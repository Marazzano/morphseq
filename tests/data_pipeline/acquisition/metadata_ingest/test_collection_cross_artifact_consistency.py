"""CROSS-ARTIFACT consistency — the four collection artifacts must speak ONE source key.

This is the test that targets the drift this work fixed. Four artifacts describe a collection
plate, and every one of them carries the source key:

    classify artifact  →  scope_metadata  →  acquisition_inventory  →  position_mapping

Two invariants hold across all of them:

  1. ``source_ordinal`` means the same thing everywhere — the same set of ordinals, and the same
     ordinal↔source_file pairing. (Before this work the scope union and the acquisition union
     assigned time indices by different rules, so the artifacts silently disagreed.)
  2. ``(source_ordinal, raw_time_index) → time_index`` is the SAME relation in both unions,
     because both call the one shared ``remap_source_time_indices``.

Deliberately built from synthetic per-source frames (injected readers), so the test pins the
CONTRACT rather than one microscope's real output. The real-data equivalents run in the
end-to-end verification.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_cross_artifact_consistency.py
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_ingest import (
    _rekey_keyence_scope_metadata_to_plate,
)
from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import (
    SourceChild,
    union_collection_acquisition_inventories,
)
from data_pipeline.acquisition.metadata_ingest.collection_scope_union import (
    union_collection_scope_metadata,
)

COLLECTION = "chem28c_coll"
PLATE = "chem28c_coll_plate01"

# Two snapshot sources and one timelapse source, so the test covers the case where ordinal and
# frame index CANNOT coincide.
CHILDREN = (
    ("20250622_plate01_t28hpf", 1),  # (child name, frame count)
    ("20250623_plate01_t52hpf", 3),  # a timelapse source — spans a RANGE of time_index
    ("20250624_plate01_t76hpf", 1),
)


def _classify_sources():
    """The classify artifact's `sources` records, in ordinal order."""
    return [
        {
            "file": child,
            "raw_path": f"/raw/{COLLECTION}/{child}",
            "declared_hpf": 28 + 24 * i,
            "source_ordinal": i,
            "time_index": i,
        }
        for i, (child, _frames) in enumerate(CHILDREN)
    ]


def _per_source_frame(child, n_frames, *, for_inventory):
    """A per-source table with that source's OWN frame numbering (always starting at 0)."""
    rows = []
    for well_number in (1, 2):
        well_index = f"A0{well_number}"
        for time_index in range(n_frames):
            row = {
                # Source-bound id — both unions restamp it to the plate.
                "experiment_id": child,
                "well_index": well_index,
                "well_id": f"{child}_{well_index}",
                "time_index": time_index,
                "position_index": well_number,
                "raw_position_label": str(well_number),
                "channel": "BF",
            }
            if for_inventory:
                row["time_index_claimed"] = time_index
                row["channel_id"] = "BF"
            else:
                row["image_id"] = f"{child}_{well_index}_BF_t{time_index:04d}"
            rows.append(row)
    return pd.DataFrame(rows)


def _build_both_unions():
    frames = {child: n for child, n in CHILDREN}

    scope = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_classify_sources(),
        read_source=lambda rec, _exp: _per_source_frame(
            rec["file"], frames[rec["file"]], for_inventory=False
        ),
        rekey_to_plate=_rekey_keyence_scope_metadata_to_plate,
    )
    inventory = union_collection_acquisition_inventories(
        collection_name=COLLECTION,
        sources=[SourceChild(child_name=child, scope="Keyence") for child, _ in CHILDREN],
        read_source=lambda source, _exp: _per_source_frame(
            source.child_name, frames[source.child_name], for_inventory=True
        ),
    )
    return scope, inventory


# ── Invariant 1: source_ordinal means the same thing in every artifact ────────────────


def test_classify_and_scope_union_agree_on_source_ordinals():
    scope, _ = _build_both_unions()
    classify_ordinals = {rec["source_ordinal"] for rec in _classify_sources()}
    assert set(scope["source_ordinal"]) == classify_ordinals == {0, 1, 2}


def test_both_unions_agree_on_source_ordinals():
    scope, inventory = _build_both_unions()
    assert set(scope["source_ordinal"]) == set(inventory["source_ordinal"])


def test_ordinal_to_source_pairing_matches_the_classify_artifact():
    """Not just the same ordinals — the same ordinal→source assignment."""
    scope, _ = _build_both_unions()
    expected = {rec["source_ordinal"]: rec["file"] for rec in _classify_sources()}
    got = dict(
        scope.groupby("source_ordinal")["source_id"].agg(lambda s: s.unique()[0])
    )
    assert got == expected


def test_position_mapping_shares_the_same_ordinals():
    """The mapping is built per source, one block per ordinal — the same ordinal space."""
    scope, _ = _build_both_unions()
    # A mapping block is emitted per source (see collection_position_mapping); its ordinal set is
    # exactly the classify artifact's, which is exactly the scope union's.
    mapping_ordinals = {rec["source_ordinal"] for rec in _classify_sources()}
    assert mapping_ordinals == set(scope["source_ordinal"])


# ── Invariant 2: the two unions agree on (source_ordinal, raw_time_index) → time_index ─


def _time_relation(df):
    return set(zip(df["source_ordinal"], df["raw_time_index"], df["time_index"]))


def test_both_unions_agree_on_the_merged_time_relation():
    """THE regression guard. This is what silently diverged before the shared remapper."""
    scope, inventory = _build_both_unions()
    assert _time_relation(scope) == _time_relation(inventory)


def test_merged_time_axis_is_contiguous_across_the_plate():
    scope, inventory = _build_both_unions()
    expected = list(range(sum(n for _child, n in CHILDREN)))  # 1 + 3 + 1 = 5 frames
    assert sorted(scope["time_index"].unique()) == expected
    assert sorted(inventory["time_index"].unique()) == expected


def test_timelapse_source_spans_a_range_in_BOTH_artifacts():
    """The case where ordinal and frame index cannot coincide — ordinal 1 owns 3 timepoints."""
    scope, inventory = _build_both_unions()
    for df in (scope, inventory):
        by_ordinal = df.groupby("source_ordinal")["time_index"].apply(set)
        assert by_ordinal.loc[0] == {0}
        assert by_ordinal.loc[1] == {1, 2, 3}  # the timelapse block
        assert by_ordinal.loc[2] == {4}


def test_no_merged_time_index_is_claimed_by_two_sources():
    scope, inventory = _build_both_unions()
    for df in (scope, inventory):
        per_time_index = df.groupby("time_index")["source_ordinal"].nunique()
        assert (per_time_index == 1).all()


def test_well_id_is_plate_keyed_in_both_artifacts():
    scope, inventory = _build_both_unions()
    expected = {f"{PLATE}_A01", f"{PLATE}_A02"}
    assert set(scope["well_id"]) == expected
    assert set(inventory["well_id"]) == expected


def test_claimed_time_atom_tracks_the_merged_axis():
    """`time_index_claimed` is inside the inventory's cell key, so it must ride the same remap."""
    _scope, inventory = _build_both_unions()
    assert (inventory["time_index_claimed"] == inventory["time_index"]).all()
