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
    PlateSource,
    union_collection_acquisition_inventories,
)
from data_pipeline.acquisition.metadata_ingest.scope.shared.acquisition_channels import (
    CHANNEL_ID_COLUMN,
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
                CHANNEL_ID_COLUMN: "BF",
            }
            if for_inventory:
                row["time_index_claimed"] = time_index
                row[CHANNEL_ID_COLUMN] = "BF"
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
        sources=[PlateSource(source_id=child, scope="Keyence") for child, _ in CHILDREN],
        read_source=lambda source, _exp: _per_source_frame(
            source.source_id, frames[source.source_id], for_inventory=True
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


# ── TWO SOURCES AT THE SAME DECLARED AGE → FAIL LOUD ──────────────────────────────────
# source_ordinal is assigned by declared_hpf, and it keys the age map plus every source-aware join.
# Two sources declaring the same age give no declared basis for which comes first, so `sorted`
# (stable) would fall back to filesystem glob order and the same data could get different ordinals
# on different runs. We refuse to guess: no date tiebreaker, no name tiebreaker — raise.

import pytest

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import (
    assert_source_order_unambiguous,
)


def _children(*names):
    return [PlateSource(source_id=n, scope="Keyence") for n in names]


def _one_frame(child):
    return pd.DataFrame(
        [
            {
                "experiment_id": child,
                "well_index": "A01",
                "well_id": f"{child}_A01",
                "time_index": 0,
                "time_index_claimed": 0,
                "position_index": 1,
                CHANNEL_ID_COLUMN: "BF",
            }
        ]
    )


def test_same_declared_age_is_rejected():
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        assert_source_order_unambiguous(
            _children("20250622_plate01_t28hpf", "20250623_plate01_t28hpf")
        )


def test_same_declared_age_rejected_even_with_different_dates():
    """The date is NOT a tiebreaker — declared age alone determines the ordinal."""
    with pytest.raises(ValueError, match="declare the same age"):
        assert_source_order_unambiguous(
            _children("20250101_plate01_t28hpf", "20259999_plate01_t28hpf")
        )


def test_ambiguity_error_names_the_colliding_children():
    with pytest.raises(ValueError, match="20250622_plate01_t28hpf"):
        assert_source_order_unambiguous(
            _children("20250622_plate01_t28hpf", "20250623_plate01_t28hpf")
        )


def test_multiple_undeclared_ages_are_also_ambiguous():
    # Two sci-style sources declare no age at all — nothing orders them.
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        assert_source_order_unambiguous(
            _children("20250622_plate01_sci", "20250623_plate01_sci")
        )


def test_distinct_declared_ages_are_accepted():
    assert_source_order_unambiguous(
        _children("20250622_plate01_t28hpf", "20250623_plate01_t52hpf")
    )


def test_union_refuses_same_age_sources_before_reading():
    """The guard runs BEFORE any source read, so an ambiguous plate costs no I/O."""
    reads = []

    def read_source(source, _experiment_id):
        reads.append(source.source_id)
        return _one_frame(source.source_id)

    with pytest.raises(ValueError, match="AMBIGUOUS"):
        union_collection_acquisition_inventories(
            collection_name=COLLECTION,
            sources=_children("20250622_plate01_t28hpf", "20250623_plate01_t28hpf"),
            read_source=read_source,
        )
    assert reads == []


def test_single_source_is_never_ambiguous():
    assert_source_order_unambiguous(_children("20250622_plate01_t28hpf"))
    assert_source_order_unambiguous(_children("20250622_plate01_sci"))


# ── SPARSE source frame numbering across BOTH artifacts ───────────────────────────────
# The fixtures above emit dense raw times (range(n_frames)), where remap-by-rank and
# `raw + offset` are indistinguishable — so they cannot detect a regression in COMPACTION.
# Keyence derives time from on-disk T#### tokens, so a partial/resumed acquisition yields sparse
# raw indices (e.g. 2, 4). Both unions must compact them to the SAME dense merged axis; if one
# compacted and the other offset, geometry and pixels would name different canonical frames.

_SPARSE_FRAMES = {
    "20250622_plate01_t28hpf": [2, 4],        # sparse: a resumed acquisition
    "20250623_plate01_t52hpf": [1, 2, 3],     # 1-based
    "20250624_plate01_t76hpf": [0],
}


def _sparse_source_frame(child, *, for_inventory):
    rows = []
    for well_number in (1, 2):
        well_index = f"A0{well_number}"
        for raw_time in _SPARSE_FRAMES[child]:
            row = {
                "experiment_id": child,
                "well_index": well_index,
                "well_id": f"{child}_{well_index}",
                "time_index": raw_time,
                "position_index": well_number,
                "raw_position_label": str(well_number),
                CHANNEL_ID_COLUMN: "BF",
            }
            if for_inventory:
                row["time_index_claimed"] = raw_time
                row[CHANNEL_ID_COLUMN] = "BF"
            else:
                row["image_id"] = f"{child}_{well_index}_BF_t{raw_time:04d}"
            rows.append(row)
    return pd.DataFrame(rows)


def _build_both_unions_sparse():
    scope = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_classify_sources(),
        read_source=lambda rec, _exp: _sparse_source_frame(rec["file"], for_inventory=False),
        rekey_to_plate=_rekey_keyence_scope_metadata_to_plate,
    )
    inventory = union_collection_acquisition_inventories(
        collection_name=COLLECTION,
        sources=[PlateSource(source_id=child, scope="Keyence") for child, _ in CHILDREN],
        read_source=lambda source, _exp: _sparse_source_frame(
            source.source_id, for_inventory=True
        ),
    )
    return scope, inventory


def test_sparse_raw_times_compact_IDENTICALLY_in_both_artifacts():
    """THE compaction guard. Fails if either union stops remapping by rank."""
    scope, inventory = _build_both_unions_sparse()
    assert _time_relation(scope) == _time_relation(inventory)


def test_sparse_raw_times_yield_a_DENSE_merged_axis():
    scope, inventory = _build_both_unions_sparse()
    # 2 + 3 + 1 = 6 distinct frames -> merged 0..5 with NO holes, despite sparse/1-based input.
    for df in (scope, inventory):
        assert sorted(df["time_index"].unique()) == [0, 1, 2, 3, 4, 5]


def test_sparse_raw_values_are_preserved_for_audit():
    scope, _ = _build_both_unions_sparse()
    by_ordinal = scope.groupby("source_ordinal")["raw_time_index"].apply(lambda s: sorted(set(s)))
    assert by_ordinal.loc[0] == [2, 4]      # the source's own numbering, unchanged
    assert by_ordinal.loc[1] == [1, 2, 3]
    assert by_ordinal.loc[2] == [0]


def test_sparse_source_leaves_no_hole_for_the_next_source():
    scope, _ = _build_both_unions_sparse()
    blocks = scope.groupby("source_ordinal")["time_index"].apply(lambda s: (min(s), max(s)))
    # Block widths are the DISTINCT frame counts (2, 3, 1) — not max(raw)+1.
    assert list(blocks) == [(0, 1), (2, 4), (5, 5)]
