"""Tests for the physical_embryo_registry contract, validator, and builder."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.build_physical_embryo_registry import (
    build_physical_embryo_registry,
    merge_physical_embryo_registry,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.physical_embryo_registry_contract import (
    EmbryoMergePolicy,
    PHYSICAL_EMBRYO_REGISTRY_PAYLOAD_COLUMNS,
    PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS,
    PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY,
    empty_physical_embryo_registry,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.validate_physical_embryo_registry import (
    validate_physical_embryo_registry,
)
from data_pipeline.shared.identifiers import build_track_id, build_well_id


# -- Frame-inventory helper (Agent B's shared contract: per-well n_sources column) -----

def _frame_inventory(well_ids, n_sources):
    """A minimal frame_inventory carrying one n_sources row per well (constant within well)."""
    return pd.DataFrame(
        [{"well_id": well_id, "n_sources": n_sources} for well_id in dict.fromkeys(well_ids)]
    )


# -- Contract -----------------------------------------------------------------

def test_required_columns_is_tuple():
    assert isinstance(PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS, tuple)
    assert PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY == ("physical_embryo_id",)


def test_empty_has_exactly_required_columns():
    empty = empty_physical_embryo_registry()
    assert list(empty.columns) == list(PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS)
    assert len(empty) == 0


def test_payload_columns_present_in_contract():
    assert PHYSICAL_EMBRYO_REGISTRY_PAYLOAD_COLUMNS == ("merge_policy", "n_sources")
    for col in PHYSICAL_EMBRYO_REGISTRY_PAYLOAD_COLUMNS:
        assert col in PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS


# -- Builder ------------------------------------------------------------------

def _frame_masks_fixture():
    """2 wells x 2 tracks, with duplicate frames per track + one no-mask NA-track row."""
    exp = "20250912"
    rows = []
    for well_index in ("B01", "C04"):
        well_id = build_well_id(exp, well_index)
        for track_index in (0, 1):
            track_id = build_track_id(well_id, track_index)
            # Two frames (masks) for the same animal -- the mint must collapse to one row.
            for time_index in (0, 1):
                rows.append(
                    {
                        "experiment_id": exp,
                        "well_id": well_id,
                        "track_id": track_id,
                        "track_id_source": "frame_masks",
                        "is_valid_mask": True,
                        "time_index": time_index,
                    }
                )
        # A no-mask placeholder row for this well -- NA track, must be dropped.
        rows.append(
            {
                "experiment_id": exp,
                "well_id": well_id,
                "track_id": pd.NA,
                "track_id_source": "no_mask_placeholder",
                "is_valid_mask": False,
                "time_index": 2,
            }
        )
    return pd.DataFrame(rows)


def _single_source_inventory(frame_masks):
    """n_sources == 1 for every well in the frame_masks fixture (legacy behavior)."""
    return _frame_inventory(frame_masks["well_id"].tolist(), n_sources=1)


def _build_legacy(frame_masks):
    return build_physical_embryo_registry(frame_masks, _single_source_inventory(frame_masks))


def test_builder_one_row_per_distinct_animal():
    fm = _frame_masks_fixture()
    registry = _build_legacy(fm)
    # 2 wells x 2 tracks = 4 animals; the no-mask rows are dropped, dup frames collapsed.
    assert len(registry) == 4
    assert list(registry.columns) == list(PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS)


def test_builder_local_embryo_index_is_one_based():
    registry = _build_legacy(_frame_masks_fixture())
    b01 = registry[registry["well_id"] == "20250912_B01"].sort_values("local_embryo_index")
    # track 0 -> embryo 1, track 1 -> embryo 2 (one-based).
    assert b01["local_embryo_index"].tolist() == [1, 2]
    assert set(b01["physical_embryo_id"]) == {"20250912_B01_e01", "20250912_B01_e02"}


def test_builder_drops_no_mask_rows():
    registry = _build_legacy(_frame_masks_fixture())
    assert (registry["track_id_source"] == "no_mask_placeholder").sum() == 0


def test_builder_stamps_normal_policy_and_n_sources():
    registry = _build_legacy(_frame_masks_fixture())
    assert set(registry["merge_policy"]) == {EmbryoMergePolicy.NORMAL.value}
    assert set(registry["n_sources"]) == {1}


def test_builder_empty_when_all_no_mask():
    fm = pd.DataFrame(
        [{"experiment_id": "20250912", "well_id": "20250912_B01", "track_id": pd.NA,
          "track_id_source": "no_mask_placeholder"}]
    )
    inv = _frame_inventory(["20250912_B01"], n_sources=1)
    assert build_physical_embryo_registry(fm, inv).empty


def test_builder_missing_column_raises():
    fm = _frame_masks_fixture().drop(columns=["track_id_source"])
    with pytest.raises(ValueError, match="missing required column"):
        build_physical_embryo_registry(fm, _single_source_inventory(_frame_masks_fixture()))


def test_builder_missing_n_sources_raises():
    fm = _frame_masks_fixture()
    inv = pd.DataFrame([{"well_id": w} for w in fm["well_id"].unique()])  # no n_sources col
    with pytest.raises(ValueError, match="n_sources"):
        build_physical_embryo_registry(fm, inv)


# -- Validator ----------------------------------------------------------------

def _valid_registry():
    return _build_legacy(_frame_masks_fixture())


def test_validator_accepts_valid():
    validate_physical_embryo_registry(_valid_registry())


def test_validator_rejects_duplicate_physical_embryo_id():
    registry = _valid_registry()
    dup = pd.concat([registry, registry.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="globally unique"):
        validate_physical_embryo_registry(dup)


def test_validator_rejects_zero_index():
    registry = _valid_registry().copy()
    registry.loc[0, "local_embryo_index"] = 0
    registry.loc[0, "physical_embryo_id"] = "20250912_B01_e00"
    with pytest.raises(ValueError, match=">= 1|one-based"):
        validate_physical_embryo_registry(registry)


def test_validator_rejects_id_not_matching_well_and_index():
    registry = _valid_registry().copy()
    # Corrupt the physical_embryo_id so it no longer matches (well_id, local_embryo_index).
    registry.loc[0, "physical_embryo_id"] = "20250912_B01_e09"
    with pytest.raises(ValueError, match="constructor-minted|encodes well_id"):
        validate_physical_embryo_registry(registry)


def test_validator_rejects_track_resolving_to_two_animals():
    registry = _valid_registry().copy()
    # Make two distinct animals share the same (well_id, track_id).
    registry.loc[1, "well_id"] = registry.loc[0, "well_id"]
    registry.loc[1, "track_id"] = registry.loc[0, "track_id"]
    with pytest.raises(ValueError, match="duplicate \\(well_id, track_id\\)|more than one"):
        validate_physical_embryo_registry(registry)


def test_validator_rejects_unknown_merge_policy():
    registry = _valid_registry().copy()
    registry.loc[0, "merge_policy"] = "smeared"
    with pytest.raises(ValueError, match="merge_policy"):
        validate_physical_embryo_registry(registry)


def test_validator_rejects_zero_n_sources():
    registry = _valid_registry().copy()
    registry.loc[0, "n_sources"] = 0
    with pytest.raises(ValueError, match="n_sources"):
        validate_physical_embryo_registry(registry)


# -- Merge (global-uniqueness enforcement) ------------------------------------

def _well_frame_masks(exp, well_index):
    well_id = build_well_id(exp, well_index)
    rows = []
    for track_index in (0, 1):
        rows.append(
            {
                "experiment_id": exp,
                "well_id": well_id,
                "track_id": build_track_id(well_id, track_index),
                "track_id_source": "frame_masks",
                "time_index": 0,
            }
        )
    return pd.DataFrame(rows)


def test_merge_concatenates_well_shards():
    exp = "20250912"
    shard_b = build_physical_embryo_registry(
        _well_frame_masks(exp, "B01"), _frame_inventory([build_well_id(exp, "B01")], 1)
    )
    shard_c = build_physical_embryo_registry(
        _well_frame_masks(exp, "C04"), _frame_inventory([build_well_id(exp, "C04")], 1)
    )
    merged = merge_physical_embryo_registry([shard_b, shard_c])
    assert len(merged) == len(shard_b) + len(shard_c)


def test_merge_rejects_colliding_physical_embryo_id():
    exp = "20250912"
    shard = build_physical_embryo_registry(
        _well_frame_masks(exp, "B01"), _frame_inventory([build_well_id(exp, "B01")], 1)
    )
    with pytest.raises(ValueError, match="globally unique"):
        merge_physical_embryo_registry([shard, shard])


# -- Legacy regression: NORMAL (n_sources==1) is byte-identical to pre-round-1 ----------
#
# The pre-round-1 builder took only frame_masks and had no merge_policy/n_sources columns.
# We regression-guard the identity part (spine + provenance) byte-for-byte, and separately
# assert the payload is the NORMAL default (added columns, legacy semantics untouched).

_LEGACY_SPINE_AND_PROVENANCE = [
    "physical_embryo_id",
    "experiment_id",
    "well_id",
    "local_embryo_index",
    "track_id",
    "track_id_source",
]


def _pre_round1_expected(frame_masks):
    """Reproduce the pre-round-1 mint exactly (single-arg, per-track), for assert_frame_equal."""
    from data_pipeline.shared.identifiers import (
        build_physical_embryo_id,
        parse_embryo_local_track_id,
        track_index_to_embryo_index,
    )

    detected = frame_masks[frame_masks["track_id"].notna()].copy()
    distinct = (
        detected[["well_id", "track_id", "experiment_id", "track_id_source"]]
        .drop_duplicates(subset=["well_id", "track_id"])
        .reset_index(drop=True)
    )
    rows = []
    for _, row in distinct.iterrows():
        well_id = str(row["well_id"])
        track_id = str(row["track_id"])
        idx = track_index_to_embryo_index(parse_embryo_local_track_id(track_id))
        rows.append(
            {
                "physical_embryo_id": build_physical_embryo_id(well_id, idx),
                "experiment_id": str(row["experiment_id"]),
                "well_id": well_id,
                "local_embryo_index": idx,
                "track_id": track_id,
                "track_id_source": str(row["track_id_source"]),
            }
        )
    return pd.DataFrame(rows, columns=_LEGACY_SPINE_AND_PROVENANCE)


def test_normal_matches_pre_round1_byte_identical():
    fm = _frame_masks_fixture()
    registry = _build_legacy(fm)
    got = registry[_LEGACY_SPINE_AND_PROVENANCE].reset_index(drop=True)
    expected = _pre_round1_expected(fm).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)
    # And the added payload is the plain NORMAL default -- legacy semantics untouched.
    assert set(registry["merge_policy"]) == {EmbryoMergePolicy.NORMAL.value}
    assert set(registry["n_sources"]) == {1}


# -- Merge policy: BRIDGE and FRACTURE (n_sources > 1) --------------------------------
#
# Snapshot acquisitions merge under one well as consecutive time_index blocks; there is no
# cross-source track_id collision (tracking runs over one time-ordered series per well). The
# merge decision is driven by the per-well n_sources COUNT from frame_inventory.

_EXP_COLL = "cilia_snapshots_coll_plate01"


def _collection_frame_masks(rows):
    """Build a frame_masks fixture. Each item is (well_index, time_index, track_index).

    time_index is the source block (one time_index per snapshot acquisition).
    """
    out = []
    for well_index, time_index, track_index in rows:
        well_id = build_well_id(_EXP_COLL, well_index)
        # a duplicate frame to prove the mint collapses per animal
        for _dup in range(2):
            out.append(
                {
                    "experiment_id": _EXP_COLL,
                    "well_id": well_id,
                    "track_id": build_track_id(well_id, track_index),
                    "track_id_source": "frame_masks",
                    "time_index": time_index,
                }
            )
    return pd.DataFrame(out)


def test_bridge_two_sources_one_track_shares_one_id():
    """n_sources=2, one distinct track_id in the well -> BRIDGE: ONE _e01 across timepoints."""
    fm = _collection_frame_masks(
        [
            ("A01", 0, 0),  # source 0 (time_index 0)
            ("A01", 1, 0),  # source 1 (time_index 1) -- SAME track_id (one time-ordered series)
        ]
    )
    well_id = build_well_id(_EXP_COLL, "A01")
    inv = _frame_inventory([well_id], n_sources=2)
    registry = build_physical_embryo_registry(fm, inv)

    # One distinct track -> BRIDGE -> one physical_embryo_id spanning the well's timepoints.
    assert set(registry["physical_embryo_id"]) == {f"{well_id}_e01"}
    assert set(registry["merge_policy"]) == {EmbryoMergePolicy.BRIDGE.value}
    assert set(registry["n_sources"]) == {2}
    validate_physical_embryo_registry(registry)


def test_fracture_two_sources_two_tracks_each_disjoint_blocks():
    """n_sources=2, 2 tracks per source -> FRACTURE: disjoint _e blocks (e01,e02 / e03,e04)."""
    fm = _collection_frame_masks(
        [
            ("A01", 0, 0),  # source 0: two animals
            ("A01", 0, 1),
            ("A01", 1, 2),  # source 1: two animals (distinct track indices -- one series per well)
            ("A01", 1, 3),
        ]
    )
    well_id = build_well_id(_EXP_COLL, "A01")
    inv = _frame_inventory([well_id], n_sources=2)
    registry = build_physical_embryo_registry(fm, inv)

    # 2 + 2 = 4 distinct animals, disjoint _e blocks, all distinct (no guessed pairing).
    assert registry["physical_embryo_id"].nunique() == 4
    assert set(registry["physical_embryo_id"]) == {
        f"{well_id}_e01",
        f"{well_id}_e02",
        f"{well_id}_e03",
        f"{well_id}_e04",
    }
    assert set(registry["merge_policy"]) == {EmbryoMergePolicy.FRACTURE.value}
    assert set(registry["n_sources"]) == {2}
    validate_physical_embryo_registry(registry)


def test_fractured_embryo_is_single_time_index_and_validator_accepts():
    """A fractured animal lives at a single time_index (flagged-OK layout); validator ACCEPTS."""
    fm = _collection_frame_masks(
        [
            ("A01", 0, 0),  # source 0: two animals -> fracture
            ("A01", 0, 1),
            ("A01", 1, 2),  # source 1: one animal
        ]
    )
    well_id = build_well_id(_EXP_COLL, "A01")
    inv = _frame_inventory([well_id], n_sources=2)
    registry = build_physical_embryo_registry(fm, inv)

    # source 0's two animals -> e01,e02 ; source 1's animal -> e03.
    ids = set(registry["physical_embryo_id"])
    assert ids == {f"{well_id}_e01", f"{well_id}_e02", f"{well_id}_e03"}
    # Each track (and thus each fractured animal) came from a single time_index block.
    per_track_times = fm.groupby("track_id")["time_index"].nunique()
    assert (per_track_times == 1).all()
    # The validator must ACCEPT this single-time_index layout (do NOT reject).
    validate_physical_embryo_registry(registry)


def test_mixed_wells_behave_independently():
    """BRIDGE and FRACTURE decisions are per-well; n_sources drives each independently."""
    fm = _collection_frame_masks(
        [
            # Well A01: one track across two sources -> BRIDGE.
            ("A01", 0, 0),
            ("A01", 1, 0),
            # Well B02: two tracks in a source -> FRACTURE (three animals total).
            ("B02", 0, 0),
            ("B02", 0, 1),
            ("B02", 1, 2),
        ]
    )
    a01 = build_well_id(_EXP_COLL, "A01")
    b02 = build_well_id(_EXP_COLL, "B02")
    inv = _frame_inventory([a01, b02], n_sources=2)
    registry = build_physical_embryo_registry(fm, inv)

    a01_rows = registry[registry["well_id"] == a01]
    b02_rows = registry[registry["well_id"] == b02]

    assert set(a01_rows["physical_embryo_id"]) == {f"{a01}_e01"}
    assert set(a01_rows["merge_policy"]) == {EmbryoMergePolicy.BRIDGE.value}
    assert set(b02_rows["physical_embryo_id"]) == {f"{b02}_e01", f"{b02}_e02", f"{b02}_e03"}
    assert set(b02_rows["merge_policy"]) == {EmbryoMergePolicy.FRACTURE.value}
    validate_physical_embryo_registry(registry)
