"""Tests for the physical_embryo_registry contract, validator, and builder."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.segmentation.physical_embryo_registry.build_physical_embryo_registry import (
    build_physical_embryo_registry,
    merge_physical_embryo_registry,
)
from data_pipeline.segmentation.physical_embryo_registry.physical_embryo_registry_contract import (
    PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS,
    PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY,
    empty_physical_embryo_registry,
)
from data_pipeline.segmentation.physical_embryo_registry.validate_physical_embryo_registry import (
    validate_physical_embryo_registry,
)
from data_pipeline.shared.identifiers import build_track_id, build_well_id


# ── Contract ──────────────────────────────────────────────────────────────────

def test_required_columns_is_tuple():
    assert isinstance(PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS, tuple)
    assert PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY == ("physical_embryo_id",)


def test_empty_has_exactly_required_columns():
    empty = empty_physical_embryo_registry()
    assert list(empty.columns) == list(PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS)
    assert len(empty) == 0


# ── Builder ───────────────────────────────────────────────────────────────────

def _frame_masks_fixture():
    """2 wells × 2 tracks, with duplicate frames per track + one no-mask NA-track row."""
    exp = "20250912"
    rows = []
    for well_index in ("B01", "C04"):
        well_id = build_well_id(exp, well_index)
        for track_index in (0, 1):
            track_id = build_track_id(well_id, track_index)
            # Two frames (masks) for the same animal — the mint must collapse to one row.
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
        # A no-mask placeholder row for this well — NA track, must be dropped.
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


def test_builder_one_row_per_distinct_animal():
    registry = build_physical_embryo_registry(_frame_masks_fixture())
    # 2 wells × 2 tracks = 4 animals; the no-mask rows are dropped, dup frames collapsed.
    assert len(registry) == 4
    assert list(registry.columns) == list(PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS)


def test_builder_local_embryo_index_is_one_based():
    registry = build_physical_embryo_registry(_frame_masks_fixture())
    b01 = registry[registry["well_id"] == "20250912_B01"].sort_values("local_embryo_index")
    # track 0 → embryo 1, track 1 → embryo 2 (one-based).
    assert b01["local_embryo_index"].tolist() == [1, 2]
    assert set(b01["physical_embryo_id"]) == {"20250912_B01_e01", "20250912_B01_e02"}


def test_builder_drops_no_mask_rows():
    registry = build_physical_embryo_registry(_frame_masks_fixture())
    assert (registry["track_id_source"] == "no_mask_placeholder").sum() == 0


def test_builder_empty_when_all_no_mask():
    fm = pd.DataFrame(
        [{"experiment_id": "20250912", "well_id": "20250912_B01", "track_id": pd.NA,
          "track_id_source": "no_mask_placeholder"}]
    )
    assert build_physical_embryo_registry(fm).empty


def test_builder_missing_column_raises():
    fm = _frame_masks_fixture().drop(columns=["track_id_source"])
    with pytest.raises(ValueError, match="missing required column"):
        build_physical_embryo_registry(fm)


# ── Validator ─────────────────────────────────────────────────────────────────

def _valid_registry():
    return build_physical_embryo_registry(_frame_masks_fixture())


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


# ── Merge (global-uniqueness enforcement) ─────────────────────────────────────

def test_merge_concatenates_well_shards():
    exp = "20250912"
    shard_b = build_physical_embryo_registry(
        _well_frame_masks(exp, "B01")
    )
    shard_c = build_physical_embryo_registry(
        _well_frame_masks(exp, "C04")
    )
    merged = merge_physical_embryo_registry([shard_b, shard_c])
    assert len(merged) == len(shard_b) + len(shard_c)


def test_merge_rejects_colliding_physical_embryo_id():
    exp = "20250912"
    shard = build_physical_embryo_registry(_well_frame_masks(exp, "B01"))
    with pytest.raises(ValueError, match="globally unique"):
        merge_physical_embryo_registry([shard, shard])


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
            }
        )
    return pd.DataFrame(rows)
