"""consolidated_features product tests (pure merge logic)."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.feature_extraction.consolidated_features.compute import (
    consolidate_feature_tables,
)
from data_pipeline.feature_extraction.consolidated_features.contract import (
    validate_consolidated_features,
)
from data_pipeline.feature_extraction.mask_geometry.compute import compute_mask_geometry_features
from tests.data_pipeline.feature_extraction._feature_fixtures import make_inputs

# Pull in a real mask_geometry table to seed the spine + core columns.


def _mask_geometry_table():
    snip, masks, inv, reg = make_inputs()
    return compute_mask_geometry_features(snip, masks, inv), reg


def test_merge_one_to_one_and_validates():
    mg, reg = _mask_geometry_table()
    # A second feature table sharing snip_id + spine, contributing one new column.
    extra = mg[["snip_id", "embryo_id", "physical_embryo_id", "experiment_id", "well_id",
                "image_id", "time_index", "channel_id"]].copy()
    extra["predicted_stage_hpf"] = 11.5

    merged = consolidate_feature_tables({"mask_geometry": mg, "stage": extra}, key="snip_id")
    assert len(merged) == len(mg)
    assert "area_um2" in merged.columns and "predicted_stage_hpf" in merged.columns
    validate_consolidated_features(merged, physical_embryo_registry_df=reg, check_sources=True)


def test_column_collision_fails_loud():
    mg, _ = _mask_geometry_table()
    collide = mg.copy()  # also has area_um2 -> non-spine collision
    with pytest.raises(ValueError, match="collision"):
        consolidate_feature_tables({"a": mg, "b": collide}, key="snip_id")


def test_duplicate_key_fails_loud():
    mg, _ = _mask_geometry_table()
    dup = pd.concat([mg, mg], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        consolidate_feature_tables({"a": dup}, key="snip_id")
