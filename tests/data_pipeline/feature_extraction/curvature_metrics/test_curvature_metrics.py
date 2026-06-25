"""curvature_metrics product tests."""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.curvature_metrics.compute import compute_curvature_features
from data_pipeline.feature_extraction.curvature_metrics.contract import (
    CURVATURE_TABLE_COLUMNS,
    validate_curvature_features,
)
from tests.data_pipeline.feature_extraction._feature_fixtures import make_inputs


def test_compute_one_row_per_snip_and_validates():
    snip, masks, inv, reg = make_inputs()
    df = compute_curvature_features(snip, masks, inv)
    assert len(df) == len(snip)
    assert list(df.columns) == CURVATURE_TABLE_COLUMNS
    validate_curvature_features(df, physical_embryo_registry_df=reg, check_sources=True)


def test_low_information_mask_yields_null_curvature_not_error():
    # A tiny 1x1 mask cannot form a >=3 point centerline -> documented null metrics, still a row.
    snip, masks, inv, reg = make_inputs(time_indices=(0,), mask_side=1)
    df = compute_curvature_features(snip, masks, inv)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["mean_curvature_per_um"])
    validate_curvature_features(df)  # nullable curvature columns pass
