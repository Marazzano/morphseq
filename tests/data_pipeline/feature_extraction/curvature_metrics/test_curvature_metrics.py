"""curvature_metrics product tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.feature_extraction.curvature_metrics.compute import (
    compute_curvature_features,
    compute_curvature_for_mask,
)
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
    # A tiny 1x1 mask cannot form a splinable skeleton -> documented null metrics, still a row.
    snip, masks, inv, reg = make_inputs(time_indices=(0,), mask_side=1)
    df = compute_curvature_features(snip, masks, inv)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["baseline_deviation_normalized"])
    assert df.iloc[0]["centerline_point_count"] == 0
    validate_curvature_features(df)  # nullable curvature columns pass


def test_elongated_curved_mask_gets_real_geodesic_spine():
    # A long, gently curved bar should skeletonize into a full spine (not a stub) and yield a
    # finite normalized baseline deviation > 0 (it bows away from the head-tail chord).
    # Thick enough (~44px wide) to survive the sigma=15 boundary smoothing, like a real embryo.
    canvas = np.zeros((300, 300), dtype=bool)
    t = np.linspace(0, np.pi, 220)
    xs = (40 + 200 * t / np.pi).astype(int)
    ys = (150 + 55 * np.sin(t)).astype(int)
    for x, y in zip(xs, ys):
        canvas[y - 22 : y + 22, x - 22 : x + 22] = True  # thick curved bar

    metrics = compute_curvature_for_mask(canvas, pixel_size_um=1.0)
    assert metrics["centerline_point_count"] > 50  # a real spine, not a collapsed stub
    assert metrics["total_length_um"] > 100
    assert np.isfinite(metrics["baseline_deviation_normalized"])
    assert metrics["baseline_deviation_normalized"] > 0
    assert metrics["arc_length_ratio"] >= 1.0
