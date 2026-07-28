"""curvature_metrics contract — schema truth for the geodesic centerline / curvature table.

Grain: one row per ``snip_id``. All measured columns are NULLABLE by contract: a low-information
mask (skeleton too short to fit a cubic B-spline) yields documented null metrics, not a bad number
or a QC flag. ``centerline_point_count`` is always a finite integer (0 for empty masks).

The metric set mirrors the validated legacy pipeline (geodesic centerline + B-spline curvature +
the baseline-deviation family), with ``baseline_deviation_normalized`` — bow as a fraction of body
length — as the headline curvature feature downstream analysis keys on.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import (
    SNIP_FEATURE_TABLE_SPINE_COLUMNS,
    validate_feature_table,
)

# Measured payload (per snip). Every column here is nullable except centerline_point_count.
CURVATURE_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "total_length_um",
    "mean_curvature_per_um",
    "baseline_deviation_um",
    "baseline_deviation_normalized",
    "max_baseline_deviation_um",
    "baseline_deviation_std_um",
    "arc_length_ratio",
    "chord_length_um",
    "keypoint_deviation_q1_um",
    "keypoint_deviation_mid_um",
    "keypoint_deviation_q3_um",
    "centerline_point_count",
)

# All curvature/deviation summaries are null when the centerline is too short to spline;
# centerline_point_count is always present (0 for an empty mask).
_NULLABLE_PAYLOAD_COLUMNS: tuple[str, ...] = tuple(
    c for c in CURVATURE_PAYLOAD_COLUMNS if c != "centerline_point_count"
)

CURVATURE_TABLE_COLUMNS: list[str] = list(SNIP_FEATURE_TABLE_SPINE_COLUMNS + CURVATURE_PAYLOAD_COLUMNS)


def validate_curvature_features(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "curvature_features",
) -> None:
    """Fail loud unless ``df`` is a valid curvature table (spine first, then features)."""
    validate_feature_table(
        df,
        required_columns=CURVATURE_TABLE_COLUMNS,
        feature_columns=CURVATURE_PAYLOAD_COLUMNS,
        nullable_feature_columns=_NULLABLE_PAYLOAD_COLUMNS,
        scope_label=scope_label,
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
    )
