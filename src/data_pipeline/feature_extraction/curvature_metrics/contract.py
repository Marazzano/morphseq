"""curvature_features contract — schema truth for centerline/curvature metrics per snip.

Grain: one row per ``snip_id``. Curvature columns are NULLABLE by contract: a low-information
mask (too few centerline points) yields documented null metrics, not a bad number or a QC flag.
``centerline_point_count`` is always a finite integer (0 for empty masks).
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import (
    SNIP_FEATURE_TABLE_ID_COLUMNS,
    validate_feature_table,
)

# Curvature/length summaries. The per-um curvature columns and centerline_length_um are nullable
# (null when the centerline has too few points); centerline_point_count is always present.
_FEATURE_COLUMNS: tuple[str, ...] = (
    "mean_curvature_per_um",
    "median_curvature_per_um",
    "max_curvature_per_um",
    "centerline_length_um",
    "centerline_point_count",
)
_NULLABLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "mean_curvature_per_um",
    "median_curvature_per_um",
    "max_curvature_per_um",
    "centerline_length_um",
)

CURVATURE_FEATURES_REQUIRED_COLUMNS: list[str] = list(SNIP_FEATURE_TABLE_ID_COLUMNS + _FEATURE_COLUMNS)


def validate_curvature_features(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "curvature_features",
) -> None:
    """Fail loud unless ``df`` is a valid curvature_features table (spine first, then features)."""
    validate_feature_table(
        df,
        required_columns=CURVATURE_FEATURES_REQUIRED_COLUMNS,
        feature_columns=_FEATURE_COLUMNS,
        nullable_feature_columns=_NULLABLE_FEATURE_COLUMNS,
        scope_label=scope_label,
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
    )
