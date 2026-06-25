"""surface_area_qc contract — schema truth for the surface-area outlier QC table.

Grain: one row per ``snip_id``. QC tables JUDGE the universe: every boolean output column ends
in ``_flag`` and must be non-null boolean. The contract validates the identity spine FIRST (via
the shared ``validate_snip_grain_identity_columns`` from the identity minting site — spine is
imported, never re-typed), then its own ``sa_outlier_flag`` column.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

SURFACE_AREA_QC_PAYLOAD_COLUMNS: tuple[str, ...] = ("sa_outlier_flag",)

SURFACE_AREA_QC_TABLE_COLUMNS: list[str] = list(SNIP_ID_SPINE_COLUMNS + SURFACE_AREA_QC_PAYLOAD_COLUMNS)


def validate_surface_area_qc(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "surface_area_qc",
) -> None:
    """Fail loud unless ``df`` is a valid surface_area_qc table (spine first, then flag)."""
    # 1. Identity spine — own level + all parents agree; optionally verified against the registry.
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )

    # 2. QC flag columns — present, non-null, boolean dtype.
    missing = [c for c in SURFACE_AREA_QC_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Expected {SURFACE_AREA_QC_TABLE_COLUMNS}."
        )

    for col in SURFACE_AREA_QC_PAYLOAD_COLUMNS:
        if df[col].isna().any():
            bad = df.loc[df[col].isna(), "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: flag column {col!r} has null value(s) for snip_id(s) {bad}. "
                "Every *_flag must be a non-null boolean QC decision."
            )
        if df[col].dtype != bool:
            raise ValueError(
                f"{scope_label}: flag column {col!r} must be boolean dtype, got {df[col].dtype}."
            )
