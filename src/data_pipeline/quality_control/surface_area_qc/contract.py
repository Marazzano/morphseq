"""surface_area_qc contract — schema truth for the surface-area outlier QC table.

Grain: one row per ``snip_id``. QC tables JUDGE the universe: every boolean output column ends
in ``_flag`` and must be non-null boolean. An applicability companion distinguishes an
exclusion-capable result from an unresolved-stage no-op. The contract validates the identity spine
FIRST (via the shared validator), then its payload.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

SURFACE_AREA_QC_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "sa_outlier_flag",
    "surface_area_qc_applicability",
)

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

    # An empty well (0 snips) has nothing to validate — a 0-row CSV round-trip always comes back
    # as `object` dtype (no True/False tokens to infer bool from), so the dtype check below would
    # otherwise reject a legitimately-empty, correctly-schemaed well.
    if df.empty:
        return

    if df["sa_outlier_flag"].isna().any():
        bad = df.loc[df["sa_outlier_flag"].isna(), "snip_id"].head(5).tolist()
        raise ValueError(
            f"{scope_label}: flag column 'sa_outlier_flag' has null value(s) "
            f"for snip_id(s) {bad}."
        )
    if df["sa_outlier_flag"].dtype != bool:
        raise ValueError(
            f"{scope_label}: flag column 'sa_outlier_flag' must be boolean dtype, "
            f"got {df['sa_outlier_flag'].dtype}."
        )

    from data_pipeline.quality_control.applicability import (
        ALLOWED_QC_APPLICABILITY,
        QC_APPLICABILITY_NOT_APPLICABLE,
    )

    applicability = df["surface_area_qc_applicability"].astype(str)
    unknown = sorted(set(applicability) - ALLOWED_QC_APPLICABILITY)
    if unknown:
        raise ValueError(
            f"{scope_label}: surface_area_qc_applicability has unknown value(s) "
            f"{unknown}."
        )
    not_applicable = applicability.eq(QC_APPLICABILITY_NOT_APPLICABLE)
    if df.loc[not_applicable, "sa_outlier_flag"].any():
        raise ValueError(
            f"{scope_label}: not-applicable rows must carry sa_outlier_flag=False."
        )
