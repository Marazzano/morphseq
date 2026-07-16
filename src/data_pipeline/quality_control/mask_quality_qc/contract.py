"""mask_quality_qc contract — schema truth for the structural mask-quality QC table.

Grain: one row per ``snip_id`` (overlap is computed per ``image_id`` internally, but the output
is per snip). Spine first (imported from the minting site, never re-typed), then the three
boolean flags. No persisted composite flag — snip_qc ORs the components itself.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

MASK_QUALITY_QC_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
)

MASK_QUALITY_QC_TABLE_COLUMNS: list[str] = list(SNIP_ID_SPINE_COLUMNS + MASK_QUALITY_QC_PAYLOAD_COLUMNS)


def validate_mask_quality_qc(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "mask_quality_qc",
) -> None:
    """Fail loud unless ``df`` is a valid mask_quality_qc table (spine first, then flags)."""
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )

    missing = [c for c in MASK_QUALITY_QC_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Expected {MASK_QUALITY_QC_TABLE_COLUMNS}."
        )

    # An empty well (0 snips) has nothing to validate — a 0-row CSV round-trip always comes back
    # as `object` dtype (no True/False tokens to infer bool from), so the dtype check below would
    # otherwise reject a legitimately-empty, correctly-schemaed well.
    if df.empty:
        return

    for col in MASK_QUALITY_QC_PAYLOAD_COLUMNS:
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
