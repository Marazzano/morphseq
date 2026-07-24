"""focus_qc contract — schema truth for the interior-structure focus QC table.

Grain: one row per ``snip_id``. Spine first (imported from the minting site, never re-typed), then
the focus metric + flag. ``focus_flag`` indicates low interior structural edge content under the
configured projection/mask heuristic — a calibration-subject QC heuristic, not a ground-truth
out-of-focus detector.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

FOCUS_QC_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "interior_strong_edge_fraction",
    "interior_n_px",
    "focus_flag",
    "focus_qc_applicability",
)

FOCUS_QC_TABLE_COLUMNS: list[str] = list(SNIP_ID_SPINE_COLUMNS + FOCUS_QC_PAYLOAD_COLUMNS)


def validate_focus_qc(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "focus_qc",
) -> None:
    """Fail loud unless ``df`` is a valid focus_qc table (spine first, then metric + flag)."""
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )

    missing = [c for c in FOCUS_QC_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Expected {FOCUS_QC_TABLE_COLUMNS}."
        )

    if df["focus_flag"].isna().any():
        bad = df.loc[df["focus_flag"].isna(), "snip_id"].head(5).tolist()
        raise ValueError(
            f"{scope_label}: focus_flag has null value(s) for snip_id(s) {bad}. "
            "focus_flag must be a non-null boolean QC decision."
        )
    if df["focus_flag"].dtype != bool:
        raise ValueError(
            f"{scope_label}: focus_flag must be boolean dtype, got {df['focus_flag'].dtype}."
        )

    from data_pipeline.quality_control.applicability import (
        ALLOWED_QC_APPLICABILITY,
        QC_APPLICABILITY_NOT_APPLICABLE,
    )

    applicability = df["focus_qc_applicability"].astype(str)
    unknown = sorted(set(applicability) - ALLOWED_QC_APPLICABILITY)
    if unknown:
        raise ValueError(
            f"{scope_label}: focus_qc_applicability has unknown value(s) {unknown}; "
            f"allowed={sorted(ALLOWED_QC_APPLICABILITY)}."
        )
    if applicability.eq(QC_APPLICABILITY_NOT_APPLICABLE).any():
        raise ValueError(
            f"{scope_label}: focus_qc is computable on a single image; use "
            "'diagnostic_only' when it must not exclude a snip."
        )

    for col in ("interior_strong_edge_fraction", "interior_n_px"):
        if df[col].isna().any():
            bad = df.loc[df[col].isna(), "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: metric column {col!r} has null value(s) for snip_id(s) {bad}. "
                "A flag without its supporting metric is not an acceptable QC product."
            )
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"{scope_label}: metric column {col!r} must be numeric, got {df[col].dtype}."
            )
