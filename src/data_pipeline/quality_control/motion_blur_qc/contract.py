"""motion_blur_qc contract — schema truth for z-stack inter-slice motion QC.

Grain: one row per ``snip_id``. Spine first (imported from the minting site, never re-typed), then
mask-pixel adjacent-z NCC summaries and ``motion_blur_flag``. The flag means the snip has too high
a fraction of valid adjacent z pairs with low in-mask NCC under the configured threshold.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

MOTION_BLUR_QC_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "mask_pixel_ncc_mean",
    "mask_pixel_ncc_min",
    "mask_pixel_ncc_p05",
    "mask_pixel_bad_pair_frac",
    "mask_pixel_longest_bad_run",
    "n_z_planes",
    "n_z_pairs",
    "n_valid_z_pairs",
    "n_flat_z_pairs",
    "n_mask_pixels",
    "motion_blur_flag",
    "motion_blur_qc_applicability",
)

MOTION_BLUR_QC_TABLE_COLUMNS: list[str] = list(
    SNIP_ID_SPINE_COLUMNS + MOTION_BLUR_QC_PAYLOAD_COLUMNS
)


def validate_motion_blur_qc(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "motion_blur_qc",
) -> None:
    """Fail loud unless ``df`` is a valid motion_blur_qc table."""
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )

    missing = [c for c in MOTION_BLUR_QC_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Expected {MOTION_BLUR_QC_TABLE_COLUMNS}."
        )

    if df["motion_blur_flag"].isna().any():
        bad = df.loc[df["motion_blur_flag"].isna(), "snip_id"].head(5).tolist()
        raise ValueError(
            f"{scope_label}: motion_blur_flag has null value(s) for snip_id(s) {bad}. "
            "motion_blur_flag must be a non-null boolean QC decision."
        )
    if df["motion_blur_flag"].dtype != bool:
        raise ValueError(
            f"{scope_label}: motion_blur_flag must be boolean dtype, "
            f"got {df['motion_blur_flag'].dtype}."
        )

    from data_pipeline.quality_control.applicability import (
        ALLOWED_QC_APPLICABILITY,
        QC_APPLICABILITY_DIAGNOSTIC_ONLY,
        QC_APPLICABILITY_NOT_APPLICABLE,
    )

    applicability = df["motion_blur_qc_applicability"].astype(str)
    unknown = sorted(set(applicability) - ALLOWED_QC_APPLICABILITY)
    if unknown:
        raise ValueError(
            f"{scope_label}: motion_blur_qc_applicability has unknown value(s) {unknown}; "
            f"allowed={sorted(ALLOWED_QC_APPLICABILITY)}."
        )
    if applicability.eq(QC_APPLICABILITY_DIAGNOSTIC_ONLY).any():
        raise ValueError(
            f"{scope_label}: motion blur is either exclusion-capable or not applicable; "
            "'diagnostic_only' is not a valid state."
        )

    not_applicable = applicability.eq(QC_APPLICABILITY_NOT_APPLICABLE)
    if df.loc[not_applicable, "motion_blur_flag"].any():
        raise ValueError(
            f"{scope_label}: not-applicable rows must carry motion_blur_flag=False."
        )

    metric_cols = tuple(
        c
        for c in MOTION_BLUR_QC_PAYLOAD_COLUMNS
        if c not in {"motion_blur_flag", "motion_blur_qc_applicability"}
    )
    for col in metric_cols:
        invalid_null = df[col].isna() & ~not_applicable
        if invalid_null.any():
            bad = df.loc[invalid_null, "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: metric column {col!r} has null value(s) for snip_id(s) {bad}. "
                "Metrics may be null only for not-applicable rows."
            )
        applicable_values = df.loc[~not_applicable, col]
        if (
            not applicable_values.empty
            and not pd.api.types.is_numeric_dtype(applicable_values)
        ):
            raise ValueError(
                f"{scope_label}: metric column {col!r} must be numeric, got {df[col].dtype}."
            )
