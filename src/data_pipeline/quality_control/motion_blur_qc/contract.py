"""motion_blur_qc contract — schema truth for z-stack inter-slice motion QC.

Grain: one row per ``snip_id``. Spine first (imported from the minting site, never re-typed), then
mask-pixel adjacent-z NCC summaries and ``motion_blur_flag``. The flag means the snip has too high
a fraction of valid adjacent z pairs with low in-mask NCC under the configured threshold.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
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

    metric_cols = tuple(c for c in MOTION_BLUR_QC_PAYLOAD_COLUMNS if c != "motion_blur_flag")
    for col in metric_cols:
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
