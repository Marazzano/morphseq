"""mask_geometry_features contract — schema truth for the geometry feature table.

Grain: one row per ``snip_id``. The contract validates the identity spine FIRST (via the
shared ``validate_snip_grain_identity_columns``), then its own measured feature columns.
Features MEASURE the universe; they do not judge it — so there are no ``_flag`` columns
here, and an empty/degenerate mask yields documented numeric values (0.0), never a QC
decision. Exclusion is QC's job downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    validate_snip_grain_identity_columns,
)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Contract — identity spine (validated by the shared spine validator) + measured feature columns
# ─────────────────────────────────────────────────────────────────────────────────────────────

# Identity spine (snip grain). These are NOT optional provenance — they are the identity-carrying
# contract, checked by validate_snip_grain_identity_columns.
_SPINE_COLUMNS: tuple[str, ...] = (
    "snip_id",
    "embryo_id",
    "physical_embryo_id",
    "experiment_id",
    "well_id",
    "image_id",
    "time_index",
    "channel_id",
)

# Measured, micron-aware geometry. Continuous values; no booleans, no ``_flag`` columns.
_FEATURE_COLUMNS: tuple[str, ...] = (
    "area_um2",
    "perimeter_um",
    "length_um",
    "width_um",
    "centroid_x_um",
    "centroid_y_um",
)

MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS: list[str] = list(_SPINE_COLUMNS + _FEATURE_COLUMNS)


def validate_mask_geometry_features(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "mask_geometry_features",
) -> None:
    """Fail loud unless ``df`` is a valid mask_geometry_features table.

    Spine first, then features. At the consume boundary pass the per-well
    ``physical_embryo_registry`` shard with ``check_sources=True`` so every row roots in a
    registered animal (the registry is the verifier of the snip_id universe).
    """
    # 1. Identity spine — own level + all parents agree; optionally verified against the registry.
    validate_snip_grain_identity_columns(
        df,
        grain="snip",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )

    # 2. Feature columns — present, non-null, finite. (snip_id uniqueness is enforced by the spine.)
    missing = [c for c in MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Expected {MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS}."
        )

    for col in _FEATURE_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            bad = df.loc[values.isna(), "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: feature column {col!r} has null/non-numeric value(s) for "
                f"snip_id(s) {bad}. Geometry features must be finite numbers (empty masks → 0.0, "
                "never null). If a mask could not be decoded, fix the upstream join."
            )
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            bad = df.loc[~np.isfinite(values.to_numpy(dtype=float)), "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: feature column {col!r} has non-finite value(s) (inf/nan) for "
                f"snip_id(s) {bad}."
            )
