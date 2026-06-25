"""Shared mechanics for snip-grain feature tables — spine validation + the RLE join scaffold.

Earned shared code: every feature product validates the identity spine first (registry as
verifier), then its own columns; and every mask-derived product runs the same
snip_inventory -> frame_masks (by mask_id) -> frame_inventory (pixel size by image_id) join,
decodes mask_rle, and calls a pure per-mask function. These helpers own that boring repetition
so each product owns only its measured columns.
"""

from __future__ import annotations

import json
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from data_pipeline.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_FRAME_PROVENANCE_COLUMNS,
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

# Identity spine + frame-derived provenance columns carried by per-snip feature tables (both defined
# once in snip_identity_contract). Use for building output rows from snip_inventory. Not the identity
# spine — do not validate with it.
SNIP_FEATURE_TABLE_SPINE_COLUMNS: tuple[str, ...] = SNIP_ID_SPINE_COLUMNS + SNIP_FRAME_PROVENANCE_COLUMNS

# Micron calibration source on a frame_inventory row (target name first, legacy fallback).
_PIXEL_SIZE_COLUMNS: tuple[str, ...] = ("source_micrometers_per_pixel", "micrometers_per_pixel")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Spine-first contract validation
# ─────────────────────────────────────────────────────────────────────────────────────────────


def validate_feature_table(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    feature_columns: Sequence[str],
    scope_label: str,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    nullable_feature_columns: Sequence[str] = (),
) -> None:
    """Validate a snip-grain feature table: identity spine first, then feature columns.

    ``feature_columns`` are required-present and finite. Columns also listed in
    ``nullable_feature_columns`` may be null (documented low-information null metrics) but, where
    present, must still be finite numbers. Features are never ``_flag`` booleans (that is QC).
    """
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Expected {list(required_columns)}."
        )

    nullable = set(nullable_feature_columns)
    for col in feature_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        coerced_nan = values.isna() & df[col].notna()  # value present but not numeric
        if coerced_nan.any():
            bad = df.loc[coerced_nan, "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: feature column {col!r} has non-numeric value(s) for "
                f"snip_id(s) {bad}."
            )
        if col not in nullable and values.isna().any():
            bad = df.loc[values.isna(), "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: feature column {col!r} has null value(s) for snip_id(s) {bad}. "
                "This column is not declared nullable; fix the compute or declare it nullable."
            )
        finite_mask = values.notna()
        if finite_mask.any() and not np.isfinite(values[finite_mask].to_numpy(dtype=float)).all():
            bad = df.loc[finite_mask & ~np.isfinite(values.where(finite_mask).to_numpy(dtype=float)), "snip_id"].head(5).tolist()
            raise ValueError(
                f"{scope_label}: feature column {col!r} has non-finite value(s) (inf) for "
                f"snip_id(s) {bad}."
            )


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Mask join scaffold — snip_inventory -> frame_masks (by mask_id) -> frame_inventory pixel size
# ─────────────────────────────────────────────────────────────────────────────────────────────


def pixel_size_for_image(frame_inventory_by_image: pd.DataFrame, image_id: str, snip_id: str) -> float:
    """Return the micron calibration for ``image_id``, failing loud if absent or non-positive."""
    if image_id not in frame_inventory_by_image.index:
        raise ValueError(
            f"feature_extraction: image_id {image_id!r} (snip {snip_id!r}) not in frame_inventory."
        )
    row = frame_inventory_by_image.loc[image_id]
    for col in _PIXEL_SIZE_COLUMNS:
        if col in row.index and not pd.isna(row[col]):
            pixel_size = float(row[col])
            if not np.isfinite(pixel_size) or pixel_size <= 0:
                raise ValueError(
                    f"feature_extraction: invalid pixel size {pixel_size!r} in {col!r} for "
                    f"image_id {image_id!r} (snip {snip_id!r})."
                )
            return pixel_size
    raise ValueError(
        f"feature_extraction: frame_inventory row for image_id {image_id!r} (snip {snip_id!r}) "
        f"carries no micron calibration. Expected one of {_PIXEL_SIZE_COLUMNS}."
    )


def compute_per_snip_mask_features(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    *,
    per_mask_fn: Callable[[np.ndarray, float], dict],
    feature_columns: Sequence[str],
    output_columns: Sequence[str],
    mask_decoder: Callable[[dict], np.ndarray] = decode_binary_mask_rle,
) -> pd.DataFrame:
    """Run ``per_mask_fn(mask, pixel_size_um)`` for every snip, returning one row per snip.

    Joins snip_inventory -> frame_masks by ``mask_id`` (cross-checking ``image_id``), decodes
    ``mask_rle``, and looks up pixel size from frame_inventory by ``image_id``. The spine columns
    are copied verbatim from snip_inventory; ``feature_columns`` come from ``per_mask_fn``'s dict.
    """
    frame_masks_by_mask = frame_masks_df.set_index("mask_id")
    frame_inventory_by_image = frame_inventory_df.set_index("image_id")

    rows: list[dict] = []
    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])
        mask_id = str(snip["mask_id"])
        image_id = str(snip["image_id"])

        if mask_id not in frame_masks_by_mask.index:
            raise ValueError(
                f"feature_extraction: mask_id {mask_id!r} (snip {snip_id!r}) not in frame_masks."
            )
        mask_row = frame_masks_by_mask.loc[mask_id]
        if str(mask_row["image_id"]) != image_id:
            raise ValueError(
                f"feature_extraction: mask_id {mask_id!r} maps to image_id {mask_row['image_id']!r} "
                f"in frame_masks but snip {snip_id!r} declares {image_id!r}."
            )

        rle = mask_row["mask_rle"]
        rle = json.loads(str(rle)) if isinstance(rle, str) else rle
        mask = mask_decoder(rle)
        pixel_size_um = pixel_size_for_image(frame_inventory_by_image, image_id, snip_id)
        metrics = per_mask_fn(mask, pixel_size_um)

        row = {col: snip[col] for col in SNIP_FEATURE_TABLE_SPINE_COLUMNS}
        for col in feature_columns:
            row[col] = metrics[col]
        rows.append(row)

    return pd.DataFrame(rows, columns=list(output_columns))
