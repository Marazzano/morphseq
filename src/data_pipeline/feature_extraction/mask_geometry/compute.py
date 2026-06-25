"""mask_geometry compute — micron-aware geometry per snip, decoded from frame_masks RLE.

Flow: validated ``snip_inventory`` defines the universe (one row per snip). For each snip we
look up its mask row in ``frame_masks`` by ``mask_id``, decode the ``mask_rle`` payload to a
binary mask, read the micron calibration from ``frame_inventory`` by ``image_id``, and compute
geometry. We reuse the existing pure per-mask function; this module only owns the join and the
RLE decode (the legacy batch loop read a separate exported PNG — we read the canonical RLE).

One row per snip, ALWAYS — including invalid/degenerate masks (the pure function returns 0.0 for
empty masks). Feature extraction measures; QC excludes. No ``is_valid_snip`` filtering here.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from data_pipeline.feature_extraction.mask_geometry_metrics import compute_mask_geometry
from data_pipeline.segmentation.masks.mask_rle import decode_binary_mask_rle

from data_pipeline.feature_extraction.shared.feature_table_utils import (
    SNIP_FEATURE_TABLE_SPINE_COLUMNS,
)
from .contract import MASK_GEOMETRY_PAYLOAD_COLUMNS, MASK_GEOMETRY_TABLE_COLUMNS

# Micron calibration source on a frame_inventory row. Target name first, legacy name as fallback;
# fail loud if neither is present (a micron-aware feature cannot guess pixel size).
_PIXEL_SIZE_COLUMNS: tuple[str, ...] = ("source_micrometers_per_pixel", "micrometers_per_pixel")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Pure per-mask geometry (spec name; delegates to the existing implementation)
# ─────────────────────────────────────────────────────────────────────────────────────────────


def compute_mask_geometry_for_mask(mask: np.ndarray, pixel_size_um: float) -> dict:
    """Return micron-aware geometry for one binary mask. Delegates to the legacy pure function."""
    return compute_mask_geometry(mask, pixel_size_um)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Per-snip batch — join snip_inventory -> frame_masks -> frame_inventory, decode RLE, compute
# ─────────────────────────────────────────────────────────────────────────────────────────────


def _pixel_size_for_image(frame_inventory_by_image: pd.DataFrame, image_id: str, snip_id: str) -> float:
    if image_id not in frame_inventory_by_image.index:
        raise ValueError(
            f"mask_geometry: image_id {image_id!r} (snip {snip_id!r}) not found in frame_inventory. "
            "Every snip's image must have a frame_inventory row carrying the micron calibration."
        )
    row = frame_inventory_by_image.loc[image_id]
    for col in _PIXEL_SIZE_COLUMNS:
        if col in row.index and not pd.isna(row[col]):
            pixel_size = float(row[col])
            if not np.isfinite(pixel_size) or pixel_size <= 0:
                raise ValueError(
                    f"mask_geometry: invalid pixel size {pixel_size!r} in column {col!r} for "
                    f"image_id {image_id!r} (snip {snip_id!r})."
                )
            return pixel_size
    raise ValueError(
        f"mask_geometry: frame_inventory row for image_id {image_id!r} (snip {snip_id!r}) carries "
        f"no micron calibration. Expected one of {_PIXEL_SIZE_COLUMNS}."
    )


def compute_mask_geometry_features(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    *,
    mask_decoder: Callable[[dict], np.ndarray] = decode_binary_mask_rle,
) -> pd.DataFrame:
    """Return one mask_geometry_features row per snip in ``snip_inventory_df``.

    Join is by ``mask_id`` (snip_inventory carries it; the most specific per-frame mask key),
    cross-checked against ``image_id``. Pixel size comes from ``frame_inventory`` by ``image_id``.
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
                f"mask_geometry: mask_id {mask_id!r} (snip {snip_id!r}) not found in frame_masks. "
                "snip_inventory and frame_masks must agree on mask identity."
            )
        mask_row = frame_masks_by_mask.loc[mask_id]
        if str(mask_row["image_id"]) != image_id:
            raise ValueError(
                f"mask_geometry: mask_id {mask_id!r} maps to image_id {mask_row['image_id']!r} in "
                f"frame_masks but snip {snip_id!r} declares image_id {image_id!r}."
            )

        rle = json.loads(str(mask_row["mask_rle"])) if isinstance(mask_row["mask_rle"], str) else mask_row["mask_rle"]
        mask = mask_decoder(rle)
        pixel_size_um = _pixel_size_for_image(frame_inventory_by_image, image_id, snip_id)
        metrics = compute_mask_geometry_for_mask(mask, pixel_size_um)

        row = {col: snip[col] for col in SNIP_FEATURE_TABLE_SPINE_COLUMNS}
        for col in MASK_GEOMETRY_PAYLOAD_COLUMNS:
            row[col] = float(metrics[col])
        rows.append(row)

    return pd.DataFrame(rows, columns=MASK_GEOMETRY_TABLE_COLUMNS)
