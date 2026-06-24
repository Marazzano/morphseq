"""mask_quality_qc compute — structural mask-trustworthiness flags, one row per snip.

Three per-snip flags, decoded from CANONICAL frame_masks RLE (not raw SAM2 mask_rle):
  - edge_flag: mask touches the image boundary within margin_pixels;
  - discontinuous_mask_flag: more than one significant connected component;
  - overlapping_mask_flag: IoU with another DISTINCT physical embryo's mask in the same image
    exceeds iou_threshold (both snips of the pair flagged).

The edge/discontinuous checks are ported from the legacy `segmentation_quality_qc.py`; the
input is re-pointed to canonical masks via the shared `decode_binary_mask_rle`. Overlap is
grouped by `image_id` (the contract-unique frame identity) within a well, and keyed on the
EXPLICIT `physical_embryo_id` column — never by parsing `snip_id`.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd
from skimage.measure import label

from data_pipeline.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
)

from .config import MaskQualityQCConfig

# snip_inventory columns this product reads beyond the spine (operational, not identity spine).
_REQUIRED_FRAME_COLUMNS: tuple[str, ...] = ("image_id", "mask_id")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Pure per-mask checks
# ─────────────────────────────────────────────────────────────────────────────────────────────


def compute_edge_flag(mask: np.ndarray, *, margin_pixels: int) -> bool:
    """True if the mask touches any image border within ``margin_pixels``."""
    return bool(
        np.any(mask[:margin_pixels, :])
        or np.any(mask[-margin_pixels:, :])
        or np.any(mask[:, :margin_pixels])
        or np.any(mask[:, -margin_pixels:])
    )


def compute_discontinuous_flag(mask: np.ndarray, *, min_component_fraction: float) -> bool:
    """True if the mask has more than one component larger than ``min_component_fraction`` of the largest."""
    labeled = label(mask)
    num_components = int(np.max(labeled))
    if num_components <= 1:
        return False
    areas = [int(np.sum(labeled == i)) for i in range(1, num_components + 1)]
    largest = max(areas)
    min_significant = largest * min_component_fraction
    num_significant = sum(1 for area in areas if area > min_significant)
    return num_significant > 1


def compute_overlap_flags_for_image(
    image_masks_by_physical_embryo: dict[str, list[tuple[str, np.ndarray]]],
    *,
    iou_threshold: float,
) -> set[str]:
    """Return the set of snip_ids flagged for overlap within one image.

    ``image_masks_by_physical_embryo`` maps physical_embryo_id -> list of (snip_id, mask) in this
    image. IoU is computed only between masks of DISTINCT physical embryos; when a pair exceeds
    the threshold BOTH snips are flagged. Same-animal overlap is not ID confusion and is ignored.
    """
    flagged: set[str] = set()
    entries = [
        (phys, snip_id, mask)
        for phys, snips in image_masks_by_physical_embryo.items()
        for (snip_id, mask) in snips
    ]
    for i in range(len(entries)):
        phys_i, snip_i, mask_i = entries[i]
        for j in range(i + 1, len(entries)):
            phys_j, snip_j, mask_j = entries[j]
            if phys_i == phys_j:
                continue  # same animal — not ID confusion
            intersection = int(np.sum(mask_i & mask_j))
            union = int(np.sum(mask_i | mask_j))
            if union > 0 and (intersection / union) > iou_threshold:
                flagged.add(snip_i)
                flagged.add(snip_j)
    return flagged


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Per-snip batch — join snip_inventory -> frame_masks, decode RLE, run checks
# ─────────────────────────────────────────────────────────────────────────────────────────────


def compute_mask_quality_qc_flags(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    *,
    config: MaskQualityQCConfig,
    mask_decoder: Callable[[dict], np.ndarray] = decode_binary_mask_rle,
) -> pd.DataFrame:
    """Return one mask_quality_qc row per snip (full spine + the three flags)."""
    missing_cols = [c for c in (*SNIP_ID_SPINE_COLUMNS, *_REQUIRED_FRAME_COLUMNS) if c not in snip_inventory_df.columns]
    if missing_cols:
        raise ValueError(
            f"mask_quality_qc: snip_inventory missing required column(s): {', '.join(missing_cols)}."
        )

    frame_masks_by_mask = frame_masks_df.set_index("mask_id")

    decoded: dict[str, np.ndarray] = {}      # snip_id -> mask
    edge: dict[str, bool] = {}
    discontinuous: dict[str, bool] = {}
    # image_id -> physical_embryo_id -> [(snip_id, mask)]
    per_image: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {}

    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])
        mask_id = str(snip["mask_id"])
        image_id = str(snip["image_id"])
        physical_embryo_id = str(snip["physical_embryo_id"])

        mask = _decode_snip_mask(frame_masks_by_mask, mask_id, image_id, snip_id, config, mask_decoder)
        if mask is None:  # missing_mask_policy != fail -> not flagged
            edge[snip_id] = False
            discontinuous[snip_id] = False
            continue

        decoded[snip_id] = mask
        edge[snip_id] = compute_edge_flag(mask, margin_pixels=config.margin_pixels)
        discontinuous[snip_id] = compute_discontinuous_flag(
            mask, min_component_fraction=config.min_component_fraction
        )
        per_image.setdefault(image_id, {}).setdefault(physical_embryo_id, []).append((snip_id, mask))

    overlapping: set[str] = set()
    for image_masks_by_physical_embryo in per_image.values():
        overlapping |= compute_overlap_flags_for_image(
            image_masks_by_physical_embryo, iou_threshold=config.iou_threshold
        )

    out = snip_inventory_df[list(SNIP_ID_SPINE_COLUMNS)].copy()
    snip_ids = out["snip_id"].astype(str)
    out["edge_flag"] = pd.array([edge[s] for s in snip_ids], dtype=bool)
    out["discontinuous_mask_flag"] = pd.array([discontinuous[s] for s in snip_ids], dtype=bool)
    out["overlapping_mask_flag"] = pd.array([s in overlapping for s in snip_ids], dtype=bool)
    return out


def _decode_snip_mask(frame_masks_by_mask, mask_id, image_id, snip_id, config, mask_decoder):
    """Decode one snip's canonical mask, failing loud (or returning None) on a missing mask."""
    if mask_id not in frame_masks_by_mask.index:
        if config.missing_mask_policy == "fail":
            raise ValueError(
                f"mask_quality_qc: mask_id {mask_id!r} (snip {snip_id!r}) not found in frame_masks. "
                "snip_inventory and frame_masks must agree on mask identity."
            )
        return None
    mask_row = frame_masks_by_mask.loc[mask_id]
    if str(mask_row["image_id"]) != image_id:
        raise ValueError(
            f"mask_quality_qc: mask_id {mask_id!r} maps to image_id {mask_row['image_id']!r} in "
            f"frame_masks but snip {snip_id!r} declares image_id {image_id!r}."
        )
    rle = mask_row["mask_rle"]
    if pd.isna(rle) if np.isscalar(rle) else (rle is None):
        if config.missing_mask_policy == "fail":
            raise ValueError(
                f"mask_quality_qc: snip {snip_id!r} (mask_id {mask_id!r}) has no mask_rle payload."
            )
        return None
    rle = json.loads(str(rle)) if isinstance(rle, str) else rle
    return mask_decoder(rle)
