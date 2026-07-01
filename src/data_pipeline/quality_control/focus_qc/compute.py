"""focus_qc compute — interior structural-edge-content heuristic, one row per snip.

Decides whether the already-materialized projection image has enough internal embryo structure to
trust the snip. Targets the "ghost / structureless embryo" failure mode: bright, well-masked
embryos that are empty inside because they are badly out of focus. This is a low-information
exclusion heuristic, not a ground-truth out-of-focus detector — see config.py for the calibration
note.

Pixels are read through the materialized-image readers (resolve via frame_inventory, never a
reconstructed path); masks are decoded from the canonical frame_masks RLE via the shared decoder.
At the focus_qc projection default (downsample_factor: 1) mask and projection dims match, so no
mask-alignment step is needed — a shape mismatch fails loud rather than silently resizing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.filters import sobel

from data_pipeline.acquisition.image_materialization.materialized_image_readers import load_projection_image
from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_FRAME_PROVENANCE_COLUMNS,
    SNIP_ID_SPINE_COLUMNS,
)

from .config import FocusQCConfig

_REQUIRED_FRAME_COLUMNS: tuple[str, ...] = ("mask_id",)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Pure per-snip metric
# ─────────────────────────────────────────────────────────────────────────────────────────────


def compute_interior_strong_edge_fraction(
    projection_image: np.ndarray,
    mask: np.ndarray,
    *,
    config: FocusQCConfig,
) -> tuple[float, int]:
    """Return (interior_strong_edge_fraction, interior_n_px) for one snip.

    Erodes the mask (primary, then a smaller fallback, then the full mask) to strip the
    silhouette boundary — the body outline is strong even in bad focus and would otherwise drown
    out the internal-structure signal. Normalizes the crop with a robust 1st-99th percentile
    rescale, runs Sobel-on-Gaussian, and reports the fraction of interior pixels whose gradient
    exceeds strong_edge_sobel_threshold.
    """
    if projection_image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"focus_qc: projection image shape {projection_image.shape[:2]} does not match mask "
            f"shape {mask.shape[:2]}. At downsample_factor=1 these must agree; a mismatch is a "
            "contract failure, not something this product silently resizes around."
        )

    interior_mask = _eroded_interior(mask, config)
    interior_n_px = int(np.sum(interior_mask))
    if interior_n_px == 0:
        return 0.0, 0

    image = projection_image
    if image.ndim == 3:
        image = image.mean(axis=2)
    image = image.astype(np.float64)

    normalized = _robust_normalize(image, mask)
    gradient_image = sobel(gaussian_filter(normalized, sigma=1.0))

    strong_edge = gradient_image > config.strong_edge_sobel_threshold
    fraction = float(np.sum(strong_edge & interior_mask)) / interior_n_px
    return fraction, interior_n_px


def _eroded_interior(mask: np.ndarray, config: FocusQCConfig) -> np.ndarray:
    bool_mask = mask.astype(bool)
    for erosion_px in (config.interior_erosion_pixels, config.interior_erosion_fallback_pixels):
        if erosion_px <= 0:
            continue
        structure = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), dtype=bool)
        eroded = binary_erosion(bool_mask, structure=structure)
        if int(np.sum(eroded)) >= config.min_interior_pixels:
            return eroded
    return bool_mask


def _robust_normalize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust 1st-99th percentile rescale over the mask + local dilated context band."""
    structure = np.ones((11, 11), dtype=bool)
    context = binary_erosion(~mask.astype(bool), structure=structure, border_value=True)
    context = ~context  # dilate the mask by the same structure
    region = image[context] if np.any(context) else image.ravel()

    lo, hi = np.percentile(region, [1, 99])
    if hi <= lo:
        return np.zeros_like(image)
    normalized = (image - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Per-snip batch — join snip_inventory -> frame_inventory (pixels) + frame_masks (masks)
# ─────────────────────────────────────────────────────────────────────────────────────────────


def compute_focus_qc(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    *,
    config: FocusQCConfig,
    image_root: str | None = None,
) -> pd.DataFrame:
    """Return one focus_qc row per snip (full spine + interior_strong_edge_fraction + focus_flag)."""
    required = (*SNIP_ID_SPINE_COLUMNS, *SNIP_FRAME_PROVENANCE_COLUMNS, *_REQUIRED_FRAME_COLUMNS)
    missing_cols = [c for c in required if c not in snip_inventory_df.columns]
    if missing_cols:
        raise ValueError(
            f"focus_qc: snip_inventory missing required column(s): {', '.join(missing_cols)}."
        )

    frame_masks_by_mask = frame_masks_df.set_index("mask_id")

    fraction_by_snip: dict[str, float] = {}
    n_px_by_snip: dict[str, int] = {}

    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])
        image_id = str(snip["image_id"])
        mask_id = str(snip["mask_id"])

        mask = _decode_snip_mask(frame_masks_by_mask, mask_id, image_id, snip_id, config)

        try:
            projection_image = load_projection_image(
                frame_inventory_df,
                image_id=image_id,
                product_key=config.projection_product_key,
                image_root=image_root,
            )
        except ValueError as exc:
            if config.missing_projection_policy == "fail":
                raise ValueError(
                    f"focus_qc: could not load projection for snip {snip_id!r} "
                    f"(image_id={image_id!r}, product_key={config.projection_product_key!r}): {exc}"
                ) from exc
            fraction_by_snip[snip_id] = 0.0
            n_px_by_snip[snip_id] = 0
            continue

        fraction, n_px = compute_interior_strong_edge_fraction(
            projection_image, mask, config=config
        )
        fraction_by_snip[snip_id] = fraction
        n_px_by_snip[snip_id] = n_px

    out = snip_inventory_df[list(SNIP_ID_SPINE_COLUMNS)].copy()
    snip_ids = out["snip_id"].astype(str)
    out["interior_strong_edge_fraction"] = [fraction_by_snip[s] for s in snip_ids]
    out["interior_n_px"] = [n_px_by_snip[s] for s in snip_ids]
    out["focus_flag"] = pd.array(
        [
            fraction_by_snip[s] < config.interior_strong_edge_fraction_threshold
            for s in snip_ids
        ],
        dtype=bool,
    )
    return out


def _decode_snip_mask(frame_masks_by_mask, mask_id, image_id, snip_id, config):
    """Decode one snip's canonical mask, failing loud (or raising per policy) on a missing mask."""
    if mask_id not in frame_masks_by_mask.index:
        raise ValueError(
            f"focus_qc: mask_id {mask_id!r} (snip {snip_id!r}) not found in frame_masks. "
            "snip_inventory and frame_masks must agree on mask identity."
        )
    mask_row = frame_masks_by_mask.loc[mask_id]
    if str(mask_row["image_id"]) != image_id:
        raise ValueError(
            f"focus_qc: mask_id {mask_id!r} maps to image_id {mask_row['image_id']!r} in "
            f"frame_masks but snip {snip_id!r} declares image_id {image_id!r}."
        )
    rle = mask_row["mask_rle"]
    if pd.isna(rle) if np.isscalar(rle) else (rle is None):
        if config.missing_mask_policy == "fail":
            raise ValueError(
                f"focus_qc: snip {snip_id!r} (mask_id {mask_id!r}) has no mask_rle payload."
            )
        return None
    rle = json.loads(str(rle)) if isinstance(rle, str) else rle
    return decode_binary_mask_rle(rle)
