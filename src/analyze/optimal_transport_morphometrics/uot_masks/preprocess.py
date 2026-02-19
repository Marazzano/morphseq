"""Preprocessing pipeline for UOT mask transport."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask

from analyze.utils.optimal_transport import UOTConfig, pad_to_divisible, UOTFrame
from .uot_grid import (
    CanonicalGridConfig,
    GridTransform,
    compute_grid_transform,
    apply_grid_transform,
    _rotate_and_scale_mask,
    CanonicalAligner,
)


def qc_mask(mask: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, dict]:
    mask_bool = mask.astype(bool)
    cleaned, stats = clean_embryo_mask(mask_bool, verbose=verbose)
    return cleaned.astype(np.uint8), stats


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("Empty mask for bbox computation.")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1


def compute_union_bbox(mask_a: np.ndarray, mask_b: np.ndarray, padding_px: int) -> Tuple[int, int, int, int]:
    union = (mask_a > 0) | (mask_b > 0)
    y0, y1, x0, x1 = _bbox_from_mask(union)
    y0 = max(0, y0 - padding_px)
    x0 = max(0, x0 - padding_px)
    y1 = min(union.shape[0], y1 + padding_px)
    x1 = min(union.shape[1], x1 + padding_px)
    return y0, y1, x0, x1


def crop_to_bbox(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return mask[y0:y1, x0:x1]


def align_masks_centroid(mask_src: np.ndarray, mask_tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    if mask_src.sum() == 0 or mask_tgt.sum() == 0:
        raise ValueError("Cannot align empty masks.")
    c_src = np.array(ndimage.center_of_mass(mask_src), dtype=np.float32)
    c_tgt = np.array(ndimage.center_of_mass(mask_tgt), dtype=np.float32)
    shift = (c_tgt - c_src).astype(np.float32)
    shifted = ndimage.shift(mask_src.astype(np.float32), shift=shift, order=0, mode="constant", cval=0.0)
    return (shifted > 0.5).astype(np.uint8), mask_tgt, (float(shift[0]), float(shift[1]))


def preprocess_pair(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    config: UOTConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    src_qc, src_stats = qc_mask(mask_src, verbose=verbose)
    tgt_qc, tgt_stats = qc_mask(mask_tgt, verbose=verbose)

    align_shift = (0.0, 0.0)
    if config.align_mode == "centroid":
        src_qc, tgt_qc, align_shift = align_masks_centroid(src_qc, tgt_qc)

    bbox = compute_union_bbox(src_qc, tgt_qc, padding_px=config.padding_px)
    src_crop = crop_to_bbox(src_qc, bbox)
    tgt_crop = crop_to_bbox(tgt_qc, bbox)

    if config.downsample_divisor and config.downsample_divisor > 1:
        src_pad, pad_hw = pad_to_divisible(src_crop, config.downsample_divisor)
        pad_h, pad_w = pad_hw
        tgt_pad = np.pad(tgt_crop, ((0, pad_h), (0, pad_w)), mode="constant")
    else:
        src_pad, tgt_pad = src_crop, tgt_crop
        pad_hw = (0, 0)

    meta = {
        "orig_shape": tuple(mask_src.shape),
        "bbox_y0y1x0x1": bbox,
        "pad_hw": pad_hw,
        "align_shift_yx": align_shift,
        "qc_stats_src": src_stats,
        "qc_stats_tgt": tgt_stats,
    }
    return src_pad, tgt_pad, meta


def preprocess_pair_canonical(
    src_frame: UOTFrame,
    tgt_frame: UOTFrame,
    config: UOTConfig,
    canonical_config: CanonicalGridConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess mask pair using canonical grid transformation.

    # TODO: Rename to source_target_joint_canonical_alignment() to better
    # reflect what this does: independently align src and tgt to canonical
    # grid, then jointly crop/pad into a shared bounding box for OT. The
    # current name is generic and doesn't convey the two-stage nature
    # (independent alignment → joint centering).

    IMPORTANT: For mask pairs, we need to preserve spatial relationships.
    This function assumes both masks are from the same embryo at the same
    resolution, so they share a common transform.

    Steps:
    1. QC clean both masks
    2. Compute shared transform based on union of masks
    3. Apply transform to both masks
    4. Return canonical masks + transform metadata

    Args:
        src_frame: Source frame with mask and metadata
        tgt_frame: Target frame with mask and metadata
        config: UOT configuration
        canonical_config: Canonical grid configuration
        verbose: Print debug info

    Returns:
        Tuple of (src_canonical, tgt_canonical, metadata)
    """
    mask_src = src_frame.embryo_mask
    mask_tgt = tgt_frame.embryo_mask

    # 1. QC clean
    src_qc, src_stats = qc_mask(mask_src, verbose=verbose)
    tgt_qc, tgt_stats = qc_mask(mask_tgt, verbose=verbose)

    # 2. Extract um_per_pixel from metadata
    src_um_per_pixel = src_frame.meta.get("um_per_pixel", np.nan)
    tgt_um_per_pixel = tgt_frame.meta.get("um_per_pixel", np.nan)

    if np.isnan(src_um_per_pixel) or np.isnan(tgt_um_per_pixel):
        raise ValueError(
            "um_per_pixel not found in frame metadata. "
            "Ensure load_mask_from_csv was used or add um_per_pixel to meta dict."
        )

    # Verify both masks are at same resolution (for same embryo)
    if abs(src_um_per_pixel - tgt_um_per_pixel) > 0.01:
        raise ValueError(
            f"Source and target have different resolutions: "
            f"{src_um_per_pixel:.3f} vs {tgt_um_per_pixel:.3f} um/px. "
            "Canonical grid preprocessing requires same-embryo mask pairs."
        )

    # 3. Load yolk masks if available (optional for now)
    src_yolk = src_frame.meta.get("yolk_mask", None)
    tgt_yolk = tgt_frame.meta.get("yolk_mask", None)

    # 4. Canonical alignment (independent per frame)
    canonical_config = CanonicalGridConfig(
        reference_um_per_pixel=canonical_config.reference_um_per_pixel,
        grid_shape_hw=canonical_config.grid_shape_hw,
        align_mode=canonical_config.align_mode,
        downsample_factor=canonical_config.downsample_factor,
        allow_flip=config.canonical_grid_allow_flip,
        anchor_mode=config.canonical_grid_anchor_mode,
        anchor_frac_yx=config.canonical_grid_anchor_frac_yx,
        clipping_threshold=config.canonical_grid_clipping_threshold,
        error_on_clip=config.canonical_grid_error_on_clip,
    )

    aligner = CanonicalAligner.from_config(canonical_config)
    use_pca = canonical_config.align_mode != "none"
    use_yolk = canonical_config.align_mode == "yolk"

    src_canonical, src_yolk_aligned, src_align_meta = aligner.align(
        src_qc,
        src_yolk,
        original_um_per_px=src_um_per_pixel,
        use_pca=use_pca,
        use_yolk=use_yolk,
    )
    tgt_canonical, tgt_yolk_aligned, tgt_align_meta = aligner.align(
        tgt_qc,
        tgt_yolk,
        original_um_per_px=tgt_um_per_pixel,
        use_pca=use_pca,
        use_yolk=use_yolk,
        reference_mask=src_canonical,
    )

    src_transform = GridTransform(
        source_um_per_pixel=src_um_per_pixel,
        target_um_per_pixel=canonical_config.reference_um_per_pixel,
        scale_factor=src_um_per_pixel / canonical_config.reference_um_per_pixel,
        rotation_rad=np.deg2rad(src_align_meta.get("rotation_deg", 0.0)),
        offset_yx_px=(0, 0),
        grid_shape_hw=canonical_config.grid_shape_hw,
        downsample_factor=canonical_config.downsample_factor,
        effective_um_per_pixel=canonical_config.reference_um_per_pixel * canonical_config.downsample_factor,
    )
    tgt_transform = GridTransform(
        source_um_per_pixel=tgt_um_per_pixel,
        target_um_per_pixel=canonical_config.reference_um_per_pixel,
        scale_factor=tgt_um_per_pixel / canonical_config.reference_um_per_pixel,
        rotation_rad=np.deg2rad(tgt_align_meta.get("rotation_deg", 0.0)),
        offset_yx_px=(0, 0),
        grid_shape_hw=canonical_config.grid_shape_hw,
        downsample_factor=canonical_config.downsample_factor,
        effective_um_per_pixel=canonical_config.reference_um_per_pixel * canonical_config.downsample_factor,
    )

    # 5. Build metadata
    meta = {
        "orig_shape_src": tuple(mask_src.shape),
        "orig_shape_tgt": tuple(mask_tgt.shape),
        "qc_stats_src": src_stats,
        "qc_stats_tgt": tgt_stats,
        "src_transform": src_transform,
        "tgt_transform": tgt_transform,
        "canonical_shape_hw": canonical_config.grid_shape_hw,
        "canonical_um_per_pixel": canonical_config.reference_um_per_pixel,
        "canonical_alignment_src": src_align_meta,
        "canonical_alignment_tgt": tgt_align_meta,
    }

    return src_canonical, tgt_canonical, meta
