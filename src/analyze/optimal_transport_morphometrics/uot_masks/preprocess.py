"""Preprocessing pipeline for UOT mask transport."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask

from src.analyze.utils.optimal_transport import UOTConfig, pad_to_divisible


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
