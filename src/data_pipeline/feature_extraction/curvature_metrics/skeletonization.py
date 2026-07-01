"""curvature_metrics skeletonization + pure curvature math.

Product-local home for the centerline/curvature computation (migrated off the legacy
``feature_extraction/core/`` layer). ``compute.py`` is the only caller; the pure
``compute_curvature_metrics`` takes a decoded mask and returns per-snip metrics — no disk I/O.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from data_pipeline.object_extraction.segmentation.shared.mask_processing import clean_embryo_mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n_components = ndimage.label(np.asarray(mask, dtype=bool))
    if n_components <= 1:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    if sizes.size <= 1:
        return np.asarray(mask, dtype=bool)
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return labeled == largest_label


def skeletonize_embryo_mask(
    mask: np.ndarray,
    *,
    min_component_size: int = 32,
) -> np.ndarray:
    """Return the cleaned skeleton of an embryo mask."""
    clean = clean_embryo_mask(mask, min_component_size=min_component_size)
    skel = skeletonize(clean)
    if not np.any(skel):
        return np.zeros_like(clean, dtype=bool)
    return _largest_component(skel)


def extract_centerline_points(
    mask: np.ndarray,
    *,
    min_component_size: int = 32,
) -> np.ndarray:
    """Return ordered centerline points in x/y pixel coordinates."""
    skel = skeletonize_embryo_mask(mask, min_component_size=min_component_size)
    coords_yx = np.argwhere(skel)
    if coords_yx.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)

    coords_xy = coords_yx[:, ::-1].astype(np.float64)
    centered = coords_xy - coords_xy.mean(axis=0, keepdims=True)
    if coords_xy.shape[0] > 3:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        order = np.argsort(centered @ axis)
        coords_xy = coords_xy[order]

    # Remove repeated points that can break arc-length derivatives.
    keep = np.ones(len(coords_xy), dtype=bool)
    if len(coords_xy) > 1:
        deltas = np.diff(coords_xy, axis=0)
        keep[1:] = np.any(np.abs(deltas) > 0, axis=1)
    return coords_xy[keep]


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values.astype(np.float64)
    window = min(int(window), int(values.size))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def compute_curvature_metrics(mask: np.ndarray, pixel_size_um: float) -> Dict[str, float]:
    """Compute curvature summaries from an embryo mask."""
    centerline_xy_px = extract_centerline_points(mask)
    if centerline_xy_px.shape[0] < 3:
        return {
            "mean_curvature_per_um": np.nan,
            "median_curvature_per_um": np.nan,
            "max_curvature_per_um": np.nan,
            "centerline_length_um": np.nan,
            "centerline_point_count": int(centerline_xy_px.shape[0]),
        }

    centerline_xy_um = centerline_xy_px * float(pixel_size_um)
    diffs = np.diff(centerline_xy_um, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    valid = s > 0
    if np.count_nonzero(valid) < 3:
        return {
            "mean_curvature_per_um": np.nan,
            "median_curvature_per_um": np.nan,
            "max_curvature_per_um": np.nan,
            "centerline_length_um": float(s[-1]),
            "centerline_point_count": int(centerline_xy_px.shape[0]),
        }

    x = _smooth_series(centerline_xy_um[:, 0])
    y = _smooth_series(centerline_xy_um[:, 1])
    dx_ds = np.gradient(x, s, edge_order=1)
    dy_ds = np.gradient(y, s, edge_order=1)
    d2x_ds2 = np.gradient(dx_ds, s, edge_order=1)
    d2y_ds2 = np.gradient(dy_ds, s, edge_order=1)
    denom = np.power(dx_ds ** 2 + dy_ds ** 2, 1.5)
    numer = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
    curvature = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    curvature = curvature[np.isfinite(curvature)]

    if curvature.size == 0:
        mean_curvature = np.nan
        median_curvature = np.nan
        max_curvature = np.nan
    else:
        mean_curvature = float(np.mean(curvature))
        median_curvature = float(np.median(curvature))
        max_curvature = float(np.max(curvature))

    return {
        "mean_curvature_per_um": mean_curvature,
        "median_curvature_per_um": median_curvature,
        "max_curvature_per_um": max_curvature,
        "centerline_length_um": float(s[-1]),
        "centerline_point_count": int(centerline_xy_px.shape[0]),
    }
