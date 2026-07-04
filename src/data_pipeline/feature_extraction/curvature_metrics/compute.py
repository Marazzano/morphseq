"""curvature_metrics compute — geodesic centerline + B-spline curvature per snip, RLE-native.

Per-mask pipeline (the validated legacy method, vendored into this product):
    decoded mask
      -> smooth_mask_boundary        (blur+re-threshold; drops fins)
      -> extract_geodesic_centerline (skeleton graph -> geodesic path -> B-spline -> analytic κ)
      -> orient_centerline_head_to_tail
      -> compute_curvature_summary   (length, mean κ, baseline-deviation family, keypoints)

Low-information masks (skeleton too short to fit a cubic spline, or an empty mask) return documented
null metrics with ``centerline_point_count`` set — still exactly one row per snip. ``compute.py``
owns only this per-mask orchestration; the join/decode scaffold lives in the shared feature utils.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import (
    compute_per_snip_mask_features,
)
from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle

from .contract import CURVATURE_PAYLOAD_COLUMNS, CURVATURE_TABLE_COLUMNS
from .curvature_functions import compute_curvature_summary
from .geodesic_centerline import extract_geodesic_centerline
from .head_tail_orientation import orient_centerline_head_to_tail
from .mask_preprocessing import smooth_mask_boundary

# Column present for every snip (never null); the measured summaries are null for low-info masks.
_ALWAYS_PRESENT_COLUMN = "centerline_point_count"


def _null_metrics(centerline_point_count: int) -> dict:
    """The documented low-information result: every summary null, only the point count carried."""
    metrics = {col: np.nan for col in CURVATURE_PAYLOAD_COLUMNS}
    metrics[_ALWAYS_PRESENT_COLUMN] = int(centerline_point_count)
    return metrics


def compute_curvature_for_mask(mask: np.ndarray, pixel_size_um: float) -> dict:
    """Return the full curvature summary for one binary mask (one snip's worth of metrics).

    Runs the vendored geodesic pipeline. Any degenerate case (empty mask, skeleton too short to
    spline, or a skeleton-graph failure) is caught and returned as documented null metrics so the
    per-snip table always gets a row — feature extraction measures, it never raises on a bad mask.
    """
    if not np.any(mask):
        return _null_metrics(0)

    try:
        smoothed = smooth_mask_boundary(mask)
        centerline = extract_geodesic_centerline(smoothed, pixel_size_um=pixel_size_um)
    except (ValueError, RuntimeError):
        return _null_metrics(0)

    point_count = len(centerline.raw_xy)
    if len(centerline.smoothed_xy) == 0 or len(centerline.curvature_per_um) == 0:
        return _null_metrics(point_count)

    oriented_xy = orient_centerline_head_to_tail(centerline.smoothed_xy, smoothed)
    mean_curvature = float(np.mean(centerline.curvature_per_um))
    summary = compute_curvature_summary(oriented_xy, pixel_size_um, mean_curvature)
    summary[_ALWAYS_PRESENT_COLUMN] = int(point_count)
    return summary


def compute_curvature_features(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    *,
    mask_decoder: Callable[[dict], np.ndarray] = decode_binary_mask_rle,
) -> pd.DataFrame:
    """Return one curvature row per snip (join by mask_id, RLE-decoded)."""
    return compute_per_snip_mask_features(
        snip_inventory_df,
        frame_masks_df,
        frame_inventory_df,
        per_mask_fn=compute_curvature_for_mask,
        feature_columns=CURVATURE_PAYLOAD_COLUMNS,
        output_columns=CURVATURE_TABLE_COLUMNS,
        mask_decoder=mask_decoder,
    )
