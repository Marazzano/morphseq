"""curvature_metrics compute — centerline length + curvature per snip, RLE-native.

Reuses the legacy pure ``compute_curvature_metrics`` (centerline via skeletonization). Low-info
masks return documented null metrics (NaN) per the contract — still one row per snip.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from data_pipeline.feature_extraction.core.curvature_metrics import compute_curvature_metrics
from data_pipeline.feature_extraction.shared.feature_table_utils import (
    compute_per_snip_mask_features,
)
from data_pipeline.segmentation.masks.mask_rle import decode_binary_mask_rle

from .contract import CURVATURE_PAYLOAD_COLUMNS, CURVATURE_TABLE_COLUMNS


def compute_curvature_for_mask(mask: np.ndarray, pixel_size_um: float) -> dict:
    """Return centerline/curvature metrics for one binary mask. Delegates to the legacy function."""
    return compute_curvature_metrics(mask, pixel_size_um)


def compute_curvature_features(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    *,
    mask_decoder: Callable[[dict], np.ndarray] = decode_binary_mask_rle,
) -> pd.DataFrame:
    """Return one curvature_features row per snip (join by mask_id, RLE-decoded)."""
    return compute_per_snip_mask_features(
        snip_inventory_df,
        frame_masks_df,
        frame_inventory_df,
        per_mask_fn=compute_curvature_for_mask,
        feature_columns=CURVATURE_PAYLOAD_COLUMNS,
        output_columns=CURVATURE_TABLE_COLUMNS,
        mask_decoder=mask_decoder,
    )
