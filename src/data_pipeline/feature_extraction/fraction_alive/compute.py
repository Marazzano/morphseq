"""fraction_alive compute — viability fraction per snip from embryo mask + VIA mask, RLE-native.

Embryo mask comes from frame_masks RLE (by mask_id); the VIA mask is a per-snip auxiliary PNG
looked up by snip_id. Reuses the legacy pure compute_fraction_alive. Empty embryo mask -> null
(documented); missing VIA mask -> policy (fail loud by default).
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd
import skimage.io as io

from data_pipeline.feature_extraction.fraction_alive._legacy_compute import compute_fraction_alive
from data_pipeline.feature_extraction.shared.feature_table_utils import SNIP_SPINE_COLUMNS
from data_pipeline.segmentation.masks.mask_rle import decode_binary_mask_rle

from .contract import FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS
from .via_masks import build_via_mask_lookup

MISSING_VIA_FAIL = "fail"
MISSING_VIA_NULL = "null"


def compute_fraction_alive_features(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    via_mask_dir,
    *,
    missing_via_policy: str = MISSING_VIA_FAIL,
    mask_decoder: Callable[[dict], np.ndarray] = decode_binary_mask_rle,
) -> pd.DataFrame:
    """Return one fraction_alive_features row per snip (embryo RLE vs per-snip VIA mask)."""
    frame_masks_by_mask = frame_masks_df.set_index("mask_id")
    via_lookup = build_via_mask_lookup(snip_inventory_df["snip_id"], via_mask_dir)

    rows: list[dict] = []
    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])
        mask_id = str(snip["mask_id"])

        mask_row = frame_masks_by_mask.loc[mask_id]
        rle = mask_row["mask_rle"]
        rle = json.loads(str(rle)) if isinstance(rle, str) else rle
        embryo_mask = mask_decoder(rle)

        via_path = via_lookup[snip_id]
        if not via_path.exists():
            if missing_via_policy == MISSING_VIA_NULL:
                fraction = np.nan
            else:
                raise ValueError(
                    f"fraction_alive: VIA mask missing for snip {snip_id!r} at {via_path}. "
                    "Generate auxiliary masks named by build_mask_id(snip_id, ...), or set "
                    "missing_via_policy='null'."
                )
        else:
            via_mask = io.imread(via_path)
            fraction = compute_fraction_alive(embryo_mask, via_mask)

        row = {col: snip[col] for col in SNIP_SPINE_COLUMNS}
        row["fraction_alive"] = fraction
        rows.append(row)

    return pd.DataFrame(rows, columns=FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS)
