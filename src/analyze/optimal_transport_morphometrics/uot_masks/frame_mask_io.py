"""Mask I/O helpers for UOT mask transport."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from src.data_pipeline.segmentation.grounded_sam2.mask_export import (
    load_labeled_mask,
    extract_individual_masks,
)

from src.analyze.utils.optimal_transport import UOTFrame, UOTFramePair


DEFAULT_USECOLS = [
    "embryo_id",
    "frame_index",
    "mask_rle",
    "mask_height_px",
    "mask_width_px",
    "image_id",
    "snip_id",
    "relative_time_s",
    "raw_time_s",
]


def _ensure_2d(mask: np.ndarray) -> np.ndarray:
    if mask.ndim > 2:
        mask = mask.squeeze()
        if mask.ndim > 2:
            mask = mask[..., 0]
    return mask


def load_mask_from_rle_counts(rle_counts: str, height_px: int, width_px: int) -> np.ndarray:
    rle_data = {"counts": rle_counts, "size": [int(height_px), int(width_px)]}
    mask = decode_mask_rle(rle_data)
    mask = _ensure_2d(mask)
    return mask.astype(np.uint8)


def load_mask_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_index: int,
    usecols: Optional[List[str]] = None,
) -> UOTFrame:
    if usecols is None:
        usecols = DEFAULT_USECOLS
    df = pd.read_csv(csv_path, usecols=usecols)
    subset = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if subset.empty:
        raise ValueError(f"No mask found for embryo_id={embryo_id} frame_index={frame_index}")
    row = subset.iloc[0]
    mask = load_mask_from_rle_counts(row["mask_rle"], row["mask_height_px"], row["mask_width_px"])
    return UOTFrame(embryo_mask=mask, meta=row.to_dict())


def load_mask_pair_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_index_src: int,
    frame_index_tgt: int,
    usecols: Optional[List[str]] = None,
) -> UOTFramePair:
    if usecols is None:
        usecols = DEFAULT_USECOLS
    df = pd.read_csv(csv_path, usecols=usecols)
    subset = df[(df["embryo_id"] == embryo_id) & (df["frame_index"].isin([frame_index_src, frame_index_tgt]))]
    if len(subset) < 2:
        raise ValueError(
            f"Expected two frames for embryo_id={embryo_id} at {frame_index_src},{frame_index_tgt}"
        )
    src_row = subset[subset["frame_index"] == frame_index_src].iloc[0]
    tgt_row = subset[subset["frame_index"] == frame_index_tgt].iloc[0]
    src = UOTFrame(
        embryo_mask=load_mask_from_rle_counts(
            src_row["mask_rle"], src_row["mask_height_px"], src_row["mask_width_px"]
        ),
        meta=src_row.to_dict(),
    )
    tgt = UOTFrame(
        embryo_mask=load_mask_from_rle_counts(
            tgt_row["mask_rle"], tgt_row["mask_height_px"], tgt_row["mask_width_px"]
        ),
        meta=tgt_row.to_dict(),
    )
    return UOTFramePair(src=src, tgt=tgt, pair_meta={"csv_path": str(csv_path)})


def load_mask_series_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_indices: Optional[Iterable[int]] = None,
    usecols: Optional[List[str]] = None,
) -> List[UOTFrame]:
    if usecols is None:
        usecols = DEFAULT_USECOLS
    df = pd.read_csv(csv_path, usecols=usecols)
    subset = df[df["embryo_id"] == embryo_id]
    if frame_indices is not None:
        frame_set = set(frame_indices)
        subset = subset[subset["frame_index"].isin(frame_set)]
    subset = subset.sort_values("frame_index")
    if subset.empty:
        raise ValueError(f"No masks found for embryo_id={embryo_id}")
    frames: List[UOTFrame] = []
    for _, row in subset.iterrows():
        mask = load_mask_from_rle_counts(row["mask_rle"], row["mask_height_px"], row["mask_width_px"])
        frames.append(UOTFrame(embryo_mask=mask, meta=row.to_dict()))
    return frames


def load_mask_from_png(
    mask_path: Path,
    embryo_id: Optional[str] = None,
    label: Optional[int] = None,
) -> np.ndarray:
    labeled = load_labeled_mask(mask_path)
    if label is None and embryo_id is None:
        label = 1
    if embryo_id is not None:
        masks = extract_individual_masks(labeled)
        if embryo_id not in masks:
            raise ValueError(f"embryo_id={embryo_id} not found in {mask_path}")
        return masks[embryo_id]
    return (labeled == label).astype(np.uint8)
