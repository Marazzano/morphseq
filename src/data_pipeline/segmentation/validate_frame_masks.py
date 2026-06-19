"""Validators for the target `frame_masks` product."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.segmentation.frame_masks_contract import FRAME_MASKS_REQUIRED_COLUMNS


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def validate_frame_mask_block(frame_masks: pd.DataFrame) -> None:
    """Validate `frame_masks` row-level structure without external joins."""
    _require_columns(frame_masks, FRAME_MASKS_REQUIRED_COLUMNS, "frame_masks")

    if frame_masks["mask_id"].duplicated().any():
        dupes = frame_masks.loc[frame_masks["mask_id"].duplicated(keep=False), "mask_id"].head(5).tolist()
        raise ValueError(f"frame_masks mask_id values must be unique; examples: {dupes}")

    bool_values = frame_masks["is_valid_mask"]
    if not bool_values.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("frame_masks is_valid_mask must be boolean")

    valid = frame_masks[frame_masks["is_valid_mask"]]
    if valid.duplicated(subset=["image_id", "track_id"]).any():
        raise ValueError("frame_masks valid rows must be unique by image_id + track_id")

    required_non_empty = [
        "segmentation_backend",
        "segmentation_model_id",
        "tracking_backend",
        "track_id_source",
    ]
    for col in required_non_empty:
        if frame_masks[col].astype(str).str.len().eq(0).any():
            raise ValueError(f"frame_masks {col} must be non-empty")

    if valid["track_id"].isna().any() or valid["track_id"].astype(str).str.len().eq(0).any():
        raise ValueError("frame_masks valid rows must have track_id")
    if valid["mask_rle"].isna().any():
        raise ValueError("frame_masks valid rows must have mask_rle")
    if valid["mask_rle_format"].isna().any() or valid["mask_rle_format"].astype(str).str.len().eq(0).any():
        raise ValueError("frame_masks valid rows must have mask_rle_format")

    numeric_cols = [
        "area_px",
        "bbox_x_min_px",
        "bbox_y_min_px",
        "bbox_x_max_px",
        "bbox_y_max_px",
        "centroid_x_px",
        "centroid_y_px",
    ]
    for col in numeric_cols:
        values = pd.to_numeric(valid[col], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"frame_masks valid {col} values must be finite")

    if (pd.to_numeric(valid["area_px"], errors="coerce") < 0).any():
        raise ValueError("frame_masks area_px must be non-negative")


def validate_frame_masks(
    frame_masks: pd.DataFrame,
    model_frame_view: pd.DataFrame,
    prompt_seeds: pd.DataFrame | None = None,
) -> None:
    """Validate frame mask rows against frame identity and prompt seeds."""
    validate_frame_mask_block(frame_masks)
    _require_columns(
        model_frame_view,
        ("image_id", "time_index", "source_image_path", "image_width_px", "image_height_px"),
        "model_frame_view",
    )

    frame_ids = set(model_frame_view["image_id"].astype(str))
    mask_frame_ids = set(frame_masks["image_id"].astype(str))
    missing = sorted(mask_frame_ids - frame_ids)
    if missing:
        raise ValueError(f"frame_masks contain image_id values outside model_frame_view: {missing[:5]}")

    frame_by_id = model_frame_view.set_index("image_id", drop=False)
    compare_cols = [
        "time_index",
        "source_image_path",
        "image_width_px",
        "image_height_px",
    ]
    for _, row in frame_masks.iterrows():
        frame = frame_by_id.loc[row["image_id"]]
        for col in compare_cols:
            if str(row[col]) != str(frame[col]):
                raise ValueError(f"frame_masks {col} disagrees with model_frame_view for {row['image_id']}")

        if bool(row["is_valid_mask"]):
            width = float(row["image_width_px"])
            height = float(row["image_height_px"])
            x0 = float(row["bbox_x_min_px"])
            y0 = float(row["bbox_y_min_px"])
            x1 = float(row["bbox_x_max_px"])
            y1 = float(row["bbox_y_max_px"])
            cx = float(row["centroid_x_px"])
            cy = float(row["centroid_y_px"])
            if not (0 <= x0 < x1 <= width):
                raise ValueError(f"frame_masks bbox x bounds outside image for {row['mask_id']}")
            if not (0 <= y0 < y1 <= height):
                raise ValueError(f"frame_masks bbox y bounds outside image for {row['mask_id']}")
            if not (0 <= cx <= width and 0 <= cy <= height):
                raise ValueError(f"frame_masks centroid outside image for {row['mask_id']}")

    if prompt_seeds is not None and "seed_id" in prompt_seeds.columns:
        known_seed_ids = set(prompt_seeds["seed_id"].dropna().astype(str))
        referenced = set(frame_masks["seed_id"].dropna().astype(str))
        missing_seeds = sorted(referenced - known_seed_ids)
        if missing_seeds:
            raise ValueError(f"frame_masks reference unknown seed_id values: {missing_seeds[:5]}")
