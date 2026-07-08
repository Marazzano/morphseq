"""Validators for the target `frame_masks` product."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.frame_masks_contract import (
    FRAME_MASKS_REQUIRED_COLUMNS,
    MAX_VALID_MASK_AREA_FRACTION,
)
from data_pipeline.shared.identifiers import build_no_mask_id, parse_mask_id, parse_track_id


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

    parsed_mask_ids = frame_masks["mask_id"].map(_parse_mask_id_or_raise)
    parsed_image_ids = parsed_mask_ids.map(lambda parsed: parsed[0])
    if (parsed_image_ids.astype(str) != frame_masks["image_id"].astype(str)).any():
        raise ValueError("frame_masks mask_id must be constructor-minted from the row image_id.")

    is_no_mask = parsed_mask_ids.map(lambda parsed: parsed[2])

    bool_values = frame_masks["is_valid_mask"]
    if not bool_values.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("frame_masks is_valid_mask must be boolean")

    valid = frame_masks[frame_masks["is_valid_mask"]]
    if valid.duplicated(subset=["image_id", "track_id"]).any():
        raise ValueError("frame_masks valid rows must be unique by image_id + track_id")

    no_mask_rows = frame_masks[is_no_mask]
    for _, row in no_mask_rows.iterrows():
        expected_mask_id = build_no_mask_id(str(row["image_id"]))
        if row["mask_id"] != expected_mask_id:
            raise ValueError(
                "frame_masks no-mask placeholder mask_id must equal build_no_mask_id(image_id)."
            )
        if not pd.isna(row["track_id"]):
            raise ValueError("frame_masks no-mask placeholder track_id must be NA.")
        if bool(row["is_valid_mask"]):
            raise ValueError("frame_masks no-mask placeholder is_valid_mask must be False.")

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
        raise ValueError("frame_masks valid rows must have constructor-minted track_id")
    valid["track_id"].map(_parse_track_id_or_raise)
    if valid["mask_rle"].isna().any():
        raise ValueError("frame_masks valid rows must have mask_rle")
    if valid["mask_rle_format"].isna().any() or valid["mask_rle_format"].astype(str).str.len().eq(0).any():
        raise ValueError("frame_masks valid rows must have mask_rle_format")

    numeric_cols = [
        "image_width_px",
        "image_height_px",
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

    valid_width = pd.to_numeric(valid["image_width_px"], errors="coerce")
    valid_height = pd.to_numeric(valid["image_height_px"], errors="coerce")
    if (valid_width <= 0).any() or (valid_height <= 0).any():
        raise ValueError("frame_masks valid image dimensions must be positive")
    if (pd.to_numeric(valid["area_px"], errors="coerce") < 0).any():
        raise ValueError("frame_masks area_px must be non-negative")
    valid_area = pd.to_numeric(valid["area_px"], errors="coerce")
    image_area_px = valid_width * valid_height
    too_large = (image_area_px > 0) & ((valid_area / image_area_px) > MAX_VALID_MASK_AREA_FRACTION)
    if bool(np.any(too_large)):
        bad = valid.loc[too_large, "mask_id"].head(5).tolist()
        raise ValueError(
            "frame_masks valid mask(s) cover too much of the frame "
            f"({MAX_VALID_MASK_AREA_FRACTION:.2f} max area fraction); examples: {bad}"
        )


def validate_frame_masks(
    frame_masks: pd.DataFrame,
    frame_inventory: pd.DataFrame,
) -> None:
    """Validate frame mask rows against frame identity, independent of prompt tables."""
    validate_frame_mask_block(frame_masks)
    _require_columns(
        frame_inventory,
        ("image_id", "time_index", "source_image_path", "image_width_px", "image_height_px"),
        "frame_inventory",
    )

    frame_ids = set(frame_inventory["image_id"].astype(str))
    mask_frame_ids = set(frame_masks["image_id"].astype(str))
    missing = sorted(mask_frame_ids - frame_ids)
    if missing:
        raise ValueError(f"frame_masks contain image_id values outside frame_inventory: {missing[:5]}")

    frame_by_id = frame_inventory.set_index("image_id", drop=False)
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
                raise ValueError(f"frame_masks {col} disagrees with frame_inventory for {row['image_id']}")

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


def validate_frame_masks_against_prompt_detections(
    frame_masks: pd.DataFrame,
    prompt_detections: pd.DataFrame,
) -> None:
    """Validate optional prompt-detection references separately from the generic contract."""
    _require_columns(frame_masks, FRAME_MASKS_REQUIRED_COLUMNS, "frame_masks")
    if "prompt_detection_id" not in prompt_detections.columns:
        raise ValueError("prompt_detections missing required column(s): prompt_detection_id")

    known_prompt_ids = set(prompt_detections["prompt_detection_id"].dropna().astype(str))
    referenced = set(frame_masks["prompt_detection_id"].dropna().astype(str))
    missing_prompts = sorted(referenced - known_prompt_ids)
    if missing_prompts:
        raise ValueError(f"frame_masks reference unknown prompt_detection_id values: {missing_prompts[:5]}")


def _parse_mask_id_or_raise(mask_id: object) -> tuple[str, int | None, bool]:
    try:
        return parse_mask_id(str(mask_id))
    except ValueError as exc:
        raise ValueError("frame_masks mask_id must be constructor-minted; use build_mask_id/build_no_mask_id.") from exc


def _parse_track_id_or_raise(track_id: object) -> tuple[str, int]:
    try:
        return parse_track_id(str(track_id))
    except ValueError as exc:
        raise ValueError("frame_masks valid track_id must be constructor-minted; use build_track_id.") from exc
