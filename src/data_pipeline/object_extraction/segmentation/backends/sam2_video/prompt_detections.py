"""SAM2 prompt-detection vocabulary and validators.

Prompt detections are the kept detection boxes that seed SAM2 tracking for one well.
This vocabulary is SAM2-specific; generic frame-mask validation lives in
`segmentation.validate_frame_masks` and does not require this table.

`prompt_seeds.py` carries legacy seed vocabulary and is intentionally retained for
Session C. This module defines the Session B adapter vocabulary independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.frame_masks_contract import FRAME_MASKS_REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

PROMPT_DETECTION_COLUMNS: tuple[str, ...] = (
    "prompt_detection_id",
    "image_id",
    "time_index",
    "bbox_x_min_px",
    "bbox_y_min_px",
    "bbox_x_max_px",
    "bbox_y_max_px",
    "is_kept",
)


def empty_prompt_detections() -> pd.DataFrame:
    return pd.DataFrame(columns=PROMPT_DETECTION_COLUMNS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def validate_sam2_prompts(
    prompt_detections: pd.DataFrame,
    frame_inventory: pd.DataFrame,
) -> None:
    """Validate SAM2 prompt detections against the frame inventory.

    Checks that kept prompt rows exist, all reference valid image_ids, bbox bounds are
    in-bounds and internally consistent, and prompt_detection_id is unique.
    """
    _require_columns(prompt_detections, PROMPT_DETECTION_COLUMNS, "prompt_detections")
    _require_columns(
        frame_inventory,
        ("image_id", "image_width_px", "image_height_px"),
        "frame_inventory",
    )

    kept = prompt_detections[prompt_detections["is_kept"].astype(bool)]
    if kept.empty:
        raise ValueError(
            "prompt_detections must contain at least one kept row (is_kept=True)"
        )

    if prompt_detections["prompt_detection_id"].duplicated().any():
        dupes = (
            prompt_detections.loc[
                prompt_detections["prompt_detection_id"].duplicated(keep=False),
                "prompt_detection_id",
            ]
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"prompt_detections prompt_detection_id must be unique; examples: {dupes}"
        )

    frame_ids = set(frame_inventory["image_id"].astype(str))
    missing_frames = sorted(
        set(kept["image_id"].astype(str)) - frame_ids
    )
    if missing_frames:
        raise ValueError(
            f"prompt_detections reference image_id values outside frame_inventory: {missing_frames[:5]}"
        )

    frame_by_id = frame_inventory.set_index("image_id", drop=False)
    bbox_cols = ["bbox_x_min_px", "bbox_y_min_px", "bbox_x_max_px", "bbox_y_max_px"]
    for col in bbox_cols:
        values = pd.to_numeric(kept[col], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"prompt_detections {col} must be finite")

    for _, row in kept.iterrows():
        image_id = str(row["image_id"])
        frame = frame_by_id.loc[image_id]
        width = float(frame["image_width_px"])
        height = float(frame["image_height_px"])
        x0 = float(row["bbox_x_min_px"])
        y0 = float(row["bbox_y_min_px"])
        x1 = float(row["bbox_x_max_px"])
        y1 = float(row["bbox_y_max_px"])
        pid = str(row["prompt_detection_id"])
        if not (0 <= x0 < x1 <= width):
            raise ValueError(
                f"prompt_detections {pid} x bounds ({x0}, {x1}) outside image width {width}"
            )
        if not (0 <= y0 < y1 <= height):
            raise ValueError(
                f"prompt_detections {pid} y bounds ({y0}, {y1}) outside image height {height}"
            )


def select_segmentation_frame_view(
    frame_inventory: pd.DataFrame,
    frame_detections: pd.DataFrame,
) -> pd.DataFrame:
    """Return the projected frame rows SAM2 should see.

    The canonical per-well frame_inventory may include multiple image products for the same
    timepoint (e.g. z_stack planes plus a focus_stack projection). Detection runs on projected BF
    frames only; segmentation must use that same model view so propagated masks cannot be assigned
    to z-stack image_ids.

    Filters to projection rows first, then to channels observed in frame_detections. Raises if any
    detection image_id falls outside the resulting view, or if the view is empty.
    """
    model_rows = frame_inventory.copy()
    if "image_product_type" in model_rows.columns:
        model_rows = model_rows[
            model_rows["image_product_type"].astype(str) == "projection"
        ].copy()
    if "channel_id" in model_rows.columns and "channel_id" in frame_detections.columns:
        channels = set(frame_detections["channel_id"].dropna().astype(str))
        if channels:
            model_rows = model_rows[model_rows["channel_id"].astype(str).isin(channels)].copy()
    detection_image_ids = set(frame_detections["image_id"].astype(str))
    missing = sorted(detection_image_ids - set(model_rows["image_id"].astype(str)))
    if missing:
        raise ValueError(
            "frame_detections reference image_id values outside the SAM2 projection "
            f"frame view: {missing[:5]}"
        )
    if model_rows.empty:
        raise ValueError("SAM2 projection frame view is empty after frame_inventory filtering.")
    return model_rows


def validate_frame_masks_against_sam2_prompts(
    frame_masks: pd.DataFrame,
    prompt_detections: pd.DataFrame,
) -> None:
    """Validate that valid mask rows reference known SAM2 prompt detections.

    Valid mask rows that carry a non-NA prompt_detection_id must reference a known
    prompt_detection_id. No-mask placeholder rows (is_valid_mask=False) are exempt.
    """
    _require_columns(frame_masks, FRAME_MASKS_REQUIRED_COLUMNS, "frame_masks")
    _require_columns(prompt_detections, ("prompt_detection_id",), "prompt_detections")

    known_ids = set(prompt_detections["prompt_detection_id"].dropna().astype(str))
    valid_rows = frame_masks[frame_masks["is_valid_mask"].astype(bool)]
    referenced = set(valid_rows["prompt_detection_id"].dropna().astype(str))
    missing = sorted(referenced - known_ids)
    if missing:
        raise ValueError(
            f"frame_masks valid rows reference unknown prompt_detection_id values: {missing[:5]}. "
            "Use prompt_detection_id values from prompt_detections."
        )
