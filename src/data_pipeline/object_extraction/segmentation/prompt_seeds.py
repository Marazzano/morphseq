"""Prompt seed helpers for segmentation backends."""

from __future__ import annotations

import numpy as np
import pandas as pd


PROMPT_SEED_COLUMNS: tuple[str, ...] = (
    "seed_id",
    "seed_image_id",
    "seed_time_index",
    "detection_id",
    "prompt_type",
    "prompt_x_min_px",
    "prompt_y_min_px",
    "prompt_x_max_px",
    "prompt_y_max_px",
)

REQUIRED_DETECTION_PROMPT_COLUMNS: tuple[str, ...] = (
    "detection_id",
    "image_id",
    "time_index",
    "bbox_x_min_px",
    "bbox_y_min_px",
    "bbox_x_max_px",
    "bbox_y_max_px",
    "is_kept",
)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def kept_frame_detections(frame_detections: pd.DataFrame) -> pd.DataFrame:
    """Return the detection rows allowed to initialize segmentation."""
    _require_columns(frame_detections, ("is_kept",), "frame_detections")
    return frame_detections[frame_detections["is_kept"].astype(bool)].copy()


def build_prompt_seeds(frame_detections: pd.DataFrame) -> pd.DataFrame:
    """Convert kept detection boxes to backend-neutral box prompt seeds."""
    _require_columns(frame_detections, REQUIRED_DETECTION_PROMPT_COLUMNS, "frame_detections")
    kept = kept_frame_detections(frame_detections)
    if kept.empty:
        return pd.DataFrame(columns=PROMPT_SEED_COLUMNS)

    kept = kept.sort_values(["time_index", "image_id", "detection_id"], kind="mergesort")
    rows: list[dict[str, object]] = []
    for seed_idx, (_, row) in enumerate(kept.iterrows()):
        seed_image_id = str(row["image_id"])
        rows.append(
            {
                "seed_id": f"{seed_image_id}_seed{seed_idx:04d}",
                "seed_image_id": seed_image_id,
                "seed_time_index": int(row["time_index"]),
                "detection_id": str(row["detection_id"]),
                "prompt_type": "box_xyxy_px",
                "prompt_x_min_px": float(row["bbox_x_min_px"]),
                "prompt_y_min_px": float(row["bbox_y_min_px"]),
                "prompt_x_max_px": float(row["bbox_x_max_px"]),
                "prompt_y_max_px": float(row["bbox_y_max_px"]),
            }
        )
    return pd.DataFrame(rows, columns=PROMPT_SEED_COLUMNS)


def validate_prompt_seeds(
    prompt_seeds: pd.DataFrame,
    frame_detections: pd.DataFrame,
    model_frame_view: pd.DataFrame,
) -> None:
    """Validate prompt seeds against kept detections and frame bounds."""
    _require_columns(prompt_seeds, PROMPT_SEED_COLUMNS, "prompt_seeds")
    _require_columns(frame_detections, REQUIRED_DETECTION_PROMPT_COLUMNS, "frame_detections")
    _require_columns(model_frame_view, ("image_id", "image_width_px", "image_height_px"), "model_frame_view")

    if prompt_seeds.empty:
        raise ValueError("prompt_seeds must contain at least one row before segment_masks runs")
    if prompt_seeds["seed_id"].duplicated().any():
        raise ValueError("prompt_seeds seed_id values must be unique")

    kept_ids = set(kept_frame_detections(frame_detections)["detection_id"].astype(str))
    seed_detection_ids = set(prompt_seeds["detection_id"].astype(str))
    missing_detections = sorted(seed_detection_ids - kept_ids)
    if missing_detections:
        raise ValueError(f"prompt_seeds reference non-kept detection_id values: {missing_detections[:5]}")

    frame_by_image = model_frame_view.set_index("image_id", drop=False)
    missing_images = sorted(set(prompt_seeds["seed_image_id"].astype(str)) - set(frame_by_image.index.astype(str)))
    if missing_images:
        raise ValueError(f"prompt_seeds reference image_id values outside model_frame_view: {missing_images[:5]}")

    if not (prompt_seeds["prompt_type"] == "box_xyxy_px").all():
        raise ValueError("prompt_seeds prompt_type must be 'box_xyxy_px'")

    numeric_cols = [
        "prompt_x_min_px",
        "prompt_y_min_px",
        "prompt_x_max_px",
        "prompt_y_max_px",
    ]
    for col in numeric_cols:
        values = pd.to_numeric(prompt_seeds[col], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"prompt_seeds {col} must be finite")

    for _, row in prompt_seeds.iterrows():
        frame = frame_by_image.loc[row["seed_image_id"]]
        width = float(frame["image_width_px"])
        height = float(frame["image_height_px"])
        x0 = float(row["prompt_x_min_px"])
        y0 = float(row["prompt_y_min_px"])
        x1 = float(row["prompt_x_max_px"])
        y1 = float(row["prompt_y_max_px"])
        if not (0 <= x0 < x1 <= width):
            raise ValueError(f"prompt seed {row['seed_id']} has x bounds outside image")
        if not (0 <= y0 < y1 <= height):
            raise ValueError(f"prompt seed {row['seed_id']} has y bounds outside image")
