"""Adapt raw SAM2 video predictor output to the `frame_masks` contract.

The raw SAM2 output is an abstracted dict:
    `sam2_raw_output: dict[int, dict[int, np.ndarray]]`
mapping `sam2_frame_index -> {object_id -> bool_mask}`.

This module owns the translation from that shape to `FRAME_MASKS_REQUIRED_COLUMNS`
rows. No SAM2 API calls happen here; the adapter is pure data transformation.
The fake predictor (Session B) and the real SAM2 backend (Session C) both feed
into this same adapter.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.backends.sam2_video.prompt_detections import (
    PROMPT_DETECTION_COLUMNS,
)
from data_pipeline.object_extraction.segmentation.frame_masks_contract import (
    FRAME_MASKS_REQUIRED_COLUMNS,
    no_mask_frame_mask_row,
)
from data_pipeline.object_extraction.segmentation.masks.mask_geometry import mask_geometry
from data_pipeline.object_extraction.segmentation.masks.mask_rle import encode_binary_mask_rle
from data_pipeline.shared.identifiers import build_mask_id, build_track_id


SAM2_BACKEND_LABEL = "sam2_video"
SAM2_RLE_FORMAT = "morphseq_rle_v1"

REQUIRED_MODEL_FRAME_VIEW_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "image_id",
    "time_index",
    "z_index",
    "channel_id",
    "source_image_path",
    "image_width_px",
    "image_height_px",
    "sam2_frame_index",
)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def _prompt_detection_id_for_object(
    object_id: int,
    image_id: str,
    prompt_detections: pd.DataFrame,
) -> object:
    """Return the prompt_detection_id that seeded this SAM2 object if known, else NA.

    SAM2 assigns object_id from the seed order. We resolve by matching object_id to the
    nth prompt detection sorted by prompt_detection_id within this image_id. Returns NA
    if no matching prompt exists (e.g. the object was propagated from another frame).
    """
    if prompt_detections is None or prompt_detections.empty:
        return pd.NA
    frame_prompts = prompt_detections[
        prompt_detections["image_id"].astype(str) == image_id
    ].sort_values("prompt_detection_id", kind="mergesort")
    if object_id < len(frame_prompts):
        return str(frame_prompts.iloc[object_id]["prompt_detection_id"])
    return pd.NA


def adapt_sam2_well_output(
    well_id: str,
    sam2_raw_output: dict[int, dict[int, np.ndarray]],
    model_frame_view: pd.DataFrame,
    prompt_detections: pd.DataFrame | None = None,
    *,
    model_id: str = "sam2:unknown",
) -> pd.DataFrame:
    """Convert raw SAM2 output to a `frame_masks` DataFrame for one well.

    Args:
        well_id: Well identifier (constructor-minted).
        sam2_raw_output: Maps sam2_frame_index → {object_id → bool_mask}.
            Frames absent from this dict had no SAM2 output and receive no-mask
            placeholder rows.
        model_frame_view: Frame rows with sam2_frame_index column added by
            `build_sam2_frame_view`. Determines the canonical frame order and carries
            identity columns.
        prompt_detections: Optional prompt detection table used to resolve
            prompt_detection_id for each valid mask row. Columns must match
            PROMPT_DETECTION_COLUMNS. Pass None to leave prompt_detection_id as NA.
        model_id: Backend model identifier string for provenance columns.

    Returns:
        DataFrame with FRAME_MASKS_REQUIRED_COLUMNS. One row per (frame, object) for
        frames with masks, one no-mask placeholder row for frames with no masks.
    """
    _require_columns(model_frame_view, REQUIRED_MODEL_FRAME_VIEW_COLUMNS, "model_frame_view")
    if prompt_detections is not None:
        _require_columns(prompt_detections, PROMPT_DETECTION_COLUMNS, "prompt_detections")

    ordered = (
        model_frame_view.copy()
        .sort_values(["time_index", "image_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []

    for _, frame_row in ordered.iterrows():
        sam2_idx = int(frame_row["sam2_frame_index"])
        image_id = str(frame_row["image_id"])
        frame_masks_for_image = sam2_raw_output.get(sam2_idx, {})

        if not frame_masks_for_image:
            rows.append(no_mask_frame_mask_row(frame_row))
            continue

        for local_mask_index, object_id in enumerate(sorted(frame_masks_for_image)):
            mask: np.ndarray = frame_masks_for_image[object_id]
            rle = encode_binary_mask_rle(mask)
            geom = mask_geometry(mask)
            prompt_det_id = _prompt_detection_id_for_object(
                object_id, image_id, prompt_detections
            )
            rows.append(
                {
                    "experiment_id": str(frame_row["experiment_id"]),
                    "well_id": str(frame_row["well_id"]),
                    "image_id": image_id,
                    "time_index": int(frame_row["time_index"]),
                    "z_index": frame_row.get("z_index", pd.NA),
                    "channel_id": str(frame_row["channel_id"]),
                    "source_image_path": str(frame_row["source_image_path"]),
                    "image_width_px": frame_row["image_width_px"],
                    "image_height_px": frame_row["image_height_px"],
                    "prompt_detection_id": prompt_det_id,
                    "sam2_object_id": object_id,
                    "mask_id": build_mask_id(image_id, local_mask_index),
                    "track_id": build_track_id(well_id, object_id),
                    "mask_rle": json.dumps(rle),
                    "mask_rle_format": SAM2_RLE_FORMAT,
                    "area_px": float(geom["area_px"]),
                    "bbox_x_min_px": float(geom["bbox_x_min_px"]),
                    "bbox_y_min_px": float(geom["bbox_y_min_px"]),
                    "bbox_x_max_px": float(geom["bbox_x_max_px"]),
                    "bbox_y_max_px": float(geom["bbox_y_max_px"]),
                    "centroid_x_px": float(geom["centroid_x_px"]),
                    "centroid_y_px": float(geom["centroid_y_px"]),
                    "mask_confidence": 1.0,
                    "is_valid_mask": True,
                    "segmentation_backend": SAM2_BACKEND_LABEL,
                    "segmentation_model_id": model_id,
                    "tracking_backend": SAM2_BACKEND_LABEL,
                    "track_id_source": "sam2_object_id",
                }
            )

    if not rows:
        return pd.DataFrame(columns=FRAME_MASKS_REQUIRED_COLUMNS)
    return pd.DataFrame(rows, columns=FRAME_MASKS_REQUIRED_COLUMNS)
