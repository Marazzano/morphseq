"""Tests for the SAM2 output adapter."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_pipeline.segmentation.backends.sam2_video.adapt_sam2_output import (
    SAM2_BACKEND_LABEL,
    SAM2_RLE_FORMAT,
    adapt_sam2_well_output,
)
from data_pipeline.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.segmentation.validate_frame_masks import validate_frame_mask_block
from data_pipeline.shared.identifiers import (
    build_image_id,
    build_mask_id,
    build_track_id,
    build_well_id,
    parse_mask_id,
    parse_track_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _well_id() -> str:
    return build_well_id("20250912", "B01")


def _model_frame_view(n_frames: int = 2) -> pd.DataFrame:
    well_id = _well_id()
    rows = []
    for t in range(n_frames):
        image_id = build_image_id(well_id, "BF", t)
        rows.append({
            "experiment_id": "20250912",
            "well_id": well_id,
            "image_id": image_id,
            "time_index": t,
            "z_index": pd.NA,
            "channel_id": "BF",
            "source_image_path": f"/data/{image_id}.png",
            "image_width_px": 100,
            "image_height_px": 80,
            "sam2_frame_index": t,
        })
    return pd.DataFrame(rows)


def _rect_mask(h: int = 80, w: int = 100, y1: int = 10, y2: int = 20, x1: int = 5, x2: int = 15) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_adapt_two_frames_two_objects():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=2)
    mask_a = _rect_mask()
    mask_b = _rect_mask(y1=30, y2=40, x1=60, x2=80)
    sam2_output = {
        0: {0: mask_a, 1: mask_b},
        1: {0: mask_a, 1: mask_b},
    }
    result = adapt_sam2_well_output(well_id, sam2_output, mfv)
    assert len(result) == 4  # 2 frames × 2 objects


def test_adapt_mask_id_and_track_id_are_constructor_minted():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=1)
    mask = _rect_mask()
    result = adapt_sam2_well_output(well_id, {0: {0: mask}}, mfv)
    assert len(result) == 1
    row = result.iloc[0]
    image_id, local_idx, is_no_mask = parse_mask_id(row["mask_id"])
    assert image_id == row["image_id"]
    assert local_idx == 0
    assert not is_no_mask
    parsed_well_id, track_idx = parse_track_id(row["track_id"])
    assert parsed_well_id == well_id
    assert track_idx == 0  # object_id 0 → track_index 0


def test_adapt_object_id_ordering():
    """local_mask_index comes from sorted(object_ids), not insertion order."""
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=1)
    mask_a = _rect_mask()
    mask_b = _rect_mask(y1=30, y2=40)
    # Provide objects in non-sorted order; adapter must sort
    result = adapt_sam2_well_output(well_id, {0: {5: mask_a, 2: mask_b}}, mfv)
    assert len(result) == 2
    local_indices = [parse_mask_id(row["mask_id"])[1] for _, row in result.iterrows()]
    assert sorted(local_indices) == [0, 1]


def test_adapt_no_mask_frame_produces_placeholder():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=2)
    mask = _rect_mask()
    # Frame 0 has a mask; frame 1 is absent from sam2_output
    result = adapt_sam2_well_output(well_id, {0: {0: mask}}, mfv)
    assert len(result) == 2
    no_mask_row = result[~result["is_valid_mask"].astype(bool)].iloc[0]
    assert no_mask_row["track_id"] is pd.NA or pd.isna(no_mask_row["track_id"])
    assert "mask_none" in no_mask_row["mask_id"]


def test_adapt_rle_round_trip():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=1)
    original_mask = _rect_mask()
    result = adapt_sam2_well_output(well_id, {0: {0: original_mask}}, mfv)
    row = result.iloc[0]
    rle = json.loads(row["mask_rle"])
    decoded = decode_binary_mask_rle(rle)
    np.testing.assert_array_equal(decoded, original_mask)


def test_adapt_rle_format_label():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=1)
    result = adapt_sam2_well_output(well_id, {0: {0: _rect_mask()}}, mfv)
    assert result.iloc[0]["mask_rle_format"] == SAM2_RLE_FORMAT


def test_adapt_backend_labels():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=1)
    result = adapt_sam2_well_output(well_id, {0: {0: _rect_mask()}}, mfv, model_id="sam2:v1")
    row = result.iloc[0]
    assert row["segmentation_backend"] == SAM2_BACKEND_LABEL
    assert row["tracking_backend"] == SAM2_BACKEND_LABEL
    assert row["segmentation_model_id"] == "sam2:v1"
    assert row["track_id_source"] == "sam2_object_id"


def test_adapt_output_passes_validate_frame_mask_block():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=3)
    mask = _rect_mask()
    # Frame 0 has 2 objects, frame 1 has 1 object, frame 2 has no masks
    sam2_output = {0: {0: mask, 1: mask}, 1: {0: mask}}
    result = adapt_sam2_well_output(well_id, sam2_output, mfv)
    validate_frame_mask_block(result)  # must not raise


def test_adapt_all_no_mask_frames():
    """All frames absent from sam2_output → all no-mask placeholders."""
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=2)
    result = adapt_sam2_well_output(well_id, {}, mfv)
    assert len(result) == 2
    assert not result["is_valid_mask"].astype(bool).any()
    validate_frame_mask_block(result)


def test_adapt_empty_model_frame_view_returns_empty():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=2).iloc[0:0]  # empty but correct schema
    result = adapt_sam2_well_output(well_id, {}, mfv)
    assert result.empty


def test_adapt_with_prompt_detections_resolves_prompt_id():
    well_id = _well_id()
    mfv = _model_frame_view(n_frames=1)
    image_id = mfv.iloc[0]["image_id"]
    prompt_det_id = f"{image_id}_det0000"
    prompt_detections = pd.DataFrame([{
        "prompt_detection_id": prompt_det_id,
        "image_id": image_id,
        "time_index": 0,
        "bbox_x_min_px": 5.0,
        "bbox_y_min_px": 5.0,
        "bbox_x_max_px": 50.0,
        "bbox_y_max_px": 40.0,
        "is_kept": True,
    }])
    result = adapt_sam2_well_output(well_id, {0: {0: _rect_mask()}}, mfv, prompt_detections)
    assert result.iloc[0]["prompt_detection_id"] == prompt_det_id
