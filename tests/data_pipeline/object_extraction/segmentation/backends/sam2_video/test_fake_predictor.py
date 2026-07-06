"""Tests for the deterministic fake SAM2 predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.backends.sam2_video.fake_predictor import (
    FakePredictor,
    segment_one_well_fake,
)
from data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video import Sam2WellInput
from data_pipeline.object_extraction.segmentation.validate_frame_masks import validate_frame_mask_block
from data_pipeline.shared.identifiers import build_image_id, build_well_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _well_id() -> str:
    return build_well_id("20250912", "B01")


def _model_frame_view(n_frames: int = 2, width: int = 100, height: int = 80) -> pd.DataFrame:
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
            "image_width_px": width,
            "image_height_px": height,
        })
    return pd.DataFrame(rows)


def _well_input(n_frames: int = 2) -> Sam2WellInput:
    well_id = _well_id()
    mfv = _model_frame_view(n_frames)
    return Sam2WellInput(
        well_id=well_id,
        model_frame_view=mfv,
        frame_detections=pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# FakePredictor unit tests
# ---------------------------------------------------------------------------

def test_fake_predictor_returns_n_objects():
    pred = FakePredictor(n_objects=3)
    masks = pred.predict_frame(80, 100, 0)
    assert len(masks) == 3
    assert set(masks.keys()) == {0, 1, 2}


def test_fake_predictor_masks_are_bool_2d():
    pred = FakePredictor(n_objects=2)
    masks = pred.predict_frame(80, 100, 0)
    for mask in masks.values():
        assert mask.dtype == bool
        assert mask.ndim == 2
        assert mask.shape == (80, 100)


def test_fake_predictor_masks_are_non_empty():
    pred = FakePredictor(n_objects=1, mask_fill_fraction=0.1)
    masks = pred.predict_frame(80, 100, 0)
    assert masks[0].any()


def test_fake_predictor_is_deterministic():
    pred = FakePredictor(n_objects=2)
    masks_a = pred.predict_frame(80, 100, 0)
    masks_b = pred.predict_frame(80, 100, 0)
    for obj_id in masks_a:
        np.testing.assert_array_equal(masks_a[obj_id], masks_b[obj_id])


def test_fake_predictor_invalid_n_objects():
    with pytest.raises(ValueError, match="n_objects"):
        FakePredictor(n_objects=0)


def test_fake_predictor_invalid_fill_fraction():
    with pytest.raises(ValueError, match="mask_fill_fraction"):
        FakePredictor(mask_fill_fraction=0.0)


# ---------------------------------------------------------------------------
# segment_one_well_fake tests
# ---------------------------------------------------------------------------

def test_segment_one_well_fake_returns_dataframe():
    pred = FakePredictor(n_objects=1)
    well = _well_input(n_frames=2)
    result = segment_one_well_fake(pred, well)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_segment_one_well_fake_output_passes_block_validator():
    pred = FakePredictor(n_objects=2)
    well = _well_input(n_frames=3)
    result = segment_one_well_fake(pred, well)
    validate_frame_mask_block(result)  # must not raise


def test_segment_one_well_fake_row_count():
    pred = FakePredictor(n_objects=2)
    n_frames = 3
    well = _well_input(n_frames=n_frames)
    result = segment_one_well_fake(pred, well)
    # 2 objects × 3 frames = 6 valid rows; no no-mask rows since all frames get masks
    assert len(result) == n_frames * 2


def test_segment_one_well_fake_is_deterministic():
    pred = FakePredictor(n_objects=1)
    well = _well_input(n_frames=2)
    result_a = segment_one_well_fake(pred, well)
    result_b = segment_one_well_fake(pred, well)
    pd.testing.assert_frame_equal(result_a, result_b)
