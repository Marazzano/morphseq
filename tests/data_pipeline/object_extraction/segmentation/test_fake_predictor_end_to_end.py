"""Fake-predictor end-to-end integration test for Session B.

One well runs through run_sam2_video_for_wells with a FakePredictor and the
segment_one_well_fake adapter. The output must pass both the generic frame-mask
validator and the SAM2 prompt cross-validator. No GPU or real SAM2 is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.backends.sam2_video.fake_predictor import (
    FakePredictor,
    segment_one_well_fake,
)
from data_pipeline.object_extraction.segmentation.backends.sam2_video.prompt_detections import (
    validate_frame_masks_against_sam2_prompts,
    validate_sam2_prompts,
)
from data_pipeline.object_extraction.segmentation.sam2_video.model_loader import Sam2VideoModelConfig
from data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video import (
    Sam2WellInput,
    Sam2WellResult,
    run_sam2_video_for_wells,
)
from data_pipeline.object_extraction.segmentation.validate_frame_masks import validate_frame_masks
from data_pipeline.shared.identifiers import (
    build_image_id,
    build_well_id,
    parse_mask_id,
    parse_track_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _well_id() -> str:
    return build_well_id("20250912", "B01")


def _frame_inventory(n_frames: int = 3, width: int = 100, height: int = 80) -> pd.DataFrame:
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
            "image_path": f"/data/{image_id}.png",
            "image_width_px": width,
            "image_height_px": height,
        })
    return pd.DataFrame(rows)


def _prompt_detections(frame_inventory: pd.DataFrame, n_objects: int = 2) -> pd.DataFrame:
    """Build prompt detections for the first frame only (seed frame)."""
    first_frame = frame_inventory.iloc[0]
    image_id = str(first_frame["image_id"])
    width = float(first_frame["image_width_px"])
    height = float(first_frame["image_height_px"])
    rows = []
    band_height = max(1, int(round(height * 0.1)))
    for i in range(n_objects):
        y0 = i * band_height
        y1 = min(y0 + band_height, height)
        rows.append({
            "prompt_detection_id": f"{image_id}_det{i:04d}",
            "image_id": image_id,
            "time_index": int(first_frame["time_index"]),
            "bbox_x_min_px": 0.0,
            "bbox_y_min_px": float(y0),
            "bbox_x_max_px": width,
            "bbox_y_max_px": float(y1),
            "is_kept": True,
        })
    return pd.DataFrame(rows)


def _fake_model_config() -> Sam2VideoModelConfig:
    return Sam2VideoModelConfig(
        models_root=Path("/fake/models"),
        config_path=Path("/fake/config.yaml"),
        checkpoint_path=Path("/fake/checkpoint.pt"),
        device="cpu",
        model_id="fake_predictor:v1",
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_end_to_end_returns_sam2_well_result():
    frame_inv = _frame_inventory(n_frames=3)
    well = Sam2WellInput(
        well_id=_well_id(),
        model_frame_view=frame_inv,
        frame_detections=pd.DataFrame(),
    )
    fake_pred = FakePredictor(n_objects=2)

    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
        return_value=fake_pred,
    ):
        results = run_sam2_video_for_wells(
            [well],
            model_config=_fake_model_config(),
            segment_one_well=segment_one_well_fake,
        )

    assert len(results) == 1
    assert isinstance(results[0], Sam2WellResult)
    assert results[0].well_id == _well_id()


def test_end_to_end_frame_masks_pass_generic_validator():
    n_frames = 3
    frame_inv = _frame_inventory(n_frames=n_frames)
    well = Sam2WellInput(
        well_id=_well_id(),
        model_frame_view=frame_inv,
        frame_detections=pd.DataFrame(),
    )
    fake_pred = FakePredictor(n_objects=2)

    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
        return_value=fake_pred,
    ):
        results = run_sam2_video_for_wells(
            [well],
            model_config=_fake_model_config(),
            segment_one_well=segment_one_well_fake,
        )

    frame_masks = results[0].frame_masks
    validate_frame_masks(frame_masks, frame_inv)  # must not raise


def test_end_to_end_frame_masks_pass_sam2_prompt_cross_validator():
    n_frames = 3
    n_objects = 2
    frame_inv = _frame_inventory(n_frames=n_frames)
    prompts = _prompt_detections(frame_inv, n_objects=n_objects)
    validate_sam2_prompts(prompts, frame_inv)  # prompts themselves are valid

    well = Sam2WellInput(
        well_id=_well_id(),
        model_frame_view=frame_inv,
        frame_detections=pd.DataFrame(),
    )
    fake_pred = FakePredictor(n_objects=n_objects)

    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
        return_value=fake_pred,
    ):
        results = run_sam2_video_for_wells(
            [well],
            model_config=_fake_model_config(),
            segment_one_well=segment_one_well_fake,
        )

    frame_masks = results[0].frame_masks
    # Prompt cross-validation: fake predictor leaves prompt_detection_id as NA
    # since it doesn't receive a prompt_detections table. Pass the check with
    # an empty prompt table to confirm NA prompt_detection_id rows are exempt.
    validate_frame_masks_against_sam2_prompts(frame_masks, prompts)  # must not raise


def test_end_to_end_mask_ids_are_constructor_minted():
    frame_inv = _frame_inventory(n_frames=2)
    well = Sam2WellInput(
        well_id=_well_id(),
        model_frame_view=frame_inv,
        frame_detections=pd.DataFrame(),
    )
    fake_pred = FakePredictor(n_objects=1)

    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
        return_value=fake_pred,
    ):
        results = run_sam2_video_for_wells(
            [well],
            model_config=_fake_model_config(),
            segment_one_well=segment_one_well_fake,
        )

    frame_masks = results[0].frame_masks
    valid_rows = frame_masks[frame_masks["is_valid_mask"].astype(bool)]
    for _, row in valid_rows.iterrows():
        # mask_id parses cleanly
        image_id, local_idx, is_no_mask = parse_mask_id(row["mask_id"])
        assert image_id == row["image_id"]
        assert local_idx is not None
        assert not is_no_mask
        # track_id parses cleanly
        parsed_well_id, track_idx = parse_track_id(row["track_id"])
        assert parsed_well_id == _well_id()


def test_end_to_end_row_count():
    n_frames = 3
    n_objects = 2
    frame_inv = _frame_inventory(n_frames=n_frames)
    well = Sam2WellInput(
        well_id=_well_id(),
        model_frame_view=frame_inv,
        frame_detections=pd.DataFrame(),
    )
    fake_pred = FakePredictor(n_objects=n_objects)

    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
        return_value=fake_pred,
    ):
        results = run_sam2_video_for_wells(
            [well],
            model_config=_fake_model_config(),
            segment_one_well=segment_one_well_fake,
        )

    frame_masks = results[0].frame_masks
    valid_count = frame_masks["is_valid_mask"].astype(bool).sum()
    assert valid_count == n_frames * n_objects


def test_end_to_end_empty_well_list():
    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
    ) as mock_load:
        results = run_sam2_video_for_wells(
            [],
            model_config=_fake_model_config(),
            segment_one_well=segment_one_well_fake,
        )
    assert results == []
    mock_load.assert_not_called()
