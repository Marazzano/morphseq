from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.segmentation.frame_masks_contract import (
    FRAME_MASKS_REQUIRED_COLUMNS,
    adapt_legacy_mask_rle_to_frame_masks,
)
from data_pipeline.segmentation.prompt_seeds import (
    build_prompt_seeds,
    kept_frame_detections,
    validate_prompt_seeds,
)
from data_pipeline.segmentation.validate_frame_masks import validate_frame_masks


def _model_frame_view() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": "20250912",
                "well_id": "20250912_B01",
                "image_id": f"20250912_B01_BF_t{t:04d}",
                "time_index": t,
                "channel_id": "BF",
                "source_image_path": f"frames/t{t:04d}.jpg",
                "image_width_px": 100,
                "image_height_px": 80,
            }
            for t in range(3)
        ]
    )


def _frame_detections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "detection_id": "det0",
                "image_id": "20250912_B01_BF_t0001",
                "time_index": 1,
                "bbox_x_min_px": 10,
                "bbox_y_min_px": 11,
                "bbox_x_max_px": 20,
                "bbox_y_max_px": 21,
                "is_kept": True,
            },
            {
                "detection_id": "det1",
                "image_id": "20250912_B01_BF_t0001",
                "time_index": 1,
                "bbox_x_min_px": 30,
                "bbox_y_min_px": 31,
                "bbox_x_max_px": 40,
                "bbox_y_max_px": 41,
                "is_kept": False,
            },
        ]
    )


def _legacy_mask_rle() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": "20250912",
                "well_id": "20250912_B01",
                "image_id": f"20250912_B01_BF_t{t:04d}",
                "embryo_id": "embryo_0",
                "channel_id": "BF",
                "time_int": t,
                "mask_rle": '{"counts": "abc", "size": [80, 100]}',
                "area_px": 42,
                "bbox_x_min": 10,
                "bbox_y_min": 11,
                "bbox_x_max": 20,
                "bbox_y_max": 21,
                "centroid_x_px": 15,
                "centroid_y_px": 16,
                "mask_confidence": 0.9,
                "source_image_path": f"frames/t{t:04d}.jpg",
                "image_width_px": 100,
                "image_height_px": 80,
                "source_backend": "sam2",
                "source_model": "hiera_l",
                "model_release": "sam2.1",
            }
            for t in range(3)
        ]
    )


def test_build_prompt_seeds_uses_only_kept_detections() -> None:
    detections = _frame_detections()
    kept = kept_frame_detections(detections)
    assert kept["detection_id"].tolist() == ["det0"]

    seeds = build_prompt_seeds(detections)
    assert seeds["detection_id"].tolist() == ["det0"]
    assert seeds.loc[0, "seed_id"] == "20250912_B01_BF_t0001_seed0000"

    validate_prompt_seeds(seeds, detections, _model_frame_view())


def test_validate_prompt_seeds_rejects_non_kept_detection() -> None:
    seeds = build_prompt_seeds(_frame_detections())
    seeds.loc[0, "detection_id"] = "det1"
    with pytest.raises(ValueError, match="non-kept detection_id"):
        validate_prompt_seeds(seeds, _frame_detections(), _model_frame_view())


def test_adapt_legacy_mask_rle_to_frame_masks_validates() -> None:
    frame_masks = adapt_legacy_mask_rle_to_frame_masks(_legacy_mask_rle())
    assert list(frame_masks.columns) == list(FRAME_MASKS_REQUIRED_COLUMNS)
    assert frame_masks["mask_id"].tolist() == [
        "20250912_B01_BF_t0000_m0000",
        "20250912_B01_BF_t0001_m0000",
        "20250912_B01_BF_t0002_m0000",
    ]
    assert frame_masks["track_id"].tolist() == ["embryo_0", "embryo_0", "embryo_0"]
    validate_frame_masks(frame_masks, _model_frame_view())


def test_validate_frame_masks_rejects_duplicate_image_track() -> None:
    frame_masks = adapt_legacy_mask_rle_to_frame_masks(_legacy_mask_rle())
    duplicate = frame_masks.iloc[[0]].copy()
    duplicate.loc[:, "mask_id"] = "extra_mask"
    frame_masks = pd.concat([frame_masks, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="image_id \\+ track_id"):
        validate_frame_masks(frame_masks, _model_frame_view())
