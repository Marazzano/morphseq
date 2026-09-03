"""Tests for SAM2 prompt detection vocabulary and validators."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.backends.sam2_video.prompt_detections import (
    PROMPT_DETECTION_COLUMNS,
    empty_prompt_detections,
    unprompted_frame_masks,
    validate_frame_masks_against_sam2_prompts,
    validate_sam2_prompts,
)
from data_pipeline.object_extraction.segmentation.frame_masks_contract import (
    FRAME_MASKS_REQUIRED_COLUMNS,
    no_mask_frame_mask_row,
)
from data_pipeline.shared.identifiers import build_image_id, build_mask_id, build_track_id, build_well_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _frame_inventory(n_frames: int = 2) -> pd.DataFrame:
    well_id = build_well_id("20250912", "B01")
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
            "image_width_px": 100,
            "image_height_px": 80,
        })
    return pd.DataFrame(rows)


def _prompt_detections(frame_inventory: pd.DataFrame, n_per_frame: int = 1) -> pd.DataFrame:
    rows = []
    for _, frame in frame_inventory.iterrows():
        for i in range(n_per_frame):
            rows.append({
                "prompt_detection_id": f"{frame['image_id']}_det{i:04d}",
                "image_id": frame["image_id"],
                "time_index": frame["time_index"],
                "bbox_x_min_px": 5.0,
                "bbox_y_min_px": 5.0,
                "bbox_x_max_px": 50.0,
                "bbox_y_max_px": 40.0,
                "is_kept": True,
            })
    return pd.DataFrame(rows)


def _valid_mask_row(image_id: str, well_id: str, local_idx: int, prompt_id: str, frame_row: dict) -> dict:
    return {
        "experiment_id": frame_row["experiment_id"],
        "well_id": well_id,
        "image_id": image_id,
        "time_index": frame_row["time_index"],
        "z_index": pd.NA,
        "channel_id": "BF",
        "image_path": frame_row["image_path"],
        "image_width_px": frame_row["image_width_px"],
        "image_height_px": frame_row["image_height_px"],
        "prompt_detection_id": prompt_id,
        "sam2_object_id": 0,
        "mask_id": build_mask_id(image_id, local_idx),
        "track_id": build_track_id(well_id, 0),
        "mask_rle": '{"shape": [80, 100], "counts": [8000]}',
        "mask_rle_format": "morphseq_rle_v1",
        "area_px": 0.0,
        "bbox_x_min_px": 5.0,
        "bbox_y_min_px": 5.0,
        "bbox_x_max_px": 50.0,
        "bbox_y_max_px": 40.0,
        "centroid_x_px": 27.5,
        "centroid_y_px": 22.5,
        "mask_confidence": 0.9,
        "is_valid_mask": True,
        "segmentation_backend": "sam2_video",
        "segmentation_model_id": "sam2:test",
        "tracking_backend": "sam2_video",
        "track_id_source": "sam2_object_id",
    }


# ---------------------------------------------------------------------------
# validate_sam2_prompts
# ---------------------------------------------------------------------------

def test_validate_sam2_prompts_valid():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv)
    validate_sam2_prompts(prompts, inv)  # should not raise


def test_validate_sam2_prompts_empty_kept_fails():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv)
    prompts["is_kept"] = False
    with pytest.raises(ValueError, match="at least one kept row"):
        validate_sam2_prompts(prompts, inv)


def test_validate_sam2_prompts_empty_table_fails():
    inv = _frame_inventory()
    prompts = empty_prompt_detections()
    with pytest.raises(ValueError, match="at least one kept row"):
        validate_sam2_prompts(prompts, inv)


def test_validate_sam2_prompts_duplicate_id_fails():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv)
    dup = prompts.iloc[[0]].copy()
    prompts = pd.concat([prompts, dup], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_sam2_prompts(prompts, inv)


def test_validate_sam2_prompts_image_id_outside_inventory_fails():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv)
    prompts.loc[0, "image_id"] = "unknown_image_id"
    with pytest.raises(ValueError, match="outside frame_inventory"):
        validate_sam2_prompts(prompts, inv)


def test_validate_sam2_prompts_x_out_of_bounds_fails():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv)
    prompts.loc[0, "bbox_x_max_px"] = 200.0  # image is only 100px wide
    with pytest.raises(ValueError, match="x bounds"):
        validate_sam2_prompts(prompts, inv)


def test_validate_sam2_prompts_y_out_of_bounds_fails():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv)
    prompts.loc[0, "bbox_y_max_px"] = 200.0  # image is only 80px tall
    with pytest.raises(ValueError, match="y bounds"):
        validate_sam2_prompts(prompts, inv)


def test_validate_sam2_prompts_inverted_bbox_fails():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv)
    prompts.loc[0, "bbox_x_min_px"] = 60.0
    prompts.loc[0, "bbox_x_max_px"] = 10.0
    with pytest.raises(ValueError, match="x bounds"):
        validate_sam2_prompts(prompts, inv)


def test_validate_sam2_prompts_missing_column_fails():
    inv = _frame_inventory()
    prompts = _prompt_detections(inv).drop(columns=["is_kept"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_sam2_prompts(prompts, inv)


# ---------------------------------------------------------------------------
# validate_frame_masks_against_sam2_prompts
# ---------------------------------------------------------------------------

def test_validate_frame_masks_against_sam2_prompts_valid():
    inv = _frame_inventory(n_frames=1)
    prompts = _prompt_detections(inv)
    frame_row = inv.iloc[0].to_dict()
    image_id = frame_row["image_id"]
    well_id = frame_row["well_id"]
    prompt_id = prompts.iloc[0]["prompt_detection_id"]
    row = _valid_mask_row(image_id, well_id, 0, prompt_id, frame_row)
    frame_masks = pd.DataFrame([row], columns=FRAME_MASKS_REQUIRED_COLUMNS)
    validate_frame_masks_against_sam2_prompts(frame_masks, prompts)  # should not raise


def test_validate_frame_masks_against_sam2_prompts_unknown_prompt_fails():
    inv = _frame_inventory(n_frames=1)
    prompts = _prompt_detections(inv)
    frame_row = inv.iloc[0].to_dict()
    image_id = frame_row["image_id"]
    well_id = frame_row["well_id"]
    row = _valid_mask_row(image_id, well_id, 0, "NONEXISTENT_PROMPT_ID", frame_row)
    frame_masks = pd.DataFrame([row], columns=FRAME_MASKS_REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="unknown prompt_detection_id"):
        validate_frame_masks_against_sam2_prompts(frame_masks, prompts)


def test_validate_frame_masks_against_sam2_prompts_no_mask_placeholder_exempt():
    """No-mask placeholder rows with NA prompt_detection_id should pass."""
    inv = _frame_inventory(n_frames=1)
    prompts = _prompt_detections(inv)
    frame_row = inv.iloc[0]
    no_mask_row = no_mask_frame_mask_row(frame_row)
    frame_masks = pd.DataFrame([no_mask_row], columns=FRAME_MASKS_REQUIRED_COLUMNS)
    validate_frame_masks_against_sam2_prompts(frame_masks, prompts)  # should not raise


# ---------------------------------------------------------------------------
# Empty wells — a well with no embryo is DATA, not an error
# ---------------------------------------------------------------------------

class TestEmptyWellIsNotAnError:
    """The ARTIFACT a well with no embryo produces, and the validator that must stay strict.

    The predicate that decides "did this well have anything?" is ``has_kept_detections``, which
    lives with the contract that mints ``is_kept``/``_det_none`` — see
    ``tests/.../detection/test_frame_detections_contract.py``. This file covers the segmentation
    side: the rows emitted when SAM2 is skipped, and the fact that ``validate_sam2_prompts`` is
    NOT relaxed to accommodate them.
    """

    def test_unprompted_frame_masks_emits_one_no_mask_row_per_frame(self):
        inv = _frame_inventory(3)
        masks = unprompted_frame_masks(inv)
        assert len(masks) == 3
        assert list(masks.columns) == list(FRAME_MASKS_REQUIRED_COLUMNS)
        assert set(masks["image_id"]) == set(inv["image_id"])
        assert not masks["is_valid_mask"].any()
        assert (masks["area_px"] == 0.0).all()

    def test_unprompted_frame_masks_passes_the_real_validator(self):
        """The whole point: this artifact must be indistinguishable from a normal empty result."""
        from data_pipeline.object_extraction.segmentation.validate_frame_masks import (
            validate_frame_masks,
        )

        inv = _frame_inventory(2)
        validate_frame_masks(unprompted_frame_masks(inv), inv)

    def test_unprompted_frame_masks_uses_the_contract_placeholder_not_a_new_vocabulary(self):
        """mask_id must be the minted no-mask id — the validator enforces this form."""
        from data_pipeline.shared.identifiers import build_no_mask_id

        inv = _frame_inventory(1)
        masks = unprompted_frame_masks(inv)
        assert masks["mask_id"].iloc[0] == build_no_mask_id(str(inv["image_id"].iloc[0]))

    def test_validate_sam2_prompts_still_rejects_an_empty_prompt_set(self):
        """The validator must NOT be relaxed: it is the guard against a broken detection stage."""
        inv = _frame_inventory(1)
        with pytest.raises(ValueError, match="at least one kept row"):
            validate_sam2_prompts(empty_prompt_detections(), inv)
