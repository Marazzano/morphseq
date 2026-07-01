"""Tests for detection/frame_detections_contract.py — manifests, id helpers, format constants."""

from data_pipeline.detection.frame_detections_contract import (
    ALLOWED_BBOX_FORMATS,
    BBOX_COLUMNS,
    CONFIDENCE_RANGE,
    NO_CANDIDATE_SUFFIX,
    REQUIRED_DETECTION_BLOCK,
    REQUIRED_FRAME_DETECTIONS_COLUMNS,
    detection_id,
    is_no_candidate_id,
    no_candidate_detection_id,
)
from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    DOWNSTREAM_FRAME_IDENTITY_BLOCK,
)


def test_required_columns_are_tuples():
    assert isinstance(REQUIRED_DETECTION_BLOCK, tuple)
    assert isinstance(REQUIRED_FRAME_DETECTIONS_COLUMNS, tuple)
    assert isinstance(ALLOWED_BBOX_FORMATS, tuple)


def test_frame_detections_columns_compose_identity_plus_detection():
    assert REQUIRED_FRAME_DETECTIONS_COLUMNS == (
        *DOWNSTREAM_FRAME_IDENTITY_BLOCK,
        *REQUIRED_DETECTION_BLOCK,
    )


def test_detection_block_has_expected_columns():
    expected = {
        "detection_id",
        "detector_backend",
        "detector_model_id",
        "class_label",
        "confidence",
        "bbox_x_min_px",
        "bbox_y_min_px",
        "bbox_x_max_px",
        "bbox_y_max_px",
        "bbox_format",
        "is_kept",
    }
    assert set(REQUIRED_DETECTION_BLOCK) == expected


def test_bbox_columns_subset_of_detection_block():
    assert set(BBOX_COLUMNS).issubset(set(REQUIRED_DETECTION_BLOCK))


def test_allowed_bbox_formats_is_xyxy_px_abs():
    assert ALLOWED_BBOX_FORMATS == ("xyxy_px_abs",)


def test_confidence_range_is_unit_interval():
    assert CONFIDENCE_RANGE == (0.0, 1.0)


def test_detection_id_format():
    assert detection_id("20250912_B01_BF_t0000", 3) == "20250912_B01_BF_t0000_det0003"


def test_no_candidate_detection_id_format():
    nc = no_candidate_detection_id("20250912_B01_BF_t0000")
    assert nc == "20250912_B01_BF_t0000_det_none"
    assert nc.endswith(NO_CANDIDATE_SUFFIX)


def test_is_no_candidate_id():
    assert is_no_candidate_id("20250912_B01_BF_t0000_det_none")
    assert not is_no_candidate_id("20250912_B01_BF_t0000_det0000")
