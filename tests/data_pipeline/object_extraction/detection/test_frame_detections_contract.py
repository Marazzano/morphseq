"""Tests for detection/frame_detections_contract.py — manifests, id helpers, format constants."""

import pandas as pd

from data_pipeline.object_extraction.detection.frame_detections_contract import (
    ALLOWED_BBOX_FORMATS,
    BBOX_COLUMNS,
    CONFIDENCE_RANGE,
    NO_CANDIDATE_SUFFIX,
    REQUIRED_DETECTION_BLOCK,
    REQUIRED_FRAME_DETECTIONS_COLUMNS,
    detection_id,
    has_kept_detections,
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


# ---------------------------------------------------------------------------
# has_kept_detections — "did this well have anything?" asked ONCE, here
# ---------------------------------------------------------------------------

def _detection_row(image_id: str, *, detection_id_value: str, is_kept: bool) -> dict:
    return {
        "image_id": image_id,
        "detection_id": detection_id_value,
        "is_kept": is_kept,
    }


class TestHasKeptDetections:
    """A well with no embryo is DATA, not an error.

    This predicate lives beside the vocabulary it reads (``is_kept`` / ``_det_none``) so that every
    consumer branching on "was anything found?" asks one question instead of re-deriving it. The
    first consumer is SAM2 prompting: SAM2 cannot be seeded with nothing, so its caller skips the
    model when this returns False.
    """

    def test_false_for_an_empty_table(self):
        assert has_kept_detections(pd.DataFrame()) is False

    def test_false_for_the_det_none_placeholder(self):
        """The exact row the detector emits for a well it found nothing in."""
        image_id = "20250912_B01_BF_t0000"
        df = pd.DataFrame([
            _detection_row(image_id, detection_id_value=no_candidate_detection_id(image_id), is_kept=False)
        ])
        assert has_kept_detections(df) is False
        assert is_no_candidate_id(df["detection_id"].iloc[0])

    def test_false_when_every_candidate_was_filtered_out(self):
        """Real candidates found, all dropped by filtering — still nothing to prompt with."""
        image_id = "20250912_B01_BF_t0000"
        df = pd.DataFrame([
            _detection_row(image_id, detection_id_value=detection_id(image_id, i), is_kept=False)
            for i in range(3)
        ])
        assert has_kept_detections(df) is False

    def test_true_when_any_row_is_kept(self):
        image_id = "20250912_B01_BF_t0000"
        df = pd.DataFrame([
            _detection_row(image_id, detection_id_value=detection_id(image_id, 0), is_kept=True),
            _detection_row(image_id, detection_id_value=detection_id(image_id, 1), is_kept=False),
        ])
        assert has_kept_detections(df) is True

    def test_false_when_the_column_is_absent(self):
        """Defensive: a frame without is_kept cannot claim to have kept anything."""
        assert has_kept_detections(pd.DataFrame([{"image_id": "x"}])) is False
