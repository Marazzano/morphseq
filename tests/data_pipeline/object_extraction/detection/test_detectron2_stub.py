"""Tests for the Detectron2 backend stub — interface-conformant, raises NotImplementedError."""

import inspect

import pytest

from data_pipeline.object_extraction.detection.backends import detectron2, groundingdino


def test_detectron2_detect_frame_raises():
    with pytest.raises(NotImplementedError):
        detectron2.detect_frame(
            model=object(),
            image_path="ignored.png",
            identity_row={"image_id": "x", "image_width_px": 10, "image_height_px": 10},
            detector_model_id="stub",
        )


def test_detectron2_adapter_signature_matches_groundingdino():
    """The stub must conform to the same detect_frame signature so the router stays backend-agnostic."""
    gd_params = list(inspect.signature(groundingdino.detect_frame).parameters)
    d2_params = list(inspect.signature(detectron2.detect_frame).parameters)
    assert gd_params == d2_params
