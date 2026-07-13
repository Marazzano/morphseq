"""Tests for detection/validate_frame_detections.py — identity (reference) + detection (schema) layers."""

import pandas as pd
import pytest

from data_pipeline.object_extraction.detection.validate_frame_detections import (
    validate_frame_detection_block,
    validate_frame_detections,
)

# --- inline builders (self-contained, matching test_frame_inventory.py house style) ----------

EXP = "20250912"
WELL_INDEX = "B01"
WELL_ID = f"{EXP}_{WELL_INDEX}"
WIDTH = 1024
HEIGHT = 768


def _image_id(time_index: int, channel: str = "BF") -> str:
    return f"{WELL_ID}_{channel}_t{time_index:04d}"


def _make_inventory(n_frames: int, channel: str = "BF") -> pd.DataFrame:
    rows = []
    for t in range(n_frames):
        rows.append({
            "experiment_id": EXP,
            "well_index": WELL_INDEX,
            "channel_id": channel,
            "time_index": t,
            "elapsed_time_s": float(t * 120),
            "image_path": f"images/{_image_id(t, channel)}.png",
            "image_micrometers_per_pixel": 0.75,
            "image_width_px": WIDTH,
            "image_height_px": HEIGHT,
        })
    return pd.DataFrame(rows)


def _identity_fields(time_index: int, channel: str = "BF") -> dict:
    return {
        "experiment_id": EXP,
        "well_id": WELL_ID,
        "image_id": _image_id(time_index, channel),
        "time_index": time_index,
        "z_index": pd.NA,
        "channel_id": channel,
        "image_path": f"images/{_image_id(time_index, channel)}.png",
        "image_width_px": WIDTH,
        "image_height_px": HEIGHT,
    }


def _kept_row(time_index: int, idx: int = 0, **overrides) -> dict:
    row = {
        **_identity_fields(time_index),
        "detection_id": f"{_image_id(time_index)}_det{idx:04d}",
        "detector_backend": "groundingdino",
        "detector_model_id": "SwinT_OGC",
        "class_label": "embryo",
        "confidence": 0.9,
        "bbox_x_min_px": 100.0,
        "bbox_y_min_px": 120.0,
        "bbox_x_max_px": 300.0,
        "bbox_y_max_px": 320.0,
        "bbox_format": "xyxy_px_abs",
        "is_kept": True,
    }
    row.update(overrides)
    return row


def _rejected_row(time_index: int, idx: int = 1, **overrides) -> dict:
    row = _kept_row(time_index, idx=idx, confidence=0.20, is_kept=False)
    row.update(overrides)
    return row


def _placeholder_row(time_index: int, **overrides) -> dict:
    row = {
        **_identity_fields(time_index),
        "detection_id": f"{_image_id(time_index)}_det_none",
        "detector_backend": "groundingdino",
        "detector_model_id": "SwinT_OGC",
        "class_label": pd.NA,
        "confidence": pd.NA,
        "bbox_x_min_px": pd.NA,
        "bbox_y_min_px": pd.NA,
        "bbox_x_max_px": pd.NA,
        "bbox_y_max_px": pd.NA,
        "bbox_format": pd.NA,
        "is_kept": False,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Composed / identity (reference) layer
# ---------------------------------------------------------------------------


def test_valid_table_passes():
    inv = _make_inventory(2)
    df = pd.DataFrame([_kept_row(0, 0), _placeholder_row(1)])
    validate_frame_detections(df, inv)  # must not raise


def test_na_z_index_passes():
    """z_index synthesized as NA is allowed (nullable)."""
    inv = _make_inventory(1)
    df = pd.DataFrame([_kept_row(0, 0)])
    assert df["z_index"].isna().all()
    validate_frame_detections(df, inv)


def test_image_id_not_in_inventory_fails():
    inv = _make_inventory(1)
    df = pd.DataFrame([_kept_row(5, 0)])  # t0005 not in a 1-frame inventory
    with pytest.raises(ValueError, match="not present in reference_frame_inventory"):
        validate_frame_detections(df, inv)


def test_carried_column_disagreement_fails():
    inv = _make_inventory(1)
    df = pd.DataFrame([_kept_row(0, 0, image_width_px=999)])
    with pytest.raises(ValueError, match="disagreeing with reference_frame_inventory"):
        validate_frame_detections(df, inv)


def test_mixed_well_id_fails():
    inv = _make_inventory(2)
    rows = [_kept_row(0, 0), _kept_row(1, 0)]
    rows[1]["well_id"] = "20250912_Z99"
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="multiple well_id"):
        validate_frame_detections(df, inv)


def test_missing_identity_column_fails():
    inv = _make_inventory(1)
    df = pd.DataFrame([_kept_row(0, 0)]).drop(columns=["channel_id"])
    with pytest.raises(ValueError, match="missing required frame identity columns"):
        validate_frame_detections(df, inv)


# ---------------------------------------------------------------------------
# Detection (schema) layer
# ---------------------------------------------------------------------------


def test_duplicate_detection_id_fails():
    df = pd.DataFrame([_kept_row(0, 0), _kept_row(0, 0)])  # same id twice
    with pytest.raises(ValueError, match="non-unique detection_id"):
        validate_frame_detection_block(df)


def test_empty_backend_fails():
    df = pd.DataFrame([_kept_row(0, 0, detector_backend="")])
    with pytest.raises(ValueError, match="empty 'detector_backend'"):
        validate_frame_detection_block(df)


def test_empty_model_id_fails():
    df = pd.DataFrame([_kept_row(0, 0, detector_model_id="  ")])
    with pytest.raises(ValueError, match="empty 'detector_model_id'"):
        validate_frame_detection_block(df)


def test_confidence_out_of_range_fails():
    df = pd.DataFrame([_kept_row(0, 0, confidence=1.5)])
    with pytest.raises(ValueError, match="confidence"):
        validate_frame_detection_block(df)


def test_nan_confidence_on_kept_fails():
    df = pd.DataFrame([_kept_row(0, 0, confidence=float("nan"))])
    with pytest.raises(ValueError, match="confidence"):
        validate_frame_detection_block(df)


def test_bbox_outside_bounds_fails():
    df = pd.DataFrame([_kept_row(0, 0, bbox_x_max_px=5000.0)])
    with pytest.raises(ValueError, match="outside image bounds"):
        validate_frame_detection_block(df)


def test_bbox_min_not_less_than_max_fails():
    df = pd.DataFrame([_kept_row(0, 0, bbox_x_min_px=300.0, bbox_x_max_px=100.0)])
    with pytest.raises(ValueError, match="not < bbox_x_max_px"):
        validate_frame_detection_block(df)


def test_bad_bbox_format_fails():
    df = pd.DataFrame([_kept_row(0, 0, bbox_format="cxcywh")])
    with pytest.raises(ValueError, match="bbox_format"):
        validate_frame_detection_block(df)


def test_kept_row_with_na_bbox_fails():
    df = pd.DataFrame([_kept_row(0, 0, bbox_x_min_px=pd.NA, bbox_y_min_px=pd.NA,
                                 bbox_x_max_px=pd.NA, bbox_y_max_px=pd.NA)])
    with pytest.raises(ValueError, match="NA bbox"):
        validate_frame_detection_block(df)


def test_kept_placeholder_id_fails():
    """A no-candidate placeholder id can never be is_kept=True."""
    row = _placeholder_row(0)
    row["is_kept"] = True
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="placeholder row.*is_kept=True"):
        validate_frame_detection_block(df)


def test_non_bool_is_kept_fails():
    df = pd.DataFrame([_kept_row(0, 0, is_kept="yes")])
    with pytest.raises(ValueError, match="is_kept must be boolean"):
        validate_frame_detection_block(df)


def test_placeholder_with_na_values_passes():
    df = pd.DataFrame([_placeholder_row(0)])
    validate_frame_detection_block(df)  # must not raise


def test_rejected_candidate_row_valid():
    df = pd.DataFrame([_kept_row(0, 0), _rejected_row(0, 1)])
    validate_frame_detection_block(df)  # must not raise


def test_missing_detection_column_fails():
    df = pd.DataFrame([_kept_row(0, 0)]).drop(columns=["confidence"])
    with pytest.raises(ValueError, match="missing required detection columns"):
        validate_frame_detection_block(df)
