"""Tests for detection/kept_frame_detections.py — consumer view + CSV round-trip."""

import pandas as pd
import pytest

from data_pipeline.detection.frame_detections_contract import (
    REQUIRED_FRAME_DETECTIONS_COLUMNS,
)
from data_pipeline.detection.kept_frame_detections import (
    kept_frame_detections,
    read_frame_detections_csv,
)

EXP = "20250912"
WELL_ID = f"{EXP}_B01"
WIDTH, HEIGHT = 1024, 768


def _image_id(t: int) -> str:
    return f"{WELL_ID}_BF_t{t:04d}"


def _make_inventory(n: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "experiment_id": EXP,
        "well_index": "B01",
        "channel_id": "BF",
        "time_index": t,
        "elapsed_time_s": float(t * 120),
        "source_image_path": f"images/{_image_id(t)}.png",
        "source_micrometers_per_pixel": 0.75,
        "image_width_px": WIDTH,
        "image_height_px": HEIGHT,
    } for t in range(n)])


def _identity(t: int) -> dict:
    return {
        "experiment_id": EXP, "well_id": WELL_ID, "image_id": _image_id(t),
        "time_index": t, "z_index": pd.NA, "channel_id": "BF",
        "source_image_path": f"images/{_image_id(t)}.png",
        "image_width_px": WIDTH, "image_height_px": HEIGHT,
    }


def _kept(t: int, idx: int) -> dict:
    return {**_identity(t), "detection_id": f"{_image_id(t)}_det{idx:04d}",
            "detector_backend": "groundingdino", "detector_model_id": "SwinT_OGC",
            "class_label": "embryo", "confidence": 0.9,
            "bbox_x_min_px": 10.0, "bbox_y_min_px": 20.0,
            "bbox_x_max_px": 110.0, "bbox_y_max_px": 120.0,
            "bbox_format": "xyxy_px_abs", "is_kept": True}


def _rejected(t: int, idx: int) -> dict:
    return {**_kept(t, idx), "confidence": 0.2, "is_kept": False}


def _placeholder(t: int) -> dict:
    return {**_identity(t), "detection_id": f"{_image_id(t)}_det_none",
            "detector_backend": "groundingdino", "detector_model_id": "SwinT_OGC",
            "class_label": pd.NA, "confidence": pd.NA,
            "bbox_x_min_px": pd.NA, "bbox_y_min_px": pd.NA,
            "bbox_x_max_px": pd.NA, "bbox_y_max_px": pd.NA,
            "bbox_format": pd.NA, "is_kept": False}


def _full_table() -> pd.DataFrame:
    return pd.DataFrame([_kept(0, 0), _rejected(0, 1), _placeholder(1)])


def test_kept_view_drops_rejected_and_placeholder():
    df = _full_table()
    view = kept_frame_detections(df)
    assert len(view) == 1
    assert view.iloc[0]["detection_id"] == f"{_image_id(0)}_det0000"
    assert view["is_kept"].all()


def test_kept_view_validates_schema_first():
    bad = _full_table()
    bad.loc[0, "bbox_format"] = "cxcywh"  # invalid on a kept row
    with pytest.raises(ValueError, match="bbox_format"):
        kept_frame_detections(bad)


def test_kept_view_with_reference_runs_full_validation():
    inv = _make_inventory(2)
    df = _full_table()
    view = kept_frame_detections(df, inv)
    assert len(view) == 1


def test_kept_view_with_reference_catches_identity_mismatch():
    inv = _make_inventory(1)  # only t0000 exists
    df = _full_table()        # references t0001 placeholder
    with pytest.raises(ValueError, match="not present in reference_frame_inventory"):
        kept_frame_detections(df, inv)


def test_csv_round_trip(tmp_path):
    df = _full_table()
    path = tmp_path / "frame_detections.csv"
    df.to_csv(path, index=False)

    loaded = read_frame_detections_csv(path)
    # is_kept must come back as real bool, not the string "True"/"False".
    assert loaded["is_kept"].dtype == bool
    assert loaded["is_kept"].tolist() == [True, False, False]
    # The kept view survives the round trip.
    assert len(kept_frame_detections(loaded)) == 1
    # All contract columns present.
    assert set(REQUIRED_FRAME_DETECTIONS_COLUMNS).issubset(set(loaded.columns))
