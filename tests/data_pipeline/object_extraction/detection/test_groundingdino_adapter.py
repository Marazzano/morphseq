"""Tests for the GroundingDINO backend adapter — native output → shared rows, flag-not-drop seam.

Inference is stubbed (monkeypatched ``detect_embryos`` / ``filter_detections``) so no GPU / weights
are needed. We assert the adapter: converts normalized boxes → absolute px, carries class_label from
phrase, marks is_kept by kept-membership (flag-not-drop), and emits a _det_none placeholder on a
frame with no candidates.
"""

import pandas as pd

import data_pipeline.object_extraction.detection.backends.groundingdino.run_groundingdino_detection as gd

WIDTH, HEIGHT = 1000, 800
IMAGE_ID = "20250912_B01_BF_t0000"


def _identity_row() -> dict:
    return {"image_id": IMAGE_ID, "image_width_px": WIDTH, "image_height_px": HEIGHT}


def test_adapter_marks_kept_and_rejected(monkeypatch):
    raw = [
        {"box_xyxy": [0.1, 0.1, 0.3, 0.3], "confidence": 0.95, "phrase": "embryo"},
        {"box_xyxy": [0.5, 0.5, 0.7, 0.7], "confidence": 0.20, "phrase": "embryo"},
    ]
    kept = [raw[0]]  # only the high-confidence one survives

    monkeypatch.setattr(gd, "detect_embryos", lambda **kw: raw)
    monkeypatch.setattr(gd, "filter_detections", lambda dets, **kw: kept)

    rows = gd.detect_frame(
        model=object(),
        image_path="ignored.png",
        identity_row=_identity_row(),
        detector_model_id="SwinT_OGC",
    )
    assert len(rows) == 2
    by_kept = {r["is_kept"] for r in rows}
    assert by_kept == {True, False}

    kept_row = next(r for r in rows if r["is_kept"])
    # 0.1*1000 -> 100, 0.3*1000 -> 300 ; 0.1*800 -> 80, 0.3*800 -> 240
    assert kept_row["bbox_x_min_px"] == 100.0
    assert kept_row["bbox_x_max_px"] == 300.0
    assert kept_row["bbox_y_min_px"] == 80.0
    assert kept_row["bbox_y_max_px"] == 240.0
    assert kept_row["bbox_format"] == "xyxy_px_abs"
    assert kept_row["class_label"] == "embryo"
    assert kept_row["detector_backend"] == "groundingdino"
    assert kept_row["detection_id"] == f"{IMAGE_ID}_det0000"


def test_adapter_emits_placeholder_on_no_candidates(monkeypatch):
    monkeypatch.setattr(gd, "detect_embryos", lambda **kw: [])
    monkeypatch.setattr(gd, "filter_detections", lambda dets, **kw: [])

    rows = gd.detect_frame(
        model=object(),
        image_path="ignored.png",
        identity_row=_identity_row(),
        detector_model_id="SwinT_OGC",
    )
    assert len(rows) == 1
    ph = rows[0]
    assert ph["detection_id"] == f"{IMAGE_ID}_det_none"
    assert ph["is_kept"] is False
    assert pd.isna(ph["confidence"])
    assert pd.isna(ph["bbox_x_min_px"])
    assert pd.isna(ph["bbox_format"])
