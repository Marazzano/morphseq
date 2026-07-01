"""Tests for detection/run_frame_detection.py — backend-agnostic router, flag-not-drop, CSV wrapper.

Inference is stubbed per-frame so the router runs without GPU / weights. We assert: multi-candidate
and no-candidate frames both appear in the output, rejected candidates are retained with is_kept=False,
the output passes the composed validator, and the CSV path wrapper round-trips.
"""

import pandas as pd
import pytest

import data_pipeline.object_extraction.detection.backends.groundingdino.run_groundingdino_detection as gd
from data_pipeline.object_extraction.detection.kept_frame_detections import read_frame_detections_csv
from data_pipeline.object_extraction.detection.run_frame_detection import (
    _projection_bf_rows,
    run_frame_detection,
    run_frame_detection_df,
)
from data_pipeline.object_extraction.detection.validate_frame_detections import validate_frame_detections

EXP = "20250912"
WELL_ID = f"{EXP}_B01"
WIDTH, HEIGHT = 1000, 800


def _image_id(t: int) -> str:
    return f"{WELL_ID}_BF_t{t:04d}"


def _make_inventory(n: int, channel: str = "BF") -> pd.DataFrame:
    return pd.DataFrame([{
        "experiment_id": EXP,
        "well_index": "B01",
        "channel_id": channel,
        "time_index": t,
        "elapsed_time_s": float(t * 120),
        "source_image_path": f"images/{WELL_ID}_{channel}_t{t:04d}.png",
        "source_micrometers_per_pixel": 0.75,
        "image_width_px": WIDTH,
        "image_height_px": HEIGHT,
    } for t in range(n)])


def _projection_row(t: int) -> dict:
    row = _make_inventory(1).iloc[0].to_dict()
    row["time_index"] = t
    row["source_image_path"] = f"images/{WELL_ID}_BF_t{t:04d}.png"
    row["z_index"] = pd.NA
    row["image_product_type"] = "projection"
    return row


def _z_stack_row(t: int, z: int) -> dict:
    row = _make_inventory(1).iloc[0].to_dict()
    row["time_index"] = t
    row["source_image_path"] = f"images/{WELL_ID}_BF_z{z:04d}_t{t:04d}.png"
    row["z_index"] = z
    row["image_product_type"] = "z_stack"
    return row


def test_projection_bf_rows_excludes_z_stack_planes():
    # A mixed inventory (one projection + two z planes for the same timepoint) must yield only the
    # projection row — a z plane must never reach a detector.
    inv = pd.DataFrame([_projection_row(0), _z_stack_row(0, 0), _z_stack_row(0, 1)])
    selected = _projection_bf_rows(inv)
    assert len(selected) == 1
    assert set(selected["image_product_type"]) == {"projection"}


def test_projection_bf_rows_keeps_all_bf_when_column_absent():
    # Back-compat: an inventory written before z_stack existed has no image_product_type column;
    # every BF row in that world IS a projection, so all BF rows are kept.
    inv = _make_inventory(2)  # no image_product_type column
    assert "image_product_type" not in inv.columns
    selected = _projection_bf_rows(inv)
    assert len(selected) == 2


def test_router_skips_z_stack_planes(monkeypatch):
    _stub_inference(monkeypatch)
    # Inventory carries the projection frame (t0000) AND its z planes; detection must run only on
    # the projection frame, so the flag-not-drop result is identical to the projection-only inventory.
    inv = pd.DataFrame([_projection_row(0), _z_stack_row(0, 0), _z_stack_row(0, 1)])
    df = run_frame_detection_df(
        inv, backend="groundingdino", model=object(), detector_model_id="SwinT_OGC",
    )
    assert set(df["image_id"]) == {_image_id(0)}
    assert df["is_kept"].sum() == 1


def _stub_inference(monkeypatch):
    """Frame t0000 has two candidates (one kept, one rejected); t0001 has none."""
    def fake_detect_embryos(*, image_path, **kw):
        if str(image_path).endswith("t0000.png"):
            return [
                {"box_xyxy": [0.1, 0.1, 0.3, 0.3], "confidence": 0.95, "phrase": "embryo"},
                {"box_xyxy": [0.5, 0.5, 0.7, 0.7], "confidence": 0.20, "phrase": "embryo"},
            ]
        return []

    def fake_filter(dets, **kw):
        return [d for d in dets if d["confidence"] >= 0.45]

    monkeypatch.setattr(gd, "detect_embryos", fake_detect_embryos)
    monkeypatch.setattr(gd, "filter_detections", fake_filter)


def test_router_df_flag_not_drop(monkeypatch):
    _stub_inference(monkeypatch)
    inv = _make_inventory(2)

    df = run_frame_detection_df(
        inv, backend="groundingdino", model=object(), detector_model_id="SwinT_OGC",
    )

    # t0000: 2 candidate rows (1 kept, 1 rejected). t0001: 1 placeholder row. Total 3.
    assert len(df) == 3
    assert df["is_kept"].sum() == 1

    t0_rows = df[df["image_id"] == _image_id(0)]
    assert len(t0_rows) == 2
    assert set(t0_rows["is_kept"]) == {True, False}

    t1_rows = df[df["image_id"] == _image_id(1)]
    assert len(t1_rows) == 1
    assert t1_rows.iloc[0]["detection_id"] == f"{_image_id(1)}_det_none"
    assert bool(t1_rows.iloc[0]["is_kept"]) is False

    # The router already validated; re-validate explicitly for belt-and-suspenders.
    validate_frame_detections(df, inv)


def test_router_only_detects_bf_channel(monkeypatch):
    _stub_inference(monkeypatch)
    # Mix BF and GFP; GFP frames must not be detected on.
    inv = pd.concat([_make_inventory(1, "BF"), _make_inventory(1, "GFP")], ignore_index=True)
    df = run_frame_detection_df(
        inv, backend="groundingdino", model=object(), detector_model_id="SwinT_OGC",
    )
    assert set(df["channel_id"]) == {"BF"}


def test_router_unknown_backend_raises():
    inv = _make_inventory(1)
    with pytest.raises(ValueError, match="Unsupported detector backend"):
        run_frame_detection_df(inv, backend="nope", model=object(), detector_model_id="x")


def test_router_detectron2_stub_raises(monkeypatch):
    inv = _make_inventory(1)
    with pytest.raises(NotImplementedError):
        run_frame_detection_df(inv, backend="detectron2", model=object(), detector_model_id="x")


def test_csv_path_wrapper_round_trips(monkeypatch, tmp_path):
    _stub_inference(monkeypatch)
    inv = _make_inventory(2)
    inv_csv = tmp_path / "frame_inventory.csv"
    inv.to_csv(inv_csv, index=False)
    out_csv = tmp_path / "frame_detections.csv"

    returned = run_frame_detection(
        inv_csv, out_csv, backend="groundingdino", model=object(), detector_model_id="SwinT_OGC",
    )
    assert out_csv.exists()

    loaded = read_frame_detections_csv(out_csv)
    assert len(loaded) == len(returned) == 3
    assert loaded["is_kept"].dtype == bool
    # Re-validate the written artifact against the inventory.
    validate_frame_detections(loaded, inv)
