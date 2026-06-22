"""Tests for data_pipeline.viz render_well entry points.

Builds synthetic frame_inventory, frame_detections, and frame_masks DataFrames
(no GPU, no real images) and asserts that MP4 output is written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.segmentation.masks.mask_rle import encode_binary_mask_rle
from data_pipeline.shared.identifiers import build_image_id, build_mask_id, build_track_id, build_well_id
from data_pipeline.viz import render_combined_video, render_detection_video, render_segmentation_video


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

_W, _H = 64, 48
_EXPERIMENT = "20250101"
_WELL_INDEX = "B01"


def _well_id() -> str:
    return build_well_id(_EXPERIMENT, _WELL_INDEX)


def _frame_inventory(n_frames: int = 2, tmp_path: Path | None = None) -> pd.DataFrame:
    well_id = _well_id()
    rows = []
    for t in range(n_frames):
        image_id = build_image_id(well_id, "BF", t)
        # Write a real black TIFF so cv2.imread succeeds.
        src = Path("/dev/null")  # cv2 returns None; render_well handles with a blank frame
        if tmp_path is not None:
            src = tmp_path / f"{image_id}.png"
            import cv2
            cv2.imwrite(str(src), np.zeros((_H, _W, 3), dtype=np.uint8))
        rows.append({
            "experiment_id": _EXPERIMENT,
            "well_id": well_id,
            "well_index": _WELL_INDEX,
            "image_id": image_id,
            "time_index": t,
            "z_index": pd.NA,
            "channel_id": "BF",
            "source_image_path": str(src),
            "image_width_px": _W,
            "image_height_px": _H,
            "elapsed_time_s": float(t * 60),
            "acquisition_time_s": float(t * 60),
            "source_micrometers_per_pixel": 1.0,
        })
    return pd.DataFrame(rows)


def _frame_detections(frame_inventory: pd.DataFrame) -> pd.DataFrame:
    well_id = _well_id()
    rows = []
    for _, inv_row in frame_inventory.iterrows():
        image_id = str(inv_row["image_id"])
        rows.append({
            "experiment_id": _EXPERIMENT,
            "well_id": well_id,
            "well_index": _WELL_INDEX,
            "image_id": image_id,
            "time_index": inv_row["time_index"],
            "z_index": pd.NA,
            "channel_id": "BF",
            "source_image_path": str(inv_row["source_image_path"]),
            "image_width_px": _W,
            "image_height_px": _H,
            "elapsed_time_s": inv_row["elapsed_time_s"],
            "acquisition_time_s": inv_row["acquisition_time_s"],
            "detection_id": f"{image_id}_det0000",
            "detector_backend": "fake",
            "detector_model_id": "fake_v0",
            "class_label": "embryo",
            "confidence": 0.9,
            "bbox_x_min_px": 4.0,
            "bbox_y_min_px": 4.0,
            "bbox_x_max_px": 30.0,
            "bbox_y_max_px": 20.0,
            "bbox_format": "xyxy_px_abs",
            "is_kept": True,
        })
    return pd.DataFrame(rows)


def _frame_masks(frame_inventory: pd.DataFrame) -> pd.DataFrame:
    well_id = _well_id()
    rows = []
    mask_arr = np.zeros((_H, _W), dtype=bool)
    mask_arr[4:20, 4:30] = True
    rle = encode_binary_mask_rle(mask_arr)

    for _, inv_row in frame_inventory.iterrows():
        image_id = str(inv_row["image_id"])
        local_idx = 0
        rows.append({
            "experiment_id": _EXPERIMENT,
            "well_id": well_id,
            "image_id": image_id,
            "time_index": inv_row["time_index"],
            "z_index": pd.NA,
            "channel_id": "BF",
            "source_image_path": str(inv_row["source_image_path"]),
            "image_width_px": _W,
            "image_height_px": _H,
            "prompt_detection_id": f"{image_id}_det0000",
            "sam2_object_id": 0,
            "mask_id": build_mask_id(image_id, local_idx),
            "track_id": build_track_id(well_id, 0),
            "mask_rle": str(rle),
            "mask_rle_format": "row_major_rle",
            "area_px": float(mask_arr.sum()),
            "bbox_x_min_px": 4.0,
            "bbox_y_min_px": 4.0,
            "bbox_x_max_px": 30.0,
            "bbox_y_max_px": 20.0,
            "centroid_x_px": 17.0,
            "centroid_y_px": 12.0,
            "mask_confidence": 0.85,
            "is_valid_mask": True,
            "segmentation_backend": "fake",
            "segmentation_model_id": "fake_v0",
            "tracking_backend": "fake",
            "track_id_source": "fake",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_render_detection_video(tmp_path: Path) -> None:
    inv = _frame_inventory(n_frames=2, tmp_path=tmp_path)
    dets = _frame_detections(inv)
    out = tmp_path / "detection.mp4"
    result = render_detection_video(_well_id(), inv, dets, tmp_path, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_segmentation_video(tmp_path: Path) -> None:
    inv = _frame_inventory(n_frames=2, tmp_path=tmp_path)
    msks = _frame_masks(inv)
    out = tmp_path / "segmentation.mp4"
    result = render_segmentation_video(_well_id(), inv, msks, tmp_path, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_combined_video(tmp_path: Path) -> None:
    inv = _frame_inventory(n_frames=2, tmp_path=tmp_path)
    dets = _frame_detections(inv)
    msks = _frame_masks(inv)
    out = tmp_path / "combined.mp4"
    result = render_combined_video(_well_id(), inv, dets, msks, tmp_path, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_detection_video_blank_frames(tmp_path: Path) -> None:
    """If source images don't exist, blank frames are used and the video is still written."""
    inv = _frame_inventory(n_frames=2, tmp_path=None)  # /dev/null paths
    dets = _frame_detections(inv)
    out = tmp_path / "blank.mp4"
    result = render_detection_video(_well_id(), inv, dets, None, out)
    assert result == out
    assert out.exists()


def test_labels_off(tmp_path: Path) -> None:
    """RenderConfig with labels disabled still produces a valid video."""
    from data_pipeline.viz.config import RenderConfig
    cfg = RenderConfig(show_well_id=False, show_embryo_labels=False)
    inv = _frame_inventory(n_frames=2, tmp_path=tmp_path)
    msks = _frame_masks(inv)
    out = tmp_path / "no_labels.mp4"
    result = render_segmentation_video(_well_id(), inv, msks, tmp_path, out, config=cfg)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_no_frame_inventory(tmp_path: Path) -> None:
    """frame_inventory=None: frame order is derived from frame_masks."""
    inv = _frame_inventory(n_frames=2, tmp_path=tmp_path)
    msks = _frame_masks(inv)
    out = tmp_path / "no_inv.mp4"
    result = render_segmentation_video(_well_id(), None, msks, tmp_path, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_combined_masks_only(tmp_path: Path) -> None:
    """render_combined_video with frame_detections=None renders masks only."""
    inv = _frame_inventory(n_frames=2, tmp_path=tmp_path)
    msks = _frame_masks(inv)
    out = tmp_path / "masks_only.mp4"
    result = render_combined_video(_well_id(), inv, None, msks, tmp_path, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
