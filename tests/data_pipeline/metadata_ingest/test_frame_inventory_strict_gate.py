"""Step 2 — strict L0–L4 frame_inventory gate (scope-aware grain + source checks)."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from data_pipeline.metadata_ingest.frame_inventory.frame_inventory_validation import (
    validate_frame_inventory,
)

EXP = "20250912"
A01 = "20250912_A01"
B01 = "20250912_B01"


def _row(well_id: str, channel: str, time_index: int, *, src: str, w: int = 16, h: int = 16) -> dict:
    well_index = well_id.split("_")[-1]
    return {
        "experiment_id": EXP,
        "well_index": well_index,
        "channel_id": channel,
        "time_index": time_index,
        "elapsed_time_s": float(time_index * 120),
        "acquisition_time_s": float(time_index * 120),
        "z_index": pd.NA,
        "image_product_type": "projection",
        "projection_method": "focus_stack",
        "source_image_path": src,
        "source_micrometers_per_pixel": 0.75,
        "image_width_px": w,
        "image_height_px": h,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _png(path: Path, w: int = 16, h: int = 16) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w, h)).save(path)
    return str(path)


# --- L3 grain (per_well, no source checks) -------------------------------------------------

def test_per_well_bf_gap_fails(tmp_path):
    # BF at 0 and 2 — not contiguous.
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(A01, "BF", 2, src="a.png"),
    ])
    with pytest.raises(ValueError, match="contiguous"):
        validate_frame_inventory(shard, tmp_path / "f.validated")


def test_per_well_ragged_channel_fails(tmp_path):
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(A01, "BF", 1, src="a.png"),
        _row(A01, "GFP", 0, src="a.png"),  # GFP missing time 1 → ragged
    ])
    with pytest.raises(ValueError, match="rectangular|time_index set"):
        validate_frame_inventory(shard, tmp_path / "f.validated")


def test_multi_timepoint_missing_elapsed_fails(tmp_path):
    rows = [_row(A01, "BF", 0, src="a.png"), _row(A01, "BF", 1, src="a.png")]
    rows[1]["elapsed_time_s"] = pd.NA
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", rows)
    with pytest.raises(ValueError, match="elapsed_time_s"):
        validate_frame_inventory(shard, tmp_path / "f.validated")


def test_per_well_rejects_many_wells(tmp_path):
    shard = _write(tmp_path / "mixed.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(B01, "BF", 0, src="a.png"),
    ])
    with pytest.raises(ValueError, match="exactly one well_index"):
        validate_frame_inventory(shard, tmp_path / "f.validated", validation_scope="per_well")


def test_merged_allows_many_wells_but_group_checks(tmp_path):
    # Two wells, each internally valid → merged scope passes.
    shard = _write(tmp_path / "merged.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(A01, "BF", 1, src="a.png"),
        _row(B01, "BF", 0, src="a.png"),
        _row(B01, "BF", 1, src="a.png"),
    ])
    validate_frame_inventory(shard, tmp_path / "f.validated", validation_scope="merged")
    assert (tmp_path / "f.validated").exists()


def test_merged_group_check_catches_one_bad_well(tmp_path):
    # B01 has a BF gap; merged scope must still catch it within the group.
    shard = _write(tmp_path / "merged.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(B01, "BF", 0, src="a.png"),
        _row(B01, "BF", 2, src="a.png"),
    ])
    with pytest.raises(ValueError, match="contiguous"):
        validate_frame_inventory(shard, tmp_path / "f.validated", validation_scope="merged")


# --- L4 sources (check_sources=True) --------------------------------------------------------

def test_check_sources_false_skips_l4(tmp_path):
    # Nonexistent path, but check_sources=False → passes (L0–L3 only).
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src="nope.png")])
    validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=False)
    assert (tmp_path / "f.validated").exists()


def test_relative_path_without_image_root_fails(tmp_path):
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src="rel.png")])
    with pytest.raises(ValueError, match="requires image_root"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_source_checks_pass_on_real_image(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src=img)])
    validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)
    assert (tmp_path / "f.validated").exists()


def test_dims_mismatch_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src=img, w=32, h=32)])
    with pytest.raises(ValueError, match="dims mismatch"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_relative_path_resolves_under_image_root(tmp_path):
    root = tmp_path / "root"
    _png(root / "imgs" / "a.png", w=16, h=16)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src="imgs/a.png")])
    validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True, image_root=root)
    assert (tmp_path / "f.validated").exists()


# --- errors report naming -------------------------------------------------------------------

def test_errors_report_named_by_well_when_derivable(tmp_path):
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(A01, "BF", 2, src="a.png"),  # gap
    ])
    with pytest.raises(ValueError):
        validate_frame_inventory(shard, tmp_path / "f.validated")
    assert (tmp_path / f"{A01}_frame_inventory.errors.md").exists()


def test_errors_report_falls_back_when_well_id_underivable(tmp_path):
    # Empty experiment_id → well_id can't be derived; fallback name must be used.
    rows = [_row(A01, "BF", 0, src="a.png")]
    rows[0]["experiment_id"] = ""
    shard = _write(tmp_path / "bad.csv", rows)
    with pytest.raises(ValueError):
        validate_frame_inventory(shard, tmp_path / "f.validated")
    assert (tmp_path / "frame_inventory.errors.md").exists()
