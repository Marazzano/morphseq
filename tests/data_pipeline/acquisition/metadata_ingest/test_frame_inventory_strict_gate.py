"""Step 2 — strict L0–L4 frame_inventory gate (scope-aware grain + image checks)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_pipeline.acquisition.metadata_ingest.frame_inventory.frame_inventory_validation import (
    validate_frame_inventory,
)
from data_pipeline.acquisition.metadata_ingest.frame_inventory.frame_inventory_validation_rules import (
    validate_focus_index_map_against_inventory,
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
        "image_path": src,
        "image_micrometers_per_pixel": 0.75,
        "image_width_px": w,
        "image_height_px": h,
        "orientation": "none",
        "image_file_format": "png",
        "pixel_dtype": "uint8",
        "downsample_factor": 1,
        "downsample_method": "none",
        "jpeg_quality": pd.NA,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _png(path: Path, w: int = 16, h: int = 16) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w, h)).save(path)
    return str(path)


def _jpg(path: Path, w: int = 16, h: int = 16) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w, h)).save(path)
    return str(path)


# --- L3 grain (per_well, no image checks) --------------------------------------------------

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
        validate_frame_inventory(shard, tmp_path / "f.validated")  # default policy = "fail"


def test_ragged_channel_warn_policy_accepts(tmp_path, caplog):
    # Same ragged GFP, but ragged_channel_policy="warn" → validates (logs a warning) instead of raising.
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(A01, "BF", 1, src="a.png"),
        _row(A01, "GFP", 0, src="a.png"),
    ])
    flag = tmp_path / "f.validated"
    import logging
    with caplog.at_level(logging.WARNING):
        validate_frame_inventory(
            shard, flag, check_sources=False, ragged_channel_policy="warn"
        )
    assert flag.exists()
    assert any("rectangular" in r.message or "time_index set" in r.message for r in caplog.records)


def test_ragged_channel_warn_still_hard_fails_ragged_BF(tmp_path):
    # "warn" only softens the cross-CHANNEL check; a ragged BF *product stream* (non-contiguous time)
    # is still a hard failure regardless of policy.
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _row(A01, "BF", 0, src="a.png"),
        _row(A01, "BF", 2, src="a.png"),  # BF skips time 1 → non-contiguous stream
    ])
    with pytest.raises(ValueError, match="contiguous"):
        validate_frame_inventory(
            shard, tmp_path / "f.validated", ragged_channel_policy="warn"
        )


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


# --- L4 images (check_sources=True) ---------------------------------------------------------

def test_check_sources_false_skips_l4(tmp_path):
    # Nonexistent path, but check_sources=False → passes (L0–L3 only).
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src="nope.png")])
    validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=False)
    assert (tmp_path / "f.validated").exists()


def test_relative_path_without_image_root_fails(tmp_path):
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src="rel.png")])
    with pytest.raises(ValueError, match="requires image_root"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_image_checks_pass_on_real_image(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src=img)])
    validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)
    assert (tmp_path / "f.validated").exists()


def test_dims_mismatch_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src=img, w=32, h=32)])
    with pytest.raises(ValueError, match="dims mismatch"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_file_format_mismatch_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    row = _row(A01, "BF", 0, src=img)
    row["image_file_format"] = "jpg"
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [row])
    with pytest.raises(ValueError, match="image_file_format mismatch"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_jpg_requires_quality(tmp_path):
    img = _jpg(tmp_path / "imgs" / "a.jpg", w=16, h=16)
    row = _row(A01, "BF", 0, src=img)
    row["image_file_format"] = "jpg"
    row["jpeg_quality"] = pd.NA
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [row])
    with pytest.raises(ValueError, match="jpeg_quality"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_positive_image_micrometers_per_pixel_required(tmp_path):
    img = _jpg(tmp_path / "imgs" / "a.jpg", w=4, h=4)
    row = _row(A01, "BF", 0, src=img, w=4, h=4)
    row["image_file_format"] = "jpg"
    row["downsample_factor"] = 4
    row["downsample_method"] = "block_mean"
    row["jpeg_quality"] = 85
    row["image_micrometers_per_pixel"] = 0.0
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [row])

    with pytest.raises(ValueError, match="image_micrometers_per_pixel"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_header_dims_are_checked_directly(tmp_path):
    img = _jpg(tmp_path / "imgs" / "a.jpg", w=4, h=4)
    row = _row(A01, "BF", 0, src=img, w=8, h=4)
    row["image_file_format"] = "jpg"
    row["downsample_factor"] = 4
    row["downsample_method"] = "block_mean"
    row["jpeg_quality"] = 85
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [row])

    with pytest.raises(ValueError, match="dims mismatch"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_relative_path_resolves_under_image_root(tmp_path):
    root = tmp_path / "root"
    _png(root / "imgs" / "a.png", w=16, h=16)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [_row(A01, "BF", 0, src="imgs/a.png")])
    validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True, image_root=root)
    assert (tmp_path / "f.validated").exists()


# --- L4b focus_index_map construction-provenance --------------------------------------------

def _npz(path: Path, *, n_z: int = 3, w: int = 16, h: int = 16, bad_range: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fim = np.zeros((h, w), dtype=np.int32)
    if bad_range:
        fim[0, 0] = n_z  # out of [0, n_z): an offset that overruns z_indices
    z_indices = np.arange(n_z, dtype=np.int32)
    np.savez(path, focus_index_map=fim, z_indices=z_indices)
    return str(path)


def _proj_row_with_provenance(well_id, time_index, *, src, fim_path):
    r = _row(well_id, "BF", time_index, src=src)
    r["focus_index_map_path"] = fim_path
    return r


def test_focus_index_map_provenance_passes(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    fim = _npz(tmp_path / "prov" / "a.npz")
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _proj_row_with_provenance(A01, 0, src=img, fim_path=fim),
    ])
    validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)
    assert (tmp_path / "f.validated").exists()


def test_focus_stack_projection_missing_provenance_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _proj_row_with_provenance(A01, 0, src=img, fim_path=pd.NA),  # column present but NA
    ])
    with pytest.raises(ValueError, match="missing focus_index_map_path"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_focus_index_map_out_of_range_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    fim = _npz(tmp_path / "prov" / "a.npz", bad_range=True)
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _proj_row_with_provenance(A01, 0, src=img, fim_path=fim),
    ])
    with pytest.raises(ValueError, match="stack-axis offsets"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_focus_index_map_wrong_shape_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    fim_path = tmp_path / "prov" / "wrong_shape.npz"
    fim_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fim_path,
        focus_index_map=np.zeros((8, 16), dtype=np.int32),
        z_indices=np.arange(3, dtype=np.int32),
    )
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _proj_row_with_provenance(A01, 0, src=img, fim_path=str(fim_path)),
    ])
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_focus_index_map_float_dtype_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png", w=16, h=16)
    fim_path = tmp_path / "prov" / "float_map.npz"
    fim_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fim_path,
        focus_index_map=np.zeros((16, 16), dtype=np.float32),
        z_indices=np.arange(3, dtype=np.int32),
    )
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [
        _proj_row_with_provenance(A01, 0, src=img, fim_path=str(fim_path)),
    ])
    with pytest.raises(ValueError, match="integer stack-axis offsets"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_zstack_row_with_provenance_fails(tmp_path):
    img = _png(tmp_path / "imgs" / "z.png", w=16, h=16)
    fim = _npz(tmp_path / "prov" / "z.npz")
    r = _row(A01, "BF", 0, src=img)
    r["z_index"] = 0
    r["image_product_type"] = "z_stack"
    r["projection_method"] = pd.NA
    r["focus_index_map_path"] = fim  # z_stack must NOT carry provenance
    shard = _write(tmp_path / f"{A01}_frame_inventory.csv", [r])
    with pytest.raises(ValueError, match="not projection/focus_stack but carries"):
        validate_frame_inventory(shard, tmp_path / "f.validated", check_sources=True)


def test_inventory_aware_zindices_match(tmp_path):
    fim = _npz(tmp_path / "prov" / "a.npz", n_z=3)
    df = pd.DataFrame([_proj_row_with_provenance(A01, 0, src="a.png", fim_path=fim)])
    acq = pd.DataFrame([
        {"channel_id": "BF", "time_index": 0, "z_index": 0},
        {"channel_id": "BF", "time_index": 0, "z_index": 1},
        {"channel_id": "BF", "time_index": 0, "z_index": 2},
    ])
    # matching labels → no raise
    validate_focus_index_map_against_inventory(df, well_acquisition_inventory_df=acq)
    # mismatched labels → raise
    acq_bad = acq.iloc[:2]  # only z 0,1 in inventory but npz has 0,1,2
    with pytest.raises(ValueError, match="do not match the ordered acquisition"):
        validate_focus_index_map_against_inventory(df, well_acquisition_inventory_df=acq_bad)


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
