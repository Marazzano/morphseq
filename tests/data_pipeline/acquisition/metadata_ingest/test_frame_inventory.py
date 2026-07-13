from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.frame_inventory import (
    merge_frame_inventory_shards,
    validate_frame_inventory,
)

EXP = "20250912"
A01 = "20250912_A01"
B01 = "20250912_B01"


def _inventory_row(well_id: str, time_index: int) -> dict:
    """One row of the LIVE flat frame_inventory contract (atoms only; ids derived by the validator)."""
    well_index = well_id.split("_")[-1]
    return {
        "experiment_id": EXP,
        "well_index": well_index,
        "channel_id": "BF",
        "time_index": time_index,
        "elapsed_time_s": float(time_index * 120),
        "acquisition_time_s": float(time_index * 120),
        "z_index": pd.NA,
        "image_product_type": "projection",
        "projection_method": "focus_stack",
        "image_path": (
            f"built_image_data/{EXP}/materialized_images/{well_id}/projection/BF/"
            f"{well_id}_BF_t{time_index:04d}.png"
        ),
        "image_micrometers_per_pixel": 0.75,
        "image_width_px": 1024,
        "image_height_px": 768,
        "orientation": "none",
        "image_file_format": "png",
        "pixel_dtype": "uint8",
        "downsample_factor": 1,
        "downsample_method": "none",
        "jpeg_quality": pd.NA,
    }


def _z_stack_inventory_row(well_id: str, time_index: int, z_index: int) -> dict:
    row = _inventory_row(well_id, time_index)
    row["z_index"] = z_index
    row["image_product_type"] = "z_stack"
    row["projection_method"] = pd.NA
    row["image_path"] = (
        f"built_image_data/{EXP}/materialized_images/{well_id}/z_stack/BF/"
        f"{well_id}_BF_z{z_index:04d}_t{time_index:04d}.png"
    )
    return row


def _write_inventory(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_validate_frame_inventory_writes_sentinel(tmp_path):
    shard = tmp_path / f"{A01}_frame_inventory.csv"
    flag = tmp_path / f"{A01}_frame_inventory.csv.validated"
    _write_inventory(shard, [_inventory_row(A01, 0)])

    validate_frame_inventory(shard, flag)

    assert flag.read_text(encoding="utf-8") == "validated\n"


def test_validate_frame_inventory_fails_on_duplicate_key(tmp_path):
    shard = tmp_path / f"{A01}_frame_inventory.csv"
    _write_inventory(shard, [_inventory_row(A01, 0), _inventory_row(A01, 0)])

    with pytest.raises(ValueError, match="(?i)duplicate"):
        validate_frame_inventory(shard, tmp_path / "bad.validated")


def test_validate_frame_inventory_allows_distinct_z_stack_planes(tmp_path):
    shard = tmp_path / f"{A01}_frame_inventory.csv"
    _write_inventory(shard, [_z_stack_inventory_row(A01, 0, 0), _z_stack_inventory_row(A01, 0, 1)])

    validate_frame_inventory(shard, tmp_path / "z.validated")

    assert (tmp_path / "z.validated").exists()


def test_validate_frame_inventory_rejects_duplicate_z_stack_plane(tmp_path):
    shard = tmp_path / f"{A01}_frame_inventory.csv"
    _write_inventory(shard, [_z_stack_inventory_row(A01, 0, 0), _z_stack_inventory_row(A01, 0, 0)])

    with pytest.raises(ValueError, match="(?i)duplicate"):
        validate_frame_inventory(shard, tmp_path / "bad.validated")


def test_validate_frame_inventory_fails_on_leaked_local_well_id(tmp_path):
    # well_index that is NOT a clean local label would compose a bad well_id; the identity-anchored
    # key routes through validate_well_id, so a leaked/invalid id fails loud HERE.
    shard = tmp_path / "bad_inventory.csv"
    row = _inventory_row(A01, 0)
    row["experiment_id"] = ""  # build_well_id → "_A01"-ish; validate_well_id rejects it
    _write_inventory(shard, [row])

    with pytest.raises(ValueError):
        validate_frame_inventory(shard, tmp_path / "bad.validated")


def test_merge_frame_inventory_shards_concatenates_and_sorts(tmp_path):
    a = tmp_path / f"{A01}_frame_inventory.csv"
    b = tmp_path / f"{B01}_frame_inventory.csv"
    out = tmp_path / f"{EXP}_frame_inventory.csv"
    _write_inventory(a, [_inventory_row(A01, 1), _inventory_row(A01, 0)])
    _write_inventory(b, [_inventory_row(B01, 0)])

    merge_frame_inventory_shards([b, a], out)

    written = pd.read_csv(out)
    # Sorted on the frame_inventory atoms (experiment_id, well_index, channel_id, time_index).
    assert list(written["well_index"]) == ["A01", "A01", "B01"]
    assert list(written["time_index"]) == [0, 1, 0]


def test_merge_frame_inventory_shards_rejects_column_drift(tmp_path):
    a = tmp_path / f"{A01}_frame_inventory.csv"
    b = tmp_path / f"{B01}_frame_inventory.csv"
    _write_inventory(a, [_inventory_row(A01, 0)])
    # Both shards must pass the required-schema check, but differ in column SET, so we hit the merge's
    # column-drift guard (not the required-schema check). Add an EXTRA non-contract column to one
    # shard — all required columns are still present, but the column lists no longer match.
    extra = pd.DataFrame([_inventory_row(B01, 0)])
    extra["unexpected_extra_column"] = "x"
    extra.to_csv(b, index=False)

    with pytest.raises(ValueError, match="expected"):
        merge_frame_inventory_shards([a, b], tmp_path / "merged.csv")
