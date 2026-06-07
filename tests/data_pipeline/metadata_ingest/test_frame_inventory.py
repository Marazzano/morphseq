from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.metadata_ingest.frame_inventory import (
    build_frame_inventory_for_well,
    merge_frame_inventory_shards,
    validate_frame_inventory,
)

EXP = "20250912"
A01 = "20250912_A01"
B01 = "20250912_B01"


def _row(well_id: str, time_int: int) -> dict:
    well_index = well_id.split("_")[-1]
    return {
        "experiment_id": EXP,
        "microscope_id": "YX1",
        "well_id": well_id,
        "well_index": well_index,
        "channel_id": "BF",
        "channel_name_raw": "Brightfield",
        "time_int": time_int,
        "frame_index": time_int,
        "image_id": f"{well_id}_BF_t{time_int:04d}",
        "stitched_image_path": f"built_image_data/{EXP}/stitched_ff_images/{well_index}/BF/{well_id}_BF_t{time_int:04d}.jpg",
        "micrometers_per_pixel": 0.75,
        "frame_interval_s": 120.0,
        "absolute_start_time": "2025-09-12T00:00:00",
        "experiment_time_s": float(time_int * 120),
        "image_width_px": 1024,
        "image_height_px": 768,
        "objective_magnification": "10",
    }


def _write_frame_contract(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_frame_inventory_for_well_filters_and_preserves_columns(tmp_path):
    source = tmp_path / "frame_contract.csv"
    rows = [_row(A01, 0), _row(A01, 1), _row(B01, 0)]
    _write_frame_contract(source, rows)

    output = tmp_path / "per_well" / A01 / f"{A01}_frame_inventory.csv"
    shard = build_frame_inventory_for_well(
        frame_contract_csv=source,
        experiment_id=EXP,
        well_id=A01,
        output_csv=output,
    )

    written = pd.read_csv(output)
    assert list(written.columns) == list(pd.DataFrame(rows).columns)
    assert list(shard["well_id"].unique()) == [A01]
    assert list(written["time_int"]) == [0, 1]


def test_build_frame_inventory_for_well_fails_when_well_absent(tmp_path):
    source = tmp_path / "frame_contract.csv"
    _write_frame_contract(source, [_row(A01, 0)])

    with pytest.raises(ValueError, match="No frame_contract rows"):
        build_frame_inventory_for_well(
            frame_contract_csv=source,
            experiment_id=EXP,
            well_id=B01,
            output_csv=tmp_path / "missing.csv",
        )


def test_validate_frame_inventory_writes_sentinel(tmp_path):
    shard = tmp_path / f"{A01}_frame_inventory.csv"
    flag = tmp_path / f"{A01}_frame_inventory.csv.validated"
    _write_frame_contract(shard, [_row(A01, 0)])

    validate_frame_inventory(shard, flag)

    assert flag.read_text(encoding="utf-8") == "validated\n"


def test_validate_frame_inventory_fails_on_duplicate_key(tmp_path):
    shard = tmp_path / f"{A01}_frame_inventory.csv"
    _write_frame_contract(shard, [_row(A01, 0), _row(A01, 0)])

    with pytest.raises(ValueError, match="Duplicate frame_inventory keys"):
        validate_frame_inventory(shard, tmp_path / "bad.validated")


def test_merge_frame_inventory_shards_concatenates_and_sorts(tmp_path):
    a = tmp_path / f"{A01}_frame_inventory.csv"
    b = tmp_path / f"{B01}_frame_inventory.csv"
    out = tmp_path / f"{EXP}_frame_inventory.csv"
    _write_frame_contract(a, [_row(A01, 1), _row(A01, 0)])
    _write_frame_contract(b, [_row(B01, 0)])

    merged = merge_frame_inventory_shards([b, a], out)

    written = pd.read_csv(out)
    assert list(merged["well_id"]) == [A01, A01, B01]
    assert list(written["time_int"]) == [0, 1, 0]


def test_merge_frame_inventory_shards_rejects_column_drift(tmp_path):
    a = tmp_path / f"{A01}_frame_inventory.csv"
    b = tmp_path / f"{B01}_frame_inventory.csv"
    _write_frame_contract(a, [_row(A01, 0)])
    bad = pd.DataFrame([_row(B01, 0)]).drop(columns=["frame_index"])
    bad.to_csv(b, index=False)

    with pytest.raises(ValueError, match="expected"):
        merge_frame_inventory_shards([a, b], tmp_path / "merged.csv")
