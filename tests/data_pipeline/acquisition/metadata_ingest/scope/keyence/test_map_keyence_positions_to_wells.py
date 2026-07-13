"""Tests for Keyence XY-position to well-marker mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.scope.keyence.map_keyence_positions_to_wells import (
    map_positions_to_wells_keyence,
)


_EXPERIMENT = "20260702_hotchem_36hpf_plate02"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_xy_position(raw_root: Path, xy_name: str, well_marker: str | None) -> None:
    position_dir = raw_root / _EXPERIMENT / xy_name
    position_dir.mkdir(parents=True, exist_ok=True)
    if well_marker is not None:
        _touch(position_dir / f"_{well_marker}")
    _touch(position_dir / f"embryo__{xy_name}_00001_Z001_CH1.tif")


def _scope_metadata_csv(path: Path) -> Path:
    df = pd.DataFrame({"well_index": ["A01", "B12", "B11", "B01"]})
    df.to_csv(path, index=False)
    return path


def test_uses_marker_file_not_xy_capture_order(tmp_path):
    raw_root = tmp_path / "raw"
    _make_xy_position(raw_root, "XY01", "A01")
    _make_xy_position(raw_root, "XY13", "B12")
    _make_xy_position(raw_root, "XY14", "B11")
    _make_xy_position(raw_root, "XY24", "B01")

    df = map_positions_to_wells_keyence(
        raw_data_dir=raw_root,
        scope_metadata_csv=_scope_metadata_csv(tmp_path / "scope.csv"),
        output_mapping_csv=tmp_path / "position_well_mapping.csv",
        output_provenance_json=tmp_path / "mapping_provenance.json",
        experiment_id=_EXPERIMENT,
    )

    observed = dict(zip(df["source_position_name"], df["well_index"]))
    assert observed == {
        "XY01": "A01",
        "XY13": "B12",
        "XY14": "B11",
        "XY24": "B01",
    }
    assert dict(zip(df["source_position_name"], df["position_index"]))["XY13"] == 13
    assert set(df["mapping_method"]) == {"keyence_xy_well_marker"}


def test_missing_marker_does_not_infer_from_xy(tmp_path):
    raw_root = tmp_path / "raw"
    _make_xy_position(raw_root, "XY13", None)

    with pytest.raises(ValueError, match="Refusing to infer plate wells from XY capture order"):
        map_positions_to_wells_keyence(
            raw_data_dir=raw_root,
            scope_metadata_csv=_scope_metadata_csv(tmp_path / "scope.csv"),
            output_mapping_csv=tmp_path / "position_well_mapping.csv",
            output_provenance_json=tmp_path / "mapping_provenance.json",
            experiment_id=_EXPERIMENT,
        )
