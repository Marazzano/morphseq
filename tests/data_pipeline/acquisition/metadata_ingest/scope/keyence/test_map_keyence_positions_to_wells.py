"""Tests for Keyence XY-position to well-marker mapping."""

from __future__ import annotations

import json
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


class TestRepeatStagePositions:
    """A well set twice during scope setup yields two XY directories with the same marker.

    Observed on four experiments (2026-08-30): 97 XY directories for 96 wells, both XY06 and XY97
    carrying ``_A06``. Downstream this unioned into one inventory and materialization refused the
    well ("Duplicate Keyence time/tile/z inventory cells"), failing the whole experiment.
    """

    def _mapping_with_repeat(self, tmp_path):
        raw_root = tmp_path / "raw"
        _make_xy_position(raw_root, "XY01", "A01")
        _make_xy_position(raw_root, "XY13", "B12")
        _make_xy_position(raw_root, "XY97", "A01")  # the re-set position, appended last
        return map_positions_to_wells_keyence(
            raw_data_dir=raw_root,
            scope_metadata_csv=_scope_metadata_csv(tmp_path / "scope.csv"),
            output_mapping_csv=tmp_path / "position_well_mapping.csv",
            output_provenance_json=tmp_path / "mapping_provenance.json",
            experiment_id=_EXPERIMENT,
        )

    def test_one_row_per_well_keeping_the_first_position(self, tmp_path):
        df = self._mapping_with_repeat(tmp_path)

        assert df["well_id"].is_unique, "a well must map to exactly one position"
        kept = dict(zip(df["well_index"], df["source_position_name"]))
        assert kept["A01"] == "XY01", "the LOWEST position_index is the original acquisition"
        assert "XY97" not in set(df["source_position_name"])

    def test_the_dropped_acquisition_stays_findable(self, tmp_path):
        # Silently discarding a complete, real acquisition is the failure mode to avoid: the
        # provenance must name what was set aside and what was kept in its place.
        self._mapping_with_repeat(tmp_path)
        provenance = json.loads((tmp_path / "mapping_provenance.json").read_text())

        dropped = provenance["dropped_repeat_positions"]
        assert [row["source_position_name"] for row in dropped] == ["XY97"]
        assert provenance["mapping_summary"]["wells_with_multiple_positions"] == 1
        assert any("XY97" in w and "XY01" in w for w in provenance["warnings"])

    def test_an_ordinary_plate_is_untouched(self, tmp_path):
        # The de-dupe must be inert where there is nothing to de-duplicate -- no dropped rows, and
        # no spurious "multiple positions" count.
        raw_root = tmp_path / "raw"
        _make_xy_position(raw_root, "XY01", "A01")
        _make_xy_position(raw_root, "XY13", "B12")
        df = map_positions_to_wells_keyence(
            raw_data_dir=raw_root,
            scope_metadata_csv=_scope_metadata_csv(tmp_path / "scope.csv"),
            output_mapping_csv=tmp_path / "position_well_mapping.csv",
            output_provenance_json=tmp_path / "mapping_provenance.json",
            experiment_id=_EXPERIMENT,
        )
        provenance = json.loads((tmp_path / "mapping_provenance.json").read_text())

        assert len(df) == 2
        assert provenance["dropped_repeat_positions"] is None
        assert provenance["mapping_summary"]["wells_with_multiple_positions"] == 0


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
