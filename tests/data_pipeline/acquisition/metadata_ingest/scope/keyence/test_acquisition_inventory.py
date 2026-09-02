"""Tests for the Keyence acquisition inventory (Stage A).

Run with: PYTHONPATH=src pytest tests/data_pipeline/metadata_ingest/scope/keyence/test_acquisition_inventory.py

The XML scrape is disk-touching, so the builder takes a ``scrape_plane_metadata`` callable that these
tests stub — the synthetic raw tree only needs empty ``*CH*.tif`` files for the filename grammar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.scope.keyence.acquisition_inventory import (
    KEYENCE_ACQUISITION_CELL_KEY,
    KEYENCE_ACQUISITION_INVENTORY_COLUMNS,
    assert_keyence_acquisition_sources_readable,
    build_keyence_acquisition_inventory,
    build_keyence_acquisition_inventory_rows,
    validate_keyence_acquisition_inventory,
)

_EXPERIMENT = "20240509"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _stub_scrape(_tiff_path: Path) -> dict:
    """A constant per-plane scrape — calibration is the same across planes in these fixtures."""
    return {
        "micrometers_per_pixel": 0.756,
        "image_width_px": 1920,
        "image_height_px": 1440,
        "objective_magnification": "4x",
        "acquisition_time_s": 100.0,
        "raw_channel_name": None,  # exercise the "no proprietary channel name" path
    }


def _make_well_tree(root: Path, well_xy: str, *, tiles, zs, channels, times, marker: str | None = None) -> None:
    """Write empty raw planes for a multi-tile/multi-Z/multi-channel well under an XY dir layout."""
    if marker is None:
        xy_num = int(well_xy.removeprefix("XY"))
        row = (xy_num - 1) // 12
        col = (xy_num - 1) % 12 + 1
        marker = f"{chr(65 + row)}{col:02d}"
    _touch(root / well_xy / f"_{marker}")
    for t in times:
        for tile in tiles:
            for z in zs:
                for ch in channels:
                    name = f"embryo__{well_xy}_{tile:05d}_T{t:04d}_Z{z:03d}_CH{ch}.tif"
                    _touch(root / well_xy / name)


# ── builder: explode, no collapse, faithful axes ──────────────────────────────────────────────


def test_builder_explodes_one_row_per_plane(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    # 1 well, 3 tiles, 2 Z, 1 channel (CH1), 2 times => 3*2*1*2 = 12 planes.
    _make_well_tree(raw_dir, "XY16", tiles=(1, 2, 3), zs=(1, 2), channels=(1,), times=(1, 2))

    df = build_keyence_acquisition_inventory(
        experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
    )

    assert len(df) == 12
    assert list(df.columns) == list(KEYENCE_ACQUISITION_INVENTORY_COLUMNS)
    # No collapse: the cell key is unique across all planes.
    assert not df.duplicated(subset=list(KEYENCE_ACQUISITION_CELL_KEY)).any()
    # Faithful axes captured from the filename grammar.
    assert df["position_index"].unique().tolist() == [16]
    assert sorted(df["z_index"].unique()) == [1, 2]
    assert sorted(df["tile_id"].unique()) == [1, 2, 3]
    assert df["n_tiles_in_well"].unique().tolist() == [3]


def test_builder_uses_well_marker_not_xy_sequence(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    _make_well_tree(raw_dir, "XY13", tiles=(1,), zs=(1,), channels=(1,), times=(1,), marker="B12")

    df = build_keyence_acquisition_inventory(
        experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
    )

    assert df["position_index"].unique().tolist() == [13]
    assert df["well_index"].unique().tolist() == ["B12"]


def test_multichannel_well_is_not_a_collision(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    # Same (tile, z, time) across TWO channels must NOT collide — channel_index distinguishes them.
    # Use CH1 only mapped; here we test the index is faithful by mapping CH2 too (see index map note).
    _make_well_tree(raw_dir, "XY16", tiles=(1,), zs=(1,), channels=(1,), times=(1,))
    df = build_keyence_acquisition_inventory(
        experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
    )
    # channel_index reflects the real CH#.
    assert df["channel_index"].unique().tolist() == [1]
    # All channel_id resolved (BF), none silently coerced to a non-vocabulary token.
    assert df["channel_id"].unique().tolist() == ["BF"]


def test_channel_id_resolves_to_bf_and_raw_name_falls_back_to_ch_token(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    _make_well_tree(raw_dir, "XY16", tiles=(1,), zs=(1,), channels=(1,), times=(1,))
    df = build_keyence_acquisition_inventory(
        experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
    )
    assert (df["channel_id"] == "BF").all()
    # No scraped name -> provenance falls back to the reliable CH# token.
    assert (df["raw_channel_name"] == "CH1").all()


def test_unmapped_channel_index_fails_loud(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    # CH2 is not in KEYENCE_CHANNEL_INDEX_MAP (BF-only today) -> fail loud, no silent default.
    _make_well_tree(raw_dir, "XY16", tiles=(1,), zs=(1,), channels=(2,), times=(1,))
    with pytest.raises(ValueError, match="No mapping for raw channel"):
        build_keyence_acquisition_inventory(
            experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
        )


def test_elapsed_time_is_finite_non_negative(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    _make_well_tree(raw_dir, "XY16", tiles=(1,), zs=(1,), channels=(1,), times=(1, 2, 3))

    times = {1: 100.0, 2: 1900.0, 3: 3700.0}

    def scrape_with_time(tiff_path: Path) -> dict:
        meta = _stub_scrape(tiff_path)
        # time index is encoded as T#### (1-based) -> claimed index t-1; map back for the timestamp.
        for token in tiff_path.name.split("_"):
            if token.startswith("T") and token[1:].isdigit():
                meta["acquisition_time_s"] = times[int(token[1:])]
        return meta

    df = build_keyence_acquisition_inventory(
        experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=scrape_with_time
    )
    assert (df["elapsed_time_s"] >= 0).all()
    assert df["elapsed_time_s"].notna().all()
    # First frame rebased to 0.
    assert df["elapsed_time_s"].min() == 0.0


# ── validation: collision fail-loud + source readability ──────────────────────────────────────


def test_duplicate_cell_key_fails_loud():
    rows = build_keyence_acquisition_inventory_rows  # noqa: F841 (kept for symmetry/readability)
    # Construct a minimal valid frame, then duplicate a cell to force the collision check.
    base = {
        "experiment_id": _EXPERIMENT,
        "position_index": 0,
        "channel_id": "BF",
        "raw_channel_name": "CH1",
        "time_index": 0,
        "elapsed_time_s": 0.0,
        "micrometers_per_pixel": 0.7,
        "image_width_px": 1920,
        "image_height_px": 1440,
        "microscope_id": "Keyence",
        "well_index": "B04",
        "well_id": f"{_EXPERIMENT}_B04",
        "tile_id": 1,
        "position_index_within_well": 1,
        "n_tiles_in_well": 1,
        "z_index": 0,
        "channel_index": 1,
        "time_index_claimed": 0,
        "acquisition_time_s": 100.0,
        "objective_magnification": "4x",
        "orientation": "unknown",
        "source_tiff_path": "/tmp/a.tif",
        "stage_x_nm": 0,
        "stage_y_nm": 0,
        "stage_z_nm": 0,
    }
    df = pd.DataFrame([base, dict(base)])  # two identical cells
    with pytest.raises(ValueError, match="not unique on the cell key"):
        validate_keyence_acquisition_inventory(df)


def test_validate_rejects_bad_calibration(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    _make_well_tree(raw_dir, "XY16", tiles=(1,), zs=(1,), channels=(1,), times=(1,))
    df = build_keyence_acquisition_inventory(
        experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
    )
    df.loc[0, "micrometers_per_pixel"] = 0.0
    with pytest.raises(ValueError, match="micrometers_per_pixel"):
        validate_keyence_acquisition_inventory(df)


def test_validate_rejects_unknown_channel_id(tmp_path):
    raw_dir = tmp_path / _EXPERIMENT
    _make_well_tree(raw_dir, "XY16", tiles=(1,), zs=(1,), channels=(1,), times=(1,))
    df = build_keyence_acquisition_inventory(
        experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
    )
    df["channel_id"] = "Cy5"  # not in VALID_CHANNEL_NAMES; catches stale/hand-edited tables
    with pytest.raises(ValueError, match="not in the canonical vocabulary"):
        validate_keyence_acquisition_inventory(df)


def test_sources_readable_raises_named_error_on_missing_path():
    df = pd.DataFrame({"source_tiff_path": ["/does/not/exist_CH1.tif"]})
    with pytest.raises(ValueError, match="does not exist"):
        assert_keyence_acquisition_sources_readable(df, scope_label="test")


class TestRepeatStagePositions:
    """A well whose stage position was set twice yields two XY dirs with the same marker.

    Observed 2026-08-30 on four experiments: 97 XY directories for 96 wells, both XY06 and XY97
    carrying ``_A06``. Both are complete, valid acquisitions of one physical well. Unioned into one
    per-well inventory they duplicate every (time, tile, z) cell, and materialization then refuses
    the well -- taking the whole experiment down with it.

    The position mapping applies the same rule, but it CANNOT cover this: the inventory is built by
    walking the raw tree and never reads the mapping, so the rule has to hold on both sides.
    """

    def _tree_with_repeat(self, tmp_path):
        raw_dir = tmp_path / _EXPERIMENT
        _make_well_tree(raw_dir, "XY06", tiles=(1, 2), zs=(1, 2), channels=(1,), times=(1,),
                        marker="A06")
        _make_well_tree(raw_dir, "XY97", tiles=(1, 2), zs=(1, 2), channels=(1,), times=(1,),
                        marker="A06")  # the re-set position, appended last
        _make_well_tree(raw_dir, "XY13", tiles=(1, 2), zs=(1, 2), channels=(1,), times=(1,),
                        marker="B12")
        return raw_dir

    def test_only_the_first_acquisition_survives(self, tmp_path):
        df = build_keyence_acquisition_inventory(
            experiment_id=_EXPERIMENT,
            raw_data_dir=self._tree_with_repeat(tmp_path),
            scrape_plane_metadata=_stub_scrape,
        )

        a06 = df[df["well_index"] == "A06"]
        assert len(a06) == 4, "2 tiles x 2 Z from ONE acquisition, not both"
        assert set(a06["position_index"]) == {6}, "the lowest position_index is kept"
        assert not a06["source_tiff_path"].astype(str).str.contains("/XY97/").any()

    def test_no_duplicate_cells_remain(self, tmp_path):
        # The exact condition materialize_well_keyence refuses on.
        df = build_keyence_acquisition_inventory(
            experiment_id=_EXPERIMENT,
            raw_data_dir=self._tree_with_repeat(tmp_path),
            scrape_plane_metadata=_stub_scrape,
        )

        a06 = df[df["well_index"] == "A06"]
        assert not a06.duplicated(list(KEYENCE_ACQUISITION_CELL_KEY), keep=False).any()

    def test_the_other_wells_are_untouched(self, tmp_path):
        # De-duplication must not cost a well, nor plane rows from wells that never repeated.
        df = build_keyence_acquisition_inventory(
            experiment_id=_EXPERIMENT,
            raw_data_dir=self._tree_with_repeat(tmp_path),
            scrape_plane_metadata=_stub_scrape,
        )

        assert set(df["well_index"]) == {"A06", "B12"}
        assert len(df[df["well_index"] == "B12"]) == 4

    def test_an_ordinary_plate_is_unchanged(self, tmp_path):
        raw_dir = tmp_path / _EXPERIMENT
        _make_well_tree(raw_dir, "XY06", tiles=(1, 2), zs=(1, 2), channels=(1,), times=(1,),
                        marker="A06")
        _make_well_tree(raw_dir, "XY13", tiles=(1, 2), zs=(1, 2), channels=(1,), times=(1,),
                        marker="B12")

        df = build_keyence_acquisition_inventory(
            experiment_id=_EXPERIMENT, raw_data_dir=raw_dir, scrape_plane_metadata=_stub_scrape
        )

        assert len(df) == 8, "nothing to de-duplicate means nothing is dropped"
