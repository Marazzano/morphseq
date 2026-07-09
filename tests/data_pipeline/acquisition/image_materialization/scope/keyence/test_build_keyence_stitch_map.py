"""Tests for build_keyence_stitch_map — mocked stitcher, no real TIFFs required."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map import (
    build_keyence_stitch_map,
)


def _make_inventory(
    n_wells: int = 2,
    n_time: int = 3,
    n_tiles: int = 3,
    n_z: int = 2,
    orientation: str = "vertical",
) -> pd.DataFrame:
    rows = []
    for w in range(n_wells):
        well_id = f"20250912_W0{w}"
        for t in range(n_time):
            for tile_id in range(n_tiles):
                for z in range(n_z):
                    rows.append({
                        "well_id": well_id,
                        "time_index": t,
                        "tile_id": str(tile_id),
                        "z_index": z,
                        "source_tiff_path": f"/fake/{well_id}_t{t}_tile{tile_id}_z{z}.tif",
                        "orientation": orientation,
                        "channel_id": "BF",
                    })
    return pd.DataFrame(rows)


def _fake_raw_coords(tile_ids: list[str], x_offset: float = 10.0) -> dict[int, list[float]]:
    return {
        idx: [0.0, x_offset * idx]
        for idx, _tile_id in enumerate(tile_ids)
    }


def _patch_io_and_stitch(tile_ids: list[str], x_offset: float = 10.0):
    fake_image = np.zeros((10, 10), dtype=np.uint8)
    fake_ff = (np.zeros((10, 10), dtype=np.float32), None)
    return [
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.skio.imread",
              return_value=fake_image),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.im_rescale",
              return_value=(np.zeros((2, 10, 10), dtype=np.float32), None, None)),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.materialize_ff_projection",
              return_value=fake_ff),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.raw_stitch2d_align",
              return_value=_fake_raw_coords(tile_ids, x_offset)),
    ]


def test_writes_coords_json(tmp_path):
    inv = _make_inventory(n_wells=1, n_time=3, n_tiles=3, orientation="vertical")
    out = tmp_path / "master_params.json"
    tile_ids = ["0", "1", "2"]

    patches = _patch_io_and_stitch(tile_ids, x_offset=10.0)
    with patches[0], patches[1], patches[2], patches[3]:
        build_keyence_stitch_map(inv, n_samples=3, out_path=out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert "coords" in data
    coords = data["coords"]
    assert set(coords.keys()) == set(tile_ids)
    assert data["metadata"]["shape"] == [3, 1]
    assert data["metadata"]["size"] == 3
    assert data["metadata"]["tile_shape"] == [10, 10]
    # stitch2d coords are [y, x]; tile 1 x=10, tile 2 x=20 across identical samples.
    assert abs(coords["0"][0] - 0.0) < 1e-6
    assert abs(coords["1"][1] - 10.0) < 1e-6
    assert abs(coords["2"][1] - 20.0) < 1e-6


def test_deterministic_seed(tmp_path):
    inv = _make_inventory(n_wells=3, n_time=5, n_tiles=3, orientation="vertical")
    out1 = tmp_path / "map1.json"
    out2 = tmp_path / "map2.json"
    tile_ids = ["0", "1", "2"]

    patches = _patch_io_and_stitch(tile_ids)
    with patches[0], patches[1], patches[2], patches[3]:
        build_keyence_stitch_map(inv, n_samples=5, out_path=out1)

    patches2 = _patch_io_and_stitch(tile_ids)
    with patches2[0], patches2[1], patches2[2], patches2[3]:
        build_keyence_stitch_map(inv, n_samples=5, out_path=out2)

    assert out1.read_bytes() == out2.read_bytes()


def test_raises_when_no_good_samples(tmp_path):
    inv = _make_inventory(n_wells=1, n_time=2, n_tiles=3)
    out = tmp_path / "map.json"

    with (
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.skio.imread",
              side_effect=OSError("file not found")),
    ):
        with pytest.raises(RuntimeError, match="no sample fully aligned"):
            build_keyence_stitch_map(inv, n_samples=5, out_path=out)

    assert not out.exists()


def test_orientation_from_inventory(tmp_path):
    inv = _make_inventory(n_wells=1, n_time=2, n_tiles=2, orientation="horizontal")
    out = tmp_path / "map.json"
    tile_ids = ["0", "1"]

    captured_orientations = []

    def _fake_align(tile_specs, orientation):
        captured_orientations.append(orientation)
        return _fake_raw_coords(tile_ids)

    fake_image = np.zeros((10, 10), dtype=np.uint8)
    fake_ff = (np.zeros((10, 10), dtype=np.float32), None)
    with (
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.skio.imread",
              return_value=fake_image),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.im_rescale",
              return_value=(np.zeros((2, 10, 10), dtype=np.float32), None, None)),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.materialize_ff_projection",
              return_value=fake_ff),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.raw_stitch2d_align",
              side_effect=_fake_align),
    ):
        build_keyence_stitch_map(inv, n_samples=2, out_path=out)

    assert all(orientation == "horizontal" for orientation in captured_orientations)


def test_raises_on_empty_inventory(tmp_path):
    inv = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        build_keyence_stitch_map(inv, n_samples=5, out_path=tmp_path / "map.json")


def test_skips_partial_alignments_and_keeps_good_samples(tmp_path):
    inv = _make_inventory(n_wells=1, n_time=3, n_tiles=3)
    out = tmp_path / "map.json"

    fake_image = np.zeros((10, 10), dtype=np.uint8)
    fake_ff = (np.zeros((10, 10), dtype=np.float32), None)
    partial = {0: [0.0, 0.0], 1: [700.0, 1.0]}
    full = {0: [0.0, 0.0], 1: [700.0, 1.0], 2: [1400.0, 2.0]}

    with (
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.skio.imread",
              return_value=fake_image),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.im_rescale",
              return_value=(np.zeros((2, 10, 10), dtype=np.float32), None, None)),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.materialize_ff_projection",
              return_value=fake_ff),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.raw_stitch2d_align",
              side_effect=[partial, full, partial]),
    ):
        build_keyence_stitch_map(inv, n_samples=3, out_path=out)

    data = json.loads(out.read_text())
    assert data["coords"] == {"0": [0.0, 0.0], "1": [700.0, 1.0], "2": [1400.0, 2.0]}
    assert data["metadata"]["shape"] == [3, 1]
