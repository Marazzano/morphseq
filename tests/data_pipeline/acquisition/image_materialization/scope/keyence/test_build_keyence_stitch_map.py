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


def _plane_path(
    well_id: str, t: int, tile_id: int, z: int, *, input_root: Path | None
) -> str:
    """One raw plane path, touched on disk when ``input_root`` is given.

    The path always carries a ``raw_image_data/`` segment because
    ``shared.path_roots.resolve_under_input_root(full_root_fallback=True)`` re-anchors on that
    segment and REJECTS a nonexistent path that lacks one.
    """
    tail = Path("raw_image_data/keyence/20250912") / f"{well_id}_t{t}_tile{tile_id}_z{z}.tif"
    if input_root is None:
        return str(Path("/fake") / tail)
    path = Path(input_root) / tail
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def _make_inventory(
    n_wells: int = 2,
    n_time: int = 3,
    n_tiles: int = 3,
    n_z: int = 2,
    orientation: str = "vertical",
    x_offset: float = 10.0,
    input_root: Path | None = None,
) -> pd.DataFrame:
    """A minimal Keyence acquisition inventory carrying every column the builder reads.

    ``stage_x_nm`` / ``stage_y_nm`` / ``micrometers_per_pixel`` are Tier-2 Keyence columns declared
    in ``scope/keyence/acquisition_inventory.KEYENCE_ACQUISITION_INVENTORY_SCOPE_COLUMNS``; the
    Stage-A stage prior (``stage_prior_offsets``) reads them, so they are NOT optional here.

    The stage coordinates are chosen so the derived prior equals the fake stitch2d fits the tests
    inject (tile i at x = ``x_offset`` * i): ``stage_prior_offsets`` computes
    ``-stage_x_nm / 1000 / micrometers_per_pixel``, so with µm/px == 1.0 the stage X for tile i is
    ``-x_offset * 1000 * i``. Without this agreement Stage C's deviation filter would reject every
    fit and the map would silently fall back to the prior.

    When ``input_root`` is given the plane files are TOUCHED on disk under it. The pixel reader is
    always mocked, but ``resolve_under_input_root(full_root_fallback=True)`` still stats the path,
    so a nonexistent file makes _build_tile_specs raise and every well is silently dropped.
    """
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
                        # Must carry a 'raw_image_data/' segment: _build_tile_specs resolves via
                        # shared.path_roots.resolve_under_input_root(full_root_fallback=True),
                        # which REJECTS a nonexistent path with no segment to re-anchor on.
                        "source_tiff_path": _plane_path(
                            well_id, t, tile_id, z, input_root=input_root
                        ),
                        "orientation": orientation,
                        "channel_id": "BF",
                        "micrometers_per_pixel": 1.0,
                        "stage_x_nm": -x_offset * 1000.0 * tile_id,
                        "stage_y_nm": 0.0,
                    })
    return pd.DataFrame(rows)


def _fake_raw_coords(tile_ids: list[str], x_offset: float = 10.0) -> dict[int, list[float]]:
    return {
        idx: [0.0, x_offset * idx]
        for idx, _tile_id in enumerate(tile_ids)
    }


_BUILD_MOD = "data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map"


def _fake_focus_group(stacks_zyx, *, config=None, device="cpu"):
    """focus_stack_group stand-in: one uint8 projection tile per input stack."""
    from types import SimpleNamespace

    tiles = tuple(
        SimpleNamespace(
            projection_u8=np.zeros((10, 10), dtype=np.uint8),
            focus_index_map=np.zeros((10, 10), dtype=np.int32),
        )
        for _ in stacks_zyx
    )
    return SimpleNamespace(tiles=tiles, intensity_lo=0, intensity_hi=65535, config=config)


def _patch_focus_group():
    return patch(f"{_BUILD_MOD}.focus_stack_group", side_effect=_fake_focus_group)


def _patch_io_and_stitch(tile_ids: list[str], x_offset: float = 10.0):
    fake_image = np.zeros((10, 10), dtype=np.uint8)
    return [
        patch(f"{_BUILD_MOD}.read_keyence_plane", return_value=fake_image),
        _patch_focus_group(),
        patch(f"{_BUILD_MOD}.raw_stitch2d_align",
              return_value=_fake_raw_coords(tile_ids, x_offset)),
    ]


def test_writes_coords_json(tmp_path):
    inv = _make_inventory(input_root=tmp_path, n_wells=1, n_time=3, n_tiles=3, orientation="vertical")
    out = tmp_path / "master_params.json"
    tile_ids = ["0", "1", "2"]

    patches = _patch_io_and_stitch(tile_ids, x_offset=10.0)
    with patches[0], patches[1], patches[2]:
        build_keyence_stitch_map(inv, n_samples=3, out_path=out, input_root=tmp_path)

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
    inv = _make_inventory(input_root=tmp_path, n_wells=3, n_time=5, n_tiles=3, orientation="vertical")
    out1 = tmp_path / "map1.json"
    out2 = tmp_path / "map2.json"
    tile_ids = ["0", "1", "2"]

    patches = _patch_io_and_stitch(tile_ids)
    with patches[0], patches[1], patches[2]:
        build_keyence_stitch_map(inv, n_samples=5, out_path=out1, input_root=tmp_path)

    patches2 = _patch_io_and_stitch(tile_ids)
    with patches2[0], patches2[1], patches2[2]:
        build_keyence_stitch_map(inv, n_samples=5, out_path=out2, input_root=tmp_path)

    assert out1.read_bytes() == out2.read_bytes()


def test_raises_when_no_good_samples(tmp_path):
    inv = _make_inventory(input_root=tmp_path, n_wells=1, n_time=2, n_tiles=3)
    out = tmp_path / "map.json"

    with (
        patch(f"{_BUILD_MOD}.read_keyence_plane",
              side_effect=OSError("file not found")),
    ):
        with pytest.raises(RuntimeError, match="no sample fully aligned"):
            build_keyence_stitch_map(inv, n_samples=5, out_path=out, input_root=tmp_path)

    assert not out.exists()


def test_orientation_from_inventory(tmp_path):
    inv = _make_inventory(input_root=tmp_path, n_wells=1, n_time=2, n_tiles=2, orientation="horizontal")
    out = tmp_path / "map.json"
    tile_ids = ["0", "1"]

    captured_orientations = []

    def _fake_align(tile_specs, orientation):
        captured_orientations.append(orientation)
        return _fake_raw_coords(tile_ids)

    fake_image = np.zeros((10, 10), dtype=np.uint8)
    with (
        patch(f"{_BUILD_MOD}.read_keyence_plane",
              return_value=fake_image),
        _patch_focus_group(),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.raw_stitch2d_align",
              side_effect=_fake_align),
    ):
        build_keyence_stitch_map(inv, n_samples=2, out_path=out, input_root=tmp_path)

    assert all(orientation == "horizontal" for orientation in captured_orientations)


def test_unknown_orientation_defaults_to_horizontal(tmp_path):
    inv = _make_inventory(input_root=tmp_path, n_wells=1, n_time=2, n_tiles=2, orientation="unknown")
    out = tmp_path / "map.json"
    tile_ids = ["0", "1"]

    captured_orientations = []

    def _fake_align(tile_specs, orientation):
        captured_orientations.append(orientation)
        return _fake_raw_coords(tile_ids)

    fake_image = np.zeros((10, 10), dtype=np.uint8)
    with (
        patch(f"{_BUILD_MOD}.read_keyence_plane",
              return_value=fake_image),
        _patch_focus_group(),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.raw_stitch2d_align",
              side_effect=_fake_align),
    ):
        build_keyence_stitch_map(inv, n_samples=2, out_path=out, input_root=tmp_path)

    assert all(orientation == "horizontal" for orientation in captured_orientations)


def test_raises_on_empty_inventory(tmp_path):
    inv = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        build_keyence_stitch_map(inv, n_samples=5, out_path=tmp_path / "map.json")


def test_skips_partial_alignments_and_keeps_good_samples(tmp_path):
    inv = _make_inventory(input_root=tmp_path, n_wells=1, n_time=3, n_tiles=3)
    out = tmp_path / "map.json"

    fake_image = np.zeros((10, 10), dtype=np.uint8)
    partial = {0: [0.0, 0.0], 1: [700.0, 1.0]}
    full = {0: [0.0, 0.0], 1: [700.0, 1.0], 2: [1400.0, 2.0]}

    with (
        patch(f"{_BUILD_MOD}.read_keyence_plane",
              return_value=fake_image),
        _patch_focus_group(),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.raw_stitch2d_align",
              side_effect=[partial, full, partial]),
    ):
        build_keyence_stitch_map(inv, n_samples=3, out_path=out, input_root=tmp_path)

    data = json.loads(out.read_text())
    assert data["coords"] == {"0": [0.0, 0.0], "1": [700.0, 1.0], "2": [1400.0, 2.0]}
    assert data["metadata"]["shape"] == [3, 1]
