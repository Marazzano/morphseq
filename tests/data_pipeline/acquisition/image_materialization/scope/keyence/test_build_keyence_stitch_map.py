"""Tests for build_keyence_stitch_map — mocked stitcher, no real TIFFs required."""

import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map import (
    build_keyence_stitch_map,
    stage_prior_offsets,
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


def test_falls_back_to_the_stage_prior_when_no_sample_aligns(tmp_path, caplog):
    """No well aligns -> the map is written from the stage prior, and SAYS so.

    POLICY CHANGE, not a regression. This test previously asserted RuntimeError("no sample fully
    aligned"). The builder deliberately stopped failing for lack of a fully-aligned sample
    (docstring, build_keyence_stitch_map:57-60): every tile index falls back to the stage prior so
    the map always covers all tiles.

    The test therefore asserts the SEMANTIC RESULT of the fallback rather than being rewritten to
    echo whatever the implementation now returns. What matters is that a map with no image evidence
    behind it is DISTINGUISHABLE from a calibrated one -- a silent fallback is how an uncalibrated
    map gets trusted as a measured one.
    """
    inv = _make_inventory(input_root=tmp_path, n_wells=1, n_time=2, n_tiles=3)
    out = tmp_path / "map.json"

    with caplog.at_level(logging.WARNING):
        with patch(f"{_BUILD_MOD}.read_keyence_plane", side_effect=OSError("file not found")):
            build_keyence_stitch_map(inv, n_samples=5, out_path=out, input_root=tmp_path)

    # 1. a map IS produced, covering every tile
    assert out.exists(), "the fallback must still write a usable map"
    written = json.loads(out.read_text())
    assert len(written["coords"]) == 3

    # 2. it records that no data-supported estimate was available
    assert written["metadata"]["alignment_source"] == "stage_prior"
    assert written["metadata"]["n_good_samples"] == 0

    # 3. the offsets are STAGE-DERIVED, not invented. Deliberately a structural check rather than
    #    a value-by-value comparison against stage_prior_offsets(): that helper is per-WELL and
    #    anchors to the CENTER tile, whereas the written map is deliberately re-normalized to a
    #    MIN-ORIGIN frame (build_keyence_stitch_map:94-102) so it matches how stitch2d renormalizes
    #    loaded params. Re-deriving the expected values here would mean copying both the anchoring
    #    and that renormalization into the test -- which is how a test starts asserting "whatever
    #    the implementation returns".
    #
    #    What must hold is that the prior is a real geometric layout, expressed in the frame the
    #    map actually ships: every axis touches its own origin, nothing is negative, and the tiles
    #    occupy genuinely distinct positions.
    offsets = np.asarray([written["coords"][str(i)] for i in range(len(written["coords"]))])
    assert (offsets >= 0).all(), "the written map is min-origin, so no offset may be negative"
    assert (offsets.min(axis=0) == 0).all(), (
        "each axis must touch zero -- otherwise the map is not in the min-origin frame that "
        "downstream materialization QC compares against"
    )
    assert len({tuple(o) for o in offsets}) == len(offsets), (
        "tiles stacked on identical offsets means no layout was derived at all"
    )

    # 4. it is loud about it
    assert any("stage prior" in r.message.lower() for r in caplog.records), (
        "a whole-map fallback with zero image evidence must warn"
    )


def test_malformed_input_still_fails(tmp_path):
    """The fallback must not become a blanket error sponge.

    "Never fails for lack of an aligned sample" is a statement about IMAGE EVIDENCE, not a licence
    to absorb structurally broken input. An empty inventory has no tiles to lay out and no stage
    positions to derive a prior from, so there is nothing to fall back TO.
    """
    with pytest.raises(ValueError, match="empty"):
        build_keyence_stitch_map(
            _make_inventory(input_root=tmp_path, n_wells=1, n_time=1, n_tiles=1).iloc[0:0],
            n_samples=5,
            out_path=tmp_path / "map.json",
            input_root=tmp_path,
        )


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
    """A well that places only some tiles still contributes those tiles; the rest use the prior.

    THE INJECTED FITS MUST AGREE WITH THE STAGE PRIOR. Stage C rejects any fit deviating from the
    prior by more than DEFAULT_CALIBRATION_THRESHOLD_PX in either axis, so fits invented
    independently of the fixture's stage coordinates are all filtered out and every tile silently
    falls back to the prior -- which makes the test pass or fail for reasons unrelated to partial
    alignment. _make_inventory places tile i at x = x_offset * i (default 10.0) in the [y, x]
    frame, so the fits must too.
    """
    inv = _make_inventory(input_root=tmp_path, n_wells=1, n_time=3, n_tiles=3)
    out = tmp_path / "map.json"

    fake_image = np.zeros((10, 10), dtype=np.uint8)
    # Center-anchored (tile 1 is the center of 3), matching what stitch2d hands back before
    # collect_center_anchored_fits re-anchors it.
    partial = {0: [0.0, -10.0], 1: [0.0, 0.0]}
    full = {0: [0.0, -10.0], 1: [0.0, 0.0], 2: [0.0, 10.0]}

    with (
        patch(f"{_BUILD_MOD}.read_keyence_plane",
              return_value=fake_image),
        _patch_focus_group(),
        patch("data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map.raw_stitch2d_align",
              side_effect=[partial, full, partial]),
    ):
        build_keyence_stitch_map(inv, n_samples=3, out_path=out, input_root=tmp_path)

    data = json.loads(out.read_text())
    # Written in the MIN-ORIGIN frame, so the center-anchored -10/0/+10 above shifts to 0/10/20.
    assert data["coords"] == {"0": [0.0, 0.0], "1": [0.0, 10.0], "2": [0.0, 20.0]}
    assert data["metadata"]["shape"] == [3, 1]
    # Tile 2 placed in only ONE of the three samples, but that one fit survived the prior filter,
    # so the map is calibrated rather than falling back for it.
    assert data["metadata"]["alignment_source"] == "aligned"
