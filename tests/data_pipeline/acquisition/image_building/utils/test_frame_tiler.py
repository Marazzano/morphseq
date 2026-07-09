import numpy as np

from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    TileSpec,
    TileTransform,
    _coords_to_transforms,
    _run_tiling_qc,
)


def test_coords_to_transforms_converts_stitch2d_yx_to_dx_dy():
    tiles = [
        TileSpec(tile_id="0", image=np.zeros((4, 4), dtype=np.uint8)),
        TileSpec(tile_id="1", image=np.zeros((4, 4), dtype=np.uint8)),
    ]

    transforms = _coords_to_transforms(tiles, {0: [0.0, 0.0], 1: [700.0, 2.0]})

    assert transforms["1"].dy_px == 700.0
    assert transforms["1"].dx_px == 2.0


def test_vertical_qc_allows_large_along_strip_shift_without_master():
    transforms = {
        "0": TileTransform("0", dx_px=0.0, dy_px=0.0, source="align"),
        "1": TileTransform("1", dx_px=1.0, dy_px=700.0, source="align"),
        "2": TileTransform("2", dx_px=2.0, dy_px=1400.0, source="align"),
    }

    qc = _run_tiling_qc(
        transforms,
        FrameTilingConfig(orientation="vertical"),
    )

    assert qc.passed
    assert qc.metrics["max_abs_shift_px"] == 1400.0
    assert qc.metrics["max_cross_axis_shift_px"] == 2.0


def test_vertical_qc_rejects_sideways_wobble_without_master():
    transforms = {
        "0": TileTransform("0", dx_px=0.0, dy_px=0.0, source="align"),
        "1": TileTransform("1", dx_px=3.0, dy_px=700.0, source="align"),
        "2": TileTransform("2", dx_px=0.0, dy_px=1400.0, source="align"),
    }

    qc = _run_tiling_qc(
        transforms,
        FrameTilingConfig(orientation="vertical"),
    )

    assert not qc.passed
    assert qc.reasons == ("cross_axis_shift_exceeds_threshold",)


def test_master_qc_compares_against_yx_reference_coords():
    transforms = {
        "0": TileTransform("0", dx_px=0.0, dy_px=0.0, source="align"),
        "1": TileTransform("1", dx_px=2.0, dy_px=700.0, source="align"),
        "2": TileTransform("2", dx_px=4.0, dy_px=1400.0, source="align"),
    }
    tiles = [
        TileSpec(tile_id=str(idx), image=np.zeros((4, 4), dtype=np.uint8))
        for idx in range(3)
    ]
    master_coords = {
        0: [0.0, 0.0],
        1: [700.0, 2.0],
        2: [1400.0, 4.0],
    }

    qc = _run_tiling_qc(
        transforms,
        FrameTilingConfig(orientation="vertical"),
        master_coords=master_coords,
        tiles=tiles,
    )

    assert qc.passed
    assert qc.metrics["max_master_deviation_px"] == 0.0
