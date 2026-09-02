import json

import numpy as np
import pytest

from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    IncompleteTileAlignmentError,
    PreComputeStitchParams,
    TileSpec,
    TileTransform,
    UnstitchableFrameError,
    _coords_to_transforms,
    _infer_layout_orientation,
    _run_tiling_qc,
    _stitch_with_stitch2d,
    stitch_frame_tiles,
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


def test_layout_orientation_uses_transform_span_not_requested_orientation():
    transforms = {
        "0": TileTransform("0", dx_px=0.0, dy_px=0.0, source="align"),
        "1": TileTransform("1", dx_px=680.0, dy_px=1.0, source="align"),
        "2": TileTransform("2", dx_px=1360.0, dy_px=0.0, source="align"),
    }

    orientation = _infer_layout_orientation(
        transforms,
        fallback_orientation="vertical",
    )

    assert orientation == "horizontal"


class TestOpenCVAlignmentFailureFallsBack:
    """A feature-poor tile makes OpenCV assert instead of returning "no matches".

    Seen on 20230525 well A07 (2026-08-30):

        cv2.error: (-215:Assertion failed) (size_t)knn <= index_->size()
                   in function 'runKnnSearch_'

    raised from FLANN's knnMatch, deep inside ``mosaic.align()``. ``cv2.error`` is not a
    RuntimeError, so stitch2d's own "Could not align tiles" handler never saw it and the exception
    escaped to kill the well -- and with it every product for that experiment. Being unable to align
    is precisely what the master-params fallback exists for, so it must be translated, not
    propagated.
    """

    def _tiles(self):
        rng = np.random.default_rng(0)
        return [
            TileSpec(tile_id=str(i), image=rng.integers(0, 255, (48, 64), dtype=np.uint8))
            for i in (1, 2, 3)
        ]

    def _boom(self, monkeypatch):
        import cv2
        import stitch2d.mosaic

        def raise_knn_assertion(self, *args, **kwargs):
            raise cv2.error(
                "OpenCV(4.10.0) /io/opencv/modules/flann/src/miniflann.cpp:521: error: "
                "(-215:Assertion failed) (size_t)knn <= index_->size() in function 'runKnnSearch_'"
            )

        monkeypatch.setattr(stitch2d.mosaic.StructuredMosaic, "align", raise_knn_assertion)

    def test_it_becomes_an_alignment_error_not_a_cv2_error(self, monkeypatch):
        self._boom(monkeypatch)

        with pytest.raises(IncompleteTileAlignmentError, match="too few keypoints"):
            _stitch_with_stitch2d(
                tiles=self._tiles(),
                orientation="horizontal",
                load_params_path=None,
                run_align=True,
            )

    def test_master_params_carry_the_frame(self, tmp_path, monkeypatch):
        self._boom(monkeypatch)
        master = tmp_path / "master.json"
        master.write_text(
            json.dumps(
                {
                    "metadata": {"shape": [1, 3], "size": 3, "tile_shape": [48, 64]},
                    "coords": {"0": [0.0, 0.0], "1": [34.0, 0.7], "2": [68.0, 1.5]},
                }
            )
        )

        result = stitch_frame_tiles(
            self._tiles(),
            FrameTilingConfig(orientation="horizontal"),
            PreComputeStitchParams(master_params_path=master),
        )

        assert result.fallback_used == "master"

    def test_with_no_master_it_still_fails_loud(self, monkeypatch):
        # The translation must not become a licence to emit a garbage mosaic. With no master coords
        # there is nothing to fall back ONTO, and a silently mis-stitched frame would pass
        # downstream QC as though it were real.
        self._boom(monkeypatch)

        with pytest.raises(UnstitchableFrameError):
            stitch_frame_tiles(
                self._tiles(),
                FrameTilingConfig(orientation="horizontal"),
                PreComputeStitchParams(),
            )
