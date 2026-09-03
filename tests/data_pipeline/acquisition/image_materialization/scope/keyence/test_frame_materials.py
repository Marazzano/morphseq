"""Tests for prepare_keyence_frame — the product-independent half of Keyence materialization.

These pin the two boundary contracts the multi-product fanout rests on:
  1. every required source plane is read EXACTLY ONCE, independent of product count;
  2. ``FrameMaterials`` always owns resolved ``tile_transforms``, whatever their provenance.
Plus the uniform-Z invariant that a corrupt plane used to violate silently.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTileResult,
    FrameTilingConfig,
    PreComputeStitchParams,
    TileTransform,
    TilingQC,
    UnstitchableFrameError,
)
from data_pipeline.acquisition.image_materialization.scope.keyence.frame_materials import (
    FrameMaterials,
    prepare_keyence_frame,
)

MODULE = "data_pipeline.acquisition.image_materialization.scope.keyence.frame_materials"

WELL_ID = "20250912_B01"
TILE_SHAPE = (2, 2)


def _write_tiff(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(TILE_SHAPE, value, dtype=np.uint16)).save(path)


def _frame_rows(tmp_path: Path, *, n_tiles: int = 2, n_z: int = 3) -> pd.DataFrame:
    rows = []
    for tile_id in range(n_tiles):
        for z_index in range(n_z):
            raw = tmp_path / "raw" / f"t{tile_id}_z{z_index}.tif"
            _write_tiff(raw, tile_id * 10 + z_index)
            rows.append(
                {
                    "tile_id": tile_id,
                    "z_index": z_index,
                    "source_tiff_path": str(raw),
                }
            )
    return pd.DataFrame(rows)


def _mock_focus(n_tiles: int):
    tiles = tuple(
        SimpleNamespace(
            projection_u8=np.full(TILE_SHAPE, i, dtype=np.uint8),
            focus_index_map=np.zeros(TILE_SHAPE, dtype=np.int32),
        )
        for i in range(n_tiles)
    )
    return SimpleNamespace(tiles=tiles, intensity_lo=0, intensity_hi=65535, config=None)


def _mock_stitch(n_tiles: int = 2, *, passed: bool = True) -> FrameTileResult:
    return FrameTileResult(
        stitched=np.zeros((2, 4), dtype=np.uint8),
        tile_transforms={
            str(i): TileTransform(str(i), dx_px=float(i * 2), dy_px=0.0, source="align")
            for i in range(n_tiles)
        },
        canvas_shape=(2, 4),
        qc=TilingQC(
            passed=passed,
            reasons=tuple() if passed else ("implausible",),
            metrics={"tile_count": float(n_tiles)},
            suggested_action="ok" if passed else "fail",
        ),
        fallback_used="none",
    )


def _prepare(frame_rows, *, mode="auto", compute_focus=True, master=None, **kw):
    return prepare_keyence_frame(
        frame_rows=frame_rows,
        channel_id="BF",
        time_index=0,
        well_id=WELL_ID,
        tiling_config=FrameTilingConfig(orientation="horizontal", mode=mode),
        fallback=PreComputeStitchParams(master_params_path=master),
        compute_focus=compute_focus,
        **kw,
    )


class TestReadCountInvariant:
    """Every required source plane is read exactly once, whatever the products want."""

    def test_reads_each_plane_exactly_once(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=3)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)) as reader, \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            _prepare(rows)
        # 2 tiles x 3 z = 6 planes, read once each. Counting calls (not "loads") is deliberate:
        # replacing one bulk loop with two would pass a laxer assertion.
        assert reader.call_count == 6

    def test_focus_and_stitch_are_each_called_once(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=3)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)) as focus, \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)) as stitch:
            _prepare(rows)
        assert focus.call_count == 1
        assert stitch.call_count == 1

    def test_prior_only_without_focus_never_calls_focus_stack_group(self, tmp_path):
        """The decision table's zero-row: prior_only + no projection = no focus reduction at all."""
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=3)
        master = tmp_path / "master_params.json"
        master.write_text('{"coords": {"0": [0, 0], "1": [2, 0]}}')
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)) as reader, \
             patch(f"{MODULE}.focus_stack_group") as focus, \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            materials = _prepare(rows, mode="prior_only", compute_focus=False, master=master)
        focus.assert_not_called()
        assert materials.focus is None
        assert reader.call_count == 6


class TestTransformContract:
    """FrameMaterials always owns resolved transforms — emitters never learn the provenance."""

    def test_transforms_populated_under_auto(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=2)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            materials = _prepare(rows)
        assert set(materials.tile_transforms) == {"0", "1"}

    def test_transforms_populated_under_prior_only(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=2)
        master = tmp_path / "master_params.json"
        master.write_text('{"coords": {"0": [0, 0], "1": [2, 0]}}')
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            materials = _prepare(rows, mode="prior_only", compute_focus=False, master=master)
        assert set(materials.tile_transforms) == {"0", "1"}

    def test_single_tile_gets_explicit_identity_transform(self, tmp_path):
        """Not None: emitters index this mapping unconditionally, so the type must not vary."""
        rows = _frame_rows(tmp_path, n_tiles=1, n_z=2)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(1)), \
             patch(f"{MODULE}.stitch_frame_tiles") as stitch:
            materials = _prepare(rows)
        stitch.assert_not_called()
        assert materials.tile_transforms["0"].source == "identity"
        assert materials.tile_transforms["0"].dx_px == 0.0

    def test_untrustworthy_geometry_refuses_before_any_product(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=2)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2, passed=False)):
            with pytest.raises(UnstitchableFrameError):
                _prepare(rows)

    def test_content_aligning_mode_without_focus_fails_loud(self, tmp_path):
        """auto aligns on image content; raw out-of-focus planes are exactly what must not be used."""
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=2)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            with pytest.raises(ValueError, match="no focus reduction was computed"):
                _prepare(rows, mode="auto", compute_focus=False)


class TestUniformZInvariant:
    """A corrupt plane must not desync focus indices from real z values."""

    def _reader_dropping(self, drop_path_fragment: str):
        def _read(path, **kwargs):
            if drop_path_fragment in str(path):
                return None
            return np.full(TILE_SHAPE, 1, dtype=np.uint16)
        return _read

    def test_ragged_tiles_fail_loud_naming_the_tile(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=3)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=self._reader_dropping("t1_z1")), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            with pytest.raises(ValueError, match=r"Ragged Keyence z planes") as excinfo:
                _prepare(rows)
        message = str(excinfo.value)
        assert "'1'" in message, message
        assert "unreadable in '1': [1]" in message, message

    def test_ragged_tiles_fail_on_prior_only_too(self, tmp_path):
        """prior_only skips focus_stack_group, so its (Z,Y,X) check cannot backstop this path."""
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=3)
        master = tmp_path / "master_params.json"
        master.write_text('{"coords": {"0": [0, 0], "1": [2, 0]}}')
        with patch(f"{MODULE}.read_keyence_plane", side_effect=self._reader_dropping("t1_z1")), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            with pytest.raises(ValueError, match=r"Ragged Keyence z planes"):
                _prepare(rows, mode="prior_only", compute_focus=False, master=master)

    def test_uniformly_dropped_plane_is_allowed(self, tmp_path):
        """Every tile losing the SAME plane stays uniform — z_indices records what survived."""
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=3)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=self._reader_dropping("_z1")), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            materials = _prepare(rows)
        assert materials.z_indices.tolist() == [0, 2]
        assert materials.tile_stacks[0].shape[0] == 2

    def test_all_planes_unreadable_for_a_tile_fails_loud(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=2)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=self._reader_dropping("t1_")):
            with pytest.raises(ValueError, match="All z-plane TIFFs unreadable"):
                _prepare(rows)


class TestPlaneLookup:
    """plane_for indexes by REAL z_index, so a filtered stack cannot return the wrong depth."""

    def test_plane_for_indexes_through_z_indices(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=3)

        def _read(path, **kwargs):
            if "_z1" in str(path):
                return None
            value = 7 if "_z2" in str(path) else 1
            return np.full(TILE_SHAPE, value, dtype=np.uint16)

        with patch(f"{MODULE}.read_keyence_plane", side_effect=_read), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            materials = _prepare(rows)
        # z_index 2 sits at POSITION 1 after z1 was dropped; positional indexing would return z0.
        assert materials.plane_for("0", 2)[0, 0] == 7

    def test_plane_for_rejects_an_absent_z_index(self, tmp_path):
        rows = _frame_rows(tmp_path, n_tiles=2, n_z=2)
        with patch(f"{MODULE}.read_keyence_plane", side_effect=lambda p, **k: np.full(TILE_SHAPE, 1, np.uint16)), \
             patch(f"{MODULE}.focus_stack_group", return_value=_mock_focus(2)), \
             patch(f"{MODULE}.stitch_frame_tiles", return_value=_mock_stitch(2)):
            materials = _prepare(rows)
        with pytest.raises(ValueError, match="appears 0 times"):
            materials.plane_for("0", 99)


def test_empty_frame_rows_fail_loud(tmp_path):
    with pytest.raises(ValueError, match="no rows for well"):
        _prepare(pd.DataFrame(columns=["tile_id", "z_index", "source_tiff_path"]))
