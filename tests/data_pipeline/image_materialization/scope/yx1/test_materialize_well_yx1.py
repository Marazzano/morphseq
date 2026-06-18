"""Tests for materialize_well_yx1 — mocked ND2 + image ops, no GPU required."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_pipeline.image_materialization.scope.yx1.materialize_well_yx1 import (
    materialize_ff_projection,
    materialize_yx1_well,
    _EMITTED_COLUMNS,
)
from data_pipeline.image_materialization.materialized_image_paths import projection_frame_path

EXP = "20250912"
WELL_INDEX = "B01"
WELL_ID = f"{EXP}_{WELL_INDEX}"
ROOT = Path("/fake/built_image_data")


def _make_inventory(n_times: int = 3, position_index: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "experiment_id": EXP,
        "well_index": WELL_INDEX,
        "position_index": position_index,
        "time_index": list(range(n_times)),
        "source_nd2_path": "/fake/exp.nd2",
        "micrometers_per_pixel": 0.65,
        "image_width_px": 512,
        "image_height_px": 512,
        "channel_index": 0,
        "z_index": 0,
        "channel": "BF",
    })


class TestMaterializeFFProjection:
    def test_returns_2d_uint8(self):
        stack = np.random.randint(0, 1000, size=(5, 64, 64), dtype=np.uint16)
        with patch(
            "data_pipeline.image_materialization.scope.yx1"
            ".materialize_well_yx1.LoG_focus_stacker"
        ) as mock_log:
            mock_log.return_value = (np.ones((64, 64), dtype=np.float32) * 100, None)
            result = materialize_ff_projection(stack, device="cpu")
        assert result.ndim == 2
        assert result.dtype == np.uint8


class TestMaterializeYX1Well:
    def _make_mock_nd2(self, n_t: int = 3, n_w: int = 5, n_z: int = 4,
                       h: int = 8, w: int = 8):
        """Return a mock nd2.ND2File whose dask array has shape (T, W, Z, Y, X)."""
        dask_arr = MagicMock()
        # _get_stack slices dask_arr[t, w, :, :, :] → shape (Z, Y, X)
        dask_arr.ndim = 5
        fake_stack = np.ones((n_z, h, w), dtype=np.uint16)
        dask_arr.__getitem__ = MagicMock(return_value=MagicMock(compute=lambda: fake_stack))

        channel_mock = MagicMock()
        channel_mock.channel.name = "BF"

        nd_mock = MagicMock()
        nd_mock.to_dask.return_value = dask_arr
        nd_mock.frame_metadata.return_value.channels = [channel_mock]
        return nd_mock

    def _run(self, inventory_df, tmp_path, candidate=True):
        nd_mock = self._make_mock_nd2(n_t=len(inventory_df))

        _mod = "data_pipeline.image_materialization.scope.yx1.materialize_well_yx1"
        with (
            patch(f"{_mod}.nd2.ND2File", return_value=nd_mock),
            patch(f"{_mod}.materialize_ff_projection",
                  return_value=np.zeros((8, 8), dtype=np.uint8)) as mock_proj,
            patch(f"{_mod}.skio.imsave") as mock_imsave,
            patch(f"{_mod}._get_stack",
                  return_value=np.ones((4, 8, 8), dtype=np.uint16)) as mock_get_stack,
        ):

            return materialize_yx1_well(
                experiment_id=EXP,
                well_id=WELL_ID,
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inventory_df,
                nd2_path=Path("/fake/exp.nd2"),
                built_image_data_dir=tmp_path,
                device="cpu",
                candidate=candidate,
            )

    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        inv = _make_inventory(n_times=3)
        df = self._run(inv, tmp_path)
        for col in _EMITTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_one_row_per_time_index(self, tmp_path):
        n = 4
        inv = _make_inventory(n_times=n)
        df = self._run(inv, tmp_path)
        assert len(df) == n

    def test_no_duplicate_rows_on_key(self, tmp_path):
        inv = _make_inventory(n_times=5)
        df = self._run(inv, tmp_path)
        key = ["experiment_id", "well_index", "channel_id", "time_index"]
        assert not df.duplicated(subset=key).any()

    def test_source_image_path_matches_layout(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df = self._run(inv, tmp_path, candidate=True)
        for _, row in df.iterrows():
            expected = projection_frame_path(
                tmp_path,
                experiment_id=row["experiment_id"],
                well_id=WELL_ID,
                channel_id=row["channel_id"],
                time_index=row["time_index"],
                candidate=True,
            )
            assert row["source_image_path"] == str(expected)

    def test_z_index_is_na_for_projection(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df = self._run(inv, tmp_path)
        assert df["z_index"].isna().all()

    def test_image_product_type_is_projection(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df = self._run(inv, tmp_path)
        assert (df["image_product_type"] == "projection").all()

    def test_projection_method_is_focus_stack(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df = self._run(inv, tmp_path)
        assert (df["projection_method"] == "focus_stack").all()

    def test_mismatched_well_id_raises(self, tmp_path):
        inv = _make_inventory(n_times=1)
        with pytest.raises(ValueError, match="well_id mismatch"):
            materialize_yx1_well(
                experiment_id=EXP,
                well_id="20250912_C04",  # wrong
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inv,
                nd2_path=Path("/fake/exp.nd2"),
                built_image_data_dir=tmp_path,
                device="cpu",
            )

    def test_ambiguous_position_index_raises(self, tmp_path):
        inv = _make_inventory(n_times=2)
        inv = inv.copy()
        inv.loc[1, "position_index"] = 99  # two different position_index values
        with pytest.raises(ValueError, match="position_index"):
            materialize_yx1_well(
                experiment_id=EXP,
                well_id=WELL_ID,
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inv,
                nd2_path=Path("/fake/exp.nd2"),
                built_image_data_dir=tmp_path,
                device="cpu",
            )

    def test_empty_inventory_raises(self, tmp_path):
        inv = _make_inventory(n_times=0)
        with pytest.raises(ValueError, match="empty"):
            materialize_yx1_well(
                experiment_id=EXP,
                well_id=WELL_ID,
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inv,
                nd2_path=Path("/fake/exp.nd2"),
                built_image_data_dir=tmp_path,
                device="cpu",
            )
