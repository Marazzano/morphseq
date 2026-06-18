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
from data_pipeline.image_materialization.materialization_plan import (
    ResolvedImageProduct,
    ResolvedMaterializationPlan,
)
from data_pipeline.shared.identifiers.constructors import build_well_id

EXP = "20250912"
WELL_INDEX = "B01"
WELL_ID = build_well_id(EXP, WELL_INDEX)  # never mint ids by hand, even in tests
ROOT = Path("/fake/built_image_data")

# The one accepted YX1 product, already resolved (identity XY composition).
IDENTITY_PLAN = ResolvedMaterializationPlan(
    products=(
        ResolvedImageProduct(
            channel_id="BF",
            image_product_type="projection",
            projection_method="focus_stack",
            xy_composition="identity",
        ),
    )
)


def _make_inventory(
    n_times: int = 3,
    position_index: int = 2,
    source_nd2_path: str = "/fake/exp.nd2",
) -> pd.DataFrame:
    # Schema-complete acquisition shard (the consume-side validator re-checks the full contract).
    # In production this is the acquisition_inventory CSV merged with position_well_mapping; the test
    # carries every YX1_ACQUISITION_INVENTORY_COLUMN so the contract validator passes structurally.
    return pd.DataFrame({
        "experiment_id": EXP,
        "raw_position_label": str(position_index),
        "position_index": position_index,
        "z_index": 0,
        "channel_index": 0,
        "channel": "BF",
        "raw_channel_name": "EYES - Dia",
        "time_index": list(range(n_times)),
        "acquisition_time_s": [100.0 * t for t in range(n_times)],
        "x_um": 10.0,
        "y_um": 20.0,
        "micrometers_per_pixel": 0.65,
        "image_width_px": 512,
        "image_height_px": 512,
        "objective_magnification": "4x",
        "microscope_id": "YX1",
        "n_z": 1,
        "source_nd2_path": source_nd2_path,
        # joined identity columns the materializer also receives:
        "well_index": WELL_INDEX,
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
        # The consume-side contract check asserts source_nd2_path EXISTS before the open is faked,
        # so point the shard at a real (empty) file under tmp_path.
        nd2_file = tmp_path / "exp.nd2"
        nd2_file.write_bytes(b"")
        inventory_df = inventory_df.copy()
        inventory_df["source_nd2_path"] = str(nd2_file)
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
                built_image_data_dir=tmp_path,
                resolved_plan=IDENTITY_PLAN,
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
                well_id=build_well_id(EXP, "C04"),  # wrong well_index on purpose
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inv,
                built_image_data_dir=tmp_path,
                resolved_plan=IDENTITY_PLAN,
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
                built_image_data_dir=tmp_path,
                resolved_plan=IDENTITY_PLAN,
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
                built_image_data_dir=tmp_path,
                resolved_plan=IDENTITY_PLAN,
                device="cpu",
            )

    def test_non_identity_resolved_product_raises(self, tmp_path):
        inv = _make_inventory(n_times=1)
        bad_plan = ResolvedMaterializationPlan(
            products=(
                ResolvedImageProduct(
                    channel_id="BF",
                    image_product_type="projection",
                    projection_method="focus_stack",
                    xy_composition="mosaic",  # YX1 should never receive this
                ),
            )
        )
        with pytest.raises(ValueError, match="identity"):
            materialize_yx1_well(
                experiment_id=EXP,
                well_id=WELL_ID,
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inv,
                built_image_data_dir=tmp_path,
                resolved_plan=bad_plan,
                device="cpu",
            )

    def test_missing_source_nd2_fails_loud_before_tensor_read(self, tmp_path):
        # Inventory points at an ND2 that was moved/deleted since extraction. The consume-side
        # contract check must fail loud (path named) at the entry guard, before any tensor read.
        inv = _make_inventory(n_times=2, source_nd2_path="/gone/missing.nd2")
        _mod = "data_pipeline.image_materialization.scope.yx1.materialize_well_yx1"
        with patch(f"{_mod}.nd2.ND2File") as mock_open:
            with pytest.raises(ValueError, match=r"missing\.nd2.*does not exist|does not exist.*missing\.nd2"):
                materialize_yx1_well(
                    experiment_id=EXP,
                    well_id=WELL_ID,
                    well_index=WELL_INDEX,
                    well_acquisition_inventory_df=inv,
                    built_image_data_dir=tmp_path,
                    resolved_plan=IDENTITY_PLAN,
                    device="cpu",
                )
            mock_open.assert_not_called()  # never reached the tensor read

    def test_smoke_cap_limits_time_indices(self, tmp_path):
        nd2_file = tmp_path / "exp.nd2"
        nd2_file.write_bytes(b"")
        inv = _make_inventory(n_times=10, source_nd2_path=str(nd2_file))
        nd_mock = self._make_mock_nd2(n_t=10)
        _mod = "data_pipeline.image_materialization.scope.yx1.materialize_well_yx1"
        with (
            patch(f"{_mod}.nd2.ND2File", return_value=nd_mock),
            patch(f"{_mod}.materialize_ff_projection",
                  return_value=np.zeros((8, 8), dtype=np.uint8)),
            patch(f"{_mod}.skio.imsave"),
            patch(f"{_mod}._get_stack",
                  return_value=np.ones((4, 8, 8), dtype=np.uint16)),
        ):
            df = materialize_yx1_well(
                experiment_id=EXP,
                well_id=WELL_ID,
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inv,
                built_image_data_dir=tmp_path,
                resolved_plan=IDENTITY_PLAN,
                device="cpu",
                smoke_max_time_indices=3,
            )
        assert len(df) == 3
