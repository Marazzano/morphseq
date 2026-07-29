"""Tests for materialize_well_yx1 — mocked ND2 + image ops, no GPU required."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_pipeline.acquisition.image_materialization.scope.yx1.materialize_well_yx1 import (
    materialize_ff_projection,
    materialize_yx1_well,
    _EMITTED_COLUMNS,
)
from data_pipeline.acquisition.image_materialization.materialized_image_paths import (
    projection_frame_path,
    z_stack_frame_path,
)
from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ResolvedImageProduct,
    ResolvedMaterializationPlan,
)
from data_pipeline.shared.identifiers.constructors import build_well_id

EXP = "20250912"
WELL_INDEX = "B01"
WELL_ID = build_well_id(EXP, WELL_INDEX)  # never mint ids by hand, even in tests

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
Z_STACK_PLAN = ResolvedMaterializationPlan(
    products=(
        ResolvedImageProduct(
            channel_id="BF",
            image_product_type="z_stack",
            projection_method=None,
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
        "channel_id": "BF",
        "raw_channel_name": "EYES - Dia",
        "time_index": list(range(n_times)),
        "acquisition_time_s": [100.0 * t for t in range(n_times)],
        # elapsed_time_s is rebased per position to its first frame (t=0 → 0.0); single position here.
        "elapsed_time_s": [100.0 * t for t in range(n_times)],
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


def _make_z_inventory(
    n_times: int = 2,
    z_indices: tuple[int, ...] = (0, 2),
    position_index: int = 2,
    source_nd2_path: str = "/fake/exp.nd2",
) -> pd.DataFrame:
    rows = []
    for t in range(n_times):
        for z in z_indices:
            rows.append({
                "experiment_id": EXP,
                "raw_position_label": str(position_index),
                "position_index": position_index,
                "z_index": z,
                "channel_index": 0,
                "channel_id": "BF",
                "raw_channel_name": "EYES - Dia",
                "time_index": t,
                "acquisition_time_s": 100.0 * t,
                "elapsed_time_s": 100.0 * t,
                "x_um": 10.0,
                "y_um": 20.0,
                "micrometers_per_pixel": 0.65,
                "image_width_px": 512,
                "image_height_px": 512,
                "objective_magnification": "4x",
                "microscope_id": "YX1",
                "n_z": 4,
                "source_nd2_path": source_nd2_path,
                "well_index": WELL_INDEX,
            })
    return pd.DataFrame(rows)


class TestMaterializeFFProjection:
    def test_returns_2d_uint8_and_focus_index_map(self):
        # materialize_ff_projection now delegates to the shared focus_stack_group; run it for
        # real (small stack, fast) and assert the output shape/dtype contract.
        stack = np.random.randint(0, 1000, size=(5, 64, 64), dtype=np.uint16)
        projection_u8, focus_index_map = materialize_ff_projection(stack, device="cpu")
        assert projection_u8.ndim == 2
        assert projection_u8.dtype == np.uint8
        assert focus_index_map.shape == (64, 64)
        assert focus_index_map.dtype == np.int32

    def test_focus_index_map_indexes_the_Z_axis(self):
        """The treason-cube guard: argmax must run over the Z axis (axis 0 of (Z, Y, X)).

        Build a real (no-mock) stack where ONE known Z plane (z=1) is in sharpest focus at a known
        region, and assert focus_index_map selects that stack-axis offset there.
        """
        # A small high-contrast feature is sharp ONLY on z=1; z=0 and z=2 are flat/blurred.
        z0 = np.full((16, 16), 100, dtype=np.uint16)
        z2 = np.full((16, 16), 100, dtype=np.uint16)
        z1 = np.full((16, 16), 100, dtype=np.uint16)
        # A sharp edge/checkerboard region drives a large LoG response on z=1.
        z1[6:10, 6:10] = 4000
        z1[7, 7] = 100
        z1[8, 8] = 100
        stack = np.stack([z0, z1, z2], axis=0)  # shape (Z=3, Y=16, X=16)

        projection_u8, focus_index_map = materialize_ff_projection(stack, device="cpu")

        z_indices = np.array([0, 1, 2], dtype=np.int32)  # the would-be acquisition labels
        assert focus_index_map.min() >= 0
        assert focus_index_map.max() < len(z_indices)
        # The sharp feature on z=1 must make the map select stack-axis offset 1 in that region.
        assert int(focus_index_map[7, 7]) == 1
        assert int(focus_index_map[6, 6]) == 1


class TestMaterializeYX1Well:
    def _make_mock_nd2(self, n_t: int = 3, n_w: int = 5, n_z: int = 4,
                       h: int = 8, w: int = 8):
        """Return a mock nd2.ND2File declaring a (T, P, Z, Y, X) layout.

        The stub must declare `sizes` and `experiment` because axis handling is now BY NAME (see
        metadata_ingest/scope/yx1/nd2_axes.py) — a mock with no named axes is rejected, which is the
        point: positional guessing is what silently mis-read real files.
        """
        dask_arr = MagicMock()
        dask_arr.ndim = 5
        fake_stack = np.ones((n_z, h, w), dtype=np.uint16)
        dask_arr.__getitem__ = MagicMock(return_value=MagicMock(compute=lambda: fake_stack))

        channel_mock = MagicMock()
        channel_mock.channel.name = "BF"

        nd_mock = MagicMock()
        nd_mock.to_dask.return_value = dask_arr
        nd_mock.frame_metadata.return_value.channels = [channel_mock]
        # Named axes, in array order, and the matching sequence loops. No C axis: this stub's array
        # is single-channel, so frameCount == T * P * Z.
        nd_mock.sizes = {"T": n_t, "P": n_w, "Z": n_z, "Y": h, "X": w}
        nd_mock.experiment = [
            type("TimeLoop", (), {})(),
            type("XYPosLoop", (), {})(),
            type("ZStackLoop", (), {})(),
        ]
        nd_mock.metadata.contents.frameCount = n_t * n_w * n_z
        return nd_mock

    def _run(
        self,
        inventory_df,
        tmp_path,
        *,
        candidate=True,
        resolved_plan=IDENTITY_PLAN,
        projection_frame=None,
        focus_index_map=None,
        stack=None,
        config=None,
    ):
        # The consume-side contract check asserts source_nd2_path EXISTS before the open is faked,
        # so point the shard at a real (empty) file under tmp_path.
        nd2_file = tmp_path / "exp.nd2"
        nd2_file.write_bytes(b"")
        inventory_df = inventory_df.copy()
        inventory_df["source_nd2_path"] = str(nd2_file)
        stack = np.ones((4, 8, 8), dtype=np.uint16) if stack is None else stack
        projection_frame = (
            np.zeros(stack.shape[1:], dtype=np.uint8)
            if projection_frame is None else projection_frame
        )
        focus_index_map = (
            np.zeros(stack.shape[1:], dtype=np.int32)
            if focus_index_map is None else focus_index_map
        )
        nd_mock = self._make_mock_nd2(
            n_t=int(inventory_df["time_index"].nunique()),
            n_z=stack.shape[0],
            h=stack.shape[1],
            w=stack.shape[2],
        )

        _mod = "data_pipeline.acquisition.image_materialization.scope.yx1.materialize_well_yx1"
        with (
            patch(f"{_mod}.nd2.ND2File", return_value=nd_mock),
            patch(f"{_mod}.materialize_ff_projection",
                  return_value=(projection_frame, focus_index_map)) as mock_proj,
            patch(f"{_mod}._get_stack",
                  return_value=stack) as mock_get_stack,
        ):
            df = materialize_yx1_well(
                experiment_id=EXP,
                well_id=WELL_ID,
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inventory_df,
                built_image_data_dir=tmp_path,
                resolved_plan=resolved_plan,
                device="cpu",
                candidate=candidate,
                config=config,
            )
        return df, mock_proj, mock_get_stack

    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        inv = _make_inventory(n_times=3)
        df, _, _ = self._run(inv, tmp_path)
        for col in _EMITTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        for removed in (
            "source_image_path",
            "source_micrometers_per_pixel",
            "source_image_width_px",
            "source_image_height_px",
        ):
            assert removed not in df.columns

    def test_one_row_per_time_index(self, tmp_path):
        n = 4
        inv = _make_inventory(n_times=n)
        df, _, _ = self._run(inv, tmp_path)
        assert len(df) == n

    def test_no_duplicate_rows_on_key(self, tmp_path):
        inv = _make_inventory(n_times=5)
        df, _, _ = self._run(inv, tmp_path)
        key = ["experiment_id", "well_index", "channel_id", "time_index", "z_index"]
        assert not df.duplicated(subset=key).any()

    def test_projection_image_path_matches_layout(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df, _, _ = self._run(inv, tmp_path, candidate=True)
        for _, row in df.iterrows():
            expected = projection_frame_path(
                tmp_path,
                experiment_id=row["experiment_id"],
                well_id=WELL_ID,
                channel_id=row["channel_id"],
                time_index=row["time_index"],
                candidate=True,
            )
            assert row["image_path"] == str(expected)

    def test_z_index_is_na_for_projection(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df, _, _ = self._run(inv, tmp_path)
        assert df["z_index"].isna().all()

    def test_projection_rows_record_materialized_and_raw_image_facts(self, tmp_path):
        inv = _make_inventory(n_times=2)
        projection = np.arange(7 * 11, dtype=np.uint8).reshape(7, 11)
        focus_map = np.zeros((7, 11), dtype=np.int32)
        stack = np.ones((4, 7, 11), dtype=np.uint16)

        df, _, _ = self._run(
            inv,
            tmp_path,
            candidate=True,
            projection_frame=projection,
            focus_index_map=focus_map,
            stack=stack,
        )

        assert (df["image_width_px"] == 11).all()
        assert (df["image_height_px"] == 7).all()
        assert np.allclose(df["image_micrometers_per_pixel"], 0.65)
        assert (df["image_file_format"] == "png").all()
        assert (df["pixel_dtype"] == "uint8").all()
        assert (df["downsample_factor"] == 1).all()
        assert (df["downsample_method"] == "none").all()
        assert df["jpeg_quality"].isna().all()
        assert (df["raw_image_source_path"] == str(tmp_path / "exp.nd2")).all()
        assert (df["raw_image_width_px"] == 512).all()
        assert (df["raw_image_height_px"] == 512).all()
        assert np.allclose(df["raw_micrometers_per_pixel"], 0.65)

        for image_path in df["image_path"]:
            with Image.open(image_path) as written:
                assert written.size == (11, 7)

    def test_projection_rows_carry_focus_index_map_path_1to1(self, tmp_path):
        from data_pipeline.acquisition.image_materialization.materialized_image_paths import (
            focus_index_map_path,
        )
        inv = _make_inventory(n_times=2)
        df, _, _ = self._run(inv, tmp_path, candidate=True)
        # every projection row carries a populated focus_index_map_path, 1:1 with its image_id
        assert df["focus_index_map_path"].notna().all()
        for _, row in df.iterrows():
            expected = focus_index_map_path(
                tmp_path, experiment_id=row["experiment_id"], well_id=WELL_ID,
                channel_id=row["channel_id"], time_index=row["time_index"], candidate=True,
            )
            assert row["focus_index_map_path"] == str(expected)
            # the .npz was actually written (np.savez is NOT mocked in _run) and is loadable
            data = np.load(expected)
            assert "focus_index_map" in data
            assert "z_indices" in data

    def test_focus_index_map_npz_has_offsets_bounded_by_z_indices(self, tmp_path):
        inv = _make_inventory(n_times=1)
        df, _, _ = self._run(inv, tmp_path, candidate=True)
        npz_path = df.iloc[0]["focus_index_map_path"]
        data = np.load(npz_path)
        fim = data["focus_index_map"]
        z_indices = data["z_indices"]
        assert fim.min() >= 0
        assert fim.max() < len(z_indices)

    def test_z_stack_emits_inventory_planes_not_array_shape(self, tmp_path):
        nd2_file = tmp_path / "exp.nd2"
        nd2_file.write_bytes(b"")
        inv = _make_z_inventory(n_times=2, z_indices=(0, 2), source_nd2_path=str(nd2_file))
        stack = np.arange(4 * 9 * 7, dtype=np.uint16).reshape(4, 9, 7)
        # Pin the factor instead of inheriting the product's 6.5 µm/px target: this fixture's
        # calibration (0.65 µm/px) would resolve to factor 10 and collapse the 9x7 synthetic stack
        # to a degenerate 1x1, destroying the dimension assertions below. This test is about row
        # enumeration coming from the inventory's z planes rather than the array shape; the write
        # policy itself is covered in test_materialized_image_write_policy.py.
        df, mock_proj, mock_get_stack = self._run(
            inv,
            tmp_path,
            candidate=True,
            resolved_plan=Z_STACK_PLAN,
            stack=stack,
            config={
                "image_materialization": {
                    "write_policies": {"BF__z_stack": {"downsample_factor": 4}}
                }
            },
        )

        assert len(df) == 4
        assert sorted(df["z_index"].unique().tolist()) == [0, 2]
        assert (df["image_product_type"] == "z_stack").all()
        assert df["projection_method"].isna().all()
        # z_stack rows carry NO focus-stack provenance.
        assert df["focus_index_map_path"].isna().all()
        assert set(df["image_id"]) == {
            f"{WELL_ID}_BF_z0000_t0000",
            f"{WELL_ID}_BF_z0002_t0000",
            f"{WELL_ID}_BF_z0000_t0001",
            f"{WELL_ID}_BF_z0002_t0001",
        }
        expected_path = z_stack_frame_path(
            tmp_path,
            experiment_id=EXP,
            well_id=WELL_ID,
            channel_id="BF",
            time_index=1,
            z_index=2,
            ext="jpg",
            candidate=True,
        )
        assert str(expected_path) in set(df["image_path"])
        assert (df["image_file_format"] == "jpg").all()
        assert (df["pixel_dtype"] == "uint8").all()
        assert (df["downsample_factor"] == 4).all()
        assert (df["downsample_method"] == "area_resize").all()
        assert (df["jpeg_quality"] == 85).all()
        assert np.allclose(df["image_micrometers_per_pixel"], 2.6)
        assert (df["raw_image_source_path"] == str(nd2_file)).all()
        assert (df["raw_image_width_px"] == 512).all()
        assert (df["raw_image_height_px"] == 512).all()
        assert np.allclose(df["raw_micrometers_per_pixel"], 0.65)
        assert (df["image_width_px"] == 2).all()
        assert (df["image_height_px"] == 2).all()
        for image_path in df["image_path"]:
            with Image.open(image_path) as written:
                assert written.size == (2, 2)
        assert mock_get_stack.call_count == 2
        mock_proj.assert_not_called()

    def test_product_grain_helper_accepts_one_resolved_product(self, tmp_path):
        inv = _make_inventory(n_times=1)
        df, _, _ = self._run(inv, tmp_path)
        assert len(df) == 1

    def test_materialize_yx1_well_rejects_multi_product_plan(self, tmp_path):
        inv = _make_inventory(n_times=1)
        multi = ResolvedMaterializationPlan(products=IDENTITY_PLAN.products + Z_STACK_PLAN.products)
        with pytest.raises(ValueError, match="exactly one resolved product"):
            materialize_yx1_well(
                experiment_id=EXP,
                well_id=WELL_ID,
                well_index=WELL_INDEX,
                well_acquisition_inventory_df=inv,
                built_image_data_dir=tmp_path,
                resolved_plan=multi,
                device="cpu",
            )

    def test_image_product_type_is_projection(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df, _, _ = self._run(inv, tmp_path)
        assert (df["image_product_type"] == "projection").all()

    def test_projection_method_is_focus_stack(self, tmp_path):
        inv = _make_inventory(n_times=2)
        df, _, _ = self._run(inv, tmp_path)
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
        _mod = "data_pipeline.acquisition.image_materialization.scope.yx1.materialize_well_yx1"
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
        _mod = "data_pipeline.acquisition.image_materialization.scope.yx1.materialize_well_yx1"
        with (
            patch(f"{_mod}.nd2.ND2File", return_value=nd_mock),
            patch(f"{_mod}.materialize_ff_projection",
                  return_value=(np.zeros((8, 8), dtype=np.uint8),
                                np.zeros((8, 8), dtype=np.int32))),
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
