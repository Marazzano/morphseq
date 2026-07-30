"""Tests for the Keyence materializer writer-policy handoff and provenance rows."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTileResult,
    TileTransform,
    TilingQC,
)
from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ResolvedImageProduct,
)
from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    validate_frame_inventory_identity_contract,
)
from data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence import (
    _EMITTED_COLUMNS,
    materialize_keyence_product_for_well,
)
from data_pipeline.shared.identifiers.constructors import build_well_id

EXPERIMENT_ID = "20250912"
WELL_INDEX = "B01"
WELL_ID = build_well_id(EXPERIMENT_ID, WELL_INDEX)
RESOLVED_PRODUCT = ResolvedImageProduct(
    channel_id="BF",
    image_product_type="projection",
    projection_method="focus_stack",
    xy_composition="mosaic",
)
Z_STACK_PRODUCT = ResolvedImageProduct(
    channel_id="BF",
    image_product_type="z_stack",
    projection_method=None,
    xy_composition="mosaic",
)


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _make_inventory(
    tmp_path: Path,
    *,
    n_tiles: int = 2,
    n_z: int = 2,
    time_index: int = 0,
    orientation: str = "horizontal",
    tile_shape: tuple[int, int] = (2, 2),
) -> pd.DataFrame:
    rows = []
    for tile_id in range(n_tiles):
        for z_index in range(n_z):
            raw_path = (
                tmp_path
                / "raw"
                / f"{WELL_ID}_tile{tile_id}_z{z_index}_t{time_index}.tif"
            )
            _write_tiff(
                raw_path,
                np.full(tile_shape, fill_value=tile_id * 10 + z_index, dtype=np.uint16),
            )
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "position_index": 13,
                    "channel_id": "BF",
                    "raw_channel_name": "CH1",
                    "time_index": time_index,
                    "elapsed_time_s": float(time_index * 60),
                    "micrometers_per_pixel": 0.5,
                    "image_width_px": tile_shape[1],
                    "image_height_px": tile_shape[0],
                    "microscope_id": "Keyence",
                    "well_index": WELL_INDEX,
                    "well_id": WELL_ID,
                    "tile_id": tile_id,
                    "position_index_within_well": tile_id,
                    "n_tiles_in_well": n_tiles,
                    "z_index": z_index,
                    "channel_index": 1,
                    "time_index_claimed": time_index,
                    "acquisition_time_s": float(time_index * 60),
                    "objective_magnification": "4x",
                    "orientation": orientation,
                    "source_tiff_path": str(raw_path),
                    "stage_x_nm": 0,
                    "stage_y_nm": 0,
                    "stage_z_nm": 0,
                }
            )
    return pd.DataFrame(rows)


def _mock_stitch_result(
    mosaic: np.ndarray,
    *,
    tile_width_px: int = 2,
) -> FrameTileResult:
    return FrameTileResult(
        stitched=mosaic,
        tile_transforms={
            "0": TileTransform(tile_id="0", dx_px=0.0, dy_px=0.0, source="align"),
            "1": TileTransform(
                tile_id="1",
                dx_px=float(tile_width_px),
                dy_px=0.0,
                source="align",
            ),
        },
        canvas_shape=mosaic.shape,
        qc=TilingQC(
            passed=True,
            reasons=tuple(),
            metrics={"tile_count": 2.0},
            suggested_action="ok",
        ),
        fallback_used="none",
    )


def _mock_focus_group(per_tile):
    """Build a FocusStackGroupResult-shaped mock from a list of (projection_u8, fim) pairs.

    Mirrors the shared primitive's public shape (``.tiles[i].projection_u8`` /
    ``.focus_index_map``) so the Keyence adapter can be tested without running the real LoG.
    """
    from types import SimpleNamespace

    tiles = tuple(
        SimpleNamespace(projection_u8=proj, focus_index_map=fim)
        for proj, fim in per_tile
    )
    return SimpleNamespace(
        tiles=tiles, intensity_lo=0, intensity_hi=65535, config=None
    )


def test_emits_final_image_metadata_and_oriented_focus_map(tmp_path):
    inventory_df = _make_inventory(tmp_path, orientation="horizontal")
    mosaic = np.array(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=np.uint8,
    )
    module = "data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence"
    with (
        patch(
            f"{module}.focus_stack_group",
            return_value=_mock_focus_group([
                (
                    np.array([[10, 20], [30, 40]], dtype=np.uint8),
                    np.zeros((2, 2), dtype=np.int32),
                ),
                (
                    np.array([[50, 60], [70, 80]], dtype=np.uint8),
                    np.ones((2, 2), dtype=np.int32),
                ),
            ]),
        ),
        patch(
            f"{module}.stitch_frame_tiles",
            return_value=_mock_stitch_result(mosaic, tile_width_px=2),
        ),
    ):
        df = materialize_keyence_product_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory_df,
            built_image_data_dir=tmp_path,
            resolved_product=RESOLVED_PRODUCT,
            device="cpu",
            candidate=True,
            config={
                "image_materialization": {
                    "write_policies": {
                        "BF__projection__focus_stack": {
                            "orientation": "vertical",
                        }
                    }
                }
            },
        )

    assert list(df.columns) == list(_EMITTED_COLUMNS)
    assert "source_image_path" not in df.columns
    row = df.iloc[0]
    assert row["image_width_px"] == 2
    assert row["image_height_px"] == 4
    assert row["image_micrometers_per_pixel"] == 0.5
    assert row["orientation"] == "vertical"
    assert row["image_file_format"] == "png"
    assert row["pixel_dtype"] == "uint8"

    with Image.open(row["image_path"]) as im:
        assert im.size == (2, 4)

    fim = np.load(row["index_map_path"])["focus_index_map"]
    np.testing.assert_array_equal(
        fim,
        np.array(
            [
                [1, 1],
                [1, 1],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_emits_raw_tile_manifest_and_written_downsampled_dimensions(tmp_path):
    inventory_df = _make_inventory(tmp_path, orientation="horizontal")
    mosaic = np.arange(16, dtype=np.uint16).reshape(4, 4)
    module = "data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence"
    with (
        patch(
            f"{module}.focus_stack_group",
            return_value=_mock_focus_group([
                (
                    np.array([[0, 1], [2, 3]], dtype=np.uint16),
                    np.zeros((2, 2), dtype=np.int32),
                ),
                (
                    np.array([[4, 5], [6, 7]], dtype=np.uint16),
                    np.ones((2, 2), dtype=np.int32),
                ),
            ]),
        ),
        patch(
            f"{module}.stitch_frame_tiles",
            return_value=_mock_stitch_result(mosaic, tile_width_px=2),
        ),
    ):
        df = materialize_keyence_product_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory_df,
            built_image_data_dir=tmp_path,
            resolved_product=RESOLVED_PRODUCT,
            device="cpu",
            candidate=False,
            config={
                "image_materialization": {
                    "write_policies": {
                        "BF__projection__focus_stack": {
                            "downsample_factor": 2,
                            "downsample_method": "block_mean",
                            "pixel_dtype": "uint8",
                            "file_format": "png",
                            "orientation": "none",
                        }
                    }
                }
            },
        )

    row = df.iloc[0]
    assert row["image_width_px"] == 2
    assert row["image_height_px"] == 2
    assert row["image_micrometers_per_pixel"] == 1.0
    assert pd.isna(row["raw_tile_path"])
    assert row["raw_tile_manifest_path"]
    assert row["raw_tile_width_px"] == 2
    assert row["raw_tile_height_px"] == 2
    assert row["raw_tile_count"] == 2
    assert row["raw_micrometers_per_pixel"] == 0.5
    assert row["downsample_factor"] == 2
    assert row["downsample_method"] == "block_mean"

    with Image.open(row["image_path"]) as im:
        assert im.size == (2, 2)

    manifest_path = Path(row["raw_tile_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["well_id"] == WELL_ID
    assert manifest["channel_id"] == "BF"
    assert len(manifest["tiles"]) == 2
    assert all(len(tile["source_tiff_paths"]) == 2 for tile in manifest["tiles"])


def test_z_stack_stitches_and_emits_one_inventory_row_per_plane(tmp_path):
    inventory_df = _make_inventory(tmp_path, n_tiles=2, n_z=2)
    mosaics = [
        np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.uint16),
        np.array([[1, 1, 11, 11], [1, 1, 11, 11]], dtype=np.uint16),
    ]
    module = "data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence"
    with patch(
        f"{module}.stitch_frame_tiles",
        side_effect=[_mock_stitch_result(mosaic) for mosaic in mosaics],
    ) as stitch:
        df = materialize_keyence_product_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory_df,
            built_image_data_dir=tmp_path,
            resolved_product=Z_STACK_PRODUCT,
            device="cpu",
            candidate=True,
            config={
                "image_materialization": {
                    "write_policies": {
                        "BF__z_stack": {
                            "downsample_factor": 1,
                            "pixel_dtype": "uint16",
                            "file_format": "tif",
                            "orientation": "none",
                            "jpeg_quality": None,
                        }
                    }
                }
            },
        )

    assert list(df.columns) == list(_EMITTED_COLUMNS)
    assert stitch.call_count == 2
    assert df["z_index"].tolist() == [0, 1]
    assert df["image_product_type"].tolist() == ["z_stack", "z_stack"]
    assert df["projection_method"].isna().all()
    assert df["index_map_path"].isna().all()
    assert df["image_id"].str.contains(r"_z000[01]_t0000$").all()
    assert df["raw_tile_count"].tolist() == [2, 2]
    # z_stack now carries the canonical display polarity like every other product (default True).
    assert df["flip_polarity"].tolist() == [True, True]
    validate_frame_inventory_identity_contract(df, scope_label="Keyence z-stack test")

    for z_index, row in df.set_index("z_index").iterrows():
        assert f"z{z_index:04d}" in Path(row["image_path"]).stem
        with Image.open(row["image_path"]) as image:
            # Written plane is the display-polarity-inverted mosaic (uint16: 65535 - v).
            np.testing.assert_array_equal(
                np.asarray(image), np.uint16(65535) - mosaics[z_index]
            )
        manifest = json.loads(Path(row["raw_tile_manifest_path"]).read_text())
        assert len(manifest["tiles"]) == 2
        assert all(tile["z_indices"] == [z_index] for tile in manifest["tiles"])
        assert all(len(tile["source_tiff_paths"]) == 1 for tile in manifest["tiles"])


def test_z_stack_rejects_missing_tile_before_writing_any_plane(tmp_path):
    time_zero = _make_inventory(tmp_path, n_tiles=2, n_z=2, time_index=0)
    time_one = _make_inventory(tmp_path, n_tiles=2, n_z=2, time_index=1)
    inventory_df = pd.concat(
        [time_zero, time_one[time_one["tile_id"] == 0]], ignore_index=True
    )

    with pytest.raises(ValueError, match="Incomplete Keyence z plane"):
        materialize_keyence_product_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory_df,
            built_image_data_dir=tmp_path,
            resolved_product=Z_STACK_PRODUCT,
            device="cpu",
            candidate=True,
        )

    assert not list((tmp_path / EXPERIMENT_ID).rglob("*z_stack*"))


def test_z_stack_uses_configured_default_jpeg_downsampling(tmp_path):
    inventory_df = _make_inventory(tmp_path, n_tiles=2, n_z=1, tile_shape=(8, 8))
    mosaic = np.arange(128, dtype=np.uint16).reshape(8, 16)
    module = "data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence"
    with patch(
        f"{module}.stitch_frame_tiles",
        return_value=_mock_stitch_result(mosaic, tile_width_px=8),
    ):
        df = materialize_keyence_product_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory_df,
            built_image_data_dir=tmp_path,
            resolved_product=Z_STACK_PRODUCT,
            device="cpu",
            candidate=True,
        )

    row = df.iloc[0]
    assert row["image_file_format"] == "jpg"
    assert row["jpeg_quality"] == 85
    assert row["downsample_factor"] == 4
    assert row["downsample_method"] == "area_resize"
    assert row["pixel_dtype"] == "uint8"
    assert row["image_height_px"] == 2
    assert row["image_width_px"] == 4
    assert row["image_micrometers_per_pixel"] == 2.0
