"""Tests for the Keyence materializer writer-policy handoff and provenance rows."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTileResult,
    TileTransform,
    TilingQC,
)
from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ResolvedImageProduct,
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
        patch(f"{module}.im_rescale", side_effect=lambda stack: (stack.astype(np.float32), None, None)),
        patch(
            f"{module}.materialize_ff_projection",
            side_effect=[
                (
                    np.array([[10, 20], [30, 40]], dtype=np.uint8),
                    np.zeros((2, 2), dtype=np.int32),
                ),
                (
                    np.array([[50, 60], [70, 80]], dtype=np.uint8),
                    np.ones((2, 2), dtype=np.int32),
                ),
            ],
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

    fim = np.load(row["focus_index_map_path"])["focus_index_map"]
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
        patch(f"{module}.im_rescale", side_effect=lambda stack: (stack.astype(np.float32), None, None)),
        patch(
            f"{module}.materialize_ff_projection",
            side_effect=[
                (
                    np.array([[0, 1], [2, 3]], dtype=np.uint16),
                    np.zeros((2, 2), dtype=np.int32),
                ),
                (
                    np.array([[4, 5], [6, 7]], dtype=np.uint16),
                    np.ones((2, 2), dtype=np.int32),
                ),
            ],
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
