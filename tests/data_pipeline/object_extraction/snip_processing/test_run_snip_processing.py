"""End-to-end test for run_snip_processing.

Uses FakePredictor to produce real frame_masks output, writes synthetic
grayscale images to a temp directory, and runs run_snip_processing end-to-end.
No GPU or real SAM2 is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.backends.sam2_video.fake_predictor import (
    FakePredictor,
    segment_one_well_fake,
)
from data_pipeline.object_extraction.segmentation.sam2_video.model_loader import Sam2VideoModelConfig
from data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video import (
    Sam2WellInput,
    run_sam2_video_for_wells,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.build_physical_embryo_registry import (
    build_physical_embryo_registry,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_INVENTORY_COLUMNS,
    validate_snip_inventory_contract,
)
from data_pipeline.shared.identifiers import build_image_id, build_well_id
from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_processing import run_snip_processing
from data_pipeline.object_extraction.snip_processing.augmentation import generate_background_noise


WELL_ID = build_well_id("20250912", "B01")
N_FRAMES = 3
N_OBJECTS = 2
IMG_W, IMG_H = 64, 64


def test_generate_background_noise_accepts_zero_variance():
    noise = generate_background_noise((4, 5), background_mean=0.0, background_std=0.0)
    assert noise.shape == (4, 5)
    assert np.all(noise == 0.0)


def _make_frame_inventory(tmp_images: Path) -> pd.DataFrame:
    rows = []
    for t in range(N_FRAMES):
        image_id = build_image_id(WELL_ID, "BF", t)
        img_path = tmp_images / f"{image_id}.png"
        img = np.random.randint(50, 200, (IMG_H, IMG_W), dtype=np.uint8)
        skio.imsave(str(img_path), img, check_contrast=False)
        rows.append({
            "experiment_id": "20250912",
            "well_id": WELL_ID,
            "image_id": image_id,
            "time_index": t,
            "z_index": pd.NA,
            "channel_id": "BF",
            "image_path": str(img_path),
            "image_width_px": IMG_W,
            "image_height_px": IMG_H,
            "image_micrometers_per_pixel": 2.17,
        })
    return pd.DataFrame(rows)


def _make_frame_masks(frame_inventory: pd.DataFrame) -> pd.DataFrame:
    well = Sam2WellInput(
        well_id=WELL_ID,
        model_frame_view=frame_inventory,
        frame_detections=pd.DataFrame(),
    )
    fake_pred = FakePredictor(n_objects=N_OBJECTS)
    config = Sam2VideoModelConfig(
        models_root=Path("/fake/models"),
        config_path=Path("/fake/config.yaml"),
        checkpoint_path=Path("/fake/checkpoint.pt"),
        device="cpu",
        model_id="fake_predictor:v1",
    )
    with patch(
        "data_pipeline.object_extraction.segmentation.sam2_video.run_sam2_video.load_sam2_video_model",
        return_value=fake_pred,
    ):
        results = run_sam2_video_for_wells(
            [well], model_config=config, segment_one_well=segment_one_well_fake,
        )
    return results[0].frame_masks


def _write_inputs(tmp_path, *, include_registry=True):
    """Write frame_inventory, frame_masks, and (optionally) the registry shard to tmp_path.

    Returns (frame_masks, frame_masks_csv, frame_inventory_csv, registry_csv). The registry is
    built from the SAME frame_masks via the real Stage-2 builder — exactly how the pipeline does
    it — so the join reproduces the identity the old mint chain produced.
    """
    tmp_images = tmp_path / "images"
    tmp_images.mkdir()

    frame_inventory = _make_frame_inventory(tmp_images)
    frame_masks = _make_frame_masks(frame_inventory)

    frame_masks_csv = tmp_path / "frame_masks.csv"
    frame_inventory_csv = tmp_path / "frame_inventory.csv"
    registry_csv = tmp_path / "physical_embryo_registry.csv"
    frame_masks.to_csv(frame_masks_csv, index=False)
    frame_inventory.to_csv(frame_inventory_csv, index=False)
    if include_registry:
        build_physical_embryo_registry(frame_masks).to_csv(registry_csv, index=False)

    return frame_masks, frame_masks_csv, frame_inventory_csv, registry_csv


def test_run_snip_processing_produces_inventory(tmp_path):
    frame_masks, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    output_csv = tmp_path / "snip_inventory.csv"
    snips_dir = tmp_path / "snips"

    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        output_csv=output_csv,
        snips_dir=snips_dir,
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    assert output_csv.exists(), "snip_inventory.csv was not written"
    df = pd.read_csv(output_csv)

    # One row per valid mask row.
    n_valid = int(frame_masks["is_valid_mask"].astype(bool).sum())
    assert len(df) == n_valid, f"expected {n_valid} rows, got {len(df)}"

    # Required columns present.
    required = [
        "snip_id", "embryo_id", "physical_embryo_id", "experiment_id", "well_id",
        "image_id", "time_index", "channel_id", "mask_id", "track_id",
        "image_path", "processed_snip_path", "is_valid_snip", "error_message",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"missing columns: {missing}"

    # All snips should be valid.
    failures = df[~df["is_valid_snip"].astype(bool)]
    assert failures.empty, f"some snips failed:\n{failures[['snip_id', 'error_message']]}"

    # snip_id is unique.
    assert not df["snip_id"].duplicated().any(), "duplicate snip_ids"

    # Pixel files exist on disk.
    for _, row in df.iterrows():
        png = tmp_path / str(row["processed_snip_path"])
        assert png.exists(), f"pixel file missing: {png}"


def test_run_snip_processing_joins_registry_physical_embryo_id(tmp_path):
    """The joined physical_embryo_id equals what the registry resolves for each (well, track)."""
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    output_csv = tmp_path / "snip_inventory.csv"

    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        output_csv=output_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    snips = pd.read_csv(output_csv)
    registry = pd.read_csv(registry_csv)
    expected = {
        (str(r["well_id"]), str(r["track_id"])): str(r["physical_embryo_id"])
        for _, r in registry.iterrows()
    }
    for _, row in snips.iterrows():
        key = (str(row["well_id"]), str(row["track_id"]))
        assert str(row["physical_embryo_id"]) == expected[key], (
            f"snip physical_embryo_id for {key} did not match the registry"
        )


def test_run_snip_processing_writes_valid_headered_inventory_for_empty_well(tmp_path):
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)
    frame_masks = pd.read_csv(frame_masks_csv).iloc[0:0]
    frame_masks.to_csv(frame_masks_csv, index=False)
    output_csv = tmp_path / "snip_inventory.csv"

    run_snip_processing(
        frame_masks_csv=frame_masks_csv,
        frame_inventory_csv=frame_inventory_csv,
        physical_embryo_registry_csv=registry_csv,
        output_csv=output_csv,
        snips_dir=tmp_path / "snips",
        output_root=tmp_path,
        target_pixel_size_um=2.17,
        output_height_px=64,
        output_width_px=64,
    )

    result = pd.read_csv(output_csv)
    assert result.empty
    assert tuple(result.columns) == SNIP_INVENTORY_COLUMNS
    validate_snip_inventory_contract(result)


def test_run_snip_processing_fails_loud_on_missing_registry_match(tmp_path):
    """A valid mask whose track is absent from the registry raises (contract violation)."""
    _, frame_masks_csv, frame_inventory_csv, registry_csv = _write_inputs(tmp_path)

    # Drop every registry row -> no track resolves -> the first valid mask must fail loud.
    pd.read_csv(registry_csv).iloc[0:0].to_csv(registry_csv, index=False)

    with pytest.raises(ValueError, match="No physical_embryo_registry entry"):
        run_snip_processing(
            frame_masks_csv=frame_masks_csv,
            frame_inventory_csv=frame_inventory_csv,
            physical_embryo_registry_csv=registry_csv,
            output_csv=tmp_path / "snip_inventory.csv",
            snips_dir=tmp_path / "snips",
            output_root=tmp_path,
            target_pixel_size_um=2.17,
            output_height_px=64,
            output_width_px=64,
        )
