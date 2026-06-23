"""Smoke run: generate real snip output files for visual inspection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
import pandas as pd
import skimage.io as skio
from unittest.mock import patch

from data_pipeline.segmentation.backends.sam2_video.fake_predictor import FakePredictor, segment_one_well_fake
from data_pipeline.segmentation.sam2_video.model_loader import Sam2VideoModelConfig
from data_pipeline.segmentation.sam2_video.run_sam2_video import Sam2WellInput, run_sam2_video_for_wells
from data_pipeline.shared.identifiers import build_image_id, build_well_id
from data_pipeline.snip_processing.entrypoints.run_snip_processing import run_snip_processing

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
IMAGES = OUT / "images"
IMAGES.mkdir(exist_ok=True)

WELL_ID = build_well_id("20250912", "B01")
N_FRAMES = 5
N_OBJECTS = 3
W, H = 128, 128

np.random.seed(42)

# Build synthetic frame inventory with real PNG images
rows = []
for t in range(N_FRAMES):
    image_id = build_image_id(WELL_ID, "BF", t)
    img_path = IMAGES / f"{image_id}.png"
    # Synthetic brightfield-ish: dark background, a few bright blobs
    img = np.full((H, W), 40, dtype=np.uint8)
    for _ in range(N_OBJECTS):
        cy, cx = np.random.randint(20, H-20), np.random.randint(20, W-20)
        rr, cc = np.ogrid[:H, :W]
        r = np.random.randint(10, 20)
        mask = (rr - cy)**2 + (cc - cx)**2 < r**2
        img[mask] = np.random.randint(150, 240)
    skio.imsave(str(img_path), img, check_contrast=False)
    rows.append({
        "experiment_id": "20250912",
        "well_id": WELL_ID,
        "image_id": image_id,
        "time_index": t,
        "z_index": pd.NA,
        "channel_id": "BF",
        "source_image_path": str(img_path),
        "image_width_px": W,
        "image_height_px": H,
        "micrometers_per_pixel": 2.17,
    })

frame_inventory = pd.DataFrame(rows)

# Run fake SAM2 to get frame_masks
well = Sam2WellInput(well_id=WELL_ID, model_frame_view=frame_inventory, frame_detections=pd.DataFrame())
fake_pred = FakePredictor(n_objects=N_OBJECTS)
config = Sam2VideoModelConfig(
    models_root=Path("/fake"), config_path=Path("/fake/c.yaml"),
    checkpoint_path=Path("/fake/cp.pt"), device="cpu", model_id="fake:v1",
)
with patch("data_pipeline.segmentation.sam2_video.run_sam2_video.load_sam2_video_model", return_value=fake_pred):
    results = run_sam2_video_for_wells([well], model_config=config, segment_one_well=segment_one_well_fake)

frame_masks = results[0].frame_masks
print(f"frame_masks: {len(frame_masks)} rows, {frame_masks['is_valid_mask'].sum()} valid")

frame_masks.to_csv(OUT / "frame_masks.csv", index=False)
frame_inventory.to_csv(OUT / "frame_inventory.csv", index=False)

# Build the physical_embryo_registry shard (identity is JOINED, not minted, in snip_processing).
from data_pipeline.segmentation.physical_embryo_registry.build_physical_embryo_registry import (
    build_physical_embryo_registry,
)
build_physical_embryo_registry(frame_masks).to_csv(
    OUT / "physical_embryo_registry.csv", index=False
)

# Run snip processing
run_snip_processing(
    frame_masks_csv=OUT / "frame_masks.csv",
    frame_inventory_csv=OUT / "frame_inventory.csv",
    physical_embryo_registry_csv=OUT / "physical_embryo_registry.csv",
    output_csv=OUT / "snip_inventory.csv",
    snips_dir=OUT / "snips",
    output_root=OUT,
    target_pixel_size_um=2.17,
    output_height_px=64,
    output_width_px=64,
)

inv = pd.read_csv(OUT / "snip_inventory.csv")
print(f"snip_inventory: {len(inv)} rows")
print(f"  valid snips: {inv['is_valid_snip'].sum()}")
print(f"  failed: {(~inv['is_valid_snip'].astype(bool)).sum()}")
print()
print(inv[["snip_id", "physical_embryo_id", "time_index", "is_valid_snip", "processed_snip_path"]].to_string())
print()
print(f"Pixel files written to: {OUT / 'snips'}")
for f in sorted((OUT / "snips").rglob("*.png")):
    print(f"  {f.relative_to(OUT)}")
