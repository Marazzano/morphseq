"""Session C GPU smoke: run real SAM2 video predictor on 20250912_B01 (3 frames).

Run with:
    PYTHONPATH=src /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/grounded_sam2/bin/python \
        scripts/smoke_sam2_real_gpu.py

Converts grayscale PNGs to RGB JPEGs in a temp dir, seeds synthetic prompt boxes,
runs the full predictor.init_state → add_new_points_or_box → propagate_in_video
pattern (matching the legacy propagation.py pattern), then adapts output through
adapt_sam2_well_output and validates with validate_frame_masks.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline.models.sam2 import load_sam2_video_predictor
from data_pipeline.segmentation.backends.sam2_video.adapt_sam2_output import (
    adapt_sam2_well_output,
)
from data_pipeline.segmentation.backends.sam2_video.prompt_detections import (
    validate_frame_masks_against_sam2_prompts,
    validate_sam2_prompts,
)
from data_pipeline.segmentation.validate_frame_masks import validate_frame_masks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Use the same symlink the pipeline uses — load_sam2_video_predictor is sensitive
# to how the models root is specified (it resolves relative config/checkpoint paths
# against sam2_pkg_dir and sam2_models_root respectively, then chdirs into sam2_pkg_dir).
SAM2_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output/models/sam2")
CONFIG_PATH = Path("configs/sam2.1/sam2.1_hiera_s.yaml")   # relative — resolved by loader
CHECKPOINT_PATH = Path("checkpoints/sam2.1_hiera_small.pt") # relative — resolved by loader
FRAME_INVENTORY_CSV = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output"
    "/acquisition/20250912/frame_inventory/per_well/20250912_B01/20250912_B01_frame_inventory.csv"
)

# Synthetic prompt boxes seeded on frame t=0 (SAM2 propagates forward from there).
# Three boxes across the 2189x2189 brightfield frame.
PROMPT_BOXES = [
    (300,  300,  900,  900),
    (1100, 300,  1700, 900),
    (700,  1100, 1400, 1800),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grayscale_png_to_rgb_jpeg(src: Path, dst: Path) -> None:
    """SAM2 expects RGB images; convert grayscale PNGs."""
    img = Image.open(src).convert("RGB")
    img.save(dst, format="JPEG", quality=95)


def _build_prompt_detections(frame_inventory: pd.DataFrame) -> pd.DataFrame:
    seed_frame = frame_inventory[frame_inventory["time_index"] == 0].iloc[0]
    image_id = str(seed_frame["image_id"])
    rows = []
    for i, (x0, y0, x1, y1) in enumerate(PROMPT_BOXES):
        rows.append({
            "prompt_detection_id": f"{image_id}_synthetic{i:04d}",
            "image_id": image_id,
            "time_index": int(seed_frame["time_index"]),
            "bbox_x_min_px": float(x0),
            "bbox_y_min_px": float(y0),
            "bbox_x_max_px": float(x1),
            "bbox_y_max_px": float(y1),
            "is_kept": True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Session C GPU smoke: real SAM2 on 20250912_B01 ===\n")

    frame_inventory = pd.read_csv(FRAME_INVENTORY_CSV)
    n_frames = len(frame_inventory)
    print(f"Frame inventory: {n_frames} frames")
    print(f"  image_ids: {frame_inventory['image_id'].tolist()}")
    w = frame_inventory['image_width_px'].iloc[0]
    h = frame_inventory['image_height_px'].iloc[0]
    print(f"  image size: {w} x {h}\n")

    prompt_detections = _build_prompt_detections(frame_inventory)
    print(f"Prompt detections: {len(prompt_detections)} boxes seeded on frame t=0")
    validate_sam2_prompts(prompt_detections, frame_inventory)
    print("  validate_sam2_prompts: PASS\n")

    print("Loading SAM2 model (grounded_sam2 env, hiera_s, cuda)...")
    predictor = load_sam2_video_predictor(
        sam2_models_root=SAM2_ROOT,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
    )
    print("  model loaded\n")

    with tempfile.TemporaryDirectory(prefix="sam2_rgb_frames_") as tmpdir:
        rgb_dir = Path(tmpdir)
        print("Converting grayscale PNGs → RGB JPEGs...")
        ordered_inv = frame_inventory.sort_values("time_index").reset_index(drop=True)
        for _, row in ordered_inv.iterrows():
            src = Path(str(row["source_image_path"]))
            dst = rgb_dir / f"{int(row['time_index']):05d}.jpg"
            _grayscale_png_to_rgb_jpeg(src, dst)
            print(f"  {src.name} → {dst.name}")

        # Build model_frame_view with sam2_frame_index = position in sorted order
        model_frame_view = ordered_inv.copy()
        model_frame_view["sam2_frame_index"] = model_frame_view.index

        well_id = str(frame_inventory["well_id"].iloc[0])

        print(f"\nInitialising SAM2 inference state over {rgb_dir}...")
        inference_state = predictor.init_state(video_path=str(rgb_dir))

        # Seed from frame 0 (t=0 maps to sam2_frame_index=0)
        seed_sam2_idx = 0
        print(f"Adding {len(prompt_detections)} box prompts on sam2_frame_index={seed_sam2_idx}...")
        for i, (_, row) in enumerate(prompt_detections.iterrows()):
            box = np.array([
                row["bbox_x_min_px"], row["bbox_y_min_px"],
                row["bbox_x_max_px"], row["bbox_y_max_px"],
            ], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=seed_sam2_idx,
                obj_id=i,
                box=box,
            )
            print(f"  obj_id={i}  box={box.tolist()}")

        print("\nPropagating masks forward...")
        sam2_raw_output: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks_for_frame: dict[int, np.ndarray] = {}
            for obj_id, logit in zip(obj_ids, mask_logits):
                arr = logit.squeeze().cpu().numpy() if hasattr(logit, "cpu") else np.squeeze(np.asarray(logit))
                mask = (arr > 0).astype(bool)
                masks_for_frame[int(obj_id)] = mask
            sam2_raw_output[int(frame_idx)] = masks_for_frame
            areas = [m.sum() for m in masks_for_frame.values()]
            print(f"  frame {frame_idx}: {len(masks_for_frame)} objects, areas={areas}")

    print(f"\nAdapter: converting SAM2 output → frame_masks...")
    frame_masks = adapt_sam2_well_output(
        well_id,
        sam2_raw_output,
        model_frame_view,
        prompt_detections,
        model_id="sam2.1_hiera_s",
    )
    valid_count = frame_masks["is_valid_mask"].astype(bool).sum()
    no_mask_count = (~frame_masks["is_valid_mask"].astype(bool)).sum()
    print(f"  frame_masks rows: {len(frame_masks)} ({valid_count} valid, {no_mask_count} no-mask)")

    print("\nValidating...")
    validate_frame_masks(frame_masks, frame_inventory)
    print("  validate_frame_masks: PASS")
    validate_frame_masks_against_sam2_prompts(frame_masks, prompt_detections)
    print("  validate_frame_masks_against_sam2_prompts: PASS")

    print("\nSample output:")
    print(frame_masks[["image_id", "mask_id", "track_id", "area_px", "is_valid_mask"]].to_string())

    print("\n=== Session C GPU smoke: PASSED ===")


if __name__ == "__main__":
    main()
