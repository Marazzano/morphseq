#!/usr/bin/env python
"""Smoke-test: run GroundingDINO frame detection on one well's real frame_inventory.

Throwaway driver — not wired into Snakemake. Proves the adapter → validator seam
against real model output for the first time.

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output \\
        env PYTHONPATH=src:$PYTHONPATH \\
        python smoke_frame_detection.py [--well WELL_ID] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

GDINO_REPO = Path("/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/GroundingDINO")
GDINO_CONFIG = GDINO_REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = GDINO_REPO / "weights/groundingdino_swint_ogc.pth"

DEFAULT_INVENTORY = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs"
    "/data_pipeline_output/experiment_metadata/20250912/20250912_frame_inventory.csv"
)
DEFAULT_WELL = "20250912_B01"
DEFAULT_OUT = Path("/tmp/frame_detections_smoke.csv")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    p.add_argument("--well", default=DEFAULT_WELL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    return p.parse_args()


def _print_summary(df: pd.DataFrame, well_inventory: pd.DataFrame) -> None:
    total = len(df)
    kept = df["is_kept"].sum()
    none_rows = df["detection_id"].astype(str).str.endswith("_det_none").sum()
    rejected = total - kept - none_rows

    print(f"\n=== Frame detection smoke summary ===")
    print(f"  Frames in inventory : {len(well_inventory)}")
    print(f"  Total output rows   : {total}")
    print(f"  is_kept=True        : {kept}")
    print(f"  rejected (is_kept=F): {rejected}")
    print(f"  _det_none placehld  : {none_rows}")

    kept_df = df[df["is_kept"].astype(bool)]
    if len(kept_df):
        conf = kept_df["confidence"].dropna()
        print(f"  confidence range    : [{conf.min():.4f}, {conf.max():.4f}]")
        # Bbox bounds check against image dimensions
        w = well_inventory["image_width_px"].iloc[0]
        h = well_inventory["image_height_px"].iloc[0]
        oob = kept_df[
            (kept_df["bbox_x_min_px"] < 0)
            | (kept_df["bbox_x_max_px"] > w)
            | (kept_df["bbox_y_min_px"] < 0)
            | (kept_df["bbox_y_max_px"] > h)
        ]
        print(f"  bbox within {w}x{h}: {'YES' if len(oob) == 0 else f'NO — {len(oob)} OOB rows'}")
    else:
        print("  confidence range    : N/A (no kept detections)")

    print()
    print(df[["image_id", "detection_id", "is_kept", "confidence",
              "bbox_x_min_px", "bbox_y_min_px", "bbox_x_max_px", "bbox_y_max_px"]].to_string())


def main() -> None:
    args = _parse_args()

    # ── 1. Load and filter inventory ──────────────────────────────────────────
    if not args.inventory.exists():
        sys.exit(f"ERROR: inventory not found: {args.inventory}")

    full_inv = pd.read_csv(args.inventory)
    well_inv = full_inv[full_inv["well_id"] == args.well].copy()
    if well_inv.empty:
        available = sorted(full_inv["well_id"].unique())
        sys.exit(f"ERROR: well {args.well!r} not in inventory. Available: {available}")

    print(f"Inventory: {args.inventory}")
    print(f"Well     : {args.well}  ({len(well_inv)} rows)")

    # Write filtered inventory to a temp file so we can pass a CSV path to the router.
    tmp_inv = Path("/tmp/_smoke_well_inventory.csv")
    well_inv.to_csv(tmp_inv, index=False)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    from data_pipeline.models.groundingdino import load_groundingdino_model

    print(f"\nLoading GroundingDINO on {args.device} ...")
    model = load_groundingdino_model(
        repo_dir=GDINO_REPO,
        config_path=GDINO_CONFIG,
        weights_path=GDINO_WEIGHTS,
        device=args.device,
    )
    print("Model loaded.")

    # ── 3. Run detection ──────────────────────────────────────────────────────
    from data_pipeline.detection import run_frame_detection
    from data_pipeline.detection.backends.groundingdino.config import GroundingDinoDetectionConfig

    print(f"\nRunning detection → {args.out} ...")
    df = run_frame_detection(
        frame_inventory_csv=tmp_inv,
        output_csv=args.out,
        backend="groundingdino",
        model=model,
        detector_model_id="SwinT_OGC",
        config=GroundingDinoDetectionConfig(device=args.device),
    )
    print(f"Detection complete. Output: {args.out}")

    # ── 4. Re-validate (belt-and-suspenders) ─────────────────────────────────
    from data_pipeline.detection.validate_frame_detections import validate_frame_detections

    validate_frame_detections(df, well_inv, context="smoke_test")
    print("validate_frame_detections: PASS")

    # ── 5. Print summary ──────────────────────────────────────────────────────
    _print_summary(df, well_inv)


if __name__ == "__main__":
    main()
