#!/usr/bin/env python3
"""
Quick existence checker for per‑experiment paths using src/run_morphseq_pipeline/paths.py.

Usage:
  python -m scripts.check_paths_experiment \
    --data-root /path/to/root \
    --exp 20250529_36hpf_ctrl_atf6 20250612_24hpf_wfs1_ctcf \
    --model-name 20241107_ds_sweep01_optimum

Prints a concise, per‑experiment report of expected files/dirs and whether they exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.run_morphseq_pipeline.paths import (
    get_stage_ref_csv,
    get_perturbation_key_csv,
    get_well_metadata_xlsx,
    get_stitched_ff_dir,
    get_built_metadata_csv,
    get_sam2_csv,
    get_sam2_masks_dir,
    get_gdino_detections_json,
    get_sam2_segmentations_json,
    get_sam2_mask_export_manifest,
    get_experiment_metadata_json,
    get_build03_output,
    get_snips_dir,
    get_build04_output,
    get_latents_csv,
    get_build06_output,
)


def _exists(p: Path) -> str:
    return "✅" if p.exists() else "❌"


def _dir_nonempty(p: Path) -> str:
    if not p.exists() or not p.is_dir():
        return "❌"
    try:
        next(p.iterdir())
        return "✅"
    except StopIteration:
        return "❌ (empty)"
    except Exception:
        return "❌"


def check_experiment(root: Path, exp: str, model_name: str) -> None:
    print(f"\n=== Experiment: {exp} ===")

    # Global refs
    stage_ref = get_stage_ref_csv(root)
    pert_key = get_perturbation_key_csv(root)
    print(f"Stage Ref: {_exists(stage_ref)} {stage_ref}")
    print(f"Perturbation Key: {_exists(pert_key)} {pert_key}")

    # Build01
    ff_dir = get_stitched_ff_dir(root, exp)
    built_meta = get_built_metadata_csv(root, exp)
    well_xlsx = get_well_metadata_xlsx(root, exp)
    print(f"Stitched FF dir: {_dir_nonempty(ff_dir)} {ff_dir}")
    print(f"Built metadata CSV: {_exists(built_meta)} {built_meta}")
    print(f"Well metadata XLSX: {_exists(well_xlsx)} {well_xlsx}")

    # SAM2
    sam2_csv = get_sam2_csv(root, exp)
    sam2_masks = get_sam2_masks_dir(root, exp)
    gdino_json = get_gdino_detections_json(root, exp)
    seg_json = get_sam2_segmentations_json(root, exp)
    mask_manifest = get_sam2_mask_export_manifest(root, exp)
    exp_meta_json = get_experiment_metadata_json(root, exp)
    print(f"SAM2 CSV: {_exists(sam2_csv)} {sam2_csv}")
    print(f"Masks dir: {_dir_nonempty(sam2_masks)} {sam2_masks}")
    print(f"GDINO: {_exists(gdino_json)} {gdino_json}")
    print(f"SAM2 seg: {_exists(seg_json)} {seg_json}")
    print(f"Mask manifest: {_exists(mask_manifest)} {mask_manifest}")
    print(f"Exp metadata JSON: {_exists(exp_meta_json)} {exp_meta_json}")

    # Build03
    b03 = get_build03_output(root, exp)
    snips = get_snips_dir(root, exp)
    print(f"Build03 df01: {_exists(b03)} {b03}")
    print(f"Snips dir: {_dir_nonempty(snips)} {snips}")

    # Build04
    b04 = get_build04_output(root, exp)
    print(f"Build04 df02: {_exists(b04)} {b04}")

    # Build06
    latents = get_latents_csv(root, model_name, exp)
    b06 = get_build06_output(root, exp)
    print(f"Latents CSV: {_exists(latents)} {latents}")
    print(f"Build06 df03: {_exists(b06)} {b06}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check per‑experiment path existence")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--exp", nargs="+", required=True, help="Experiment name(s)")
    ap.add_argument("--model-name", default="20241107_ds_sweep01_optimum")
    args = ap.parse_args()

    root = Path(args.data_root)
    print(f"Root: {root}")
    print(f"Model: {args.model_name}")

    for exp in args.exp:
        check_experiment(root, exp, args.model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

