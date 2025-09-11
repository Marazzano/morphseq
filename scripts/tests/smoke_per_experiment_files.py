#!/usr/bin/env python3
"""
Smoke test for per-experiment SAM2 files (zero heavy imports, instant).

Usage:
  python scripts/tests/smoke_per_experiment_files.py \
    --data-root /path/to/morphseq_playground \
    --experiments 20250529_30hpf_wfs1_ctcf,20250529_36hpf_ctrl_atf6_extras
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiments", required=True, help="Comma-separated experiment IDs")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    for exp in experiments:
        sam2_root = data_root / "sam2_pipeline_files"
        meta_path = sam2_root / "raw_data_organized" / exp / f"experiment_metadata_{exp}.json"
        gdino_path = sam2_root / "detections" / f"gdino_detections_{exp}.json"
        seg_path   = sam2_root / "segmentation" / f"grounded_sam_segmentations_{exp}.json"
        csv_path   = sam2_root / "sam2_expr_files" / f"sam2_metadata_{exp}.csv"

        print(f"\n=== {exp} ===")
        print(f"experiment_metadata_path: {meta_path}  ->  {'OK' if meta_path.exists() else 'MISSING'}")
        print(f"gdino_detections_path:   {gdino_path}  ->  {'OK' if gdino_path.exists() else 'MISSING'}")
        print(f"sam2_segmentations_path: {seg_path}    ->  {'OK' if seg_path.exists() else 'MISSING'}")
        print(f"sam2_csv_path:            {csv_path}    ->  {'OK' if csv_path.exists() else 'MISSING'}")


if __name__ == "__main__":
    sys.exit(main())
