"""
11_merge_chunks.py
==================
After the 11_focus_stacked_full_scan.qsub array job completes, merge all
projected-domain chunk CSVs into one ff_metrics_full.csv under the organized
outputs tree.

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/11_merge_chunks.py
"""

from pathlib import Path
import pandas as pd

SCAN_DIR = Path(__file__).resolve().parent / "outputs/projected/scan"
CHUNK_DIR = SCAN_DIR / "chunks"
CSV_OUT = SCAN_DIR / "ff_metrics_full.csv"

chunks = sorted(CHUNK_DIR.glob("chunk_*.csv"))
print(f"Found {len(chunks)} chunk files in {CHUNK_DIR}")

dfs = [pd.read_csv(c) for c in chunks if c.stat().st_size > 0]
df = pd.concat(dfs, ignore_index=True).sort_values(["t", "well"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["t", "well"], keep="last")

df.to_csv(CSV_OUT, index=False)
print(f"Saved -> {CSV_OUT}  ({len(df)} rows)")
print(df[["rel_entropy_mean", "ff_rel_entropy", "ff_lap_abs_ratio"]].describe())
