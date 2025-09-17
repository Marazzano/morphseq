#!/usr/bin/env python
"""Concatenate selected Build06 CSVs into a single file.

Edit the CSV_BASENAMES list below and run the script. It will grab each file
from BUILD06_DIR, tag rows with their source basename, and write the combined
table to OUTPUT_PATH.
"""

from pathlib import Path

import pandas as pd


# Directory containing the per-experiment Build06 CSVs.
BUILD06_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"
)

# Basenames (or relative paths) of the CSVs you want to combine.
CSV_BASENAMES = [
    # Example entries — replace with the CSV filenames you need.
    # "df03_final_output_with_latents_20250305.csv",
    # "df03_final_output_with_latents_20250912.csv",
]

# Where to write the combined CSV.
OUTPUT_PATH = Path(__file__).with_name("combined_build06.csv")


def main() -> None:
    if not CSV_BASENAMES:
        raise SystemExit("Populate CSV_BASENAMES with the files you want to merge.")

    frames = []
    for name in CSV_BASENAMES:
        csv_path = (BUILD06_DIR / name).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df["source_csv"] = csv_path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, copy=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows from {len(frames)} CSVs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

