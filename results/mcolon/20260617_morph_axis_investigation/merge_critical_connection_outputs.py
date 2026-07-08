"""
Merge per-candidate critical-connection gate outputs from the SGE array run.

Run after all array tasks complete:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/merge_critical_connection_outputs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from critical_connection_validation import (  # noqa: E402
    TABLE_DIR,
    candidate_specs,
    make_plot,
    summarize,
)


def expected_files() -> list[Path]:
    return [TABLE_DIR / f"critical_connection_{name}.csv" for name, _ in candidate_specs()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="critical_connection_validation")
    args = parser.parse_args()

    files = [path for path in expected_files() if path.exists()]
    missing = [path.name for path in expected_files() if not path.exists()]
    if not files:
        raise SystemExit(f"No per-candidate outputs found in {TABLE_DIR}")
    if missing:
        print(f"Warning: missing {len(missing)} expected output(s): {missing}")

    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    csv_out = TABLE_DIR / f"{args.output_prefix}.csv"
    df.to_csv(csv_out, index=False)
    summary = summarize(df, output_prefix=args.output_prefix)
    plot_out = make_plot(summary, output_prefix=args.output_prefix)
    print(f"Merged {len(files)} file(s)")
    print(f"Saved: {csv_out}")
    print(f"Saved: {TABLE_DIR / f'{args.output_prefix}_summary.csv'}")
    print(f"Saved: {TABLE_DIR / f'{args.output_prefix}_scores.csv'}")
    print(f"Saved: {plot_out}")


if __name__ == "__main__":
    main()
