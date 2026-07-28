#!/usr/bin/env python
"""Timed benchmark for candidate focus metrics on existing 2D images."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from posthoc_focus_candidate_metric_comparison import (
    FOCUS_CSV,
    RANDOM_SEED,
    metric_row,
    focus_status,
)


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="20251125")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FOCUS_CSV)
    df = df[(df["experiment_id"].astype(str) == str(args.experiment)) & df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["ff_rel_entropy", "ff_lap_abs_ratio"]).copy()
    df["status"] = focus_status(df)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    rows: list[dict] = []
    start = time.monotonic()
    deadline = start + args.seconds

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if time.monotonic() >= deadline:
            break
        base = row.to_dict()
        try:
            rows.append({**base, **metric_row(row)})
        except Exception as exc:
            rows.append({**base, "candidate_metric_error": str(exc)})

        if len(rows) % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            elapsed = time.monotonic() - start
            rate = len(rows) / elapsed if elapsed > 0 else np.nan
            print(f"{len(rows)} rows in {elapsed:.1f}s ({rate:.2f} rows/s)", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    elapsed = time.monotonic() - start
    rate = len(out_df) / elapsed if elapsed > 0 else np.nan
    print(f"Saved {len(out_df)} rows -> {args.out}", flush=True)
    print(f"Elapsed {elapsed:.1f}s; rate {rate:.2f} rows/s", flush=True)
    print(f"Estimated 20251125 full not-dead time: {len(df) / rate / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
