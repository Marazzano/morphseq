"""Run the minimal V0 modal-organization benchmark.

This script is the first validation layer above the V0 density composition:

- reuse the fixed V0 distributions
- validate the composed density peak count against the expected recipe label
- sample multiple replicates per distribution
- compute the four current V0 metrics
- write a compact per-distribution summary table
- write a small pairwise ordering table for the expected ladders

The goal is not to build the full registry yet. The goal is to keep the V0
corpus runnable as a regression target while the metric definitions settle.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphseq_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/morphseq_xdg_cache")
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.core.density_composition import (  # noqa: E402
    compose_density_truth,
    realize_from_truth,
    validate_composed_density_truth,
)
from morphseq_investigation.core.peak_counting import count_mass_significant_modes  # noqa: E402
from morphseq_investigation.core.support_geometry import evaluate_kde_on_grid  # noqa: E402
from morphseq_investigation.plotting.modal_distribution_plotting import compute_v0_metric_summary  # noqa: E402
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS  # noqa: E402


OUT_DIR = RUN_DIR / "tables" / "modal_v0_benchmark"


def _expected_truth_peak_count(distribution_id: str) -> int:
    if distribution_id.startswith("one_peak_"):
        return 1
    if distribution_id.startswith("two_peaks_"):
        return 2
    if distribution_id.startswith("three_peaks_"):
        return 3
    raise ValueError(f"Cannot infer expected peak count for {distribution_id!r}")


def _expected_bridge_labels(distribution_id: str) -> dict[str, str]:
    if distribution_id == "two_peaks_no_bridge":
        return {"bridge_left_right": "no_bridge"}
    if distribution_id == "two_peaks_low_bridge":
        return {"bridge_left_right": "low_bridge"}
    if distribution_id == "two_peaks_high_bridge":
        return {"bridge_left_right": "high_bridge"}
    return {}


def _summarize_metric(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _paired_win_rate(left: np.ndarray, right: np.ndarray, *, greater_is_better: bool = True) -> tuple[float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size == 0:
        return float("nan"), float("nan")
    delta = left - right
    if greater_is_better:
        win_rate = float(np.mean(delta > 0))
        return win_rate, float(np.median(delta))
    win_rate = float(np.mean(delta < 0))
    return win_rate, float(np.median(delta))


def _observed_peak_count(points: np.ndarray, truth_grid) -> tuple[int, float | None]:
    xx = truth_grid.xx
    yy = truth_grid.yy
    density = evaluate_kde_on_grid(points, xx, yy)
    detail = count_mass_significant_modes(density)
    return detail[0], detail[1]


def build_benchmark_rows(n: int, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_rows: list[dict[str, object]] = []
    rep_rows: list[dict[str, object]] = []

    for dist_index, dist in enumerate(V0_DISTRIBUTIONS):
        truth = compose_density_truth(dist.density_spec)
        expected_peak_count = _expected_truth_peak_count(dist.distribution_id)
        validation = validate_composed_density_truth(
            truth,
            expected_peak_count=expected_peak_count,
            expected_bridge_labels=_expected_bridge_labels(dist.distribution_id),
        )
        truth_peak_count = validation.resolved_peak_count
        truth_split_level = count_mass_significant_modes(truth.composed_grid.density)[1]
        bridge_labels = ";".join(br.bridge_region_label for br in validation.bridge_regions) or "n/a"

        truth_rows.append(
            {
                "distribution_id": dist.distribution_id,
                "note": dist.note,
                "expected_peak_count": expected_peak_count,
                "truth_peak_count": truth_peak_count,
                "truth_split_level": truth_split_level,
                "bridge_labels": bridge_labels,
                "total_mass": validation.total_mass,
                "x_min": truth.composed_grid.grid.x_min,
                "x_max": truth.composed_grid.grid.x_max,
                "y_min": truth.composed_grid.grid.y_min,
                "y_max": truth.composed_grid.grid.y_max,
            }
        )

        for rep in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, rep, dist_index]))
            realization = realize_from_truth(truth, n=n, rng=rng)
            metric_summary = compute_v0_metric_summary(
                realization.points,
                distribution_id=dist.distribution_id,
            )
            observed_peak_count, observed_split_level = _observed_peak_count(
                realization.points,
                truth.composed_grid.grid,
            )
            rep_rows.append(
                {
                    "distribution_id": dist.distribution_id,
                    "rep": rep,
                    "seed": seed,
                    "n": n,
                    "truth_peak_count": truth_peak_count,
                    "observed_peak_count": observed_peak_count,
                    "observed_peak_match": observed_peak_count == truth_peak_count,
                    "observed_split_level": observed_split_level,
                    **metric_summary,
                }
            )

    return pd.DataFrame(truth_rows), pd.DataFrame(rep_rows)


def build_summary_table(truth_df: pd.DataFrame, rep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = rep_df.groupby("distribution_id", sort=False)

    for _, truth_row in truth_df.iterrows():
        dist_id = str(truth_row["distribution_id"])
        sub = grouped.get_group(dist_id)
        row = {
            "distribution_id": dist_id,
            "note": truth_row["note"],
            "truth_peak_count": int(truth_row["truth_peak_count"]),
            "truth_split_level": truth_row["truth_split_level"],
            "observed_peak_count_match_rate": float(np.mean(sub["observed_peak_match"].astype(float))),
        }
        for metric in ("hdr_concentration_auc", "valley_depth", "mst_max_edge", "fiedler"):
            stats = _summarize_metric(sub[metric].to_numpy())
            row[f"{metric}_median"] = stats["median"]
            row[f"{metric}_q25"] = stats["q25"]
            row[f"{metric}_q75"] = stats["q75"]
        stats = _summarize_metric(sub["observed_peak_count"].to_numpy())
        row["observed_peak_count_median"] = stats["median"]
        row["observed_peak_count_q25"] = stats["q25"]
        row["observed_peak_count_q75"] = stats["q75"]
        row["resolution_artifact_expected"] = row["observed_peak_count_match_rate"] < 1.0
        rows.append(row)

    return pd.DataFrame(rows)


def build_pairwise_table(rep_df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("concentration", "one_peak_compact", "one_peak_diffuse", "hdr_concentration_auc", True),
        ("bridge", "two_peaks_no_bridge", "two_peaks_low_bridge", "valley_depth", True),
        ("bridge", "two_peaks_low_bridge", "two_peaks_high_bridge", "valley_depth", True),
        ("bridge", "two_peaks_no_bridge", "two_peaks_low_bridge", "mst_max_edge", True),
        ("bridge", "two_peaks_low_bridge", "two_peaks_high_bridge", "mst_max_edge", True),
        ("bridge", "two_peaks_no_bridge", "two_peaks_low_bridge", "fiedler", True),
        ("bridge", "two_peaks_low_bridge", "two_peaks_high_bridge", "fiedler", True),
        ("modal_count", "three_peaks_compact", "one_peak_compact", "observed_peak_count", True),
    ]

    rows: list[dict[str, object]] = []
    for ladder, left_id, right_id, metric, greater_is_better in checks:
        left = rep_df.loc[rep_df["distribution_id"] == left_id, metric].to_numpy()
        right = rep_df.loc[rep_df["distribution_id"] == right_id, metric].to_numpy()
        win_rate, median_delta = _paired_win_rate(left, right, greater_is_better=greater_is_better)
        rows.append(
            {
                "ladder": ladder,
                "metric": metric,
                "left_id": left_id,
                "right_id": right_id,
                "expected_relation": "left > right" if greater_is_better else "left < right",
                "paired_win_rate": win_rate,
                "median_delta": median_delta,
                "pass": bool(np.isfinite(win_rate) and win_rate >= 0.80),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, out_path: Path, *, n: int, reps: int, seed: int) -> None:
    passed = int(pairwise_df["pass"].sum())
    total = int(len(pairwise_df))
    lines = [
        f"V0 modal benchmark",
        f"n = {n}, reps = {reps}, seed = {seed}",
        "",
        f"Truth peak counts: all {len(summary_df)} distributions matched expected counts.",
        f"Pairwise checks passed: {passed}/{total}",
        "",
        "Pairwise failures:",
    ]
    failed = pairwise_df.loc[~pairwise_df["pass"]]
    if failed.empty:
        lines.append("  none")
    else:
        for _, row in failed.iterrows():
            lines.append(
                f"  - {row['left_id']} vs {row['right_id']} on {row['metric']}: "
                f"win_rate={row['paired_win_rate']:.2f}, delta={row['median_delta']:.3f}"
            )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="Sample size per replicate.")
    parser.add_argument("--reps", type=int, default=5, help="Number of replicates per V0 distribution.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for tables.")
    parser.add_argument(
        "--strict-pairwise",
        dest="strict_pairwise",
        action="store_true",
        help="Raise if expected pairwise checks fail.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict_pairwise",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(strict_pairwise=False)
    args = parser.parse_args()

    truth_df, rep_df = build_benchmark_rows(args.n, args.reps, args.seed)
    summary_df = build_summary_table(truth_df, rep_df)
    pairwise_df = build_pairwise_table(rep_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = args.out_dir / f"v0_truth_peaks_n{args.n}_reps{args.reps}_seed{args.seed}.csv"
    summary_path = args.out_dir / f"v0_metric_summary_n{args.n}_reps{args.reps}_seed{args.seed}.csv"
    pairwise_path = args.out_dir / f"v0_pairwise_orderings_n{args.n}_reps{args.reps}_seed{args.seed}.csv"
    report_path = args.out_dir / f"v0_report_n{args.n}_reps{args.reps}_seed{args.seed}.md"

    truth_df.to_csv(truth_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    write_report(summary_df, pairwise_df, report_path, n=args.n, reps=args.reps, seed=args.seed)

    print(f"Saved: {truth_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {pairwise_path}")
    print(f"Saved: {report_path}")

    if args.strict_pairwise and not bool(pairwise_df["pass"].all()):
        failed = pairwise_df.loc[~pairwise_df["pass"], ["ladder", "metric", "left_id", "right_id", "paired_win_rate"]]
        raise RuntimeError(f"V0 benchmark pairwise checks failed:\n{failed.to_string(index=False)}")


if __name__ == "__main__":
    main()
