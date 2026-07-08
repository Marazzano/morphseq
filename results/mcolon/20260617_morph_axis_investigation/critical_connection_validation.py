"""
Focused validation for alpha-like support-connectivity statistics.

This is candidate triage, not framework adoption. It evaluates
critical_connection_ratio on the same target/control synthetic gate used for the
KDE bandwidth work, using the existing WT bootstrap-null machinery.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/critical_connection_validation.py
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from support_geometry import (  # noqa: E402
    compute_support_geometry,
    critical_connection_ratio,
    critical_major_connection_ratio,
)
from synthetic_scenarios import SCENARIOS, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "support_connectivity"
PLOT_DIR = RUN_DIR / "plots" / "support_connectivity"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GATE_SCENARIOS = [
    "unimodal_compact",
    "variance_only",
    "crescent",
    "spiral",
    "outliers",
    "two_discrete",
    "three_discrete",
    "small_middle",
]
TARGETS = {"two_discrete", "three_discrete", "small_middle"}
CONTROLS = {"unimodal_compact", "variance_only", "crescent", "spiral", "outliers"}
WT_POOL_N = 3000
SIG_ALPHA = 0.05
_WT_POOL = None


def candidate_specs(candidate_names: list[str] | None = None) -> list[tuple[str, dict]]:
    specs = [
        ("largest_k3_mass90", {"fn": "largest", "k": 3, "target_mass": 0.90}),
        ("largest_k5_mass90", {"fn": "largest", "k": 5, "target_mass": 0.90}),
        ("major_k3_min05_cov85", {"fn": "major", "k": 3, "min_component_mass": 0.05, "covered_mass": 0.85}),
        ("major_k5_min05_cov85", {"fn": "major", "k": 5, "min_component_mass": 0.05, "covered_mass": 0.85}),
        ("major_k3_min10_cov85", {"fn": "major", "k": 3, "min_component_mass": 0.10, "covered_mass": 0.85}),
        ("major_k5_min10_cov85", {"fn": "major", "k": 5, "min_component_mass": 0.10, "covered_mass": 0.85}),
        ("major_k3_min10_cov90", {"fn": "major", "k": 3, "min_component_mass": 0.10, "covered_mass": 0.90}),
        ("major_k5_min10_cov90", {"fn": "major", "k": 5, "min_component_mass": 0.10, "covered_mass": 0.90}),
    ]
    if candidate_names is None:
        return specs
    requested = set(candidate_names)
    known = {name for name, _ in specs}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown candidate(s): {unknown}. Known candidates: {sorted(known)}")
    return [(name, params) for name, params in specs if name in requested]


def _init_worker(wt_pool: np.ndarray) -> None:
    global _WT_POOL
    _WT_POOL = wt_pool


def _run_one(task: tuple[int, int, str, dict, int, int]) -> dict:
    scen_idx, seed, candidate, params, n, n_resample = task
    scen = SCENARIOS[scen_idx]
    wt_pool = _WT_POOL
    if wt_pool is None:
        wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))

    rng = np.random.default_rng(1000 + 100 * seed + n)
    pts = scen.generator(n, rng)
    params = dict(params)
    fn_name = params.pop("fn")
    if fn_name == "largest":
        stat_fn = lambda x: critical_connection_ratio(x, **params)
    elif fn_name == "major":
        stat_fn = lambda x: critical_major_connection_ratio(x, **params)
    else:
        raise ValueError(f"Unknown critical-connection candidate type: {fn_name!r}")
    bundle = compute_support_geometry(
        pts,
        wt_pool,
        n_resample=n_resample,
        rng=np.random.default_rng(7000 + seed),
        statistics={"critical_connection_ratio": stat_fn},
    )
    sr = bundle.results["critical_connection_ratio"]
    return {
        "candidate": candidate,
        "scenario": scen.name,
        "expected_support": scen.expected_support,
        "n": n,
        "seed": seed,
        "stat": sr.stat,
        "pvalue": sr.pvalue,
        "significant": sr.pvalue < SIG_ALPHA,
        "reference_stat": sr.reference_stat,
    }


def run_sweep(
    n: int,
    n_seeds: int,
    n_resample: int,
    n_workers: int,
    candidate_names: list[str] | None = None,
    output_prefix: str = "critical_connection_validation",
) -> pd.DataFrame:
    wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))
    scenario_indices = [i for i, scen in enumerate(SCENARIOS) if scen.name in GATE_SCENARIOS]
    candidates = candidate_specs(candidate_names)
    tasks = [
        (scen_idx, seed, candidate, params, n, n_resample)
        for scen_idx in scenario_indices
        for seed in range(n_seeds)
        for candidate, params in candidates
    ]
    partial = TABLE_DIR / f"{output_prefix}.partial.csv"
    partial.unlink(missing_ok=True)
    rows = []
    total = len(tasks)
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        _init_worker(wt_pool)
        iterator = map(_run_one, tasks)
    else:
        pool = mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(wt_pool,))
        iterator = pool.imap_unordered(_run_one, tasks, chunksize=4)

    try:
        for done, row in enumerate(iterator, start=1):
            rows.append(row)
            if done % 25 == 0 or done == total:
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(f"  completed {done}/{total} partial={partial}", flush=True)
    finally:
        if n_workers != 1:
            pool.close()
            pool.join()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, output_prefix: str = "critical_connection_validation") -> pd.DataFrame:
    summary = (
        df.groupby(["candidate", "scenario", "expected_support"], as_index=False)
        .agg(significant_rate=("significant", "mean"),
             median_pvalue=("pvalue", "median"),
             median_stat=("stat", "median"),
             median_reference=("reference_stat", "median"))
    )
    scores = []
    for candidate, sub in summary.groupby("candidate"):
        target_rate = sub[sub["scenario"].isin(TARGETS)]["significant_rate"].mean()
        control_rate = sub[sub["scenario"].isin(CONTROLS)]["significant_rate"].mean()
        scores.append({
            "candidate": candidate,
            "target_rate": target_rate,
            "control_rate": control_rate,
            "score": target_rate - control_rate,
        })
    pd.DataFrame(scores).sort_values(["score", "target_rate"], ascending=False).to_csv(
        TABLE_DIR / f"{output_prefix}_scores.csv",
        index=False,
    )
    summary.to_csv(TABLE_DIR / f"{output_prefix}_summary.csv", index=False)
    return summary


def make_plot(summary: pd.DataFrame, output_prefix: str = "critical_connection_validation") -> Path:
    candidates = [name for name, _ in candidate_specs() if name in set(summary["candidate"])]
    pivot = (
        summary.pivot(index="scenario", columns="candidate", values="significant_rate")
        .reindex(index=GATE_SCENARIOS, columns=candidates)
    )
    expected = {s.name: s.expected_support for s in SCENARIOS}

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(candidates)))
    ax.set_xticklabels(candidates, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(GATE_SCENARIOS)))
    ax.set_yticklabels([f"{name} ({expected[name][0]})" for name in GATE_SCENARIOS])
    ax.set_title(
        f"critical_connection_ratio focused gate\ncell = fraction significant at p<{SIG_ALPHA}",
        fontsize=12,
        fontweight="bold",
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            color = "white" if val < 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("significant fraction")
    fig.tight_layout()
    out = PLOT_DIR / f"{output_prefix}_heatmap.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-resample", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--output-prefix", default="critical_connection_validation")
    args = parser.parse_args()

    candidates = candidate_specs(args.candidates)
    print("Critical-connection focused validation")
    print(f"  scenarios={GATE_SCENARIOS}")
    print(f"  candidates={[name for name, _ in candidates]}")
    print(f"  n={args.sample_size} n_seeds={args.n_seeds} n_resample={args.n_resample} n_workers={args.n_workers}")
    df = run_sweep(
        args.sample_size,
        args.n_seeds,
        args.n_resample,
        args.n_workers,
        candidate_names=args.candidates,
        output_prefix=args.output_prefix,
    )
    csv_out = TABLE_DIR / f"{args.output_prefix}.csv"
    df.to_csv(csv_out, index=False)
    summary = summarize(df, output_prefix=args.output_prefix)
    plot_out = make_plot(summary, output_prefix=args.output_prefix)
    print(f"Saved: {csv_out}")
    print(f"Saved: {TABLE_DIR / f'{args.output_prefix}_summary.csv'}")
    print(f"Saved: {TABLE_DIR / f'{args.output_prefix}_scores.csv'}")
    print(f"Saved: {plot_out}")


if __name__ == "__main__":
    main()
