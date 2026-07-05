"""
Synthetic validation sweep for candidate adaptive KDE estimators.

This tunes only against synthetic scenarios, not real anchor genes. Outputs are
kept separate from the bandwidth-sensitivity diagnostic:

    tables/density_improvements/
    plots/density_improvements/

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/adaptive_kde_validation.py
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
sys.path.insert(0, str(RUN_DIR))

from support_geometry import (  # noqa: E402
    KDESpec,
    compute_support_geometry,
    knn_adaptive_kde_spec,
    normalize_shape,
    scipy_gaussian_kde_spec,
    valley_depth,
    valley_detection_detail,
)
from synthetic_scenarios import SCENARIOS, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "density_improvements"
PLOT_DIR = RUN_DIR / "plots" / "density_improvements"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_N = 40
DEFAULT_N_SEEDS = 3
DEFAULT_N_RESAMPLE = 50
WT_POOL_N = 3000
SIG_ALPHA = 0.05

CONTROLS = {"unimodal_compact", "variance_only", "crescent", "spiral", "outliers"}
TARGETS = {"two_discrete", "three_discrete", "small_middle"}

_WT_POOL = None


def candidate_specs() -> list[tuple[str, KDESpec]]:
    return [
        ("scott", scipy_gaussian_kde_spec()),
        ("knn_k3_bw0.4_clip1.5", knn_adaptive_kde_spec(k=3, bw_scale=0.4, max_factor=1.5)),
        ("knn_k3_bw0.5_clip1.5", knn_adaptive_kde_spec(k=3, bw_scale=0.5, max_factor=1.5)),
        ("knn_k3_bw0.6_clip1.5", knn_adaptive_kde_spec(k=3, bw_scale=0.6, max_factor=1.5)),
        ("knn_k5_bw0.5_clip1.5", knn_adaptive_kde_spec(k=5, bw_scale=0.5, max_factor=1.5)),
        ("knn_k5_bw0.6_clip1.5", knn_adaptive_kde_spec(k=5, bw_scale=0.6, max_factor=1.5)),
        ("knn_k5_bw0.6", knn_adaptive_kde_spec(k=5, bw_scale=0.6)),
        ("knn_k5_bw0.8", knn_adaptive_kde_spec(k=5, bw_scale=0.8)),
        ("knn_k8_bw0.6", knn_adaptive_kde_spec(k=8, bw_scale=0.6)),
        ("knn_k8_bw0.7", knn_adaptive_kde_spec(k=8, bw_scale=0.7)),
        ("knn_k10_bw0.6", knn_adaptive_kde_spec(k=10, bw_scale=0.6)),
        ("knn_k10_bw0.7", knn_adaptive_kde_spec(k=10, bw_scale=0.7)),
        ("knn_k10_bw0.8", knn_adaptive_kde_spec(k=10, bw_scale=0.8)),
        ("knn_k10_bw1.0", knn_adaptive_kde_spec(k=10, bw_scale=1.0)),
        ("knn_k15_bw1.0", knn_adaptive_kde_spec(k=15, bw_scale=1.0)),
    ]


def _init_worker(wt_pool: np.ndarray) -> None:
    global _WT_POOL
    _WT_POOL = wt_pool


def _component_summary(detail: dict) -> tuple[int, str]:
    masses = np.asarray(detail.get("component_mass_fracs", []), dtype=float)
    if masses.size == 0:
        return 0, ""
    return int(detail.get("n_components_at_split", 0)), ";".join(f"{m:.4f}" for m in masses)


def _run_one_task(task: tuple[int, int, str, KDESpec, int, int]) -> dict:
    scen_idx, seed, candidate, spec, n, n_resample = task
    scen = SCENARIOS[scen_idx]
    wt_pool = _WT_POOL
    if wt_pool is None:
        wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))

    rng = np.random.default_rng(1000 + 100 * seed + n)
    pts_raw = scen.generator(n, rng)
    pts_norm = normalize_shape(pts_raw)
    bundle = compute_support_geometry(
        pts_raw,
        wt_pool,
        n_resample=n_resample,
        rng=np.random.default_rng(7000 + seed),
        statistics={"valley_depth": valley_depth},
        kde=spec,
    )
    vd = bundle.results["valley_depth"]
    detail = valley_detection_detail(pts_norm, kde=spec)
    n_comp, mass_fracs = _component_summary(detail)
    return {
        "candidate": candidate,
        "scenario": scen.name,
        "expected_support": scen.expected_support,
        "n": n,
        "seed": seed,
        "valley_depth": vd.stat,
        "valley_pvalue": vd.pvalue,
        "valley_significant": vd.pvalue < SIG_ALPHA,
        "n_components_at_split": n_comp,
        "component_mass_fracs": mass_fracs,
    }


def _write_partial(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(TABLE_DIR / "adaptive_kde_validation.partial.csv", index=False)


def _select_candidate_specs(candidate_names: set[str] | None = None) -> list[tuple[str, KDESpec]]:
    return [
        (name, spec) for name, spec in candidate_specs()
        if candidate_names is None or name in candidate_names
    ]


def _select_scenario_indices(scenario_names: set[str] | None = None) -> list[int]:
    return [
        i for i, scen in enumerate(SCENARIOS)
        if scenario_names is None or scen.name in scenario_names
    ]


def run_sweep(
    n: int,
    n_seeds: int,
    n_resample: int,
    n_workers: int,
    scenario_names: set[str] | None = None,
    candidate_names: set[str] | None = None,
) -> pd.DataFrame:
    wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))
    partial = TABLE_DIR / "adaptive_kde_validation.partial.csv"
    partial.unlink(missing_ok=True)
    specs = _select_candidate_specs(candidate_names)
    scen_indices = _select_scenario_indices(scenario_names)
    tasks = [
        (scen_idx, seed, candidate, spec, n, n_resample)
        for scen_idx in scen_indices
        for seed in range(n_seeds)
        for candidate, spec in specs
    ]
    rows = []
    total = len(tasks)
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        _init_worker(wt_pool)
        iterator = map(_run_one_task, tasks)
    else:
        pool = mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(wt_pool,))
        iterator = pool.imap_unordered(_run_one_task, tasks, chunksize=3)

    try:
        for done, row in enumerate(iterator, start=1):
            rows.append(row)
            if done % 25 == 0 or done == total:
                _write_partial(rows)
                print(f"  completed {done}/{total} partial={partial}", flush=True)
    finally:
        if n_workers != 1:
            pool.close()
            pool.join()

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["candidate", "scenario", "expected_support"], as_index=False)
        .agg(significant_rate=("valley_significant", "mean"),
             median_pvalue=("valley_pvalue", "median"),
             median_valley_depth=("valley_depth", "median"))
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
    score_df = pd.DataFrame(scores).sort_values(["score", "target_rate"], ascending=False)
    summary.to_csv(TABLE_DIR / "adaptive_kde_validation_summary.csv", index=False)
    score_df.to_csv(TABLE_DIR / "adaptive_kde_validation_scores.csv", index=False)
    return summary


def make_plot(summary: pd.DataFrame) -> Path:
    available_scenarios = set(summary["scenario"])
    order = [s.name for s in SCENARIOS if s.name in available_scenarios]
    available = set(summary["candidate"])
    candidates = [name for name, _ in candidate_specs() if name in available]
    pivot = (
        summary.pivot(index="scenario", columns="candidate", values="significant_rate")
        .reindex(index=order, columns=candidates)
    )

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    expected = {s.name: s.expected_support for s in SCENARIOS}
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{name} ({expected[name][0]})" for name in pivot.index], fontsize=9)
    ax.set_title(
        f"adaptive KDE candidate sweep\ncell = fraction valley_depth significant at p<{SIG_ALPHA}",
        fontsize=12,
        fontweight="bold",
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            color = "white" if val < 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("significant fraction")
    fig.tight_layout()
    out = PLOT_DIR / "adaptive_kde_candidate_heatmap.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=DEFAULT_N)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--n-resample", type=int, default=DEFAULT_N_RESAMPLE)
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--candidates", nargs="*", default=None)
    args = parser.parse_args()

    scenario_names = set(args.scenarios) if args.scenarios else None
    candidate_names = set(args.candidates) if args.candidates else None
    selected_scenarios = [SCENARIOS[i].name for i in _select_scenario_indices(scenario_names)]
    selected_candidates = [name for name, _ in _select_candidate_specs(candidate_names)]

    print("Adaptive KDE synthetic validation")
    print(f"  scenarios={selected_scenarios}")
    print(f"  candidates={selected_candidates}")
    print(f"  n={args.sample_size} n_seeds={args.n_seeds} n_resample={args.n_resample} n_workers={args.n_workers}")
    df = run_sweep(
        args.sample_size,
        args.n_seeds,
        args.n_resample,
        args.n_workers,
        scenario_names=scenario_names,
        candidate_names=candidate_names,
    )
    csv_out = TABLE_DIR / "adaptive_kde_validation.csv"
    df.to_csv(csv_out, index=False)
    summary = summarize(df)
    plot_out = make_plot(summary)
    print(f"Saved: {csv_out}")
    print(f"Saved: {TABLE_DIR / 'adaptive_kde_validation_summary.csv'}")
    print(f"Saved: {TABLE_DIR / 'adaptive_kde_validation_scores.csv'}")
    print(f"Saved: {plot_out}")


if __name__ == "__main__":
    main()
