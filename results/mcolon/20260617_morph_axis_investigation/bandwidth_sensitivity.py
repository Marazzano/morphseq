"""
Bandwidth-sensitivity diagnostic for valley_depth.

Runs the real support-geometry bootstrap machinery across scalar KDE bandwidth
multipliers for every synthetic scenario and sample size. This is a diagnostic gate:
inspect the heatmap before implementing any adaptive KDE candidate.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/bandwidth_sensitivity.py
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
    compute_support_geometry,
    normalize_shape,
    scipy_gaussian_kde_spec,
    valley_depth,
    valley_detection_detail,
)
from synthetic_scenarios import SAMPLE_SIZES, SCENARIOS, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
PLOT_DIR = RUN_DIR / "plots"
TABLE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

BANDWIDTH_SCALES = [0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
DEFAULT_N_SEEDS = 5
DEFAULT_N_RESAMPLE = 80
WT_POOL_N = 3000
SIG_ALPHA = 0.05
_WT_POOL = None


def _init_worker(wt_pool: np.ndarray) -> None:
    global _WT_POOL
    _WT_POOL = wt_pool


def _component_summary(detail: dict) -> tuple[int, str]:
    masses = np.asarray(detail.get("component_mass_fracs", []), dtype=float)
    if masses.size == 0:
        return 0, ""
    return int(detail.get("n_components_at_split", 0)), ";".join(f"{m:.4f}" for m in masses)


def _run_one_task(task: tuple[int, int, int, float, int]) -> dict:
    scen_idx, n, seed, bw_scale, n_resample = task
    scen = SCENARIOS[scen_idx]
    wt_pool = _WT_POOL
    if wt_pool is None:
        wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))

    pts_rng = np.random.default_rng(1000 + 100 * seed + n)
    pts_raw = scen.generator(n, pts_rng)
    pts_norm = normalize_shape(pts_raw)
    kde = scipy_gaussian_kde_spec(bw_scale=bw_scale)
    bundle = compute_support_geometry(
        pts_raw,
        wt_pool,
        n_resample=n_resample,
        rng=np.random.default_rng(7000 + seed),
        statistics={"valley_depth": valley_depth},
        kde=kde,
    )
    vd = bundle.results["valley_depth"]
    detail = valley_detection_detail(pts_norm, kde=kde)
    n_comp, mass_fracs = _component_summary(detail)
    return {
        "scenario": scen.name,
        "expected_support": scen.expected_support,
        "sample_size": n,
        "seed": seed,
        "bandwidth_scale": bw_scale,
        "valley_depth": vd.stat,
        "valley_pvalue": vd.pvalue,
        "valley_percentile": vd.percentile,
        "valley_significant": vd.pvalue < SIG_ALPHA,
        "n_components_at_split": n_comp,
        "component_mass_fracs": mass_fracs,
    }


def _write_partial(rows: list[dict], partial_out: Path) -> None:
    pd.DataFrame(rows).to_csv(partial_out, index=False)


def run_sweep(n_seeds: int, n_resample: int, n_workers: int) -> pd.DataFrame:
    wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))
    partial_out = TABLE_DIR / "bandwidth_sensitivity.partial.csv"
    partial_out.unlink(missing_ok=True)
    rows = []

    tasks = [
        (scen_idx, n, seed, bw_scale, n_resample)
        for scen_idx in range(len(SCENARIOS))
        for n in SAMPLE_SIZES
        for seed in range(n_seeds)
        for bw_scale in BANDWIDTH_SCALES
    ]
    total = len(tasks)
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        _init_worker(wt_pool)
        iterator = map(_run_one_task, tasks)
        for done, row in enumerate(iterator, start=1):
            rows.append(row)
            if done % 25 == 0 or done == total:
                _write_partial(rows, partial_out)
                print(f"  completed {done}/{total}  partial={partial_out}", flush=True)
    else:
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(wt_pool,)) as pool:
            for done, row in enumerate(pool.imap_unordered(_run_one_task, tasks, chunksize=4), start=1):
                rows.append(row)
                if done % 25 == 0 or done == total:
                    _write_partial(rows, partial_out)
                    print(f"  completed {done}/{total}  partial={partial_out}", flush=True)

    return pd.DataFrame(rows)


def make_heatmap(df: pd.DataFrame) -> Path:
    summary = (
        df.groupby(["scenario", "expected_support", "bandwidth_scale"], as_index=False)
        .agg(significant_rate=("valley_significant", "mean"),
             median_pvalue=("valley_pvalue", "median"),
             median_valley_depth=("valley_depth", "median"))
    )
    order = [s.name for s in SCENARIOS]
    pivot = (
        summary.pivot(index="scenario", columns="bandwidth_scale", values="significant_rate")
        .reindex(order)
    )
    expected = {s.name: s.expected_support for s in SCENARIOS}

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    labels = [f"{name} ({expected[name][0]})" for name in pivot.index]
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Scott bandwidth multiplier")
    ax.set_title(
        "valley_depth bandwidth sensitivity\n"
        f"cell = fraction significant at p<{SIG_ALPHA}; c=connected, d=discrete",
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
    out = PLOT_DIR / "bandwidth_sensitivity.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)

    summary.to_csv(TABLE_DIR / "bandwidth_sensitivity_summary.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--n-resample", type=int, default=DEFAULT_N_RESAMPLE)
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    print("Bandwidth-sensitivity diagnostic")
    print(f"  scenarios={len(SCENARIOS)} sample_sizes={SAMPLE_SIZES}")
    print(f"  bandwidth_scales={BANDWIDTH_SCALES}")
    print(f"  n_seeds={args.n_seeds} n_resample={args.n_resample} n_workers={args.n_workers}")

    df = run_sweep(n_seeds=args.n_seeds, n_resample=args.n_resample, n_workers=args.n_workers)
    csv_out = TABLE_DIR / "bandwidth_sensitivity.csv"
    df.to_csv(csv_out, index=False)
    plot_out = make_heatmap(df)
    print(f"Saved: {csv_out}")
    print(f"Saved: {plot_out}")


if __name__ == "__main__":
    main()
