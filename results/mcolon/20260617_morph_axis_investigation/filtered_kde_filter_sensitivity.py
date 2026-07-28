"""
Benchmark KDE prefilters for high-resolution component detection.

This quantifies a few principled filtering choices:
  - no filter
  - KDE HDR island filter: remove points landing in tiny pilot-KDE islands
  - point pilot-density filter: remove the lowest pilot-density points

Then recompute a sensitive KDE and count major HDR components.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from filtered_kde_visual_diagnostic import (  # noqa: E402
    DensitySpec,
    FilterSpec,
    PanelSpec,
    evaluate_panel,
)

TABLE_DIR = RUN_DIR / "tables" / "support_method_diagnostics"
PLOT_DIR = RUN_DIR / "plots" / "support_method_diagnostics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def make_specs() -> list[PanelSpec]:
    specs: list[PanelSpec] = []
    density_specs = [
        DensitySpec("scott_x0.45", "scott", 0.45),
        DensitySpec("scott_x0.55", "scott", 0.55),
        DensitySpec("scott_x0.65", "scott", 0.65),
    ]
    filters: list[FilterSpec | None] = [None]
    for pilot_bw in (0.45, 0.55, 0.65):
        for mass_level in (0.90, 0.95, 0.98):
            for min_mass in (0.01, 0.03, 0.05, 0.08):
                filters.append(FilterSpec(
                    f"kdemass_bw{pilot_bw:g}_hdr{mass_level:g}_drop{min_mass:g}",
                    kind="kde_mass",
                    k=5,
                    eps_mult=1.0,
                    radius_mode="global",
                    min_component_frac=0.08,
                    pilot_bw_scale=pilot_bw,
                    pilot_mass_level=mass_level,
                    min_kde_component_mass=min_mass,
                ))
    for pilot_bw in (0.45, 0.55, 0.65):
        for q in (0.025, 0.05, 0.10):
            filters.append(FilterSpec(
                f"pointdens_bw{pilot_bw:g}_q{q:g}",
                kind="point_density",
                k=5,
                eps_mult=1.0,
                radius_mode="global",
                min_component_frac=0.08,
                pilot_bw_scale=pilot_bw,
                min_point_density_quantile=q,
            ))
    for filt in filters:
        for dens in density_specs:
            filter_name = "none" if filt is None else filt.name
            specs.append(PanelSpec(
                f"{filter_name}__{dens.name}",
                filt,
                dens,
                min_density_mass=0.08,
            ))
    return specs


def eval_one(args_tuple: tuple[str, int, int, int, list[PanelSpec]]) -> pd.DataFrame:
    case, seed, sample_size, grid_size, specs = args_tuple
    rows, _ = evaluate_panel(case, seed, sample_size, grid_size, specs)
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["panel", "density", "filter", "min_density_mass"]
    for key_vals, sub in raw.groupby(keys, dropna=False):
        row = dict(zip(keys, key_vals, strict=True))
        by_case = {}
        for case, csub in sub.groupby("case"):
            by_case[f"{case}_ge2"] = float(np.mean(csub["n_major_components"] >= 2))
            by_case[f"{case}_ge3"] = float(np.mean(csub["n_major_components"] >= 3))
            by_case[f"{case}_mean"] = float(np.mean(csub["n_major_components"]))
            by_case[f"{case}_removed_mean"] = float(np.mean(csub["n_removed"]))
        false_cases = ["spiral", "crescent", "outliers", "wt_null"]
        false_ge2 = sum(by_case.get(f"{case}_ge2", 0.0) for case in false_cases)
        three_ge3 = by_case.get("three_discrete_ge3", 0.0)
        three_ge2 = by_case.get("three_discrete_ge2", 0.0)
        row.update({
            "three_ge3": three_ge3,
            "three_ge2": three_ge2,
            "false_ge2_sum": false_ge2,
            "priority_score": 3.0 * three_ge3 + three_ge2 - false_ge2,
            **by_case,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["priority_score", "three_ge3", "false_ge2_sum", "spiral_ge2", "wt_null_ge2"],
        ascending=[False, False, True, True, True],
    )


def make_plot(scores: pd.DataFrame, prefix: str) -> Path:
    top = scores.head(30).copy()
    labels = [f"{r.density}\n{str(r['filter'])[:24]}" for _, r in top.iterrows()]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(top)), 5.6))
    ax.bar(x - 0.2, top["three_ge3"], width=0.2, label="three >=3", color="#1B9E77")
    ax.bar(x, top.get("spiral_ge2", 0), width=0.2, label="spiral >=2", color="#D95F02")
    ax.bar(x + 0.2, top.get("outliers_ge2", 0), width=0.2, label="outliers >=2", color="#7570B3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Filtered KDE sensitivity")
    fig.tight_layout()
    out = PLOT_DIR / f"{prefix}_top_filters.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--cases", nargs="*", default=["three_discrete", "spiral", "crescent", "outliers", "wt_null"])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--output-prefix", default="filtered_kde_filter_sensitivity")
    args = parser.parse_args()

    specs = make_specs()
    tasks = [
        (case, seed, args.sample_size, args.grid_size, specs)
        for case in args.cases
        for seed in range(args.n_seeds)
    ]
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            dfs = list(ex.map(eval_one, tasks))
    else:
        dfs = []
        for i, task in enumerate(tasks, start=1):
            dfs.append(eval_one(task))
            print(f"  completed {i}/{len(tasks)} case-seeds", flush=True)

    raw = pd.concat(dfs, ignore_index=True)
    scores = summarize(raw)
    raw_path = TABLE_DIR / f"{args.output_prefix}_raw.csv"
    score_path = TABLE_DIR / f"{args.output_prefix}_scores.csv"
    raw.to_csv(raw_path, index=False)
    scores.to_csv(score_path, index=False)
    plot_path = make_plot(scores, args.output_prefix)
    print(f"Saved: {raw_path}")
    print(f"Saved: {score_path}")
    print(f"Saved: {plot_path}")
    print(scores.head(35).to_string(index=False))


if __name__ == "__main__":
    main()
