"""
Focused component-count sensitivity for support discreteness.

This isolates the central failure mode:
  - three_discrete should show stable multiple major components.
  - spiral should remain one major component.
  - WT-null draws should remain one major component.

The sweep varies KDE bandwidth and the minimum component mass used to filter tiny
outlier-induced components.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from support_geometry import (  # noqa: E402
    DEFAULT_HDR_MASS_LEVELS,
    GRID_SIZE,
    KDESpec,
    _kde_grid,
    knn_adaptive_kde_spec,
    normalize_shape,
    scipy_gaussian_kde_spec,
)
from synthetic_scenarios import SCENARIOS_BY_NAME, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "support_method_diagnostics"
PLOT_DIR = RUN_DIR / "plots" / "support_method_diagnostics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def kde_specs() -> list[tuple[str, KDESpec]]:
    specs = [
        (f"scott_x{scale:g}", scipy_gaussian_kde_spec(bw_scale=scale))
        for scale in (0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.80, 1.00)
    ]
    for k in (3, 5, 8, 10):
        for bw in (0.25, 0.35, 0.45, 0.55, 0.65):
            specs.append((f"knn_k{k}_bw{bw:g}", knn_adaptive_kde_spec(k=k, bw_scale=bw)))
    return specs


def _fmt(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return ""
    return ";".join(f"{v:.{digits}f}" for v in vals)


def _hdr_thresholds(density: np.ndarray) -> list[tuple[float, float]]:
    flat = np.asarray(density, dtype=float).ravel()
    total = float(flat.sum())
    if total <= 0:
        return [(float(m), np.nan) for m in DEFAULT_HDR_MASS_LEVELS]
    ordered = np.sort(flat)[::-1]
    csum = np.cumsum(ordered) / total
    rows = []
    for mass in DEFAULT_HDR_MASS_LEVELS:
        idx = int(np.searchsorted(csum, mass, side="left"))
        idx = min(idx, len(ordered) - 1)
        rows.append((float(mass), float(ordered[idx])))
    return rows


def _component_masses(density: np.ndarray, threshold: float) -> np.ndarray:
    labels, n_labels = ndi_label(density >= threshold)
    total = float(density.sum())
    if total <= 0 or n_labels == 0:
        return np.array([], dtype=float)
    masses = np.array([
        density[labels == label].sum() / total
        for label in range(1, n_labels + 1)
    ], dtype=float)
    return np.sort(masses)[::-1]


def profile_components(points: np.ndarray, kde_name: str, kde: KDESpec, grid_size: int, min_masses: list[float]) -> list[dict]:
    _, _, density = _kde_grid(points, grid_size=grid_size, kde=kde)
    peak = float(density.max())
    rows = []
    for mass_level, threshold in _hdr_thresholds(density):
        masses = _component_masses(density, threshold)
        for min_mass in min_masses:
            major = masses[masses >= min_mass]
            rows.append({
                "kde": kde_name,
                "grid_size": grid_size,
                "mass_level": mass_level,
                "threshold_peak_frac": float(threshold / peak) if peak > 0 else np.nan,
                "min_component_mass": float(min_mass),
                "n_components": int(masses.size),
                "n_major_components": int(major.size),
                "largest_component_mass": float(masses[0]) if masses.size else 0.0,
                "second_component_mass": float(masses[1]) if masses.size > 1 else 0.0,
                "third_component_mass": float(masses[2]) if masses.size > 2 else 0.0,
                "component_masses": _fmt(masses),
            })
    return rows


def scenario_points(case: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + 100 * seed + n)
    if case == "wt_null":
        pts = wt_reference(n, rng)
    else:
        pts = SCENARIOS_BY_NAME[case].generator(n, rng)
    return normalize_shape(pts)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["case", "kde", "grid_size", "min_component_mass", "seed"]):
        case, kde, grid_size, min_mass, seed = keys
        counts = sub.sort_values("mass_level")["n_major_components"].to_numpy(dtype=int)
        rows.append({
            "case": case,
            "kde": kde,
            "grid_size": int(grid_size),
            "min_component_mass": float(min_mass),
            "seed": int(seed),
            "max_major_components": int(np.max(counts)),
            "levels_ge2": int(np.sum(counts >= 2)),
            "levels_ge3": int(np.sum(counts >= 3)),
            "count_path": ";".join(map(str, counts.tolist())),
        })
    seed_summary = pd.DataFrame(rows)
    agg = (
        seed_summary.groupby(["case", "kde", "grid_size", "min_component_mass"], as_index=False)
        .agg(
            mean_max_major=("max_major_components", "mean"),
            rate_ge2_any=("levels_ge2", lambda x: float(np.mean(np.asarray(x) > 0))),
            rate_ge3_any=("levels_ge3", lambda x: float(np.mean(np.asarray(x) > 0))),
            median_levels_ge2=("levels_ge2", "median"),
            median_levels_ge3=("levels_ge3", "median"),
        )
    )
    return seed_summary, agg


def score_methods(agg: pd.DataFrame) -> pd.DataFrame:
    idx = ["kde", "grid_size", "min_component_mass"]
    wide = agg.pivot_table(
        index=idx,
        columns="case",
        values=["rate_ge2_any", "rate_ge3_any", "median_levels_ge2", "median_levels_ge3"],
        fill_value=0.0,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for col in [
        "rate_ge3_any_three_discrete",
        "rate_ge2_any_spiral",
        "rate_ge2_any_wt_null",
        "median_levels_ge3_three_discrete",
    ]:
        if col not in wide:
            wide[col] = 0.0
    wide["selectivity_score"] = (
        wide["rate_ge3_any_three_discrete"]
        - wide["rate_ge2_any_spiral"]
        - wide["rate_ge2_any_wt_null"]
        + 0.05 * wide["median_levels_ge3_three_discrete"]
    )
    return wide.sort_values(
        ["selectivity_score", "rate_ge3_any_three_discrete", "rate_ge2_any_spiral", "rate_ge2_any_wt_null"],
        ascending=[False, False, True, True],
    )


def make_plot(score_df: pd.DataFrame, prefix: str) -> Path:
    top = score_df.head(25).copy()
    labels = [
        f"{r.kde}\nmin={r.min_component_mass:g}"
        for r in top.itertuples()
    ]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(max(9, 0.42 * len(top)), 5.0))
    ax.bar(x - 0.22, top["rate_ge3_any_three_discrete"], width=0.22, label="three >=3", color="#1B9E77")
    ax.bar(x, top["rate_ge2_any_spiral"], width=0.22, label="spiral >=2", color="#D95F02")
    ax.bar(x + 0.22, top["rate_ge2_any_wt_null"], width=0.22, label="WT >=2", color="#7570B3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("seed fraction")
    ax.set_title("Component-count selectivity: recover three_discrete, avoid spiral/WT")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = PLOT_DIR / f"{prefix}_top_selectivity.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--cases", nargs="*", default=["three_discrete", "spiral", "wt_null"])
    parser.add_argument("--min-component-masses", nargs="*", type=float, default=[0.01, 0.03, 0.05, 0.08, 0.10])
    parser.add_argument("--output-prefix", default="component_bandwidth_sensitivity")
    args = parser.parse_args()

    rows = []
    specs = kde_specs()
    total = len(args.cases) * args.n_seeds * len(specs)
    done = 0
    for case in args.cases:
        for seed in range(args.n_seeds):
            pts = scenario_points(case, args.sample_size, seed)
            for kde_name, kde in specs:
                rows.extend(profile_components(pts, kde_name, kde, args.grid_size, args.min_component_masses))
                for row in rows[-len(DEFAULT_HDR_MASS_LEVELS) * len(args.min_component_masses):]:
                    row.update({"case": case, "seed": seed, "n": args.sample_size})
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  completed {done}/{total}", flush=True)

    df = pd.DataFrame(rows)
    seed_summary, agg = summarize(df)
    scores = score_methods(agg)
    prefix = args.output_prefix
    df.to_csv(TABLE_DIR / f"{prefix}_profiles.csv", index=False)
    seed_summary.to_csv(TABLE_DIR / f"{prefix}_seed_summary.csv", index=False)
    agg.to_csv(TABLE_DIR / f"{prefix}_summary.csv", index=False)
    scores.to_csv(TABLE_DIR / f"{prefix}_scores.csv", index=False)
    plot_path = make_plot(scores, prefix)
    print(f"Saved: {TABLE_DIR / f'{prefix}_profiles.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_seed_summary.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_summary.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_scores.csv'}")
    print(f"Saved: {plot_path}")
    print(scores.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
