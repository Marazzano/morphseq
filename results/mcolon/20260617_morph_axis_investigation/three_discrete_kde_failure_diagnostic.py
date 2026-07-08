"""
Focused diagnostic for the three_discrete KDE/component failure.

Question: does the HDR/component method fail because the component-count logic is
bad, or because the KDE density field has already merged the three modes before
components are counted?

Outputs:
  tables/support_method_diagnostics/three_discrete_kde_failure_*.csv
  plots/support_method_diagnostics/three_discrete_kde_failure_*.png
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
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from support_geometry import (  # noqa: E402
    DEFAULT_HDR_MASS_LEVELS,
    GRID_SIZE,
    MIN_COMPONENT_MASS_FRAC,
    KDESpec,
    _kde_grid,
    knn_adaptive_kde_spec,
    normalize_shape,
    scipy_gaussian_kde_spec,
    valley_detection_detail,
)
from synthetic_scenarios import SCENARIOS_BY_NAME  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "support_method_diagnostics"
PLOT_DIR = RUN_DIR / "plots" / "support_method_diagnostics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PEAK_FRACS = np.round(np.r_[np.linspace(0.97, 0.70, 7), np.linspace(0.65, 0.05, 13)], 3)


def _fmt(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return ""
    return ";".join(f"{v:.{digits}f}" for v in vals)


def kde_specs() -> list[tuple[str, KDESpec]]:
    specs = [
        (f"scott_x{scale:g}", scipy_gaussian_kde_spec(bw_scale=scale))
        for scale in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.20)
    ]
    specs.extend([
        ("knn_k5_bw0.4", knn_adaptive_kde_spec(k=5, bw_scale=0.4)),
        ("knn_k5_bw0.6", knn_adaptive_kde_spec(k=5, bw_scale=0.6)),
        ("knn_k10_bw0.4", knn_adaptive_kde_spec(k=10, bw_scale=0.4)),
        ("knn_k10_bw0.6", knn_adaptive_kde_spec(k=10, bw_scale=0.6)),
    ])
    return specs


def _component_masses(density: np.ndarray, mask: np.ndarray) -> tuple[int, int, np.ndarray]:
    labels, n_labels = ndi_label(mask)
    total = float(np.sum(density))
    if total <= 0 or n_labels == 0:
        return int(n_labels), 0, np.array([], dtype=float)
    masses = np.array([
        density[labels == label].sum() / total
        for label in range(1, n_labels + 1)
    ], dtype=float)
    major = masses[masses >= MIN_COMPONENT_MASS_FRAC]
    return int(n_labels), int(major.size), np.sort(masses)[::-1]


def _hdr_thresholds(density: np.ndarray, mass_levels: np.ndarray) -> list[tuple[float, float]]:
    flat = np.asarray(density, dtype=float).ravel()
    total = float(flat.sum())
    if total <= 0:
        return [(float(m), np.nan) for m in mass_levels]
    ordered = np.sort(flat)[::-1]
    csum = np.cumsum(ordered) / np.sum(ordered)
    out = []
    for mass in mass_levels:
        idx = int(np.searchsorted(csum, mass, side="left"))
        idx = min(idx, len(ordered) - 1)
        out.append((float(mass), float(ordered[idx])))
    return out


def density_component_tables(points: np.ndarray, kde_name: str, kde: KDESpec, grid_size: int) -> tuple[list[dict], list[dict], dict]:
    xx, yy, density = _kde_grid(points, grid_size=grid_size, kde=kde)
    peak = float(np.max(density))
    total = float(np.sum(density))
    peak_rows = []
    for frac in PEAK_FRACS:
        threshold = float(frac * peak)
        n_comp, n_major, masses = _component_masses(density, density >= threshold)
        peak_rows.append({
            "kde": kde_name,
            "grid_size": grid_size,
            "peak_frac": float(frac),
            "density_threshold": threshold,
            "n_components": n_comp,
            "n_major_components": n_major,
            "component_masses": _fmt(masses),
        })

    hdr_rows = []
    for mass, threshold in _hdr_thresholds(density, DEFAULT_HDR_MASS_LEVELS):
        n_comp, n_major, masses = _component_masses(density, density >= threshold)
        hdr_rows.append({
            "kde": kde_name,
            "grid_size": grid_size,
            "mass_level": mass,
            "density_threshold": threshold,
            "threshold_peak_frac": float(threshold / peak) if peak > 0 else np.nan,
            "n_components": n_comp,
            "n_major_components": n_major,
            "component_masses": _fmt(masses),
        })

    valley = valley_detection_detail(points, grid_size=grid_size, kde=kde)
    summary = {
        "kde": kde_name,
        "grid_size": grid_size,
        "peak": peak,
        "total_density": total,
        "max_peak_major_components": max(r["n_major_components"] for r in peak_rows),
        "max_hdr_major_components": max(r["n_major_components"] for r in hdr_rows),
        "n_peak_levels_with_3_major": sum(r["n_major_components"] >= 3 for r in peak_rows),
        "n_hdr_levels_with_3_major": sum(r["n_major_components"] >= 3 for r in hdr_rows),
        "valley_frac": float(valley["valley_frac"]),
        "valley_n_components": int(valley["n_components_at_split"]),
        "valley_component_masses": _fmt(valley["component_mass_fracs"]),
    }
    return peak_rows, hdr_rows, summary


def radius_component_table(points: np.ndarray) -> list[dict]:
    dmat = squareform(pdist(points))
    n = len(points)
    sorted_d = np.sort(dmat, axis=1)
    kth = sorted_d[:, max(1, min(5, n - 1))]
    scale = float(np.median(kth[kth > 1e-12]))
    rows = []
    for ratio in np.round(np.linspace(0.5, 3.0, 26), 2):
        adjacency = (dmat <= ratio * scale) & (dmat > 0)
        n_comp, labels = connected_components(adjacency, directed=False)
        sizes = np.bincount(labels, minlength=n_comp)
        rows.append({
            "radius_over_median_knn5": float(ratio),
            "radius": float(ratio * scale),
            "n_components": int(n_comp),
            "component_sizes": _fmt(np.sort(sizes)[::-1], 0),
        })
    return rows


def make_density_panel(points: np.ndarray, summaries: pd.DataFrame, grid_size: int, out_prefix: str) -> Path:
    selected = [
        "scott_x0.15", "scott_x0.2", "scott_x0.3", "scott_x0.5",
        "scott_x0.8", "scott_x1", "knn_k10_bw0.4", "knn_k10_bw0.6",
    ]
    spec_map = dict(kde_specs())
    n_cols = 4
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax, name in zip(axes.ravel(), selected):
        kde = spec_map[name]
        xx, yy, density = _kde_grid(points, grid_size=grid_size, kde=kde)
        peak = float(np.max(density))
        ax.contourf(xx, yy, density, levels=16, cmap="Blues")
        ax.contour(xx, yy, density, levels=[0.2 * peak, 0.5 * peak, 0.8 * peak],
                   colors=["#B35806", "#008837", "#7B3294"], linewidths=1.1)
        ax.scatter(points[:, 0], points[:, 1], s=18, c="#111111", edgecolors="white", linewidths=0.35)
        row = summaries[(summaries["kde"] == name) & (summaries["grid_size"] == grid_size)].iloc[0]
        ax.set_title(
            f"{name}\npeak max={int(row.max_peak_major_components)} "
            f"HDR max={int(row.max_hdr_major_components)} valley={row.valley_frac:.2f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    out = PLOT_DIR / f"{out_prefix}_density_panel_grid{grid_size}.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-sizes", nargs="*", type=int, default=[30, 60, 120])
    parser.add_argument("--output-prefix", default="three_discrete_kde_failure")
    args = parser.parse_args()

    scenario = SCENARIOS_BY_NAME["three_discrete"]
    rng = np.random.default_rng(1000 + 100 * args.seed + args.sample_size)
    raw = scenario.generator(args.sample_size, rng)
    points = normalize_shape(raw)

    peak_rows = []
    hdr_rows = []
    summary_rows = []
    for grid_size in args.grid_sizes:
        for name, kde in kde_specs():
            p_rows, h_rows, summary = density_component_tables(points, name, kde, grid_size)
            peak_rows.extend(p_rows)
            hdr_rows.extend(h_rows)
            summary_rows.append(summary)

    peak_df = pd.DataFrame(peak_rows)
    hdr_df = pd.DataFrame(hdr_rows)
    summary_df = pd.DataFrame(summary_rows)
    radius_df = pd.DataFrame(radius_component_table(points))

    prefix = args.output_prefix
    peak_df.to_csv(TABLE_DIR / f"{prefix}_peak_component_sweep.csv", index=False)
    hdr_df.to_csv(TABLE_DIR / f"{prefix}_hdr_component_sweep.csv", index=False)
    summary_df.to_csv(TABLE_DIR / f"{prefix}_summary.csv", index=False)
    radius_df.to_csv(TABLE_DIR / f"{prefix}_radius_graph_components.csv", index=False)
    plot_path = make_density_panel(points, summary_df, max(args.grid_sizes), prefix)

    print(f"Saved: {TABLE_DIR / f'{prefix}_peak_component_sweep.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_hdr_component_sweep.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_summary.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_radius_graph_components.csv'}")
    print(f"Saved: {plot_path}")
    print(summary_df.sort_values(["grid_size", "kde"]).to_string(index=False))


if __name__ == "__main__":
    main()
