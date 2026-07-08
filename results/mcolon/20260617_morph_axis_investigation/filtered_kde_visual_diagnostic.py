"""
Visual diagnostic for outlier-filtered, high-resolution KDE component detection.

Goal:
  keep a fine KDE bandwidth so real discrete modes can separate, but remove tiny
  unsupported point-cloud islands before density estimation so outliers do not
  become modes.

Outputs:
  plots/support_method_diagnostics/<prefix>_panel.png
  tables/support_method_diagnostics/<prefix>_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from support_geometry import DEFAULT_HDR_MASS_LEVELS, _kde_grid, knn_adaptive_kde_spec, normalize_shape, scipy_gaussian_kde_spec  # noqa: E402
from synthetic_scenarios import SCENARIOS_BY_NAME, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "support_method_diagnostics"
PLOT_DIR = RUN_DIR / "plots" / "support_method_diagnostics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FilterSpec:
    name: str
    kind: str
    k: int
    eps_mult: float
    radius_mode: str
    min_component_frac: float
    pilot_bw_scale: float = 0.55
    pilot_mass_level: float = 0.95
    min_kde_component_mass: float = 0.03
    min_point_density_quantile: float = 0.05


@dataclass(frozen=True)
class DensitySpec:
    name: str
    estimator: str
    bw_scale: float
    k: int = 5


@dataclass(frozen=True)
class PanelSpec:
    name: str
    filter_spec: FilterSpec | None
    density_spec: DensitySpec
    min_density_mass: float


def scenario_points(case: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + 100 * seed + n)
    if case == "wt_null":
        pts = wt_reference(n, rng)
    else:
        pts = SCENARIOS_BY_NAME[case].generator(n, rng)
    return normalize_shape(pts)


def _fmt(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals)
    if vals.size == 0:
        return ""
    return ";".join(f"{float(v):.{digits}f}" for v in vals)


def local_scales(points: np.ndarray, k: int) -> np.ndarray:
    d = squareform(pdist(points))
    k_eff = max(1, min(k, len(points) - 1))
    sigma = np.sort(d, axis=1)[:, k_eff]
    positive = sigma[sigma > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.where(sigma > 1e-12, sigma, fallback)


def _point_grid_labels(points: np.ndarray, xx: np.ndarray, yy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    xs = xx[0, :]
    ys = yy[:, 0]
    x_idx = np.clip(np.searchsorted(xs, points[:, 0]), 1, len(xs) - 1)
    y_idx = np.clip(np.searchsorted(ys, points[:, 1]), 1, len(ys) - 1)
    x_left = x_idx - 1
    y_left = y_idx - 1
    xi = np.where(np.abs(xs[x_idx] - points[:, 0]) < np.abs(xs[x_left] - points[:, 0]), x_idx, x_left)
    yi = np.where(np.abs(ys[y_idx] - points[:, 1]) < np.abs(ys[y_left] - points[:, 1]), y_idx, y_left)
    return labels[yi, xi].astype(int)


def _hdr_threshold(density: np.ndarray, mass_level: float) -> float:
    flat = density.ravel()
    total = float(flat.sum())
    if total <= 0:
        return float("inf")
    ordered = np.sort(flat)[::-1]
    csum = np.cumsum(ordered) / total
    idx = min(int(np.searchsorted(csum, mass_level, side="left")), len(ordered) - 1)
    return float(ordered[idx])


def filter_points(points: np.ndarray, spec: FilterSpec | None, grid_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(points)
    if spec is None:
        keep = np.ones(n, dtype=bool)
        return keep, np.zeros(n, dtype=int), {
            "filter": "none",
            "n_kept": n,
            "n_removed": 0,
            "filter_component_sizes": str(n),
        }

    if spec.kind == "kde_mass":
        xx, yy, density = _kde_grid(
            points,
            grid_size=grid_size,
            kde=scipy_gaussian_kde_spec(bw_scale=spec.pilot_bw_scale),
        )
        threshold = _hdr_threshold(density, spec.pilot_mass_level)
        labels, n_components = ndi_label(density >= threshold)
        total = float(density.sum())
        masses = np.array([
            float(density[labels == label].sum() / total)
            for label in range(1, n_components + 1)
        ])
        kept_components = np.flatnonzero(masses >= spec.min_kde_component_mass) + 1
        point_labels = _point_grid_labels(points, xx, yy, labels)
        keep = np.isin(point_labels, kept_components)
        return keep, point_labels, {
            "filter": spec.name,
            "n_kept": int(np.sum(keep)),
            "n_removed": int(np.sum(~keep)),
            "filter_component_sizes": _fmt(np.sort(masses)[::-1]),
            "filter_min_size": spec.min_kde_component_mass,
            "filter_pilot_bw": spec.pilot_bw_scale,
            "filter_pilot_mass_level": spec.pilot_mass_level,
        }

    if spec.kind == "point_density":
        xx, yy, density = _kde_grid(
            points,
            grid_size=grid_size,
            kde=scipy_gaussian_kde_spec(bw_scale=spec.pilot_bw_scale),
        )
        # Reuse nearest-grid density as a transparent diagnostic approximation.
        labels = np.arange(density.size, dtype=int).reshape(density.shape) + 1
        point_cells = _point_grid_labels(points, xx, yy, labels) - 1
        point_density = density.ravel()[np.clip(point_cells, 0, density.size - 1)]
        floor = float(np.quantile(point_density, spec.min_point_density_quantile))
        keep = point_density >= floor
        return keep, np.zeros(n, dtype=int), {
            "filter": spec.name,
            "n_kept": int(np.sum(keep)),
            "n_removed": int(np.sum(~keep)),
            "filter_component_sizes": _fmt(np.sort(point_density)[::-1]),
            "filter_min_size": spec.min_point_density_quantile,
            "filter_pilot_bw": spec.pilot_bw_scale,
            "filter_pilot_mass_level": np.nan,
        }

    if spec.kind != "point_graph":
        raise ValueError(f"Unknown filter kind: {spec.kind}")

    d = squareform(pdist(points))
    sigma = local_scales(points, spec.k)
    if spec.radius_mode == "global":
        radius = spec.eps_mult * float(np.median(sigma))
        adj = (d <= radius) & (d > 0)
    elif spec.radius_mode == "local_max":
        radius = spec.eps_mult * np.maximum(sigma[:, None], sigma[None, :])
        adj = (d <= radius) & (d > 0)
    elif spec.radius_mode == "local_mean":
        radius = spec.eps_mult * 0.5 * (sigma[:, None] + sigma[None, :])
        adj = (d <= radius) & (d > 0)
    elif spec.radius_mode == "local_min":
        radius = spec.eps_mult * np.minimum(sigma[:, None], sigma[None, :])
        adj = (d <= radius) & (d > 0)
    else:
        raise ValueError(f"Unknown radius_mode: {spec.radius_mode}")

    n_components, labels = connected_components(csr_matrix(adj), directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    min_size = int(np.ceil(spec.min_component_frac * n))
    keep_components = np.flatnonzero(sizes >= min_size)
    keep = np.isin(labels, keep_components)
    return keep, labels, {
        "filter": spec.name,
        "n_kept": int(np.sum(keep)),
        "n_removed": int(np.sum(~keep)),
        "filter_component_sizes": ";".join(str(int(v)) for v in np.sort(sizes)[::-1]),
        "filter_min_size": min_size,
        "filter_pilot_bw": np.nan,
        "filter_pilot_mass_level": np.nan,
    }


def kde_for_spec(spec: DensitySpec):
    if spec.estimator == "scott":
        return scipy_gaussian_kde_spec(bw_scale=spec.bw_scale)
    if spec.estimator == "knn":
        return knn_adaptive_kde_spec(k=spec.k, bw_scale=spec.bw_scale)
    raise ValueError(f"Unknown density estimator: {spec.estimator}")


def hdr_components(density: np.ndarray, min_mass: float) -> dict:
    total = float(density.sum())
    peak = float(density.max())
    best = {
        "n_components": 0,
        "n_major_components": 0,
        "mass_level": np.nan,
        "threshold": np.nan,
        "threshold_peak_frac": np.nan,
        "component_masses": np.array([], dtype=float),
        "labels": np.zeros_like(density, dtype=int),
    }
    if total <= 0 or peak <= 0:
        return best
    ordered = np.sort(density.ravel())[::-1]
    csum = np.cumsum(ordered) / total
    for mass_level in DEFAULT_HDR_MASS_LEVELS:
        idx = min(int(np.searchsorted(csum, mass_level, side="left")), len(ordered) - 1)
        threshold = float(ordered[idx])
        labels, n_labels = ndi_label(density >= threshold)
        masses = np.array([
            float(density[labels == label].sum() / total)
            for label in range(1, n_labels + 1)
        ])
        major = masses[masses >= min_mass]
        if len(major) > best["n_major_components"]:
            best = {
                "n_components": int(n_labels),
                "n_major_components": int(len(major)),
                "mass_level": float(mass_level),
                "threshold": threshold,
                "threshold_peak_frac": float(threshold / peak),
                "component_masses": np.sort(masses)[::-1],
                "labels": labels,
            }
    return best


def panel_specs() -> list[PanelSpec]:
    kde_mass_filter = FilterSpec(
        "pilotHDR95_dropMass<3pct",
        kind="kde_mass",
        k=5,
        eps_mult=1.0,
        radius_mode="global",
        min_component_frac=0.08,
        pilot_bw_scale=0.55,
        pilot_mass_level=0.95,
        min_kde_component_mass=0.03,
    )
    kde_mass_filter_strict = FilterSpec(
        "pilotHDR90_dropMass<5pct",
        kind="kde_mass",
        k=5,
        eps_mult=1.0,
        radius_mode="global",
        min_component_frac=0.08,
        pilot_bw_scale=0.55,
        pilot_mass_level=0.90,
        min_kde_component_mass=0.05,
    )
    point_density_filter = FilterSpec(
        "pilotPointDensity_drop5pct",
        kind="point_density",
        k=5,
        eps_mult=1.0,
        radius_mode="global",
        min_component_frac=0.08,
        pilot_bw_scale=0.55,
        min_point_density_quantile=0.05,
    )
    graph_filter = FilterSpec("graph_localmax_k5_e1.05_min8pct", kind="point_graph", k=5, eps_mult=1.05, radius_mode="local_max", min_component_frac=0.08)
    return [
        PanelSpec("raw_scott0.55", None, DensitySpec("scott_x0.55", "scott", 0.55), 0.08),
        PanelSpec("kdemass_scott0.55", kde_mass_filter, DensitySpec("scott_x0.55", "scott", 0.55), 0.08),
        PanelSpec("kdemass_scott0.45", kde_mass_filter, DensitySpec("scott_x0.45", "scott", 0.45), 0.08),
        PanelSpec("kdemass_strict_scott0.55", kde_mass_filter_strict, DensitySpec("scott_x0.55", "scott", 0.55), 0.08),
        PanelSpec("pointdens_scott0.55", point_density_filter, DensitySpec("scott_x0.55", "scott", 0.55), 0.08),
        PanelSpec("graph_scott0.55", graph_filter, DensitySpec("scott_x0.55", "scott", 0.55), 0.08),
    ]


def evaluate_panel(case: str, seed: int, n: int, grid_size: int, specs: list[PanelSpec]) -> tuple[list[dict], dict]:
    points = scenario_points(case, n, seed)
    rows = []
    plot_data = {"case": case, "seed": seed, "points": points, "panels": []}
    for spec in specs:
        keep, filter_labels, filter_info = filter_points(points, spec.filter_spec, grid_size)
        kept_points = points[keep]
        if len(kept_points) < 3:
            continue
        xx, yy, density = _kde_grid(kept_points, grid_size=grid_size, kde=kde_for_spec(spec.density_spec))
        comp = hdr_components(density, spec.min_density_mass)
        row = {
            "case": case,
            "seed": seed,
            "n": n,
            "panel": spec.name,
            "density": spec.density_spec.name,
            "min_density_mass": spec.min_density_mass,
            "n_density_components": comp["n_components"],
            "n_major_components": comp["n_major_components"],
            "hdr_mass_level": comp["mass_level"],
            "threshold_peak_frac": comp["threshold_peak_frac"],
            "component_masses": _fmt(comp["component_masses"]),
            **filter_info,
        }
        rows.append(row)
        plot_data["panels"].append({
            "spec": spec,
            "keep": keep,
            "filter_labels": filter_labels,
            "xx": xx,
            "yy": yy,
            "density": density,
            "components": comp,
            "row": row,
        })
    return rows, plot_data


def make_plot(plot_data_by_case: list[dict], prefix: str) -> Path:
    n_rows = len(plot_data_by_case)
    n_cols = len(plot_data_by_case[0]["panels"]) if plot_data_by_case else 0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.5 * n_rows), squeeze=False)
    colors = ["#D95F02", "#1B9E77", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]
    for r, pdata in enumerate(plot_data_by_case):
        points = pdata["points"]
        for c, panel in enumerate(pdata["panels"]):
            ax = axes[r, c]
            xx = panel["xx"]
            yy = panel["yy"]
            density = panel["density"]
            comp = panel["components"]
            labels = comp["labels"]
            ax.contourf(xx, yy, density, levels=16, cmap="Blues")
            if np.isfinite(comp["threshold"]):
                major_ids = []
                masses = comp["component_masses"]
                for label_id in range(1, int(labels.max()) + 1):
                    mass = float(density[labels == label_id].sum() / density.sum())
                    if mass >= panel["spec"].min_density_mass:
                        major_ids.append(label_id)
                for j, label_id in enumerate(major_ids):
                    ax.contour(
                        xx,
                        yy,
                        labels == label_id,
                        levels=[0.5],
                        colors=[colors[j % len(colors)]],
                        linewidths=1.8,
                    )
            keep = panel["keep"]
            ax.scatter(points[keep, 0], points[keep, 1], s=16, c="#111111", edgecolors="white", linewidths=0.3)
            if np.any(~keep):
                ax.scatter(points[~keep, 0], points[~keep, 1], s=34, c="#D7191C", marker="x", linewidths=1.2)
            row = panel["row"]
            if c == 0:
                ax.set_ylabel(f"{pdata['case']}\nseed {pdata['seed']}", fontsize=9)
            ax.set_title(
                f"{panel['spec'].name}\nkept {row['n_kept']}/{row['n']}  major={row['n_major_components']}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    out = PLOT_DIR / f"{prefix}_panel.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--cases", nargs="*", default=["three_discrete", "spiral", "outliers", "wt_null"])
    parser.add_argument("--output-prefix", default="filtered_kde_visual_diagnostic")
    args = parser.parse_args()

    specs = panel_specs()
    all_rows = []
    plot_data = []
    for case in args.cases:
        rows, pdata = evaluate_panel(case, args.seed, args.sample_size, args.grid_size, specs)
        all_rows.extend(rows)
        plot_data.append(pdata)

    summary = pd.DataFrame(all_rows)
    summary_path = TABLE_DIR / f"{args.output_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_path = make_plot(plot_data, args.output_prefix)
    print(f"Saved: {summary_path}")
    print(f"Saved: {plot_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
