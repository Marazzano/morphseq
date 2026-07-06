"""
Search candidate component strategies for discrete-vs-continuum support.

This extends beyond isotropic KDE:
  - DBSCAN over local-scale-normalized eps grids.
  - radius and mutual-kNN graph component counts.
  - anisotropic local-PCA KDE, intended to smooth along curved manifolds while
    avoiding across-gap smoothing.

The target pattern is:
  three_discrete: >=3 major components
  spiral / crescent / outliers / WT null: <=1 major component
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
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.neighbors import NearestNeighbors

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from support_geometry import DEFAULT_HDR_MASS_LEVELS, GRID_SIZE, _kde_grid, normalize_shape, scipy_gaussian_kde_spec  # noqa: E402
from synthetic_scenarios import SCENARIOS_BY_NAME, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "support_method_diagnostics"
PLOT_DIR = RUN_DIR / "plots" / "support_method_diagnostics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals)
    if vals.size == 0:
        return ""
    if np.issubdtype(vals.dtype, np.integer):
        return ";".join(str(int(v)) for v in vals)
    return ";".join(f"{float(v):.{digits}f}" for v in vals)


def scenario_points(case: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + 100 * seed + n)
    if case == "wt_null":
        pts = wt_reference(n, rng)
    else:
        pts = SCENARIOS_BY_NAME[case].generator(n, rng)
    return normalize_shape(pts)


def local_scale(points: np.ndarray, k: int) -> np.ndarray:
    d = squareform(pdist(points))
    k_eff = max(1, min(k, len(points) - 1))
    sigma = np.sort(d, axis=1)[:, k_eff]
    positive = sigma[sigma > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.where(sigma > 1e-12, sigma, fallback)


def major_count_from_labels(labels: np.ndarray, n: int, min_mass: float) -> tuple[int, int, str]:
    valid = labels[labels >= 0]
    if valid.size == 0:
        return 0, 0, ""
    _, counts = np.unique(valid, return_counts=True)
    counts = np.sort(counts)[::-1]
    major = counts[counts >= np.ceil(min_mass * n)]
    return int(len(counts)), int(len(major)), _fmt(counts, 0)


def graph_major_count(adjacency: np.ndarray, min_mass: float) -> tuple[int, int, str]:
    n_comp, labels = connected_components(adjacency, directed=False)
    counts = np.bincount(labels, minlength=n_comp)
    counts = np.sort(counts)[::-1]
    major = counts[counts >= np.ceil(min_mass * len(labels))]
    return int(n_comp), int(len(major)), _fmt(counts, 0)


def dbscan_rows(points: np.ndarray, min_mass_values: list[float]) -> list[dict]:
    rows = []
    n = len(points)
    for k_scale in (3, 5, 8):
        sigma = local_scale(points, k_scale)
        base = float(np.median(sigma))
        for eps_mult in (0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.7, 2.0):
            for min_samples in (3, 4, 5):
                labels = DBSCAN(eps=eps_mult * base, min_samples=min_samples).fit_predict(points)
                noise_frac = float(np.mean(labels < 0))
                for min_mass in min_mass_values:
                    n_comp, n_major, sizes = major_count_from_labels(labels, n, min_mass)
                    rows.append({
                        "method": f"dbscan_k{k_scale}_eps{eps_mult:g}_min{min_samples}",
                        "family": "dbscan",
                        "min_component_mass": min_mass,
                        "n_components": n_comp,
                        "n_major_components": n_major,
                        "component_sizes": sizes,
                        "noise_frac": noise_frac,
                    })
    return rows


def optics_rows(points: np.ndarray, min_mass_values: list[float]) -> list[dict]:
    rows = []
    n = len(points)
    for min_samples in (3, 5, 8):
        try:
            opt = OPTICS(min_samples=min_samples, max_eps=np.inf).fit(points)
        except Exception:
            continue
        reach = opt.reachability_[np.isfinite(opt.reachability_)]
        if reach.size == 0:
            continue
        for q in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
            eps = float(np.quantile(reach, q))
            labels = cluster_optics_dbscan(
                reachability=opt.reachability_,
                core_distances=opt.core_distances_,
                ordering=opt.ordering_,
                eps=eps,
            )
            noise_frac = float(np.mean(labels < 0))
            for min_mass in min_mass_values:
                n_comp, n_major, sizes = major_count_from_labels(labels, n, min_mass)
                rows.append({
                    "method": f"optics_min{min_samples}_q{q:g}",
                    "family": "optics",
                    "min_component_mass": min_mass,
                    "n_components": n_comp,
                    "n_major_components": n_major,
                    "component_sizes": sizes,
                    "noise_frac": noise_frac,
                })
    return rows


def graph_rows(points: np.ndarray, min_mass_values: list[float]) -> list[dict]:
    rows = []
    n = len(points)
    d = squareform(pdist(points))
    for k in (2, 3, 4, 5, 6, 8, 10):
        order = np.argsort(d, axis=1)
        knn = np.zeros((n, n), dtype=bool)
        for i in range(n):
            knn[i, order[i, 1:min(k + 1, n)]] = True
        for mode, adj in (
            ("mutual", knn & knn.T),
            ("union", knn | knn.T),
        ):
            for min_mass in min_mass_values:
                n_comp, n_major, sizes = graph_major_count(adj, min_mass)
                rows.append({
                    "method": f"{mode}_knn{k}",
                    "family": "knn_graph",
                    "min_component_mass": min_mass,
                    "n_components": n_comp,
                    "n_major_components": n_major,
                    "component_sizes": sizes,
                    "noise_frac": 0.0,
                })
    sigma = local_scale(points, 5)
    base = float(np.median(sigma))
    for eps_mult in (0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.1):
        adj = (d <= eps_mult * base) & (d > 0)
        for min_mass in min_mass_values:
            n_comp, n_major, sizes = graph_major_count(adj, min_mass)
            rows.append({
                "method": f"radius_knn5_eps{eps_mult:g}",
                "family": "radius_graph",
                "min_component_mass": min_mass,
                "n_components": n_comp,
                "n_major_components": n_major,
                "component_sizes": sizes,
                "noise_frac": 0.0,
            })
    return rows


def anisotropic_density(points: np.ndarray, grid_size: int, k: int, scale: float, anisotropy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    pad = 1.0
    xs = np.linspace(pts[:, 0].min() - pad, pts[:, 0].max() + pad, grid_size)
    ys = np.linspace(pts[:, 1].min() - pad, pts[:, 1].max() + pad, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    nn = NearestNeighbors(n_neighbors=max(2, min(k + 1, n))).fit(pts)
    dists, idxs = nn.kneighbors(pts)
    density = np.zeros(len(grid), dtype=float)
    for i in range(n):
        nbrs = pts[idxs[i, 1:]]
        if len(nbrs) < 2:
            cov = np.eye(2)
        else:
            centered = nbrs - nbrs.mean(axis=0)
            cov = np.cov(centered.T)
            if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
                cov = np.eye(2)
        vals, vecs = np.linalg.eigh(cov + 1e-6 * np.eye(2))
        vals = np.maximum(vals, 1e-5)
        # Preserve local tangent direction but avoid needle kernels.
        vals = np.array([max(vals[0], vals[1] / anisotropy), vals[1]])
        cov = vecs @ np.diag(vals * scale * scale) @ vecs.T
        inv = np.linalg.inv(cov)
        det = max(float(np.linalg.det(cov)), 1e-12)
        diff = grid - pts[i]
        q = np.einsum("ij,jk,ik->i", diff, inv, diff)
        density += np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(det))
    density = density.reshape(xx.shape) / n
    return xx, yy, density


def component_count_from_density(density: np.ndarray, min_mass: float) -> tuple[int, int, str]:
    flat = density.ravel()
    total = float(flat.sum())
    if total <= 0:
        return 0, 0, ""
    max_major = 0
    max_components = 0
    best_masses = np.array([], dtype=float)
    for mass_level in DEFAULT_HDR_MASS_LEVELS:
        ordered = np.sort(flat)[::-1]
        csum = np.cumsum(ordered) / total
        threshold = float(ordered[min(int(np.searchsorted(csum, mass_level, side="left")), len(ordered) - 1)])
        labels, n_labels = ndi_label(density >= threshold)
        masses = np.array([
            density[labels == label].sum() / total
            for label in range(1, n_labels + 1)
        ], dtype=float)
        major = masses[masses >= min_mass]
        if len(major) > max_major:
            max_major = len(major)
            max_components = n_labels
            best_masses = np.sort(masses)[::-1]
    return int(max_components), int(max_major), _fmt(best_masses)


def density_rows(points: np.ndarray, min_mass_values: list[float], grid_size: int) -> list[dict]:
    rows = []
    for scale in (0.35, 0.45, 0.55, 0.65, 0.80, 1.0):
        _, _, density = _kde_grid(points, grid_size=grid_size, kde=scipy_gaussian_kde_spec(bw_scale=scale))
        for min_mass in min_mass_values:
            n_comp, n_major, masses = component_count_from_density(density, min_mass)
            rows.append({
                "method": f"scott_x{scale:g}",
                "family": "isotropic_kde",
                "min_component_mass": min_mass,
                "n_components": n_comp,
                "n_major_components": n_major,
                "component_sizes": masses,
                "noise_frac": 0.0,
            })
    for k in (4, 6, 8, 10):
        for scale in (0.6, 0.8, 1.0, 1.25):
            for anis in (2.0, 4.0, 8.0, 16.0):
                _, _, density = anisotropic_density(points, grid_size=grid_size, k=k, scale=scale, anisotropy=anis)
                for min_mass in min_mass_values:
                    n_comp, n_major, masses = component_count_from_density(density, min_mass)
                    rows.append({
                        "method": f"anisok{k}_s{scale:g}_a{anis:g}",
                        "family": "anisotropic_kde",
                        "min_component_mass": min_mass,
                        "n_components": n_comp,
                        "n_major_components": n_major,
                        "component_sizes": masses,
                        "noise_frac": 0.0,
                    })
    return rows


def score(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in seed_df.groupby(["method", "family", "min_component_mass"]):
        method, family, min_mass = keys
        by_case = {}
        for case, csub in sub.groupby("case"):
            by_case[f"{case}_ge2"] = float(np.mean(csub["n_major_components"] >= 2))
            by_case[f"{case}_ge3"] = float(np.mean(csub["n_major_components"] >= 3))
            by_case[f"{case}_mean_major"] = float(np.mean(csub["n_major_components"]))
        three = by_case.get("three_discrete_ge3", 0.0)
        false_cases = ["spiral", "crescent", "outliers", "wt_null"]
        false = sum(by_case.get(f"{case}_ge2", 0.0) for case in false_cases)
        rows.append({
            "method": method,
            "family": family,
            "min_component_mass": min_mass,
            "three_ge3": three,
            "false_ge2_sum": false,
            "selectivity_score": three - false,
            **by_case,
        })
    return pd.DataFrame(rows).sort_values(
        ["selectivity_score", "three_ge3", "false_ge2_sum"],
        ascending=[False, False, True],
    )


def make_plot(scores: pd.DataFrame, prefix: str) -> Path:
    top = scores.head(30).copy()
    labels = [f"{r.method}\nmin={r.min_component_mass:g}" for r in top.itertuples()]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(top)), 5.5))
    ax.bar(x - 0.18, top["three_ge3"], width=0.18, label="three >=3", color="#1B9E77")
    false = top[[c for c in top.columns if c.endswith("_ge2") and not c.startswith("three_discrete")]].sum(axis=1)
    ax.bar(x, false, width=0.18, label="false >=2 sum", color="#D95F02")
    ax.bar(x + 0.18, top["selectivity_score"], width=0.18, label="score", color="#7570B3")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(min(-1.2, float(top["selectivity_score"].min()) - 0.1), max(1.1, float(top["three_ge3"].max()) + 0.1))
    ax.set_title("Geometry strategy search")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = PLOT_DIR / f"{prefix}_top_methods.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--cases", nargs="*", default=["three_discrete", "spiral", "crescent", "outliers", "wt_null"])
    parser.add_argument("--min-component-masses", nargs="*", type=float, default=[0.03, 0.05, 0.08, 0.10])
    parser.add_argument("--families", nargs="*", default=["dbscan", "optics", "graph", "density"])
    parser.add_argument("--output-prefix", default="geometry_strategy_search")
    args = parser.parse_args()

    rows = []
    total = len(args.cases) * args.n_seeds
    done = 0
    for case in args.cases:
        for seed in range(args.n_seeds):
            pts = scenario_points(case, args.sample_size, seed)
            case_rows = []
            if "dbscan" in args.families:
                case_rows.extend(dbscan_rows(pts, args.min_component_masses))
            if "optics" in args.families:
                case_rows.extend(optics_rows(pts, args.min_component_masses))
            if "graph" in args.families:
                case_rows.extend(graph_rows(pts, args.min_component_masses))
            if "density" in args.families:
                case_rows.extend(density_rows(pts, args.min_component_masses, args.grid_size))
            for row in case_rows:
                row.update({"case": case, "seed": seed, "n": args.sample_size})
            rows.extend(case_rows)
            done += 1
            print(f"  completed {done}/{total} case-seeds", flush=True)

    df = pd.DataFrame(rows)
    scores = score(df)
    prefix = args.output_prefix
    df.to_csv(TABLE_DIR / f"{prefix}_raw.csv", index=False)
    scores.to_csv(TABLE_DIR / f"{prefix}_scores.csv", index=False)
    plot_path = make_plot(scores, prefix)
    print(f"Saved: {TABLE_DIR / f'{prefix}_raw.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_scores.csv'}")
    print(f"Saved: {plot_path}")
    print(scores.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
