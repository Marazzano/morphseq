"""
Search density-component detectors with a point-graph continuity guard.

The failure mode in the pure KDE/HDR component detector is:
  - a narrow bandwidth recovers three_discrete,
  - but the same bandwidth cuts continuous low-density manifolds such as spiral.

This script keeps the sensitive KDE step, then asks whether the apparent density
components are connected by locally supported point-cloud paths. If they are,
they are merged before counting major support components.
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from support_geometry import DEFAULT_HDR_MASS_LEVELS, _kde_grid, normalize_shape, scipy_gaussian_kde_spec  # noqa: E402
from synthetic_scenarios import SCENARIOS_BY_NAME, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "support_method_diagnostics"
PLOT_DIR = RUN_DIR / "plots" / "support_method_diagnostics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def scenario_points(case: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + 100 * seed + n)
    if case == "wt_null":
        pts = wt_reference(n, rng)
    else:
        pts = SCENARIOS_BY_NAME[case].generator(n, rng)
    return normalize_shape(pts)


def _fmt(vals: list[float] | np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals)
    if vals.size == 0:
        return ""
    return ";".join(f"{float(v):.{digits}f}" for v in vals)


def local_scale(points: np.ndarray, k: int) -> np.ndarray:
    d = squareform(pdist(points))
    k_eff = max(1, min(k, len(points) - 1))
    sigma = np.sort(d, axis=1)[:, k_eff]
    positive = sigma[sigma > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.where(sigma > 1e-12, sigma, fallback)


def local_radius_graph(points: np.ndarray, k: int, eps_mult: float, mode: str) -> csr_matrix:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    d = squareform(pdist(pts))
    sigma = local_scale(pts, k)
    if mode == "global":
        radius = eps_mult * float(np.median(sigma))
        adj = (d <= radius) & (d > 0)
    elif mode == "local_or":
        radius = eps_mult * np.maximum(sigma[:, None], sigma[None, :])
        adj = (d <= radius) & (d > 0)
    elif mode == "local_and":
        radius = eps_mult * np.minimum(sigma[:, None], sigma[None, :])
        adj = (d <= radius) & (d > 0)
    elif mode == "mutual_knn":
        n_neighbors = max(2, min(k + 1, n))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(pts)
        _, idx = nn.kneighbors(pts)
        knn = np.zeros((n, n), dtype=bool)
        for i in range(n):
            knn[i, idx[i, 1:]] = True
        adj = knn & knn.T
    elif mode == "union_knn":
        n_neighbors = max(2, min(k + 1, n))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(pts)
        _, idx = nn.kneighbors(pts)
        knn = np.zeros((n, n), dtype=bool)
        for i in range(n):
            knn[i, idx[i, 1:]] = True
        adj = knn | knn.T
    else:
        raise ValueError(f"Unknown graph mode: {mode}")
    return csr_matrix(adj)


def density_threshold(density: np.ndarray, mass_level: float) -> float:
    flat = density.ravel()
    total = float(flat.sum())
    if total <= 0:
        return float("inf")
    ordered = np.sort(flat)[::-1]
    csum = np.cumsum(ordered) / total
    return float(ordered[min(int(np.searchsorted(csum, mass_level, side="left")), len(ordered) - 1)])


def point_component_ids(points: np.ndarray, xx: np.ndarray, yy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    xs = xx[0, :]
    ys = yy[:, 0]
    x_idx = np.clip(np.searchsorted(xs, points[:, 0]), 1, len(xs) - 1)
    y_idx = np.clip(np.searchsorted(ys, points[:, 1]), 1, len(ys) - 1)
    x_left = x_idx - 1
    y_left = y_idx - 1
    choose_right = np.abs(xs[x_idx] - points[:, 0]) < np.abs(xs[x_left] - points[:, 0])
    choose_up = np.abs(ys[y_idx] - points[:, 1]) < np.abs(ys[y_left] - points[:, 1])
    xi = np.where(choose_right, x_idx, x_left)
    yi = np.where(choose_up, y_idx, y_left)
    return labels[yi, xi].astype(int)


def density_component_snapshots(
    points: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    density: np.ndarray,
) -> list[dict]:
    total_density = float(density.sum())
    snapshots = []
    for mass_level in DEFAULT_HDR_MASS_LEVELS:
        threshold = density_threshold(density, float(mass_level))
        labels, n_labels = ndi_label(density >= threshold)
        if n_labels == 0:
            continue
        density_masses = np.array([
            float(density[labels == label].sum() / total_density)
            for label in range(1, n_labels + 1)
        ])
        snapshots.append({
            "hdr_mass_level": float(mass_level),
            "density_masses": density_masses,
            "point_ids": point_component_ids(points, xx, yy, labels),
        })
    return snapshots


def guarded_component_count(
    points: np.ndarray,
    snapshots: list[dict],
    graph_labels: np.ndarray,
    *,
    min_density_mass: float,
    min_point_frac: float,
) -> dict:
    n = len(points)

    best = {
        "n_density_components": 0,
        "n_guarded_components": 0,
        "hdr_mass_level": np.nan,
        "density_component_masses": "",
        "guarded_density_masses": "",
        "guarded_point_counts": "",
        "seeded_point_frac": 0.0,
    }
    for snapshot in snapshots:
        density_masses = snapshot["density_masses"]
        major_density_labels = np.where(density_masses >= min_density_mass)[0] + 1
        if major_density_labels.size == 0:
            continue

        point_ids = snapshot["point_ids"]
        seeded = np.isin(point_ids, major_density_labels)
        seeded_frac = float(np.mean(seeded))
        guarded_masses = {}
        guarded_points = {}
        for density_label in major_density_labels:
            seed_graph_components = np.unique(graph_labels[point_ids == density_label])
            if seed_graph_components.size == 0:
                continue
            # If multiple point-graph components land in one density island, keep
            # the dominant graph component. This avoids a single outlier pixel
            # giving an otherwise valid island several graph identities.
            counts = np.array([np.sum(graph_labels[point_ids == density_label] == gc) for gc in seed_graph_components])
            graph_component = int(seed_graph_components[int(np.argmax(counts))])
            guarded_masses[graph_component] = guarded_masses.get(graph_component, 0.0) + float(density_masses[density_label - 1])
            guarded_points[graph_component] = guarded_points.get(graph_component, 0) + int(np.sum(point_ids == density_label))

        min_points = int(np.ceil(min_point_frac * n))
        kept = [
            gc for gc, mass in guarded_masses.items()
            if mass >= min_density_mass and guarded_points.get(gc, 0) >= min_points
        ]
        n_guarded = len(kept)
        if n_guarded > best["n_guarded_components"]:
            best = {
                "n_density_components": int(len(major_density_labels)),
                "n_guarded_components": int(n_guarded),
                "hdr_mass_level": float(snapshot["hdr_mass_level"]),
                "density_component_masses": _fmt(np.sort(density_masses[density_masses >= min_density_mass])[::-1]),
                "guarded_density_masses": _fmt(sorted((guarded_masses[gc] for gc in kept), reverse=True)),
                "guarded_point_counts": ";".join(str(int(guarded_points[gc])) for gc in kept),
                "seeded_point_frac": seeded_frac,
            }
    return best


def evaluate_case(points: np.ndarray, grid_size: int, min_masses: list[float], min_point_fracs: list[float]) -> list[dict]:
    rows = []
    density_specs: list[tuple[str, float]] = [
        (f"scott_x{scale:g}", scale)
        for scale in (0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85)
    ]

    graph_cache: dict[tuple[str, int, float], np.ndarray] = {}
    for graph_mode in ("global", "local_and", "local_or", "mutual_knn", "union_knn"):
        for graph_k in (3, 4, 5, 6, 8):
            eps_values = (1.0,) if graph_mode.endswith("_knn") else (0.85, 1.0, 1.15, 1.3, 1.5, 1.8)
            for eps_mult in eps_values:
                graph = local_radius_graph(points, graph_k, eps_mult, graph_mode)
                _, graph_labels = connected_components(graph, directed=False)
                graph_cache[(graph_mode, graph_k, eps_mult)] = graph_labels

    density_cache: dict[str, list[dict]] = {}
    for density_name, scale in density_specs:
        xx, yy, density = _kde_grid(points, grid_size=grid_size, kde=scipy_gaussian_kde_spec(bw_scale=scale))
        density_cache[density_name] = density_component_snapshots(points, xx, yy, density)

    for density_name, snapshots in density_cache.items():
        for (graph_mode, graph_k, eps_mult), graph_labels in graph_cache.items():
            for min_mass in min_masses:
                for min_point_frac in min_point_fracs:
                    out = guarded_component_count(
                        points,
                        snapshots,
                        graph_labels,
                        min_density_mass=min_mass,
                        min_point_frac=min_point_frac,
                    )
                    rows.append({
                        "method": f"{density_name}__{graph_mode}_k{graph_k}_eps{eps_mult:g}",
                        "density": density_name,
                        "graph_mode": graph_mode,
                        "graph_k": graph_k,
                        "eps_mult": eps_mult,
                        "min_component_mass": min_mass,
                        "min_point_frac": min_point_frac,
                        **out,
                    })
    return rows


def score(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method", "density", "graph_mode", "graph_k", "eps_mult", "min_component_mass", "min_point_frac"]
    for key_vals, sub in raw.groupby(keys):
        row = dict(zip(keys, key_vals, strict=True))
        by_case = {}
        for case, csub in sub.groupby("case"):
            by_case[f"{case}_ge2"] = float(np.mean(csub["n_guarded_components"] >= 2))
            by_case[f"{case}_ge3"] = float(np.mean(csub["n_guarded_components"] >= 3))
            by_case[f"{case}_mean"] = float(np.mean(csub["n_guarded_components"]))
            by_case[f"{case}_density_mean"] = float(np.mean(csub["n_density_components"]))
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
    labels = [f"{r.density}\n{r.graph_mode} k{r.graph_k} e{r.eps_mult:g}\nm={r.min_component_mass:g}" for r in top.itertuples()]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(max(11, 0.48 * len(top)), 5.8))
    ax.bar(x - 0.2, top["three_ge3"], width=0.2, label="three >=3", color="#1B9E77")
    ax.bar(x, top["false_ge2_sum"], width=0.2, label="false >=2 sum", color="#D95F02")
    ax.bar(x + 0.2, top["priority_score"], width=0.2, label="priority", color="#7570B3")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.legend(frameon=False)
    ax.set_title("Density components with graph continuity guard")
    fig.tight_layout()
    out = PLOT_DIR / f"{prefix}_top_methods.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--cases", nargs="*", default=["three_discrete", "spiral", "crescent", "outliers", "wt_null"])
    parser.add_argument("--min-component-masses", nargs="*", type=float, default=[0.05, 0.08, 0.10])
    parser.add_argument("--min-point-fracs", nargs="*", type=float, default=[0.05, 0.08, 0.10])
    parser.add_argument("--output-prefix", default="density_graph_guard_search")
    args = parser.parse_args()

    rows = []
    total = len(args.cases) * args.n_seeds
    done = 0
    for case in args.cases:
        for seed in range(args.n_seeds):
            points = scenario_points(case, args.sample_size, seed)
            case_rows = evaluate_case(points, args.grid_size, args.min_component_masses, args.min_point_fracs)
            for row in case_rows:
                row.update({"case": case, "seed": seed, "n": args.sample_size})
            rows.extend(case_rows)
            done += 1
            print(f"  completed {done}/{total} case-seeds", flush=True)

    raw = pd.DataFrame(rows)
    scores = score(raw)
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
