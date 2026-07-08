"""
Broad synthetic diagnostics for support-discreteness methods.

This is deliberately diagnostic, not production gating. It benchmarks the current
KDE baselines (Scott and kNN adaptive k=10/bw=0.6) plus graph, MST, local-scale,
and HDR-component methods across the synthetic scenarios. Outputs are raw enough
to inspect failure modes rather than only final calls.

Run a small smoke test:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/support_method_diagnostics.py \
        --sample-size 40 --n-seeds 1 --n-resample 5 --output-prefix smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label
from scipy.sparse.csgraph import connected_components, laplacian, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from support_geometry import (  # noqa: E402
    DEFAULT_HDR_MASS_LEVELS,
    GRID_SIZE,
    MIN_COMPONENT_MASS_FRAC,
    KDESpec,
    _delaunay_edges,
    _kde_grid,
    _knn_adjacency,
    conductance,
    critical_connection_ratio,
    critical_major_connection_ratio,
    evaluate_kde_on_grid,
    fiedler_value,
    hdr_concentration_auc,
    knn_adaptive_kde_spec,
    mst_max_edge,
    normalize_shape,
    scipy_gaussian_kde_spec,
    valley_depth,
    valley_detection_detail,
)
from synthetic_scenarios import SCENARIOS, SCENARIOS_BY_NAME, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "support_method_diagnostics"
PLOT_DIR = RUN_DIR / "plots" / "support_method_diagnostics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WT_POOL_N = 3000
SIG_ALPHA = 0.05
DEFAULT_KDES = ("scott", "knn_k10_bw0.6")

_WT_POOL = None
_KDE_SPECS = None


def _init_worker(wt_pool: np.ndarray, kdes: list[tuple[str, KDESpec]]) -> None:
    global _WT_POOL, _KDE_SPECS
    _WT_POOL = wt_pool
    _KDE_SPECS = kdes


def kde_specs(names: list[str]) -> list[tuple[str, KDESpec]]:
    known = {
        "scott": scipy_gaussian_kde_spec(),
        "knn_k10_bw0.6": knn_adaptive_kde_spec(k=10, bw_scale=0.6),
    }
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"Unknown KDE(s): {unknown}. Known: {sorted(known)}")
    return [(name, known[name]) for name in names]


def _fmt(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return ""
    return ";".join(f"{v:.{digits}f}" for v in vals)


def _pvalue(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    return float(np.mean(null >= obs))


def _null_summary(prefix: str, null: np.ndarray) -> dict:
    null = np.asarray(null, dtype=float)
    return {
        "null_mean": float(np.mean(null)),
        "null_median": float(np.median(null)),
        "null_q05": float(np.quantile(null, 0.05)),
        "null_q95": float(np.quantile(null, 0.95)),
        f"{prefix}_null_mean": float(np.mean(null)),
        f"{prefix}_null_median": float(np.median(null)),
        f"{prefix}_null_q05": float(np.quantile(null, 0.05)),
        f"{prefix}_null_q95": float(np.quantile(null, 0.95)),
    }


def _local_scale_distance_matrix(points: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    dmat = squareform(pdist(pts))
    if len(pts) < 2:
        return dmat, np.ones(len(pts)), dmat
    k_eff = max(1, min(int(k), len(pts) - 1))
    sigma = np.sort(dmat, axis=1)[:, k_eff]
    positive = sigma[sigma > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    sigma = np.where(sigma <= 1e-12, fallback, sigma)
    denom = np.sqrt(np.outer(sigma, sigma))
    scaled = dmat / np.where(denom <= 1e-12, 1.0, denom)
    return dmat, sigma, scaled


def _mst_edge_diagnostics_from_dmat(dmat: np.ndarray) -> dict:
    n = dmat.shape[0]
    if n < 3:
        return {
            "mst_max_edge_ratio": 0.0,
            "mst_top2_ratio": 0.0,
            "mst_edges_gt_q90_ratio": 0.0,
            "mst_edges": "",
        }
    mst = minimum_spanning_tree(dmat)
    edges = np.asarray(mst.data, dtype=float)
    edges = edges[edges > 0]
    if edges.size == 0:
        return {
            "mst_max_edge_ratio": 0.0,
            "mst_top2_ratio": 0.0,
            "mst_edges_gt_q90_ratio": 0.0,
            "mst_edges": "",
        }
    med = float(np.median(edges))
    ratios = edges / med if med > 1e-12 else np.zeros_like(edges)
    top = np.sort(ratios)[::-1]
    q90 = float(np.quantile(ratios, 0.90)) if ratios.size else 0.0
    return {
        "mst_max_edge_ratio": float(top[0]),
        "mst_top2_ratio": float(top[1]) if top.size > 1 else 0.0,
        "mst_edges_gt_q90_ratio": int(np.sum(ratios >= q90)),
        "mst_edges": _fmt(np.sort(ratios)[::-1][:10], 3),
    }


def raw_mst_max_edge(points: np.ndarray) -> float:
    dmat = squareform(pdist(points))
    return _mst_edge_diagnostics_from_dmat(dmat)["mst_max_edge_ratio"]


def local_scaled_mst_max_edge(points: np.ndarray, k: int = 5) -> float:
    _, _, scaled = _local_scale_distance_matrix(points, k=k)
    return _mst_edge_diagnostics_from_dmat(scaled)["mst_max_edge_ratio"]


def local_scaled_mst_top2(points: np.ndarray, k: int = 5) -> float:
    _, _, scaled = _local_scale_distance_matrix(points, k=k)
    return _mst_edge_diagnostics_from_dmat(scaled)["mst_top2_ratio"]


def _major_component_profile_from_density(
    density: np.ndarray,
    mass_levels: np.ndarray,
    min_component_mass: float = MIN_COMPONENT_MASS_FRAC,
) -> list[dict]:
    flat = np.asarray(density, dtype=float).ravel()
    total = float(flat.sum())
    if total <= 0:
        return []
    order = np.argsort(flat)[::-1]
    sorted_density = flat[order]
    cumulative = np.cumsum(sorted_density) / total
    rows = []
    for mass in mass_levels:
        idx = int(np.searchsorted(cumulative, mass, side="left"))
        idx = min(idx, len(sorted_density) - 1)
        threshold = float(sorted_density[idx])
        labels, n_labels = ndi_label(density >= threshold)
        component_masses = np.array([
            density[labels == lbl].sum() / total for lbl in range(1, n_labels + 1)
        ])
        major = component_masses[component_masses >= min_component_mass]
        rows.append({
            "mass_level": float(mass),
            "density_threshold": threshold,
            "n_components": int(n_labels),
            "n_major_components": int(major.size),
            "largest_component_mass": float(component_masses.max()) if component_masses.size else 0.0,
            "second_component_mass": float(np.sort(component_masses)[-2]) if component_masses.size > 1 else 0.0,
            "major_component_masses": _fmt(np.sort(major)[::-1], 4),
        })
    return rows


def hdr_component_profile(
    points: np.ndarray,
    kde: KDESpec,
    mass_levels: np.ndarray = DEFAULT_HDR_MASS_LEVELS,
    grid_size: int = GRID_SIZE,
) -> list[dict]:
    _, _, density = _kde_grid(points, grid_size=grid_size, kde=kde)
    return _major_component_profile_from_density(density, np.asarray(mass_levels, dtype=float))


def hdr_component_stats(points: np.ndarray, kde: KDESpec) -> dict:
    profile = hdr_component_profile(points, kde=kde)
    if not profile:
        return {
            "hdr_max_major_components": 0.0,
            "hdr_component_auc": 0.0,
            "hdr_second_component_mass_max": 0.0,
        }
    mass = np.array([r["mass_level"] for r in profile], dtype=float)
    counts = np.array([r["n_major_components"] for r in profile], dtype=float)
    second = np.array([r["second_component_mass"] for r in profile], dtype=float)
    return {
        "hdr_max_major_components": float(np.max(counts)),
        "hdr_component_auc": float(np.trapz(np.maximum(counts - 1.0, 0.0), mass)),
        "hdr_second_component_mass_max": float(np.max(second)),
    }


def _graph_diagnostics(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    k = max(3, min(n - 1, int(np.ceil(np.log2(n))))) if n > 1 else 1
    if n < 4:
        return {
            "graph_k": k,
            "graph_n_components": 0,
            "graph_component_sizes": "",
            "graph_eigenvalues": "",
            "graph_near_zero_eigenvalues": 0,
            "fiedler_stat": 0.0,
            "conductance_stat": 0.0,
            "fiedler_split_frac": 0.0,
        }
    adj = _knn_adjacency(pts, k=k)
    n_comp, labels = connected_components(adj > 0, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    lap = laplacian(adj, normed=True)
    eigvals, eigvecs = np.linalg.eigh(lap)
    eigvals = np.sort(np.maximum(eigvals, 0.0))
    order = np.argsort(eigvals)
    fvec = eigvecs[:, order[1]] if eigvecs.shape[1] > 1 else eigvecs[:, 0]
    side = fvec >= 0
    split_frac = min(float(np.mean(side)), float(np.mean(~side)))
    return {
        "graph_k": k,
        "graph_n_components": int(n_comp),
        "graph_component_sizes": _fmt(np.sort(sizes)[::-1], 0),
        "graph_eigenvalues": _fmt(eigvals[:8], 5),
        "graph_near_zero_eigenvalues": int(np.sum(eigvals < 1e-6)),
        "fiedler_stat": fiedler_value(pts),
        "conductance_stat": conductance(pts),
        "fiedler_split_frac": split_frac,
    }


def _point_rows(scenario: str, seed: int, points: np.ndarray) -> list[dict]:
    pts = np.asarray(points, dtype=float)
    _, sigma, _ = _local_scale_distance_matrix(pts, k=5)
    return [
        {
            "scenario": scenario,
            "seed": seed,
            "point_index": i,
            "x_norm": float(pt[0]),
            "y_norm": float(pt[1]),
            "knn5_sigma": float(sigma[i]),
        }
        for i, pt in enumerate(pts)
    ]


def _evaluate_stat(
    name: str,
    fn,
    points: np.ndarray,
    null_points: list[np.ndarray],
) -> dict:
    obs = float(fn(points))
    null = np.array([fn(sample) for sample in null_points], dtype=float)
    row = {
        "metric": name,
        "observed": obs,
        "pvalue": _pvalue(obs, null),
        "significant": _pvalue(obs, null) < SIG_ALPHA,
        "null_values": _fmt(null, 5),
    }
    row.update(_null_summary(name, null))
    return row


def _task(
    scenario_name: str,
    seed: int,
    n: int,
    n_resample: int,
    wt_pool: np.ndarray,
    kdes: list[tuple[str, KDESpec]],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    scenario = SCENARIOS_BY_NAME[scenario_name]
    rng = np.random.default_rng(1000 + 100 * seed + n)
    points_raw = scenario.generator(n, rng)
    points = normalize_shape(points_raw)
    ref = normalize_shape(wt_pool)
    null_rng = np.random.default_rng(7000 + 100 * seed + n)
    draw_indices = [null_rng.choice(len(ref), size=n, replace=True) for _ in range(n_resample)]
    null_points = [ref[idx] for idx in draw_indices]

    metric_rows = []
    profile_rows = []
    graph_rows = []
    mst_rows = []
    point_rows = _point_rows(scenario_name, seed, points)

    base = {
        "scenario": scenario_name,
        "expected_support": scenario.expected_support,
        "seed": seed,
        "n": n,
    }

    graph_diag = _graph_diagnostics(points)
    raw_dmat, sigma, scaled_dmat = _local_scale_distance_matrix(points, k=5)
    raw_mst_diag = _mst_edge_diagnostics_from_dmat(raw_dmat)
    scaled_mst_diag = _mst_edge_diagnostics_from_dmat(scaled_dmat)
    graph_rows.append({**base, **graph_diag})
    mst_rows.append({
        **base,
        "space": "raw_distance",
        "local_sigma_median": float(np.median(sigma)),
        "local_sigma_q05": float(np.quantile(sigma, 0.05)),
        "local_sigma_q95": float(np.quantile(sigma, 0.95)),
        **raw_mst_diag,
    })
    mst_rows.append({
        **base,
        "space": "local_scaled_distance",
        "local_sigma_median": float(np.median(sigma)),
        "local_sigma_q05": float(np.quantile(sigma, 0.05)),
        "local_sigma_q95": float(np.quantile(sigma, 0.95)),
        **scaled_mst_diag,
    })

    graph_stats = {
        "mst_max_edge": mst_max_edge,
        "critical_conn_90": critical_connection_ratio,
        "critical_major_conn": critical_major_connection_ratio,
        "fiedler": fiedler_value,
        "conductance": conductance,
        "local_scaled_mst_max": local_scaled_mst_max_edge,
        "local_scaled_mst_top2": local_scaled_mst_top2,
    }
    for metric, fn in graph_stats.items():
        metric_rows.append({**base, "kde": "none", **_evaluate_stat(metric, fn, points, null_points)})

    for kde_name, kde in kdes:
        kde_stats = {
            "valley_depth": lambda x, kk=kde: valley_depth(x, kde=kk),
            "hdr_area_concentration": lambda x, kk=kde: hdr_concentration_auc(x, relative=True, kde=kk),
            "hdr_max_major_components": lambda x, kk=kde: hdr_component_stats(x, kk)["hdr_max_major_components"],
            "hdr_component_auc": lambda x, kk=kde: hdr_component_stats(x, kk)["hdr_component_auc"],
            "hdr_second_component_mass_max": (
                lambda x, kk=kde: hdr_component_stats(x, kk)["hdr_second_component_mass_max"]
            ),
        }
        for metric, fn in kde_stats.items():
            metric_rows.append({**base, "kde": kde_name, **_evaluate_stat(metric, fn, points, null_points)})

        valley = valley_detection_detail(points, kde=kde)
        profile = hdr_component_profile(points, kde=kde)
        for row in profile:
            profile_rows.append({**base, "kde": kde_name, **row})
        profile_rows.append({
            **base,
            "kde": kde_name,
            "mass_level": np.nan,
            "density_threshold": valley["level"] if valley["level"] is not None else np.nan,
            "n_components": valley["n_components_at_split"],
            "n_major_components": valley["n_components_at_split"],
            "largest_component_mass": np.nan,
            "second_component_mass": np.nan,
            "major_component_masses": _fmt(valley["component_mass_fracs"], 4),
            "profile_type": "valley_split",
            "valley_frac": valley["valley_frac"],
        })

    return metric_rows, profile_rows, graph_rows, mst_rows, point_rows


def _task_from_args(args: tuple[str, int, int, int]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    scenario_name, seed, n, n_resample = args
    if _WT_POOL is None or _KDE_SPECS is None:
        raise RuntimeError("worker globals were not initialized")
    return _task(scenario_name, seed, n, n_resample, _WT_POOL, _KDE_SPECS)


def summarize_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        df.groupby(["kde", "metric", "scenario", "expected_support"], as_index=False)
        .agg(
            significant_rate=("significant", "mean"),
            median_pvalue=("pvalue", "median"),
            median_observed=("observed", "median"),
            median_null=("null_median", "median"),
        )
    )
    scores = []
    controls = {s.name for s in SCENARIOS if s.expected_support == "connected"}
    targets = {s.name for s in SCENARIOS if s.expected_support == "discrete"}
    for (kde, metric), sub in summary.groupby(["kde", "metric"]):
        target_rate = sub[sub["scenario"].isin(targets)]["significant_rate"].mean()
        control_rate = sub[sub["scenario"].isin(controls)]["significant_rate"].mean()
        crescent_rate = float(sub.loc[sub["scenario"] == "crescent", "significant_rate"].mean())
        outlier_rate = float(sub.loc[sub["scenario"] == "outliers", "significant_rate"].mean())
        three_rate = float(sub.loc[sub["scenario"] == "three_discrete", "significant_rate"].mean())
        small_middle_rate = float(sub.loc[sub["scenario"] == "small_middle", "significant_rate"].mean())
        scores.append({
            "kde": kde,
            "metric": metric,
            "target_rate": target_rate,
            "control_rate": control_rate,
            "score": target_rate - control_rate,
            "crescent_rate": crescent_rate,
            "outliers_rate": outlier_rate,
            "three_discrete_rate": three_rate,
            "small_middle_rate": small_middle_rate,
            "priority_score": three_rate + small_middle_rate - crescent_rate - outlier_rate,
        })
    score_df = pd.DataFrame(scores).sort_values(
        ["priority_score", "score", "three_discrete_rate"],
        ascending=False,
    )
    return summary, score_df


def make_score_plot(score_df: pd.DataFrame, out_prefix: str) -> Path:
    labels = [f"{r.kde}:{r.metric}" for r in score_df.itertuples()]
    vals = score_df["priority_score"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 0.38 * len(labels)), 4.8))
    colors = ["#4C78A8" if v >= 0 else "#B279A2" for v in vals]
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("three + small_middle - crescent - outliers")
    ax.set_title("Support-method diagnostic priority score")
    fig.tight_layout()
    out = PLOT_DIR / f"{out_prefix}_priority_scores.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-resample", type=int, default=100)
    parser.add_argument("--scenarios", nargs="*", default=[s.name for s in SCENARIOS])
    parser.add_argument("--kdes", nargs="*", default=list(DEFAULT_KDES))
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-prefix", default="support_method_diagnostics")
    args = parser.parse_args()

    unknown = sorted(set(args.scenarios) - set(SCENARIOS_BY_NAME))
    if unknown:
        raise ValueError(f"Unknown scenario(s): {unknown}")

    specs = kde_specs(args.kdes)
    wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))
    print("Support-method diagnostics")
    print(json.dumps({
        "sample_size": args.sample_size,
        "n_seeds": args.n_seeds,
        "n_resample": args.n_resample,
        "n_workers": args.n_workers,
        "scenarios": args.scenarios,
        "kdes": args.kdes,
        "output_prefix": args.output_prefix,
    }, indent=2))

    metric_rows = []
    profile_rows = []
    graph_rows = []
    mst_rows = []
    point_rows = []
    tasks = [
        (scenario, seed, args.sample_size, args.n_resample)
        for scenario in args.scenarios
        for seed in range(args.n_seeds)
    ]
    n_workers = max(1, int(args.n_workers))
    if n_workers == 1:
        _init_worker(wt_pool, specs)
        iterator = map(_task_from_args, tasks)
    else:
        pool = mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(wt_pool, specs))
        iterator = pool.imap_unordered(_task_from_args, tasks, chunksize=1)
    try:
        for i, rows in enumerate(iterator, start=1):
            m, p, g, mst, pts = rows
            metric_rows.extend(m)
            profile_rows.extend(p)
            graph_rows.extend(g)
            mst_rows.extend(mst)
            point_rows.extend(pts)
            if i % 5 == 0 or i == len(tasks):
                print(f"  completed {i}/{len(tasks)} scenario-seed tasks", flush=True)
    finally:
        if n_workers != 1:
            pool.close()
            pool.join()

    metrics = pd.DataFrame(metric_rows)
    profiles = pd.DataFrame(profile_rows)
    graph = pd.DataFrame(graph_rows)
    mst = pd.DataFrame(mst_rows)
    points = pd.DataFrame(point_rows)
    summary, scores = summarize_metrics(metrics)

    prefix = args.output_prefix
    metrics.to_csv(TABLE_DIR / f"{prefix}_metric_raw.csv", index=False)
    profiles.to_csv(TABLE_DIR / f"{prefix}_hdr_component_profiles.csv", index=False)
    graph.to_csv(TABLE_DIR / f"{prefix}_graph_diagnostics.csv", index=False)
    mst.to_csv(TABLE_DIR / f"{prefix}_mst_local_scale_diagnostics.csv", index=False)
    points.to_csv(TABLE_DIR / f"{prefix}_point_local_scales.csv", index=False)
    summary.to_csv(TABLE_DIR / f"{prefix}_metric_summary.csv", index=False)
    scores.to_csv(TABLE_DIR / f"{prefix}_metric_scores.csv", index=False)
    plot_path = make_score_plot(scores, prefix)

    print(f"Saved: {TABLE_DIR / f'{prefix}_metric_raw.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_hdr_component_profiles.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_graph_diagnostics.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_mst_local_scale_diagnostics.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_point_local_scales.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_metric_summary.csv'}")
    print(f"Saved: {TABLE_DIR / f'{prefix}_metric_scores.csv'}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
