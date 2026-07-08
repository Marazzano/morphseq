"""
Search local-scale MST cuts as a support-component detector.

The density detector fails because bandwidth controls both mode recovery and
manifold continuity. This detector instead asks whether connecting the point
cloud requires unusually long edges relative to local neighbor spacing. Cutting
only those edges should keep curved continua intact while splitting genuinely
separate support islands. Small outlier islands are filtered by point mass.
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from support_geometry import normalize_shape  # noqa: E402
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


def local_scale(points: np.ndarray, k: int) -> np.ndarray:
    d = squareform(pdist(points))
    k_eff = max(1, min(k, len(points) - 1))
    sigma = np.sort(d, axis=1)[:, k_eff]
    positive = sigma[sigma > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.where(sigma > 1e-12, sigma, fallback)


def normalized_mst_edges(points: np.ndarray, k: int, scale_mode: str) -> pd.DataFrame:
    d = squareform(pdist(points))
    sigma = local_scale(points, k)
    if scale_mode == "raw_median":
        denom = np.full_like(d, float(np.median(sigma)))
    elif scale_mode == "local_min":
        denom = np.minimum(sigma[:, None], sigma[None, :])
    elif scale_mode == "local_mean":
        denom = 0.5 * (sigma[:, None] + sigma[None, :])
    elif scale_mode == "local_geom":
        denom = np.sqrt(sigma[:, None] * sigma[None, :])
    elif scale_mode == "local_max":
        denom = np.maximum(sigma[:, None], sigma[None, :])
    else:
        raise ValueError(f"Unknown scale mode: {scale_mode}")
    weights = d / np.maximum(denom, 1e-12)
    np.fill_diagonal(weights, 0.0)
    mst = minimum_spanning_tree(csr_matrix(weights)).tocoo()
    rows = []
    for i, j, weight in zip(mst.row, mst.col, mst.data, strict=True):
        rows.append({
            "i": int(i),
            "j": int(j),
            "raw_dist": float(d[i, j]),
            "norm_dist": float(weight),
            "sigma_i": float(sigma[i]),
            "sigma_j": float(sigma[j]),
        })
    return pd.DataFrame(rows)


def component_count(points: np.ndarray, edge_df: pd.DataFrame, tau: float, min_point_frac: float) -> dict:
    n = len(points)
    keep = edge_df[edge_df["norm_dist"] <= tau]
    if keep.empty:
        adj = csr_matrix((n, n), dtype=bool)
    else:
        row = np.concatenate([keep["i"].to_numpy(int), keep["j"].to_numpy(int)])
        col = np.concatenate([keep["j"].to_numpy(int), keep["i"].to_numpy(int)])
        data = np.ones(len(row), dtype=bool)
        adj = csr_matrix((data, (row, col)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)
    counts = np.sort(np.bincount(labels, minlength=n_components))[::-1]
    min_points = int(np.ceil(min_point_frac * n))
    major = counts[counts >= min_points]
    cut_edges = edge_df[edge_df["norm_dist"] > tau]["norm_dist"].to_numpy(float)
    return {
        "n_components": int(n_components),
        "n_major_components": int(len(major)),
        "component_sizes": ";".join(str(int(v)) for v in major),
        "max_norm_edge": float(edge_df["norm_dist"].max()),
        "second_max_norm_edge": float(edge_df["norm_dist"].nlargest(2).iloc[-1]) if len(edge_df) > 1 else np.nan,
        "n_cut_edges": int(len(cut_edges)),
        "cut_edge_norms": ";".join(f"{v:.4f}" for v in np.sort(cut_edges)[::-1][:8]),
    }


def evaluate_case(points: np.ndarray, taus: list[float], min_point_fracs: list[float]) -> list[dict]:
    rows = []
    for k in (2, 3, 4, 5, 6, 8, 10):
        for scale_mode in ("raw_median", "local_min", "local_mean", "local_geom", "local_max"):
            edge_df = normalized_mst_edges(points, k, scale_mode)
            for tau in taus:
                for min_point_frac in min_point_fracs:
                    out = component_count(points, edge_df, tau, min_point_frac)
                    rows.append({
                        "method": f"mstgap_{scale_mode}_k{k}_tau{tau:g}",
                        "scale_mode": scale_mode,
                        "k": k,
                        "tau": tau,
                        "min_point_frac": min_point_frac,
                        **out,
                    })
    return rows


def score(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method", "scale_mode", "k", "tau", "min_point_frac"]
    for key_vals, sub in raw.groupby(keys):
        row = dict(zip(keys, key_vals, strict=True))
        by_case = {}
        for case, csub in sub.groupby("case"):
            by_case[f"{case}_ge2"] = float(np.mean(csub["n_major_components"] >= 2))
            by_case[f"{case}_ge3"] = float(np.mean(csub["n_major_components"] >= 3))
            by_case[f"{case}_mean"] = float(np.mean(csub["n_major_components"]))
            by_case[f"{case}_max_edge_mean"] = float(np.mean(csub["max_norm_edge"]))
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
    labels = [f"{r.scale_mode}\nk{r.k} tau{r.tau:g}\nmin{r.min_point_frac:g}" for r in top.itertuples()]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(top)), 5.5))
    ax.bar(x - 0.2, top["three_ge3"], width=0.2, label="three >=3", color="#1B9E77")
    ax.bar(x, top["false_ge2_sum"], width=0.2, label="false >=2 sum", color="#D95F02")
    ax.bar(x + 0.2, top["priority_score"], width=0.2, label="priority", color="#7570B3")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.legend(frameon=False)
    ax.set_title("Local-scale MST gap component search")
    fig.tight_layout()
    out = PLOT_DIR / f"{prefix}_top_methods.png"
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--cases", nargs="*", default=["three_discrete", "spiral", "crescent", "outliers", "wt_null"])
    parser.add_argument("--taus", nargs="*", type=float, default=[1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0])
    parser.add_argument("--min-point-fracs", nargs="*", type=float, default=[0.05, 0.08, 0.10, 0.15])
    parser.add_argument("--output-prefix", default="mst_gap_component_search")
    args = parser.parse_args()

    rows = []
    total = len(args.cases) * args.n_seeds
    done = 0
    for case in args.cases:
        for seed in range(args.n_seeds):
            points = scenario_points(case, args.sample_size, seed)
            case_rows = evaluate_case(points, args.taus, args.min_point_fracs)
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
