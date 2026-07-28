"""
Targeted validation for HDR mass-area concentration curves.

This is intentionally narrow: by default it runs only the weak-separation case and
the small-middle case. The primary question is whether dense target mass occupies
less KDE area than matched-N WT null resamples.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/hdr_area_validation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from support_geometry import (  # noqa: E402
    DEFAULT_HDR_MASS_LEVELS,
    KDESpec,
    hdr_concentration_auc,
    hdr_mass_area_profile,
    knn_adaptive_kde_spec,
    normalize_shape,
    scipy_gaussian_kde_spec,
    valley_depth,
)
from synthetic_scenarios import SCENARIOS_BY_NAME, wt_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables" / "density_improvements"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCENARIOS = ["weak_separation", "small_middle"]
WT_POOL_N = 3000
SIG_ALPHA = 0.05


def kde_specs(names: list[str]) -> list[tuple[str, KDESpec]]:
    known = {
        "scott": scipy_gaussian_kde_spec(),
        "knn_k10_bw0.8": knn_adaptive_kde_spec(k=10, bw_scale=0.8),
        "knn_k5_bw0.6": knn_adaptive_kde_spec(k=5, bw_scale=0.6),
        "knn_k3_bw0.6_clip1.5": knn_adaptive_kde_spec(k=3, bw_scale=0.6, max_factor=1.5),
    }
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"Unknown KDE candidate(s): {unknown}. Known: {sorted(known)}")
    return [(name, known[name]) for name in names]


def _profile_rows(
    scenario: str,
    seed: int,
    kde_name: str,
    pts: np.ndarray,
    wt_pool: np.ndarray,
    n_resample: int,
    rng: np.random.Generator,
    kde: KDESpec,
    space: str,
) -> list[dict]:
    pts_eval = normalize_shape(pts) if space == "normalized" else np.asarray(pts, dtype=float)
    ref_eval = normalize_shape(wt_pool) if space == "normalized" else np.asarray(wt_pool, dtype=float)
    observed = hdr_mass_area_profile(pts_eval, kde=kde)
    null_areas = []
    null_relative = []
    for _ in range(n_resample):
        idx = rng.choice(len(ref_eval), size=len(pts), replace=True)
        prof = hdr_mass_area_profile(ref_eval[idx], kde=kde)
        null_areas.append(prof["areas"])
        null_relative.append(prof["relative_areas"])
    null_areas = np.asarray(null_areas, dtype=float)
    null_relative = np.asarray(null_relative, dtype=float)

    rows = []
    for j, mass in enumerate(observed["mass_levels"]):
        rows.append({
            "scenario": scenario,
            "seed": seed,
            "kde": kde_name,
            "space": space,
            "mass_level": float(mass),
            "hdr_area": float(observed["areas"][j]),
            "hdr_relative_area": float(observed["relative_areas"][j]),
            "hdr_density_threshold": float(observed["thresholds"][j]),
            "null_median_area": float(np.median(null_areas[:, j])),
            "null_median_relative_area": float(np.median(null_relative[:, j])),
            "area_log_ratio_to_null_median": float(
                np.log(observed["areas"][j] / np.median(null_areas[:, j]))
            ),
            "relative_area_log_ratio_to_null_median": float(
                np.log(observed["relative_areas"][j] / np.median(null_relative[:, j]))
            ),
        })
    return rows


def _run_one(
    scenario_name: str,
    seed: int,
    n: int,
    n_resample: int,
    wt_pool: np.ndarray,
    kde_name: str,
    kde: KDESpec,
    space: str,
) -> tuple[dict, list[dict]]:
    scenario = SCENARIOS_BY_NAME[scenario_name]
    rng = np.random.default_rng(1000 + 100 * seed + n)
    pts = scenario.generator(n, rng)
    pts_eval = normalize_shape(pts) if space == "normalized" else np.asarray(pts, dtype=float)
    ref_eval = normalize_shape(wt_pool) if space == "normalized" else np.asarray(wt_pool, dtype=float)
    stat_fns = {
        "hdr_abs": lambda x: hdr_concentration_auc(x, kde=kde),
        "hdr_rel": lambda x: hdr_concentration_auc(x, relative=True, kde=kde),
        "valley": lambda x: valley_depth(x, kde=kde),
    }
    observed = {name: fn(pts_eval) for name, fn in stat_fns.items()}
    null = {name: [] for name in stat_fns}
    null_rng = np.random.default_rng(7000 + seed)
    for _ in range(n_resample):
        idx = null_rng.choice(len(ref_eval), size=len(pts_eval), replace=True)
        sample = ref_eval[idx]
        for name, fn in stat_fns.items():
            null[name].append(fn(sample))
    null = {name: np.asarray(vals, dtype=float) for name, vals in null.items()}
    pvalues = {name: float(np.mean(null[name] >= observed[name])) for name in stat_fns}
    references = {name: float(np.median(null[name])) for name in stat_fns}
    profile_rows = _profile_rows(
        scenario_name,
        seed,
        kde_name,
        pts,
        wt_pool,
        n_resample,
        np.random.default_rng(9000 + seed),
        kde,
        space,
    )
    summary_row = {
        "scenario": scenario_name,
        "expected_support": scenario.expected_support,
        "seed": seed,
        "n": n,
        "kde": kde_name,
        "space": space,
        "hdr_abs_auc": -observed["hdr_abs"],
        "hdr_abs_null_median_auc": -references["hdr_abs"],
        "hdr_abs_pvalue": pvalues["hdr_abs"],
        "hdr_abs_significant": pvalues["hdr_abs"] < SIG_ALPHA,
        "hdr_rel_auc": -observed["hdr_rel"],
        "hdr_rel_null_median_auc": -references["hdr_rel"],
        "hdr_rel_pvalue": pvalues["hdr_rel"],
        "hdr_rel_significant": pvalues["hdr_rel"] < SIG_ALPHA,
        "valley_depth": observed["valley"],
        "valley_pvalue": pvalues["valley"],
        "valley_significant": pvalues["valley"] < SIG_ALPHA,
    }
    return summary_row, profile_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--kdes", nargs="*", default=["scott"])
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-resample", type=int, default=100)
    parser.add_argument("--space", choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--output-prefix", default="hdr_area_targeted")
    args = parser.parse_args()

    unknown_scenarios = sorted(set(args.scenarios) - set(SCENARIOS_BY_NAME))
    if unknown_scenarios:
        raise ValueError(f"Unknown scenario(s): {unknown_scenarios}")

    wt_pool = wt_reference(WT_POOL_N, np.random.default_rng(1))
    rows = []
    profile_rows = []
    combos = [
        (scenario_name, seed, kde_name, kde)
        for scenario_name in args.scenarios
        for seed in range(args.n_seeds)
        for kde_name, kde in kde_specs(args.kdes)
    ]
    print("HDR mass-area targeted validation")
    print(f"  scenarios={args.scenarios}")
    print(f"  kdes={args.kdes}")
    print(
        f"  n={args.sample_size} n_seeds={args.n_seeds} "
        f"n_resample={args.n_resample} space={args.space} "
        f"mass_levels={DEFAULT_HDR_MASS_LEVELS.tolist()}"
    )
    for i, (scenario_name, seed, kde_name, kde) in enumerate(combos, start=1):
        row, prof = _run_one(
            scenario_name,
            seed,
            args.sample_size,
            args.n_resample,
            wt_pool,
            kde_name,
            kde,
            args.space,
        )
        rows.append(row)
        profile_rows.extend(prof)
        print(
            f"  {i}/{len(combos)} {scenario_name} seed={seed} kde={kde_name} "
            f"abs_p={row['hdr_abs_pvalue']:.3f} rel_p={row['hdr_rel_pvalue']:.3f} "
            f"valley_p={row['valley_pvalue']:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    profile_df = pd.DataFrame(profile_rows)
    summary = (
        df.groupby(["scenario", "expected_support", "kde", "space"], as_index=False)
        .agg(
            hdr_abs_significant_rate=("hdr_abs_significant", "mean"),
            hdr_rel_significant_rate=("hdr_rel_significant", "mean"),
            valley_significant_rate=("valley_significant", "mean"),
            median_hdr_abs_pvalue=("hdr_abs_pvalue", "median"),
            median_hdr_rel_pvalue=("hdr_rel_pvalue", "median"),
            median_valley_pvalue=("valley_pvalue", "median"),
            median_hdr_abs_auc=("hdr_abs_auc", "median"),
            median_hdr_abs_null_auc=("hdr_abs_null_median_auc", "median"),
            median_hdr_rel_auc=("hdr_rel_auc", "median"),
            median_hdr_rel_null_auc=("hdr_rel_null_median_auc", "median"),
        )
    )
    curve_summary = (
        profile_df.groupby(["scenario", "kde", "mass_level"], as_index=False)
        .agg(
            median_area_log_ratio=("area_log_ratio_to_null_median", "median"),
            median_relative_area_log_ratio=("relative_area_log_ratio_to_null_median", "median"),
        )
    )

    row_out = TABLE_DIR / f"{args.output_prefix}.csv"
    summary_out = TABLE_DIR / f"{args.output_prefix}_summary.csv"
    profile_out = TABLE_DIR / f"{args.output_prefix}_profiles.csv"
    curve_out = TABLE_DIR / f"{args.output_prefix}_curve_summary.csv"
    df.to_csv(row_out, index=False)
    summary.to_csv(summary_out, index=False)
    profile_df.to_csv(profile_out, index=False)
    curve_summary.to_csv(curve_out, index=False)
    print(f"Saved: {row_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {profile_out}")
    print(f"Saved: {curve_out}")


if __name__ == "__main__":
    main()
