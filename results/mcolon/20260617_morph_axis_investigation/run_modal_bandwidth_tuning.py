"""Geometry-derived KDE bandwidth tuning for the modal V0 anchors."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.core.bandwidth_tuning import (  # noqa: E402
    DEFAULT_CONNECTIVITY_MASS,
    DEFAULT_KNN_K,
    DEFAULT_MULTIPLIERS,
    DensityGeometryMeasurement,
    evaluate_isotropic_gaussian_kde_from_dist2,
    measure_density_geometry,
    precompute_squared_distances,
    propose_bandwidth_candidates,
)
from morphseq_investigation.core.density_composition import (  # noqa: E402
    ComposedDensityTruth,
    compose_density_truth,
    realize_from_truth,
    validate_composed_density_truth,
)
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS_BY_ID  # noqa: E402


OUT_DIR = RUN_DIR / "tables" / "modal_bandwidth_tuning"
RULE_NAMES = (
    "median_kNN_distance",
    "q90_kNN_distance",
    "median_MST_edge_length",
    "q90_MST_edge_length",
    "longest_non_outlier_MST_edge",
    "connectivity_90_radius",
    "global_R50",
    "global_R80",
)
BRIDGE_LADDER = (
    "two_peaks_no_bridge",
    "two_peaks_low_bridge",
    "two_peaks_high_bridge",
)
ONE_PEAK_CASES = (
    "one_peak_compact",
    "one_peak_diffuse",
    "one_peak_spiral",
)
COMPACT_MULTI_PEAK_CASES = (
    "two_peaks_no_bridge",
    "three_peaks_compact",
)
ANCHOR_CASES = (
    "one_peak_compact",
    "one_peak_diffuse",
    "one_peak_spiral",
    "two_peaks_no_bridge",
    "two_peaks_low_bridge",
    "two_peaks_high_bridge",
    "three_peaks_compact",
)

_TRUTH_BY_ID: dict[str, ComposedDensityTruth] = {}
_TRUTH_MEASUREMENT_BY_ID: dict[str, DensityGeometryMeasurement] = {}
_CONFIG: "CalibrationConfig" | None = None
_DIST_INDEX_BY_ID: dict[str, int] = {}


@dataclass(frozen=True)
class CalibrationConfig:
    sample_sizes: tuple[int, ...]
    n_seeds: int
    rule_names: tuple[str, ...]
    multipliers: tuple[float, ...]
    knn_k: int
    connectivity_mass: float
    bridge_width_multiplier: float
    min_component_mass_frac: float
    sweep_steps: int


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_float_list(values: tuple[float, ...], digits: int = 4) -> str:
    if not values:
        return ""
    return ";".join(_fmt_float(value, digits=digits) for value in values)


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


def _measurement_bridge(measurement: DensityGeometryMeasurement):
    return measurement.bridge_regions[0] if measurement.bridge_regions else None


def _expected_truth_peak_count(distribution_id: str) -> int:
    if distribution_id.startswith("one_peak_"):
        return 1
    if distribution_id.startswith("two_peaks_"):
        return 2
    if distribution_id.startswith("three_peaks_"):
        return 3
    raise ValueError(f"Cannot infer expected peak count for {distribution_id!r}")


def _expected_bridge_labels(distribution_id: str) -> dict[str, str]:
    if distribution_id == "two_peaks_no_bridge":
        return {"bridge_left_right": "no_bridge"}
    if distribution_id == "two_peaks_low_bridge":
        return {"bridge_left_right": "low_bridge"}
    if distribution_id == "two_peaks_high_bridge":
        return {"bridge_left_right": "high_bridge"}
    return {}


def _truth_row(distribution_id: str, truth: ComposedDensityTruth, measurement: DensityGeometryMeasurement) -> dict[str, object]:
    bridge = _measurement_bridge(measurement)
    return {
        "distribution_id": distribution_id,
        "note": V0_DISTRIBUTIONS_BY_ID[distribution_id].note,
        "expected_peak_count": _expected_truth_peak_count(distribution_id),
        "truth_peak_count": measurement.peak_count,
        "truth_peak_density": measurement.peak_density,
        "truth_peak_heights": _fmt_float_list(measurement.peak_heights),
        "truth_peak_height_min": measurement.peak_height_min,
        "truth_peak_height_median": measurement.peak_height_median,
        "truth_peak_height_max": measurement.peak_height_max,
        "truth_global_saddle_density": measurement.saddle_density,
        "truth_global_valley_density_ratio": measurement.valley_density_ratio,
        "truth_global_valley_depth": measurement.valley_depth,
        "truth_valley_density_ratio": measurement.valley_density_ratio,
        "truth_valley_depth": measurement.valley_depth,
        "truth_bridge_pair_valley_density_ratio": measurement.bridge_pair_valley_density_ratio,
        "truth_bridge_pair_valley_depth": measurement.bridge_pair_valley_depth,
        "truth_bridge_region_density_ratio": bridge.bridge_region_density_ratio if bridge else np.nan,
        "truth_bridge_region_mass_fraction": bridge.bridge_region_mass_fraction if bridge else np.nan,
        "truth_bridge_region_label": bridge.bridge_region_label if bridge else "",
        "truth_total_mass": measurement.total_mass,
        "truth_peak_count_matches_expected": bool(
            measurement.peak_count == _expected_truth_peak_count(distribution_id)
        ),
        "truth_grid_x_min": truth.composed_grid.grid.x_min if truth.composed_grid.grid else np.nan,
        "truth_grid_x_max": truth.composed_grid.grid.x_max if truth.composed_grid.grid else np.nan,
        "truth_grid_y_min": truth.composed_grid.grid.y_min if truth.composed_grid.grid else np.nan,
        "truth_grid_y_max": truth.composed_grid.grid.y_max if truth.composed_grid.grid else np.nan,
    }


def _build_candidate_row(
    *,
    distribution_id: str,
    n: int,
    seed: int,
    candidate,
    truth_measurement: DensityGeometryMeasurement,
    estimated_measurement: DensityGeometryMeasurement,
) -> dict[str, object]:
    truth_bridge = _measurement_bridge(truth_measurement)
    est_bridge = _measurement_bridge(estimated_measurement)
    truth_peak_count = truth_measurement.peak_count
    estimated_peak_count = estimated_measurement.peak_count

    truth_global_valley_ratio = truth_measurement.valley_density_ratio
    est_global_valley_ratio = estimated_measurement.valley_density_ratio
    truth_global_valley_depth = truth_measurement.valley_depth
    est_global_valley_depth = estimated_measurement.valley_depth
    truth_global_saddle = truth_measurement.saddle_density
    est_global_saddle = estimated_measurement.saddle_density

    truth_bridge_pair_ratio = truth_measurement.bridge_pair_valley_density_ratio
    est_bridge_pair_ratio = estimated_measurement.bridge_pair_valley_density_ratio
    truth_bridge_pair_depth = truth_measurement.bridge_pair_valley_depth
    est_bridge_pair_depth = estimated_measurement.bridge_pair_valley_depth

    global_valley_ratio_error = np.nan
    log_global_valley_ratio_error = np.nan
    global_valley_depth_error = np.nan
    if _is_finite(truth_global_valley_ratio) and _is_finite(est_global_valley_ratio):
        global_valley_ratio_error = float(est_global_valley_ratio - truth_global_valley_ratio)
        if truth_global_valley_ratio > 0 and est_global_valley_ratio > 0:
            log_global_valley_ratio_error = float(np.log(est_global_valley_ratio) - np.log(truth_global_valley_ratio))
    if _is_finite(truth_global_valley_depth) and _is_finite(est_global_valley_depth):
        global_valley_depth_error = float(est_global_valley_depth - truth_global_valley_depth)

    bridge_pair_valley_ratio_error = np.nan
    log_bridge_pair_valley_ratio_error = np.nan
    bridge_pair_valley_depth_error = np.nan
    if _is_finite(truth_bridge_pair_ratio) and _is_finite(est_bridge_pair_ratio):
        bridge_pair_valley_ratio_error = float(est_bridge_pair_ratio - truth_bridge_pair_ratio)
        if truth_bridge_pair_ratio > 0 and est_bridge_pair_ratio > 0:
            log_bridge_pair_valley_ratio_error = float(np.log(est_bridge_pair_ratio) - np.log(truth_bridge_pair_ratio))
    if _is_finite(truth_bridge_pair_depth) and _is_finite(est_bridge_pair_depth):
        bridge_pair_valley_depth_error = float(est_bridge_pair_depth - truth_bridge_pair_depth)

    truth_bridge_ratio = truth_bridge.bridge_region_density_ratio if truth_bridge else np.nan
    est_bridge_ratio = est_bridge.bridge_region_density_ratio if est_bridge else np.nan
    truth_bridge_mass = truth_bridge.bridge_region_mass_fraction if truth_bridge else np.nan
    est_bridge_mass = est_bridge.bridge_region_mass_fraction if est_bridge else np.nan
    truth_bridge_label = truth_bridge.bridge_region_label if truth_bridge else ""
    est_bridge_label = est_bridge.bridge_region_label if est_bridge else ""

    notes: list[str] = []
    if estimated_peak_count > truth_peak_count:
        notes.append("false_split")
    elif estimated_peak_count < truth_peak_count:
        notes.append("false_merge")
    if truth_peak_count > 1 and not _is_finite(est_global_valley_ratio):
        notes.append("no_estimated_global_valley")
    if _is_finite(truth_bridge_pair_ratio) and not _is_finite(est_bridge_pair_ratio):
        notes.append("no_estimated_bridge_pair_valley")
    if truth_bridge and est_bridge and truth_bridge_label != est_bridge_label:
        notes.append("bridge_label_mismatch")

    return {
        "distribution_id": distribution_id,
        "n": int(n),
        "seed": int(seed),
        "bandwidth_rule": candidate.bandwidth_rule,
        "geometry_scale": float(candidate.geometry_scale),
        "bandwidth_multiplier": float(candidate.bandwidth_multiplier),
        "bandwidth": float(candidate.bandwidth),
        "truth_peak_count": int(truth_peak_count),
        "estimated_peak_count": int(estimated_peak_count),
        "peak_count_error": int(estimated_peak_count - truth_peak_count),
        "false_split_flag": bool(estimated_peak_count > truth_peak_count),
        "false_merge_flag": bool(estimated_peak_count < truth_peak_count),
        "truth_peak_density": truth_measurement.peak_density,
        "estimated_peak_density": estimated_measurement.peak_density,
        "truth_peak_heights": _fmt_float_list(truth_measurement.peak_heights),
        "estimated_peak_heights": _fmt_float_list(estimated_measurement.peak_heights),
        "truth_peak_height_median": truth_measurement.peak_height_median,
        "estimated_peak_height_median": estimated_measurement.peak_height_median,
        "peak_height_median_error": float(estimated_measurement.peak_height_median - truth_measurement.peak_height_median),
        "truth_global_saddle_density": truth_global_saddle,
        "estimated_global_saddle_density": est_global_saddle,
        "global_saddle_density_error": float(est_global_saddle - truth_global_saddle) if _is_finite(truth_global_saddle) and _is_finite(est_global_saddle) else np.nan,
        "truth_global_valley_density_ratio": truth_global_valley_ratio,
        "estimated_global_valley_density_ratio": est_global_valley_ratio,
        "global_valley_ratio_error": global_valley_ratio_error,
        "log_global_valley_ratio_error": log_global_valley_ratio_error,
        "truth_global_valley_depth": truth_global_valley_depth,
        "estimated_global_valley_depth": est_global_valley_depth,
        "global_valley_depth_error": global_valley_depth_error,
        "truth_valley_density_ratio": truth_global_valley_ratio,
        "estimated_valley_density_ratio": est_global_valley_ratio,
        "valley_ratio_error": global_valley_ratio_error,
        "log_valley_ratio_error": log_global_valley_ratio_error,
        "truth_valley_depth": truth_global_valley_depth,
        "estimated_valley_depth": est_global_valley_depth,
        "valley_depth_error": global_valley_depth_error,
        "truth_bridge_pair_valley_density_ratio": truth_bridge_pair_ratio,
        "estimated_bridge_pair_valley_density_ratio": est_bridge_pair_ratio,
        "bridge_pair_valley_ratio_error": bridge_pair_valley_ratio_error,
        "log_bridge_pair_valley_ratio_error": log_bridge_pair_valley_ratio_error,
        "truth_bridge_pair_valley_depth": truth_bridge_pair_depth,
        "estimated_bridge_pair_valley_depth": est_bridge_pair_depth,
        "bridge_pair_valley_depth_error": bridge_pair_valley_depth_error,
        "truth_bridge_region_density_ratio": truth_bridge_ratio,
        "estimated_bridge_region_density_ratio": est_bridge_ratio,
        "bridge_region_density_ratio_error": float(est_bridge_ratio - truth_bridge_ratio) if _is_finite(truth_bridge_ratio) and _is_finite(est_bridge_ratio) else np.nan,
        "truth_bridge_region_mass_fraction": truth_bridge_mass,
        "estimated_bridge_region_mass_fraction": est_bridge_mass,
        "bridge_region_mass_fraction_error": float(est_bridge_mass - truth_bridge_mass) if _is_finite(truth_bridge_mass) and _is_finite(est_bridge_mass) else np.nan,
        "truth_bridge_region_label": truth_bridge_label,
        "estimated_bridge_region_label": est_bridge_label,
        "truth_peak_height_min": truth_measurement.peak_height_min,
        "estimated_peak_height_min": estimated_measurement.peak_height_min,
        "truth_peak_height_max": truth_measurement.peak_height_max,
        "estimated_peak_height_max": estimated_measurement.peak_height_max,
        "truth_total_mass": truth_measurement.total_mass,
        "estimated_total_mass": estimated_measurement.total_mass,
        "notes": ";".join(notes),
    }


def _measure_truth(distribution_id: str) -> tuple[ComposedDensityTruth, DensityGeometryMeasurement, dict[str, str]]:
    spec = V0_DISTRIBUTIONS_BY_ID[distribution_id].density_spec
    truth = compose_density_truth(spec, keep_component_fields=False)
    validation = validate_composed_density_truth(
        truth,
        expected_peak_count=_expected_truth_peak_count(distribution_id),
        expected_bridge_labels=_expected_bridge_labels(distribution_id),
    )
    measurement = measure_density_geometry(
        truth.composed_grid,
        spec,
        bridge_width_multiplier=_CONFIG.bridge_width_multiplier if _CONFIG is not None else 0.8,
        min_component_mass_frac=_CONFIG.min_component_mass_frac if _CONFIG is not None else 0.10,
        sweep_steps=_CONFIG.sweep_steps if _CONFIG is not None else 200,
    )
    if not validation.peak_count_matches:
        raise RuntimeError(f"truth peak count validation failed for {distribution_id!r}")
    return truth, measurement, {k: v for k, v in _expected_bridge_labels(distribution_id).items()}


def _summarize_rule_multiplier(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (bandwidth_rule, n, bandwidth_multiplier), sub in df.groupby(
        ["bandwidth_rule", "n", "bandwidth_multiplier"],
        sort=False,
    ):
        sub = sub.copy()
        global_valley_mask = (
            sub["truth_global_valley_density_ratio"].map(_is_finite)
            & sub["estimated_global_valley_density_ratio"].map(_is_finite)
        )
        bridge_pair_mask = (
            sub["distribution_id"].isin(BRIDGE_LADDER)
            & sub["truth_bridge_pair_valley_density_ratio"].map(_is_finite)
            & sub["estimated_bridge_pair_valley_density_ratio"].map(_is_finite)
        )
        global_valley_sub = sub.loc[global_valley_mask]
        bridge_pair_sub = sub.loc[bridge_pair_mask]
        bridge_sub = sub.loc[sub["distribution_id"].isin(BRIDGE_LADDER)]
        one_peak_sub = sub.loc[sub["distribution_id"].isin(ONE_PEAK_CASES)]
        compact_multi_sub = sub.loc[sub["distribution_id"].isin(COMPACT_MULTI_PEAK_CASES)]

        valley_order_ok = []
        bridge_order_ok = []
        for seed, seed_sub in bridge_sub.groupby("seed", sort=False):
            order = seed_sub.set_index("distribution_id")
            if not all(case in order.index for case in BRIDGE_LADDER):
                continue
            valley_vals = order.loc[list(BRIDGE_LADDER), "estimated_bridge_pair_valley_depth"].to_numpy(dtype=float)
            bridge_vals = order.loc[list(BRIDGE_LADDER), "estimated_bridge_region_density_ratio"].to_numpy(dtype=float)
            if np.all(np.isfinite(valley_vals)):
                valley_order_ok.append(bool(valley_vals[0] > valley_vals[1] > valley_vals[2]))
            if np.all(np.isfinite(bridge_vals)):
                bridge_order_ok.append(bool(bridge_vals[2] > bridge_vals[1] > bridge_vals[0]))

        row = {
            "bandwidth_rule": bandwidth_rule,
            "n": int(n),
            "bandwidth_multiplier": float(bandwidth_multiplier),
            "n_rows": int(len(sub)),
            "n_global_valley_rows": int(len(global_valley_sub)),
            "n_bridge_pair_rows": int(len(bridge_pair_sub)),
            "n_bridge_rows": int(len(bridge_sub)),
            "median_abs_global_valley_ratio_error": float(np.median(np.abs(global_valley_sub["global_valley_ratio_error"]))) if len(global_valley_sub) else np.nan,
            "median_abs_log_global_valley_ratio_error": float(np.median(np.abs(global_valley_sub["log_global_valley_ratio_error"]))) if len(global_valley_sub) else np.nan,
            "median_abs_bridge_pair_valley_ratio_error": float(np.median(np.abs(bridge_pair_sub["bridge_pair_valley_ratio_error"]))) if len(bridge_pair_sub) else np.nan,
            "median_abs_log_bridge_pair_valley_ratio_error": float(np.median(np.abs(bridge_pair_sub["log_bridge_pair_valley_ratio_error"]))) if len(bridge_pair_sub) else np.nan,
            "median_abs_bridge_pair_valley_depth_error": float(np.median(np.abs(bridge_pair_sub["bridge_pair_valley_depth_error"]))) if len(bridge_pair_sub) else np.nan,
            "valley_ordering_recovery_rate": float(np.mean(valley_order_ok)) if valley_order_ok else np.nan,
            "bridge_ordering_recovery_rate": float(np.mean(bridge_order_ok)) if bridge_order_ok else np.nan,
            "false_split_rate": float(np.mean(one_peak_sub["false_split_flag"].astype(float))) if len(one_peak_sub) else np.nan,
            "false_merge_rate": float(np.mean(compact_multi_sub["false_merge_flag"].astype(float))) if len(compact_multi_sub) else np.nan,
            "peak_count_sanity_rate": float(np.mean(sub["estimated_peak_count"].eq(sub["truth_peak_count"]).astype(float))),
            "median_selected_bandwidth": float(np.median(sub["bandwidth"])),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [
            "median_abs_log_bridge_pair_valley_ratio_error",
            "median_abs_bridge_pair_valley_depth_error",
            "median_abs_log_global_valley_ratio_error",
            "median_abs_global_valley_ratio_error",
            "valley_ordering_recovery_rate",
            "bridge_ordering_recovery_rate",
            "peak_count_sanity_rate",
            "false_split_rate",
            "false_merge_rate",
            "median_selected_bandwidth",
        ],
        ascending=[True, True, True, True, False, False, False, True, True, True],
        na_position="last",
    )
    return summary


def _select_best_rule_multiplier(summary: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[dict[str, object]] = []
    for (bandwidth_rule, n), sub in summary.groupby(["bandwidth_rule", "n"], sort=False):
        best = sub.sort_values(
            [
                "median_abs_log_bridge_pair_valley_ratio_error",
                "median_abs_bridge_pair_valley_depth_error",
                "median_abs_log_global_valley_ratio_error",
                "median_abs_global_valley_ratio_error",
                "valley_ordering_recovery_rate",
                "bridge_ordering_recovery_rate",
                "peak_count_sanity_rate",
                "false_split_rate",
                "false_merge_rate",
                "median_selected_bandwidth",
            ],
            ascending=[True, True, True, True, False, False, False, True, True, True],
            na_position="last",
        ).iloc[0]
        selected_rows.append({
            **best.to_dict(),
            "selected_by": "bridge_pair_error_then_ordering_then_sanity",
        })
    return pd.DataFrame(selected_rows)


def _write_report(
    truth_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    out_path: Path,
    *,
    config: CalibrationConfig,
) -> None:
    truth_cols = [
        "distribution_id",
        "truth_peak_count",
        "truth_global_valley_density_ratio",
        "truth_bridge_pair_valley_density_ratio",
        "truth_bridge_region_density_ratio",
        "truth_bridge_region_mass_fraction",
        "truth_bridge_region_label",
    ]
    selected_cols = [
        "n",
        "bandwidth_rule",
        "bandwidth_multiplier",
        "median_selected_bandwidth",
        "median_abs_log_bridge_pair_valley_ratio_error",
        "median_abs_bridge_pair_valley_depth_error",
        "median_abs_log_global_valley_ratio_error",
        "median_abs_global_valley_ratio_error",
        "valley_ordering_recovery_rate",
        "bridge_ordering_recovery_rate",
        "false_split_rate",
        "false_merge_rate",
        "peak_count_sanity_rate",
    ]
    lines: list[str] = [
        "# Modal V0 geometry-derived bandwidth calibration",
        "",
        f"Sample sizes: {list(config.sample_sizes)}",
        f"Seeds per sample size: {config.n_seeds}",
        f"Rules: {', '.join(config.rule_names)}",
        f"Multipliers: {list(config.multipliers)}",
        "",
        "## Truth anchor check",
        "",
        truth_df.loc[:, truth_cols].to_string(index=False),
        "",
        "## Selected rule per sample size",
        "",
        selected_df.loc[:, selected_cols].to_string(index=False),
        "",
        "## Notes",
        "",
    ]
    failures = selected_df.loc[
        (selected_df["valley_ordering_recovery_rate"] < 1.0)
        | (selected_df["bridge_ordering_recovery_rate"] < 1.0)
        | (selected_df["peak_count_sanity_rate"] < 1.0)
    ]
    lines.extend(
        [
            "Pairwise bridge metrics use the first two mode components in V0.",
            "Bridge-region metrics use explicit bridge components when present, and the no-bridge anchor keeps a synthetic corridor measurement.",
            "",
        ]
    )
    if failures.empty:
        lines.append("  none of the selected rules violated ordering or peak-count sanity on the tested seeds.")
    else:
        for _, row in failures.iterrows():
            lines.append(
                f"- n={int(row['n'])} {row['bandwidth_rule']} x {row['bandwidth_multiplier']:.2f}: "
                f"valley={row['valley_ordering_recovery_rate']:.2f}, "
                f"bridge={row['bridge_ordering_recovery_rate']:.2f}, "
                f"peak={row['peak_count_sanity_rate']:.2f}"
            )
    out_path.write_text("\n".join(lines) + "\n")


def _init_worker(
    truth_by_id: dict[str, ComposedDensityTruth],
    truth_measurement_by_id: dict[str, DensityGeometryMeasurement],
    config: CalibrationConfig,
    dist_index_by_id: dict[str, int],
) -> None:
    global _TRUTH_BY_ID, _TRUTH_MEASUREMENT_BY_ID, _CONFIG, _DIST_INDEX_BY_ID
    _TRUTH_BY_ID = truth_by_id
    _TRUTH_MEASUREMENT_BY_ID = truth_measurement_by_id
    _CONFIG = config
    _DIST_INDEX_BY_ID = dist_index_by_id


def _run_task(task: tuple[str, int, int]) -> list[dict[str, object]]:
    distribution_id, n, seed = task
    truth = _TRUTH_BY_ID[distribution_id]
    truth_measurement = _TRUTH_MEASUREMENT_BY_ID[distribution_id]
    config = _CONFIG
    if config is None:
        raise RuntimeError("worker not initialized")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(n), int(_DIST_INDEX_BY_ID[distribution_id])]))
    realization = realize_from_truth(truth, n=n, rng=rng)
    points = realization.points

    grid = truth.composed_grid
    if grid.grid is None:
        raise ValueError("truth grid is missing its canonical grid")
    grid_points = np.column_stack([grid.xx.ravel(), grid.yy.ravel()])
    dist2 = precompute_squared_distances(grid_points, points, chunk_size=2048)
    candidates = propose_bandwidth_candidates(
        points,
        rule_names=config.rule_names,
        multipliers=config.multipliers,
        knn_k=config.knn_k,
        connectivity_mass=config.connectivity_mass,
    )

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        density_flat = evaluate_isotropic_gaussian_kde_from_dist2(
            dist2,
            candidate.bandwidth,
            cell_area=grid.grid.cell_area,
            normalize_grid=True,
        )
        density = density_flat.reshape(grid.density.shape)
        measurement = measure_density_geometry(
            type(grid)(grid=grid.grid, xx=grid.xx, yy=grid.yy, density=density),
            truth.density_spec,
            bridge_width_multiplier=config.bridge_width_multiplier,
            min_component_mass_frac=config.min_component_mass_frac,
            sweep_steps=config.sweep_steps,
        )
        rows.append(
            _build_candidate_row(
                distribution_id=distribution_id,
                n=n,
                seed=seed,
                candidate=candidate,
                truth_measurement=truth_measurement,
                estimated_measurement=measurement,
            )
        )
    return rows


def _build_truth_tables(config: CalibrationConfig) -> tuple[pd.DataFrame, dict[str, ComposedDensityTruth], dict[str, DensityGeometryMeasurement]]:
    truth_rows: list[dict[str, object]] = []
    truth_by_id: dict[str, ComposedDensityTruth] = {}
    truth_measurement_by_id: dict[str, DensityGeometryMeasurement] = {}

    for distribution_id in ANCHOR_CASES:
        truth, measurement, _ = _measure_truth(distribution_id)
        truth_by_id[distribution_id] = truth
        truth_measurement_by_id[distribution_id] = measurement
        truth_rows.append(_truth_row(distribution_id, truth, measurement))

    return pd.DataFrame(truth_rows), truth_by_id, truth_measurement_by_id


def run_sweep(
    *,
    sample_sizes: tuple[int, ...],
    n_seeds: int,
    n_workers: int,
    config: CalibrationConfig,
    truth_by_id: dict[str, ComposedDensityTruth],
    truth_measurement_by_id: dict[str, DensityGeometryMeasurement],
) -> pd.DataFrame:
    tasks = [
        (distribution_id, n, seed)
        for distribution_id in ANCHOR_CASES
        for n in sample_sizes
        for seed in range(n_seeds)
    ]
    rows: list[dict[str, object]] = []

    if n_workers <= 1:
        _init_worker(truth_by_id, truth_measurement_by_id, config, {k: i for i, k in enumerate(ANCHOR_CASES)})
        for done, task_rows in enumerate(map(_run_task, tasks), start=1):
            rows.extend(task_rows)
            if done % 5 == 0 or done == len(tasks):
                print(f"  completed {done}/{len(tasks)} realizations", flush=True)
    else:
        with mp.Pool(
            processes=int(n_workers),
            initializer=_init_worker,
            initargs=(truth_by_id, truth_measurement_by_id, config, {k: i for i, k in enumerate(ANCHOR_CASES)}),
        ) as pool:
            for done, task_rows in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), start=1):
                rows.extend(task_rows)
                if done % 5 == 0 or done == len(tasks):
                    print(f"  completed {done}/{len(tasks)} realizations", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[80, 160])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rules", nargs="*", default=list(RULE_NAMES))
    parser.add_argument("--multipliers", nargs="*", type=float, default=list(DEFAULT_MULTIPLIERS))
    parser.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    parser.add_argument("--connectivity-mass", type=float, default=DEFAULT_CONNECTIVITY_MASS)
    parser.add_argument("--bridge-width-multiplier", type=float, default=0.8)
    parser.add_argument("--min-component-mass-frac", type=float, default=0.10)
    parser.add_argument("--sweep-steps", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sample_sizes = tuple(int(n) for n in args.sample_sizes)
    config = CalibrationConfig(
        sample_sizes=sample_sizes,
        n_seeds=int(args.n_seeds),
        rule_names=tuple(str(rule) for rule in args.rules),
        multipliers=tuple(float(multiplier) for multiplier in args.multipliers),
        knn_k=int(args.knn_k),
        connectivity_mass=float(args.connectivity_mass),
        bridge_width_multiplier=float(args.bridge_width_multiplier),
        min_component_mass_frac=float(args.min_component_mass_frac),
        sweep_steps=int(args.sweep_steps),
    )

    truth_df, truth_by_id, truth_measurement_by_id = _build_truth_tables(config)
    candidate_df = run_sweep(
        sample_sizes=sample_sizes,
        n_seeds=config.n_seeds,
        n_workers=int(args.n_workers),
        config=config,
        truth_by_id=truth_by_id,
        truth_measurement_by_id=truth_measurement_by_id,
    )
    rule_multiplier_summary = _summarize_rule_multiplier(candidate_df)
    selected_df = _select_best_rule_multiplier(rule_multiplier_summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = args.out_dir / "modal_bandwidth_tuning_truth.csv"
    candidate_path = args.out_dir / "modal_bandwidth_tuning_candidates.csv"
    summary_path = args.out_dir / "modal_bandwidth_tuning_summary.csv"
    rule_summary_path = args.out_dir / "modal_bandwidth_tuning_rule_summary.csv"
    report_path = args.out_dir / "modal_bandwidth_tuning_report.md"

    truth_df.to_csv(truth_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    rule_multiplier_summary.to_csv(rule_summary_path, index=False)
    selected_df.to_csv(summary_path, index=False)
    _write_report(truth_df, rule_multiplier_summary, selected_df, report_path, config=config)

    print(f"Saved: {truth_path}")
    print(f"Saved: {candidate_path}")
    print(f"Saved: {rule_summary_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
