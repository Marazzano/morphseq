"""Quick diagnostic for an iterative, control-anchored correction field.

This operates only on the existing two-dimensional aligned-UMAP initializer::

    xy_corrected[i, t] = xy_raw[i, t] + field[experiment[i], t]

The bridge experiment (20251207_pbx) is fixed.  At each iteration and overlap
bin, controls in another experiment are softly matched to bridge controls.
The robust mean match residual is smoothed over time and added as a damped
increment to that experiment's field.  No time coordinate or latent feature is
changed, and the same translation is applied to every genotype in an
experiment/time bin.

This is intentionally an MVP, not a production batch-correction API.  Its main
output is a pair of HTML viewers and a small set of diagnostics that answer:
does a translation-only field visibly close the experiment seam?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_CACHE = Path("/tmp") / "morphseq_20260703_correction_field"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
_CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist, pdist, squareform

import analyze.trajectory_condensation as tc


DEFAULT_INPUT = _HERE / "figures" / "condensed_raw_zmub_relative_api" / "condensed_positions.npz"
DEFAULT_OUT = _HERE / "figures" / "iterative_correction_field_v0"
REFERENCE_EXPERIMENT = "20251207_pbx"
CONTROL_LABELS = {"inj_ctrl", "wik_ab"}


def experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if parts[1].isalpha() else parts[0]


def robust_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean after one conservative MAD trim on residual magnitude."""
    if len(values) == 0:
        return np.full(2, np.nan)
    center = np.average(values, axis=0, weights=weights)
    radii = np.linalg.norm(values - center, axis=1)
    med = np.median(radii)
    mad = 1.4826 * np.median(np.abs(radii - med))
    keep = radii <= med + 3.0 * max(mad, 1e-8)
    if not np.any(keep):
        keep = np.ones(len(values), dtype=bool)
    return np.average(values[keep], axis=0, weights=weights[keep])


def soft_match_residual(target: np.ndarray, reference: np.ndarray, bandwidth: float) -> np.ndarray:
    """Return one translation residual from soft target-to-reference matches."""
    d2 = cdist(target, reference, metric="sqeuclidean")
    # Row shifting keeps nearest matches numerically alive even before clouds overlap.
    logits = -(d2 - d2.min(axis=1, keepdims=True)) / (2.0 * bandwidth**2)
    weights = np.exp(np.clip(logits, -50.0, 0.0))
    weights /= weights.sum(axis=1, keepdims=True)
    matched_reference = weights @ reference
    confidence = weights.max(axis=1)
    return robust_weighted_mean(matched_reference - target, confidence)


def smooth_residual(
    observed_times: np.ndarray,
    residuals: np.ndarray,
    all_times: np.ndarray,
    smoothness: float,
) -> np.ndarray:
    """Smooth residuals; do not alter pre-overlap coordinates.

    The early 20260304-only initializer defines the incoming coordinate frame.
    Its field must remain zero through 44 hpf so that the correction learned at
    the first overlap bin can repair, rather than preserve, the 44->48 seam.
    After the evidence interval, constant extrapolation preserves continuity.
    """
    order = np.argsort(observed_times)
    x = observed_times[order]
    y = residuals[order]
    result = np.empty((len(all_times), 2), dtype=float)
    for dim in range(2):
        if len(x) == 1:
            result[:, dim] = y[0, dim]
            continue
        degree = min(3, len(x) - 1)
        variance_scale = max(float(np.var(y[:, dim])), 1e-6)
        spline = UnivariateSpline(
            x,
            y[:, dim],
            k=degree,
            s=smoothness * len(x) * variance_scale,
            ext=3,
        )
        result[:, dim] = spline(all_times)
        result[all_times < x[0], dim] = 0.0
    return result


def cross_same_ratio(coords: np.ndarray, experiments: np.ndarray) -> float:
    if len(coords) < 3 or len(set(experiments)) < 2:
        return np.nan
    distances = squareform(pdist(coords))
    upper = np.triu_indices(len(coords), k=1)
    same = (experiments[:, None] == experiments[None, :])[upper]
    values = distances[upper]
    if not np.any(same) or not np.any(~same):
        return np.nan
    return float(values[~same].mean() / values[same].mean())


def diagnostic_rows(
    raw: np.ndarray,
    corrected: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    experiments: np.ndarray,
    controls: np.ndarray,
) -> list[dict]:
    rows = []
    for ti, time in enumerate(times):
        use = mask[:, ti] & controls
        exps = experiments[use]
        if len(set(exps)) < 2:
            continue
        rows.append(
            {
                "time_bin": float(time),
                "n_controls": int(use.sum()),
                "n_experiments": len(set(exps)),
                "cross_same_ratio_raw": cross_same_ratio(raw[use, ti], exps),
                "cross_same_ratio_corrected": cross_same_ratio(corrected[use, ti], exps),
            }
        )
    return rows


def render_field(fields: np.ndarray, experiments: np.ndarray, times: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ei, experiment in enumerate(experiments):
        axes[0].plot(times, fields[ei, :, 0], marker="o", ms=3, label=experiment)
        axes[1].plot(times, fields[ei, :, 1], marker="o", ms=3, label=experiment)
    for axis, label in zip(axes, ("initializer x shift", "initializer y shift")):
        axis.axhline(0.0, color="0.7", lw=1)
        axis.set_xlabel("hpf")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Iterative control-derived translation field")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_overlap_comparison(
    raw: np.ndarray,
    corrected: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    experiments: np.ndarray,
    controls: np.ndarray,
    out_path: Path,
) -> None:
    overlap_indices = [
        ti
        for ti in range(len(times))
        if len(set(experiments[mask[:, ti] & controls])) >= 2
    ]
    if overlap_indices and overlap_indices[0] > 0:
        overlap_indices.insert(0, overlap_indices[0] - 1)
    colors = {
        "20251207_pbx": "#6A3D9A",
        "20260304": "#1B9E77",
        "20260306": "#D95F02",
    }
    fig, axes = plt.subplots(2, len(overlap_indices), figsize=(2.6 * len(overlap_indices), 5.2))
    all_xy = np.concatenate(
        [raw[mask], corrected[mask]], axis=0
    )
    xlim = np.nanpercentile(all_xy[:, 0], [1, 99])
    ylim = np.nanpercentile(all_xy[:, 1], [1, 99])
    for column, ti in enumerate(overlap_indices):
        use = mask[:, ti] & controls
        for row, (coords, stage) in enumerate(((raw, "raw"), (corrected, "corrected"))):
            axis = axes[row, column]
            for experiment in sorted(set(experiments[use])):
                selected = use & (experiments == experiment)
                axis.scatter(
                    coords[selected, ti, 0],
                    coords[selected, ti, 1],
                    s=18,
                    alpha=0.8,
                    color=colors.get(experiment, "0.5"),
                    label=experiment,
                )
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{times[ti]:g} hpf")
            if column == 0:
                axis.set_ylabel(stage)
            axis.set_xticks([])
            axis.set_yticks([])
    present_experiments = sorted(set(experiments[controls]))
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=colors.get(exp, "0.5"), label=exp)
        for exp in present_experiments
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Control overlap: translation field gate", y=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--smoothness", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input, allow_pickle=True)
    raw = np.asarray(data["x0"], dtype=float)
    mask = np.asarray(data["mask"], dtype=bool)
    times = np.asarray(data["time_values"], dtype=float)
    embryo_ids = np.asarray(data["embryo_ids"]).astype(str)
    labels = np.asarray(data["labels"]).astype(str)
    experiment_per_embryo = np.asarray([experiment_of(value) for value in embryo_ids])
    experiment_names = np.asarray(sorted(set(experiment_per_embryo)))
    if REFERENCE_EXPERIMENT not in experiment_names:
        raise ValueError(f"Reference experiment {REFERENCE_EXPERIMENT!r} is absent")
    controls = np.isin(labels, sorted(CONTROL_LABELS))

    # A data-scaled matching radius: twice the median nearest-control spacing.
    if args.bandwidth is None:
        spacings = []
        for ti in range(len(times)):
            use = mask[:, ti] & controls
            if use.sum() >= 3:
                distances = cdist(raw[use, ti], raw[use, ti])
                np.fill_diagonal(distances, np.inf)
                spacings.extend(distances.min(axis=1).tolist())
        bandwidth = 2.0 * float(np.median(spacings))
    else:
        bandwidth = args.bandwidth
    if bandwidth <= 0:
        raise ValueError("Matching bandwidth must be positive")

    fields = np.zeros((len(experiment_names), len(times), 2), dtype=float)
    exp_index = {name: idx for idx, name in enumerate(experiment_names)}
    corrected = raw.copy()
    history = []
    best_score = np.inf
    best_iteration = 0
    best_fields = fields.copy()
    best_corrected = corrected.copy()

    for iteration in range(args.iterations):
        increments = np.zeros_like(fields)
        observed_counts = {}
        for experiment in experiment_names:
            if experiment == REFERENCE_EXPERIMENT:
                continue
            observed_times = []
            observed_residuals = []
            target_exp = experiment_per_embryo == experiment
            ref_exp = experiment_per_embryo == REFERENCE_EXPERIMENT
            for ti, time in enumerate(times):
                target = mask[:, ti] & controls & target_exp
                reference = mask[:, ti] & controls & ref_exp
                if target.sum() < 2 or reference.sum() < 2:
                    continue
                residual = soft_match_residual(
                    corrected[target, ti], corrected[reference, ti], bandwidth
                )
                observed_times.append(time)
                observed_residuals.append(residual)
            observed_counts[experiment] = len(observed_times)
            if not observed_times:
                continue
            smoothed = smooth_residual(
                np.asarray(observed_times),
                np.asarray(observed_residuals),
                times,
                args.smoothness,
            )
            increments[exp_index[experiment]] = args.step_size * smoothed

        fields += increments
        # Reapply the accumulated field to the untouched initializer.
        corrected = raw.copy()
        for embryo_index, experiment in enumerate(experiment_per_embryo):
            corrected[embryo_index, mask[embryo_index]] += fields[
                exp_index[experiment], mask[embryo_index]
            ]
        max_update = float(np.linalg.norm(increments, axis=2).max())
        current_diagnostics = pd.DataFrame(
            diagnostic_rows(raw, corrected, mask, times, experiment_per_embryo, controls)
        )
        mixing_score = float(
            np.mean(np.abs(np.log(current_diagnostics["cross_same_ratio_corrected"])))
        )
        history.append(
            {
                "iteration": iteration + 1,
                "max_update": max_update,
                "mixing_score": mixing_score,
                **observed_counts,
            }
        )
        if mixing_score < best_score:
            best_score = mixing_score
            best_iteration = iteration + 1
            best_fields = fields.copy()
            best_corrected = corrected.copy()
        print(
            f"iteration {iteration + 1:02d}: max field update={max_update:.5f}, "
            f"mixing score={mixing_score:.5f}"
        )
        if max_update < args.tolerance:
            break

    # Correspondences can oscillate slightly; emit the best observed iterate.
    fields = best_fields
    corrected = best_corrected
    rows = diagnostic_rows(raw, corrected, mask, times, experiment_per_embryo, controls)
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(args.out_dir / "overlap_diagnostics.csv", index=False)
    pd.DataFrame(history).to_csv(args.out_dir / "iteration_history.csv", index=False)

    overlap_evidence = np.zeros((len(experiment_names), len(times)), dtype=bool)
    ref_exp = experiment_per_embryo == REFERENCE_EXPERIMENT
    for ei, experiment in enumerate(experiment_names):
        if experiment == REFERENCE_EXPERIMENT:
            continue
        target_exp = experiment_per_embryo == experiment
        for ti in range(len(times)):
            overlap_evidence[ei, ti] = bool(
                (mask[:, ti] & controls & target_exp).sum() >= 2
                and (mask[:, ti] & controls & ref_exp).sum() >= 2
            )

    out_npz = args.out_dir / "corrected_initializer.npz"
    np.savez(
        out_npz,
        x0=corrected,
        x0_raw=raw,
        mask=mask,
        time_values=times,
        embryo_ids=embryo_ids,
        labels=labels,
        experiments=experiment_per_embryo,
        experiment_names=experiment_names,
        fields=fields,
        overlap_evidence=overlap_evidence,
        reference_experiment=REFERENCE_EXPERIMENT,
        bandwidth=bandwidth,
    )
    render_field(fields, experiment_names, times, args.out_dir / "translation_field.png")
    render_overlap_comparison(
        raw,
        corrected,
        mask,
        times,
        experiment_per_embryo,
        controls,
        args.out_dir / "control_overlap_before_after.png",
    )

    experiment_colors = {
        "20251207_pbx": "#6A3D9A",
        "20260304": "#1B9E77",
        "20260306": "#D95F02",
    }
    genotype_colors = {
        "inj_ctrl": "#2166AC",
        "wik_ab": "#808080",
        "pbx1b_crispant": "#9467bd",
        "pbx4_crispant": "#F7B267",
        "pbx1b_pbx4_crispant": "#B2182B",
    }
    tc.time_slice_html(
        corrected,
        mask,
        times,
        labels=experiment_per_embryo,
        color_map=experiment_colors,
        embryo_ids=embryo_ids,
        title="iterative translation-field corrected initializer | experiment",
        output_path=args.out_dir / "corrected_by_experiment.html",
    )
    tc.time_slice_html(
        corrected,
        mask,
        times,
        labels=labels,
        color_map=genotype_colors,
        embryo_ids=embryo_ids,
        title="iterative translation-field corrected initializer | genotype",
        output_path=args.out_dir / "corrected_by_genotype.html",
    )

    summary = {
        "input": str(args.input),
        "reference_experiment": REFERENCE_EXPERIMENT,
        "control_labels": sorted(CONTROL_LABELS),
        "iterations_completed": len(history),
        "selected_iteration": best_iteration,
        "step_size": args.step_size,
        "bandwidth": bandwidth,
        "smoothness": args.smoothness,
        "tolerance": args.tolerance,
        "mean_overlap_ratio_raw": float(diagnostics["cross_same_ratio_raw"].mean()),
        "mean_overlap_ratio_corrected": float(diagnostics["cross_same_ratio_corrected"].mean()),
        "mixing_score": best_score,
    }
    boundary_before = np.flatnonzero(times < times[np.flatnonzero(np.any(overlap_evidence, axis=0))[0]])[-1]
    boundary_after = boundary_before + 1
    boundary_common = (
        mask[:, boundary_before]
        & mask[:, boundary_after]
        & controls
        & (experiment_per_embryo == "20260304")
    )
    if np.any(boundary_common):
        raw_jump = raw[boundary_common, boundary_after] - raw[boundary_common, boundary_before]
        corrected_jump = (
            corrected[boundary_common, boundary_after]
            - corrected[boundary_common, boundary_before]
        )
        summary.update(
            {
                "boundary": f"{times[boundary_before]:g}->{times[boundary_after]:g} hpf",
                "boundary_n_paired_controls": int(boundary_common.sum()),
                "boundary_centroid_jump_raw": float(np.linalg.norm(raw_jump.mean(axis=0))),
                "boundary_centroid_jump_corrected": float(
                    np.linalg.norm(corrected_jump.mean(axis=0))
                ),
                "boundary_mean_paired_jump_raw": float(
                    np.linalg.norm(raw_jump, axis=1).mean()
                ),
                "boundary_mean_paired_jump_corrected": float(
                    np.linalg.norm(corrected_jump, axis=1).mean()
                ),
            }
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostic bundle -> {args.out_dir}")


if __name__ == "__main__":
    main()
