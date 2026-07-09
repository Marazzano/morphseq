"""Merge per-task .npz outputs from run_resolved_peak_permutation_array.qsub
into the same resolved_peak_null_test_table schema the serial
run_resolved_peak_v0_smoke_test.py --run-permutation path produces.

Run after all array tasks complete:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/merge_resolved_peak_permutation_outputs.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.core.resolved_peak_analysis import EmpiricalNullSpec, reduce_permutation_null_test  # noqa: E402
from morphseq_investigation.core.resolved_peak_metrics import ResolvedPeakDistributionSummary, ResolvedPeakRunContext  # noqa: E402

from run_resolved_peak_permutation_array_task import OUT_DIR as ARRAY_OUT_DIR  # noqa: E402
from run_resolved_peak_v0_smoke_test import ANALYSIS_SPEC, OUT_DIR as SMOKE_OUT_DIR  # noqa: E402


def _bare_summary(distribution_id: str, source_type: str, metric_names: list[str], values: np.ndarray) -> ResolvedPeakDistributionSummary:
    """Reconstruct just enough of ResolvedPeakDistributionSummary to feed
    reduce_permutation_null_test -- only metric_name attributes and
    distribution_id/source_type are read there, not the full object.
    """
    fields = {name: float(value) for name, value in zip(metric_names, values)}
    return ResolvedPeakDistributionSummary(
        distribution_id=distribution_id,
        source_type=source_type,
        number_of_peaks=int(fields.get("number_of_peaks", np.nan)) if "number_of_peaks" in fields else 0,
        assigned_support_fraction=fields.get("assigned_support_fraction", float("nan")),
        unassigned_support_fraction=float("nan"),
        across_peak_total_support_fraction_mean=float("nan"),
        across_peak_total_support_fraction_skew=float("nan"),
        across_peak_radius_mean=fields.get("across_peak_radius_mean", float("nan")),
        across_peak_radius_skew=float("nan"),
        across_peak_cv_radius_from_center_mean=fields.get("across_peak_cv_radius_from_center_mean", float("nan")),
        across_peak_distance_mean=fields.get("across_peak_distance_mean", float("nan")),
        across_peak_r80_density_mean=fields.get("across_peak_r80_density_mean", float("nan")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-dir", type=Path, default=ARRAY_OUT_DIR)
    parser.add_argument(
        "--n-tasks", type=int, default=None,
        help="If given, merge only task_*_of_{n_tasks:03d}.npz files. Prevents stale "
             "outputs from a prior run with a different task count from colliding with "
             "the current run. If omitted, all task_*_of_*.npz are merged and a mixed "
             "n_tasks is a hard error.",
    )
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--realization-seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=SMOKE_OUT_DIR)
    parser.add_argument(
        "--out-name", default="resolved_peak_null_test_array.csv",
        help="Output filename, kept distinct from the serial path's resolved_peak_null_test.csv.",
    )
    args = parser.parse_args()

    pattern = f"task_*_of_{args.n_tasks:03d}.npz" if args.n_tasks is not None else "task_*_of_*.npz"
    files = sorted(glob.glob(str(args.array_dir / pattern)))
    if not files:
        raise SystemExit(f"No array task outputs matching {pattern!r} found in {args.array_dir}")

    metrics: list[str] | None = None
    observed_delta_ref: np.ndarray | None = None
    reference_values_ref: np.ndarray | None = None
    target_values_ref: np.ndarray | None = None
    null_delta_by_metric: dict[str, list[np.ndarray]] = {}
    total_draws = 0
    expected_n_tasks: int | None = None
    seen_task_ids: set[int] = set()

    for path in files:
        data = np.load(path, allow_pickle=True)
        file_metrics = list(data["metrics"])
        if metrics is None:
            metrics = file_metrics
            null_delta_by_metric = {metric: [] for metric in metrics}
        elif file_metrics != metrics:
            raise SystemExit(f"{path}: metric order {file_metrics} does not match {metrics}")

        if observed_delta_ref is None:
            observed_delta_ref = data["observed_delta"]
            reference_values_ref = data["reference_metric_values"]
            target_values_ref = data["target_metric_values"]
        elif not np.allclose(data["observed_delta"], observed_delta_ref):
            raise SystemExit(
                f"{path}: observed_delta {data['observed_delta']} does not match "
                f"{observed_delta_ref} from an earlier task -- realization seeds diverged across tasks."
            )

        n_tasks = int(data["n_tasks"])
        expected_n_tasks = expected_n_tasks or n_tasks
        if n_tasks != expected_n_tasks:
            raise SystemExit(f"{path}: n_tasks={n_tasks} does not match {expected_n_tasks} from an earlier task")
        seen_task_ids.add(int(data["task_id"]))

        n_draws = int(data["n_draws"])
        total_draws += n_draws
        for metric in metrics:
            key = f"null_delta__{metric}"
            if key in data.files and n_draws > 0:
                null_delta_by_metric[metric].append(data[key])

    missing_tasks = sorted(set(range(1, expected_n_tasks + 1)) - seen_task_ids)
    if missing_tasks:
        raise SystemExit(f"Missing array task output(s) for task_id(s): {missing_tasks} (found {len(files)} of {expected_n_tasks})")

    print(f"Merged {len(files)} task file(s), {total_draws} total permutation draws.")

    observed_delta = dict(zip(metrics, observed_delta_ref))
    reference_summary = _bare_summary("observed_reference", "empirical", metrics, reference_values_ref)
    target_summary = _bare_summary("observed_target", "empirical", metrics, target_values_ref)
    null_deltas = {
        metric: np.concatenate(chunks) if chunks else np.array([], dtype=float)
        for metric, chunks in null_delta_by_metric.items()
    }

    null_spec = EmpiricalNullSpec(method="pooled_label_permutation", n_draws=total_draws, alternative="two-sided")
    context = ResolvedPeakRunContext(
        analysis_id="resolved_peak_v0_smoke_test_permutation_array",
        scenario_id="one_peak_compact_vs_three_peaks_compact",
        replicate_id=f"seed{args.realization_seed}",
        seed=args.realization_seed,
        n=args.n,
        bandwidth_rule=ANALYSIS_SPEC.bandwidth_rule,
        bandwidth_multiplier=ANALYSIS_SPEC.bandwidth_multiplier,
        bandwidth_value=float("nan"),
        peak_detector_method=ANALYSIS_SPEC.peak_detector_method,
        canonical_grid_id="one_peak_compact",
        assignment_rule=ANALYSIS_SPEC.assignment_rule,
    )

    null_test_df = reduce_permutation_null_test(
        observed_delta=observed_delta,
        reference_summary=reference_summary,
        target_summary=target_summary,
        null_deltas=null_deltas,
        analysis_spec=ANALYSIS_SPEC,
        null_spec=null_spec,
        context=context,
        metrics=tuple(metrics),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    null_test_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(null_test_df.to_string(index=False))


if __name__ == "__main__":
    main()
