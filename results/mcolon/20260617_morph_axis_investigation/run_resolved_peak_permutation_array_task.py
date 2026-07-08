"""One SGE array-task slice of the one_peak_compact vs three_peaks_compact
pooled-label permutation null test.

Each task computes `--n-draws-total / --n-tasks` permutation draws (the
parallelizable unit factored out as `run_resolved_peak_permutation_draws`)
using a task-local RNG stream, and writes its null_delta slice plus the
(cheap, identical-across-tasks) observed_delta and observed summaries to a
per-task .npz file. After all tasks complete, run
`merge_resolved_peak_permutation_outputs.py` to concatenate slices and
perform the final `run_empirical_null_test` reduction -- reduction must run
once, over the pooled draws, not per task.

Task index resolution, in priority order:
    --task-id (1-based, for manual/local testing)
    $SGE_TASK_ID (1-based, set by SGE for array jobs)

Usage (local smoke test of task 1 of 4, 20 draws total):
    conda run -n segmentation_grounded_sam --no-capture-output python \
        run_resolved_peak_permutation_array_task.py \
        --task-id 1 --n-tasks 4 --n-draws-total 20

Usage (SGE array job): see run_resolved_peak_permutation_array.qsub
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphseq_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/morphseq_xdg_cache")
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.core.density_composition import compose_density_truth, realize_from_truth  # noqa: E402
from morphseq_investigation.core.resolved_peak_analysis import (  # noqa: E402
    compute_observed_delta,
    run_resolved_peak_permutation_draws,
)
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS_BY_ID  # noqa: E402

from run_resolved_peak_v0_smoke_test import ANALYSIS_SPEC  # noqa: E402

OUT_DIR = RUN_DIR / "morphseq_investigation" / "tables" / "resolved_peak_permutation_array"

REFERENCE_SCENARIO = "one_peak_compact"
TARGET_SCENARIO = "three_peaks_compact"
POOLED_N = 80


def _resolve_task_index(args: argparse.Namespace) -> int:
    if args.task_id is not None:
        return args.task_id
    sge_task_id = os.environ.get("SGE_TASK_ID")
    if sge_task_id is None:
        raise SystemExit("No --task-id given and $SGE_TASK_ID is not set.")
    return int(sge_task_id)


def _draws_for_task(task_index_zero_based: int, n_tasks: int, n_draws_total: int) -> int:
    """Split n_draws_total across n_tasks as evenly as possible (remainder to the first tasks)."""
    base, remainder = divmod(n_draws_total, n_tasks)
    return base + (1 if task_index_zero_based < remainder else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=None, help="1-based task index; overrides $SGE_TASK_ID.")
    parser.add_argument("--n-tasks", type=int, required=True, help="Total number of array tasks.")
    parser.add_argument("--n-draws-total", type=int, default=500, help="Total permutation draws across all tasks.")
    parser.add_argument("--n", type=int, default=POOLED_N, help="Sample size per group (must match observed groups).")
    parser.add_argument("--realization-seed", type=int, default=7, help="Seed for realizing the observed reference/target point clouds. Must match across all tasks and the merge step.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    task_id = _resolve_task_index(args)
    if not (1 <= task_id <= args.n_tasks):
        raise SystemExit(f"task_id={task_id} out of range for n_tasks={args.n_tasks}")
    task_index_zero_based = task_id - 1

    n_draws = _draws_for_task(task_index_zero_based, args.n_tasks, args.n_draws_total)

    reference_dist = V0_DISTRIBUTIONS_BY_ID[REFERENCE_SCENARIO]
    target_dist = V0_DISTRIBUTIONS_BY_ID[TARGET_SCENARIO]

    reference_truth = compose_density_truth(reference_dist.density_spec)
    target_truth = compose_density_truth(target_dist.density_spec)

    # Same realization seed on every task -> identical observed reference/target
    # point clouds across the whole array, so observed_delta agrees bit-for-bit.
    # Matches run_resolved_peak_v0_smoke_test.run_permutation_comparison exactly.
    rng_reference = np.random.default_rng(np.random.SeedSequence([args.realization_seed, 1]))
    rng_target = np.random.default_rng(np.random.SeedSequence([args.realization_seed, 2]))
    reference_points = realize_from_truth(reference_truth, n=args.n, rng=rng_reference).points
    target_points = realize_from_truth(target_truth, n=args.n, rng=rng_target).points

    canonical_grid = reference_truth.composed_grid.grid

    if n_draws == 0:
        print(f"Task {task_id}/{args.n_tasks}: 0 draws assigned, nothing to do.")
        null_deltas = {}
    else:
        # Distinct, reproducible per-task draw stream: same [seed, 3] root as
        # the serial path's permutation RNG, with task_id appended so tasks
        # never collide, independent of $JOB_ID or wall-clock time.
        draw_rng = np.random.default_rng(np.random.SeedSequence([args.realization_seed, 3, task_id]))
        null_deltas = run_resolved_peak_permutation_draws(
            reference_points=reference_points,
            target_points=target_points,
            analysis_spec=ANALYSIS_SPEC,
            canonical_grid=canonical_grid,
            n_draws=n_draws,
            rng=draw_rng,
        )

    observed_delta, reference_summary, target_summary = compute_observed_delta(
        reference_points=reference_points,
        target_points=target_points,
        analysis_spec=ANALYSIS_SPEC,
        canonical_grid=canonical_grid,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"task_{task_id:03d}_of_{args.n_tasks:03d}.npz"
    np.savez(
        out_path,
        task_id=task_id,
        n_tasks=args.n_tasks,
        n_draws=n_draws,
        metrics=np.array(list(observed_delta.keys())),
        observed_delta=np.array(list(observed_delta.values()), dtype=float),
        reference_metric_values=np.array(
            [getattr(reference_summary, m) for m in observed_delta], dtype=float
        ),
        target_metric_values=np.array(
            [getattr(target_summary, m) for m in observed_delta], dtype=float
        ),
        **{f"null_delta__{metric}": values for metric, values in null_deltas.items()},
    )
    print(f"Task {task_id}/{args.n_tasks}: wrote {n_draws} draws -> {out_path}")


if __name__ == "__main__":
    main()
