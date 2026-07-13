"""Dedicated timing benchmark for the bootstrap-vote speed optimizations
(Steps 1-2 of the resolve-hot-path speedup plan).

Deliberately NOT a pytest test: wall-clock assertions are fragile across
laptops, cluster nodes, CI workers, BLAS builds, CPU load, and thermal state.
This script measures performance and emits JSON/CSV; it enforces no runtime
threshold. Correctness is covered separately by
tests/test_vote_only_resolve_speedup.py and v0/validate_v0_resolved_peaks.py.

Checkpoints, each run against the SAME fixed distribution/seed/draw count:
  baseline            -- pre-Step-1/2 equivalent: full resolve per draw
                          (resolve_points_with_analysis_spec, discarding the
                          extra fields) + connectivity always computed
  step1_only          -- detection-only primitive, connectivity still always computed
  step2_only          -- full resolve per draw, but connectivity gated off
  combined            -- both (the current, actually-shipped code path)

Usage:
  python v0/benchmark_bootstrap_peak_resolution.py [--n-draws 1000] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from morphseq_investigation.core import bandwidth_tuning
from morphseq_investigation.core.distribution_records import derive_shared_grid
from morphseq_investigation.core.resolved_peak_analysis import (
    ResolvedPeakAnalysisSpec,
    _compute_peak_detection_with_analysis_spec,
    _resolve_points_single_pass,
)

SEED = 20260617
N_POINTS = 200
BOOTSTRAP_SAMPLE_FRACTION = 0.80


def _fixed_distribution(seed: int = SEED, n_points: int = N_POINTS) -> np.ndarray:
    """Two well-separated clusters -- representative of a typical resolve, not
    a worst/best case. Fixed seed so every checkpoint sees identical draws."""
    rng = np.random.default_rng(seed)
    n_left = n_points // 2
    n_right = n_points - n_left
    left = rng.normal(loc=[-2.0, 0.0], scale=0.3, size=(n_left, 2))
    right = rng.normal(loc=[2.0, 0.0], scale=0.3, size=(n_right, 2))
    return np.concatenate([left, right], axis=0)


def _analysis_spec() -> ResolvedPeakAnalysisSpec:
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule="median_kNN_distance",
        bandwidth_multiplier=1.0,
        peak_detector_method="kde_peak_basins_sample_support",
        min_sample_fraction=0.10,
    )


def _draws(points: np.ndarray, n_draws: int, seed: int, sample_fraction: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(points)
    sample_size = max(1, int(round(n * sample_fraction)))
    return [
        points[rng.integers(0, n, size=sample_size)]
        for _ in range(n_draws)
    ]


def _force_connectivity_always_on():
    """Context manager-free monkeypatch: bandwidth_geometry_scales always
    computes connectivity_90_radius, matching pre-Step-2 behavior (this
    module's own `bandwidth_geometry_scales(points)` call before the
    `include_connectivity_radius` param existed always defaulted to True
    internally -- i.e. it never gated the union-find sweep off)."""
    orig = bandwidth_tuning.bandwidth_geometry_scales

    def _always_full(pts, **kwargs):
        kwargs["include_connectivity_radius"] = True
        return orig(pts, **kwargs)

    return orig, _always_full


def _draw_baseline(points, grid, spec) -> tuple[int, tuple]:
    """Pre-Step-1/2 equivalent: full resolve, connectivity always computed
    (both Step 1's detection-only shortcut and Step 2's connectivity gate
    reverted via monkeypatch -- resolve_points_with_analysis_spec's internals
    are otherwise unchanged, so this isolates "neither optimization applied")."""
    orig, always_full = _force_connectivity_always_on()
    bandwidth_tuning.bandwidth_geometry_scales = always_full
    try:
        resolved = _resolve_points_single_pass(
            distribution_id="bench_draw", points=points, canonical_grid=grid, analysis_spec=spec,
        )
    finally:
        bandwidth_tuning.bandwidth_geometry_scales = orig
    centers = tuple(peak.geometry.center_coordinate for peak in resolved.peaks)
    return resolved.number_of_peaks, centers


def _draw_step1_only(points, grid, spec) -> tuple[int, tuple]:
    """Detection-only primitive (Step 1), but connectivity still unconditionally
    computed (Step 2 reverted via monkeypatch) -- isolates Step 1's own effect."""
    orig, always_full = _force_connectivity_always_on()
    bandwidth_tuning.bandwidth_geometry_scales = always_full
    try:
        _density_grid, _points_array, detection_result = _compute_peak_detection_with_analysis_spec(
            points, grid, spec,
        )
    finally:
        bandwidth_tuning.bandwidth_geometry_scales = orig
    accepted = tuple(d for d in detection_result.candidate_details if d.accepted)
    centers = tuple((float(d.peak_x), float(d.peak_y)) for d in accepted)
    return len(accepted), centers


def _draw_step2_only(points, grid, spec) -> tuple[int, tuple]:
    """Full resolve per draw (pre-Step-1 shape, no monkeypatch needed --
    resolve_points_with_analysis_spec already gates connectivity off
    internally, i.e. Step 2 is live here; Step 1's detection-only shortcut is
    the thing NOT applied on this checkpoint)."""
    resolved = _resolve_points_single_pass(
        distribution_id="bench_draw", points=points, canonical_grid=grid, analysis_spec=spec,
    )
    centers = tuple(peak.geometry.center_coordinate for peak in resolved.peaks)
    return resolved.number_of_peaks, centers


def _draw_combined(points, grid, spec) -> tuple[int, tuple]:
    """Current shipped code path: detection-only + connectivity gated."""
    _density_grid, _points_array, detection_result = _compute_peak_detection_with_analysis_spec(
        points, grid, spec,
    )
    accepted = tuple(d for d in detection_result.candidate_details if d.accepted)
    centers = tuple((float(d.peak_x), float(d.peak_y)) for d in accepted)
    return len(accepted), centers


CHECKPOINTS = {
    "baseline": _draw_baseline,
    "step1_only": _draw_step1_only,
    "step2_only": _draw_step2_only,
    "combined": _draw_combined,
}


def run_checkpoint(name: str, draw_fn, draws: list[np.ndarray], grid, spec) -> dict:
    """Times CPU time (ru_utime), not wall-clock. On a shared/oversubscribed
    cluster node (confirmed here: 1 CPU allocated via Cpus_allowed_list, but
    load average 4-6 from other tenants), wall-clock is preempted
    unpredictably between checkpoints -- CPU time actually consumed by this
    process is immune to that noise."""
    start_cpu_s = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    t0 = time.perf_counter()
    for sample_points in draws:
        draw_fn(sample_points, grid, spec)
    wall_time_s = time.perf_counter() - t0
    end_cpu_s = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    cpu_time_s = end_cpu_s - start_cpu_s
    end_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    n_draws = len(draws)
    return {
        "implementation_checkpoint": name,
        "n_draws": n_draws,
        "n_jobs": 1,
        "sweep_steps": None,
        "wall_time_s": wall_time_s,
        "cpu_time_s": cpu_time_s,
        "cpu_time_per_draw_ms": (cpu_time_s / n_draws) * 1000.0 if n_draws else float("nan"),
        "time_per_draw_ms": (wall_time_s / n_draws) * 1000.0 if n_draws else float("nan"),
        "draws_per_second": n_draws / wall_time_s if wall_time_s > 0 else float("nan"),
        "max_rss_mb": end_rss_kb / 1024.0,
        "seed": SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON results")
    args = parser.parse_args()

    points = _fixed_distribution()
    grid = derive_shared_grid(points, points, grid_size=61)
    spec = _analysis_spec()
    draws = _draws(points, args.n_draws, seed=SEED, sample_fraction=BOOTSTRAP_SAMPLE_FRACTION)

    results = []
    for name, draw_fn in CHECKPOINTS.items():
        result = run_checkpoint(name, draw_fn, draws, grid, spec)
        results.append(result)
        print(
            f"{name:>12s}: {result['cpu_time_s']:8.3f}s CPU "
            f"({result['cpu_time_per_draw_ms']:7.3f} ms/draw CPU) | "
            f"{result['wall_time_s']:8.3f}s wall "
            f"({result['time_per_draw_ms']:7.3f} ms/draw wall, "
            f"{result['draws_per_second']:8.1f} draws/s), "
            f"{result['max_rss_mb']:7.1f} MB RSS"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
