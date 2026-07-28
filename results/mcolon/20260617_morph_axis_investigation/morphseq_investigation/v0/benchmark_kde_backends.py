"""Compares this codebase's hand-rolled isotropic Gaussian KDE evaluator
against sklearn.neighbors.KernelDensity, to answer: is the dense
BLAS-matmul-based custom evaluator (bandwidth_tuning.py's
precompute_squared_distances + evaluate_isotropic_gaussian_kde_from_dist2)
still the right choice, or has a general-purpose library backend caught up?

Verifies numerical equivalence first (they must produce the same density
surface, not just similar timings), then benchmarks CPU time (not wall-clock
-- see the caveat below).

Result as of this writing (2-D, ~160 samples, 61x61 grid): numerically
identical (ratio 1.0 +/- ~1e-12), hand-rolled ~3x faster than sklearn across
all its algorithm variants (auto/kd_tree/ball_tree). Dense wins at this scale
because a Gaussian kernel's infinite support means tree methods still visit
most points anyway, paying pure traversal overhead for no pruning benefit.
Documented on evaluate_isotropic_gaussian_kde_from_dist2's docstring -- rerun
this script rather than trusting that note indefinitely if the workload
changes (higher dimensions, much larger sample counts, sparse query
locations, or approximate-evaluation tolerances could all flip this result).

CPU-time caveat: on a shared/oversubscribed cluster node (single allocated
CPU, other tenants' load visible via `uptime`), wall-clock is preempted
unpredictably -- this script measures resource.getrusage(...).ru_utime
instead, which is immune to that noise.

Usage:
  python v0/benchmark_kde_backends.py [--n-points 160] [--grid-size 61] [--n-calls 100]
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from morphseq_investigation.core.bandwidth_tuning import (
    bandwidth_geometry_scales,
    evaluate_isotropic_gaussian_kde_from_dist2,
    precompute_squared_distances,
)

SEED = 20260617


def _fixed_setup(n_points: int, grid_size: int):
    rng = np.random.default_rng(SEED)
    points = rng.normal(size=(n_points, 2))
    xs = np.linspace(-3, 3, grid_size)
    ys = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    scales = bandwidth_geometry_scales(points, include_connectivity_radius=False)
    h = scales["median_kNN_distance"]
    return points, grid_points, h


def _cpu_time() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_utime


def _bench(fn, n_calls: int) -> float:
    fn()  # warmup (JIT/allocator warmup, first-call overhead)
    t0 = _cpu_time()
    for _ in range(n_calls):
        fn()
    return (_cpu_time() - t0) / n_calls * 1000.0


def check_equivalence(points: np.ndarray, grid_points: np.ndarray, h: float) -> None:
    from sklearn.neighbors import KernelDensity

    dist2 = precompute_squared_distances(grid_points, points)
    manual_density = evaluate_isotropic_gaussian_kde_from_dist2(
        dist2, h, cell_area=None, normalize_grid=False,
    )
    kde = KernelDensity(bandwidth=h, kernel="gaussian", metric="euclidean").fit(points)
    sklearn_density = np.exp(kde.score_samples(grid_points))

    ratio = sklearn_density / manual_density
    print(f"Numerical equivalence check: ratio mean={ratio.mean():.12f}, std={ratio.std():.2e}")
    np.testing.assert_allclose(sklearn_density, manual_density, rtol=1e-8, atol=1e-10)
    print("PASS: hand-rolled and sklearn produce numerically identical density surfaces.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-points", type=int, default=160)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--n-calls", type=int, default=100)
    args = parser.parse_args()

    from sklearn.neighbors import KernelDensity

    points, grid_points, h = _fixed_setup(args.n_points, args.grid_size)

    check_equivalence(points, grid_points, h)

    def hand_rolled():
        dist2 = precompute_squared_distances(grid_points, points)
        return evaluate_isotropic_gaussian_kde_from_dist2(dist2, h)

    def sklearn_algo(algorithm: str):
        def _fn():
            kde = KernelDensity(
                bandwidth=h, kernel="gaussian", metric="euclidean", algorithm=algorithm,
            ).fit(points)
            return np.exp(kde.score_samples(grid_points))
        return _fn

    candidates = [
        ("hand_rolled", hand_rolled),
        ("sklearn_auto", sklearn_algo("auto")),
        ("sklearn_kd_tree", sklearn_algo("kd_tree")),
        ("sklearn_ball_tree", sklearn_algo("ball_tree")),
    ]

    print(f"n_points={args.n_points}, grid_size={args.grid_size}x{args.grid_size}, n_calls={args.n_calls}\n")
    results = []
    for name, fn in candidates:
        ms = _bench(fn, args.n_calls)
        results.append((name, ms))
        print(f"{name:20s}: {ms:8.3f} ms/call CPU")

    baseline_ms = results[0][1]
    print()
    for name, ms in results[1:]:
        factor = ms / baseline_ms
        print(f"{name} is {factor:.2f}x the hand-rolled cost")


if __name__ == "__main__":
    main()
