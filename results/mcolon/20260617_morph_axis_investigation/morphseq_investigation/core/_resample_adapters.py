"""Thin typed wrapper over `analyze.utils.resampling` (COMPOSE_single_path_plan
Sec 1.11 / 5.1).

The raw engine hands the statistic a `data` dict and injects a magic
`"indices"` key (subsample) or requires `"X1"`/`"X2"` group names -- a
stringly-typed handshake. This module owns those conventions so
`resolved_peak_analysis.py` never talks to `analyze.utils.resampling`
directly, only to the typed helper here.

Stage 2a adds `bootstrap_peak_vote`, wrapping `resample.subsample` the same
way `run_permutation_draws` wraps `resample.permute_groups`: same typed
seam, same `_FAILED`/`n_failed` -> NaN-padding discipline for invalid draws
(here: dropped rather than NaN-padded, since a vote's "invalid draw" is
"does not vote," not a numeric quantity to average -- see `n_failed`
handling below for the reasoning).

Note: `resolved_peak_analysis.py` already computes the OBSERVED delta itself
(`compute_observed_delta`) and reduces observed-vs-null via the existing
`run_empirical_null_test` (unchanged, stable engine -- see
`resolved_peak_metrics.py`). This adapter's job is narrowly the DRAW
generation: produce `n_draws` null_delta values per metric under pooled-label
permutation, nothing more, so it composes cleanly with that existing
reduction step instead of duplicating it.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

import analyze.utils.resampling as resample


def run_permutation_draws(
    *,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    metrics: tuple[str, ...],
    resolve_and_summarize: Callable[[np.ndarray], Mapping[str, float]],
    n_draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run `n_draws` pooled-label permutation draws via
    `analyze.utils.resampling.permute_groups`, returning raw null_delta arrays
    per metric (`{metric_name: ndarray[n_draws]}`).

    `resolve_and_summarize(points) -> {metric_name: value, ...}` is the single
    per-group resolve+summarize call (SINGLE_PASS per COMPOSE_single_path_plan
    Sec 1.10); this adapter does not know or care what it does internally, it
    only orchestrates the two-group draw and the per-metric delta statistic.

    Per the resampling package's documented SeedSequence clean break, the
    exact draws differ from the retired hand-rolled loop for a "matching"
    seed -- expected, not a bug.
    """
    reference_points = np.asarray(reference_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    if n_draws <= 0:
        return {metric: np.asarray([], dtype=float) for metric in metrics}

    def _delta_statistic(data: dict, rng: np.random.Generator | None) -> dict[str, float]:
        reference_summary = resolve_and_summarize(np.asarray(data["X1"], dtype=float))
        target_summary = resolve_and_summarize(np.asarray(data["X2"], dtype=float))
        return {
            metric: float(target_summary[metric]) - float(reference_summary[metric])
            for metric in metrics
        }

    spec = resample.permute_groups(a="X1", b="X2")
    stat = resample.statistic(
        "resolved_peak_metric_deltas",
        _delta_statistic,
        outputs=list(metrics),
    )
    out = resample.run(
        data={"X1": reference_points, "X2": target_points},
        spec=spec,
        statistic=stat,
        n_iters=n_draws,
        seed=seed,
        store="all",
    )

    # The engine catches per-draw exceptions internally (e.g. a degenerate
    # permuted split too small for KDE) and returns a `_FAILED` sentinel for
    # that draw, which its own success-partitioning already excludes from
    # `out.samples`. If we returned only `out.samples`, a caller requesting
    # `n_draws=500` with 100 engine-level failures would silently get
    # 400-length arrays with no signal anything failed -- downstream validity
    # checks (`run_empirical_null_test`'s `n_null`/`valid_null_fraction`)
    # would then treat the truncated array as the full, honestly-achieved
    # null population. Per COMPOSE_single_path_plan Sec 1.11, the engine's
    # `_FAILED` sentinel IS the intended "invalid draw" signal, and the
    # existing "invalid draw = non-vote, excluded" pattern already used
    # elsewhere is: pad failed draws back in as NaN (restoring the array to
    # its full requested length) so `run_empirical_null_test`'s existing
    # `np.isfinite` valid-null filtering -- not a new parallel concept --
    # naturally discounts them from `n_valid_null`/`valid_null_fraction`.
    samples = out.samples or []
    n_failed = int(out.n_failed)
    return {
        metric: np.concatenate(
            [
                np.asarray([sample[metric] for sample in samples], dtype=float),
                np.full(n_failed, np.nan, dtype=float),
            ]
        )
        for metric in metrics
    }


def bootstrap_peak_vote(
    *,
    points: np.ndarray,
    resolve_draw: Callable[[np.ndarray], tuple[int, Sequence[tuple[float, float]]]],
    n_draws: int,
    sample_fraction: float,
    min_sample_size: int,
    seed: int,
) -> tuple[list[int], list[tuple[tuple[float, float], ...]], int]:
    """Run `n_draws` bootstrap-subsample draws via
    `analyze.utils.resampling.subsample`, each resolving a reduced-sample-size
    draw with `resolve_draw(sample_points) -> (count, centers)`.

    Returns `(counts, centers_by_draw, n_failed)`:
      - `counts`: one accepted-peak-count int per SUCCESSFUL draw
      - `centers_by_draw`: the parallel list of that draw's candidate centers
        (transient use only -- consumed to build a `PeakSeedSet`, never
        persisted on the final `ResolvedPeakDistribution`; Stage 2b scope)
      - `n_failed`: engine-level `_FAILED` count (a crashed/degenerate draw
        is NOT a vote for "0 modes" -- it is simply excluded, per plan Sec 1.5)

    The draw sample size mirrors the pre-refactor `_resampled_mode_count` in
    `valley_visualization.py`: `max(min_sample_size, ceil(sample_fraction * n))`,
    capped at `n`, so the MIN_EMBRYOS floor logic carries over unchanged.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n_draws <= 0 or n == 0:
        return [], [], 0

    sample_n = min(n, max(int(min_sample_size), int(np.ceil(float(sample_fraction) * n))))

    def _vote_statistic(data: dict, rng: np.random.Generator | None) -> dict:
        indices = data.get("indices")
        if indices is None:
            # The engine's unperturbed "observed" call (see `resample._engine.run`)
            # has no "indices" key -- use the full-data pass so it does not
            # crash preflight; its count/centers are not read by this adapter
            # (only `out.samples`, i.e. the per-draw perturbed results, are).
            sample_points = data["points"]
        else:
            sample_points = data["points"][np.asarray(indices)]
        count, centers = resolve_draw(sample_points)
        return {"count": int(count), "centers": tuple(tuple(map(float, c)) for c in centers)}

    # Pass an explicit `size=` (rather than `frac=`) so the draw count is
    # exactly `sample_n` (which already folds in the MIN_EMBRYOS floor) --
    # avoids a float round-trip through `frac = sample_n / n`.
    spec = resample.subsample(size=sample_n)
    stat = resample.statistic(
        "peak_count_vote",
        _vote_statistic,
        outputs=["count", "centers"],
    )
    out = resample.run(
        data={"n": n, "points": points},
        spec=spec,
        statistic=stat,
        n_iters=int(n_draws),
        seed=int(seed),
        store="all",
    )

    samples = out.samples or []
    counts = [int(sample["count"]) for sample in samples]
    centers_by_draw = [tuple(sample["centers"]) for sample in samples]
    return counts, centers_by_draw, int(out.n_failed)


__all__ = ["bootstrap_peak_vote", "run_permutation_draws"]
