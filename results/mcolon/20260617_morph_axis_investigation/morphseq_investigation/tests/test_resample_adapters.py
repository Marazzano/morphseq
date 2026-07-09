"""Tests for `core._resample_adapters.run_permutation_draws` failure surfacing.

The underlying `analyze.utils.resampling` engine catches per-draw exceptions
internally and returns a `_FAILED` sentinel for that draw (excluded from
`out.samples`). `run_permutation_draws` must not silently return a truncated
array when draws fail -- it must pad the failures back in as NaN so the
array's length always equals the requested `n_draws`, letting the existing
`run_empirical_null_test` `np.isfinite` valid-null accounting (unchanged)
correctly reflect the true failure rate instead of an artificially-100%-valid
truncated population.
"""
from __future__ import annotations

import numpy as np
import pytest

from morphseq_investigation.core._resample_adapters import bootstrap_peak_vote, run_permutation_draws
from morphseq_investigation.core.resolved_peak_metrics import run_empirical_null_test


def test_run_permutation_draws_pads_failed_draws_with_nan():
    reference_points = np.arange(20.0).reshape(10, 2)
    target_points = np.arange(20.0, 40.0).reshape(10, 2)

    call_count = {"n": 0}
    # The engine calls resolve_and_summarize twice (X1, X2) for its internal
    # statistic dry-run and twice more for the "observed" statistic (both on
    # the ORIGINAL, unpermuted data) before any real per-draw permutation
    # happens -- those 4 calls must succeed or `resample.run` raises straight
    # out of preflight instead of going through the per-draw try/except. Only
    # start failing after that warm-up so we're actually exercising per-draw
    # failure handling, not preflight.
    WARMUP_CALLS = 4

    def _flaky_resolve_and_summarize(points: np.ndarray) -> dict[str, float]:
        # Fail every 3rd call (deterministic) to simulate a degenerate permuted
        # split raising inside `resolve_and_summarize` (e.g. too few points for
        # KDE), independent of RNG draw content.
        call_count["n"] += 1
        if call_count["n"] > WARMUP_CALLS and call_count["n"] % 3 == 0:
            raise ValueError("simulated degenerate split")
        return {"metric_a": float(np.mean(points))}

    n_draws = 30
    result = run_permutation_draws(
        reference_points=reference_points,
        target_points=target_points,
        metrics=("metric_a",),
        resolve_and_summarize=_flaky_resolve_and_summarize,
        n_draws=n_draws,
        seed=12345,
    )

    values = result["metric_a"]
    # Length must always equal the requested n_draws -- failed draws are padded
    # in, not dropped, so the array length is never silently truncated.
    assert values.shape == (n_draws,)

    n_finite = int(np.isfinite(values).sum())
    n_nan = int(np.isnan(values).sum())
    assert n_finite + n_nan == n_draws
    # Some draws should have failed (the flaky statistic raises ~1/3 of calls,
    # each draw makes 2 calls -- reference + target resolve).
    assert n_nan > 0
    assert n_finite > 0


def test_failed_draws_reduce_valid_null_fraction_downstream():
    """The padded-NaN array must flow through run_empirical_null_test's
    existing n_null/n_valid_null/valid_null_fraction/test_is_valid accounting
    and correctly report a degraded (or invalidated) null test when enough
    draws failed -- not silently report 100% validity against a truncated
    population.
    """
    reference_points = np.arange(20.0).reshape(10, 2)
    target_points = np.arange(20.0, 40.0).reshape(10, 2)

    call_count = {"n": 0}
    # Same warm-up guard as above: the first 4 calls are the engine's
    # preflight dry-run + observed-statistic calls on unpermuted data and
    # must succeed, or `resample.run` raises out of preflight instead of
    # reaching the per-draw failure path this test targets.
    WARMUP_CALLS = 4

    def _always_fail_80pct(points: np.ndarray) -> dict[str, float]:
        call_count["n"] += 1
        # Fail 4 out of every 5 real per-draw calls (80% failure rate),
        # deterministic and independent of permutation content.
        if call_count["n"] > WARMUP_CALLS and call_count["n"] % 5 != 0:
            raise ValueError("simulated degenerate split")
        return {"metric_a": float(np.mean(points))}

    n_draws = 50
    result = run_permutation_draws(
        reference_points=reference_points,
        target_points=target_points,
        metrics=("metric_a",),
        resolve_and_summarize=_always_fail_80pct,
        n_draws=n_draws,
        seed=999,
    )
    values = result["metric_a"]
    assert values.shape == (n_draws,)

    null_result = run_empirical_null_test(
        observed_value=5.0, null_values=values, alternative="greater", min_valid_null_fraction=0.8,
    )
    # n_null must reflect the full requested draw count (honest denominator),
    # not just however many happened to succeed.
    assert null_result.n_null == n_draws
    assert null_result.n_valid_null <= n_draws
    assert null_result.valid_null_fraction == pytest.approx(null_result.n_valid_null / n_draws)
    # With a high failure rate, the honest valid_null_fraction should be well
    # under the 0.8 threshold, correctly invalidating the test rather than
    # reporting test_is_valid=True against a truncated 100%-valid array.
    if null_result.valid_null_fraction < 0.8:
        assert not null_result.test_is_valid


def test_bootstrap_peak_vote_returns_counts_and_centers_per_successful_draw():
    points = np.arange(40.0).reshape(20, 2)

    def _resolve_draw(sample_points):
        # A deterministic, cheap stand-in for resolve_points_with_analysis_spec:
        # "count" = 2 always, "centers" = the sample mean twice (arbitrary but
        # exercises the (count, centers) contract bootstrap_peak_vote expects).
        mean = tuple(np.mean(sample_points, axis=0))
        return 2, (mean, mean)

    counts, centers_by_draw, n_failed = bootstrap_peak_vote(
        points=points,
        resolve_draw=_resolve_draw,
        n_draws=15,
        sample_fraction=0.8,
        min_sample_size=5,
        seed=42,
    )
    assert n_failed == 0
    assert len(counts) == 15
    assert all(count == 2 for count in counts)
    assert len(centers_by_draw) == 15
    assert all(len(centers) == 2 for centers in centers_by_draw)


def test_bootstrap_peak_vote_min_sample_size_floor_and_frac_cap():
    """Mirrors valley_visualization.py's pre-refactor sample_n formula:
    min(n, max(min_sample_size, ceil(sample_fraction * n)))."""
    points = np.arange(20.0).reshape(10, 2)
    seen_sizes = []

    def _resolve_draw(sample_points):
        seen_sizes.append(len(sample_points))
        return 1, ((0.0, 0.0),)

    bootstrap_peak_vote(
        points=points,
        resolve_draw=_resolve_draw,
        n_draws=5,
        sample_fraction=0.1,  # ceil(0.1*10)=1, but min_sample_size=6 floors it
        min_sample_size=6,
        seed=1,
    )
    # The engine's unperturbed "observed" call (see resample._engine.run) also
    # invokes the statistic once on the FULL (unsubsampled) data before any
    # per-draw perturbation, and preflight's dry run makes one more perturbed
    # call before the real loop -- so `seen_sizes` has a couple of extra
    # entries beyond the 5 real draws. What matters here is that every
    # PERTURBED call (i.e. every call that isn't on the full unsubsampled
    # data) sees exactly the floored sample_n=6, never the raw
    # ceil(0.1*10)=1.
    per_draw_sizes = [size for size in seen_sizes if size != len(points)]
    assert len(per_draw_sizes) >= 5
    assert all(size == 6 for size in per_draw_sizes)


def test_bootstrap_peak_vote_empty_points_returns_empty():
    counts, centers_by_draw, n_failed = bootstrap_peak_vote(
        points=np.empty((0, 2)),
        resolve_draw=lambda pts: (0, ()),
        n_draws=10,
        sample_fraction=0.8,
        min_sample_size=5,
        seed=1,
    )
    assert counts == []
    assert centers_by_draw == []
    assert n_failed == 0


def test_run_permutation_draws_no_failures_all_finite():
    reference_points = np.arange(20.0).reshape(10, 2)
    target_points = np.arange(20.0, 40.0).reshape(10, 2)

    def _never_fail(points: np.ndarray) -> dict[str, float]:
        return {"metric_a": float(np.mean(points))}

    n_draws = 15
    result = run_permutation_draws(
        reference_points=reference_points,
        target_points=target_points,
        metrics=("metric_a",),
        resolve_and_summarize=_never_fail,
        n_draws=n_draws,
        seed=7,
    )
    values = result["metric_a"]
    assert values.shape == (n_draws,)
    assert np.isfinite(values).all()
