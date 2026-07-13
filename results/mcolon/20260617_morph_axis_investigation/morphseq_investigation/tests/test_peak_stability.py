"""Unit tests for `core.peak_stability`: vote contracts and resolution summary,
PeakSeedSet construction (COMPOSE_single_path_plan Sec 1.5 / 1.5d).
"""

from __future__ import annotations

import pytest

from morphseq_investigation.core.peak_stability import (
    PeakCountRobustnessPolicy,
    PeakCountVote,
    PeakVotingSpec,
    build_consensus_seed_set,
    compute_peak_count_stability,
)


def test_peak_count_vote_rejects_frequency_sum_mismatch():
    with pytest.raises(ValueError):
        PeakCountVote(
            peak_count_frequencies={1: 4, 2: 76},
            n_draws_requested=80,
            n_draws_valid=79,  # mismatched: frequencies sum to 80
            sample_fraction=0.8,
        )


def test_peak_count_vote_frequency_and_probability_helpers():
    vote = PeakCountVote(
        peak_count_frequencies={1: 4, 2: 76},
        n_draws_requested=80,
        n_draws_valid=80,
        sample_fraction=0.8,
    )
    assert vote.frequency_for(2) == 76
    assert vote.frequency_for(3) == 0
    assert vote.probability_for(2) == pytest.approx(0.95)
    assert vote.probability_for(3) == 0.0


def test_compute_peak_count_stability_stable_vote():
    vote = PeakCountVote(
        peak_count_frequencies={1: 4, 2: 76},
        n_draws_requested=80,
        n_draws_valid=80,
        sample_fraction=0.8,
    )
    stability = compute_peak_count_stability(vote, PeakCountRobustnessPolicy(min_mode_frequency=0.80))

    assert stability.mode_peak_count == 2
    assert stability.mode_frequency == pytest.approx(0.95)
    assert stability.count_is_stable is True
    assert stability.valid_draw_fraction == pytest.approx(1.0)
    assert stability.mean_peak_count == pytest.approx((1 * 4 + 2 * 76) / 80)


def test_compute_peak_count_stability_unstable_vote_below_threshold():
    vote = PeakCountVote(
        peak_count_frequencies={1: 45, 2: 55},
        n_draws_requested=100,
        n_draws_valid=100,
        sample_fraction=0.8,
    )
    stability = compute_peak_count_stability(vote, PeakCountRobustnessPolicy(min_mode_frequency=0.80))

    assert stability.mode_peak_count == 2
    assert stability.mode_frequency == pytest.approx(0.55)
    assert stability.count_is_stable is False


def test_compute_peak_count_stability_excludes_invalid_draws_from_variance():
    """An invalid draw is NOT a vote for '0 modes' -- it is excluded, so
    valid_draw_fraction reflects the loss but mean/variance stay computed
    only over the valid votes (plan Sec 1.5)."""
    vote = PeakCountVote(
        peak_count_frequencies={2: 40},
        n_draws_requested=50,  # 10 draws failed/invalid
        n_draws_valid=40,
        sample_fraction=0.8,
    )
    stability = compute_peak_count_stability(vote, PeakCountRobustnessPolicy())

    assert stability.mean_peak_count == pytest.approx(2.0)
    assert stability.peak_count_variance == pytest.approx(0.0)
    assert stability.valid_draw_fraction == pytest.approx(0.8)


def test_compute_peak_count_stability_no_valid_draws_raises():
    vote = PeakCountVote(
        peak_count_frequencies={},
        n_draws_requested=10,
        n_draws_valid=0,
        sample_fraction=0.8,
    )
    with pytest.raises(ValueError, match="at least one valid draw"):
        compute_peak_count_stability(vote, PeakCountRobustnessPolicy())


def test_tied_mode_selects_larger_warns_and_is_nonrobust():
    vote = PeakCountVote({1: 5, 2: 5}, 10, 10, 0.8)
    with pytest.warns(RuntimeWarning, match="tied mode"):
        summary = compute_peak_count_stability(
            vote,
            PeakCountRobustnessPolicy(min_mode_frequency=0.5),
            PeakVotingSpec(n_draws=10, sample_fraction=0.8, min_valid_draws=5),
        )
    assert summary.resolved_peak_count == 2
    assert summary.mode_was_tied is True
    assert summary.is_robust is False


def test_insufficient_valid_draws_retains_mode_but_is_nonrobust():
    vote = PeakCountVote({3: 4}, 10, 4, 0.8)
    summary = compute_peak_count_stability(
        vote,
        PeakCountRobustnessPolicy(min_mode_frequency=0.5),
        PeakVotingSpec(n_draws=10, sample_fraction=0.8, min_valid_draws=5),
    )
    assert summary.resolved_peak_count == 3
    assert summary.has_enough_valid_draws is False
    assert summary.is_robust is False


def test_resolution_summary_retains_complete_contracts():
    voting_spec = PeakVotingSpec(n_draws=10, sample_fraction=0.7, min_valid_draws=6)
    policy = PeakCountRobustnessPolicy(min_mode_frequency=0.75)
    vote = PeakCountVote({2: 8}, 10, 8, 0.7)
    summary = compute_peak_count_stability(vote, policy, voting_spec)
    assert summary.peak_count_vote is vote
    assert summary.voting_spec is voting_spec
    assert summary.robustness_policy is policy


def test_build_consensus_seed_set_two_peaks_two_qualifying_draws():
    draw_centers = [
        [(-2.0, 0.0), (2.0, 0.0)],
        [(-2.1, 0.1), (2.1, -0.1)],
        [(0.0, 0.0)],  # a draw that voted for 1 peak -- not qualifying, ignored
    ]
    seed_set = build_consensus_seed_set(
        target_peak_count=2, draw_centers=draw_centers, n_draws_at_target=2
    )

    assert seed_set.target_peak_count == 2
    assert len(seed_set.seeds) == 2
    assert seed_set.construction_method == "bootstrap_consensus_centroid"
    centers = sorted(seed_set.centers, key=lambda c: c[0])
    assert centers[0][0] == pytest.approx(-2.05, abs=0.05)
    assert centers[1][0] == pytest.approx(2.05, abs=0.05)


def test_build_consensus_seed_set_raises_when_no_draw_matches_target():
    with pytest.raises(ValueError):
        build_consensus_seed_set(
            target_peak_count=3,
            draw_centers=[[(-2.0, 0.0), (2.0, 0.0)]],
            n_draws_at_target=0,
        )


def test_build_consensus_seed_set_zero_target_returns_empty():
    seed_set = build_consensus_seed_set(target_peak_count=0, draw_centers=[], n_draws_at_target=0)
    assert seed_set.target_peak_count == 0
    assert seed_set.seeds == ()
