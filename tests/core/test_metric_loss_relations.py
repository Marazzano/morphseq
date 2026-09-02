from __future__ import annotations

import pytest
import torch

from src.core.losses.metric_loss import (
    MetricLossConfigurationError,
    ZeroPositiveAnchorError,
    build_metric_target_masks,
    supcon_lout_from_logits,
)
from src.core.metric import (
    MetricGroupDefinition,
    MetricRelation,
    MetricRelationPolicy,
    PairRelation,
)


def _symmetric_policy() -> MetricRelationPolicy:
    return MetricRelationPolicy(
        name="test_only_loss_relations",
        version="fixture-1",
        scope="test_only",
        symmetry="symmetric",
        diagonal_behavior="positive",
        groups=tuple(MetricGroupDefinition(name=name) for name in ("alpha", "beta", "gamma")),
        pair_relations=(
            PairRelation("alpha", "alpha", MetricRelation.POSITIVE, "diagonal"),
            PairRelation("alpha", "beta", MetricRelation.NEGATIVE, "explicit_negative"),
            PairRelation("alpha", "gamma", MetricRelation.EXCLUDED, "explicit_exclusion"),
            PairRelation("beta", "beta", MetricRelation.POSITIVE, "diagonal"),
            PairRelation("beta", "gamma", MetricRelation.NEGATIVE, "explicit_negative"),
            PairRelation("gamma", "gamma", MetricRelation.POSITIVE, "diagonal"),
        ),
        rule_precedence=("diagonal", "explicit_negative", "explicit_exclusion"),
        exclusion_rule_names=("explicit_exclusion",),
    )


def _asymmetric_policy() -> MetricRelationPolicy:
    return MetricRelationPolicy(
        name="test_only_directed_loss_relations",
        version="fixture-2",
        scope="test_only",
        symmetry="asymmetric",
        diagonal_behavior="positive",
        groups=(MetricGroupDefinition("alpha"), MetricGroupDefinition("beta")),
        pair_relations=(
            PairRelation("alpha", "alpha", MetricRelation.POSITIVE, "diagonal"),
            PairRelation("alpha", "beta", MetricRelation.POSITIVE, "directed_positive"),
            PairRelation("beta", "alpha", MetricRelation.EXCLUDED, "directed_exclusion"),
            PairRelation("beta", "beta", MetricRelation.POSITIVE, "diagonal"),
        ),
        rule_precedence=("diagonal", "directed_positive", "directed_exclusion"),
        exclusion_rule_names=("directed_exclusion",),
    )


def test_relation_targets_pin_positive_negative_excluded_and_age_window() -> None:
    targets = build_metric_target_masks(
        num_features=5,
        explicit_positive_pairs=(),
        relation_policy=_symmetric_policy(),
        metric_groups=("alpha", "alpha", "alpha", "beta", "gamma"),
        ages=torch.tensor([10.0, 11.0, 12.1, 10.0, -1000.0]),
        loss_age_window=1.0,
        sampler_age_window=0.25,
    )

    # Same-class relation inside the closed loss window is positive.
    assert targets.positive[0, 1]
    assert targets.denominator[0, 1]
    # The same positive class relation outside the loss window becomes negative.
    assert not targets.positive[0, 2]
    assert targets.denominator[0, 2]
    # Explicit class negatives enter only the denominator.
    assert not targets.positive[0, 3]
    assert targets.denominator[0, 3]
    # Exclusion wins regardless of an arbitrarily large age difference.
    assert not targets.positive[0, 4]
    assert not targets.denominator[0, 4]
    # Self comparisons never enter either mask.
    assert not targets.positive.diagonal().any()
    assert not targets.denominator.diagonal().any()
    assert "loss_age_window=1" in targets.context
    assert "sampler_age_window=0.25" in targets.context


def test_explicit_paired_views_remain_positive() -> None:
    targets = build_metric_target_masks(
        num_features=2,
        explicit_positive_pairs=((0, 1),),
        relation_policy=_symmetric_policy(),
        metric_groups=("alpha", "gamma"),
        ages=(0.0, 100.0),
        loss_age_window=0.0,
        sampler_age_window=0.0,
    )

    assert torch.equal(
        targets.positive,
        torch.tensor([[False, True], [True, False]]),
    )
    assert torch.equal(targets.positive, targets.denominator)


def test_symmetric_relation_policy_propagates_both_directions() -> None:
    targets = build_metric_target_masks(
        num_features=2,
        explicit_positive_pairs=(),
        relation_policy=_symmetric_policy(),
        metric_groups=("alpha", "beta"),
        ages=(3.0, 3.0),
        loss_age_window=1.0,
    )

    assert not targets.positive.any()
    assert targets.denominator[0, 1]
    assert targets.denominator[1, 0]


def test_asymmetric_relation_policy_propagates_direction_without_symmetrizing() -> None:
    targets = build_metric_target_masks(
        num_features=2,
        explicit_positive_pairs=(),
        relation_policy=_asymmetric_policy(),
        metric_groups=("alpha", "beta"),
        ages=(3.0, 3.0),
        loss_age_window=1.0,
    )

    assert targets.positive[0, 1]
    assert targets.denominator[0, 1]
    assert not targets.positive[1, 0]
    assert not targets.denominator[1, 0]


def test_excluded_logits_enter_neither_numerator_nor_denominator() -> None:
    targets = build_metric_target_masks(
        num_features=4,
        explicit_positive_pairs=((0, 1), (2, 3)),
        relation_policy=_symmetric_policy(),
        metric_groups=("alpha", "alpha", "gamma", "gamma"),
        ages=(0.0, 0.0, 0.0, 0.0),
        loss_age_window=1.0,
    )
    logits = torch.tensor(
        [[0.0, 1.0, -2.0, -3.0], [1.0, 0.0, -4.0, -5.0],
         [-2.0, -4.0, 0.0, 2.0], [-3.0, -5.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    changed = logits.clone()
    changed[:2, 2:] = 1e100
    changed[2:, :2] = -1e100

    assert supcon_lout_from_logits(logits, targets) == pytest.approx(0.0)
    assert supcon_lout_from_logits(changed, targets) == pytest.approx(0.0)


def test_zero_positive_anchors_name_indices_policy_and_both_windows() -> None:
    targets = build_metric_target_masks(
        num_features=2,
        explicit_positive_pairs=(),
        relation_policy=_symmetric_policy(),
        metric_groups=("alpha", "beta"),
        ages=(0.0, 0.0),
        loss_age_window=2.0,
        sampler_age_window=0.5,
    )

    with pytest.raises(ZeroPositiveAnchorError) as exc_info:
        supcon_lout_from_logits(torch.zeros((2, 2)), targets)
    message = str(exc_info.value)
    assert "anchor_indices=[0, 1]" in message
    assert "test_only_loss_relations@fixture-1" in message
    assert "loss_age_window=2" in message
    assert "sampler_age_window=0.5" in message


@pytest.mark.parametrize("bad_age", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ages_fail_with_feature_index(bad_age: float) -> None:
    with pytest.raises(MetricLossConfigurationError, match=r"nonfinite feature indices=\[1\]"):
        build_metric_target_masks(
            num_features=2,
            explicit_positive_pairs=((0, 1),),
            relation_policy=_symmetric_policy(),
            metric_groups=("alpha", "alpha"),
            ages=(0.0, bad_age),
            loss_age_window=1.0,
        )
