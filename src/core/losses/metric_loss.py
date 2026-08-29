"""Pure relation-aware SupCon loss helpers.

The class relationship semantics in this module come exclusively from
``MetricRelationPolicy.relation``.  Boolean masks are derived per batch and are
runtime data, not a second policy representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch

from src.core.metric import MetricRelation, MetricRelationPolicy


class MetricLossError(ValueError):
    """Base error for invalid metric-loss inputs or targets."""


class MetricLossConfigurationError(MetricLossError):
    """Metric-loss configuration is invalid or incomplete."""


class ZeroPositiveAnchorError(MetricLossError):
    """At least one anchor has no positive in the current batch."""


@dataclass(frozen=True)
class MetricTargetMasks:
    """Positive and denominator membership derived for one feature batch."""

    positive: torch.Tensor
    denominator: torch.Tensor
    context: str


def validate_metric_parameters(
    *,
    temperature: float,
    metric_weight: float,
    sampler_age_window: float,
    loss_age_window: float | None,
) -> None:
    """Validate scalar configuration without coupling the two age windows."""

    _require_finite_scalar("contrastive_temperature", temperature, strictly_positive=True)
    _require_finite_scalar("metric_weight", metric_weight, nonnegative=True)
    _require_finite_scalar(
        "sampler_age_window", sampler_age_window, nonnegative=True
    )
    if loss_age_window is not None:
        _require_finite_scalar(
            "loss_age_window", loss_age_window, nonnegative=True
        )


def build_metric_target_masks(
    *,
    num_features: int,
    explicit_positive_pairs: Iterable[tuple[int, int]],
    relation_policy: MetricRelationPolicy | None = None,
    metric_groups: Sequence[str] | None = None,
    ages: Sequence[float] | torch.Tensor | None = None,
    loss_age_window: float | None = None,
    sampler_age_window: float | None = None,
    device: torch.device | str | None = None,
) -> MetricTargetMasks:
    """Derive positive and denominator masks from C1 relation semantics.

    When ``relation_policy`` is absent, this is an explicitly paired-view-only
    target: supplied pairs are positive and all other non-self comparisons are
    negative.  Relation-aware targets require group names, finite ages, and an
    explicit loss-age window.  A positive class relation outside that window is
    negative; an excluded relation remains absent from both masks.
    """

    if not isinstance(num_features, int) or isinstance(num_features, bool) or num_features < 2:
        raise MetricLossConfigurationError(
            f"num_features must be an integer >= 2, got {num_features!r}"
        )

    positive = torch.zeros((num_features, num_features), dtype=torch.bool)
    denominator = ~torch.eye(num_features, dtype=torch.bool)

    if relation_policy is None:
        if metric_groups is not None or ages is not None or loss_age_window is not None:
            raise MetricLossConfigurationError(
                "paired-view-only targets cannot accept metric_groups, ages, or "
                "loss_age_window without a MetricRelationPolicy"
            )
        context = "policy=paired_view_only, loss_age_window=not_applicable"
    else:
        if not isinstance(relation_policy, MetricRelationPolicy):
            raise MetricLossConfigurationError(
                "relation_policy must be a C1 MetricRelationPolicy instance"
            )
        if metric_groups is None or len(metric_groups) != num_features:
            actual = None if metric_groups is None else len(metric_groups)
            raise MetricLossConfigurationError(
                f"metric_groups length must equal num_features={num_features}, got {actual}"
            )
        if any(not isinstance(group, str) or not group for group in metric_groups):
            raise MetricLossConfigurationError(
                "metric_groups must contain non-empty canonical group-name strings"
            )
        if loss_age_window is None:
            raise MetricLossConfigurationError(
                f"relation policy {relation_policy.name!r}@{relation_policy.version} "
                "requires an explicit loss_age_window"
            )
        _require_finite_scalar("loss_age_window", loss_age_window, nonnegative=True)
        if ages is None:
            raise MetricLossConfigurationError(
                f"relation policy {relation_policy.name!r}@{relation_policy.version} "
                "requires one finite age per feature"
            )
        age_values = torch.as_tensor(ages, dtype=torch.float64).flatten()
        if age_values.numel() != num_features:
            raise MetricLossConfigurationError(
                f"ages length must equal num_features={num_features}, got {age_values.numel()}"
            )
        if not torch.isfinite(age_values).all():
            bad = torch.nonzero(~torch.isfinite(age_values), as_tuple=False).flatten().tolist()
            raise MetricLossConfigurationError(
                f"ages must be finite; nonfinite feature indices={bad!r}"
            )

        denominator.zero_()
        age_is_close = (
            torch.abs(age_values.unsqueeze(1) - age_values.unsqueeze(0))
            <= float(loss_age_window)
        )
        for anchor_index, anchor_group in enumerate(metric_groups):
            for comparison_index, comparison_group in enumerate(metric_groups):
                if anchor_index == comparison_index:
                    continue
                relation = relation_policy.relation(anchor_group, comparison_group)
                if relation is MetricRelation.EXCLUDED:
                    continue
                denominator[anchor_index, comparison_index] = True
                if (
                    relation is MetricRelation.POSITIVE
                    and age_is_close[anchor_index, comparison_index]
                ):
                    positive[anchor_index, comparison_index] = True

        sampler_context = (
            "deferred"
            if sampler_age_window is None
            else _format_finite_context_scalar("sampler_age_window", sampler_age_window)
        )
        context = (
            f"policy={relation_policy.name}@{relation_policy.version}, "
            f"loss_age_window={float(loss_age_window):g}, "
            f"sampler_age_window={sampler_context}"
        )

    for pair in explicit_positive_pairs:
        if len(pair) != 2:
            raise MetricLossConfigurationError(
                f"explicit positive pair must have two indices, got {pair!r}"
            )
        left, right = pair
        if (
            not isinstance(left, int)
            or isinstance(left, bool)
            or not isinstance(right, int)
            or isinstance(right, bool)
            or not 0 <= left < num_features
            or not 0 <= right < num_features
        ):
            raise MetricLossConfigurationError(
                f"explicit positive pair {pair!r} is outside feature indices "
                f"[0, {num_features})"
            )
        if left == right:
            raise MetricLossConfigurationError(
                f"explicit positive pair cannot be a self-comparison: {pair!r}"
            )
        positive[left, right] = True
        positive[right, left] = True
        denominator[left, right] = True
        denominator[right, left] = True

    positive.fill_diagonal_(False)
    denominator.fill_diagonal_(False)
    if torch.any(positive & ~denominator):
        raise AssertionError("positive metric targets must also be denominator members")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return MetricTargetMasks(
        positive=positive.to(target_device),
        denominator=denominator.to(target_device),
        context=context,
    )


def supcon_lout_from_logits(
    logits: torch.Tensor,
    targets: MetricTargetMasks,
) -> torch.Tensor:
    """Compute SupCon ``L_out`` with the positive average outside the log."""

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise MetricLossConfigurationError(
            f"logits must be a square rank-2 tensor, got shape={tuple(logits.shape)!r}"
        )
    if not logits.is_floating_point():
        raise MetricLossConfigurationError(
            f"logits must be floating point, got dtype={logits.dtype}"
        )
    expected_shape = tuple(logits.shape)
    if tuple(targets.positive.shape) != expected_shape or tuple(targets.denominator.shape) != expected_shape:
        raise MetricLossConfigurationError(
            "target masks must match logits shape; "
            f"logits={expected_shape!r}, positive={tuple(targets.positive.shape)!r}, "
            f"denominator={tuple(targets.denominator.shape)!r}"
        )
    positive = targets.positive.to(device=logits.device, dtype=torch.bool)
    denominator = targets.denominator.to(device=logits.device, dtype=torch.bool)
    if torch.any(positive & ~denominator):
        raise MetricLossConfigurationError(
            "every positive target must also be present in its denominator"
        )
    if torch.any(torch.diagonal(positive)) or torch.any(torch.diagonal(denominator)):
        raise MetricLossConfigurationError("self-comparisons must be excluded")
    if not torch.isfinite(logits[denominator]).all():
        raise MetricLossConfigurationError(
            "eligible contrastive logits must be finite"
        )

    positive_counts = positive.sum(dim=1)
    zero_positive = torch.nonzero(positive_counts == 0, as_tuple=False).flatten()
    if zero_positive.numel():
        raise ZeroPositiveAnchorError(
            "metric loss anchors have no legal batch positive; "
            f"anchor_indices={zero_positive.tolist()!r}, {targets.context}"
        )

    eligible_logits = logits.masked_fill(~denominator, -torch.inf)
    log_denominator = torch.logsumexp(eligible_logits, dim=1)
    if not torch.isfinite(log_denominator).all():
        bad = torch.nonzero(~torch.isfinite(log_denominator), as_tuple=False).flatten()
        raise MetricLossConfigurationError(
            "metric loss anchors have no finite denominator; "
            f"anchor_indices={bad.tolist()!r}, {targets.context}"
        )

    log_probabilities = logits - log_denominator.unsqueeze(1)
    positive_log_probability_sum = torch.where(
        positive,
        log_probabilities,
        torch.zeros((), dtype=logits.dtype, device=logits.device),
    ).sum(dim=1)
    per_anchor = -(positive_log_probability_sum / positive_counts.to(logits.dtype))
    loss = per_anchor.mean()
    if not torch.isfinite(loss):
        raise MetricLossConfigurationError(
            f"metric loss is nonfinite under {targets.context}"
        )
    return loss


def euclidean_supcon_lout(
    features: torch.Tensor,
    targets: MetricTargetMasks,
    *,
    temperature: float,
) -> torch.Tensor:
    """Compute stable SupCon ``L_out`` using legacy-normalized Euclidean logits."""

    _require_finite_scalar("contrastive_temperature", temperature, strictly_positive=True)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise MetricLossConfigurationError(
            f"features must have shape [N>=2, D>=1], got {tuple(features.shape)!r}"
        )
    if not features.is_floating_point():
        raise MetricLossConfigurationError(
            f"features must be floating point, got dtype={features.dtype}"
        )
    if not torch.isfinite(features).all():
        bad = torch.nonzero(~torch.isfinite(features), as_tuple=False).tolist()
        raise MetricLossConfigurationError(
            f"features must be finite; nonfinite coordinates={bad!r}"
        )

    # The legacy expression sqrt(cdist(features)^2 / (D / 2)) is exactly
    # cdist(features) / sqrt(D / 2).  Computing distances in float64 avoids the
    # unnecessary square and its float32 overflow while preserving that scale.
    compute_features = features.to(torch.float64)
    distances = torch.cdist(compute_features, compute_features, p=2)
    scale = math.sqrt(features.shape[1] / 2.0)
    logits = -distances / (scale * float(temperature))
    return supcon_lout_from_logits(logits, targets)


def _require_finite_scalar(
    name: str,
    value: float,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if isinstance(value, bool):
        raise MetricLossConfigurationError(f"{name} must be a real scalar, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricLossConfigurationError(
            f"{name} must be a finite real scalar, got {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        raise MetricLossConfigurationError(f"{name} must be finite, got {value!r}")
    if strictly_positive and numeric <= 0:
        raise MetricLossConfigurationError(f"{name} must be > 0, got {value!r}")
    if nonnegative and numeric < 0:
        raise MetricLossConfigurationError(f"{name} must be >= 0, got {value!r}")


def _format_finite_context_scalar(name: str, value: float) -> str:
    _require_finite_scalar(name, value, nonnegative=True)
    return f"{float(value):g}"
