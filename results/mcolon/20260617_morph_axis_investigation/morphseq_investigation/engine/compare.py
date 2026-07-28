"""Purpose-specific comparisons over the unified label ontology.

Peak correspondence is deliberately absent: peak matching has its own deferred
policy design.  This module currently owns partition agreement and a structured
summary of label groups across a catalog ``DistributionComparison``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Hashable, Mapping, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .catalog import DistributionComparison
from .density import DEFAULT_DENSITY_SPEC, _bandwidth
from .grid import build_shared_grid, evaluate_density
from .objects import (
    DensityEstimate,
    DensityEstimateSpec,
    Distribution,
    Grid,
    LabelGroup,
    UNASSIGNED_LABEL,
)


def _readonly_mapping(values):
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class LabelGroupSummaryRow:
    """One comparison member's label summary with structured coordinates."""

    comparison_coordinates: Mapping[str, Hashable]
    member_value: Hashable
    distribution_id: str
    distribution_coordinates: Mapping[str, Hashable]
    label_group_name: str
    sample_set_count: int
    assigned_sample_count: int
    unassigned_sample_count: int
    peak_count: int | None
    is_robust: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison_coordinates", _readonly_mapping(self.comparison_coordinates))
        object.__setattr__(self, "distribution_coordinates", _readonly_mapping(self.distribution_coordinates))


@dataclass(frozen=True)
class PartitionAgreementResult:
    distribution_id: str
    left_label_group_name: str
    right_label_group_name: str
    cross_tab: Mapping[Hashable, Mapping[Hashable, int]] = field(default_factory=dict)
    agreement_metric: str = "adjusted_rand_score"
    agreement_score: float = 0.0
    n_samples_compared: int = 0
    left_only_sample_ids: tuple[str, ...] = ()
    right_only_sample_ids: tuple[str, ...] = ()
    unassigned_both_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cross_tab",
            _readonly_mapping({key: _readonly_mapping(value) for key, value in self.cross_tab.items()}),
        )
        object.__setattr__(self, "left_only_sample_ids", tuple(self.left_only_sample_ids))
        object.__setattr__(self, "right_only_sample_ids", tuple(self.right_only_sample_ids))
        object.__setattr__(self, "unassigned_both_sample_ids", tuple(self.unassigned_both_sample_ids))


def summarize_label_groups(
    comparison: DistributionComparison,
    label_group_name: str,
) -> tuple[LabelGroupSummaryRow, ...]:
    """Summarize one label across comparison members without flattening coordinates.

    Sample-set counts are taken from the authoritative derived view
    ``Distribution.sample_sets``. Missing labels raise rather than silently
    dropping a comparison member.
    """
    rows = []
    for member_value, distribution in comparison.members.items():
        group = distribution.get_label_group(label_group_name)
        sample_sets = distribution.sample_sets(label_group_name)
        assigned = sum(value != UNASSIGNED_LABEL for value in group.assignments.values())
        rows.append(
            LabelGroupSummaryRow(
                comparison_coordinates=comparison.coordinates,
                member_value=member_value,
                distribution_id=distribution.distribution_id,
                distribution_coordinates=distribution.coordinates,
                label_group_name=group.name,
                sample_set_count=len(sample_sets),
                assigned_sample_count=assigned,
                unassigned_sample_count=len(group.assignments) - assigned,
                peak_count=group.peak_count,
                is_robust=group.is_robust,
            )
        )
    return tuple(rows)


def label_group_agreement(left_lg: LabelGroup, right_lg: LabelGroup) -> PartitionAgreementResult:
    """Chance-corrected agreement between two partitions of one distribution."""
    if left_lg.distribution_id != right_lg.distribution_id:
        raise ValueError("label_group_agreement requires the same distribution_id")
    if set(left_lg.assignments) != set(right_lg.assignments):
        raise ValueError("label_group_agreement requires identical sample membership")

    sample_ids = tuple(left_lg.assignments)
    left_assigned = {sid for sid in sample_ids if left_lg.assignments[sid] != UNASSIGNED_LABEL}
    right_assigned = {sid for sid in sample_ids if right_lg.assignments[sid] != UNASSIGNED_LABEL}
    common = tuple(sid for sid in sample_ids if sid in left_assigned and sid in right_assigned)
    left_only = tuple(sid for sid in sample_ids if sid in left_assigned - right_assigned)
    right_only = tuple(sid for sid in sample_ids if sid in right_assigned - left_assigned)
    neither = tuple(sid for sid in sample_ids if sid not in left_assigned | right_assigned)

    cross_tab = {
        left: {right: 0 for right in right_lg.categories()}
        for left in left_lg.categories()
    }
    for sid in common:
        cross_tab[left_lg.assignments[sid]][right_lg.assignments[sid]] += 1

    score = (
        float(adjusted_rand_score(
            [left_lg.assignments[sid] for sid in common],
            [right_lg.assignments[sid] for sid in common],
        ))
        if common else float("nan")
    )
    return PartitionAgreementResult(
        distribution_id=left_lg.distribution_id,
        left_label_group_name=left_lg.name,
        right_label_group_name=right_lg.name,
        cross_tab=cross_tab,
        agreement_score=score,
        n_samples_compared=len(common),
        left_only_sample_ids=left_only,
        right_only_sample_ids=right_only,
        unassigned_both_sample_ids=neither,
    )


def _feature_values(distribution: Distribution, features: tuple[str, ...]) -> np.ndarray:
    missing = tuple(name for name in features if name not in distribution.feature_names)
    if missing:
        raise ValueError(
            f"distribution {distribution.distribution_id!r} lacks comparison "
            f"feature(s) {missing}; has {distribution.feature_names}"
        )
    return np.column_stack([distribution.feature_column(name) for name in features])


def _density_spec(distribution: Distribution, label_group: str | None) -> DensityEstimateSpec:
    if label_group is not None:
        group = distribution.get_label_group(label_group)
        if group.density is not None:
            return group.density.spec
    if distribution.shared_density is not None:
        return distribution.shared_density.spec
    return DEFAULT_DENSITY_SPEC


def _estimate_on_grid(
    distribution: Distribution,
    features: tuple[str, ...],
    values: np.ndarray,
    grid: Grid,
    spec: DensityEstimateSpec,
) -> DensityEstimate:
    field = evaluate_density(grid, values, _bandwidth(values, spec))
    return DensityEstimate(
        distribution_id=distribution.distribution_id,
        feature_names=features,
        spec=spec,
        grid=grid,
        density_grid=field,
    )


def _overlap(left: DensityEstimate, right: DensityEstimate) -> float:
    if left.grid.grid_id != right.grid.grid_id:
        raise ValueError("density overlap requires exact shared-grid identity")
    cell_volume = float(np.prod([
        axis[1] - axis[0] if len(axis) > 1 else 1.0
        for axis in left.grid.axis_values
    ]))
    return float(np.minimum(
        left.density_grid.density, right.density_grid.density
    ).sum() * cell_volume)


@dataclass(frozen=True)
class NullTestResult:
    """Immutable evidence for an optional permutation test of density overlap."""

    state: str = "not_tested"
    observed_overlap: float | None = None
    null_overlaps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    valid_draws: int = 0
    p_value: float | None = None
    alpha: float = 0.05
    n_draws: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        values = np.array(self.null_overlaps, dtype=float, copy=True)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("null_overlaps must be a finite one-dimensional array")
        values.setflags(write=False)
        object.__setattr__(self, "null_overlaps", values)
        if self.state not in {"not_tested", "invalid", "nonsignificant", "significant"}:
            raise ValueError(f"unknown null-test state {self.state!r}")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.n_draws < 0:
            raise ValueError("n_draws must be non-negative")
        if self.valid_draws < 0 or self.valid_draws > self.n_draws:
            raise ValueError("valid_draws must be in [0, n_draws]")
        if self.valid_draws != len(values):
            raise ValueError("valid_draws must equal len(null_overlaps)")

        observed_is_finite = (
            self.observed_overlap is not None
            and np.isfinite(float(self.observed_overlap))
        )
        p_is_finite = self.p_value is not None and np.isfinite(float(self.p_value))
        if self.state == "not_tested":
            if self.n_draws != 0 or self.valid_draws != 0 or self.p_value is not None:
                raise ValueError("not_tested evidence cannot contain null draws or a p-value")
        elif self.state == "invalid":
            if self.n_draws < 1 or self.p_value is not None or not observed_is_finite:
                raise ValueError("invalid evidence requires draws and an observed value, but no p-value")
        else:
            if self.n_draws < 1 or self.valid_draws < 1 or not observed_is_finite:
                raise ValueError("tested evidence requires observed and valid null values")
            if not p_is_finite or not 0.0 <= float(self.p_value) <= 1.0:
                raise ValueError("tested evidence requires a finite p-value in [0, 1]")
            significant = float(self.p_value) < float(self.alpha)
            if significant != (self.state == "significant"):
                raise ValueError("state must agree with p_value < alpha")


@dataclass(frozen=True)
class DescriptiveComparison:
    """One directed, aligned target-versus-reference comparison."""

    reference: Distribution
    target: Distribution
    features: tuple[str, ...]
    grid: Grid
    reference_density: DensityEstimate
    target_density: DensityEstimate
    density_overlap: float
    label_group_name: str | None = None
    reference_label_group: LabelGroup | None = None
    target_label_group: LabelGroup | None = None
    reference_value: Hashable | None = None
    target_value: Hashable | None = None
    coordinates: Mapping[str, Hashable] = field(default_factory=dict)
    null_test: NullTestResult = field(default_factory=NullTestResult)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "coordinates", _readonly_mapping(self.coordinates))

    def test_nulls(
        self, *, n_draws: int = 200, seed: int | None = None, alpha: float = 0.05
    ) -> "DescriptiveComparison":
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        rng = np.random.default_rng(seed)
        ref_values = _feature_values(self.reference, self.features)
        target_values = _feature_values(self.target, self.features)
        pooled = np.concatenate((ref_values, target_values), axis=0)
        n_ref = len(ref_values)
        null_values: list[float] = []
        for _ in range(n_draws):
            try:
                order = rng.permutation(len(pooled))
                left = pooled[order[:n_ref]]
                right = pooled[order[n_ref:]]
                left_est = _estimate_on_grid(
                    self.reference, self.features, left, self.grid,
                    self.reference_density.spec,
                )
                right_est = _estimate_on_grid(
                    self.target, self.features, right, self.grid,
                    self.target_density.spec,
                )
                value = _overlap(left_est, right_est)
                if np.isfinite(value):
                    null_values.append(value)
            except (ValueError, FloatingPointError):
                # A failed permutation is an invalid draw, not a reason to
                # discard valid evidence from independent draws.
                continue
        valid = np.asarray([value for value in null_values if np.isfinite(value)])
        if not len(valid):
            result = NullTestResult(
                state="invalid", observed_overlap=self.density_overlap,
                n_draws=n_draws, seed=seed, alpha=alpha,
            )
        else:
            # Smaller overlap is the directed evidence of distributional change.
            p_value = float((1 + np.count_nonzero(valid <= self.density_overlap)) / (len(valid) + 1))
            result = NullTestResult(
                state="significant" if p_value < alpha else "nonsignificant",
                observed_overlap=self.density_overlap, null_overlaps=valid,
                valid_draws=len(valid), p_value=p_value, alpha=alpha,
                n_draws=n_draws, seed=seed,
            )
        return replace(self, null_test=result)


@dataclass(frozen=True)
class DescriptiveComparisons:
    comparisons: tuple[DescriptiveComparison, ...]
    across: str | None = None
    reference_value: Hashable | None = None
    target_values: tuple[Hashable, ...] = ()
    match_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparisons", tuple(self.comparisons))
        object.__setattr__(self, "target_values", tuple(self.target_values))
        object.__setattr__(self, "match_on", tuple(self.match_on))

    def test_nulls(
        self, *, n_draws: int = 200, seed: int | None = None, alpha: float = 0.05
    ) -> "DescriptiveComparisons":
        seeds = np.random.SeedSequence(seed).spawn(len(self.comparisons))
        tested = tuple(
            comparison.test_nulls(
                n_draws=n_draws,
                seed=int(child.generate_state(1)[0]),
                alpha=alpha,
            )
            for comparison, child in zip(self.comparisons, seeds)
        )
        return replace(self, comparisons=tested)


def compare_distributions(
    *,
    reference: Distribution,
    targets: Sequence[Distribution],
    label_group: str | None = None,
    features: Sequence[str] | None = None,
    grid: Grid | None = None,
    grid_size: int = 40,
) -> DescriptiveComparisons:
    """Prepare immutable 1-D/2-D comparisons directly from raw objects.

    Every target is independently aligned with the reference. Densities use
    complete distribution membership; a requested label group is retained only
    as plot-ready overlay metadata.
    """
    targets = tuple(targets)
    if not targets:
        raise ValueError("compare_distributions requires at least one target")
    if grid is not None:
        resolved_features = tuple(grid.feature_names)
        if features is not None and tuple(features) != resolved_features:
            raise ValueError("supplied features must exactly match grid.feature_names")
    elif features is not None:
        resolved_features = tuple(features)
    elif len(reference.feature_names) == 1:
        resolved_features = reference.feature_names
    else:
        raise ValueError("multi-feature comparison requires explicit ordered features")
    if len(resolved_features) not in (1, 2):
        raise ValueError("comparison density preparation currently supports only 1-D or 2-D")

    prepared: list[DescriptiveComparison] = []
    for target in targets:
        ref_values = _feature_values(reference, resolved_features)
        target_values = _feature_values(target, resolved_features)
        ref_group = reference.get_label_group(label_group) if label_group is not None else None
        target_group = target.get_label_group(label_group) if label_group is not None else None
        pair_grid = grid or build_shared_grid(
            resolved_features,
            ref_values,
            target_values,
            reference.sample_ids,
            target.sample_ids,
            resolution=grid_size,
        )
        ref_spec = _density_spec(reference, label_group)
        target_spec = _density_spec(target, label_group)
        ref_density = _estimate_on_grid(reference, resolved_features, ref_values, pair_grid, ref_spec)
        target_density = _estimate_on_grid(target, resolved_features, target_values, pair_grid, target_spec)
        prepared.append(DescriptiveComparison(
            reference=reference,
            target=target,
            features=resolved_features,
            grid=pair_grid,
            reference_density=ref_density,
            target_density=target_density,
            density_overlap=_overlap(ref_density, target_density),
            label_group_name=label_group,
            reference_label_group=ref_group,
            target_label_group=target_group,
        ))
    return DescriptiveComparisons(tuple(prepared))


__all__ = [
    "DescriptiveComparison",
    "DescriptiveComparisons",
    "LabelGroupSummaryRow",
    "NullTestResult",
    "PartitionAgreementResult",
    "compare_distributions",
    "label_group_agreement",
    "summarize_label_groups",
]
