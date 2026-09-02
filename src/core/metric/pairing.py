"""Split-local, indexed metric-pair selection over resolved observations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.core.metric.relations import (
    MetricRelation,
    MetricRelationPolicy,
    UnknownMetricGroupError,
)


class MetricPairingError(ValueError):
    """Base error for invalid pair policies, indexes, or requests."""


class MetricPairingConfigurationError(MetricPairingError):
    """The declared pair-selection mechanism is incomplete or inconsistent."""


@dataclass(frozen=True)
class SameEmbryoCandidatePolicy:
    """Control explicit paired-view positives from the same physical embryo."""

    enabled: bool = True
    allow_same_observation: bool = True
    relation_semantics: Literal["explicit_pair_positive"] = "explicit_pair_positive"


@dataclass(frozen=True)
class DifferentEmbryoCandidatePolicy:
    """Control positives from other embryos through C1's relation authority."""

    enabled: bool = True
    relation_semantics: Literal["positive_only"] = "positive_only"


@dataclass(frozen=True)
class MetricPairingPolicy:
    """Named mechanism policy for one split-local metric-pair index."""

    name: str
    version: str
    stage_column: str
    stage_source: str
    sampler_age_window: float
    same_embryo: SameEmbryoCandidatePolicy = SameEmbryoCandidatePolicy()
    different_embryo: DifferentEmbryoCandidatePolicy = DifferentEmbryoCandidatePolicy()
    same_embryo_probability: float = 0.5
    observation_weighting: Literal["uniform_observation"] = "uniform_observation"
    base_seed: int = 0
    scientific_policy: bool = False

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "stage_column", "stage_source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise MetricPairingConfigurationError(
                    f"metric pairing {field_name} must be a non-empty string"
                )
        _require_finite_nonnegative("sampler_age_window", self.sampler_age_window)
        probability = _require_probability(
            "same_embryo_probability", self.same_embryo_probability
        )
        if not isinstance(self.same_embryo, SameEmbryoCandidatePolicy):
            raise MetricPairingConfigurationError(
                "same_embryo must be a SameEmbryoCandidatePolicy"
            )
        if not isinstance(self.different_embryo, DifferentEmbryoCandidatePolicy):
            raise MetricPairingConfigurationError(
                "different_embryo must be a DifferentEmbryoCandidatePolicy"
            )
        if not self.same_embryo.enabled and not self.different_embryo.enabled:
            raise MetricPairingConfigurationError(
                "at least one of same_embryo or different_embryo candidates must be enabled"
            )
        if not self.same_embryo.enabled and probability != 0.0:
            raise MetricPairingConfigurationError(
                "same_embryo_probability must be 0 when same_embryo candidates are disabled"
            )
        if not self.different_embryo.enabled and probability != 1.0:
            raise MetricPairingConfigurationError(
                "same_embryo_probability must be 1 when different_embryo candidates are disabled"
            )
        if self.observation_weighting != "uniform_observation":
            raise MetricPairingConfigurationError(
                "C2 supports only explicit uniform_observation weighting; "
                f"got {self.observation_weighting!r}"
            )
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise MetricPairingConfigurationError(
                "base_seed must be a non-negative integer"
            )
        if not isinstance(self.scientific_policy, bool):
            raise MetricPairingConfigurationError("scientific_policy must be a boolean")


@dataclass(frozen=True)
class PairCandidateCounts:
    same_embryo: int
    different_embryo: int

    @property
    def total(self) -> int:
        return self.same_embryo + self.different_embryo


@dataclass(frozen=True)
class PairSelection:
    anchor_index: int
    other_index: int
    candidate_kind: Literal["same_embryo", "different_embryo"]
    seed: int


@dataclass(frozen=True)
class PairQueryDiagnostics:
    """Evidence that a query uses bounded indexes rather than a cohort scan."""

    group_range_lookups: int
    embryo_range_lookups: int
    same_embryo_positions_examined: int
    full_length_boolean_allocations: int = 0
    index_strategy: str = "sorted_age_ranges_by_group_and_physical_embryo"


@dataclass(frozen=True)
class _AgeSortedPool:
    dataset_indices: np.ndarray
    ages: np.ndarray

    @classmethod
    def build(
        cls,
        dataset_indices: Sequence[int],
        *,
        ages: np.ndarray,
        snip_ids: Sequence[str],
    ) -> "_AgeSortedPool":
        ordered = sorted(
            (int(index) for index in dataset_indices),
            key=lambda index: (float(ages[index]), snip_ids[index]),
        )
        index_array = np.asarray(ordered, dtype=np.int64)
        return cls(
            dataset_indices=index_array,
            ages=np.asarray([ages[index] for index in ordered], dtype=np.float64),
        )

    def bounds(self, age: float, window: float) -> tuple[int, int]:
        lower = int(np.searchsorted(self.ages, age - window, side="left"))
        upper = int(np.searchsorted(self.ages, age + window, side="right"))
        return lower, upper


class MetricPairIndex:
    """One immutable index for one resolved split and one C1 relation policy."""

    REQUIRED_COLUMNS = frozenset(
        {"snip_id", "physical_embryo_id", "split", "metric_group"}
    )

    def __init__(
        self,
        resolved_split_table: pd.DataFrame,
        *,
        relation_policy: MetricRelationPolicy,
        pairing_policy: MetricPairingPolicy,
        split: str | None = None,
    ) -> None:
        if not isinstance(resolved_split_table, pd.DataFrame):
            raise TypeError(
                "MetricPairIndex requires a resolved split pandas DataFrame; "
                f"got {type(resolved_split_table)!r}"
            )
        if not isinstance(relation_policy, MetricRelationPolicy):
            raise MetricPairingConfigurationError(
                "relation_policy must be a C1 MetricRelationPolicy"
            )
        if not isinstance(pairing_policy, MetricPairingPolicy):
            raise MetricPairingConfigurationError(
                "pairing_policy must be a MetricPairingPolicy"
            )
        required = self.REQUIRED_COLUMNS | {pairing_policy.stage_column}
        missing = sorted(required - set(resolved_split_table.columns))
        if missing:
            raise MetricPairingConfigurationError(
                f"pair policy {pairing_policy.name!r} requires missing resolved columns: {missing!r}"
            )
        if resolved_split_table.empty:
            raise MetricPairingConfigurationError(
                f"pair policy {pairing_policy.name!r} received an empty split table"
            )

        table = resolved_split_table.reset_index(drop=True)
        self._require_opaque_strings(table, "snip_id")
        self._require_opaque_strings(table, "physical_embryo_id")
        self._require_opaque_strings(table, "split")
        self._require_opaque_strings(table, "metric_group")
        duplicated = table["snip_id"].duplicated(keep=False)
        if duplicated.any():
            duplicate_ids = sorted(set(table.loc[duplicated, "snip_id"]))
            raise MetricPairingConfigurationError(
                "uniform_observation pairing requires one selected asset row per snip_id; "
                "sibling product/z assets must be resolved as one observation before pairing. "
                f"duplicate snip_id values={duplicate_ids!r}"
            )

        split_names = tuple(dict.fromkeys(table["split"]))
        if len(split_names) != 1:
            raise MetricPairingConfigurationError(
                "MetricPairIndex is split-local and cannot index multiple splits; "
                f"observed splits={split_names!r}"
            )
        if split is not None and split_names[0] != split:
            raise MetricPairingConfigurationError(
                f"requested split={split!r} but indexed rows carry split={split_names[0]!r}"
            )

        raw_ages = table[pairing_policy.stage_column]
        if raw_ages.map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise MetricPairingConfigurationError(
                f"pair policy {pairing_policy.name!r} stage column "
                f"{pairing_policy.stage_column!r} contains boolean values"
            )
        ages = pd.to_numeric(raw_ages, errors="coerce").to_numpy(dtype=np.float64)
        invalid_age_indices = np.flatnonzero(~np.isfinite(ages)).tolist()
        if invalid_age_indices:
            bad_ids = table.iloc[invalid_age_indices]["snip_id"].tolist()
            raise MetricPairingConfigurationError(
                f"pair policy {pairing_policy.name!r} requires finite stage values from "
                f"{pairing_policy.stage_source!r} column={pairing_policy.stage_column!r}; "
                f"offending snip_id values={bad_ids!r}"
            )

        groups = tuple(table["metric_group"])
        unknown_groups = sorted(set(groups) - set(relation_policy.group_names))
        if unknown_groups:
            raise UnknownMetricGroupError(
                f"pair policy {pairing_policy.name!r} contains groups absent from relation "
                f"policy {relation_policy.name!r}@{relation_policy.version}: {unknown_groups!r}"
            )

        self.policy = pairing_policy
        self.relation_policy = relation_policy
        self.split = split_names[0]
        self.snip_ids = tuple(table["snip_id"])
        self.physical_embryo_ids = tuple(table["physical_embryo_id"])
        self.metric_groups = groups
        self.ages = ages
        all_indices = tuple(range(len(table)))

        group_indices: dict[str, list[int]] = {}
        embryo_indices: dict[str, list[int]] = {}
        for index in all_indices:
            group_indices.setdefault(groups[index], []).append(index)
            embryo_indices.setdefault(self.physical_embryo_ids[index], []).append(index)
        self._group_pools = {
            group: _AgeSortedPool.build(
                indices,
                ages=ages,
                snip_ids=self.snip_ids,
            )
            for group, indices in sorted(group_indices.items())
        }
        self._embryo_pools = {
            embryo_id: _AgeSortedPool.build(
                indices,
                ages=ages,
                snip_ids=self.snip_ids,
            )
            for embryo_id, indices in sorted(embryo_indices.items())
        }
        self._embryo_pool_positions: dict[int, int] = {}
        for pool in self._embryo_pools.values():
            self._embryo_pool_positions.update(
                {
                    int(dataset_index): position
                    for position, dataset_index in enumerate(pool.dataset_indices)
                }
            )

        self._group_embryo_positions: dict[tuple[str, str], np.ndarray] = {}
        for group, pool in self._group_pools.items():
            by_embryo: dict[str, list[int]] = {}
            for position, dataset_index in enumerate(pool.dataset_indices):
                embryo_id = self.physical_embryo_ids[int(dataset_index)]
                by_embryo.setdefault(embryo_id, []).append(position)
            for embryo_id, positions in by_embryo.items():
                self._group_embryo_positions[(group, embryo_id)] = np.asarray(
                    positions, dtype=np.int64
                )

        self._positive_groups = {
            anchor_group: tuple(
                candidate_group
                for candidate_group in sorted(self._group_pools)
                if relation_policy.relation(anchor_group, candidate_group)
                is MetricRelation.POSITIVE
            )
            for anchor_group in sorted(set(groups))
        }

    def __len__(self) -> int:
        return len(self.snip_ids)

    def candidate_counts(self, anchor_index: int) -> PairCandidateCounts:
        self._validate_anchor_index(anchor_index)
        same_count = self._same_embryo_count(anchor_index)
        different_count = sum(
            count for _, _, _, _, count in self._different_group_ranges(anchor_index)
        )
        return PairCandidateCounts(same_count, different_count)

    def query_diagnostics(self, anchor_index: int) -> PairQueryDiagnostics:
        self._validate_anchor_index(anchor_index)
        different_ranges = self._different_group_ranges(anchor_index)
        examined = sum(len(excluded) for _, _, _, excluded, _ in different_ranges)
        return PairQueryDiagnostics(
            group_range_lookups=len(self._positive_groups[self.metric_groups[anchor_index]]),
            embryo_range_lookups=(
                1 if self.policy.same_embryo.enabled else 0
            ) + len(different_ranges),
            same_embryo_positions_examined=examined,
        )

    def candidate_indices(
        self,
        anchor_index: int,
        candidate_kind: Literal["same_embryo", "different_embryo"] | None = None,
    ) -> tuple[int, ...]:
        """Materialize legal candidates for preflight diagnostics/error recovery only."""

        self._validate_anchor_index(anchor_index)
        candidates: list[int] = []
        if candidate_kind in (None, "same_embryo") and self.policy.same_embryo.enabled:
            pool, lower, upper, excluded = self._same_embryo_range(anchor_index)
            excluded_set = set(excluded)
            candidates.extend(
                int(pool.dataset_indices[position])
                for position in range(lower, upper)
                if position not in excluded_set
            )
        if candidate_kind in (None, "different_embryo") and self.policy.different_embryo.enabled:
            for pool, lower, upper, excluded, _ in self._different_group_ranges(anchor_index):
                excluded_set = set(excluded)
                candidates.extend(
                    int(pool.dataset_indices[position])
                    for position in range(lower, upper)
                    if position not in excluded_set
                )
        return tuple(
            sorted(set(candidates), key=lambda index: self.snip_ids[index])
        )

    def select_same_embryo(self, anchor_index: int, choice: int) -> int:
        pool, lower, upper, excluded = self._same_embryo_range(anchor_index)
        count = upper - lower - len(excluded)
        position = _select_kth_available(lower, upper, excluded, choice, count=count)
        return int(pool.dataset_indices[position])

    def select_different_embryo(self, anchor_index: int, choice: int) -> int:
        ranges = self._different_group_ranges(anchor_index)
        total = sum(count for _, _, _, _, count in ranges)
        if not 0 <= choice < total:
            raise MetricPairingError(
                f"different-embryo choice={choice} is outside candidate count={total}"
            )
        remainder = choice
        for pool, lower, upper, excluded, count in ranges:
            if remainder < count:
                position = _select_kth_available(
                    lower, upper, excluded, remainder, count=count
                )
                return int(pool.dataset_indices[position])
            remainder -= count
        raise AssertionError("different-embryo candidate selection exhausted unexpectedly")

    def _same_embryo_count(self, anchor_index: int) -> int:
        if not self.policy.same_embryo.enabled:
            return 0
        _, lower, upper, excluded = self._same_embryo_range(anchor_index)
        return upper - lower - len(excluded)

    def _same_embryo_range(
        self, anchor_index: int
    ) -> tuple[_AgeSortedPool, int, int, tuple[int, ...]]:
        embryo_id = self.physical_embryo_ids[anchor_index]
        pool = self._embryo_pools[embryo_id]
        lower, upper = pool.bounds(
            self.ages[anchor_index], self.policy.sampler_age_window
        )
        excluded: tuple[int, ...] = ()
        if not self.policy.same_embryo.allow_same_observation:
            anchor_position = self._embryo_pool_positions[anchor_index]
            if lower <= anchor_position < upper:
                excluded = (anchor_position,)
        return pool, lower, upper, excluded

    def _different_group_ranges(
        self, anchor_index: int
    ) -> tuple[tuple[_AgeSortedPool, int, int, tuple[int, ...], int], ...]:
        if not self.policy.different_embryo.enabled:
            return ()
        anchor_group = self.metric_groups[anchor_index]
        embryo_id = self.physical_embryo_ids[anchor_index]
        age = self.ages[anchor_index]
        ranges = []
        for candidate_group in self._positive_groups[anchor_group]:
            pool = self._group_pools.get(candidate_group)
            if pool is None:
                continue
            lower, upper = pool.bounds(age, self.policy.sampler_age_window)
            same_positions = self._group_embryo_positions.get(
                (candidate_group, embryo_id), np.empty(0, dtype=np.int64)
            )
            left = int(np.searchsorted(same_positions, lower, side="left"))
            right = int(np.searchsorted(same_positions, upper, side="left"))
            excluded = tuple(int(value) for value in same_positions[left:right])
            count = upper - lower - len(excluded)
            if count:
                ranges.append((pool, lower, upper, excluded, count))
        return tuple(ranges)

    def _validate_anchor_index(self, anchor_index: int) -> None:
        if (
            isinstance(anchor_index, bool)
            or not isinstance(anchor_index, (int, np.integer))
            or not 0 <= int(anchor_index) < len(self)
        ):
            raise IndexError(anchor_index)

    @staticmethod
    def _require_opaque_strings(table: pd.DataFrame, column: str) -> None:
        invalid = [
            index
            for index, value in enumerate(table[column])
            if not isinstance(value, str) or not value
        ]
        if invalid:
            raise MetricPairingConfigurationError(
                f"resolved {column} values must be non-empty opaque strings; "
                f"offending row indices={invalid!r}"
            )


class MetricPairSampler:
    """Deterministically sample one indexed legal positive for each anchor."""

    def __init__(self, pair_index: MetricPairIndex, *, rank: int = 0) -> None:
        if not isinstance(pair_index, MetricPairIndex):
            raise TypeError("pair_index must be a MetricPairIndex")
        _require_nonnegative_integer("rank", rank)
        self.pair_index = pair_index
        self.rank = int(rank)
        self.epoch = 0

    @property
    def policy(self) -> MetricPairingPolicy:
        return self.pair_index.policy

    @property
    def snip_ids(self) -> tuple[str, ...]:
        return self.pair_index.snip_ids

    def set_epoch(self, epoch: int) -> None:
        _require_nonnegative_integer("epoch", epoch)
        self.epoch = int(epoch)

    def sample(
        self,
        anchor_index: int,
        *,
        worker_id: int = 0,
        rank: int | None = None,
        draw_index: int = 0,
    ) -> PairSelection:
        _require_nonnegative_integer("worker_id", worker_id)
        _require_nonnegative_integer("draw_index", draw_index)
        effective_rank = self.rank if rank is None else rank
        _require_nonnegative_integer("rank", effective_rank)
        counts = self.pair_index.candidate_counts(anchor_index)
        if not counts.total:
            raise MetricPairingError(
                f"anchor snip_id={self.pair_index.snip_ids[anchor_index]!r} has no legal "
                f"positive under pair policy {self.policy.name!r}@{self.policy.version}"
            )
        seed = derive_pair_seed(
            base_seed=self.policy.base_seed,
            policy_name=self.policy.name,
            policy_version=self.policy.version,
            split=self.pair_index.split,
            anchor_snip_id=self.pair_index.snip_ids[anchor_index],
            epoch=self.epoch,
            rank=int(effective_rank),
            worker_id=int(worker_id),
            draw_index=int(draw_index),
        )
        rng = np.random.default_rng(seed)
        if counts.same_embryo and counts.different_embryo:
            use_same = rng.random() < self.policy.same_embryo_probability
        else:
            use_same = bool(counts.same_embryo)
        if use_same:
            other_index = self.pair_index.select_same_embryo(
                anchor_index, int(rng.integers(counts.same_embryo))
            )
            candidate_kind = "same_embryo"
        else:
            other_index = self.pair_index.select_different_embryo(
                anchor_index, int(rng.integers(counts.different_embryo))
            )
            candidate_kind = "different_embryo"
        return PairSelection(
            anchor_index=int(anchor_index),
            other_index=other_index,
            candidate_kind=candidate_kind,
            seed=seed,
        )


def derive_pair_seed(
    *,
    base_seed: int,
    policy_name: str,
    policy_version: str,
    split: str,
    anchor_snip_id: str,
    epoch: int,
    rank: int,
    worker_id: int,
    draw_index: int,
) -> int:
    """Derive a stable 64-bit seed without Python's process-randomized hash."""

    payload = json.dumps(
        [
            str(base_seed),
            policy_name,
            policy_version,
            split,
            anchor_snip_id,
            str(epoch),
            str(rank),
            str(worker_id),
            str(draw_index),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _select_kth_available(
    lower: int,
    upper: int,
    excluded: Sequence[int],
    choice: int,
    *,
    count: int,
) -> int:
    if not 0 <= choice < count:
        raise MetricPairingError(
            f"candidate choice={choice} is outside candidate count={count}"
        )
    position = lower + choice
    for excluded_position in excluded:
        if excluded_position <= position:
            position += 1
        else:
            break
    if not lower <= position < upper:
        raise AssertionError("indexed candidate position escaped its age range")
    return position


def _require_finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise MetricPairingConfigurationError(f"{name} must be a real scalar")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricPairingConfigurationError(
            f"{name} must be a finite non-negative scalar, got {value!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise MetricPairingConfigurationError(
            f"{name} must be finite and >= 0, got {value!r}"
        )
    return numeric


def _require_probability(name: str, value: float) -> float:
    numeric = _require_finite_nonnegative(name, value)
    if numeric > 1:
        raise MetricPairingConfigurationError(f"{name} must be in [0, 1], got {value!r}")
    return numeric


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise MetricPairingConfigurationError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
