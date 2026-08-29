"""One validated semantic authority for metric-group relationships."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal, Mapping


RELATION_POLICY_FORMAT_VERSION = "1.0"

TAG_UNCERTAIN = "uncertain"
TAG_CONTROL = "control"
TAG_WILD_TYPE = "wild_type"
TAG_CRISPANT = "crispant"
TAG_WT_AB = "wild_type_ab"
TAG_WT_WIK = "wild_type_wik"
TAG_WT_MUTANT_BACKGROUND = "wild_type_mutant_background"
TAG_WT_FLUORESCENT_MARKER = "wild_type_fluorescent_marker"

KNOWN_GROUP_TAGS = frozenset(
    {
        TAG_UNCERTAIN,
        TAG_CONTROL,
        TAG_WILD_TYPE,
        TAG_CRISPANT,
        TAG_WT_AB,
        TAG_WT_WIK,
        TAG_WT_MUTANT_BACKGROUND,
        TAG_WT_FLUORESCENT_MARKER,
    }
)

PolicyScope = Literal["test_only", "compatibility_candidate", "scientific"]
Symmetry = Literal["symmetric", "asymmetric"]
DiagonalBehavior = Literal[
    "explicit", "positive", "uncertain_excluded_otherwise_positive"
]


class MetricRelationError(ValueError):
    """Base error for relation policy failures."""


class MetricRelationValidationError(MetricRelationError):
    """The relation policy declaration is invalid."""


class MetricRelationCoverageError(MetricRelationError):
    """One or more declared metric-group pairs have no relationship."""


class UnknownMetricGroupError(MetricRelationError):
    """A lookup requested a group absent from the relation policy."""


class MetricRelation(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class MetricGroupDefinition:
    """Canonical metric-group identity plus explicit compatibility-rule tags."""

    name: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise MetricRelationValidationError("metric group name must be non-empty")
        tags = tuple(sorted(self.tags))
        object.__setattr__(self, "tags", tags)
        duplicates = sorted(tag for tag in set(tags) if tags.count(tag) > 1)
        if duplicates:
            raise MetricRelationValidationError(
                f"metric group {self.name!r} has duplicate tags: {duplicates!r}"
            )
        unknown = sorted(set(tags) - KNOWN_GROUP_TAGS)
        if unknown:
            raise MetricRelationValidationError(
                f"metric group {self.name!r} has unknown tags: {unknown!r}"
            )
        wt_subtypes = {
            TAG_WT_AB,
            TAG_WT_WIK,
            TAG_WT_MUTANT_BACKGROUND,
            TAG_WT_FLUORESCENT_MARKER,
        }
        if set(tags) & wt_subtypes and TAG_WILD_TYPE not in tags:
            raise MetricRelationValidationError(
                f"metric group {self.name!r} has a wild-type subtype tag without "
                f"{TAG_WILD_TYPE!r}"
            )

    def has(self, tag: str) -> bool:
        return tag in self.tags


@dataclass(frozen=True)
class PairRelation:
    """One explicit class-pair relation and its named policy rule."""

    left_group: str
    right_group: str
    relation: MetricRelation
    rule_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.left_group, str) or not self.left_group:
            raise MetricRelationValidationError("left_group must be non-empty")
        if not isinstance(self.right_group, str) or not self.right_group:
            raise MetricRelationValidationError("right_group must be non-empty")
        try:
            relation = MetricRelation(self.relation)
        except ValueError as exc:
            raise MetricRelationValidationError(
                f"invalid relation {self.relation!r}; expected positive, negative, or excluded"
            ) from exc
        object.__setattr__(self, "relation", relation)
        if not isinstance(self.rule_name, str) or not self.rule_name.strip():
            raise MetricRelationValidationError("pair relation rule_name must be non-empty")


@dataclass(frozen=True)
class MetricRelationPolicy:
    """Complete versioned relationship table and its validation declarations.

    ``relation`` is the sole public semantic lookup.  Truth tables and future
    runtime caches are derived from this same authority.
    """

    name: str
    version: str
    scope: PolicyScope
    symmetry: Symmetry
    diagonal_behavior: DiagonalBehavior
    groups: tuple[MetricGroupDefinition, ...]
    pair_relations: tuple[PairRelation, ...]
    rule_precedence: tuple[str, ...]
    exclusion_rule_names: tuple[str, ...]
    format_version: str = RELATION_POLICY_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != RELATION_POLICY_FORMAT_VERSION:
            raise MetricRelationValidationError(
                f"unsupported relation policy format_version={self.format_version!r}; "
                f"expected {RELATION_POLICY_FORMAT_VERSION!r}"
            )
        for field_name, value in (("name", self.name), ("version", self.version)):
            if not isinstance(value, str) or not value.strip():
                raise MetricRelationValidationError(
                    f"relation policy {field_name} must be a non-empty string"
                )
        if self.scope not in ("test_only", "compatibility_candidate", "scientific"):
            raise MetricRelationValidationError(f"invalid policy scope={self.scope!r}")
        if self.symmetry not in ("symmetric", "asymmetric"):
            raise MetricRelationValidationError(
                f"invalid symmetry declaration={self.symmetry!r}"
            )
        if self.diagonal_behavior not in (
            "explicit",
            "positive",
            "uncertain_excluded_otherwise_positive",
        ):
            raise MetricRelationValidationError(
                f"invalid diagonal_behavior={self.diagonal_behavior!r}"
            )

        groups = tuple(sorted(self.groups, key=lambda group: group.name))
        object.__setattr__(self, "groups", groups)
        if not groups:
            raise MetricRelationValidationError("relation policy has no metric groups")
        names = tuple(group.name for group in groups)
        if len(set(names)) != len(names):
            duplicates = sorted(name for name in set(names) if names.count(name) > 1)
            raise MetricRelationValidationError(
                f"relation policy has duplicate metric groups: {duplicates!r}"
            )

        precedence = tuple(self.rule_precedence)
        object.__setattr__(self, "rule_precedence", precedence)
        if not precedence or any(not isinstance(rule, str) or not rule for rule in precedence):
            raise MetricRelationValidationError(
                "relation policy must declare non-empty rule_precedence"
            )
        if len(set(precedence)) != len(precedence):
            raise MetricRelationValidationError("rule_precedence contains duplicates")
        exclusion_rules = tuple(sorted(self.exclusion_rule_names))
        object.__setattr__(self, "exclusion_rule_names", exclusion_rules)
        if len(set(exclusion_rules)) != len(exclusion_rules):
            raise MetricRelationValidationError("exclusion_rule_names contains duplicates")
        undeclared_exclusion_rules = set(exclusion_rules) - set(precedence)
        if undeclared_exclusion_rules:
            raise MetricRelationValidationError(
                "exclusion rules are absent from rule_precedence: "
                f"{sorted(undeclared_exclusion_rules)!r}"
            )

        entries = tuple(self.pair_relations)
        relation_index: dict[tuple[str, str], PairRelation] = {}
        unknown_groups: set[str] = set()
        for entry in entries:
            if not isinstance(entry, PairRelation):
                raise MetricRelationValidationError(
                    "pair_relations must contain PairRelation instances"
                )
            unknown_groups.update(
                group_name
                for group_name in (entry.left_group, entry.right_group)
                if group_name not in names
            )
            if entry.rule_name not in precedence:
                raise MetricRelationValidationError(
                    f"pair {entry.left_group!r} x {entry.right_group!r} uses undeclared "
                    f"rule_name={entry.rule_name!r}"
                )
            if entry.relation is MetricRelation.EXCLUDED:
                if entry.rule_name not in exclusion_rules:
                    raise MetricRelationValidationError(
                        f"excluded pair {entry.left_group!r} x {entry.right_group!r} "
                        f"uses rule {entry.rule_name!r} absent from exclusion_rule_names"
                    )
            elif entry.rule_name in exclusion_rules:
                raise MetricRelationValidationError(
                    f"non-excluded pair {entry.left_group!r} x {entry.right_group!r} "
                    f"uses declared exclusion rule {entry.rule_name!r}"
                )
            key = self._normalize_pair(entry.left_group, entry.right_group)
            if key in relation_index:
                raise MetricRelationValidationError(
                    f"duplicate relation declaration for pair={key!r} under "
                    f"symmetry={self.symmetry!r}"
                )
            relation_index[key] = PairRelation(
                left_group=key[0],
                right_group=key[1],
                relation=entry.relation,
                rule_name=entry.rule_name,
            )
        if unknown_groups:
            raise MetricRelationValidationError(
                f"pair relations reference unknown metric groups: {sorted(unknown_groups)!r}"
            )

        expected = set(self._expected_pair_keys(names))
        missing = sorted(expected - set(relation_index))
        extra = sorted(set(relation_index) - expected)
        if missing or extra:
            raise MetricRelationCoverageError(
                f"relation policy {self.name!r}@{self.version} does not completely cover "
                f"declared groups; uncovered_pairs={missing!r}, extra_pairs={extra!r}"
            )

        group_by_name = {group.name: group for group in groups}
        for group_name in names:
            entry = relation_index[(group_name, group_name)]
            group = group_by_name[group_name]
            if self.diagonal_behavior == "positive":
                expected_relation = MetricRelation.POSITIVE
            elif self.diagonal_behavior == "uncertain_excluded_otherwise_positive":
                expected_relation = (
                    MetricRelation.EXCLUDED
                    if group.has(TAG_UNCERTAIN)
                    else MetricRelation.POSITIVE
                )
            else:
                continue
            if entry.relation is not expected_relation:
                raise MetricRelationValidationError(
                    f"diagonal relation for group={group_name!r} is "
                    f"{entry.relation.value!r}, expected {expected_relation.value!r} "
                    f"from diagonal_behavior={self.diagonal_behavior!r}"
                )

        object.__setattr__(
            self,
            "pair_relations",
            tuple(relation_index[key] for key in sorted(relation_index)),
        )
        object.__setattr__(self, "_relation_index", relation_index)

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.groups)

    def relation(self, left_group: str, right_group: str) -> MetricRelation:
        """Return the sole authoritative semantic relation for two known groups."""

        unknown = sorted(
            group for group in {left_group, right_group} if group not in self.group_names
        )
        if unknown:
            raise UnknownMetricGroupError(
                f"unknown metric groups={unknown!r} for relation policy "
                f"{self.name!r}@{self.version}"
            )
        key = self._normalize_pair(left_group, right_group)
        try:
            return self._relation_index[key].relation
        except KeyError as exc:
            raise MetricRelationCoverageError(
                f"relation policy {self.name!r}@{self.version} has no relation for "
                f"pair={key!r}"
            ) from exc

    def truth_table(self) -> str:
        """Return a deterministic, complete, human-readable class×class TSV."""

        lines = ["left_group\tright_group\trelation\trule_name"]
        for left_group in self.group_names:
            for right_group in self.group_names:
                key = self._normalize_pair(left_group, right_group)
                entry = self._relation_index[key]
                lines.append(
                    f"{left_group}\t{right_group}\t"
                    f"{self.relation(left_group, right_group).value}\t{entry.rule_name}"
                )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Serialize declarations and the complete canonical pair table."""

        entries = sorted(
            self.pair_relations,
            key=lambda entry: self._normalize_pair(entry.left_group, entry.right_group),
        )
        return {
            "format_version": self.format_version,
            "name": self.name,
            "version": self.version,
            "scope": self.scope,
            "symmetry": self.symmetry,
            "diagonal_behavior": self.diagonal_behavior,
            "groups": [
                {"name": group.name, "tags": list(group.tags)} for group in self.groups
            ],
            "rule_precedence": list(self.rule_precedence),
            "exclusion_rule_names": list(self.exclusion_rule_names),
            "pair_relations": [
                {
                    "left_group": entry.left_group,
                    "right_group": entry.right_group,
                    "relation": entry.relation.value,
                    "rule_name": entry.rule_name,
                }
                for entry in entries
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricRelationPolicy":
        try:
            return cls(
                format_version=payload.get(
                    "format_version", RELATION_POLICY_FORMAT_VERSION
                ),
                name=payload["name"],
                version=payload["version"],
                scope=payload["scope"],
                symmetry=payload["symmetry"],
                diagonal_behavior=payload["diagonal_behavior"],
                groups=tuple(
                    MetricGroupDefinition(
                        name=group["name"], tags=tuple(group.get("tags", ()))
                    )
                    for group in payload["groups"]
                ),
                pair_relations=tuple(
                    PairRelation(
                        left_group=entry["left_group"],
                        right_group=entry["right_group"],
                        relation=entry["relation"],
                        rule_name=entry["rule_name"],
                    )
                    for entry in payload["pair_relations"]
                ),
                rule_precedence=tuple(payload["rule_precedence"]),
                exclusion_rule_names=tuple(payload["exclusion_rule_names"]),
            )
        except (KeyError, TypeError) as exc:
            raise MetricRelationValidationError(
                f"invalid relation policy payload: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, text: str) -> "MetricRelationPolicy":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MetricRelationValidationError(
                f"invalid relation policy JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise MetricRelationValidationError("relation policy JSON must contain an object")
        return cls.from_dict(payload)

    def _normalize_pair(self, left_group: str, right_group: str) -> tuple[str, str]:
        if self.symmetry == "symmetric" and right_group < left_group:
            return (right_group, left_group)
        return (left_group, right_group)

    def _expected_pair_keys(self, names: tuple[str, ...]) -> Iterable[tuple[str, str]]:
        for left_index, left_group in enumerate(names):
            start = left_index if self.symmetry == "symmetric" else 0
            for right_group in names[start:]:
                yield (left_group, right_group)


CANDIDATE_RULE_PRECEDENCE = (
    "candidate_uncertain_exclusion",
    "candidate_same_non_uncertain_positive",
    "legacy_control_control_exclusion",
    "legacy_control_wild_type_exclusion",
    "legacy_different_crispants_exclusion",
    "legacy_crispant_wild_type_exclusion",
    "legacy_ab_wik_wild_type_exclusion",
    "legacy_mutant_background_wild_type_exclusion",
    "legacy_fluorescent_marker_wild_type_exclusion",
    "explicit_intentional_positive",
    "explicit_intentional_negative",
    "explicit_intentional_exclusion",
)

CANDIDATE_EXCLUSION_RULES = (
    "candidate_uncertain_exclusion",
    "legacy_control_control_exclusion",
    "legacy_control_wild_type_exclusion",
    "legacy_different_crispants_exclusion",
    "legacy_crispant_wild_type_exclusion",
    "legacy_ab_wik_wild_type_exclusion",
    "legacy_mutant_background_wild_type_exclusion",
    "legacy_fluorescent_marker_wild_type_exclusion",
    "explicit_intentional_exclusion",
)


def build_d35_compatibility_candidate(
    *,
    groups: Iterable[MetricGroupDefinition],
    explicit_relations: Iterable[PairRelation],
    version: str,
    name: str = "d35_compatibility_candidate_not_scientific",
) -> MetricRelationPolicy:
    """Build the non-scientific D35 compatibility candidate with full coverage.

    Candidate and legacy exclusions have precedence over explicit pair content.
    Supplying a shadowed explicit pair is rejected instead of silently ignoring it.
    """

    group_tuple = tuple(sorted(groups, key=lambda group: group.name))
    group_by_name = {group.name: group for group in group_tuple}
    if len(group_by_name) != len(group_tuple):
        raise MetricRelationValidationError("candidate metric groups contain duplicates")

    explicit_index: dict[tuple[str, str], PairRelation] = {}
    unknown: set[str] = set()
    allowed_explicit_rules = {
        "explicit_intentional_positive",
        "explicit_intentional_negative",
        "explicit_intentional_exclusion",
    }
    for entry in explicit_relations:
        unknown.update(
            group_name
            for group_name in (entry.left_group, entry.right_group)
            if group_name not in group_by_name
        )
        if entry.rule_name not in allowed_explicit_rules:
            raise MetricRelationValidationError(
                f"candidate explicit pair {entry.left_group!r} x {entry.right_group!r} "
                f"must use one of {sorted(allowed_explicit_rules)!r}; got "
                f"{entry.rule_name!r}"
            )
        expected_rule = f"explicit_intentional_{entry.relation.value}"
        if entry.rule_name != expected_rule:
            raise MetricRelationValidationError(
                f"candidate explicit pair {entry.left_group!r} x {entry.right_group!r} "
                f"relation={entry.relation.value!r} requires rule_name={expected_rule!r}"
            )
        key = _symmetric_pair(entry.left_group, entry.right_group)
        if key in explicit_index:
            raise MetricRelationValidationError(
                f"duplicate candidate explicit relation for pair={key!r}"
            )
        explicit_index[key] = entry
    if unknown:
        raise MetricRelationValidationError(
            f"candidate explicit relations reference unknown groups: {sorted(unknown)!r}"
        )

    pair_relations: list[PairRelation] = []
    uncovered: list[tuple[str, str]] = []
    shadowed: list[tuple[str, str]] = []
    names = tuple(group_by_name)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index:]:
            left = group_by_name[left_name]
            right = group_by_name[right_name]
            candidate = _candidate_relation(left, right)
            explicit = explicit_index.get((left_name, right_name))
            if candidate is not None:
                if explicit is not None:
                    shadowed.append((left_name, right_name))
                relation, rule_name = candidate
            elif explicit is not None:
                relation, rule_name = explicit.relation, explicit.rule_name
            else:
                uncovered.append((left_name, right_name))
                continue
            pair_relations.append(
                PairRelation(
                    left_group=left_name,
                    right_group=right_name,
                    relation=relation,
                    rule_name=rule_name,
                )
            )
    if shadowed:
        raise MetricRelationValidationError(
            "explicit candidate relations are shadowed by higher-precedence D35/legacy "
            f"rules for pairs: {shadowed!r}"
        )
    if uncovered:
        raise MetricRelationCoverageError(
            f"D35 compatibility candidate {name!r}@{version} has uncovered class pairs; "
            "add explicit intentional positive/negative/exclusion declarations for: "
            f"{uncovered!r}"
        )

    return MetricRelationPolicy(
        name=name,
        version=version,
        scope="compatibility_candidate",
        symmetry="symmetric",
        diagonal_behavior="uncertain_excluded_otherwise_positive",
        groups=group_tuple,
        pair_relations=tuple(pair_relations),
        rule_precedence=CANDIDATE_RULE_PRECEDENCE,
        exclusion_rule_names=CANDIDATE_EXCLUSION_RULES,
    )


def build_test_only_policy(
    group_names: Iterable[str], *, version: str = "1"
) -> MetricRelationPolicy:
    """Build an unmistakably trivial plumbing policy, never a science policy."""

    names = tuple(sorted(group_names))
    groups = tuple(MetricGroupDefinition(name=name) for name in names)
    entries = tuple(
        PairRelation(
            left_group=left_name,
            right_group=right_name,
            relation=(
                MetricRelation.POSITIVE
                if left_name == right_name
                else MetricRelation.NEGATIVE
            ),
            rule_name=(
                "test_only_same_group_positive"
                if left_name == right_name
                else "test_only_different_group_negative"
            ),
        )
        for left_index, left_name in enumerate(names)
        for right_name in names[left_index:]
    )
    return MetricRelationPolicy(
        name="test_only_trivial_metric_relations_not_scientific",
        version=version,
        scope="test_only",
        symmetry="symmetric",
        diagonal_behavior="positive",
        groups=groups,
        pair_relations=entries,
        rule_precedence=(
            "test_only_same_group_positive",
            "test_only_different_group_negative",
        ),
        exclusion_rule_names=(),
    )


def validate_policy_for_preset(
    policy: MetricRelationPolicy, *, scientific_preset: bool
) -> None:
    """Prevent test/candidate content from loading as a scientific policy."""

    if scientific_preset and policy.scope != "scientific":
        raise MetricRelationValidationError(
            f"scientific preset cannot use relation policy {policy.name!r}@{policy.version} "
            f"with scope={policy.scope!r}"
        )


def _candidate_relation(
    left: MetricGroupDefinition, right: MetricGroupDefinition
) -> tuple[MetricRelation, str] | None:
    if left.has(TAG_UNCERTAIN) or right.has(TAG_UNCERTAIN):
        return (MetricRelation.EXCLUDED, "candidate_uncertain_exclusion")
    if left.name == right.name:
        return (MetricRelation.POSITIVE, "candidate_same_non_uncertain_positive")
    if left.has(TAG_CONTROL) and right.has(TAG_CONTROL):
        return (MetricRelation.EXCLUDED, "legacy_control_control_exclusion")
    if _one_has_each(left, right, TAG_CONTROL, TAG_WILD_TYPE):
        return (MetricRelation.EXCLUDED, "legacy_control_wild_type_exclusion")
    if left.has(TAG_CRISPANT) and right.has(TAG_CRISPANT):
        return (MetricRelation.EXCLUDED, "legacy_different_crispants_exclusion")
    if _one_has_each(left, right, TAG_CRISPANT, TAG_WILD_TYPE):
        return (MetricRelation.EXCLUDED, "legacy_crispant_wild_type_exclusion")
    if _one_has_each(left, right, TAG_WT_AB, TAG_WT_WIK):
        return (MetricRelation.EXCLUDED, "legacy_ab_wik_wild_type_exclusion")
    if _one_has_each(left, right, TAG_WT_MUTANT_BACKGROUND, TAG_WILD_TYPE):
        return (
            MetricRelation.EXCLUDED,
            "legacy_mutant_background_wild_type_exclusion",
        )
    if _one_has_each(left, right, TAG_WT_FLUORESCENT_MARKER, TAG_WILD_TYPE):
        return (
            MetricRelation.EXCLUDED,
            "legacy_fluorescent_marker_wild_type_exclusion",
        )
    return None


def _one_has_each(
    left: MetricGroupDefinition,
    right: MetricGroupDefinition,
    first_tag: str,
    second_tag: str,
) -> bool:
    return (left.has(first_tag) and right.has(second_tag)) or (
        left.has(second_tag) and right.has(first_tag)
    )


def _symmetric_pair(left_group: str, right_group: str) -> tuple[str, str]:
    if right_group < left_group:
        return (right_group, left_group)
    return (left_group, right_group)
