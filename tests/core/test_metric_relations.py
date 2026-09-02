from __future__ import annotations

from dataclasses import replace

import pytest

from src.core.metric import (
    MetricGroupDefinition,
    MetricMappingArtifact,
    MetricMappingEntry,
    MetricRelation,
    MetricRelationCoverageError,
    MetricRelationPolicy,
    MetricRelationValidationError,
    PairRelation,
    UnknownMetricGroupError,
    build_d35_compatibility_candidate,
    build_metric_provenance_payload,
    build_test_only_policy,
    load_metric_provenance_payload,
    validate_metric_bundle_for_preset,
    validate_policy_for_preset,
)
from src.core.metric.relations import (
    TAG_CONTROL,
    TAG_CRISPANT,
    TAG_UNCERTAIN,
    TAG_WILD_TYPE,
    TAG_WT_AB,
    TAG_WT_FLUORESCENT_MARKER,
    TAG_WT_MUTANT_BACKGROUND,
    TAG_WT_WIK,
)


def _group(name: str, *tags: str) -> MetricGroupDefinition:
    return MetricGroupDefinition(name=name, tags=tags)


def _candidate(
    groups: tuple[MetricGroupDefinition, ...],
    explicit_relations: tuple[PairRelation, ...] = (),
) -> MetricRelationPolicy:
    return build_d35_compatibility_candidate(
        groups=groups,
        explicit_relations=explicit_relations,
        version="candidate-fixture-1",
    )


def _explicit(
    left: str, right: str, relation: MetricRelation
) -> PairRelation:
    return PairRelation(
        left_group=left,
        right_group=right,
        relation=relation,
        rule_name=f"explicit_intentional_{relation.value}",
    )


def test_uncertain_relations_are_excluded_including_the_diagonal() -> None:
    policy = _candidate((_group("crispant", TAG_CRISPANT), _group("uncertain", TAG_UNCERTAIN)))

    assert policy.relation("uncertain", "uncertain") is MetricRelation.EXCLUDED
    assert policy.relation("uncertain", "crispant") is MetricRelation.EXCLUDED
    assert policy.relation("crispant", "uncertain") is MetricRelation.EXCLUDED


def test_non_uncertain_diagonals_are_positive() -> None:
    policy = _candidate((_group("crispant", TAG_CRISPANT),))
    assert policy.relation("crispant", "crispant") is MetricRelation.POSITIVE


def test_different_crispants_preserve_named_legacy_exclusion() -> None:
    policy = _candidate(
        (_group("crispant_a", TAG_CRISPANT), _group("crispant_b", TAG_CRISPANT))
    )

    assert policy.relation("crispant_a", "crispant_b") is MetricRelation.EXCLUDED
    assert "legacy_different_crispants_exclusion" in policy.truth_table()


@pytest.mark.parametrize(
    ("groups", "expected_rule"),
    [
        (
            (_group("control_a", TAG_CONTROL), _group("control_b", TAG_CONTROL)),
            "legacy_control_control_exclusion",
        ),
        (
            (_group("control", TAG_CONTROL), _group("wt", TAG_WILD_TYPE)),
            "legacy_control_wild_type_exclusion",
        ),
        (
            (_group("crispant", TAG_CRISPANT), _group("wt", TAG_WILD_TYPE)),
            "legacy_crispant_wild_type_exclusion",
        ),
        (
            (
                _group("wt_ab", TAG_WILD_TYPE, TAG_WT_AB),
                _group("wt_wik", TAG_WILD_TYPE, TAG_WT_WIK),
            ),
            "legacy_ab_wik_wild_type_exclusion",
        ),
        (
            (
                _group("mutant_background_wt", TAG_WILD_TYPE, TAG_WT_MUTANT_BACKGROUND),
                _group("wt", TAG_WILD_TYPE),
            ),
            "legacy_mutant_background_wild_type_exclusion",
        ),
        (
            (
                _group("fluorescent_wt", TAG_WILD_TYPE, TAG_WT_FLUORESCENT_MARKER),
                _group("wt", TAG_WILD_TYPE),
            ),
            "legacy_fluorescent_marker_wild_type_exclusion",
        ),
    ],
)
def test_other_legacy_exclusions_are_explicitly_named(groups, expected_rule) -> None:
    policy = _candidate(groups)
    assert policy.relation(groups[0].name, groups[1].name) is MetricRelation.EXCLUDED
    assert expected_rule in policy.truth_table()


def test_intentional_negative_must_be_explicitly_declared() -> None:
    groups = (_group("inhibitor_24hpf"), _group("inhibitor_30hpf"))
    with pytest.raises(MetricRelationCoverageError) as exc_info:
        _candidate(groups)
    assert "uncovered class pairs" in str(exc_info.value)
    assert "inhibitor_24hpf" in str(exc_info.value)
    assert "inhibitor_30hpf" in str(exc_info.value)

    policy = _candidate(
        groups,
        (_explicit("inhibitor_24hpf", "inhibitor_30hpf", MetricRelation.NEGATIVE),),
    )
    assert (
        policy.relation("inhibitor_24hpf", "inhibitor_30hpf")
        is MetricRelation.NEGATIVE
    )
    assert "explicit_intentional_negative" in policy.truth_table()


def test_explicit_relation_cannot_silently_override_higher_precedence_exclusion() -> None:
    groups = (_group("crispant_a", TAG_CRISPANT), _group("crispant_b", TAG_CRISPANT))
    with pytest.raises(MetricRelationValidationError, match="shadowed.*crispant_a"):
        _candidate(
            groups,
            (_explicit("crispant_a", "crispant_b", MetricRelation.NEGATIVE),),
        )


def test_direct_incomplete_policy_fails_before_lookup_without_negative_fallthrough() -> None:
    with pytest.raises(MetricRelationCoverageError, match="uncovered_pairs.*alpha.*beta"):
        MetricRelationPolicy(
            name="invalid_incomplete_fixture",
            version="1",
            scope="test_only",
            symmetry="symmetric",
            diagonal_behavior="positive",
            groups=(_group("alpha"), _group("beta")),
            pair_relations=(
                PairRelation("alpha", "alpha", MetricRelation.POSITIVE, "same"),
                PairRelation("beta", "beta", MetricRelation.POSITIVE, "same"),
            ),
            rule_precedence=("same",),
            exclusion_rule_names=(),
        )


def test_symmetric_policy_rejects_reverse_duplicate_and_lookup_is_symmetric() -> None:
    policy = build_test_only_policy(("beta", "alpha"))
    assert policy.relation("alpha", "beta") is MetricRelation.NEGATIVE
    assert policy.relation("beta", "alpha") is MetricRelation.NEGATIVE

    reverse_only = replace(
        policy,
        pair_relations=tuple(
            PairRelation(
                "beta",
                "alpha",
                entry.relation,
                entry.rule_name,
            )
            if (entry.left_group, entry.right_group) == ("alpha", "beta")
            else entry
            for entry in policy.pair_relations
        ),
    )
    assert reverse_only.to_json() == policy.to_json()

    duplicate = PairRelation(
        "beta", "alpha", MetricRelation.NEGATIVE, "test_only_different_group_negative"
    )
    with pytest.raises(MetricRelationValidationError, match="duplicate relation"):
        replace(policy, pair_relations=policy.pair_relations + (duplicate,))


def test_asymmetric_policy_requires_and_preserves_both_directions() -> None:
    policy = MetricRelationPolicy(
        name="test_only_asymmetric_fixture",
        version="1",
        scope="test_only",
        symmetry="asymmetric",
        diagonal_behavior="positive",
        groups=(_group("alpha"), _group("beta")),
        pair_relations=(
            PairRelation("alpha", "alpha", MetricRelation.POSITIVE, "diagonal"),
            PairRelation("beta", "beta", MetricRelation.POSITIVE, "diagonal"),
            PairRelation("alpha", "beta", MetricRelation.POSITIVE, "directed_positive"),
            PairRelation("beta", "alpha", MetricRelation.EXCLUDED, "directed_exclusion"),
        ),
        rule_precedence=("diagonal", "directed_positive", "directed_exclusion"),
        exclusion_rule_names=("directed_exclusion",),
    )

    assert policy.relation("alpha", "beta") is MetricRelation.POSITIVE
    assert policy.relation("beta", "alpha") is MetricRelation.EXCLUDED

    with pytest.raises(MetricRelationCoverageError, match="uncovered_pairs.*beta.*alpha"):
        replace(
            policy,
            pair_relations=tuple(
                entry
                for entry in policy.pair_relations
                if (entry.left_group, entry.right_group) != ("beta", "alpha")
            ),
        )


def test_unknown_group_lookup_fails_by_group_and_policy() -> None:
    policy = build_test_only_policy(("alpha", "beta"))
    with pytest.raises(UnknownMetricGroupError, match="not_declared.*test_only_trivial"):
        policy.relation("alpha", "not_declared")


def test_truth_table_is_complete_deterministic_and_serialization_round_trips() -> None:
    first = build_test_only_policy(("zeta", "alpha"), version="fixture-2")
    second = build_test_only_policy(("alpha", "zeta"), version="fixture-2")

    assert first.truth_table() == second.truth_table()
    assert first.to_json() == second.to_json()
    assert len(first.truth_table().strip().splitlines()) == 5
    restored = MetricRelationPolicy.from_json(first.to_json())
    assert restored.to_json() == first.to_json()
    assert restored.relation("zeta", "alpha") is MetricRelation.NEGATIVE


def test_test_and_candidate_policies_cannot_load_through_scientific_preset() -> None:
    test_policy = build_test_only_policy(("alpha",))
    candidate = _candidate((_group("alpha"),))

    for policy in (test_policy, candidate):
        with pytest.raises(MetricRelationValidationError, match="scientific preset"):
            validate_policy_for_preset(policy, scientific_preset=True)
        validate_policy_for_preset(policy, scientific_preset=False)


def test_mapping_and_relation_policy_round_trip_as_exact_provenance_payload() -> None:
    mapping = MetricMappingArtifact(
        name="test_only_mapping",
        version="map-v7",
        metadata_columns=("label",),
        entries=(
            MetricMappingEntry(("A",), "alpha"),
            MetricMappingEntry(("B",), "beta"),
        ),
        scientific_policy=False,
    )
    relations = build_test_only_policy(("alpha", "beta"), version="rel-v9")

    payload = build_metric_provenance_payload(
        mapping=mapping, relation_policy=relations
    )
    restored_mapping, restored_relations = load_metric_provenance_payload(payload)

    assert payload["mapping"]["version"] == "map-v7"
    assert payload["relation_policy"]["version"] == "rel-v9"
    assert restored_mapping.to_json() == mapping.to_json()
    assert restored_relations.to_json() == relations.to_json()
    with pytest.raises(MetricRelationValidationError, match="scientific preset"):
        validate_metric_bundle_for_preset(
            mapping=mapping,
            relation_policy=relations,
            scientific_preset=True,
        )


def test_provenance_rejects_mapping_groups_absent_from_relation_coverage() -> None:
    mapping = MetricMappingArtifact(
        name="test_only_mapping",
        version="1",
        metadata_columns=("label",),
        entries=(MetricMappingEntry(("A",), "mapping_only_group"),),
    )
    policy = build_test_only_policy(("relation_only_group",))

    with pytest.raises(
        MetricRelationCoverageError, match="mapping_only_group.*relation_only_group"
    ):
        build_metric_provenance_payload(mapping=mapping, relation_policy=policy)
