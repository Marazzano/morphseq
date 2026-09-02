from __future__ import annotations

import pandas as pd
import pytest

from src.core.metric import (
    MetricGroupDefinition,
    MetricRelation,
    MetricRelationPolicy,
    PairRelation,
)
from src.core.metric.pair_preflight import NoLegalPositiveError, preflight_pair_index
from src.core.metric.pairing import (
    DifferentEmbryoCandidatePolicy,
    MetricPairIndex,
    MetricPairingConfigurationError,
    MetricPairingPolicy,
    SameEmbryoCandidatePolicy,
)


def _relation_policy() -> MetricRelationPolicy:
    return MetricRelationPolicy(
        name="test_only_pair_preflight_relations",
        version="1",
        scope="test_only",
        symmetry="symmetric",
        diagonal_behavior="positive",
        groups=tuple(
            MetricGroupDefinition(name=name) for name in ("alpha", "beta", "gamma")
        ),
        pair_relations=(
            PairRelation("alpha", "alpha", MetricRelation.POSITIVE, "same_group"),
            PairRelation("alpha", "beta", MetricRelation.NEGATIVE, "negative"),
            PairRelation("alpha", "gamma", MetricRelation.EXCLUDED, "excluded"),
            PairRelation("beta", "beta", MetricRelation.POSITIVE, "same_group"),
            PairRelation("beta", "gamma", MetricRelation.NEGATIVE, "negative"),
            PairRelation("gamma", "gamma", MetricRelation.POSITIVE, "same_group"),
        ),
        rule_precedence=("same_group", "negative", "excluded"),
        exclusion_rule_names=("excluded",),
    )


def _policy(
    *,
    window: float = 1.0,
    same_enabled: bool = True,
    allow_same_observation: bool = False,
    different_enabled: bool = True,
) -> MetricPairingPolicy:
    probability = 0.5
    if not same_enabled:
        probability = 0.0
    elif not different_enabled:
        probability = 1.0
    return MetricPairingPolicy(
        name="test_only_indexed_pairing",
        version="1",
        stage_column="pair_stage_hpf",
        stage_source="test fixture stage axis v1",
        sampler_age_window=window,
        same_embryo=SameEmbryoCandidatePolicy(
            enabled=same_enabled,
            allow_same_observation=allow_same_observation,
        ),
        different_embryo=DifferentEmbryoCandidatePolicy(enabled=different_enabled),
        same_embryo_probability=probability,
        base_seed=41,
    )


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snip_id": ("opaque::a0", "opaque::a1", "opaque::a2", "opaque::b0", "opaque::b1"),
            "physical_embryo_id": (
                "physical::unrelated-x",
                "physical::unrelated-x",
                "physical::unrelated-y",
                "physical::unrelated-z",
                "physical::unrelated-w",
            ),
            "split": ("train",) * 5,
            "metric_group": ("alpha", "alpha", "alpha", "beta", "beta"),
            "pair_stage_hpf": (10.0, 10.5, 11.0, 10.0, 10.5),
        }
    )


def test_preflight_checks_every_anchor_and_reports_candidate_surfaces() -> None:
    pair_index = MetricPairIndex(
        _table(), relation_policy=_relation_policy(), pairing_policy=_policy(), split="train"
    )

    report = preflight_pair_index(pair_index)

    assert report.anchor_count == 5
    assert report.anchors_with_same_embryo_candidates == 2
    assert report.anchors_with_different_embryo_candidates == 5
    assert report.minimum_candidate_count > 0


def test_no_legal_positive_names_anchor_policy_stage_source_and_window() -> None:
    table = _table().iloc[[0]].copy()
    policy = _policy(same_enabled=False, different_enabled=True)
    pair_index = MetricPairIndex(
        table, relation_policy=_relation_policy(), pairing_policy=policy, split="train"
    )

    with pytest.raises(NoLegalPositiveError) as exc_info:
        preflight_pair_index(pair_index)

    message = str(exc_info.value)
    assert "opaque::a0" in message
    assert "test_only_indexed_pairing" in message
    assert "test fixture stage axis v1" in message
    assert "sampler_age_window=1" in message


def test_age_window_is_closed_and_relation_exclusions_are_honored() -> None:
    table = pd.DataFrame(
        {
            "snip_id": ("anchor", "boundary", "outside", "excluded"),
            "physical_embryo_id": ("embryo-a", "embryo-b", "embryo-c", "embryo-d"),
            "split": ("train",) * 4,
            "metric_group": ("alpha", "alpha", "alpha", "gamma"),
            "pair_stage_hpf": (10.0, 11.0, 11.0001, 10.0),
        }
    )
    pair_index = MetricPairIndex(
        table,
        relation_policy=_relation_policy(),
        pairing_policy=_policy(same_enabled=False),
    )

    assert pair_index.candidate_indices(0) == (1,)
    assert pair_index.candidate_counts(0).different_embryo == 1


def test_index_is_strictly_split_local() -> None:
    table = _table()
    table.loc[table.index[-1], "split"] = "eval"

    with pytest.raises(MetricPairingConfigurationError, match="cannot index multiple splits"):
        MetricPairIndex(
            table, relation_policy=_relation_policy(), pairing_policy=_policy()
        )


def test_physical_embryo_id_is_consumed_as_an_opaque_explicit_column() -> None:
    table = _table().iloc[:2].copy()
    table["snip_id"] = ("no-shared-prefix-a", "no-shared-prefix-b")
    table["physical_embryo_id"] = ("same-opaque-animal", "same-opaque-animal")
    pair_index = MetricPairIndex(
        table,
        relation_policy=_relation_policy(),
        pairing_policy=_policy(different_enabled=False),
    )

    assert pair_index.candidate_indices(0, "same_embryo") == (1,)


def test_sibling_assets_cannot_silently_overweight_one_observation() -> None:
    table = _table().iloc[:2].copy()
    table.loc[table.index[1], "snip_id"] = table.iloc[0]["snip_id"]
    table["z_index"] = (0, 1)

    with pytest.raises(
        MetricPairingConfigurationError,
        match=r"uniform_observation.*sibling product/z assets.*opaque::a0",
    ):
        MetricPairIndex(
            table, relation_policy=_relation_policy(), pairing_policy=_policy()
        )
