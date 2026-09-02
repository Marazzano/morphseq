from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.core.data import dataset_configs
from src.core.data.dataset_configs import PipelineMetricDataConfig
from src.core.data.manifest_types import MetricMappingPolicy
from src.core.metric import (
    MetricMappingArtifact,
    MetricMappingEntry,
    build_test_only_policy,
)
from src.core.metric.pairing import (
    DifferentEmbryoCandidatePolicy,
    MetricPairIndex,
    MetricPairSampler,
    MetricPairingPolicy,
    SameEmbryoCandidatePolicy,
    derive_pair_seed,
)
from tests.core.fixtures.manifest_v2 import BF_PRODUCT, synthetic_manifest_v2


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snip_id": ("anchor", "same-animal", "other-a", "other-b", "negative"),
            "physical_embryo_id": ("embryo-x", "embryo-x", "embryo-y", "embryo-z", "embryo-q"),
            "split": ("train",) * 5,
            "metric_group": ("alpha", "alpha", "alpha", "alpha", "beta"),
            "pair_stage_hpf": (10.0, 10.25, 9.5, 10.5, 10.0),
        }
    )


def _policy(
    *,
    same_probability: float,
    same_enabled: bool = True,
    different_enabled: bool = True,
) -> MetricPairingPolicy:
    return MetricPairingPolicy(
        name="test_only_reproducible_pairing",
        version="fixture-1",
        stage_column="pair_stage_hpf",
        stage_source="fixture inferred axis",
        sampler_age_window=0.5,
        same_embryo=SameEmbryoCandidatePolicy(
            enabled=same_enabled,
            allow_same_observation=False,
        ),
        different_embryo=DifferentEmbryoCandidatePolicy(enabled=different_enabled),
        same_embryo_probability=same_probability,
        base_seed=917,
    )


def _index(policy: MetricPairingPolicy, table: pd.DataFrame | None = None) -> MetricPairIndex:
    return MetricPairIndex(
        _table() if table is None else table,
        relation_policy=build_test_only_policy(("alpha", "beta")),
        pairing_policy=policy,
        split="train",
    )


def test_same_and_different_embryo_selection_are_separate_explicit_policies() -> None:
    same_sampler = MetricPairSampler(
        _index(_policy(same_probability=1.0, different_enabled=False))
    )
    different_sampler = MetricPairSampler(
        _index(_policy(same_probability=0.0, same_enabled=False))
    )

    same = same_sampler.sample(0)
    different = different_sampler.sample(0)

    assert same.candidate_kind == "same_embryo"
    assert same_sampler.snip_ids[same.other_index] == "same-animal"
    assert different.candidate_kind == "different_embryo"
    assert different_sampler.snip_ids[different.other_index] in {"other-a", "other-b"}


def test_sampling_is_stable_under_resolved_row_reordering() -> None:
    policy = _policy(same_probability=0.0, same_enabled=False)
    original = _table()
    reordered = original.iloc[[4, 2, 0, 3, 1]].reset_index(drop=True)
    first = MetricPairSampler(_index(policy, original))
    second = MetricPairSampler(_index(policy, reordered))
    first_anchor = first.snip_ids.index("anchor")
    second_anchor = second.snip_ids.index("anchor")

    first_selection = first.sample(first_anchor)
    second_selection = second.sample(second_anchor)

    assert first.snip_ids[first_selection.other_index] == second.snip_ids[
        second_selection.other_index
    ]


def test_worker_rank_and_epoch_seed_derivation_is_reproducible_and_explicit() -> None:
    kwargs = {
        "base_seed": 11,
        "policy_name": "test_only_seed",
        "policy_version": "1",
        "split": "train",
        "anchor_snip_id": "opaque-anchor",
        "epoch": 3,
        "rank": 2,
        "worker_id": 4,
        "draw_index": 0,
    }
    seed = derive_pair_seed(**kwargs)

    assert seed == derive_pair_seed(**kwargs)
    assert seed != derive_pair_seed(**{**kwargs, "worker_id": 5})
    assert seed != derive_pair_seed(**{**kwargs, "rank": 3})
    assert seed != derive_pair_seed(**{**kwargs, "epoch": 4})

    sampler = MetricPairSampler(_index(_policy(same_probability=0.0, same_enabled=False)), rank=2)
    sampler.set_epoch(3)
    first = sampler.sample(0, worker_id=4)
    second = sampler.sample(0, worker_id=4)
    assert first == second
    assert first.seed != sampler.sample(0, worker_id=5).seed


def test_query_uses_sorted_indexes_without_full_cohort_boolean_allocation() -> None:
    row_count = 5000
    table = pd.DataFrame(
        {
            "snip_id": [f"opaque-{index:05d}" for index in range(row_count)],
            "physical_embryo_id": [f"physical-{index:05d}" for index in range(row_count)],
            "split": ["train"] * row_count,
            "metric_group": ["alpha"] * row_count,
            "pair_stage_hpf": [float(index % 100) for index in range(row_count)],
        }
    )
    policy = MetricPairingPolicy(
        name="test_only_large_index",
        version="1",
        stage_column="pair_stage_hpf",
        stage_source="fixture stage",
        sampler_age_window=0.0,
        same_embryo=SameEmbryoCandidatePolicy(enabled=False),
        different_embryo=DifferentEmbryoCandidatePolicy(enabled=True),
        same_embryo_probability=0.0,
    )
    pair_index = MetricPairIndex(
        table,
        relation_policy=build_test_only_policy(("alpha",)),
        pairing_policy=policy,
    )

    diagnostics = pair_index.query_diagnostics(0)

    assert diagnostics.index_strategy == "sorted_age_ranges_by_group_and_physical_embryo"
    assert diagnostics.group_range_lookups == 1
    assert diagnostics.same_embryo_positions_examined == 1
    assert diagnostics.full_length_boolean_allocations == 0
    assert pair_index.candidate_counts(0).different_embryo == 49


def test_pipeline_metric_config_maps_preflights_and_builds_split_local_dataset(
    tmp_path, monkeypatch
) -> None:
    manifest = synthetic_manifest_v2(tmp_path)
    for row_number, path_string in enumerate(manifest.asset_table["processed_snip_path"]):
        image_path = Path(path_string)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((24, 18), row_number * 17, dtype=np.uint8)).save(image_path)

    mapping = MetricMappingArtifact(
        name="test_only_manifest_metric_mapping",
        version="1",
        metadata_columns=("genotype",),
        entries=(
            MetricMappingEntry(
                source_values=("source-spelling/control",), metric_group="alpha"
            ),
        ),
        scientific_policy=False,
    )
    manifest_policy = replace(
        manifest.policy,
        metric_mapping=MetricMappingPolicy(
            enabled=True,
            name=mapping.name,
            scientific_policy=False,
        ),
    )
    manifest = replace(manifest, policy=manifest_policy)
    monkeypatch.setattr(
        dataset_configs,
        "build_pipeline_manifest",
        lambda *args, **kwargs: manifest,
    )
    pairing = MetricPairingPolicy(
        name="test_only_manifest_same_observation_pairs",
        version="1",
        stage_column="predicted_stage_hpf",
        stage_source="fixture-clock-v1 predicted_stage_hpf",
        sampler_age_window=0.0,
        same_embryo=SameEmbryoCandidatePolicy(
            enabled=True, allow_same_observation=True
        ),
        different_embryo=DifferentEmbryoCandidatePolicy(enabled=False),
        same_embryo_probability=1.0,
        base_seed=29,
        scientific_policy=False,
    )
    config = PipelineMetricDataConfig(
        pipeline_output_root=tmp_path,
        manifest_policy=manifest_policy,
        metric_mapping_artifact=mapping,
        relation_policy=build_test_only_policy(("alpha",)),
        pairing_policy=pairing,
        input_dim=(1, 24, 18),
        batch_size=1,
        num_workers=0,
    )

    result = config.make_metadata()
    dataset = config.create_dataset(split="train")

    def fail_candidate_materialization(*args, **kwargs):
        raise AssertionError("normal __getitem__ materialized its candidate set")

    monkeypatch.setattr(
        config.pair_indices["train"],
        "candidate_indices",
        fail_candidate_materialization,
    )
    item = dataset[0]

    assert result.resolved_sample_table["metric_group"].tolist() == [
        "alpha",
        "alpha",
        "alpha",
    ]
    assert result.resolved_sample_table["metric_group_code"].tolist() == [0, 0, 0]
    assert {report.split for report in config.pair_preflight_reports} == {
        "train",
        "eval",
        "test",
    }
    assert item.data.shape == (2, 1, 24, 18)
    assert item.self_stats == ["animal::alpha", 24.0, "alpha"]
    assert item.other_stats == item.self_stats
    assert item.pair_candidate_kind == "same_embryo"
    assert item.pair_policy_name == pairing.name
    assert item.pair_stage_source == pairing.stage_source
    assert config.metric_provenance_payload["mapping"]["name"] == mapping.name
