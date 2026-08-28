from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.core.data.manifest_types import (
    AssetSelectionPolicy,
    ManifestPolicy,
    MetricMappingPolicy,
    SplitPolicy,
)
from src.core.data.pipeline_contracts import (
    CURRENT_WRITER_SYMBOLS,
    PipelineContractImportError,
    fingerprint_artifact,
    read_artifact_table,
    resolve_experiment_paths,
)
from src.core.data.pipeline_manifest import (
    assign_group_splits,
    parse_boolean,
    validate_manifest_tables,
)


def _observation_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snip_id": "opaque-observation-A",
                "physical_embryo_id": "opaque-animal-A",
                "embryo_id": "opaque-embryo-A",
                "well_id": "opaque-well-A",
                "experiment_id": "exp-A",
            },
            {
                "snip_id": "opaque-observation-B",
                "physical_embryo_id": "opaque-animal-B",
                "embryo_id": "opaque-embryo-B",
                "well_id": "opaque-well-B",
                "experiment_id": "exp-A",
            },
        ]
    )


def _asset_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snip_id": "opaque-observation-A",
                "snip_product_key": "BF-product",
                "z_index": pd.NA,
                "physical_embryo_id": "opaque-animal-A",
                "embryo_id": "opaque-embryo-A",
                "well_id": "opaque-well-A",
                "experiment_id": "exp-A",
            },
            {
                "snip_id": "opaque-observation-A",
                "snip_product_key": "RFP-product",
                "z_index": pd.NA,
                "physical_embryo_id": "opaque-animal-A",
                "embryo_id": "opaque-embryo-A",
                "well_id": "opaque-well-A",
                "experiment_id": "exp-A",
            },
            {
                "snip_id": "opaque-observation-A",
                "snip_product_key": "BF-z-product",
                "z_index": 0,
                "physical_embryo_id": "opaque-animal-A",
                "embryo_id": "opaque-embryo-A",
                "well_id": "opaque-well-A",
                "experiment_id": "exp-A",
            },
        ]
    )


def test_observation_and_asset_contract_accepts_products_and_z_planes():
    validate_manifest_tables(_observation_table(), _asset_table())


def test_observation_key_must_be_unique():
    observations = pd.concat([_observation_table(), _observation_table().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate snip_id.*opaque-observation-A"):
        validate_manifest_tables(observations, _asset_table())


def test_asset_compound_key_treats_null_projection_as_real_key_member():
    assets = pd.concat([_asset_table(), _asset_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate.*projection null"):
        validate_manifest_tables(_observation_table(), assets)


def test_orphan_asset_names_opaque_observation():
    assets = _asset_table().copy()
    assets.loc[0, "snip_id"] = "opaque-orphan"
    with pytest.raises(ValueError, match="orphan.*opaque-orphan"):
        validate_manifest_tables(_observation_table(), assets)


def test_conflicting_asset_parent_names_key_and_field():
    assets = _asset_table().copy()
    assets.loc[0, "physical_embryo_id"] = "different-animal"
    with pytest.raises(ValueError, match="BF-product.*physical_embryo_id"):
        validate_manifest_tables(_observation_table(), assets)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (1, True), (0, False), ("True", True), ("False", False)],
)
def test_safe_boolean_parser(value, expected):
    assert parse_boolean(value, column="flag", identity="opaque") is expected


def test_safe_boolean_parser_rejects_unknown_string():
    with pytest.raises(ValueError, match="unrecognized boolean"):
        parse_boolean("not-a-bool", column="flag", identity="opaque")


def test_manifest_policy_requires_explicit_unique_experiments(manifest_policy_factory):
    policy = manifest_policy_factory()
    with pytest.raises(ValueError, match="contains duplicates"):
        ManifestPolicy(
            pipeline_output_root=policy.pipeline_output_root,
            experiment_ids=("exp-A", "exp-A"),
            assets=policy.assets,
            validity=policy.validity,
            qc=policy.qc,
            stage=policy.stage,
            covariates=policy.covariates,
            splits=policy.splits,
            metric_mapping=policy.metric_mapping,
        )


def test_constant_metric_mapping_is_unmistakably_test_only():
    with pytest.raises(ValueError, match="name must contain"):
        MetricMappingPolicy(True, "single_group", "v1", constant_group="all")
    with pytest.raises(ValueError, match="cannot be scientific"):
        MetricMappingPolicy(
            True,
            "test_only_single_group",
            "v1",
            scientific_policy=True,
            constant_group="all",
        )


def _split_observations(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "physical_embryo_id": [f"opaque-group-{index}" for index in range(n)],
            "experiment_id": ["exp-A"] * n,
        }
    )


def test_group_splits_are_disjoint_reorder_and_growth_stable(manifest_policy_factory):
    split = SplitPolicy(
        enabled=True,
        ratios=(("train", 0.7), ("eval", 0.2), ("test", 0.1)),
        required_splits=("train", "eval", "test"),
        tolerance=0.08,
        hash_salt="stable-test",
    )
    policy = manifest_policy_factory(split_policy=split)
    original = _split_observations()
    _, assignments = assign_group_splits(original, policy)
    _, reordered = assign_group_splits(original.sample(frac=1, random_state=17), policy)
    grown = pd.concat(
        [
            original,
            pd.DataFrame(
                {
                    "physical_embryo_id": [f"new-group-{index}" for index in range(50)],
                    "experiment_id": ["exp-A"] * 50,
                }
            ),
        ],
        ignore_index=True,
    )
    _, grown_assignments = assign_group_splits(grown, policy)
    assert assignments == reordered
    assert all(grown_assignments[group] == value for group, value in assignments.items())
    assert set(assignments.values()) == {"train", "eval", "test"}


def test_explicit_test_experiment_goes_wholly_to_test(manifest_policy_factory):
    observations = pd.DataFrame(
        {
            "physical_embryo_id": ["shared-group", "shared-group", "other-group"],
            "experiment_id": ["exp-test", "exp-train", "exp-train"],
        }
    )
    split = SplitPolicy(
        enabled=True,
        ratios=(("train", 1.0), ("eval", 0.0), ("test", 0.0)),
        test_experiments=("exp-test",),
        required_splits=("train", "test"),
        tolerance=0,
    )
    policy = manifest_policy_factory(
        experiment_ids=("exp-train", "exp-test"), split_policy=split
    )
    table, assignments = assign_group_splits(observations, policy)
    assert assignments["shared-group"] == "test"
    assert table.set_index("physical_embryo_id").at["shared-group", "assignment_source"] == (
        "explicit_test_experiment"
    )


def test_empty_required_split_fails_unconditionally(manifest_policy_factory):
    split = SplitPolicy(
        enabled=True,
        ratios=(("train", 1.0), ("eval", 0.0), ("test", 0.0)),
        required_splits=("eval",),
        tolerance=1.0,
    )
    policy = manifest_policy_factory(split_policy=split)
    with pytest.raises(ValueError, match="non-empty split 'eval'"):
        assign_group_splits(_split_observations(2), policy)


def test_missing_parquet_engine_is_actionable(monkeypatch, tmp_path):
    path = tmp_path / "snip_qc.parquet"
    path.write_bytes(b"not-relevant")

    def unavailable(*args, **kwargs):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd, "read_parquet", unavailable)
    with pytest.raises(RuntimeError, match="install.*pyarrow.*QC must not be skipped"):
        read_artifact_table(path, source_name="snip_qc", experiment_id="exp-A")


def test_source_inventory_records_path_stats_rows_and_hash(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("snip_id\nopaque-A\n", encoding="utf-8")
    record = fingerprint_artifact(
        experiment_id="exp-A",
        source_name="snip_inventory",
        path=path,
        required=True,
        row_count=1,
        schema_version="writer-v2",
    )
    assert record.exists
    assert record.path == path
    assert record.size_bytes == path.stat().st_size
    assert record.mtime_ns == path.stat().st_mtime_ns
    assert record.row_count == 1
    assert len(record.sha256) == 64
    assert record.schema_version == "writer-v2"


def test_current_writer_symbols_cover_every_declared_source():
    joined = "\n".join(CURRENT_WRITER_SYMBOLS)
    for token in (
        "SNIP_INVENTORY_WRITE_COLUMNS",
        "REQUIRED_FRAME_INVENTORY_COLUMNS",
        "STAGE_PREDICTION_TABLE_COLUMNS",
        "SNIP_QC_TABLE_COLUMNS",
        "REQUIRED_PLATE_METADATA_COLUMNS",
        "REQUIRED_COLLECTION_PROVENANCE_KEYS",
        "artifact_path",
        "resolve_from_root",
    ):
        assert token in joined


def test_hydra_manifest_policy_forces_root_and_ordered_experiments():
    path = Path("src/core/hydra_configs/data/pipeline_manifest_smoke.yaml")
    config = OmegaConf.load(path)
    assert OmegaConf.is_missing(config, "pipeline_output_root")
    assert OmegaConf.is_missing(config, "experiment_ids")
    assert config.assets.vanilla_product_key == (
        "BF__projection__focus_stack__clahe_blend"
    )
    assert config.stage.enabled is False
    assert config.metric_mapping.enabled is False


def test_pipeline_import_packaging_failure_is_actionable_without_path_mutation(monkeypatch):
    import src.core.data.pipeline_contracts as contracts

    def missing_package(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(contracts.importlib, "import_module", missing_package)
    with pytest.raises(PipelineContractImportError, match="Do not work around.*PYTHONPATH"):
        resolve_experiment_paths(
            output_root=Path("/pipeline/output"), experiment_id="explicit-experiment"
        )
