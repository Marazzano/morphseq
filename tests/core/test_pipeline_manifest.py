from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

import src.core.data.dataset_configs as dataset_configs
from src.core.data.dataset_configs import PipelineDataConfig
from src.core.data.manifest_types import (
    CovariatePolicy,
    ManifestPolicy,
    MetricMappingPolicy,
    QCPolicy,
    SplitPolicy,
    StagePolicy,
)
from src.core.data.pipeline_manifest import (
    assign_group_splits,
    build_pipeline_manifest,
    preflight_pipeline_sources,
)
from tests.core.fixtures.pipeline_source_tables import (
    BF_PRODUCT,
    RFP_PRODUCT,
    write_pipeline_source_fixture,
)
from tests.core.fixtures.manifest_v2 import synthetic_manifest_v2


def _policy(
    experiment_ids: tuple[str, ...],
    *,
    qc: QCPolicy | None = None,
    stage: StagePolicy | None = None,
    covariates: CovariatePolicy | None = None,
    splits: SplitPolicy | None = None,
) -> ManifestPolicy:
    return ManifestPolicy(
        name="test_only_pipeline_manifest",
        version="1",
        experiment_ids=experiment_ids,
        allowed_product_keys=(BF_PRODUCT, RFP_PRODUCT),
        selected_product_key=BF_PRODUCT,
        qc=qc or QCPolicy(name="fixture_qc", version="1"),
        stage=stage or StagePolicy(enabled=False),
        covariates=covariates or CovariatePolicy(),
        splits=splits or SplitPolicy(enabled=False),
    )


def test_build_carries_exact_temperature_time_stage_qc_and_assets(tmp_path: Path) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(
        tmp_path, experiment_id, booleans_as_strings=True
    )
    result = build_pipeline_manifest(
        tmp_path,
        _policy((experiment_id,)),
        source_paths={experiment_id: paths},
    )

    assert result.observation_table["snip_id"].tolist() == [
        "observation-opaque-0",
        "observation-opaque-1",
        "observation-opaque-2",
        "observation-opaque-3",
    ]
    assert len(result.asset_table) == 5
    assert result.asset_table.iloc[:2]["snip_product_key"].tolist() == [
        BF_PRODUCT,
        RFP_PRODUCT,
    ]
    assert result.resolved_sample_table["snip_product_key"].eq(BF_PRODUCT).all()
    assert result.resolved_sample_table["z_index"].isna().all()
    assert result.observation_table["incubation_temperature_c"].tolist() == pytest.approx(
        [27.5, 27.6, 27.7, 27.8]
    )
    assert result.observation_table["elapsed_time_s"].tolist() == pytest.approx(
        [0.0, 600.0, 1200.0, 1800.0]
    )
    assert set(result.observation_table["elapsed_time_status"]) == {"available"}
    assert set(result.observation_table["temperature_status"]) == {"available"}
    assert set(result.observation_table["stage_status"]) == {"predicted"}
    assert set(result.observation_table["stage_model_version"]) == {"fixture-stage-v1"}
    assert set(result.observation_table["qc_status"]) == {"evaluated"}
    assert result.observation_table["use_snip"].eq(True).all()  # noqa: E712
    assert result.observation_table["sa_outlier_flag"].eq(False).all()  # noqa: E712
    assert set(result.observation_table["start_age_source"]) == {"plate_metadata"}
    assert result.source_inventory[0].sha256
    assert len(result.source_inventory[0].sha256) == 64
    assert result.source_inventory[0].row_count == 5


def test_invalid_asset_with_null_path_is_preserved_then_counted_out(
    tmp_path: Path,
) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    snips = pd.read_csv(paths.snip_inventory)
    invalid_snip_id = str(snips.loc[0, "snip_id"])
    snips["error_message"] = snips["error_message"].astype(object)
    snips.loc[0, "is_valid_snip"] = False
    snips.loc[0, "processed_snip_path"] = pd.NA
    snips.loc[0, "error_message"] = "fixture materialization failure"
    snips.to_csv(paths.snip_inventory, index=False)

    result = build_pipeline_manifest(
        tmp_path,
        _policy((experiment_id,)),
        source_paths={experiment_id: paths},
    )

    invalid_asset = result.asset_table[
        result.asset_table["snip_id"].astype(str).eq(invalid_snip_id)
        & result.asset_table["snip_product_key"].eq(BF_PRODUCT)
    ]
    assert len(invalid_asset) == 1
    assert invalid_asset.iloc[0]["is_valid_snip"] == False  # noqa: E712
    assert pd.isna(invalid_asset.iloc[0]["processed_snip_path"])
    assert invalid_snip_id not in set(result.resolved_sample_table["snip_id"].astype(str))
    valid_filter = next(
        record
        for record in result.cohort_report.filters
        if record.filter_name == "valid_asset"
    )
    assert (valid_filter.rows_in, valid_filter.rows_out) == (4, 3)


def test_valid_asset_with_null_path_fails_by_asset_key(tmp_path: Path) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    snips = pd.read_csv(paths.snip_inventory)
    snips.loc[0, "processed_snip_path"] = pd.NA
    snips.to_csv(paths.snip_inventory, index=False)

    with pytest.raises(
        ValueError,
        match="valid assets require non-null processed_snip_path.*observation-opaque-0",
    ):
        build_pipeline_manifest(
            tmp_path,
            _policy((experiment_id,)),
            source_paths={experiment_id: paths},
        )


@pytest.mark.parametrize("bad_time_index", [-1, 1.5, "not-an-integer", float("inf")])
def test_time_index_must_be_nonnegative_integer(
    tmp_path: Path, bad_time_index: object
) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    snips = pd.read_csv(paths.snip_inventory)
    snips["time_index"] = snips["time_index"].astype(object)
    snips.loc[0, "time_index"] = bad_time_index
    snips.to_csv(paths.snip_inventory, index=False)

    with pytest.raises(
        ValueError,
        match="opaque-experiment:snip_inventory.time_index.*non-negative integers",
    ):
        build_pipeline_manifest(
            tmp_path,
            _policy((experiment_id,)),
            source_paths={experiment_id: paths},
        )


@pytest.mark.parametrize(
    ("column", "bad_value", "error_pattern"),
    [
        ("stage_prediction_status", None, "null stage_prediction_status"),
        ("stage_prediction_status", "unknown-status", "unknown stage_prediction_status"),
        ("predicted_stage_hpf", "not-a-stage", "nonnumeric or nonfinite"),
        ("predicted_stage_hpf", float("inf"), "nonnumeric or nonfinite"),
        ("predicted_stage_hpf", None, "status/value mismatch"),
        ("stage_prediction_status", "missing_temperature", "status/value mismatch"),
    ],
)
def test_stage_status_and_value_corruption_fails_with_row_identity(
    tmp_path: Path,
    column: str,
    bad_value: object,
    error_pattern: str,
) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    stage = pd.read_csv(paths.stage_predictions)
    stage[column] = stage[column].astype(object)
    stage.loc[0, column] = bad_value
    stage.to_csv(paths.stage_predictions, index=False)

    with pytest.raises(
        ValueError,
        match=rf"opaque-experiment.*{error_pattern}.*observation-opaque-0",
    ):
        build_pipeline_manifest(
            tmp_path,
            _policy((experiment_id,)),
            source_paths={experiment_id: paths},
        )


def test_conflicting_frame_times_fail_by_well_time_and_products(tmp_path: Path) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    frames = pd.read_csv(paths.frame_inventory)
    frames.loc[1, "elapsed_time_s"] = 123.0
    frames.to_csv(paths.frame_inventory, index=False)
    with pytest.raises(ValueError, match="well-opaque-0.*time_index=0.*products"):
        build_pipeline_manifest(
            tmp_path,
            _policy((experiment_id,)),
            source_paths={experiment_id: paths},
        )


def test_relative_snip_and_mask_paths_use_declared_path_resolver(tmp_path: Path) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    snips = pd.read_csv(paths.snip_inventory)
    snips["processed_snip_path"] = [f"pixels/asset-{i}.png" for i in range(len(snips))]
    snips["embryo_mask_snip_path"] = [f"masks/asset-{i}.png" for i in range(len(snips))]
    snips.to_csv(paths.snip_inventory, index=False)
    calls: list[tuple[str, Path]] = []

    def resolver(path_string: str, output_root: Path) -> Path:
        calls.append((path_string, output_root))
        return output_root / path_string

    result = build_pipeline_manifest(
        tmp_path,
        _policy((experiment_id,)),
        source_paths={experiment_id: paths},
        asset_path_resolver=resolver,
    )
    assert result.asset_table.loc[0, "processed_snip_path"] == str(
        tmp_path / "pixels/asset-0.png"
    )
    assert result.asset_table.loc[0, "embryo_mask_snip_path"] == str(
        tmp_path / "masks/asset-0.png"
    )
    assert ("pixels/asset-0.png", tmp_path) in calls


def test_collection_start_age_uses_declared_source_ordinal_not_time_guess(
    tmp_path: Path,
) -> None:
    experiment_id = "opaque-collection"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id, collection=True)
    result = build_pipeline_manifest(
        tmp_path,
        _policy((experiment_id,)),
        source_paths={experiment_id: paths},
    )
    assert result.observation_table["start_age_hpf"].tolist() == [30, 31, 32, 33]
    assert set(result.observation_table["start_age_source"]) == {"collection_provenance"}


def test_collection_without_source_ordinal_mapping_never_guesses_from_time(
    tmp_path: Path,
) -> None:
    experiment_id = "opaque-collection"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id, collection=True)
    paths = replace(paths, acquisition_inventory=None)
    result = build_pipeline_manifest(
        tmp_path,
        _policy((experiment_id,)),
        source_paths={experiment_id: paths},
    )
    assert result.observation_table["start_age_hpf"].isna().all()
    assert set(result.observation_table["start_age_source"]) == {"unavailable"}
    assert any(
        issue.code == "collection_source_ordinal_unavailable"
        for issue in result.schema_report.issues
    )


def test_qc_and_stage_keep_no_artifact_row_missing_and_unavailable_distinct(
    tmp_path: Path,
) -> None:
    first = "opaque-experiment-a"
    second = "opaque-experiment-b"
    paths_a = write_pipeline_source_fixture(tmp_path, first, row_count=3)
    stage = pd.read_csv(paths_a.stage_predictions).drop(
        columns=["stage_prediction_status", "model_version"]
    )
    stage = stage.iloc[:-1].copy()
    stage.to_csv(paths_a.stage_predictions, index=False)
    qc = pd.read_parquet(paths_a.snip_qc).iloc[:-1].copy()
    qc.to_parquet(paths_a.snip_qc, index=False)

    paths_b = write_pipeline_source_fixture(
        tmp_path,
        second,
        row_count=2,
        identity_offset=100,
        include_stage=False,
        include_qc=False,
    )
    policy = _policy(
        (first, second),
        qc=QCPolicy(
            name="fixture_absence_allowed",
            version="1",
            accepted_statuses=("evaluated", "row_missing", "no_artifact"),
            require_use_snip=None,
        ),
    )
    result = build_pipeline_manifest(
        tmp_path,
        policy,
        source_paths={first: paths_a, second: paths_b},
    )
    first_rows = result.observation_table[
        result.observation_table["experiment_id"].eq(first)
    ]
    second_rows = result.observation_table[
        result.observation_table["experiment_id"].eq(second)
    ]
    assert first_rows["stage_status"].tolist() == ["unavailable", "unavailable", "row_missing"]
    assert first_rows["qc_status"].tolist() == ["evaluated", "evaluated", "row_missing"]
    assert set(second_rows["stage_status"]) == {"no_artifact"}
    assert set(second_rows["qc_status"]) == {"no_artifact"}
    assert second_rows["use_snip"].isna().all()


def test_requested_missing_per_flag_policy_fails_by_experiment_and_flag(
    tmp_path: Path,
) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    qc = pd.read_parquet(paths.snip_qc).drop(columns=["sa_outlier_flag"])
    qc.to_parquet(paths.snip_qc, index=False)
    policy = _policy(
        (experiment_id,),
        qc=QCPolicy(
            name="fixture_per_flag",
            version="1",
            required_flags_false=("sa_outlier_flag",),
        ),
    )
    with pytest.raises(ValueError, match="opaque-experiment.*sa_outlier_flag"):
        build_pipeline_manifest(
            tmp_path,
            policy,
            source_paths={experiment_id: paths},
        )


def test_one_to_one_stage_and_many_to_one_plate_contracts_fail_loudly(
    tmp_path: Path,
) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    stage = pd.read_csv(paths.stage_predictions)
    pd.concat([stage, stage.iloc[[0]]], ignore_index=True).to_csv(
        paths.stage_predictions, index=False
    )
    with pytest.raises(ValueError, match="stage_predictions.*duplicate key"):
        build_pipeline_manifest(
            tmp_path,
            _policy((experiment_id,)),
            source_paths={experiment_id: paths},
        )

    paths = write_pipeline_source_fixture(tmp_path / "plate-case", experiment_id)
    plate = pd.read_csv(paths.plate_metadata)
    pd.concat([plate, plate.iloc[[0]]], ignore_index=True).to_csv(
        paths.plate_metadata, index=False
    )
    with pytest.raises(ValueError, match="plate_metadata.*duplicate key"):
        build_pipeline_manifest(
            tmp_path,
            _policy((experiment_id,)),
            source_paths={experiment_id: paths},
        )


def test_group_splits_are_disjoint_reorder_stable_and_growth_stable() -> None:
    train_experiment = "opaque-train-experiment"
    test_experiment = "opaque-test-experiment"
    rows = [
        {
            "physical_embryo_id": f"opaque-animal-{index}",
            "experiment_id": train_experiment,
        }
        for index in range(300)
    ] + [
        {"physical_embryo_id": f"held-out-animal-{index}", "experiment_id": test_experiment}
        for index in range(10)
    ]
    resolved = pd.DataFrame(rows)
    policy = _policy(
        (train_experiment, test_experiment),
        splits=SplitPolicy(
            train_fraction=0.7,
            eval_fraction=0.2,
            test_fraction=0.1,
            tolerance=0.08,
            explicit_test_experiments=(test_experiment,),
        ),
    )
    original = assign_group_splits(resolved, policy).set_index("physical_embryo_id")["split"]
    reordered = assign_group_splits(
        resolved.sample(frac=1.0, random_state=7), policy
    ).set_index("physical_embryo_id")["split"]
    pd.testing.assert_series_equal(original.sort_index(), reordered.sort_index())
    assert original.loc[[f"held-out-animal-{i}" for i in range(10)]].eq("test").all()

    growth = pd.concat(
        [
            resolved,
            pd.DataFrame(
                {
                    "physical_embryo_id": [f"new-opaque-animal-{i}" for i in range(50)],
                    "experiment_id": train_experiment,
                }
            ),
        ],
        ignore_index=True,
    )
    grown = assign_group_splits(growth, policy).set_index("physical_embryo_id")["split"]
    assert original.to_dict() == grown.loc[original.index].to_dict()
    assert set(original) == {"train", "eval", "test"}


def test_split_tolerance_and_empty_required_splits_are_enforced() -> None:
    resolved = pd.DataFrame(
        {"physical_embryo_id": ["one-animal"], "experiment_id": ["one-experiment"]}
    )
    empty_policy = _policy(
        ("one-experiment",),
        splits=SplitPolicy(tolerance=1.0),
    )
    with pytest.raises(ValueError, match="required split.*empty"):
        assign_group_splits(resolved, empty_policy)

    tolerance_policy = replace(
        empty_policy,
        splits=SplitPolicy(tolerance=0.0, required_splits=()),
    )
    with pytest.raises(ValueError, match="split ratios exceed tolerance"):
        assign_group_splits(resolved, tolerance_policy)


def test_inference_switches_disable_qc_stage_splits_and_metric_mapping(
    tmp_path: Path,
) -> None:
    experiment_id = "opaque-inference"
    paths = write_pipeline_source_fixture(
        tmp_path, experiment_id, include_stage=False, include_qc=False
    )
    policy = _policy(
        (experiment_id,),
        qc=QCPolicy(name="disabled", version="1", enabled=False),
        stage=StagePolicy(enabled=False),
        splits=SplitPolicy(enabled=False),
    )
    result = build_pipeline_manifest(
        tmp_path,
        policy,
        source_paths={experiment_id: paths},
    )
    assert len(result.resolved_sample_table) == 4
    assert result.resolved_sample_table["split"].isna().all()
    assert result.split_assignments.empty

    science_policy = replace(
        policy,
        metric_mapping=MetricMappingPolicy(
            enabled=True, name="scientific_mapping", scientific_policy=True
        ),
    )
    with pytest.raises(ValueError, match="test_only or dummy"):
        build_pipeline_manifest(
            tmp_path,
            science_policy,
            source_paths={experiment_id: paths},
        )


def test_deterministic_experiment_observation_and_product_order(tmp_path: Path) -> None:
    first = "opaque-first"
    second = "opaque-second"
    first_paths = write_pipeline_source_fixture(
        tmp_path, first, row_count=2, identity_offset=20
    )
    second_paths = write_pipeline_source_fixture(
        tmp_path, second, row_count=2, identity_offset=40
    )
    policy = replace(
        _policy((second, first)),
        allowed_product_keys=(RFP_PRODUCT, BF_PRODUCT),
    )
    result = build_pipeline_manifest(
        tmp_path,
        policy,
        source_paths={first: first_paths, second: second_paths},
    )
    assert result.observation_table["experiment_id"].tolist() == [second, second, first, first]
    first_observation_assets = result.asset_table[
        result.asset_table["snip_id"].eq("observation-opaque-40")
    ]
    assert first_observation_assets["snip_product_key"].tolist() == [RFP_PRODUCT, BF_PRODUCT]


def test_preflight_collects_schema_variant_and_build_error(tmp_path: Path) -> None:
    experiment_id = "opaque-preflight"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    snips = pd.read_csv(paths.snip_inventory).drop(columns=["snip_product_key"])
    snips.to_csv(paths.snip_inventory, index=False)
    report = preflight_pipeline_sources(
        tmp_path,
        _policy((experiment_id,)),
        source_paths={experiment_id: paths},
    )
    assert report.build_error is not None
    assert "snip_product_key" in report.build_error
    assert any(
        issue.code == "missing_required_columns" and "snip_product_key" in issue.columns
        for issue in report.schema_report.issues
    )
    assert report.experiment_summaries[0]["product_keys"] == ()


def test_pipeline_data_config_matches_split_aware_a2_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = synthetic_manifest_v2()
    captured: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(dataset_configs, "BasicDataset", FakeDataset)
    config = PipelineDataConfig(
        pipeline_output_root=Path("/declared-output"),
        manifest_policy=fixture.policy,
        input_dim=(1, 144, 64),
        batch_size=7,
        num_workers=2,
        loader_seed=13,
        drop_last_train=True,
        pin_memory=True,
        persistent_workers=True,
    )
    config.manifest_result = fixture
    dataset = config.create_dataset(split="eval")
    assert isinstance(dataset, FakeDataset)
    resolved_table = captured.pop("resolved_sample_table")
    assert resolved_table is fixture.resolved_sample_table
    assert captured == {
        "product_key": fixture.policy.selected_product_key,
        "input_dim": (1, 144, 64),
        "split": "eval",
        "transform": None,
        "max_decode_failures": 10,
    }
    assert (config.batch_size, config.num_workers, config.loader_seed) == (7, 2, 13)
    assert (config.drop_last_train, config.pin_memory, config.persistent_workers) == (
        True,
        True,
        True,
    )


def test_pipeline_data_config_passes_injected_path_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = synthetic_manifest_v2()
    captured: dict[str, object] = {}

    def artifact_path_fn(*args: object, **kwargs: object) -> Path:
        return Path("/declared-artifact")

    def asset_path_resolver(path_string: str, output_root: Path) -> Path:
        return output_root / path_string

    def fake_build(*args: object, **kwargs: object):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fixture

    monkeypatch.setattr(dataset_configs, "build_pipeline_manifest", fake_build)
    config = PipelineDataConfig(
        pipeline_output_root=Path("/declared-output"),
        manifest_policy=fixture.policy,
        artifact_path_fn=artifact_path_fn,
        asset_path_resolver=asset_path_resolver,
        artifact_path_authority="declared.config:artifact_path",
        asset_path_authority="declared.config:asset_path",
    )
    assert config.make_metadata() is fixture
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["artifact_path_fn"] is artifact_path_fn
    assert kwargs["asset_path_resolver"] is asset_path_resolver
    assert config.artifact_path_authority == "declared.config:artifact_path"
    assert config.asset_path_authority == "declared.config:asset_path"


def test_manifest_smoke_hydra_config_requires_explicit_authorities() -> None:
    path = Path("src/core/hydra_configs/data/pipeline_manifest_smoke.yaml")
    config = OmegaConf.load(path)
    assert OmegaConf.is_missing(config, "pipeline_output_root")
    assert OmegaConf.is_missing(config.manifest_policy, "experiment_ids")
    assert config.manifest_policy.name == "temporary_pipeline_manifest_smoke"
    assert config.manifest_policy.z_selection_mode == "projection"
    assert config.input_dim == [1, 288, 128]
