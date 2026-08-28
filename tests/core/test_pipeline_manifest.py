from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.core.data.manifest_types import (
    CovariatePolicy,
    ExperimentTables,
    MetricMappingPolicy,
    QCPolicy,
    SplitPolicy,
    StagePolicy,
    ValidityPolicy,
)
from src.core.data.pipeline_manifest import (
    build_manifest_from_tables,
    select_vanilla_assets,
)


BF_PRODUCT = "BF__projection__focus_stack__clahe_blend"
RFP_PRODUCT = "RFP__projection__max__no_change"
Z_PRODUCT = "BF__z_stack__no_change"


def test_builds_unique_observations_product_aware_assets_and_exact_vanilla_view(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    result = build_manifest_from_tables(manifest_policy_factory(), [bundle])

    assert result.observation_table["snip_id"].is_unique
    assert len(result.observation_table) == 3
    assert len(result.asset_table) == 7
    first_id = "observation::exp-A::0"
    sibling_assets = result.asset_table.loc[result.asset_table["snip_id"].eq(first_id)]
    assert sibling_assets["snip_product_key"].tolist() == [
        BF_PRODUCT,
        RFP_PRODUCT,
        Z_PRODUCT,
        Z_PRODUCT,
        Z_PRODUCT,
    ]
    assert sibling_assets["z_index"].tolist()[:2] == [pd.NA, pd.NA]
    assert sibling_assets.loc[sibling_assets["snip_product_key"].eq(Z_PRODUCT), "z_index"].tolist() == [
        0,
        1,
        2,
    ]
    assert result.selected_sample_view["snip_product_key"].eq(BF_PRODUCT).all()
    assert result.selected_sample_view["z_index"].isna().all()
    assert "use_snip" not in result.asset_table.columns
    assert "qc_status" not in result.asset_table.columns


def test_temperature_elapsed_time_stage_and_start_age_are_exact(
    manifest_bundle_factory, manifest_policy_factory
):
    result = build_manifest_from_tables(
        manifest_policy_factory(required_covariates=("incubation_temperature_c", "elapsed_time_s")),
        [manifest_bundle_factory()],
    )
    observations = result.observation_table
    assert observations["incubation_temperature_c"].tolist() == [27.0, 27.1, 27.2]
    assert observations["temperature_status"].tolist() == ["available"] * 3
    assert observations["temperature_source"].eq("plate_metadata.temperature").all()
    assert observations["elapsed_time_s"].tolist() == [7.0, 67.0, 127.0]
    assert observations["elapsed_time_status"].tolist() == ["available"] * 3
    assert observations["time_index"].tolist() == [0, 1, 2]
    assert observations["predicted_stage_hpf"].tolist() == [21.0, 22.0, 23.0]
    assert observations["stage_status"].eq("predicted").all()
    assert observations["stage_model_version"].eq("kimmel1995_temp_rate_v1").all()
    assert observations["plate_start_age_hpf"].tolist() == [20.0, 21.0, 22.0]
    assert observations["start_age_hpf"].tolist() == [20.0, 21.0, 22.0]
    assert observations["start_age_source"].eq("plate_metadata").all()


def test_collection_start_age_uses_source_ordinal_not_merged_time_index(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory(n_observations=3)
    observations = bundle.snip_inventory[["well_id", "time_index"]].drop_duplicates()
    acquisition = observations.copy()
    acquisition["experiment_id"] = "exp-A"
    acquisition["source_ordinal"] = [4, 4, 9]
    provenance = {
        "experiment_id": "exp-A",
        "is_collection": True,
        "sources": [],
        "start_age_by_source_ordinal": {"4": 31, "9": 55},
        "start_age_by_time_index": {"4": 31, "9": 55},
    }
    bundle = replace(
        bundle,
        acquisition_inventory=acquisition,
        collection_provenance=provenance,
    )
    result = build_manifest_from_tables(manifest_policy_factory(), [bundle])
    assert result.observation_table["source_ordinal"].tolist() == [4, 4, 9]
    assert result.observation_table["collection_start_age_hpf"].tolist() == [31.0, 31.0, 55.0]
    assert result.observation_table["start_age_hpf"].tolist() == [31.0, 31.0, 55.0]
    assert result.observation_table["start_age_source"].eq("collection_provenance").all()


def test_collection_without_source_ordinal_is_unavailable_not_guessed(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    bundle = replace(
        bundle,
        collection_provenance={
            "experiment_id": "exp-A",
            "is_collection": True,
            "sources": [],
            "start_age_by_source_ordinal": {"0": 42},
            "start_age_by_time_index": {"0": 42},
        },
        acquisition_inventory=None,
    )
    result = build_manifest_from_tables(manifest_policy_factory(), [bundle])
    assert result.observation_table["start_age_hpf"].isna().all()
    assert result.observation_table["start_age_source"].eq("unavailable").all()
    assert any(
        issue.code == "collection_source_ordinal_unavailable"
        for issue in result.validation_report.issues
    )


def test_conflicting_repeated_biological_parent_fails_by_snip_and_field(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    inventory = bundle.snip_inventory.copy()
    same_snip = inventory["snip_id"].eq("observation::exp-A::0")
    inventory.loc[inventory.index[same_snip][1], "physical_embryo_id"] = "conflict"
    with pytest.raises(ValueError, match="observation::exp-A::0.*physical_embryo_id"):
        build_manifest_from_tables(
            manifest_policy_factory(), [replace(bundle, snip_inventory=inventory)]
        )


def test_ids_stay_opaque_strings_and_time_index_is_integer(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    numeric_id = bundle.snip_inventory.copy()
    numeric_id.loc[0, "snip_id"] = 123
    with pytest.raises(ValueError, match="opaque ID/key column 'snip_id'"):
        build_manifest_from_tables(
            manifest_policy_factory(), [replace(bundle, snip_inventory=numeric_id)]
        )

    fractional_time = bundle.snip_inventory.copy()
    fractional_time["time_index"] = fractional_time["time_index"].astype(float)
    fractional_time.loc[0, "time_index"] = 0.5
    with pytest.raises(ValueError, match="time_index must be a non-negative integer"):
        build_manifest_from_tables(
            manifest_policy_factory(), [replace(bundle, snip_inventory=fractional_time)]
        )


def test_selected_product_zero_and_multiple_failures_name_available_assets(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    policy = manifest_policy_factory()
    missing = bundle.snip_inventory.copy()
    missing.loc[
        missing["snip_id"].eq("observation::exp-A::1"), "snip_product_key"
    ] = RFP_PRODUCT
    with pytest.raises(ValueError, match="observation::exp-A::1.*0 matching.*available assets"):
        build_manifest_from_tables(policy, [replace(bundle, snip_inventory=missing)])

    duplicate = pd.concat(
        [
            bundle.snip_inventory,
            bundle.snip_inventory.loc[
                bundle.snip_inventory["snip_id"].eq("observation::exp-A::1")
            ],
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate.*projection null"):
        build_manifest_from_tables(policy, [replace(bundle, snip_inventory=duplicate)])


def test_conflicting_frame_times_fail_with_product_plane_detail(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    frame = bundle.frame_inventory.copy()
    key = frame["well_id"].eq("well-exp-A-0")
    frame.loc[frame.index[key][1], "elapsed_time_s"] = 99.0
    with pytest.raises(ValueError, match="elapsed times conflict.*product/plane"):
        build_manifest_from_tables(
            manifest_policy_factory(), [replace(bundle, frame_inventory=frame)]
        )


def test_one_to_one_join_coverage_reports_missing_and_extra_ids(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    stage = bundle.stage_predictions.iloc[:-1].copy()
    extra = stage.iloc[[0]].copy()
    extra["snip_id"] = "stage-extra-observation"
    stage = pd.concat([stage, extra], ignore_index=True)
    qc = bundle.snip_qc.iloc[:-1].copy()
    result = build_manifest_from_tables(
        manifest_policy_factory(stage_enabled=False, qc_enabled=False),
        [replace(bundle, stage_predictions=stage, snip_qc=qc)],
    )
    stage_report = next(
        report for report in result.validation_report.joins if report.source_name == "stage_predictions"
    )
    qc_report = next(
        report for report in result.validation_report.joins if report.source_name == "snip_qc"
    )
    assert "observation::exp-A::2" in stage_report.missing_left_keys
    assert "stage-extra-observation" in stage_report.extra_source_keys
    assert qc_report.missing_left_keys == ("observation::exp-A::2",)
    assert result.observation_table.loc[2, "stage_status"] == "row_missing"
    assert result.observation_table.loc[2, "qc_status"] == "row_missing"


def test_many_to_one_plate_join_duplicate_fails(manifest_bundle_factory, manifest_policy_factory):
    bundle = manifest_bundle_factory()
    plate = pd.concat([bundle.plate_metadata, bundle.plate_metadata.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="many-to-one.*duplicate source rows"):
        build_manifest_from_tables(
            manifest_policy_factory(), [replace(bundle, plate_metadata=plate)]
        )


def test_qc_and_stage_three_state_absence_is_not_failure(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory(qc=False, stage=False)
    result = build_manifest_from_tables(
        manifest_policy_factory(qc_enabled=False, stage_enabled=False), [bundle]
    )
    observations = result.observation_table
    assert observations["qc_status"].eq("no_artifact").all()
    assert observations["use_snip"].isna().all()
    assert observations["stage_status"].eq("unavailable").all()
    assert observations["selected_by_policy"].all()


def test_stage_schema_without_status_keeps_value_and_marks_unavailable(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    legacy_stage = bundle.stage_predictions.drop(
        columns=["stage_prediction_status", "model_version"]
    )
    result = build_manifest_from_tables(
        manifest_policy_factory(stage_enabled=False),
        [replace(bundle, stage_predictions=legacy_stage)],
    )
    assert result.observation_table["predicted_stage_hpf"].notna().all()
    assert result.observation_table["stage_status"].eq("unavailable").all()
    schema = next(
        report
        for report in result.validation_report.schemas
        if report.source_name == "stage_predictions"
    )
    assert schema.status == "loaded"


def test_string_false_qc_flag_is_not_truthy(manifest_bundle_factory, manifest_policy_factory):
    bundle = manifest_bundle_factory()
    result = build_manifest_from_tables(
        manifest_policy_factory(exclude_flags=("sa_outlier_flag",)), [bundle]
    )
    assert result.observation_table["sa_outlier_flag"].eq(False).all()
    assert result.observation_table["selected_by_policy"].all()


def test_requested_missing_per_flag_fails_by_experiment_and_flag(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    qc = bundle.snip_qc.drop(columns="sa_outlier_flag")
    with pytest.raises(ValueError, match="exp-A.*sa_outlier_flag"):
        build_manifest_from_tables(
            manifest_policy_factory(exclude_flags=("sa_outlier_flag",)),
            [replace(bundle, snip_qc=qc)],
        )


def test_inference_switches_disable_qc_stage_split_and_metric(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory(qc=False, stage=False)
    policy = manifest_policy_factory(qc_enabled=False, stage_enabled=False)
    policy = replace(
        policy,
        validity=ValidityPolicy(False, "inference_validity_disabled"),
        splits=SplitPolicy(enabled=False, required_splits=()),
        metric_mapping=MetricMappingPolicy(False, "inference_metric_disabled", "v1"),
    )
    result = build_manifest_from_tables(policy, [bundle])
    assert result.observation_table["selected_by_policy"].all()
    assert result.observation_table["split"].isna().all()
    assert "metric_group" not in result.observation_table
    assert result.split_assignments.empty


def test_metric_test_stub_switch_is_separate(manifest_bundle_factory, manifest_policy_factory):
    result = build_manifest_from_tables(
        manifest_policy_factory(metric_enabled=True), [manifest_bundle_factory()]
    )
    assert result.observation_table["metric_group"].eq("test_group").all()
    assert not result.policy.metric_mapping.scientific_policy


def test_validity_filter_counts_invalid_selected_asset(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory()
    inventory = bundle.snip_inventory.copy()
    target = inventory["snip_id"].eq("observation::exp-A::2") & inventory[
        "snip_product_key"
    ].eq(BF_PRODUCT)
    inventory.loc[target, "is_valid_snip"] = False
    result = build_manifest_from_tables(
        manifest_policy_factory(), [replace(bundle, snip_inventory=inventory)]
    )
    assert result.observation_table["selected_by_policy"].tolist() == [True, True, False]
    validity_count = next(
        count
        for count in result.cohort_report.counts
        if count.filter_name.startswith("validity:")
    )
    assert validity_count.rows_removed == 1


def test_explicit_experiment_order_and_output_are_deterministic(
    manifest_bundle_factory, manifest_policy_factory
):
    first = manifest_bundle_factory("exp-A", n_observations=2)
    second = manifest_bundle_factory("exp-B", n_observations=2)
    policy = manifest_policy_factory(experiment_ids=("exp-B", "exp-A"))
    one = build_manifest_from_tables(policy, [first, second])
    two = build_manifest_from_tables(policy, [second, first])
    expected = (
        "observation::exp-B::0",
        "observation::exp-B::1",
        "observation::exp-A::0",
        "observation::exp-A::1",
    )
    assert one.observation_order == expected
    assert two.observation_order == expected
    assert one.asset_order == two.asset_order


def test_missing_snip_z_column_reports_current_writer_gap_and_normalizes_null(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory(include_siblings=False)
    inventory = bundle.snip_inventory.drop(columns="z_index")
    result = build_manifest_from_tables(
        manifest_policy_factory(), [replace(bundle, snip_inventory=inventory)]
    )
    assert result.asset_table["z_index"].isna().all()
    assert str(result.asset_table["z_index"].dtype) == "Int64"
    assert any(
        issue.code == "current_z_snip_writer_gap"
        for issue in result.validation_report.issues
    )


def test_relative_asset_path_requires_pipeline_path_authority(
    manifest_bundle_factory, manifest_policy_factory
):
    bundle = manifest_bundle_factory(include_siblings=False)
    inventory = bundle.snip_inventory.copy()
    inventory.loc[0, "processed_snip_path"] = "relative/asset.png"
    with pytest.raises(ValueError, match="requires the pipeline path resolver"):
        build_manifest_from_tables(
            manifest_policy_factory(), [replace(bundle, snip_inventory=inventory)]
        )
    result = build_manifest_from_tables(
        manifest_policy_factory(),
        [replace(bundle, snip_inventory=inventory)],
        path_resolver=lambda value, root: root / value,
    )
    assert result.asset_table.loc[0, "processed_snip_path"] == (
        "/pipeline/output/relative/asset.png"
    )
    assert result.asset_table.loc[0, "processed_snip_path_source"] == "relative/asset.png"


def test_selector_can_be_called_again_on_frozen_tables(
    manifest_bundle_factory, manifest_policy_factory
):
    policy = manifest_policy_factory()
    result = build_manifest_from_tables(policy, [manifest_bundle_factory()])
    selected = select_vanilla_assets(result.observation_table, result.asset_table, policy)
    assert selected["snip_id"].tolist() == list(result.observation_order)
