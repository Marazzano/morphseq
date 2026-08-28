from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib

import pandas as pd
import pytest

from src.core.data.manifest_types import (
    ManifestPolicy,
    QCPolicy,
    SplitPolicy,
    StagePolicy,
)
from src.core.data.pipeline_contracts import (
    SOURCE_SYMBOL_CITATIONS,
    TABLE_CONTRACTS,
    normalize_boolean_series,
)
from src.core.data.pipeline_manifest import (
    PipelinePackagingError,
    build_pipeline_manifest,
    resolve_pipeline_source_paths,
    select_vanilla_assets,
    validate_manifest_tables,
)
from tests.core.fixtures.manifest_v2 import BF_PRODUCT, synthetic_manifest_v2
from tests.core.fixtures.pipeline_source_tables import write_pipeline_source_fixture


def _policy(experiment_id: str) -> ManifestPolicy:
    return ManifestPolicy(
        name="test_only_contract",
        version="1",
        experiment_ids=(experiment_id,),
        allowed_product_keys=(BF_PRODUCT,),
        selected_product_key=BF_PRODUCT,
        qc=QCPolicy(name="fixture_qc", version="1"),
        stage=StagePolicy(enabled=False),
        splits=SplitPolicy(enabled=False),
    )


def test_current_writer_symbols_are_named_for_every_declared_source() -> None:
    assert set(SOURCE_SYMBOL_CITATIONS) == {
        "snip_inventory",
        "frame_inventory",
        "stage_predictions",
        "snip_qc",
        "plate_metadata",
        "collection_provenance",
        "acquisition_inventory",
        "path_authority",
        "snip_path_authority",
    }
    assert all(":" in citation for citations in SOURCE_SYMBOL_CITATIONS.values() for citation in citations)
    frame_contract = TABLE_CONTRACTS["frame_inventory"]
    assert set(frame_contract.current_writer_columns) == {
        "experiment_id",
        "well_index",
        "channel_id",
        "time_index",
        "z_index",
        "image_product_type",
        "projection_method",
        "acquisition_time_s",
        "elapsed_time_s",
        "image_path",
        "image_micrometers_per_pixel",
        "image_width_px",
        "image_height_px",
        "n_sources",
        "orientation",
        "image_file_format",
        "pixel_dtype",
        "downsample_factor",
        "downsample_method",
        "jpeg_quality",
        "flip_polarity",
    }
    assert set(frame_contract.adapter_required_columns) != set(
        frame_contract.current_writer_columns
    )


def test_observation_and_asset_keys_include_null_z() -> None:
    fixture = synthetic_manifest_v2()
    validate_manifest_tables(fixture.observation_table, fixture.asset_table)

    duplicate_observation = pd.concat(
        [fixture.observation_table, fixture.observation_table.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate key.*snip_id"):
        validate_manifest_tables(duplicate_observation, fixture.asset_table)

    duplicate_projection = pd.concat(
        [fixture.asset_table, fixture.asset_table.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate key.*z_index"):
        validate_manifest_tables(fixture.observation_table, duplicate_projection)


def test_two_products_and_ordered_z_planes_share_one_observation() -> None:
    fixture = synthetic_manifest_v2()
    alpha = fixture.asset_table[fixture.asset_table["snip_id"].eq("snip::alpha")]
    assert len(alpha) == 5
    z_rows = alpha[alpha["z_index"].notna()]
    assert z_rows["z_index"].tolist() == [0, 1, 2]
    assert fixture.observation_table["snip_id"].tolist().count("snip::alpha") == 1


def test_orphan_asset_fails_by_opaque_id() -> None:
    fixture = synthetic_manifest_v2()
    orphan = fixture.asset_table.iloc[[0]].copy()
    orphan["snip_id"] = "opaque-orphan"
    assets = pd.concat([fixture.asset_table, orphan], ignore_index=True)
    with pytest.raises(ValueError, match="opaque-orphan"):
        validate_manifest_tables(fixture.observation_table, assets)


def test_selected_product_zero_and_multiple_fail_by_snip_id() -> None:
    fixture = synthetic_manifest_v2()
    missing = fixture.asset_table[
        ~fixture.asset_table["snip_id"].eq("snip::beta")
    ].copy()
    with pytest.raises(ValueError, match="snip::beta.*0 matching"):
        select_vanilla_assets(fixture.observation_table, missing, fixture.policy)

    duplicate = pd.concat(
        [fixture.asset_table, fixture.asset_table.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="snip::alpha.*2 matching"):
        select_vanilla_assets(fixture.observation_table, duplicate, fixture.policy)


def test_safe_string_boolean_parsing_does_not_treat_false_as_truthy() -> None:
    parsed = normalize_boolean_series(
        pd.Series(["True", "False", True, False, 1, 0]),
        experiment_id="opaque-experiment",
        source_name="snip_qc",
        column="use_snip",
    )
    assert parsed.tolist() == [True, False, True, False, True, False]
    with pytest.raises(ValueError, match="unparseable boolean"):
        normalize_boolean_series(
            pd.Series(["false"]),
            experiment_id="opaque-experiment",
            source_name="snip_qc",
            column="use_snip",
        )


def test_path_resolution_calls_only_declared_pipeline_authority() -> None:
    calls: list[tuple[object, ...]] = []

    def artifact_path(root: Path, step: str, artifact: str, experiment_id: str, **kwargs: object) -> Path:
        calls.append((root, step, artifact, experiment_id, kwargs))
        return Path(root) / step / experiment_id / artifact

    paths = resolve_pipeline_source_paths(
        Path("/declared-output"), "opaque-experiment", artifact_path_fn=artifact_path
    )
    assert paths.frame_inventory == Path(
        "/declared-output/frame_inventory/opaque-experiment/inventory"
    )
    assert [call[1:3] for call in calls] == [
        ("snip_inventory", "snip_inventory"),
        ("frame_inventory", "inventory"),
        ("stage_predictions", "stage_predictions"),
        ("snip_qc", "verdict"),
        ("ingest_plate_metadata", "csv"),
        ("collection_provenance", "provenance"),
    ]


def test_failed_pipeline_import_reports_packaging_discrepancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fail_pipeline_import(name: str, package: str | None = None) -> object:
        if name == "src.data_pipeline.pipeline_orchestrator.orchestration.paths":
            raise ModuleNotFoundError("No module named 'data_pipeline'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_pipeline_import)
    with pytest.raises(PipelinePackagingError, match="Do not set PYTHONPATH"):
        resolve_pipeline_source_paths(Path("/declared-output"), "opaque-experiment")


def test_missing_parquet_engine_is_actionable_and_qc_is_not_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)

    def no_engine(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd, "read_parquet", no_engine)
    with pytest.raises(RuntimeError, match="install.*pyarrow.*QC was not skipped"):
        build_pipeline_manifest(
            tmp_path,
            _policy(experiment_id),
            source_paths={experiment_id: paths},
        )


def test_conflicting_parent_identity_fails_without_parsing_ids(tmp_path: Path) -> None:
    experiment_id = "opaque-experiment"
    paths = write_pipeline_source_fixture(tmp_path, experiment_id)
    snips = pd.read_csv(paths.snip_inventory)
    sibling = snips[snips["snip_id"].eq("observation-opaque-0")].index[-1]
    snips.loc[sibling, "physical_embryo_id"] = "different-opaque-animal"
    snips.to_csv(paths.snip_inventory, index=False)
    with pytest.raises(ValueError, match="observation-opaque-0.*physical_embryo_id"):
        build_pipeline_manifest(
            tmp_path,
            _policy(experiment_id),
            source_paths={experiment_id: paths},
        )


def test_non_projection_mode_is_rejected_by_vanilla_selector() -> None:
    fixture = synthetic_manifest_v2()
    policy = replace(fixture.policy, z_selection_mode="all_planes")
    with pytest.raises(ValueError, match="requires z_selection_mode='projection'"):
        select_vanilla_assets(fixture.observation_table, fixture.asset_table, policy)
