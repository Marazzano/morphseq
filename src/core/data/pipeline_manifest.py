"""Build deterministic core observation/asset tables from declared pipeline artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.core.data.manifest_types import (
    CohortFilterRecord,
    CohortReport,
    ManifestPolicy,
    PipelineManifestResult,
    SchemaIssue,
    SchemaReport,
    SourceArtifact,
)
from src.core.data.pipeline_contracts import (
    COLLECTION_PROVENANCE_REQUIRED_KEYS,
    QC_FLAG_SUFFIX,
    SNIP_IDENTITY_COLUMNS,
    SOURCE_SYMBOL_CITATIONS,
    TABLE_CONTRACTS,
    inspect_table_schema,
    normalize_boolean_series,
    qc_flag_and_applicability_columns,
    require_non_null,
    require_table_contract,
    require_unique,
)


SOURCE_ORDER: tuple[str, ...] = (
    "snip_inventory",
    "frame_inventory",
    "stage_predictions",
    "snip_qc",
    "plate_metadata",
    "collection_provenance",
    "acquisition_inventory",
)

STAGE_SOURCE_STATUSES: frozenset[str] = frozenset(
    {"predicted", "missing_start_age_hpf", "missing_temperature"}
)

OBSERVATION_PARENT_COLUMNS: tuple[str, ...] = SNIP_IDENTITY_COLUMNS

ASSET_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "processed_snip_path",
    "legacy_flat_snip_path",
    "is_valid_snip",
    "error_message",
    "source_image_product_key",
    "image_path",
    "embryo_mask",
    "embryo_mask_snip_path",
    "mask_id",
    "track_id",
    "crop_x_min_px",
    "crop_y_min_px",
    "crop_x_max_px",
    "crop_y_max_px",
    "crop_width_px",
    "crop_height_px",
    "crop_x_min_um",
    "crop_y_min_um",
    "crop_x_max_um",
    "crop_y_max_um",
    "source_micrometers_per_pixel",
    "snip_micrometers_per_pixel",
    "orientation_policy",
    "orientation_source",
    "no_yolk_policy",
    "rotation_angle_rad",
    "flip_x",
    "crop_center_um_x",
    "crop_center_um_y",
    "source_height_px",
    "source_width_px",
    "source_um_per_px",
    "target_um_per_px",
    "output_height_px",
    "output_width_px",
    "border_mode",
    "image_interpolation",
    "mask_interpolation",
    "realized_scale_y",
    "realized_scale_x",
    "centering",
    "snip_transform_id",
    "resolved_transform_chain_json",
    "output_grid_id",
    "pixel_dtype",
)


class PipelinePackagingError(RuntimeError):
    """The pipeline path authority cannot be imported from the installed package."""


@dataclass(frozen=True)
class ExperimentSourcePaths:
    """Explicit paths and authorities for one configured experiment.

    Runtime-declared drop-in paths are passed here explicitly.  The adapter never
    searches for a runtime config or substitutes an undeclared fallback.
    """

    experiment_id: str
    snip_inventory: Path
    frame_inventory: Path
    plate_metadata: Path
    stage_predictions: Path | None = None
    snip_qc: Path | None = None
    collection_provenance: Path | None = None
    acquisition_inventory: Path | None = None
    collection_provenance_applicable: bool | None = None
    authorities: Mapping[str, str] = field(default_factory=dict)

    def source_path(self, source_name: str) -> Path | None:
        return getattr(self, source_name)


@dataclass(frozen=True)
class SourcePreflightSummary:
    experiment_id: str
    source_name: str
    path: Path | None
    exists: bool
    row_count: int | None
    columns: tuple[str, ...]
    schema_version: str | None
    size_bytes: int | None
    mtime_ns: int | None
    sha256: str | None
    authority: str


@dataclass(frozen=True)
class PipelinePreflightReport:
    sources: tuple[SourcePreflightSummary, ...]
    schema_report: SchemaReport
    experiment_summaries: tuple[Mapping[str, Any], ...]
    build_error: str | None


ArtifactPathFunction = Callable[..., Path]
AssetPathResolver = Callable[[str, Path], Path]


def resolve_pipeline_source_paths(
    output_root: Path,
    experiment_id: str,
    *,
    artifact_path_fn: ArtifactPathFunction | None = None,
) -> ExperimentSourcePaths:
    """Resolve canonical merged artifacts through the pipeline path authority only."""

    if artifact_path_fn is None:
        try:
            module = importlib.import_module(
                "src.data_pipeline.pipeline_orchestrator.orchestration.paths"
            )
            artifact_path_fn = module.artifact_path
        except ModuleNotFoundError as exc:
            raise PipelinePackagingError(
                "Cannot import the pipeline artifact_path authority from a clean repository-root "
                "Python environment: src.data_pipeline.pipeline_orchestrator.orchestration imports "
                "the unavailable top-level 'data_pipeline' package. Do not set PYTHONPATH or modify "
                "sys.path; pass explicit ExperimentSourcePaths from a declared runtime/config "
                "authority until pipeline packaging is repaired."
            ) from exc

    root = Path(output_root)
    common = {"experiment_id": experiment_id}
    return ExperimentSourcePaths(
        experiment_id=experiment_id,
        snip_inventory=artifact_path_fn(
            root,
            "snip_inventory",
            "snip_inventory",
            path_mode="merged",
            **common,
        ),
        frame_inventory=artifact_path_fn(
            root,
            "frame_inventory",
            "inventory",
            path_mode="merged",
            **common,
        ),
        stage_predictions=artifact_path_fn(
            root,
            "stage_predictions",
            "stage_predictions",
            path_mode="merged",
            **common,
        ),
        snip_qc=artifact_path_fn(
            root,
            "snip_qc",
            "verdict",
            path_mode="merged",
            **common,
        ),
        plate_metadata=artifact_path_fn(
            root,
            "ingest_plate_metadata",
            "csv",
            **common,
        ),
        collection_provenance=artifact_path_fn(
            root,
            "collection_provenance",
            "provenance",
            **common,
        ),
        authorities={name: SOURCE_SYMBOL_CITATIONS["path_authority"][0] for name in SOURCE_ORDER},
    )


def build_pipeline_manifest(
    output_root: Path,
    policy: ManifestPolicy,
    *,
    source_paths: Mapping[str, ExperimentSourcePaths] | None = None,
    artifact_path_fn: ArtifactPathFunction | None = None,
    asset_path_resolver: AssetPathResolver | None = None,
) -> PipelineManifestResult:
    """Build and validate the manifest for an explicit ordered experiment list."""

    _validate_policy(policy)
    paths_by_experiment = _resolve_all_sources(
        Path(output_root), policy, source_paths=source_paths, artifact_path_fn=artifact_path_fn
    )

    observation_parts: list[pd.DataFrame] = []
    asset_parts: list[pd.DataFrame] = []
    source_inventory: list[SourceArtifact] = []
    schema_issues: list[SchemaIssue] = []

    for experiment_order, experiment_id in enumerate(policy.experiment_ids):
        paths = paths_by_experiment[experiment_id]
        tables, payloads, inventory = _read_experiment_sources(paths, policy=policy)
        source_inventory.extend(inventory)

        for source_name in ("snip_inventory", "frame_inventory", "plate_metadata"):
            schema_issues.extend(
                require_table_contract(
                    tables[source_name],
                    experiment_id=experiment_id,
                    contract=TABLE_CONTRACTS[source_name],
                )
            )
        for source_name in ("stage_predictions", "snip_qc", "acquisition_inventory"):
            if source_name in tables:
                schema_issues.extend(
                    require_table_contract(
                        tables[source_name],
                        experiment_id=experiment_id,
                        contract=TABLE_CONTRACTS[source_name],
                    )
                )
        for source_name in (
            "snip_inventory",
            "frame_inventory",
            "stage_predictions",
            "acquisition_inventory",
        ):
            if source_name not in tables:
                continue
            prepared = tables[source_name].copy()
            prepared["time_index"] = _nonnegative_integer(
                prepared["time_index"],
                label=f"{experiment_id}:{source_name}.time_index",
            )
            tables[source_name] = prepared

        snips = tables["snip_inventory"]
        frames = tables["frame_inventory"]
        plate = tables["plate_metadata"]
        if policy.qc.enabled and "snip_qc" in tables:
            for flag in policy.qc.required_flags_false:
                if flag not in tables["snip_qc"].columns:
                    raise ValueError(
                        f"policy {policy.qc.name!r}: experiment {experiment_id!r} snip_qc "
                        f"schema lacks requested flag {flag!r}; no fallback to use_snip is allowed."
                    )

        _validate_experiment_column(snips, experiment_id, "snip_inventory")
        _validate_experiment_column(frames, experiment_id, "frame_inventory")
        _validate_experiment_column(plate, experiment_id, "plate_metadata")

        assets, asset_issues = _build_asset_table(
            snips,
            experiment_id,
            policy,
            output_root=Path(output_root),
            asset_path_resolver=asset_path_resolver,
        )
        schema_issues.extend(asset_issues)
        observations = _collapse_observations(snips, experiment_id)
        observations, join_issues = _add_observation_metadata(
            observations,
            frames=frames,
            stage=tables.get("stage_predictions"),
            qc=tables.get("snip_qc"),
            plate=plate,
            collection_provenance=payloads.get("collection_provenance"),
            acquisition_inventory=tables.get("acquisition_inventory"),
            collection_provenance_applicable=paths.collection_provenance_applicable,
            experiment_id=experiment_id,
        )
        schema_issues.extend(join_issues)

        observations["_experiment_order"] = experiment_order
        observations["_observation_order"] = range(len(observations))
        assets["_experiment_order"] = experiment_order
        observation_parts.append(observations)
        asset_parts.append(assets)

    observations = pd.concat(observation_parts, ignore_index=True, sort=False)
    assets = pd.concat(asset_parts, ignore_index=True, sort=False)
    observations, assets = _order_manifest_tables(observations, assets, policy)
    validate_manifest_tables(observations, assets)

    selected_assets = select_vanilla_assets(observations, assets, policy)
    resolved = observations.merge(
        selected_assets,
        on="snip_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    resolved, filter_records = _apply_cohort_policy(resolved, policy)

    if policy.splits.enabled:
        split_assignments = assign_group_splits(resolved, policy)
        split_map = split_assignments.set_index("physical_embryo_id")["split"]
        resolved["split"] = resolved["physical_embryo_id"].map(split_map)
        observations["split"] = observations["physical_embryo_id"].map(split_map)
    else:
        split_assignments = pd.DataFrame(columns=["physical_embryo_id", "split"])
        resolved["split"] = pd.NA
        observations["split"] = pd.NA

    observations = observations.drop(columns=["_experiment_order", "_observation_order"])
    assets = assets.drop(columns=["_experiment_order", "_observation_order"])
    resolved = resolved.drop(columns=["_experiment_order", "_observation_order"])
    diagnostics = {
        "selected_observations": len(resolved),
        "observation_rows": len(observations),
        "asset_rows": len(assets),
        "source_authorities": {
            experiment_id: dict(paths_by_experiment[experiment_id].authorities)
            for experiment_id in policy.experiment_ids
        },
    }
    return PipelineManifestResult(
        observation_table=observations.reset_index(drop=True),
        asset_table=assets.reset_index(drop=True),
        resolved_sample_table=resolved.reset_index(drop=True),
        source_inventory=tuple(source_inventory),
        schema_report=SchemaReport(tuple(schema_issues)),
        cohort_report=CohortReport(tuple(filter_records), diagnostics),
        split_assignments=split_assignments.reset_index(drop=True),
        policy=policy,
    )


def validate_manifest_tables(observations: pd.DataFrame, assets: pd.DataFrame) -> None:
    """Validate the two public grains without parsing or reconstructing any ID."""

    required_observation = list(OBSERVATION_PARENT_COLUMNS)
    missing_observation = [c for c in required_observation if c not in observations.columns]
    if missing_observation:
        raise ValueError(f"observation_table: missing required columns {missing_observation}.")
    require_non_null(observations, required_observation, label="observation_table")
    _nonnegative_integer(
        observations["time_index"], label="observation_table.time_index"
    )
    require_unique(observations, ["snip_id"], label="observation_table")

    required_asset = [
        "snip_id",
        "snip_product_key",
        "z_index",
        "processed_snip_path",
        "is_valid_snip",
    ]
    missing_asset = [c for c in required_asset if c not in assets.columns]
    if missing_asset:
        raise ValueError(f"asset_table: missing required columns {missing_asset}.")
    require_non_null(
        assets,
        ["snip_id", "snip_product_key", "is_valid_snip"],
        label="asset_table",
    )
    valid_assets = normalize_boolean_series(
        assets["is_valid_snip"],
        experiment_id="manifest",
        source_name="asset_table",
        column="is_valid_snip",
    )
    missing_valid_paths = valid_assets & assets["processed_snip_path"].isna()
    if missing_valid_paths.any():
        examples = assets.loc[
            missing_valid_paths,
            ["snip_id", "snip_product_key", "z_index"],
        ].head(5).to_dict("records")
        raise ValueError(
            "asset_table: valid assets require non-null processed_snip_path; "
            f"offending asset keys={examples}."
        )
    require_unique(
        assets,
        ["snip_id", "snip_product_key", "z_index"],
        label="asset_table",
    )
    observation_ids = set(observations["snip_id"].astype(str))
    orphan_ids = sorted(set(assets["snip_id"].astype(str)) - observation_ids)
    if orphan_ids:
        raise ValueError(
            f"asset_table: orphan snip_id values have no observation row: {orphan_ids[:5]}."
        )


def select_vanilla_assets(
    observations: pd.DataFrame,
    assets: pd.DataFrame,
    policy: ManifestPolicy,
) -> pd.DataFrame:
    """Return exactly one configured projection asset for every observation."""

    if policy.z_selection_mode != "projection":
        raise ValueError(
            f"policy {policy.name!r}: vanilla selection requires z_selection_mode='projection', "
            f"got {policy.z_selection_mode!r}."
        )
    selected = assets.loc[
        assets["snip_product_key"].eq(policy.selected_product_key)
        & assets["z_index"].isna()
    ].copy()
    by_snip = {str(key): group for key, group in selected.groupby("snip_id", sort=False)}
    asset_columns = [
        c for c in assets.columns if c not in ("_experiment_order", "_observation_order")
    ]
    rows: list[pd.Series] = []
    for snip_id in observations["snip_id"].astype(str):
        matches = by_snip.get(snip_id)
        if matches is None or len(matches) != 1:
            available = assets.loc[
                assets["snip_id"].astype(str).eq(snip_id),
                ["snip_product_key", "z_index"],
            ].to_dict("records")
            count = 0 if matches is None else len(matches)
            raise ValueError(
                f"policy {policy.name!r}: snip_id {snip_id!r} has {count} matching asset(s) for "
                f"product {policy.selected_product_key!r} with null z_index; available={available}."
            )
        rows.append(matches.iloc[0][asset_columns])
    return pd.DataFrame(rows, columns=asset_columns).reset_index(drop=True)


def assign_group_splits(resolved: pd.DataFrame, policy: ManifestPolicy) -> pd.DataFrame:
    """Assign stable content-hash splits at physical-embryo grain."""

    split_policy = policy.splits
    fractions = (
        split_policy.train_fraction,
        split_policy.eval_fraction,
        split_policy.test_fraction,
    )
    if any(value < 0 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError(
            f"policy {policy.name!r}: split fractions must be non-negative and sum to 1; "
            f"got {fractions}."
        )
    if resolved.empty:
        raise ValueError(f"policy {policy.name!r}: no observations remain for split assignment.")

    group_experiments = (
        resolved.groupby("physical_embryo_id", sort=False)["experiment_id"]
        .agg(lambda values: tuple(dict.fromkeys(values.astype(str))))
    )
    explicit_test = set(split_policy.explicit_test_experiments)
    assignments: list[dict[str, str]] = []
    unpinned: list[str] = []
    for physical_embryo_id, experiments in group_experiments.items():
        group_id = str(physical_embryo_id)
        if explicit_test.intersection(experiments):
            split = "test"
        else:
            split = _hash_split(group_id, split_policy)
            unpinned.append(group_id)
        assignments.append({"physical_embryo_id": group_id, "split": split})

    result = pd.DataFrame(assignments, columns=["physical_embryo_id", "split"])
    observed_splits = set(result["split"])
    missing_required = [name for name in split_policy.required_splits if name not in observed_splits]
    if missing_required:
        raise ValueError(
            f"policy {policy.name!r}: required split(s) {missing_required} are empty after "
            f"assigning {len(result)} physical_embryo_id groups."
        )

    if unpinned:
        unpinned_rows = result[result["physical_embryo_id"].isin(unpinned)]
        targets = dict(zip(("train", "eval", "test"), fractions))
        actual = unpinned_rows["split"].value_counts(normalize=True).to_dict()
        outside = {
            name: (actual.get(name, 0.0), target)
            for name, target in targets.items()
            if abs(actual.get(name, 0.0) - target) > split_policy.tolerance
        }
        if outside:
            raise ValueError(
                f"policy {policy.name!r}: unpinned physical_embryo_id split ratios exceed "
                f"tolerance {split_policy.tolerance}; actual/target={outside}."
            )
    return result


def preflight_pipeline_sources(
    output_root: Path,
    policy: ManifestPolicy,
    *,
    source_paths: Mapping[str, ExperimentSourcePaths] | None = None,
    artifact_path_fn: ArtifactPathFunction | None = None,
    asset_path_resolver: AssetPathResolver | None = None,
) -> PipelinePreflightReport:
    """Read declared sources and collect schema/join summaries without changing artifacts."""

    paths_by_experiment = _resolve_all_sources(
        Path(output_root), policy, source_paths=source_paths, artifact_path_fn=artifact_path_fn
    )
    sources: list[SourcePreflightSummary] = []
    issues: list[SchemaIssue] = []
    experiment_summaries: list[Mapping[str, Any]] = []
    for experiment_id in policy.experiment_ids:
        paths = paths_by_experiment[experiment_id]
        tables: dict[str, pd.DataFrame] = {}
        for source_name in SOURCE_ORDER:
            path = paths.source_path(source_name)
            authority = paths.authorities.get(source_name, "unspecified")
            if path is None or not Path(path).is_file():
                sources.append(
                    SourcePreflightSummary(
                        experiment_id,
                        source_name,
                        path,
                        False,
                        None,
                        (),
                        None,
                        None,
                        None,
                        None,
                        authority,
                    )
                )
                if source_name in ("snip_inventory", "frame_inventory", "plate_metadata"):
                    issues.append(
                        SchemaIssue(
                            experiment_id,
                            source_name,
                            "warning",
                            "missing_artifact",
                            f"{experiment_id}: declared {source_name} artifact is missing at {path}.",
                        )
                    )
                continue
            if source_name == "collection_provenance":
                payload = _read_json(Path(path), experiment_id, source_name)
                columns = tuple(payload)
                row_count = None
                schema_version = None
            else:
                table = _read_table(Path(path), experiment_id, source_name)
                tables[source_name] = table
                columns = tuple(table.columns)
                row_count = len(table)
                schema_version = _schema_version(source_name, table)
                if source_name in TABLE_CONTRACTS:
                    issues.extend(
                        inspect_table_schema(
                            table,
                            experiment_id=experiment_id,
                            contract=TABLE_CONTRACTS[source_name],
                        )
                    )
            resolved_path = Path(path).resolve()
            stat = resolved_path.stat()
            sources.append(
                SourcePreflightSummary(
                    experiment_id,
                    source_name,
                    resolved_path,
                    True,
                    row_count,
                    columns,
                    schema_version,
                    stat.st_size,
                    stat.st_mtime_ns,
                    _sha256_file(resolved_path),
                    authority,
                )
            )

        snips = tables.get("snip_inventory", pd.DataFrame())
        frames = tables.get("frame_inventory", pd.DataFrame())
        stage = tables.get("stage_predictions", pd.DataFrame())
        qc = tables.get("snip_qc", pd.DataFrame())
        snip_ids = set(snips["snip_id"].astype(str)) if "snip_id" in snips else set()
        summary = {
            "experiment_id": experiment_id,
            "row_counts": {name: len(table) for name, table in tables.items()},
            "product_keys": (
                tuple(dict.fromkeys(snips["snip_product_key"].dropna().astype(str)))
                if "snip_product_key" in snips
                else ()
            ),
            "z_index_null": int(snips["z_index"].isna().sum()) if "z_index" in snips else None,
            "z_index_non_null": int(snips["z_index"].notna().sum()) if "z_index" in snips else None,
            "stage_join": _join_count_summary(snip_ids, stage),
            "qc_join": _join_count_summary(snip_ids, qc),
            "temperature_available": (
                int(tables["plate_metadata"]["temperature"].notna().sum())
                if "plate_metadata" in tables and "temperature" in tables["plate_metadata"]
                else None
            ),
            "elapsed_time_available": (
                int(frames["elapsed_time_s"].notna().sum())
                if "elapsed_time_s" in frames
                else None
            ),
            "policy_filter_counts": {
                "is_valid_snip": _value_counts(snips, "is_valid_snip"),
                "use_snip": _value_counts(qc, "use_snip"),
                "stage_status": _value_counts(stage, "stage_prediction_status"),
                "selected_product_rows": (
                    int(snips["snip_product_key"].astype(str).eq(policy.selected_product_key).sum())
                    if "snip_product_key" in snips
                    else None
                ),
            },
        }
        experiment_summaries.append(summary)

    try:
        build_pipeline_manifest(
            output_root,
            policy,
            source_paths=paths_by_experiment,
            artifact_path_fn=artifact_path_fn,
            asset_path_resolver=asset_path_resolver,
        )
    except Exception as exc:  # preflight records the first fatal build contract without hiding it.
        build_error = f"{type(exc).__name__}: {exc}"
    else:
        build_error = None
    return PipelinePreflightReport(
        sources=tuple(sources),
        schema_report=SchemaReport(tuple(issues)),
        experiment_summaries=tuple(experiment_summaries),
        build_error=build_error,
    )


def _resolve_all_sources(
    output_root: Path,
    policy: ManifestPolicy,
    *,
    source_paths: Mapping[str, ExperimentSourcePaths] | None,
    artifact_path_fn: ArtifactPathFunction | None,
) -> dict[str, ExperimentSourcePaths]:
    if source_paths is None:
        return {
            experiment_id: resolve_pipeline_source_paths(
                output_root, experiment_id, artifact_path_fn=artifact_path_fn
            )
            for experiment_id in policy.experiment_ids
        }
    missing = [experiment_id for experiment_id in policy.experiment_ids if experiment_id not in source_paths]
    extra = [experiment_id for experiment_id in source_paths if experiment_id not in policy.experiment_ids]
    if missing or extra:
        raise ValueError(
            f"policy {policy.name!r}: explicit source_paths must cover the ordered experiment_ids "
            f"exactly; missing={missing}, extra={extra}."
        )
    resolved: dict[str, ExperimentSourcePaths] = {}
    for experiment_id in policy.experiment_ids:
        paths = source_paths[experiment_id]
        if paths.experiment_id != experiment_id:
            raise ValueError(
                f"policy {policy.name!r}: source path key {experiment_id!r} contains "
                f"ExperimentSourcePaths for {paths.experiment_id!r}."
            )
        resolved[experiment_id] = paths
    return resolved


def _read_experiment_sources(
    paths: ExperimentSourcePaths,
    *,
    policy: ManifestPolicy,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], list[SourceArtifact]]:
    tables: dict[str, pd.DataFrame] = {}
    payloads: dict[str, dict[str, Any]] = {}
    inventory: list[SourceArtifact] = []
    required_sources = {"snip_inventory", "frame_inventory", "plate_metadata"}
    if (
        policy.stage.enabled
        and policy.stage.required
        and "no_artifact" not in policy.stage.accepted_statuses
    ):
        required_sources.add("stage_predictions")
    if policy.qc.enabled and "no_artifact" not in policy.qc.accepted_statuses:
        required_sources.add("snip_qc")
    if paths.collection_provenance_applicable is True:
        required_sources.add("collection_provenance")

    for source_name in SOURCE_ORDER:
        path = paths.source_path(source_name)
        if path is None or not Path(path).is_file():
            if source_name in required_sources:
                raise FileNotFoundError(
                    f"{paths.experiment_id}: required {source_name} artifact for policy "
                    f"{policy.name!r} is missing at declared path {path}."
                )
            continue
        resolved_path = Path(path).resolve()
        if source_name == "collection_provenance":
            payloads[source_name] = _read_json(resolved_path, paths.experiment_id, source_name)
            row_count = None
            schema_version = None
        else:
            table = _read_table(resolved_path, paths.experiment_id, source_name)
            tables[source_name] = table
            row_count = len(table)
            schema_version = _schema_version(source_name, table)
        stat = resolved_path.stat()
        inventory.append(
            SourceArtifact(
                experiment_id=paths.experiment_id,
                source_name=source_name,
                path=resolved_path,
                schema_version=schema_version,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                row_count=row_count,
                sha256=_sha256_file(resolved_path),
            )
        )
    return tables, payloads, inventory


def _read_table(path: Path, experiment_id: str, source_name: str) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                f"{experiment_id}: cannot read {source_name} Parquet artifact {path}; install "
                "a pandas Parquet engine such as pyarrow. QC was not skipped."
            ) from exc
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            f"{experiment_id}: failed to read declared {source_name} table {path}: {exc}"
        ) from exc


def _read_json(path: Path, experiment_id: str, source_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"{experiment_id}: failed to read declared {source_name} JSON {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{experiment_id}: {source_name} must contain one JSON object.")
    return payload


def _build_asset_table(
    snips: pd.DataFrame,
    experiment_id: str,
    policy: ManifestPolicy,
    *,
    output_root: Path,
    asset_path_resolver: AssetPathResolver | None,
) -> tuple[pd.DataFrame, tuple[SchemaIssue, ...]]:
    require_non_null(snips, SNIP_IDENTITY_COLUMNS, label=f"{experiment_id}:snip_inventory")
    require_non_null(
        snips,
        ["snip_product_key", "is_valid_snip"],
        label=f"{experiment_id}:snip_inventory",
    )
    table = snips.copy()
    table["is_valid_snip"] = normalize_boolean_series(
        table["is_valid_snip"],
        experiment_id=experiment_id,
        source_name="snip_inventory",
        column="is_valid_snip",
    )
    issues: list[SchemaIssue] = []
    if "z_index" not in table.columns:
        table["z_index"] = pd.array([pd.NA] * len(table), dtype="Int64")
        issues.append(
            SchemaIssue(
                experiment_id,
                "snip_inventory",
                "info",
                "z_index_not_emitted_by_current_snip_writer",
                f"{experiment_id}: snip_inventory has no z_index column; current rendered rows "
                "are represented as null-z assets. Synthetic z rows remain covered by contract tests.",
                ("z_index",),
            )
        )
    else:
        table["z_index"] = _nullable_nonnegative_integer(
            table["z_index"], label=f"{experiment_id}:snip_inventory.z_index"
        )
    for column in ("processed_snip_path", "embryo_mask", "embryo_mask_snip_path"):
        if column in table.columns:
            table[column] = table[column].map(
                lambda value: _resolve_snip_asset_path(
                    value,
                    output_root=output_root,
                    asset_path_resolver=asset_path_resolver,
                )
            )
    require_unique(
        table,
        ["snip_id", "snip_product_key", "z_index"],
        label=f"{experiment_id}:asset candidates",
    )
    configured = set(policy.allowed_product_keys)
    table = table[table["snip_product_key"].astype(str).isin(configured)].copy()
    columns = ["snip_id", "snip_product_key", "z_index"] + [
        column for column in ASSET_PAYLOAD_COLUMNS if column in table.columns
    ]
    return table[columns].reset_index(drop=True), tuple(issues)


def _collapse_observations(snips: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snip_id, group in snips.groupby("snip_id", sort=False, dropna=False):
        row: dict[str, Any] = {"snip_id": snip_id}
        for column in OBSERVATION_PARENT_COLUMNS:
            values = group[column].drop_duplicates()
            if len(values) != 1:
                raise ValueError(
                    f"{experiment_id}: snip_id {snip_id!r} has conflicting biological parent "
                    f"field {column!r}: {values.tolist()}."
                )
            row[column] = values.iloc[0]
        rows.append(row)
    observations = pd.DataFrame(rows, columns=OBSERVATION_PARENT_COLUMNS)
    require_unique(observations, ["snip_id"], label=f"{experiment_id}:observations")
    return observations


def _add_observation_metadata(
    observations: pd.DataFrame,
    *,
    frames: pd.DataFrame,
    stage: pd.DataFrame | None,
    qc: pd.DataFrame | None,
    plate: pd.DataFrame,
    collection_provenance: dict[str, Any] | None,
    acquisition_inventory: pd.DataFrame | None,
    collection_provenance_applicable: bool | None,
    experiment_id: str,
) -> tuple[pd.DataFrame, tuple[SchemaIssue, ...]]:
    issues: list[SchemaIssue] = []
    result = observations.copy()

    timing = _frame_timing(frames, experiment_id)
    result = result.merge(
        timing, on=["well_id", "time_index"], how="left", validate="many_to_one", sort=False
    )
    result["elapsed_time_status"] = "row_missing"
    has_frame = result["_frame_row_present"].fillna(False).astype(bool)
    result.loc[has_frame, "elapsed_time_status"] = "missing"
    result.loc[has_frame & result["elapsed_time_s"].notna(), "elapsed_time_status"] = "available"
    result = result.drop(columns=["_frame_row_present"])

    require_unique(plate, ["well_id"], label=f"{experiment_id}:plate_metadata")
    observed_wells = set(result["well_id"].astype(str))
    plate_wells = set(plate["well_id"].astype(str))
    missing_wells = sorted(observed_wells - plate_wells)
    if missing_wells:
        raise ValueError(
            f"{experiment_id}: plate_metadata has no row for observation well_id(s) {missing_wells[:5]}."
        )
    extra_wells = sorted(plate_wells - observed_wells)
    if extra_wells:
        issues.append(
            _join_issue(experiment_id, "plate_metadata", "extra_well_ids", extra_wells)
        )
    plate_columns = ["well_id", "temperature", "start_age_hpf", "genotype", "medium"] + [
        column for column in ("strain", "chem_perturbation") if column in plate.columns
    ]
    result = result.merge(
        plate[plate_columns], on="well_id", how="left", validate="many_to_one", sort=False
    )
    result = result.rename(columns={"temperature": "incubation_temperature_c"})
    numeric_temperature = pd.to_numeric(result["incubation_temperature_c"], errors="coerce")
    bad_temperature = result["incubation_temperature_c"].notna() & numeric_temperature.isna()
    if bad_temperature.any():
        offenders = result.loc[bad_temperature, "snip_id"].astype(str).tolist()[:5]
        raise ValueError(
            f"{experiment_id}: plate_metadata.temperature is non-numeric for snip_id(s) {offenders}."
        )
    result["incubation_temperature_c"] = numeric_temperature
    result["temperature_status"] = result["incubation_temperature_c"].map(
        lambda value: "available" if pd.notna(value) else "missing"
    )

    result, start_age_issues = _resolve_start_age(
        result,
        collection_provenance=collection_provenance,
        acquisition_inventory=acquisition_inventory,
        collection_provenance_applicable=collection_provenance_applicable,
        experiment_id=experiment_id,
    )
    issues.extend(start_age_issues)
    result, stage_issues = _join_stage(result, stage, experiment_id)
    issues.extend(stage_issues)
    result, qc_issues = _join_qc(result, qc, experiment_id)
    issues.extend(qc_issues)
    return result, tuple(issues)


def _frame_timing(frames: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    require_non_null(
        frames, ["well_id", "time_index"], label=f"{experiment_id}:frame_inventory"
    )
    work = frames.copy()
    work["elapsed_time_s"] = pd.to_numeric(work["elapsed_time_s"], errors="coerce")
    finite = work["elapsed_time_s"].dropna().map(math.isfinite)
    if not finite.all():
        raise ValueError(f"{experiment_id}: frame_inventory.elapsed_time_s has non-finite values.")
    if (work["elapsed_time_s"].dropna() < 0).any():
        raise ValueError(f"{experiment_id}: frame_inventory.elapsed_time_s has negative values.")

    rows: list[dict[str, Any]] = []
    for (well_id, time_index), group in work.groupby(
        ["well_id", "time_index"], sort=False, dropna=False
    ):
        values = group["elapsed_time_s"].dropna().unique()
        mixed_missing = group["elapsed_time_s"].isna().any() and group["elapsed_time_s"].notna().any()
        if len(values) > 1 or mixed_missing:
            product_columns = [
                c
                for c in (
                    "image_product_type",
                    "projection_method",
                    "channel_id",
                    "z_index",
                    "elapsed_time_s",
                )
                if c in group.columns
            ]
            raise ValueError(
                f"{experiment_id}: frame_inventory rows disagree on elapsed_time_s for "
                f"well_id={well_id!r}, time_index={time_index!r}; "
                f"products={group[product_columns].to_dict('records')}."
            )
        rows.append(
            {
                "well_id": well_id,
                "time_index": time_index,
                "elapsed_time_s": values[0] if len(values) else pd.NA,
                "_frame_row_present": True,
            }
        )
    return pd.DataFrame(
        rows, columns=["well_id", "time_index", "elapsed_time_s", "_frame_row_present"]
    )


def _resolve_start_age(
    observations: pd.DataFrame,
    *,
    collection_provenance: dict[str, Any] | None,
    acquisition_inventory: pd.DataFrame | None,
    collection_provenance_applicable: bool | None,
    experiment_id: str,
) -> tuple[pd.DataFrame, tuple[SchemaIssue, ...]]:
    result = observations.copy()
    issues: list[SchemaIssue] = []
    explicit_single = collection_provenance_applicable is False
    if collection_provenance is not None:
        _validate_collection_provenance(collection_provenance, experiment_id)
        is_collection = bool(collection_provenance["is_collection"])
    elif explicit_single:
        is_collection = False
    elif collection_provenance_applicable is True:
        raise FileNotFoundError(
            f"{experiment_id}: collection provenance is declared applicable but no artifact was supplied."
        )
    else:
        result["start_age_hpf"] = pd.NA
        result["start_age_source"] = "unavailable"
        issues.append(
            SchemaIssue(
                experiment_id,
                "collection_provenance",
                "warning",
                "collection_status_undeclared",
                f"{experiment_id}: no collection provenance or explicit single-acquisition "
                "authority was supplied; start_age_hpf was not guessed from plate metadata.",
            )
        )
        return result, tuple(issues)

    if not is_collection:
        result["start_age_hpf"] = pd.to_numeric(result["start_age_hpf"], errors="coerce")
        result["start_age_source"] = result["start_age_hpf"].map(
            lambda value: "plate_metadata" if pd.notna(value) else "unavailable"
        )
        return result, tuple(issues)

    if acquisition_inventory is None:
        result["start_age_hpf"] = pd.NA
        result["start_age_source"] = "unavailable"
        issues.append(
            SchemaIssue(
                experiment_id,
                "acquisition_inventory",
                "warning",
                "collection_source_ordinal_unavailable",
                f"{experiment_id}: collection provenance is present but no declared acquisition "
                "inventory maps (well_id,time_index) to source_ordinal; start age was not guessed "
                "from merged time_index.",
            )
        )
        return result, tuple(issues)

    require_table_contract(
        acquisition_inventory,
        experiment_id=experiment_id,
        contract=TABLE_CONTRACTS["acquisition_inventory"],
    )
    mapping = acquisition_inventory[
        ["well_id", "time_index", "source_ordinal"]
    ].drop_duplicates()
    require_unique(
        mapping,
        ["well_id", "time_index"],
        label=f"{experiment_id}:collection source ordinal",
    )
    result = result.merge(
        mapping, on=["well_id", "time_index"], how="left", validate="many_to_one", sort=False
    )
    age_map = collection_provenance["start_age_by_source_ordinal"]

    def _age(source_ordinal: object) -> object:
        if pd.isna(source_ordinal):
            return pd.NA
        return age_map.get(str(int(source_ordinal)), pd.NA)

    result["start_age_hpf"] = result["source_ordinal"].map(_age)
    result["start_age_source"] = result["start_age_hpf"].map(
        lambda value: "collection_provenance" if pd.notna(value) else "unavailable"
    )
    result = result.drop(columns=["source_ordinal"])
    return result, tuple(issues)


def _join_stage(
    observations: pd.DataFrame,
    stage: pd.DataFrame | None,
    experiment_id: str,
) -> tuple[pd.DataFrame, tuple[SchemaIssue, ...]]:
    result = observations.copy()
    if stage is None:
        result["predicted_stage_hpf"] = pd.NA
        result["stage_status"] = "no_artifact"
        result["stage_model_version"] = pd.NA
        return result, ()
    _validate_experiment_column(stage, experiment_id, "stage_predictions")
    require_unique(stage, ["snip_id"], label=f"{experiment_id}:stage_predictions")
    _validate_join_identity(result, stage, experiment_id, "stage_predictions")
    prepared = _validate_stage_values(stage, experiment_id)
    issues = list(_join_coverage_issues(result, prepared, experiment_id, "stage_predictions"))
    columns = ["snip_id", "predicted_stage_hpf"] + [
        c for c in ("stage_prediction_status", "model_version") if c in prepared.columns
    ]
    joined = result.merge(
        prepared[columns], on="snip_id", how="left", validate="one_to_one", sort=False
    )
    matched = joined["snip_id"].astype(str).isin(set(prepared["snip_id"].astype(str)))
    if "stage_prediction_status" in joined:
        joined["stage_status"] = joined["stage_prediction_status"].where(matched, "row_missing")
        joined = joined.drop(columns=["stage_prediction_status"])
    else:
        joined["stage_status"] = matched.map(lambda value: "unavailable" if value else "row_missing")
    if "model_version" in joined:
        joined = joined.rename(columns={"model_version": "stage_model_version"})
    else:
        joined["stage_model_version"] = pd.NA
    return joined, tuple(issues)


def _validate_stage_values(stage: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    """Enforce the live stage status/value contract without converting corruption to absence."""

    prepared = stage.copy()

    def examples(mask: pd.Series) -> list[dict[str, object]]:
        columns = ["snip_id", "predicted_stage_hpf"]
        if "stage_prediction_status" in prepared.columns:
            columns.insert(1, "stage_prediction_status")
        return prepared.loc[mask, columns].head(5).to_dict("records")

    raw_stage = prepared["predicted_stage_hpf"]
    numeric_stage = pd.to_numeric(raw_stage, errors="coerce")
    finite_or_null = numeric_stage.map(
        lambda value: pd.isna(value) or math.isfinite(float(value))
    )
    invalid_stage = raw_stage.notna() & (numeric_stage.isna() | ~finite_or_null)
    if invalid_stage.any():
        raise ValueError(
            f"{experiment_id}: stage_predictions has nonnumeric or nonfinite "
            f"predicted_stage_hpf; offending rows={examples(invalid_stage)}."
        )
    prepared["predicted_stage_hpf"] = numeric_stage

    if "stage_prediction_status" in prepared.columns:
        null_status = prepared["stage_prediction_status"].isna()
        if null_status.any():
            raise ValueError(
                f"{experiment_id}: stage_predictions has null stage_prediction_status; "
                f"offending rows={examples(null_status)}."
            )
        status = prepared["stage_prediction_status"].astype(str)
        unknown_status = ~status.isin(STAGE_SOURCE_STATUSES)
        if unknown_status.any():
            raise ValueError(
                f"{experiment_id}: stage_predictions has unknown stage_prediction_status; "
                f"allowed={sorted(STAGE_SOURCE_STATUSES)}; "
                f"offending rows={examples(unknown_status)}."
            )
        predicted = status.eq("predicted")
        incoherent = (predicted & numeric_stage.isna()) | (
            ~predicted & numeric_stage.notna()
        )
        if incoherent.any():
            raise ValueError(
                f"{experiment_id}: stage_predictions status/value mismatch; status='predicted' "
                "requires finite predicted_stage_hpf and unresolved statuses require null; "
                f"offending rows={examples(incoherent)}."
            )
    if "model_version" in prepared.columns and prepared["model_version"].isna().any():
        missing_version = prepared["model_version"].isna()
        raise ValueError(
            f"{experiment_id}: stage_predictions has null model_version; "
            f"offending rows={examples(missing_version)}."
        )
    return prepared


def _join_qc(
    observations: pd.DataFrame,
    qc: pd.DataFrame | None,
    experiment_id: str,
) -> tuple[pd.DataFrame, tuple[SchemaIssue, ...]]:
    result = observations.copy()
    if qc is None:
        result["use_snip"] = pd.array([pd.NA] * len(result), dtype="boolean")
        result["qc_fail_reasons"] = pd.NA
        result["qc_status"] = "no_artifact"
        result["qc_schema_version"] = pd.NA
        return result, ()
    _validate_experiment_column(qc, experiment_id, "snip_qc")
    require_unique(qc, ["snip_id"], label=f"{experiment_id}:snip_qc")
    _validate_join_identity(result, qc, experiment_id, "snip_qc")
    issues = list(_join_coverage_issues(result, qc, experiment_id, "snip_qc"))
    prepared = qc.copy()
    bool_columns = ["use_snip"] + [c for c in prepared.columns if c.endswith(QC_FLAG_SUFFIX)]
    for column in bool_columns:
        prepared[column] = normalize_boolean_series(
            prepared[column],
            experiment_id=experiment_id,
            source_name="snip_qc",
            column=column,
        )
    carry = ["snip_id", "use_snip", "qc_fail_reasons"] + list(
        qc_flag_and_applicability_columns(prepared)
    )
    carry = list(dict.fromkeys(carry))
    joined = result.merge(
        prepared[carry], on="snip_id", how="left", validate="one_to_one", sort=False
    )
    matched = joined["snip_id"].astype(str).isin(set(prepared["snip_id"].astype(str)))
    joined["qc_status"] = matched.map(lambda value: "evaluated" if value else "row_missing")
    if "qc_schema_version" not in joined:
        joined["qc_schema_version"] = _schema_version("snip_qc", prepared)
    return joined, tuple(issues)


def _apply_cohort_policy(
    resolved: pd.DataFrame,
    policy: ManifestPolicy,
) -> tuple[pd.DataFrame, list[CohortFilterRecord]]:
    result = resolved.copy()
    records: list[CohortFilterRecord] = []

    def apply(mask: pd.Series, name: str, reason: str) -> None:
        nonlocal result
        mask = mask.fillna(False).astype(bool)
        before = result
        for experiment_id in policy.experiment_ids:
            in_count = int(before["experiment_id"].astype(str).eq(experiment_id).sum())
            out_count = int(
                (before["experiment_id"].astype(str).eq(experiment_id) & mask).sum()
            )
            records.append(
                CohortFilterRecord(experiment_id, name, in_count, out_count, reason)
            )
        result = before.loc[mask].copy()

    if policy.require_valid_snip:
        apply(result["is_valid_snip"].eq(True), "valid_asset", "is_valid_snip is true")  # noqa: E712
    if policy.qc.enabled:
        apply(
            result["qc_status"].isin(policy.qc.accepted_statuses),
            "qc_status",
            f"qc_status in {policy.qc.accepted_statuses}",
        )
        if policy.qc.require_use_snip is not None:
            apply(
                result["use_snip"].eq(policy.qc.require_use_snip),
                "qc_verdict",
                f"use_snip is {policy.qc.require_use_snip}",
            )
        for flag in policy.qc.required_flags_false:
            if flag not in result.columns:
                experiments = tuple(dict.fromkeys(result["experiment_id"].astype(str)))
                raise ValueError(
                    f"policy {policy.qc.name!r}: requested QC flag {flag!r} is absent for "
                    f"experiment(s) {experiments}; no fallback to use_snip is allowed."
                )
            for experiment_id in policy.experiment_ids:
                subset = result[result["experiment_id"].astype(str).eq(experiment_id)]
                if len(subset) and subset[flag].isna().all():
                    raise ValueError(
                        f"policy {policy.qc.name!r}: requested QC flag {flag!r} is absent from "
                        f"experiment {experiment_id!r}; no fallback to use_snip is allowed."
                    )
            apply(result[flag].eq(False), f"qc_flag:{flag}", f"{flag} is false")  # noqa: E712
    if policy.stage.enabled and policy.stage.required:
        accepted = policy.stage.accepted_statuses
        status_mask = result["stage_status"].isin(accepted) if accepted else pd.Series(True, index=result.index)
        apply(
            result["predicted_stage_hpf"].notna() & status_mask,
            "stage_required",
            f"finite stage with status in {accepted or ('any',)}",
        )
    if policy.covariates.enabled:
        for column in policy.covariates.required_columns:
            if column not in result.columns:
                raise ValueError(
                    f"policy {policy.name!r}: required covariate column {column!r} is absent."
                )
            apply(result[column].notna(), f"covariate:{column}", f"{column} is available")
    return result.reset_index(drop=True), records


def _order_manifest_tables(
    observations: pd.DataFrame,
    assets: pd.DataFrame,
    policy: ManifestPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = observations.sort_values(
        ["_experiment_order", "_observation_order"], kind="stable"
    ).reset_index(drop=True)
    observation_order = {
        str(snip_id): index for index, snip_id in enumerate(observations["snip_id"])
    }
    assets["_observation_order"] = assets["snip_id"].astype(str).map(observation_order)
    product_order = {key: index for index, key in enumerate(policy.allowed_product_keys)}
    assets["_product_order"] = assets["snip_product_key"].astype(str).map(product_order)
    assets["_z_null_order"] = assets["z_index"].notna().astype(int)
    assets = assets.sort_values(
        ["_experiment_order", "_observation_order", "_product_order", "_z_null_order", "z_index"],
        kind="stable",
        na_position="first",
    ).drop(columns=["_product_order", "_z_null_order"])
    return observations.reset_index(drop=True), assets.reset_index(drop=True)


def _validate_policy(policy: ManifestPolicy) -> None:
    if not policy.experiment_ids:
        raise ValueError(f"policy {policy.name!r}: experiment_ids must be an explicit non-empty list.")
    if len(set(policy.experiment_ids)) != len(policy.experiment_ids):
        raise ValueError(f"policy {policy.name!r}: experiment_ids contains duplicates.")
    if not policy.allowed_product_keys:
        raise ValueError(f"policy {policy.name!r}: allowed_product_keys must be explicit and non-empty.")
    if policy.selected_product_key not in policy.allowed_product_keys:
        raise ValueError(
            f"policy {policy.name!r}: selected_product_key {policy.selected_product_key!r} is not "
            "in allowed_product_keys."
        )
    unknown_test = sorted(set(policy.splits.explicit_test_experiments) - set(policy.experiment_ids))
    if unknown_test:
        raise ValueError(
            f"policy {policy.name!r}: explicit test experiments are not configured: {unknown_test}."
        )
    if policy.metric_mapping.enabled:
        name = policy.metric_mapping.name or ""
        if "test_only" not in name and "dummy" not in name:
            raise ValueError(
                f"policy {policy.name!r}: Track A only permits metric mappings named test_only or "
                f"dummy, got {name!r}."
            )
        if policy.metric_mapping.scientific_policy:
            raise ValueError(
                f"policy {policy.name!r}: Track A metric mapping must set scientific_policy=False."
            )


def _validate_experiment_column(
    table: pd.DataFrame, experiment_id: str, source_name: str
) -> None:
    if "experiment_id" not in table.columns:
        return
    values = tuple(dict.fromkeys(table["experiment_id"].dropna().astype(str)))
    if values != (experiment_id,):
        raise ValueError(
            f"{experiment_id}: declared {source_name} rows carry experiment_id values {values}."
        )


def _validate_join_identity(
    observations: pd.DataFrame,
    source: pd.DataFrame,
    experiment_id: str,
    source_name: str,
) -> None:
    overlapping = [c for c in SNIP_IDENTITY_COLUMNS if c != "snip_id" and c in source.columns]
    by_observation = observations.set_index("snip_id")
    for _, row in source.iterrows():
        snip_id = row["snip_id"]
        if snip_id not in by_observation.index:
            continue
        expected = by_observation.loc[snip_id]
        for column in overlapping:
            if str(row[column]) != str(expected[column]):
                raise ValueError(
                    f"{experiment_id}: {source_name} snip_id {snip_id!r} has conflicting "
                    f"identity field {column!r}: source={row[column]!r}, "
                    f"observation={expected[column]!r}."
                )


def _join_coverage_issues(
    observations: pd.DataFrame,
    source: pd.DataFrame,
    experiment_id: str,
    source_name: str,
) -> tuple[SchemaIssue, ...]:
    observation_ids = set(observations["snip_id"].astype(str))
    source_ids = set(source["snip_id"].astype(str))
    issues: list[SchemaIssue] = []
    missing = sorted(observation_ids - source_ids)
    extra = sorted(source_ids - observation_ids)
    if missing:
        issues.append(_join_issue(experiment_id, source_name, "missing_snip_ids", missing))
    if extra:
        issues.append(_join_issue(experiment_id, source_name, "extra_snip_ids", extra))
    return tuple(issues)


def _join_issue(
    experiment_id: str, source_name: str, code: str, identifiers: list[str]
) -> SchemaIssue:
    return SchemaIssue(
        experiment_id,
        source_name,
        "warning",
        code,
        f"{experiment_id}: {source_name} {code} count={len(identifiers)}; "
        f"examples={identifiers[:5]}.",
    )


def _validate_collection_provenance(payload: dict[str, Any], experiment_id: str) -> None:
    missing = [key for key in COLLECTION_PROVENANCE_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"{experiment_id}: collection_provenance missing required key(s) {missing}."
        )
    if payload["experiment_id"] != experiment_id:
        raise ValueError(
            f"{experiment_id}: collection_provenance declares experiment_id "
            f"{payload['experiment_id']!r}."
        )
    if not isinstance(payload["is_collection"], bool):
        raise ValueError(f"{experiment_id}: collection_provenance.is_collection must be bool.")
    age_map = payload["start_age_by_source_ordinal"]
    if not isinstance(age_map, dict):
        raise ValueError(
            f"{experiment_id}: collection_provenance.start_age_by_source_ordinal must be a dict."
        )
    if age_map != payload["start_age_by_time_index"]:
        raise ValueError(
            f"{experiment_id}: collection provenance canonical and legacy age maps disagree."
        )


def _nullable_nonnegative_integer(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_token = values.notna() & numeric.isna()
    boolean_token = values.map(lambda value: isinstance(value, bool)) & values.notna()
    non_finite = numeric.notna() & ~numeric.map(
        lambda value: pd.isna(value) or math.isfinite(float(value))
    )
    non_integer = numeric.notna() & numeric.ne(numeric.round())
    negative = numeric.notna() & numeric.lt(0)
    invalid = invalid_token | boolean_token | non_finite | non_integer | negative
    if invalid.any():
        bad = values[invalid].tolist()[:5]
        raise ValueError(f"{label}: expected nullable non-negative integers; got {bad}.")
    return pd.Series(pd.array(numeric, dtype="Int64"), index=values.index)


def _nonnegative_integer(values: pd.Series, *, label: str) -> pd.Series:
    parsed = _nullable_nonnegative_integer(values, label=label)
    if parsed.isna().any():
        source_rows = values.index[parsed.isna()].tolist()[:5]
        raise ValueError(
            f"{label}: expected non-null non-negative integers; "
            f"null values at source rows {source_rows}."
        )
    return parsed


def _resolve_snip_asset_path(
    value: object,
    *,
    output_root: Path,
    asset_path_resolver: AssetPathResolver | None,
) -> object:
    if value is None or pd.isna(value):
        return value
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    resolver = asset_path_resolver
    if resolver is None:
        try:
            module = importlib.import_module(
                "src.data_pipeline.object_extraction.snip_processing.io"
            )

            def _pipeline_resolver(path_string: str, root: Path) -> Path:
                return module.resolve_from_root(path_string, output_root=root)

            resolver = _pipeline_resolver
        except ModuleNotFoundError as exc:
            raise PipelinePackagingError(
                "Cannot import the pipeline resolve_from_root snip path authority from a clean "
                "repository-root Python environment. Do not set PYTHONPATH or modify sys.path; "
                "pass an explicit asset_path_resolver until pipeline packaging is repaired."
            ) from exc
    return str(Path(resolver(str(value), output_root)))


def _hash_split(physical_embryo_id: str, split_policy: Any) -> str:
    payload = (split_policy.salt + "\0" + physical_embryo_id).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / (1 << 64)
    if value < split_policy.train_fraction:
        return "train"
    if value < split_policy.train_fraction + split_policy.eval_fraction:
        return "eval"
    return "test"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_version(source_name: str, table: pd.DataFrame) -> str | None:
    if source_name == "snip_qc":
        payload = "\0".join(map(str, table.columns)).encode("utf-8")
        return "columns_sha256:" + hashlib.sha256(payload).hexdigest()
    for column in ("schema_version", "qc_schema_version"):
        if column in table.columns:
            values = tuple(dict.fromkeys(table[column].dropna().astype(str)))
            if len(values) == 1:
                return values[0]
    return None


def _join_count_summary(snip_ids: set[str], source: pd.DataFrame) -> Mapping[str, int] | None:
    if "snip_id" not in source:
        return None
    source_ids = set(source["snip_id"].astype(str))
    return {
        "matched": len(snip_ids & source_ids),
        "missing": len(snip_ids - source_ids),
        "extra": len(source_ids - snip_ids),
    }


def _value_counts(table: pd.DataFrame, column: str) -> Mapping[str, int] | None:
    if column not in table:
        return None
    values = table[column].map(lambda value: "<null>" if pd.isna(value) else str(value))
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).items()}
