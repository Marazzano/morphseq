"""Observation/asset manifest adapter for explicit pipeline experiments.

The filesystem entrypoint is :func:`build_pipeline_manifest`.  The pure
:func:`build_manifest_from_tables` seam implements all identity, join, selection, split, and
reporting behavior and is also the executable contract shared with dataset work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.data.manifest_types import (
    AssetKey,
    CohortReport,
    ExperimentTables,
    FilterCount,
    JoinReport,
    ManifestPolicy,
    ManifestResult,
    PathResolver,
    SchemaIssue,
    SourceArtifactRecord,
    SourceSchemaReport,
    ValidationReport,
)
from src.core.data.pipeline_contracts import (
    ADAPTER_REQUIRED_COLUMNS,
    SourceContract,
    current_writer_contracts,
    fingerprint_artifact,
    read_artifact_table,
    read_collection_provenance,
    resolve_experiment_paths,
    resolve_pipeline_path,
)


_OBSERVATION_IDENTITY_COLUMNS = (
    "snip_id",
    "embryo_id",
    "physical_embryo_id",
    "experiment_id",
    "well_id",
    "image_id",
    "time_index",
    "channel_id",
)
_ASSET_PARENT_COLUMNS = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
)
_QC_SPINE_COLUMNS = {
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
}


def parse_boolean(value: Any, *, column: str, identity: str) -> bool | pd._libs.missing.NAType:
    """Parse booleans without the ``bool('False')`` trap."""

    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(
        f"{identity}: column {column!r} has unrecognized boolean value {value!r}; "
        "accepted values are bool, 0/1, and strings true/false/0/1"
    )


def _normalize_boolean_column(
    frame: pd.DataFrame, column: str, *, experiment_id: str, identity_column: str
) -> None:
    if column not in frame.columns:
        return
    parsed = []
    for row_index, value in frame[column].items():
        identity = (
            str(frame.at[row_index, identity_column])
            if identity_column in frame.columns
            else f"row {row_index}"
        )
        parsed.append(
            parse_boolean(
                value,
                column=column,
                identity=f"experiment {experiment_id!r}, {identity_column} {identity!r}",
            )
        )
    frame[column] = pd.array(parsed, dtype="boolean")


def _normalize_z_index(
    frame: pd.DataFrame, *, source_name: str, experiment_id: str
) -> bool:
    """Return whether the source lacked z_index and was normalized to projection-null."""

    if "z_index" not in frame.columns:
        frame["z_index"] = pd.array([pd.NA] * len(frame), dtype="Int64")
        return True
    numeric = pd.to_numeric(frame["z_index"], errors="coerce")
    invalid_token = frame["z_index"].notna() & numeric.isna()
    fractional = numeric.notna() & numeric.mod(1).ne(0)
    negative = numeric.notna() & numeric.lt(0)
    bad = invalid_token | fractional | negative
    if bad.any():
        examples = frame.loc[bad, [c for c in ("snip_id", "z_index") if c in frame]].head(5)
        raise ValueError(
            f"experiment {experiment_id!r} {source_name}: z_index must be a nullable "
            f"non-negative integer; examples: {examples.to_dict('records')}"
        )
    frame["z_index"] = pd.array(numeric, dtype="Int64")
    return False


def _missing_values_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    source_name: str,
    experiment_id: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"experiment {experiment_id!r} {source_name}: missing required identity/key "
            f"column(s) {missing}; columns present={sorted(frame.columns)}"
        )


def _require_opaque_strings(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    source_name: str,
    experiment_id: str,
) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        invalid = [
            value
            for value in frame[column].tolist()
            if not isinstance(value, str) or not value
        ]
        if invalid:
            raise ValueError(
                f"experiment {experiment_id!r} {source_name}: opaque ID/key column "
                f"{column!r} must contain non-empty strings; examples={invalid[:5]}"
            )


def _normalize_time_index(
    frame: pd.DataFrame, *, source_name: str, experiment_id: str
) -> None:
    if "time_index" not in frame.columns:
        return
    numeric = pd.to_numeric(frame["time_index"], errors="coerce")
    invalid = numeric.isna() | numeric.mod(1).ne(0) | numeric.lt(0)
    if invalid.any():
        columns = [column for column in ("snip_id", "well_id", "time_index") if column in frame]
        examples = frame.loc[invalid, columns].head(5).to_dict("records")
        raise ValueError(
            f"experiment {experiment_id!r} {source_name}: time_index must be a non-negative "
            f"integer; examples={examples}"
        )
    frame["time_index"] = numeric.astype("int64")


def _require_one_experiment(
    frame: pd.DataFrame, *, source_name: str, experiment_id: str
) -> None:
    if "experiment_id" not in frame.columns or frame.empty:
        return
    present = tuple(dict.fromkeys(frame["experiment_id"].dropna().astype(str)))
    if present != (experiment_id,):
        raise ValueError(
            f"explicit experiment {experiment_id!r} resolved {source_name} rows for "
            f"experiment_id values {present}; sources are never silently filtered"
        )


def _schema_report(
    frame: pd.DataFrame | None,
    *,
    source_name: str,
    experiment_id: str,
    path: Path | None,
    contract: SourceContract | None,
    status: str | None = None,
) -> SourceSchemaReport:
    if frame is None:
        return SourceSchemaReport(
            experiment_id=experiment_id,
            source_name=source_name,
            path=path,
            status=status or "absent",
            columns=(),
            dtypes=(),
            writer_symbol=contract.writer_symbol if contract else None,
        )
    columns = tuple(str(column) for column in frame.columns)
    writer_columns = set(contract.current_writer_columns) if contract else set()
    return SourceSchemaReport(
        experiment_id=experiment_id,
        source_name=source_name,
        path=path,
        status=status or "loaded",
        columns=columns,
        dtypes=tuple((str(column), str(frame[column].dtype)) for column in frame.columns),
        missing_adapter_required=tuple(
            column
            for column in (contract.adapter_required if contract else ())
            if column not in frame.columns
        ),
        missing_current_writer=tuple(
            column
            for column in (contract.current_writer_columns if contract else ())
            if column not in frame.columns
        ),
        unexpected_columns=tuple(column for column in columns if writer_columns and column not in writer_columns),
        writer_symbol=contract.writer_symbol if contract else None,
    )


def _default_contracts() -> dict[str, SourceContract]:
    """Contract shape for the pure seam when pipeline symbols are deliberately not imported."""

    return {
        name: SourceContract(
            source_name=name,
            adapter_required=required,
            current_writer_columns=required,
            writer_symbol="MANIFEST_SCHEMA.md v2 adapter-required subset",
        )
        for name, required in ADAPTER_REQUIRED_COLUMNS.items()
    }


def _collapse_observations(
    inventory: pd.DataFrame, *, experiment_id: str
) -> pd.DataFrame:
    _require_columns(
        inventory,
        _OBSERVATION_IDENTITY_COLUMNS,
        source_name="snip_inventory",
        experiment_id=experiment_id,
    )
    for column in _OBSERVATION_IDENTITY_COLUMNS:
        if inventory[column].isna().any():
            examples = inventory.loc[inventory[column].isna(), "snip_id"].head(5).tolist()
            raise ValueError(
                f"experiment {experiment_id!r} snip_inventory: required observation column "
                f"{column!r} is null for snip_id(s) {examples}"
            )

    rows: list[dict[str, Any]] = []
    for snip_id, group in inventory.groupby("snip_id", sort=False, dropna=False):
        first = group.iloc[0]
        for column in _OBSERVATION_IDENTITY_COLUMNS[1:]:
            conflicts = [
                value
                for value in group[column].tolist()[1:]
                if not _missing_values_equal(first[column], value)
            ]
            if conflicts:
                values = group[column].tolist()
                raise ValueError(
                    f"experiment {experiment_id!r}: snip_id {snip_id!r} has conflicting "
                    f"biological parent field {column!r}: {values}"
                )
        row = {column: first[column] for column in _OBSERVATION_IDENTITY_COLUMNS}
        for optional_parent in ("mask_id", "track_id"):
            if optional_parent in group.columns:
                first_value = first[optional_parent]
                if any(
                    not _missing_values_equal(first_value, value)
                    for value in group[optional_parent].tolist()[1:]
                ):
                    raise ValueError(
                        f"experiment {experiment_id!r}: snip_id {snip_id!r} has conflicting "
                        f"parent field {optional_parent!r}"
                    )
                row[optional_parent] = first_value
        row["_source_observation_order"] = int(group["_source_asset_order"].min())
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_asset_path_columns(
    assets: pd.DataFrame,
    *,
    output_root: Path,
    experiment_id: str,
    path_resolver: PathResolver | None,
) -> None:
    for column in ("processed_snip_path", "embryo_mask_snip_path", "embryo_mask", "image_path"):
        if column not in assets.columns:
            continue
        raw_column = f"{column}_source"
        assets[raw_column] = assets[column]
        resolved: list[Any] = []
        for row_index, value in assets[column].items():
            if value is None or pd.isna(value):
                resolved.append(pd.NA)
                continue
            path = Path(str(value))
            if path.is_absolute():
                resolved.append(str(path))
            elif path_resolver is None:
                key = {
                    "snip_id": assets.at[row_index, "snip_id"],
                    "snip_product_key": assets.at[row_index, "snip_product_key"],
                }
                raise ValueError(
                    f"experiment {experiment_id!r}: relative {column} {value!r} for asset {key} "
                    "requires the pipeline path resolver"
                )
            else:
                resolved.append(str(path_resolver(str(value), output_root)))
        assets[column] = resolved


def _asset_key_tuple(row: pd.Series) -> tuple[str, str, int | None]:
    z_value = None if pd.isna(row["z_index"]) else int(row["z_index"])
    return str(row["snip_id"]), str(row["snip_product_key"]), z_value


def validate_manifest_tables(observations: pd.DataFrame, assets: pd.DataFrame) -> None:
    """Validate the frozen v2 table relationship without parsing any identifier."""

    _require_columns(
        observations,
        ("snip_id", "physical_embryo_id", "embryo_id", "well_id", "experiment_id"),
        source_name="observation_table",
        experiment_id="<combined>",
    )
    _require_columns(
        assets,
        ("snip_id", "snip_product_key", "z_index"),
        source_name="asset_table",
        experiment_id="<combined>",
    )
    duplicate_observations = observations["snip_id"].duplicated(keep=False)
    if duplicate_observations.any():
        ids = observations.loc[duplicate_observations, "snip_id"].astype(str).unique().tolist()
        raise ValueError(f"observation_table has duplicate snip_id value(s): {ids[:5]}")

    seen: dict[tuple[str, str, int | None], int] = {}
    duplicates: list[tuple[str, str, int | None]] = []
    for row_index, row in assets.iterrows():
        key = _asset_key_tuple(row)
        if key in seen:
            duplicates.append(key)
        else:
            seen[key] = int(row_index)
    if duplicates:
        raise ValueError(
            "asset_table has duplicate (snip_id, snip_product_key, z_index) key(s), including "
            f"projection null as a real key member: {duplicates[:5]}"
        )

    observation_ids = set(observations["snip_id"].astype(str))
    orphan_ids = sorted(set(assets["snip_id"].astype(str)) - observation_ids)
    if orphan_ids:
        raise ValueError(f"asset_table has orphan snip_id value(s): {orphan_ids[:5]}")

    by_observation = observations.set_index(observations["snip_id"].astype(str))
    for _, asset in assets.iterrows():
        snip_id = str(asset["snip_id"])
        observation = by_observation.loc[snip_id]
        for column in _ASSET_PARENT_COLUMNS:
            if column not in assets.columns or column not in observations.columns:
                continue
            if not _missing_values_equal(asset[column], observation[column]):
                raise ValueError(
                    f"asset { _asset_key_tuple(asset)!r} conflicts with observation snip_id "
                    f"{snip_id!r} on parent field {column!r}: asset={asset[column]!r}, "
                    f"observation={observation[column]!r}"
                )


def _join_frame_time(
    observations: pd.DataFrame,
    frame_inventory: pd.DataFrame,
    *,
    experiment_id: str,
    joins: list[JoinReport],
) -> None:
    _require_columns(
        frame_inventory,
        ("well_id", "time_index", "elapsed_time_s"),
        source_name="frame_inventory",
        experiment_id=experiment_id,
    )
    frame = frame_inventory.copy()
    frame["time_index"] = pd.to_numeric(frame["time_index"], errors="raise").astype(int)
    raw_elapsed = frame["elapsed_time_s"].copy()
    frame["elapsed_time_s"] = pd.to_numeric(raw_elapsed, errors="coerce")
    unparseable = raw_elapsed.notna() & frame["elapsed_time_s"].isna()
    nonfinite = frame["elapsed_time_s"].notna() & ~np.isfinite(frame["elapsed_time_s"])
    negative = frame["elapsed_time_s"].notna() & frame["elapsed_time_s"].lt(0)
    if (unparseable | nonfinite | negative).any():
        columns = [
            column
            for column in (
                "well_id",
                "time_index",
                "channel_id",
                "image_product_type",
                "projection_method",
                "z_index",
                "elapsed_time_s",
            )
            if column in frame.columns
        ]
        examples = frame.loc[unparseable | nonfinite | negative, columns].head(10).to_dict("records")
        raise ValueError(
            f"experiment {experiment_id!r} frame_inventory has invalid elapsed_time_s; "
            f"offending product/plane rows: {examples}"
        )

    time_by_key: dict[tuple[str, int], float | None] = {}
    for key, group in frame.groupby(["well_id", "time_index"], sort=False, dropna=False):
        values = tuple(dict.fromkeys(float(v) for v in group["elapsed_time_s"].dropna()))
        if len(values) > 1:
            detail_columns = [
                column
                for column in (
                    "well_id",
                    "time_index",
                    "channel_id",
                    "image_product_type",
                    "projection_method",
                    "z_index",
                    "elapsed_time_s",
                )
                if column in group.columns
            ]
            raise ValueError(
                f"experiment {experiment_id!r} frame elapsed times conflict for "
                f"(well_id, time_index)={key!r} across product/plane rows: "
                f"{group[detail_columns].to_dict('records')}"
            )
        time_by_key[(str(key[0]), int(key[1]))] = values[0] if values else None

    observation_keys = {
        (str(row.well_id), int(row.time_index))
        for row in observations[["well_id", "time_index"]].itertuples(index=False)
    }
    frame_keys = set(time_by_key)
    joins.append(
        JoinReport(
            experiment_id,
            "frame_inventory",
            "observation(well_id,time_index)",
            "frame(well_id,time_index,product,plane)",
            tuple(f"{well_id}|{time_index}" for well_id, time_index in sorted(observation_keys - frame_keys)),
            tuple(f"{well_id}|{time_index}" for well_id, time_index in sorted(frame_keys - observation_keys)),
        )
    )
    elapsed_values: list[float] = []
    elapsed_status: list[str] = []
    for row in observations.itertuples(index=False):
        key = (str(row.well_id), int(row.time_index))
        if key not in time_by_key:
            elapsed_values.append(np.nan)
            elapsed_status.append("row_missing")
        elif time_by_key[key] is None:
            elapsed_values.append(np.nan)
            elapsed_status.append("missing")
        else:
            elapsed_values.append(float(time_by_key[key]))
            elapsed_status.append("available")
    observations["elapsed_time_s"] = elapsed_values
    observations["elapsed_time_status"] = elapsed_status


def _join_plate_metadata(
    observations: pd.DataFrame,
    plate_metadata: pd.DataFrame,
    *,
    experiment_id: str,
    joins: list[JoinReport],
) -> None:
    _require_columns(
        plate_metadata,
        ("well_id", "temperature", "start_age_hpf", "genotype", "medium"),
        source_name="plate_metadata",
        experiment_id=experiment_id,
    )
    duplicate_wells = plate_metadata["well_id"].duplicated(keep=False)
    if duplicate_wells.any():
        examples = plate_metadata.loc[duplicate_wells, ["well_id"]].head(10).to_dict("records")
        raise ValueError(
            f"experiment {experiment_id!r} plate_metadata must be many-to-one on well_id; "
            f"duplicate source rows: {examples}"
        )
    plate_by_well = plate_metadata.set_index(plate_metadata["well_id"].astype(str))
    observation_wells = set(observations["well_id"].astype(str))
    plate_wells = set(plate_by_well.index)
    joins.append(
        JoinReport(
            experiment_id,
            "plate_metadata",
            "observation(well_id)",
            "plate_metadata(well_id)",
            tuple(sorted(observation_wells - plate_wells)),
            tuple(sorted(plate_wells - observation_wells)),
        )
    )

    carried_columns = [
        column
        for column in ("genotype", "medium", "strain", "chem_perturbation")
        if column in plate_metadata.columns
    ]
    for column in carried_columns:
        observations[column] = [
            plate_by_well.at[str(well_id), column]
            if str(well_id) in plate_by_well.index
            else pd.NA
            for well_id in observations["well_id"]
        ]

    temperatures: list[float] = []
    temperature_statuses: list[str] = []
    plate_ages: list[float] = []
    for well_id in observations["well_id"].astype(str):
        if well_id not in plate_by_well.index:
            temperatures.append(np.nan)
            temperature_statuses.append("row_missing")
            plate_ages.append(np.nan)
            continue
        plate = plate_by_well.loc[well_id]
        temperature = pd.to_numeric(pd.Series([plate["temperature"]]), errors="coerce").iloc[0]
        if pd.isna(plate["temperature"]):
            temperatures.append(np.nan)
            temperature_statuses.append("missing")
        elif pd.isna(temperature):
            raise ValueError(
                f"experiment {experiment_id!r} well_id {well_id!r}: plate temperature is "
                f"not numeric: {plate['temperature']!r}"
            )
        elif not math.isfinite(float(temperature)):
            raise ValueError(
                f"experiment {experiment_id!r} well_id {well_id!r}: plate temperature is "
                f"non-finite: {plate['temperature']!r}"
            )
        else:
            temperatures.append(float(temperature))
            temperature_statuses.append("available")
        raw_age = plate["start_age_hpf"]
        age = pd.to_numeric(pd.Series([raw_age]), errors="coerce").iloc[0]
        if not pd.isna(raw_age) and pd.isna(age):
            raise ValueError(
                f"experiment {experiment_id!r} well_id {well_id!r}: plate start_age_hpf "
                f"is not numeric: {raw_age!r}"
            )
        plate_ages.append(np.nan if pd.isna(age) else float(age))
    observations["incubation_temperature_c"] = temperatures
    observations["temperature_status"] = temperature_statuses
    observations["temperature_source"] = "plate_metadata.temperature"
    observations["plate_start_age_hpf"] = plate_ages


def _join_stage(
    observations: pd.DataFrame,
    stage: pd.DataFrame | None,
    *,
    experiment_id: str,
    joins: list[JoinReport],
) -> None:
    if stage is None:
        observations["predicted_stage_hpf"] = np.nan
        observations["stage_status"] = "unavailable"
        observations["stage_model_version"] = pd.NA
        joins.append(
            JoinReport(
                experiment_id,
                "stage_predictions",
                "observation(snip_id)",
                "absent",
                tuple(observations["snip_id"].astype(str)),
                (),
            )
        )
        return
    _require_columns(
        stage,
        ("snip_id", "predicted_stage_hpf"),
        source_name="stage_predictions",
        experiment_id=experiment_id,
    )
    duplicate = stage["snip_id"].duplicated(keep=False)
    if duplicate.any():
        ids = stage.loc[duplicate, "snip_id"].astype(str).unique().tolist()
        raise ValueError(
            f"experiment {experiment_id!r} stage_predictions is not one-to-one on snip_id: {ids[:5]}"
        )
    stage_by_id = stage.set_index(stage["snip_id"].astype(str))
    observation_ids = set(observations["snip_id"].astype(str))
    stage_ids = set(stage_by_id.index)
    joins.append(
        JoinReport(
            experiment_id,
            "stage_predictions",
            "observation(snip_id)",
            "stage_predictions(snip_id)",
            tuple(sorted(observation_ids - stage_ids)),
            tuple(sorted(stage_ids - observation_ids)),
        )
    )
    values: list[float] = []
    statuses: list[str] = []
    versions: list[Any] = []
    status_column = (
        "stage_prediction_status" if "stage_prediction_status" in stage.columns else None
    )
    version_column = "model_version" if "model_version" in stage.columns else None
    for snip_id in observations["snip_id"].astype(str):
        if snip_id not in stage_by_id.index:
            values.append(np.nan)
            statuses.append("row_missing")
            versions.append(pd.NA)
            continue
        row = stage_by_id.loc[snip_id]
        raw_value = row["predicted_stage_hpf"]
        value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
        if not pd.isna(raw_value) and pd.isna(value):
            raise ValueError(
                f"experiment {experiment_id!r} stage_predictions: snip_id {snip_id!r} has "
                f"non-numeric predicted_stage_hpf {raw_value!r}"
            )
        values.append(np.nan if pd.isna(value) else float(value))
        if status_column and pd.isna(row[status_column]):
            raise ValueError(
                f"experiment {experiment_id!r} stage_predictions: snip_id {snip_id!r} has "
                "null stage_prediction_status"
            )
        statuses.append(str(row[status_column]) if status_column else "unavailable")
        versions.append(row[version_column] if version_column else pd.NA)
    observations["predicted_stage_hpf"] = values
    observations["stage_status"] = statuses
    observations["stage_model_version"] = versions


def _join_qc(
    observations: pd.DataFrame,
    qc: pd.DataFrame | None,
    *,
    experiment_id: str,
    joins: list[JoinReport],
    schema_version: str | None,
) -> tuple[str, ...]:
    if qc is None:
        observations["use_snip"] = pd.array([pd.NA] * len(observations), dtype="boolean")
        observations["qc_fail_reasons"] = pd.NA
        observations["qc_status"] = "no_artifact"
        observations["qc_schema_version"] = "no_artifact"
        joins.append(
            JoinReport(
                experiment_id,
                "snip_qc",
                "observation(snip_id)",
                "absent",
                tuple(observations["snip_id"].astype(str)),
                (),
            )
        )
        return ()
    _require_columns(
        qc,
        ("snip_id", "use_snip", "qc_fail_reasons"),
        source_name="snip_qc",
        experiment_id=experiment_id,
    )
    duplicate = qc["snip_id"].duplicated(keep=False)
    if duplicate.any():
        ids = qc.loc[duplicate, "snip_id"].astype(str).unique().tolist()
        raise ValueError(
            f"experiment {experiment_id!r} snip_qc is not one-to-one on snip_id: {ids[:5]}"
        )
    payload_columns = [column for column in qc.columns if column not in _QC_SPINE_COLUMNS]
    flag_columns = tuple(column for column in payload_columns if str(column).endswith("_flag"))
    normalized = qc.copy()
    _normalize_boolean_column(
        normalized, "use_snip", experiment_id=experiment_id, identity_column="snip_id"
    )
    if normalized["use_snip"].isna().any():
        ids = normalized.loc[normalized["use_snip"].isna(), "snip_id"].head(5).tolist()
        raise ValueError(
            f"experiment {experiment_id!r} snip_qc: use_snip is null for snip_id(s) {ids}"
        )
    for flag in flag_columns:
        _normalize_boolean_column(
            normalized, flag, experiment_id=experiment_id, identity_column="snip_id"
        )
        if normalized[flag].isna().any():
            ids = normalized.loc[normalized[flag].isna(), "snip_id"].head(5).tolist()
            raise ValueError(
                f"experiment {experiment_id!r} snip_qc: flag {flag!r} is null for "
                f"snip_id(s) {ids}"
            )
    qc_by_id = normalized.set_index(normalized["snip_id"].astype(str))
    observation_ids = set(observations["snip_id"].astype(str))
    qc_ids = set(qc_by_id.index)
    joins.append(
        JoinReport(
            experiment_id,
            "snip_qc",
            "observation(snip_id)",
            "snip_qc(snip_id)",
            tuple(sorted(observation_ids - qc_ids)),
            tuple(sorted(qc_ids - observation_ids)),
        )
    )
    for column in payload_columns:
        values = []
        for snip_id in observations["snip_id"].astype(str):
            values.append(qc_by_id.at[snip_id, column] if snip_id in qc_by_id.index else pd.NA)
        if column == "use_snip" or column in flag_columns:
            observations[column] = pd.array(values, dtype="boolean")
        else:
            observations[column] = values
    observations["qc_status"] = [
        "evaluated" if snip_id in qc_by_id.index else "row_missing"
        for snip_id in observations["snip_id"].astype(str)
    ]
    observations["qc_schema_version"] = (
        schema_version
        or (
            str(normalized["qc_schema_version"].iloc[0])
            if "qc_schema_version" in normalized.columns and len(normalized)
            else "unversioned"
        )
    )
    return flag_columns


def _join_start_age(
    observations: pd.DataFrame,
    provenance: Mapping[str, Any] | None,
    acquisition: pd.DataFrame | None,
    *,
    experiment_id: str,
    joins: list[JoinReport],
    issues: list[SchemaIssue],
) -> None:
    observations["collection_start_age_hpf"] = np.nan
    observations["source_ordinal"] = pd.array([pd.NA] * len(observations), dtype="Int64")
    if provenance is None:
        observations["start_age_hpf"] = np.nan
        observations["start_age_source"] = "unavailable"
        issues.append(
            SchemaIssue(
                "warning",
                experiment_id,
                "collection_provenance",
                "collection_provenance_absent",
                "start age was left unavailable because collection status cannot be guessed",
            )
        )
        return
    missing_keys = [
        key
        for key in ADAPTER_REQUIRED_COLUMNS["collection_provenance"]
        if key not in provenance
    ]
    if missing_keys:
        raise ValueError(
            f"experiment {experiment_id!r}: collection provenance is missing required key(s) "
            f"{missing_keys}"
        )
    if str(provenance.get("experiment_id")) != experiment_id:
        raise ValueError(
            f"experiment {experiment_id!r}: collection provenance declares experiment_id "
            f"{provenance.get('experiment_id')!r}"
        )
    is_collection = provenance.get("is_collection")
    if not isinstance(is_collection, bool):
        raise ValueError(
            f"experiment {experiment_id!r}: collection provenance is_collection must be bool"
        )
    if not is_collection:
        sources = provenance["sources"]
        if (
            not isinstance(sources, Sequence)
            or isinstance(sources, (str, bytes))
            or len(sources) != 1
        ):
            raise ValueError(
                f"experiment {experiment_id!r}: non-collection provenance must declare exactly "
                "one source record"
            )
        declared_ordinal = (
            sources[0].get("source_ordinal")
            if isinstance(sources[0], Mapping)
            else None
        )
        if not isinstance(declared_ordinal, int) or isinstance(declared_ordinal, bool):
            raise ValueError(
                f"experiment {experiment_id!r}: non-collection source record lacks integer "
                "source_ordinal"
            )
        observations["start_age_hpf"] = observations["plate_start_age_hpf"]
        observations["start_age_source"] = np.where(
            observations["plate_start_age_hpf"].notna(), "plate_metadata", "unavailable"
        )
        observations["source_ordinal"] = pd.array(
            [declared_ordinal] * len(observations), dtype="Int64"
        )
        return

    if acquisition is None:
        observations["start_age_hpf"] = np.nan
        observations["start_age_source"] = "unavailable"
        issues.append(
            SchemaIssue(
                "warning",
                experiment_id,
                "acquisition_inventory",
                "collection_source_ordinal_unavailable",
                "collection start age requires explicit acquisition inventory source_ordinal; "
                "merged time_index was not used as a substitute",
            )
        )
        return
    _require_columns(
        acquisition,
        ("well_id", "time_index", "source_ordinal"),
        source_name="acquisition_inventory",
        experiment_id=experiment_id,
    )
    mapping_frame = acquisition[["well_id", "time_index", "source_ordinal"]].drop_duplicates()
    conflicts = (
        mapping_frame.groupby(["well_id", "time_index"], dropna=False)["source_ordinal"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicts.any():
        keys = conflicts[conflicts].index.tolist()[:5]
        raise ValueError(
            f"experiment {experiment_id!r}: acquisition inventory maps (well_id,time_index) "
            f"to multiple source_ordinal values: {keys}"
        )
    source_by_frame = {
        (str(row.well_id), int(row.time_index)): int(row.source_ordinal)
        for row in mapping_frame.itertuples(index=False)
    }
    observation_keys = {
        (str(row.well_id), int(row.time_index))
        for row in observations[["well_id", "time_index"]].itertuples(index=False)
    }
    source_keys = set(source_by_frame)
    joins.append(
        JoinReport(
            experiment_id,
            "acquisition_inventory",
            "observation(well_id,time_index)",
            "acquisition(well_id,time_index)",
            tuple(f"{well}|{time}" for well, time in sorted(observation_keys - source_keys)),
            tuple(f"{well}|{time}" for well, time in sorted(source_keys - observation_keys)),
        )
    )
    age_map = provenance.get("start_age_by_source_ordinal")
    if not isinstance(age_map, Mapping):
        raise ValueError(
            f"experiment {experiment_id!r}: collection provenance lacks canonical "
            "start_age_by_source_ordinal mapping"
        )
    resolved: list[float] = []
    raw_collection: list[float] = []
    sources: list[Any] = []
    statuses: list[str] = []
    for row in observations.itertuples(index=False):
        frame_key = (str(row.well_id), int(row.time_index))
        if frame_key not in source_by_frame:
            sources.append(pd.NA)
            raw_collection.append(np.nan)
            resolved.append(np.nan)
            statuses.append("unavailable")
            continue
        ordinal = source_by_frame[frame_key]
        sources.append(ordinal)
        raw_age = age_map.get(str(ordinal))
        if raw_age is None:
            raw_collection.append(np.nan)
            resolved.append(np.nan)
            statuses.append("unavailable")
        else:
            raw_collection.append(float(raw_age))
            resolved.append(float(raw_age))
            statuses.append("collection_provenance")
    observations["source_ordinal"] = pd.array(sources, dtype="Int64")
    observations["collection_start_age_hpf"] = raw_collection
    observations["start_age_hpf"] = resolved
    observations["start_age_source"] = statuses


def select_vanilla_assets(
    observations: pd.DataFrame, assets: pd.DataFrame, policy: ManifestPolicy
) -> pd.DataFrame:
    """Resolve exactly one configured product/null-z asset for each supplied observation."""

    rows: list[pd.Series] = []
    for snip_id in observations["snip_id"].astype(str):
        available = assets.loc[assets["snip_id"].astype(str).eq(snip_id)]
        matches = available.loc[
            available["snip_product_key"].astype(str).eq(policy.assets.vanilla_product_key)
            & available["z_index"].isna()
        ]
        if len(matches) != 1:
            available_keys = [_asset_key_tuple(row) for _, row in available.iterrows()]
            raise ValueError(
                f"vanilla asset policy {policy.assets.vanilla_product_key!r}: snip_id "
                f"{snip_id!r} has {len(matches)} matching null-z asset(s); available assets="
                f"{available_keys}"
            )
        rows.append(matches.iloc[0])
    if not rows:
        return assets.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def assign_group_splits(
    observations: pd.DataFrame, policy: ManifestPolicy
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assign opaque physical embryo IDs using deterministic blake2b content hashing."""

    if not policy.splits.enabled:
        empty = pd.DataFrame(
            columns=["physical_embryo_id", "split", "assignment_source"]
        )
        return empty, {}
    ratios = policy.splits.ratios
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for name, ratio in ratios:
        running += ratio
        cumulative.append((name, running))

    group_experiments = (
        observations.groupby("physical_embryo_id", sort=False)["experiment_id"]
        .agg(lambda values: tuple(dict.fromkeys(str(value) for value in values)))
    )
    explicit_tests = set(policy.splits.test_experiments)
    assignments: dict[str, str] = {}
    sources: dict[str, str] = {}
    for group_id, experiments in group_experiments.items():
        opaque_id = str(group_id)
        if explicit_tests.intersection(experiments):
            assignments[opaque_id] = "test"
            sources[opaque_id] = "explicit_test_experiment"
            continue
        digest = hashlib.blake2b(
            f"{policy.splits.hash_salt}\0{opaque_id}".encode("utf-8"), digest_size=8
        ).digest()
        unit = int.from_bytes(digest, "big") / float(2**64)
        split = cumulative[-1][0]
        for name, upper in cumulative:
            if unit < upper:
                split = name
                break
        assignments[opaque_id] = split
        sources[opaque_id] = "blake2b"

    for split in policy.splits.required_splits:
        if split not in assignments.values():
            raise ValueError(
                f"split policy requires non-empty split {split!r}, but no physical_embryo_id "
                "was assigned to it"
            )

    unpinned = [group for group, source in sources.items() if source == "blake2b"]
    if unpinned:
        for name, target in ratios:
            achieved = sum(assignments[group] == name for group in unpinned) / len(unpinned)
            if abs(achieved - target) > policy.splits.tolerance:
                raise ValueError(
                    f"split ratio tolerance failed on unpinned physical_embryo_id groups for "
                    f"{name!r}: target={target:.6f}, achieved={achieved:.6f}, "
                    f"tolerance={policy.splits.tolerance:.6f}, n_groups={len(unpinned)}"
                )

    table = pd.DataFrame(
        [
            {
                "physical_embryo_id": group,
                "split": assignments[group],
                "assignment_source": sources[group],
            }
            for group in sorted(assignments)
        ]
    )
    return table, assignments


def _record_filter(
    before: pd.Series,
    after: pd.Series,
    observations: pd.DataFrame,
    *,
    name: str,
    reason: str,
    counts: list[FilterCount],
) -> None:
    for experiment_id in tuple(dict.fromkeys(observations["experiment_id"].astype(str))):
        experiment = observations["experiment_id"].astype(str).eq(experiment_id)
        counts.append(
            FilterCount(
                experiment_id,
                name,
                int((before & experiment).sum()),
                int((after & experiment).sum()),
                reason,
            )
        )


def _apply_metric_mapping(observations: pd.DataFrame, policy: ManifestPolicy) -> None:
    mapping_policy = policy.metric_mapping
    if not mapping_policy.enabled:
        return
    if mapping_policy.constant_group is not None:
        observations["metric_group"] = mapping_policy.constant_group
        return
    assert mapping_policy.source_column is not None
    if mapping_policy.source_column not in observations.columns:
        raise ValueError(
            f"metric mapping policy {mapping_policy.name!r} requires source column "
            f"{mapping_policy.source_column!r}"
        )
    mapping = dict(mapping_policy.mapping)
    source = observations[mapping_policy.source_column]
    uncovered = sorted(
        {str(value) for value in source.dropna() if str(value) not in mapping}
    )
    if uncovered:
        raise ValueError(
            f"metric mapping policy {mapping_policy.name!r} leaves source values uncovered: "
            f"{uncovered}"
        )
    observations["metric_group"] = [
        mapping.get(str(value), pd.NA) if not pd.isna(value) else pd.NA for value in source
    ]


def build_manifest_from_tables(
    policy: ManifestPolicy,
    experiments: Sequence[ExperimentTables],
    *,
    source_inventory: Sequence[SourceArtifactRecord] = (),
    path_resolver: PathResolver | None = None,
    writer_contracts: Mapping[str, SourceContract] | None = None,
) -> ManifestResult:
    """Build the v2 manifest from one table bundle per explicit experiment."""

    by_experiment = {bundle.experiment_id: bundle for bundle in experiments}
    if len(by_experiment) != len(experiments):
        raise ValueError("experiment table bundles contain duplicate experiment_id values")
    missing_bundles = [exp for exp in policy.experiment_ids if exp not in by_experiment]
    extra_bundles = sorted(set(by_experiment) - set(policy.experiment_ids))
    if missing_bundles or extra_bundles:
        raise ValueError(
            f"experiment bundles must exactly match the explicit ordered experiment_ids; "
            f"missing={missing_bundles}, extra={extra_bundles}"
        )
    contracts = dict(writer_contracts or _default_contracts())
    schemas: list[SourceSchemaReport] = []
    joins: list[JoinReport] = []
    issues: list[SchemaIssue] = []
    observation_parts: list[pd.DataFrame] = []
    asset_parts: list[pd.DataFrame] = []
    qc_flags_by_experiment: dict[str, tuple[str, ...]] = {}

    for experiment_order, experiment_id in enumerate(policy.experiment_ids):
        bundle = by_experiment[experiment_id]
        inventory = bundle.snip_inventory.copy()
        frame = bundle.frame_inventory.copy()
        plate = bundle.plate_metadata.copy()
        stage = bundle.stage_predictions.copy() if bundle.stage_predictions is not None else None
        qc = bundle.snip_qc.copy() if bundle.snip_qc is not None else None
        acquisition = (
            bundle.acquisition_inventory.copy()
            if bundle.acquisition_inventory is not None
            else None
        )
        for source_name, table in (
            ("snip_inventory", inventory),
            ("frame_inventory", frame),
            ("plate_metadata", plate),
            ("stage_predictions", stage),
            ("snip_qc", qc),
            ("acquisition_inventory", acquisition),
        ):
            if table is not None:
                _require_one_experiment(
                    table, source_name=source_name, experiment_id=experiment_id
                )

        inventory_missing_z = _normalize_z_index(
            inventory, source_name="snip_inventory", experiment_id=experiment_id
        )
        _normalize_z_index(frame, source_name="frame_inventory", experiment_id=experiment_id)
        if inventory_missing_z:
            issues.append(
                SchemaIssue(
                    "warning",
                    experiment_id,
                    "snip_inventory",
                    "current_z_snip_writer_gap",
                    "source has no explicit z_index; rows were normalized to nullable projection "
                    "z_index and no z-plane rendering capability is claimed",
                    ("z_index",),
                )
            )

        _require_columns(
            inventory,
            ADAPTER_REQUIRED_COLUMNS["snip_inventory"],
            source_name="snip_inventory",
            experiment_id=experiment_id,
        )
        _require_columns(
            frame,
            ADAPTER_REQUIRED_COLUMNS["frame_inventory"],
            source_name="frame_inventory",
            experiment_id=experiment_id,
        )
        _require_columns(
            plate,
            ADAPTER_REQUIRED_COLUMNS["plate_metadata"],
            source_name="plate_metadata",
            experiment_id=experiment_id,
        )
        _normalize_boolean_column(
            inventory,
            "is_valid_snip",
            experiment_id=experiment_id,
            identity_column="snip_id",
        )
        if inventory["is_valid_snip"].isna().any():
            ids = inventory.loc[inventory["is_valid_snip"].isna(), "snip_id"].head(5).tolist()
            raise ValueError(
                f"experiment {experiment_id!r}: is_valid_snip is null for snip_id(s) {ids}"
            )
        _require_opaque_strings(
            inventory,
            (
                "experiment_id",
                "well_id",
                "physical_embryo_id",
                "embryo_id",
                "snip_id",
                "image_id",
                "channel_id",
                "snip_product_key",
            ),
            source_name="snip_inventory",
            experiment_id=experiment_id,
        )
        _require_opaque_strings(
            frame,
            ("experiment_id", "well_id", "channel_id"),
            source_name="frame_inventory",
            experiment_id=experiment_id,
        )
        _require_opaque_strings(
            plate,
            ("experiment_id", "well_id"),
            source_name="plate_metadata",
            experiment_id=experiment_id,
        )
        _normalize_time_index(
            inventory, source_name="snip_inventory", experiment_id=experiment_id
        )
        _normalize_time_index(
            frame, source_name="frame_inventory", experiment_id=experiment_id
        )
        if acquisition is not None:
            _normalize_time_index(
                acquisition,
                source_name="acquisition_inventory",
                experiment_id=experiment_id,
            )
        inventory["_source_asset_order"] = np.arange(len(inventory), dtype=np.int64)
        observations = _collapse_observations(inventory, experiment_id=experiment_id)

        assets = inventory.copy()
        _resolve_asset_path_columns(
            assets,
            output_root=policy.pipeline_output_root,
            experiment_id=experiment_id,
            path_resolver=path_resolver,
        )
        valid_missing_path = assets["is_valid_snip"].eq(True) & assets[
            "processed_snip_path"
        ].isna()
        if valid_missing_path.any():
            examples = [
                _asset_key_tuple(row)
                for _, row in assets.loc[valid_missing_path].head(5).iterrows()
            ]
            raise ValueError(
                f"experiment {experiment_id!r}: valid asset(s) lack processed_snip_path: {examples}"
            )

        _join_frame_time(observations, frame, experiment_id=experiment_id, joins=joins)
        _join_plate_metadata(
            observations, plate, experiment_id=experiment_id, joins=joins
        )
        _join_stage(observations, stage, experiment_id=experiment_id, joins=joins)
        qc_flags_by_experiment[experiment_id] = _join_qc(
            observations,
            qc,
            experiment_id=experiment_id,
            joins=joins,
            schema_version=bundle.schema_versions.get("snip_qc"),
        )
        _join_start_age(
            observations,
            bundle.collection_provenance,
            acquisition,
            experiment_id=experiment_id,
            joins=joins,
            issues=issues,
        )
        observations["_experiment_order"] = experiment_order
        assets["_experiment_order"] = experiment_order
        observation_parts.append(observations)
        asset_parts.append(assets)

        source_tables = {
            "snip_inventory": inventory,
            "frame_inventory": frame,
            "plate_metadata": plate,
            "stage_predictions": stage,
            "snip_qc": qc,
            "acquisition_inventory": acquisition,
        }
        for source_name, table in source_tables.items():
            report = _schema_report(
                table,
                source_name=source_name,
                experiment_id=experiment_id,
                path=bundle.source_paths.get(source_name),
                contract=contracts.get(source_name),
            )
            schemas.append(report)
            if report.missing_current_writer:
                issues.append(
                    SchemaIssue(
                        "warning",
                        experiment_id,
                        source_name,
                        "live_artifact_differs_from_current_writer",
                        "artifact omits columns declared by the current writer symbol",
                        report.missing_current_writer,
                    )
                )
        provenance_keys = (
            tuple(str(key) for key in bundle.collection_provenance)
            if bundle.collection_provenance is not None
            else ()
        )
        collection_contract = contracts.get("collection_provenance")
        schemas.append(
            SourceSchemaReport(
                experiment_id,
                "collection_provenance",
                bundle.source_paths.get("collection_provenance"),
                "loaded" if bundle.collection_provenance is not None else "absent",
                provenance_keys,
                tuple((key, type(bundle.collection_provenance[key]).__name__) for key in provenance_keys)
                if bundle.collection_provenance is not None
                else (),
                missing_adapter_required=tuple(
                    key
                    for key in (
                        collection_contract.adapter_required if collection_contract else ()
                    )
                    if key not in provenance_keys
                ),
                missing_current_writer=tuple(
                    key
                    for key in (
                        collection_contract.current_writer_columns
                        if collection_contract
                        else ()
                    )
                    if key not in provenance_keys
                ),
                unexpected_columns=tuple(
                    key
                    for key in provenance_keys
                    if collection_contract
                    and key not in collection_contract.current_writer_columns
                ),
                writer_symbol=(
                    collection_contract.writer_symbol
                    if collection_contract
                    else "data_pipeline.acquisition.metadata_ingest."
                    "collection_provenance_contract.REQUIRED_COLLECTION_PROVENANCE_KEYS"
                ),
            )
        )

    observations = pd.concat(observation_parts, ignore_index=True)
    assets = pd.concat(asset_parts, ignore_index=True)
    observations = observations.sort_values(
        ["_experiment_order", "_source_observation_order"], kind="stable"
    ).reset_index(drop=True)
    observations["observation_row_index"] = np.arange(len(observations), dtype=np.int64)
    observation_rank = dict(
        zip(observations["snip_id"].astype(str), observations["observation_row_index"])
    )

    present_products = tuple(dict.fromkeys(assets["snip_product_key"].astype(str)))
    undeclared_products = sorted(set(present_products) - set(policy.assets.product_order))
    if undeclared_products:
        raise ValueError(
            "asset source contains product keys absent from the explicit product_order: "
            f"{undeclared_products}"
        )
    product_rank = {product: rank for rank, product in enumerate(policy.assets.product_order)}
    assets["_observation_rank"] = assets["snip_id"].astype(str).map(observation_rank)
    assets["_product_rank"] = assets["snip_product_key"].astype(str).map(product_rank)
    assets["_z_null_rank"] = assets["z_index"].notna().astype(int)
    assets["_z_sort"] = assets["z_index"].fillna(-1).astype(int)
    assets = assets.sort_values(
        ["_observation_rank", "_product_rank", "_z_null_rank", "_z_sort", "_source_asset_order"],
        kind="stable",
    ).reset_index(drop=True)
    assets["asset_row_index"] = np.arange(len(assets), dtype=np.int64)
    validate_manifest_tables(observations, assets)

    _apply_metric_mapping(observations, policy)
    counts: list[FilterCount] = []
    candidate = pd.Series(True, index=observations.index, dtype=bool)

    before = candidate.copy()
    if policy.qc.enabled:
        for experiment_id in policy.experiment_ids:
            missing_flags = sorted(
                set(policy.qc.exclude_flags) - set(qc_flags_by_experiment[experiment_id])
            )
            if missing_flags:
                raise ValueError(
                    f"QC policy {policy.qc.name!r}: experiment {experiment_id!r} source schema "
                    f"lacks requested individual flag(s) {missing_flags}"
                )
        keep_qc = observations["qc_status"].isin(policy.qc.accepted_statuses)
        if policy.qc.require_use_snip:
            keep_qc &= observations["use_snip"].fillna(False).astype(bool)
        for flag in policy.qc.exclude_flags:
            keep_qc &= ~observations[flag].fillna(False).astype(bool)
        candidate &= keep_qc
    _record_filter(
        before,
        candidate,
        observations,
        name=f"qc:{policy.qc.name}",
        reason="explicit QC status/use_snip/flag policy" if policy.qc.enabled else "disabled",
        counts=counts,
    )

    before = candidate.copy()
    if policy.stage.enabled:
        keep_stage = observations["stage_status"].isin(policy.stage.accepted_statuses)
        if policy.stage.require_value:
            keep_stage &= observations["predicted_stage_hpf"].notna()
        candidate &= keep_stage
    _record_filter(
        before,
        candidate,
        observations,
        name=f"stage:{policy.stage.name}",
        reason="explicit stage status/value policy" if policy.stage.enabled else "disabled",
        counts=counts,
    )

    before = candidate.copy()
    for column in policy.covariates.required_columns:
        if column not in observations.columns:
            raise ValueError(
                f"covariate policy {policy.covariates.name!r} requires absent column {column!r}"
            )
        candidate &= observations[column].notna()
    _record_filter(
        before,
        candidate,
        observations,
        name=f"covariates:{policy.covariates.name}",
        reason=f"required non-null columns={policy.covariates.required_columns}",
        counts=counts,
    )

    before = candidate.copy()
    candidate_observations = observations.loc[candidate]
    selected_assets = select_vanilla_assets(candidate_observations, assets, policy)
    _record_filter(
        before,
        candidate,
        observations,
        name=f"asset:{policy.assets.vanilla_product_key}:projection_null",
        reason="exactly one configured product with null z_index",
        counts=counts,
    )

    before = candidate.copy()
    if policy.validity.enabled:
        valid_by_id = dict(
            zip(
                selected_assets["snip_id"].astype(str),
                selected_assets["is_valid_snip"].astype(bool),
            )
        )
        candidate &= observations["snip_id"].astype(str).map(valid_by_id).fillna(False)
        selected_assets = selected_assets.loc[
            selected_assets["snip_id"].astype(str).isin(
                set(observations.loc[candidate, "snip_id"].astype(str))
            )
        ].reset_index(drop=True)
    _record_filter(
        before,
        candidate,
        observations,
        name=f"validity:{policy.validity.name}",
        reason="selected asset is_valid_snip" if policy.validity.enabled else "disabled",
        counts=counts,
    )

    selected_observations = observations.loc[candidate].copy()
    split_table, assignments = assign_group_splits(selected_observations, policy)
    observations["selected_by_policy"] = candidate
    observations["split"] = pd.NA
    if assignments:
        observations.loc[candidate, "split"] = selected_observations[
            "physical_embryo_id"
        ].astype(str).map(assignments).to_numpy()
        split_counts = (
            observations.loc[candidate]
            .groupby("physical_embryo_id", dropna=False)["split"]
            .nunique(dropna=False)
        )
        if split_counts.gt(1).any():
            offenders = split_counts[split_counts.gt(1)].index.astype(str).tolist()[:5]
            raise AssertionError(
                "physical_embryo_id groups crossed splits after assignment: "
                f"{offenders}"
            )

    asset_by_snip = selected_assets.set_index(selected_assets["snip_id"].astype(str))
    sample_rows: list[dict[str, Any]] = []
    for _, observation in observations.loc[candidate].iterrows():
        snip_id = str(observation["snip_id"])
        asset = asset_by_snip.loc[snip_id]
        row = observation.to_dict()
        for column, value in asset.items():
            if column not in row:
                row[column] = value
        row["snip_product_key"] = asset["snip_product_key"]
        row["z_index"] = asset["z_index"]
        row["asset_row_index"] = int(asset["asset_row_index"])
        sample_rows.append(row)
    selected_view = pd.DataFrame(sample_rows)

    public_observations = observations.drop(
        columns=["_experiment_order", "_source_observation_order"], errors="ignore"
    )
    public_assets = assets.drop(
        columns=[
            "_experiment_order",
            "_source_asset_order",
            "_observation_rank",
            "_product_rank",
            "_z_null_rank",
            "_z_sort",
        ],
        errors="ignore",
    )
    selected_view = selected_view.drop(
        columns=["_experiment_order", "_source_observation_order"], errors="ignore"
    )
    validate_manifest_tables(public_observations, public_assets)
    observation_order = tuple(public_observations["snip_id"].astype(str))
    asset_order = tuple(
        AssetKey(*_asset_key_tuple(row)) for _, row in public_assets.iterrows()
    )
    policy_name = "/".join(
        (
            policy.qc.name,
            policy.stage.name,
            policy.covariates.name,
            policy.assets.vanilla_product_key,
        )
    )
    return ManifestResult(
        observation_table=public_observations.reset_index(drop=True),
        asset_table=public_assets.reset_index(drop=True),
        selected_sample_view=selected_view.reset_index(drop=True),
        observation_order=observation_order,
        asset_order=asset_order,
        source_inventory=tuple(source_inventory),
        validation_report=ValidationReport(tuple(schemas), tuple(joins), tuple(issues)),
        cohort_report=CohortReport(policy_name, tuple(counts)),
        split_assignments=split_table,
        policy=policy,
    )


def build_pipeline_manifest(policy: ManifestPolicy) -> ManifestResult:
    """Resolve and read explicit experiments through live pipeline helpers, then build tables."""

    contracts = current_writer_contracts()
    bundles: list[ExperimentTables] = []
    inventory_records: list[SourceArtifactRecord] = []
    scopes = policy.acquisition_scopes
    for experiment_id in policy.experiment_ids:
        resolved = resolve_experiment_paths(
            output_root=policy.pipeline_output_root,
            experiment_id=experiment_id,
            acquisition_scope=scopes.get(experiment_id),
        )
        paths = resolved.paths
        required = {
            "snip_inventory": True,
            "frame_inventory": True,
            "plate_metadata": True,
            "stage_predictions": policy.stage.enabled,
            "snip_qc": policy.qc.enabled,
            "collection_provenance": "start_age_hpf" in policy.covariates.required_columns,
            "acquisition_inventory": False,
        }
        tables: dict[str, pd.DataFrame | None] = {}
        for source_name in (
            "snip_inventory",
            "frame_inventory",
            "plate_metadata",
            "stage_predictions",
            "snip_qc",
            "acquisition_inventory",
        ):
            path = paths.get(source_name)
            if path is None or not path.exists():
                if required[source_name]:
                    missing_path = path or Path("<unresolved>")
                    raise FileNotFoundError(
                        f"experiment {experiment_id!r}: required {source_name} artifact is "
                        f"missing: {missing_path}"
                    )
                tables[source_name] = None
                if path is not None:
                    inventory_records.append(
                        fingerprint_artifact(
                            experiment_id=experiment_id,
                            source_name=source_name,
                            path=path,
                            required=False,
                            row_count=None,
                        )
                    )
                continue
            table = read_artifact_table(
                path, source_name=source_name, experiment_id=experiment_id
            )
            tables[source_name] = table
            inventory_records.append(
                fingerprint_artifact(
                    experiment_id=experiment_id,
                    source_name=source_name,
                    path=path,
                    required=required[source_name],
                    row_count=len(table),
                )
            )

        provenance_path = paths["collection_provenance"]
        if provenance_path.exists():
            provenance = read_collection_provenance(
                provenance_path, experiment_id=experiment_id
            )
            inventory_records.append(
                fingerprint_artifact(
                    experiment_id=experiment_id,
                    source_name="collection_provenance",
                    path=provenance_path,
                    required=required["collection_provenance"],
                    row_count=1,
                )
            )
        elif required["collection_provenance"]:
            raise FileNotFoundError(
                f"experiment {experiment_id!r}: required collection provenance artifact is "
                f"missing: {provenance_path}"
            )
        else:
            provenance = None
            inventory_records.append(
                fingerprint_artifact(
                    experiment_id=experiment_id,
                    source_name="collection_provenance",
                    path=provenance_path,
                    required=False,
                    row_count=None,
                )
            )
        bundles.append(
            ExperimentTables(
                experiment_id=experiment_id,
                snip_inventory=tables["snip_inventory"],  # type: ignore[arg-type]
                frame_inventory=tables["frame_inventory"],  # type: ignore[arg-type]
                plate_metadata=tables["plate_metadata"],  # type: ignore[arg-type]
                stage_predictions=tables["stage_predictions"],
                snip_qc=tables["snip_qc"],
                collection_provenance=provenance,
                acquisition_inventory=tables["acquisition_inventory"],
                source_paths=paths,
            )
        )
    return build_manifest_from_tables(
        policy,
        bundles,
        source_inventory=inventory_records,
        path_resolver=resolve_pipeline_path,
        writer_contracts=contracts,
    )
