#!/usr/bin/env python3
"""Reproduce the Phase-1 stage-estimation reliability audit.

This audit is intentionally read-only with respect to source artifacts.  Pipeline
experiments come only from an explicit ordered authority table; the pipeline output
tree is never globbed.  Derived CSV/JSON files are written under --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_PIPELINE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
)
DEFAULT_TRAINING_METADATA = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/"
    "embryo_metadata_df_train.csv"
)
DEFAULT_AGE_KEY = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/"
    "age_key.csv"
)
DEFAULT_METRIC_KEY = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/"
    "metric_key.csv"
)

EXPECTED_STAGE_STATUS = {
    "predicted",
    "missing_start_age_hpf",
    "missing_temperature",
}
STAGE_AXIS_COLUMNS = {
    "current_core_default": "inferred_stage_hpf",
    "nominal_clock": "predicted_stage_hpf",
    "legacy_mlp": "inferred_stage_hpf_reg",
}


@dataclass(frozen=True)
class Artifact:
    experiment_id: str
    kind: str
    path: Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Audit live and legacy stage estimation without modifying producers or artifacts."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument(
        "--experiment-authority",
        type=Path,
        default=repo_root
        / "docs/refactors/core-model/reports/recon_tables/availability_schema.csv",
        help="Ordered experiment authority; first occurrence defines experiment order.",
    )
    parser.add_argument(
        "--training-metadata", type=Path, default=DEFAULT_TRAINING_METADATA
    )
    parser.add_argument("--legacy-age-key", type=Path, default=DEFAULT_AGE_KEY)
    parser.add_argument("--metric-key", type=Path, default=DEFAULT_METRIC_KEY)
    parser.add_argument(
        "--surface-reference",
        type=Path,
        default=repo_root
        / "src/data_pipeline/quality_control/surface_area_qc/references/"
        "surface_area_reference_v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "docs/refactors/core-model/reports/stage_estimation",
    )
    parser.add_argument(
        "--membership-sample-size",
        type=int,
        default=256,
        help="Deterministic evenly spaced anchors used for candidate-set comparisons.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_experiments(path: Path) -> list[str]:
    authority = pd.read_csv(path, usecols=["experiment_id"], dtype="string")
    values = authority["experiment_id"].dropna().astype(str).tolist()
    # dict preserves first occurrence order in Python 3.10.
    experiments = list(dict.fromkeys(values))
    if not experiments:
        raise ValueError(f"Experiment authority contains no experiment_id values: {path}")
    return experiments


def artifact_paths(root: Path, experiment_id: str) -> dict[str, Path]:
    exp = experiment_id
    return {
        "stage": root
        / "feature_extraction"
        / exp
        / "stage_predictions"
        / f"{exp}_stage_predictions.csv",
        "inventory": root
        / "object_extraction"
        / exp
        / "snips"
        / f"{exp}_snip_inventory.csv",
        "frame_inventory": root
        / "acquisition"
        / exp
        / "frame_inventory"
        / f"{exp}_frame_inventory.csv",
        "plate": root
        / "acquisition"
        / exp
        / "ingest_metadata"
        / "plate_metadata.csv",
        "collection_provenance": root
        / "acquisition"
        / exp
        / "ingest_metadata"
        / "collection_provenance.json",
        "mask_geometry": root
        / "feature_extraction"
        / exp
        / "mask_geometry"
        / f"{exp}_mask_geometry.csv",
        "surface_area_qc": root
        / "quality_control"
        / exp
        / "surface_area_qc"
        / f"{exp}_surface_area_qc.csv",
        "analysis_ready": root
        / "analysis_ready"
        / exp
        / "analysis_ready"
        / f"{exp}_analysis_ready.parquet",
    }


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite_count(series: pd.Series) -> int:
    values = safe_numeric(series).to_numpy(dtype=float)
    return int(np.isfinite(values).sum())


def quantiles(values: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def first_existing_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def parse_bool(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    )
    return mapped.astype("boolean")


def check_unique(frame: pd.DataFrame, key: str, label: str) -> None:
    if key not in frame.columns:
        raise ValueError(f"{label} is missing required key column {key!r}")
    duplicate = frame[key].duplicated(keep=False)
    if duplicate.any():
        example = frame.loc[duplicate, key].astype(str).iloc[0]
        raise ValueError(f"{label} is not one-to-one on {key}; example duplicate={example!r}")


def source_inventory(
    experiments: Sequence[str], pipeline_root: Path
) -> tuple[pd.DataFrame, list[Artifact]]:
    rows: list[dict[str, object]] = []
    artifacts: list[Artifact] = []
    for exp in experiments:
        for kind, path in artifact_paths(pipeline_root, exp).items():
            exists = path.is_file()
            rows.append(
                {
                    "experiment_order": experiments.index(exp),
                    "experiment_id": exp,
                    "artifact_kind": kind,
                    "path": str(path),
                    "present": exists,
                    "size_bytes": path.stat().st_size if exists else None,
                    "mtime_utc": (
                        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                        if exists
                        else None
                    ),
                }
            )
            artifacts.append(Artifact(exp, kind, path))
    return pd.DataFrame(rows), artifacts


def stage_artifact_audit(
    experiments: Sequence[str], pipeline_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    summaries: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    used: list[Path] = []
    for order, exp in enumerate(experiments):
        path = artifact_paths(pipeline_root, exp)["stage"]
        if not path.is_file():
            continue
        used.append(path)
        stage = pd.read_csv(path, low_memory=False)
        required = {"snip_id", "physical_embryo_id", "predicted_stage_hpf"}
        missing = sorted(required - set(stage.columns))
        if missing:
            raise ValueError(f"Stage artifact {path} is missing required columns: {missing}")
        check_unique(stage, "snip_id", str(path))
        values = safe_numeric(stage["predicted_stage_hpf"])
        schema = "with_status" if "stage_prediction_status" in stage.columns else "no_status"
        if "stage_prediction_status" in stage.columns:
            status = stage["stage_prediction_status"].fillna("<null>").astype(str)
        else:
            status = pd.Series("unavailable", index=stage.index, dtype="string")
        for key, count in status.value_counts(dropna=False).items():
            statuses.append(
                {
                    "experiment_id": exp,
                    "stage_schema": schema,
                    "effective_status": key,
                    "row_count": int(count),
                    "status_declared_by_artifact": "stage_prediction_status" in stage.columns,
                }
            )
        sort_columns = [
            column
            for column in ("physical_embryo_id", "time_index")
            if column in stage.columns
        ]
        negative_steps = None
        step_count = None
        if len(sort_columns) == 2:
            ordered = stage.assign(_stage=values).sort_values(sort_columns, kind="mergesort")
            diffs = ordered.groupby("physical_embryo_id", sort=False)["_stage"].diff()
            finite_diffs = diffs[np.isfinite(diffs)]
            negative_steps = int((finite_diffs < -1e-9).sum())
            step_count = int(len(finite_diffs))
            zero_steps = int((np.abs(finite_diffs) <= 1e-9).sum())
            median_step = float(finite_diffs.median()) if len(finite_diffs) else None
            p95_step = float(finite_diffs.quantile(0.95)) if len(finite_diffs) else None
            max_step = float(finite_diffs.max()) if len(finite_diffs) else None
        else:
            zero_steps = None
            median_step = None
            p95_step = None
            max_step = None
        group_sizes = stage.groupby("physical_embryo_id", sort=False).size()
        summaries.append(
            {
                "experiment_order": order,
                "experiment_id": exp,
                "path": str(path),
                "stage_schema": schema,
                "row_count": int(len(stage)),
                "unique_snip_count": int(stage["snip_id"].nunique(dropna=True)),
                "unique_embryo_count": int(stage["physical_embryo_id"].nunique(dropna=True)),
                "finite_stage_count": int(np.isfinite(values).sum()),
                "missing_stage_count": int(values.isna().sum()),
                "stage_min": values.min(skipna=True),
                "stage_max": values.max(skipna=True),
                "model_versions": "|".join(
                    sorted(stage.get("model_version", pd.Series(dtype=str)).dropna().astype(str).unique())
                ),
                "unexpected_status_count": int(
                    (~status.isin(EXPECTED_STAGE_STATUS | {"unavailable"})).sum()
                ),
                "max_snips_per_embryo": int(group_sizes.max()) if len(group_sizes) else 0,
                "trajectory_kind": (
                    "time_series" if len(group_sizes) and int(group_sizes.max()) > 1 else "snapshot"
                ),
                "ordered_finite_step_count": step_count,
                "negative_stage_step_count": negative_steps,
                "zero_stage_step_count": zero_steps,
                "median_stage_step_hpf": median_step,
                "p95_stage_step_hpf": p95_step,
                "max_stage_step_hpf": max_step,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(statuses), used


def collection_provenance_audit(
    experiments: Sequence[str], pipeline_root: Path
) -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, object]] = []
    used: list[Path] = []
    for exp in experiments:
        path = artifact_paths(pipeline_root, exp)["collection_provenance"]
        if not path.is_file():
            continue
        used.append(path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        canonical_map = payload.get("start_age_by_source_ordinal")
        legacy_map = payload.get("start_age_by_time_index")
        sources = payload.get("sources") or []
        rows.append(
            {
                "experiment_id": exp,
                "path": str(path),
                "declared_experiment_id_matches": str(payload.get("experiment_id")) == exp,
                "is_collection": bool(payload.get("is_collection")),
                "source_count": len(sources),
                "canonical_start_age_entry_count": (
                    len(canonical_map) if isinstance(canonical_map, dict) else None
                ),
                "legacy_start_age_entry_count": (
                    len(legacy_map) if isinstance(legacy_map, dict) else None
                ),
                "source_ordinal_missing_count": sum(
                    source.get("source_ordinal") is None for source in sources
                ),
                "declared_hpf_missing_count": sum(
                    source.get("declared_hpf") is None for source in sources
                ),
            }
        )
    return pd.DataFrame(rows), used


def _elapsed_by_image(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    time_col = first_existing_column(frame, ("elapsed_time_s", "experiment_time_s", "time_s"))
    if time_col is None:
        raise ValueError(f"{label} has none of the recognized elapsed-time columns")
    if "image_id" not in frame.columns:
        raise ValueError(f"{label} has no image_id")
    values = frame[["image_id", time_col]].copy()
    values[time_col] = safe_numeric(values[time_col])
    conflicts = values.groupby("image_id")[time_col].nunique(dropna=True)
    if (conflicts > 1).any():
        image_id = str(conflicts[conflicts > 1].index[0])
        raise ValueError(f"{label} has conflicting elapsed times for image_id={image_id!r}")
    return values.drop_duplicates("image_id").rename(columns={time_col: "elapsed_time_s"})


def formula_reconstruction(
    experiments: Sequence[str], pipeline_root: Path
) -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, object]] = []
    used: list[Path] = []
    for exp in experiments:
        paths = artifact_paths(pipeline_root, exp)
        required = [paths[key] for key in ("stage", "frame_inventory", "plate")]
        if not all(path.is_file() for path in required):
            continue
        used.extend(required)
        stage = pd.read_csv(paths["stage"], low_memory=False)
        frame = pd.read_csv(paths["frame_inventory"], low_memory=False)
        plate = pd.read_csv(paths["plate"], low_memory=False)
        elapsed = _elapsed_by_image(frame, str(paths["frame_inventory"]))
        for key, table, label in (
            ("image_id", stage, "stage"),
            ("well_id", stage, "stage"),
            ("well_id", plate, "plate"),
        ):
            if key not in table.columns:
                raise ValueError(f"{exp} {label} table is missing {key}")
        if plate["well_id"].duplicated().any():
            raise ValueError(f"{exp} plate metadata has duplicate well_id")
        merged = stage.merge(elapsed, on="image_id", how="left", validate="many_to_one")
        merged = merged.merge(
            plate[["well_id", "start_age_hpf", "temperature"]],
            on="well_id",
            how="left",
            validate="many_to_one",
        )
        for column in ("elapsed_time_s", "start_age_hpf", "temperature", "predicted_stage_hpf"):
            merged[column] = safe_numeric(merged[column])
        predicted = merged["start_age_hpf"] + merged["elapsed_time_s"] / 3600.0 * (
            0.055 * merged["temperature"] - 0.57
        )
        error = merged["predicted_stage_hpf"] - predicted
        finite = np.isfinite(error)
        provenance_present = paths["collection_provenance"].is_file()
        provenance_is_collection = None
        if provenance_present:
            used.append(paths["collection_provenance"])
            with paths["collection_provenance"].open(encoding="utf-8") as handle:
                provenance = json.load(handle)
            provenance_is_collection = bool(provenance.get("is_collection"))
        rows.append(
            {
                "experiment_id": exp,
                "row_count": int(len(merged)),
                "formula_comparable_count": int(finite.sum()),
                "missing_elapsed_count": int(merged["elapsed_time_s"].isna().sum()),
                "missing_start_age_count": int(merged["start_age_hpf"].isna().sum()),
                "missing_temperature_count": int(merged["temperature"].isna().sum()),
                "max_abs_formula_error_hpf": (
                    float(np.nanmax(np.abs(error[finite]))) if finite.any() else None
                ),
                "mean_abs_formula_error_hpf": (
                    float(np.nanmean(np.abs(error[finite]))) if finite.any() else None
                ),
                "temperature_values_c": "|".join(
                    str(value)
                    for value in sorted(merged["temperature"].dropna().unique().tolist())
                ),
                "start_age_values_hpf": "|".join(
                    str(value)
                    for value in sorted(merged["start_age_hpf"].dropna().unique().tolist())
                ),
                "collection_provenance_present": provenance_present,
                "collection_provenance_is_collection": provenance_is_collection,
                "interpretation": (
                    "current_entrypoint_inputs_present"
                    if provenance_present
                    else "historical_formula_reconstruction_only_missing_required_collection_provenance"
                ),
            }
        )
    return pd.DataFrame(rows), list(dict.fromkeys(used))


def legacy_axis_audit(
    metadata_path: Path, age_key_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    wanted = list(STAGE_AXIS_COLUMNS.values()) + [
        "snip_id",
        "embryo_id",
        "experiment_id",
        "experiment_date",
        "short_pert_name",
        "temperature",
        "manual_stage_hpf",
    ]
    header = pd.read_csv(metadata_path, nrows=0).columns
    metadata = pd.read_csv(
        metadata_path, usecols=[column for column in wanted if column in header], low_memory=False
    )
    age = pd.read_csv(age_key_path, low_memory=False)
    check_unique(metadata, "snip_id", str(metadata_path))
    check_unique(age, "snip_id", str(age_key_path))
    joined = age.merge(
        metadata,
        on="snip_id",
        how="left",
        suffixes=("_age_key", "_metadata"),
        validate="one_to_one",
        indicator=True,
    )
    rows: list[dict[str, object]] = []
    for axis, column in (
        ("current_core_default", "inferred_stage_hpf"),
        ("nominal_clock", "predicted_stage_hpf"),
        ("manual_anatomical", "manual_stage_hpf"),
    ):
        values = metadata[column] if column in metadata else pd.Series(np.nan, index=metadata.index)
        rows.append(
            {
                "universe": "training_metadata",
                "axis": axis,
                "column": column,
                **quantiles(safe_numeric(values)),
            }
        )
    rows.append(
        {
            "universe": "legacy_age_key",
            "axis": "legacy_mlp_age_key",
            "column": "inferred_stage_hpf_reg",
            **quantiles(joined["inferred_stage_hpf_reg"]),
        }
    )
    rows.append(
        {
            "universe": "legacy_age_key",
            "axis": "nominal_clock_age_key",
            "column": "calc_stage_hpf",
            **quantiles(joined["calc_stage_hpf"]),
        }
    )
    for axis, column in STAGE_AXIS_COLUMNS.items():
        values = joined[column] if column in joined else pd.Series(np.nan, index=joined.index)
        rows.append(
            {
                "universe": "exact_age_key_metadata_join",
                "axis": axis,
                "column": column,
                **quantiles(safe_numeric(values)),
            }
        )
    experiment_col = first_existing_column(
        joined, ("experiment_id", "experiment_date_age_key", "experiment_date_metadata")
    )
    by_experiment: list[dict[str, object]] = []
    if experiment_col:
        for experiment, group in joined.groupby(experiment_col, dropna=False, sort=True):
            calc = safe_numeric(group["calc_stage_hpf"])
            mlp = safe_numeric(group["inferred_stage_hpf_reg"])
            delta = mlp - calc
            embryo_col = first_existing_column(group, ("embryo_id_age_key", "embryo_id_metadata"))
            max_per_embryo = (
                int(group.groupby(embryo_col).size().max()) if embryo_col and len(group) else 0
            )
            by_experiment.append(
                {
                    "experiment_id": experiment,
                    "row_count": int(len(group)),
                    "finite_mlp_count": int(np.isfinite(mlp).sum()),
                    "median_mlp_minus_calc_hpf": float(np.nanmedian(delta)),
                    "p05_mlp_minus_calc_hpf": float(np.nanquantile(delta, 0.05)),
                    "p95_mlp_minus_calc_hpf": float(np.nanquantile(delta, 0.95)),
                    "max_snips_per_embryo": max_per_embryo,
                    "trajectory_kind": "time_series" if max_per_embryo > 1 else "snapshot",
                    "temperature_values_c": "|".join(
                        str(value)
                        for value in sorted(
                            safe_numeric(
                                group[
                                    first_existing_column(
                                        group, ("temperature_age_key", "temperature_metadata")
                                    )
                                ]
                            )
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    ),
                }
            )
    provenance_columns = [
        column for column in ("train_dir", "model_name", "architecture_name") if column in age
    ]
    provenance = age[provenance_columns].drop_duplicates().copy()
    provenance.insert(0, "provenance_tuple_count", len(provenance))
    provenance["age_key_path"] = str(age_key_path)
    provenance["metadata_match_count"] = int((joined["_merge"] == "both").sum())
    provenance["metadata_missing_count"] = int((joined["_merge"] == "left_only").sum())
    return (
        pd.DataFrame(rows),
        pd.DataFrame(by_experiment),
        provenance,
        joined,
        [metadata_path, age_key_path],
    )


def exact_crosswalk_summary(
    experiments: Sequence[str], pipeline_root: Path, legacy_joined: pd.DataFrame
) -> pd.DataFrame:
    stage_snips: set[str] = set()
    stage_embryos: set[str] = set()
    for exp in experiments:
        path = artifact_paths(pipeline_root, exp)["stage"]
        if not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=lambda column: column in {"snip_id", "physical_embryo_id"})
        stage_snips.update(frame.get("snip_id", pd.Series(dtype=str)).dropna().astype(str))
        stage_embryos.update(
            frame.get("physical_embryo_id", pd.Series(dtype=str)).dropna().astype(str)
        )
    legacy_snips = set(legacy_joined["snip_id"].dropna().astype(str))
    legacy_embryo_col = first_existing_column(
        legacy_joined, ("embryo_id_age_key", "embryo_id_metadata")
    )
    legacy_embryos = (
        set(legacy_joined[legacy_embryo_col].dropna().astype(str)) if legacy_embryo_col else set()
    )
    return pd.DataFrame(
        [
            {
                "key": "snip_id",
                "current_unique": len(stage_snips),
                "legacy_unique": len(legacy_snips),
                "exact_intersection": len(stage_snips & legacy_snips),
                "crosswalk_method": "exact_string_equality_only",
            },
            {
                "key": "physical_embryo_id_vs_legacy_embryo_id",
                "current_unique": len(stage_embryos),
                "legacy_unique": len(legacy_embryos),
                "exact_intersection": len(stage_embryos & legacy_embryos),
                "crosswalk_method": "exact_string_equality_only",
            },
        ]
    )


def _surface_flags(stage: np.ndarray, area: np.ndarray, reference: pd.DataFrame) -> np.ndarray:
    ref_stage = safe_numeric(reference["stage_hpf"]).to_numpy(dtype=float)
    lower = 0.9 * np.interp(stage, ref_stage, safe_numeric(reference["p5"]).to_numpy(float))
    upper = 1.4 * np.interp(stage, ref_stage, safe_numeric(reference["p95"]).to_numpy(float))
    return (area < lower) | (area > upper)


def surface_sensitivity(
    experiments: Sequence[str], pipeline_root: Path, reference_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    reference = pd.read_csv(reference_path)
    used: list[Path] = [reference_path]
    detail: list[dict[str, object]] = []
    offsets = (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0)
    for exp in experiments:
        paths = artifact_paths(pipeline_root, exp)
        needed = [paths[key] for key in ("stage", "mask_geometry", "surface_area_qc")]
        if not all(path.is_file() for path in needed):
            continue
        used.extend(needed)
        stage = pd.read_csv(paths["stage"], low_memory=False)
        geometry = pd.read_csv(paths["mask_geometry"], low_memory=False)
        stored = pd.read_csv(paths["surface_area_qc"], low_memory=False)
        for table, label in ((stage, "stage"), (geometry, "geometry"), (stored, "surface QC")):
            check_unique(table, "snip_id", f"{exp} {label}")
        area_col = first_existing_column(geometry, ("area_um2", "surface_area_um"))
        if area_col is None:
            raise ValueError(f"{exp} mask geometry has no recognized area column")
        merged = stage[["snip_id", "predicted_stage_hpf"]].merge(
            geometry[["snip_id", area_col]], on="snip_id", how="inner", validate="one_to_one"
        )
        stored_columns = ["snip_id", "sa_outlier_flag"]
        if "surface_area_qc_applicability" in stored:
            stored_columns.append("surface_area_qc_applicability")
        merged = merged.merge(stored[stored_columns], on="snip_id", how="left", validate="one_to_one")
        axis = safe_numeric(merged["predicted_stage_hpf"]).to_numpy(float)
        area = safe_numeric(merged[area_col]).to_numpy(float)
        valid = np.isfinite(axis) & np.isfinite(area)
        baseline = np.full(len(merged), False)
        baseline[valid] = _surface_flags(axis[valid], area[valid], reference)
        stored_flag = parse_bool(merged["sa_outlier_flag"])
        comparable = valid & stored_flag.notna().to_numpy()
        for offset in offsets:
            shifted = np.full(len(merged), False)
            shifted[valid] = _surface_flags(axis[valid] + offset, area[valid], reference)
            detail.append(
                {
                    "experiment_id": exp,
                    "scenario": "additive_stage_offset",
                    "stage_shift_hpf": offset,
                    "joined_row_count": int(len(merged)),
                    "applicable_row_count": int(valid.sum()),
                    "flagged_count": int(shifted[valid].sum()),
                    "flagged_fraction": float(shifted[valid].mean()) if valid.any() else None,
                    "flag_flip_vs_recomputed_baseline_count": int((shifted[valid] != baseline[valid]).sum()),
                    "stored_flag_comparable_count": int(comparable.sum()),
                    "stored_vs_recomputed_baseline_disagreement_count": int(
                        (
                            stored_flag[comparable].astype(bool).to_numpy()
                            != baseline[comparable]
                        ).sum()
                    ),
                }
            )
    frame = pd.DataFrame(detail)
    if frame.empty:
        aggregate = pd.DataFrame()
    else:
        aggregate = (
            frame.groupby(["scenario", "stage_shift_hpf"], as_index=False)
            .agg(
                experiment_count=("experiment_id", "nunique"),
                applicable_row_count=("applicable_row_count", "sum"),
                flagged_count=("flagged_count", "sum"),
                flag_flip_vs_recomputed_baseline_count=(
                    "flag_flip_vs_recomputed_baseline_count",
                    "sum",
                ),
                stored_flag_comparable_count=("stored_flag_comparable_count", "sum"),
                stored_vs_recomputed_baseline_disagreement_count=(
                    "stored_vs_recomputed_baseline_disagreement_count",
                    "sum",
                ),
            )
        )
        aggregate["flagged_fraction"] = (
            aggregate["flagged_count"] / aggregate["applicable_row_count"]
        )
        aggregate["flip_fraction"] = (
            aggregate["flag_flip_vs_recomputed_baseline_count"]
            / aggregate["applicable_row_count"]
        )
    return frame, aggregate, list(dict.fromkeys(used))


def read_metric_key(path: Path) -> pd.DataFrame:
    metric = pd.read_csv(path, index_col=0)
    metric.index = metric.index.astype(str)
    metric.columns = metric.columns.astype(str)
    metric = metric.apply(pd.to_numeric, errors="coerce")
    if set(metric.index) != set(metric.columns):
        raise ValueError("Metric key row/column label sets differ")
    return metric.loc[metric.index, metric.index]


def _axis_and_group(legacy_joined: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"snip_id": legacy_joined["snip_id"].astype(str)})
    result["embryo_id"] = legacy_joined[
        first_existing_column(legacy_joined, ("embryo_id_age_key", "embryo_id_metadata"))
    ].astype("string")
    result["short_pert_name"] = legacy_joined[
        first_existing_column(
            legacy_joined, ("short_pert_name_age_key", "short_pert_name_metadata")
        )
    ].astype("string")
    experiment_col = first_existing_column(
        legacy_joined, ("experiment_id", "experiment_date_age_key", "experiment_date_metadata")
    )
    temperature_col = first_existing_column(
        legacy_joined, ("temperature_age_key", "temperature_metadata")
    )
    result["experiment_id"] = legacy_joined[experiment_col].astype("string")
    result["temperature_c"] = safe_numeric(legacy_joined[temperature_col])
    per_experiment_max = result.groupby("experiment_id", dropna=False)["embryo_id"].transform(
        lambda embryos: embryos.map(embryos.value_counts()).max()
    )
    result["trajectory_kind"] = np.where(per_experiment_max > 1, "time_series", "snapshot")
    for axis, column in STAGE_AXIS_COLUMNS.items():
        result[axis] = safe_numeric(legacy_joined[column])
    return result


def _candidate_count(
    values: np.ndarray,
    group_codes: np.ndarray,
    embryo_codes: np.ndarray,
    positive_group_codes: dict[int, np.ndarray],
    window: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    same = np.zeros(n, dtype=np.int64)
    other = np.zeros(n, dtype=np.int64)
    valid = np.isfinite(values) & (group_codes >= 0) & (embryo_codes >= 0)
    groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for group_code in np.unique(group_codes[valid]):
        indices = np.flatnonzero(valid & (group_codes == group_code))
        order = np.argsort(values[indices], kind="mergesort")
        groups[int(group_code)] = (indices[order], values[indices][order])
    for group_code, (indices, sorted_values) in groups.items():
        anchors = values[indices]
        same[indices] = np.searchsorted(sorted_values, anchors + window, side="right") - np.searchsorted(
            sorted_values, anchors - window, side="left"
        )
        for target_code in positive_group_codes.get(group_code, np.empty(0, dtype=int)):
            target = groups.get(int(target_code))
            if target is None:
                continue
            target_values = target[1]
            other[indices] += np.searchsorted(
                target_values, anchors + window, side="right"
            ) - np.searchsorted(target_values, anchors - window, side="left")
    # Relation-positive candidates from the anchor embryo are illegal "other" samples.
    by_embryo: dict[int, np.ndarray] = {}
    for embryo in np.unique(embryo_codes[valid]):
        by_embryo[int(embryo)] = np.flatnonzero(valid & (embryo_codes == embryo))
    for embryo_indices in by_embryo.values():
        for anchor_index in embryo_indices:
            related = positive_group_codes.get(
                int(group_codes[anchor_index]), np.empty(0, dtype=int)
            )
            illegal = embryo_indices[np.isin(group_codes[embryo_indices], related)]
            illegal = illegal[np.abs(values[illegal] - values[anchor_index]) <= window]
            other[anchor_index] -= len(illegal)
    same[~valid] = 0
    other[~valid] = 0
    return same, other


def metric_window_sensitivity(
    legacy_joined: pd.DataFrame, metric_path: Path, sample_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    data = _axis_and_group(legacy_joined)
    metric = read_metric_key(metric_path)
    missing_groups = sorted(set(data["short_pert_name"].dropna().astype(str)) - set(metric.index))
    if missing_groups:
        raise ValueError(f"Metric key does not cover groups: {missing_groups[:5]}")
    group_labels = metric.index.tolist()
    group_to_code = {label: index for index, label in enumerate(group_labels)}
    group_codes = data["short_pert_name"].map(group_to_code).fillna(-1).astype(int).to_numpy()
    embryo_codes, _ = pd.factorize(data["embryo_id"], sort=False, use_na_sentinel=True)
    positive_group_codes = {
        group_to_code[source]: np.array(
            [group_to_code[target] for target in group_labels if metric.loc[source, target] == 1],
            dtype=int,
        )
        for source in group_labels
    }
    summary: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for dimension in ("trajectory_kind", "temperature_c", "experiment_id"):
        for stratum_value, indices in data.groupby(dimension, dropna=False).groups.items():
            for axis in STAGE_AXIS_COLUMNS:
                values = data.loc[indices, axis].to_numpy(float)
                coverage.append(
                    {
                        "axis": axis,
                        "stratum": dimension,
                        "stratum_value": stratum_value,
                        "row_count": int(len(values)),
                        "finite_count": int(np.isfinite(values).sum()),
                        "missing_count": int((~np.isfinite(values)).sum()),
                    }
                )
    cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for axis in STAGE_AXIS_COLUMNS:
        values = data[axis].to_numpy(float)
        finite = np.isfinite(values)
        for window in (1.5, 3.0):
            same, other = _candidate_count(
                values, group_codes, embryo_codes, positive_group_codes, window
            )
            cache[(axis, window)] = (same, other)
            valid_same = same[finite]
            valid_other = other[finite]
            summary.append(
                {
                    "axis": axis,
                    "window_hpf": window,
                    "finite_anchor_count": int(finite.sum()),
                    "self_inclusive_zero_count": int((valid_same == 0).sum()),
                    "self_exclusive_zero_count": int((valid_same <= 1).sum()),
                    "other_relation_zero_count": int((valid_other == 0).sum()),
                    "self_inclusive_median_candidates": float(np.median(valid_same)),
                    "self_inclusive_p05_candidates": float(np.quantile(valid_same, 0.05)),
                    "self_inclusive_p95_candidates": float(np.quantile(valid_same, 0.95)),
                    "other_relation_median_candidates": float(np.median(valid_other)),
                    "other_relation_p05_candidates": float(np.quantile(valid_other, 0.05)),
                    "other_relation_p95_candidates": float(np.quantile(valid_other, 0.95)),
                    "scope_note": "pooled pre-split upper bound; relation content preserved from metric_key",
                }
            )
    n = len(data)
    if sample_size <= 0:
        sample_indices = np.array([], dtype=int)
    else:
        # Snapshot rows are a small minority, so a global evenly spaced sample can miss them.
        # Allocate evenly across the declared trajectory strata, then sample each stratum in
        # stable age-key row order.  This is descriptive sensitivity sampling, not weighting.
        trajectory_values = sorted(data["trajectory_kind"].dropna().unique().tolist())
        quota = max(1, sample_size // max(len(trajectory_values), 1))
        sample_parts: list[np.ndarray] = []
        for trajectory in trajectory_values:
            candidates = np.flatnonzero(
                data["trajectory_kind"].to_numpy() == trajectory
            )
            sample_parts.append(
                candidates[
                    np.unique(
                        np.linspace(
                            0, max(len(candidates) - 1, 0), min(quota, len(candidates)), dtype=int
                        )
                    )
                ]
            )
        sample_indices = np.unique(np.concatenate(sample_parts)) if sample_parts else np.array([], dtype=int)
    membership: list[dict[str, object]] = []
    for window in (1.5, 3.0):
        base_values = data["current_core_default"].to_numpy(float)
        for alt_axis in ("nominal_clock", "legacy_mlp"):
            alt_values = data[alt_axis].to_numpy(float)
            for index in sample_indices:
                if not np.isfinite(base_values[index]) or not np.isfinite(alt_values[index]):
                    continue
                positive_codes = positive_group_codes.get(
                    int(group_codes[index]), np.empty(0, dtype=int)
                )
                relation_mask = np.isin(group_codes, positive_codes) & (
                    embryo_codes != embryo_codes[index]
                )
                base_members = set(
                    np.flatnonzero(
                        relation_mask & (np.abs(base_values - base_values[index]) <= window)
                    ).tolist()
                )
                alt_members = set(
                    np.flatnonzero(
                        relation_mask & (np.abs(alt_values - alt_values[index]) <= window)
                    ).tolist()
                )
                union = base_members | alt_members
                intersection = base_members & alt_members
                membership.append(
                    {
                        "snip_id": data.iloc[index]["snip_id"],
                        "experiment_id": data.iloc[index]["experiment_id"],
                        "temperature_c": data.iloc[index]["temperature_c"],
                        "trajectory_kind": data.iloc[index]["trajectory_kind"],
                        "alternative_axis": alt_axis,
                        "window_hpf": window,
                        "baseline_candidate_count": len(base_members),
                        "alternative_candidate_count": len(alt_members),
                        "intersection_count": len(intersection),
                        "union_count": len(union),
                        "jaccard": len(intersection) / len(union) if union else 1.0,
                    }
                )
    membership_frame = pd.DataFrame(membership)
    strata: list[pd.DataFrame] = []
    for dimension in ("trajectory_kind", "temperature_c", "experiment_id"):
        grouped = (
            membership_frame.groupby(
                ["alternative_axis", "window_hpf", dimension], dropna=False, as_index=False
            )
            .agg(
                sampled_anchor_count=("snip_id", "size"),
                median_jaccard=("jaccard", "median"),
                p05_jaccard=("jaccard", lambda values: values.quantile(0.05)),
                p95_jaccard=("jaccard", lambda values: values.quantile(0.95)),
                median_baseline_candidate_count=("baseline_candidate_count", "median"),
                median_alternative_candidate_count=("alternative_candidate_count", "median"),
            )
            .rename(columns={dimension: "stratum_value"})
        )
        grouped.insert(2, "stratum", dimension)
        strata.append(grouped)
    return (
        pd.DataFrame(summary),
        pd.DataFrame(coverage),
        membership_frame,
        pd.concat(strata, ignore_index=True),
        [metric_path],
    )


def consumer_inventory(repo_root: Path) -> pd.DataFrame:
    patterns = (
        "predicted_stage_hpf|inferred_stage_hpf|stage_hpf|stage_prediction_status|"
        "calc_stage_hpf|manual_stage_hpf"
    )
    command = [
        "rg",
        "-l",
        patterns,
        "--glob",
        "!docs/refactors/core-model/_archive/**",
        "--glob",
        "*.py",
        "--glob",
        "*.smk",
        "--glob",
        "*.yaml",
        "--glob",
        "*.yml",
        "--glob",
        "*.sh",
        "--glob",
        "*.ipynb",
        ".",
    ]
    result = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    rows: list[dict[str, object]] = []
    for raw in sorted(line for line in result.stdout.splitlines() if line):
        relative = raw[2:] if raw.startswith("./") else raw
        path = repo_root / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        token_counts = {
            token: text.count(token)
            for token in (
                "predicted_stage_hpf",
                "inferred_stage_hpf_reg",
                "inferred_stage_hpf",
                "manual_stage_hpf",
                "stage_prediction_status",
                "stage_hpf",
            )
        }
        if relative.startswith(("_Archive/", "src/data_pipeline/_archive/")):
            category = "archived_or_backup"
        elif relative.startswith("src/data_pipeline/feature_extraction/stage_predictions/"):
            category = "live_stage_producer"
        elif relative.startswith("src/data_pipeline/quality_control/surface_area_qc/"):
            category = "stage_conditioned_qc"
        elif relative.startswith("src/data_pipeline/analysis_ready/"):
            category = "analysis_ready_consumer"
        elif relative.startswith("src/core/"):
            category = "core_training_or_inference"
        elif relative.startswith("src/build/"):
            category = "legacy_build_or_estimator"
        elif relative.startswith(("src/analyze/", "src/morphseq/", "src/app/")):
            category = "downstream_analysis_or_app"
        elif relative.startswith("tests/"):
            category = "test"
        elif relative.endswith(".ipynb") or relative.startswith(("results/", "dev/")):
            category = "exploratory_notebook"
        elif relative.startswith("src/data_pipeline/pipeline_orchestrator/"):
            category = "orchestration"
        elif relative.startswith("scripts/"):
            category = "script"
        elif relative.startswith("src/legacy/") or relative.startswith("src/vae/"):
            category = "legacy_model"
        else:
            category = "script_or_orchestration"
        if relative.startswith("src/data_pipeline/feature_extraction/stage_predictions/"):
            treatment = "producer_or_contract"
        elif "surface_area_qc" in relative:
            treatment = "qc_conditioning_axis"
        elif relative.endswith("src/core/data/dataset_classes.py") or relative == "src/core/data/dataset_classes.py":
            treatment = "metric_pair_constraint_or_dataset_metadata"
        elif "loss_functions.py" in relative or "loss_configs.py" in relative:
            treatment = "metric_loss_target_or_window_config"
        elif relative.startswith("src/data_pipeline/analysis_ready/"):
            treatment = "metadata_join_filter_order_or_plot_axis"
        elif relative.startswith("src/build/"):
            treatment = "legacy_producer_default_or_build_metadata"
        elif relative.endswith(".ipynb") or "report" in relative or "viz" in relative:
            treatment = "plot_exploration_or_reporting"
        elif "config" in relative:
            treatment = "configuration_or_fallback"
        else:
            treatment = "metadata_filter_order_model_input_or_unknown_static_hit"
        rows.append(
            {
                "path": relative,
                "category": category,
                "stage_treatment": treatment,
                "suffix": path.suffix,
                "token_counts_json": json.dumps(token_counts, sort_keys=True),
                "inventory_method": "static token hit; inspect source before inferring runtime reachability",
            }
        )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, label in (
        (args.experiment_authority, "experiment authority"),
        (args.training_metadata, "training metadata"),
        (args.legacy_age_key, "legacy age key"),
        (args.metric_key, "metric key"),
        (args.surface_reference, "surface-area reference"),
    ):
        require_file(path, label)

    experiments = ordered_experiments(args.experiment_authority)
    inventory, _ = source_inventory(experiments, args.pipeline_root)
    stage_summary, stage_status, stage_used = stage_artifact_audit(
        experiments, args.pipeline_root
    )
    collection_summary, collection_used = collection_provenance_audit(
        experiments, args.pipeline_root
    )
    formula, formula_used = formula_reconstruction(experiments, args.pipeline_root)
    (
        legacy_summary,
        legacy_by_experiment,
        legacy_provenance,
        legacy_joined,
        legacy_used,
    ) = legacy_axis_audit(args.training_metadata, args.legacy_age_key)
    crosswalk = exact_crosswalk_summary(experiments, args.pipeline_root, legacy_joined)
    surface_detail, surface_aggregate, surface_used = surface_sensitivity(
        experiments, args.pipeline_root, args.surface_reference
    )
    (
        metric_summary,
        metric_axis_coverage,
        membership,
        membership_strata,
        metric_used,
    ) = metric_window_sensitivity(legacy_joined, args.metric_key, args.membership_sample_size)
    consumers = consumer_inventory(args.repo_root)

    source_files = list(
        dict.fromkeys(
            [args.experiment_authority]
            + stage_used
            + collection_used
            + formula_used
            + legacy_used
            + surface_used
            + metric_used
        )
    )
    fingerprints = pd.DataFrame(
        [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ]
    )

    outputs = {
        "source_inventory.csv": inventory,
        "stage_by_experiment.csv": stage_summary,
        "stage_status_counts.csv": stage_status,
        "collection_provenance.csv": collection_summary,
        "formula_reconstruction.csv": formula,
        "legacy_axis_summary.csv": legacy_summary,
        "legacy_axis_by_experiment.csv": legacy_by_experiment,
        "legacy_age_key_provenance.csv": legacy_provenance,
        "exact_crosswalk.csv": crosswalk,
        "surface_area_stage_sensitivity_by_experiment.csv": surface_detail,
        "surface_area_stage_sensitivity.csv": surface_aggregate,
        "metric_window_sensitivity.csv": metric_summary,
        "metric_axis_coverage_strata.csv": metric_axis_coverage,
        "metric_membership_sample.csv": membership,
        "metric_membership_strata.csv": membership_strata,
        "consumer_inventory.csv": consumers,
        "source_fingerprints.csv": fingerprints,
    }
    for filename, frame in outputs.items():
        write_csv(frame, args.output_dir / filename)

    summary = {
        "measurement_date_utc": datetime.now(tz=timezone.utc).isoformat(),
        "repo_root": str(args.repo_root),
        "pipeline_root": str(args.pipeline_root),
        "experiment_authority": str(args.experiment_authority),
        "ordered_experiment_count": len(experiments),
        "ordered_experiments": experiments,
        "artifact_presence_counts": {
            key: int(value)
            for key, value in inventory.groupby("artifact_kind")["present"].sum().items()
        },
        "stage_artifact_count": int(len(stage_summary)),
        "stage_row_count": int(stage_summary["row_count"].sum()),
        "stage_finite_count": int(stage_summary["finite_stage_count"].sum()),
        "stage_status_counts": {
            str(key): int(value)
            for key, value in stage_status.groupby("effective_status")["row_count"].sum().items()
        },
        "collection_provenance_audit": {
            "artifact_count": int(len(collection_summary)),
            "declared_collection_count": int(collection_summary["is_collection"].sum()),
        },
        "formula_reconstruction_experiment_count": int(len(formula)),
        "formula_reconstruction_current_entrypoint_ready_count": int(
            (formula["interpretation"] == "current_entrypoint_inputs_present").sum()
        ),
        "consumer_inventory_file_count": int(len(consumers)),
        "identifiability": {
            "exact_current_to_legacy_snip_crosswalk": int(
                crosswalk.loc[crosswalk["key"] == "snip_id", "exact_intersection"].iloc[0]
            ),
            "manual_anatomical_anchor_count_in_legacy_training_metadata": int(
                legacy_summary.loc[
                    legacy_summary["axis"] == "manual_anatomical", "n"
                ].iloc[0]
            ),
            "accuracy_conclusion": "not_identifiable_from_audited_artifacts",
        },
        "derived_outputs": sorted(outputs),
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
