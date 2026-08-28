#!/usr/bin/env python3
"""Read-only, explicit-source audit of the stage-conditioned surface-area QC rule.

This study intentionally does not import ``data_pipeline``.  The repository's current
``morphseq-env`` installation cannot resolve that top-level package from the repository root
without forbidden ``PYTHONPATH``/``sys.path`` manipulation.  Instead, the small path table below
is a literal transcription of the merged artifact templates registered in
``src/data_pipeline/pipeline_orchestrator/orchestration/paths.py``.  Experiment membership comes
only from ``--experiment-list``; bundle ingress paths come only from the declared
``--experiment-manifest``.  No output-tree glob is used.

The script writes report-only tables and figures.  It never writes beneath the pipeline output or
SeaHub bundle roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REFERENCE = Path(
    "src/data_pipeline/quality_control/surface_area_qc/references/"
    "surface_area_reference_v1.csv"
)

# Literal merged-path templates from paths.py.  See its PIPELINE_STEPS entries and
# _experiment_step_dir/step_dir/artifact_path implementation (paths.py:125-180,294-374,
# 411-440,559-569,648-689,710-719,799-809,864-883,1083-1175).
CANONICAL_MERGED_PATHS: dict[str, tuple[str, str, str]] = {
    "frame_inventory": ("acquisition", "frame_inventory", "{experiment}_frame_inventory.csv"),
    "plate_metadata": ("acquisition", "ingest_metadata", "plate_metadata.csv"),
    "collection_provenance": (
        "acquisition",
        "ingest_metadata",
        "collection_provenance.json",
    ),
    "snip_inventory": (
        "object_extraction",
        "snips",
        "{experiment}_snip_inventory.csv",
    ),
    "physical_embryo_registry": (
        "object_extraction",
        "physical_embryo_registry",
        "{experiment}_physical_embryo_registry.csv",
    ),
    "frame_masks": (
        "object_extraction",
        "frame_masks",
        "{experiment}_frame_masks.csv",
    ),
    "mask_geometry": (
        "feature_extraction",
        "mask_geometry",
        "{experiment}_mask_geometry.csv",
    ),
    "stage_predictions": (
        "feature_extraction",
        "stage_predictions",
        "{experiment}_stage_predictions.csv",
    ),
    "surface_area_qc": (
        "quality_control",
        "surface_area_qc",
        "{experiment}_surface_area_qc.csv",
    ),
    "mask_quality_qc": (
        "quality_control",
        "mask_quality_qc",
        "{experiment}_mask_quality_qc.csv",
    ),
    "death_detection_qc": (
        "quality_control",
        "death_detection",
        "{experiment}_death_detection_qc.csv",
    ),
    "focus_qc": (
        "quality_control",
        "focus_qc",
        "{experiment}_focus_qc.csv",
    ),
    "motion_blur_qc": (
        "quality_control",
        "motion_blur_qc",
        "{experiment}_motion_blur_qc.csv",
    ),
    "snip_qc": (
        "quality_control",
        "snip_qc",
        "{experiment}_snip_qc.parquet",
    ),
}

IDENTITY_COLUMNS = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
)

STAGE_BINS = (-np.inf, 18.0, 24.0, 30.0, 36.0, 48.0, 60.0, 72.0, np.inf)
STAGE_LABELS = ("<18", "18-24", "24-30", "30-36", "36-48", "48-60", "60-72", ">72")


@dataclass(frozen=True)
class LoadedExperiment:
    rows: pd.DataFrame
    source_records: list[dict[str, object]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute and audit surface-area QC from an explicit ordered experiment list. "
            "Only report artifacts are written."
        )
    )
    parser.add_argument("--experiment-list", type=Path, required=True)
    parser.add_argument(
        "--experiment-limit",
        type=int,
        default=None,
        help="Use the first N non-empty entries, preserving authority order.",
    )
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--pipeline-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-version", default="v1")
    parser.add_argument("--k-lower-min", type=float, default=0.70)
    parser.add_argument("--k-lower-max", type=float, default=0.90)
    parser.add_argument("--k-lower-step", type=float, default=0.05)
    parser.add_argument("--k-upper", type=float, default=1.40)
    parser.add_argument("--production-k-lower", type=float, default=0.90)
    parser.add_argument("--review-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--stage-offsets-hpf",
        default="-3,-1.5,0,1.5,3",
        help="Comma-separated offline stage offsets for the Track-E sensitivity seam.",
    )
    return parser


def _read_experiments(path: Path, limit: int | None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"experiment-list authority is missing: {path}")
    experiments = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not experiments:
        raise ValueError(f"experiment-list authority is empty: {path}")
    duplicates = pd.Series(experiments)[pd.Series(experiments).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"experiment-list authority has duplicate IDs: {duplicates[:10]}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--experiment-limit must be positive when supplied")
        experiments = experiments[:limit]
    return experiments


def _canonical_path(root: Path, source: str, experiment_id: str) -> Path:
    stage, product_dir, filename = CANONICAL_MERGED_PATHS[source]
    return root / stage / experiment_id / product_dir / filename.format(experiment=experiment_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_source(
    path: Path,
    *,
    experiment_id: str | None,
    source: str,
    authority: str,
    used: bool,
    row_count: int | None = None,
    error: str = "",
) -> dict[str, object]:
    exists = path.is_file()
    record: dict[str, object] = {
        "experiment_id": experiment_id,
        "source": source,
        "authority": authority,
        "used": used,
        "path": str(path.resolve()),
        "exists": exists,
        "row_count": row_count,
        "size_bytes": None,
        "mtime_ns": None,
        "sha256": None,
        "error": error,
    }
    if exists:
        stat = path.stat()
        record.update(
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            sha256=_sha256(path),
        )
    return record


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required source {label!r} is missing: {path}")
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except ImportError as exc:
        raise RuntimeError(
            f"required source {label!r} at {path} cannot be read because no Parquet engine is "
            f"available: {exc}"
        ) from exc


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}; available={sorted(df.columns)}")


def _require_unique(df: pd.DataFrame, key: str, label: str) -> None:
    _require_columns(df, [key], label)
    duplicates = df.loc[df[key].astype(str).duplicated(keep=False), key].astype(str).unique().tolist()
    if duplicates:
        raise ValueError(f"{label}: duplicate {key} values {duplicates[:10]}")


def _coerce_bool(series: pd.Series, label: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values: list[bool] = []
    for value in series:
        if pd.isna(value):
            raise ValueError(f"{label}: null boolean value")
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            values.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in mapping:
            values.append(mapping[value.strip().lower()])
        else:
            raise ValueError(f"{label}: unrecognized boolean value {value!r}")
    return pd.Series(values, index=series.index, dtype=bool)


def _assert_identity_agreement(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str,
    left_label: str,
    right_label: str,
) -> None:
    common = [column for column in IDENTITY_COLUMNS if column in left.columns and column in right.columns]
    check = left[[key, *[c for c in common if c != key]]].merge(
        right[[key, *[c for c in common if c != key]]],
        on=key,
        how="inner",
        validate="one_to_one",
        suffixes=("__left", "__right"),
    )
    for column in common:
        if column == key:
            continue
        mismatch = check[f"{column}__left"].astype(str) != check[f"{column}__right"].astype(str)
        if mismatch.any():
            offenders = check.loc[mismatch, key].astype(str).head(10).tolist()
            raise ValueError(
                f"identity disagreement for {column!r} between {left_label} and {right_label}; "
                f"offending {key}s={offenders}"
            )


def _merge_exact_payload(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    key: str,
    payload: Sequence[str],
    base_label: str,
    other_label: str,
) -> pd.DataFrame:
    _require_unique(base, key, base_label)
    _require_unique(other, key, other_label)
    left_ids = set(base[key].astype(str))
    right_ids = set(other[key].astype(str))
    if left_ids != right_ids:
        raise ValueError(
            f"{other_label}: {key} coverage differs from {base_label}; "
            f"only_base={sorted(left_ids-right_ids)[:10]}, only_other={sorted(right_ids-left_ids)[:10]}"
        )
    _assert_identity_agreement(
        base,
        other,
        key=key,
        left_label=base_label,
        right_label=other_label,
    )
    _require_columns(other, payload, other_label)
    return base.merge(other[[key, *payload]], on=key, how="left", validate="one_to_one")


def _manifest_rows(path: Path, experiments: Sequence[str]) -> pd.DataFrame:
    manifest = _read_table(path, "experiment_manifest")
    _require_columns(
        manifest,
        [
            "experiment_id",
            "frame_inventory_csv",
            "plate_metadata_csv",
            "precomputed_frame_masks_csv",
            "runtime_config_yaml",
        ],
        "experiment_manifest",
    )
    _require_unique(manifest, "experiment_id", "experiment_manifest")
    manifest["experiment_id"] = manifest["experiment_id"].astype(str)
    by_id = manifest.set_index("experiment_id")
    missing = [experiment for experiment in experiments if experiment not in by_id.index]
    if missing:
        raise ValueError(f"experiment_manifest lacks selected experiment IDs: {missing}")
    return by_id.loc[list(experiments)].reset_index()


def _load_experiment(
    experiment_id: str,
    manifest_row: pd.Series,
    pipeline_root: Path,
) -> LoadedExperiment:
    required_names = ("plate_metadata", "snip_inventory", "mask_geometry", "stage_predictions", "snip_qc")
    paths = {name: _canonical_path(pipeline_root, name, experiment_id) for name in CANONICAL_MERGED_PATHS}
    loaded = {name: _read_table(paths[name], f"{experiment_id}:{name}") for name in required_names}
    bundle_frame_path = Path(str(manifest_row["frame_inventory_csv"]))
    bundle_masks_path = Path(str(manifest_row["precomputed_frame_masks_csv"]))
    bundle_frame = _read_table(bundle_frame_path, f"{experiment_id}:declared_bundle_frame_inventory")
    bundle_masks = _read_table(bundle_masks_path, f"{experiment_id}:declared_bundle_frame_masks")

    snips = loaded["snip_inventory"].copy()
    _require_columns(
        snips,
        [
            *IDENTITY_COLUMNS,
            "image_id",
            "mask_id",
            "time_index",
            "processed_snip_path",
            "embryo_mask_snip_path",
            "is_valid_snip",
        ],
        f"{experiment_id}:snip_inventory",
    )
    _require_unique(snips, "snip_id", f"{experiment_id}:snip_inventory")
    if not snips["experiment_id"].astype(str).eq(experiment_id).all():
        raise ValueError(f"{experiment_id}: snip_inventory contains a different experiment_id")
    snips["source_row_order"] = np.arange(len(snips), dtype=int)
    snips["is_valid_snip"] = _coerce_bool(
        snips["is_valid_snip"], f"{experiment_id}:snip_inventory.is_valid_snip"
    )

    geometry_payload = ["area_um2", "perimeter_um", "length_um", "width_um"]
    rows = _merge_exact_payload(
        snips,
        loaded["mask_geometry"],
        key="snip_id",
        payload=geometry_payload,
        base_label=f"{experiment_id}:snip_inventory",
        other_label=f"{experiment_id}:mask_geometry",
    )
    stage_payload = ["predicted_stage_hpf", "model_version", "stage_prediction_status"]
    rows = _merge_exact_payload(
        rows,
        loaded["stage_predictions"],
        key="snip_id",
        payload=stage_payload,
        base_label=f"{experiment_id}:snip_inventory",
        other_label=f"{experiment_id}:stage_predictions",
    )

    qc = loaded["snip_qc"].copy()
    _require_columns(
        qc,
        [
            *IDENTITY_COLUMNS,
            "sa_outlier_flag",
            "surface_area_qc_applicability",
            "use_snip",
            "qc_fail_reasons",
        ],
        f"{experiment_id}:snip_qc",
    )
    flag_columns = sorted(column for column in qc.columns if column.endswith("_flag"))
    for column in [*flag_columns, "use_snip"]:
        qc[column] = _coerce_bool(qc[column], f"{experiment_id}:snip_qc.{column}")
    applicability_columns = sorted(column for column in qc.columns if column.endswith("_applicability"))
    qc_payload = [*flag_columns, *applicability_columns, "use_snip", "qc_fail_reasons"]
    rows = _merge_exact_payload(
        rows,
        qc,
        key="snip_id",
        payload=qc_payload,
        base_label=f"{experiment_id}:snip_inventory",
        other_label=f"{experiment_id}:snip_qc",
    )

    plate = loaded["plate_metadata"].copy()
    _require_columns(plate, ["well_id"], f"{experiment_id}:plate_metadata")
    _require_unique(plate, "well_id", f"{experiment_id}:plate_metadata")
    plate_columns = [
        "well_id",
        "genotype",
        "chem_perturbation",
        "perturbation",
        "perturbation_key",
        "perturbation_domain",
        "temperature",
        "start_age_hpf",
        "stage_hpf",
        "source_scope",
        "image_kind",
        "calibration_status",
        "scale_estimation_status",
        "calibration_method",
        "calibration_reference_version",
    ]
    plate_columns = [column for column in plate_columns if column in plate.columns]
    rows = rows.merge(plate[plate_columns], on="well_id", how="left", validate="many_to_one")

    _require_columns(
        bundle_frame,
        ["image_id", "well_id", "time_index", "elapsed_time_s"],
        f"{experiment_id}:declared_bundle_frame_inventory",
    )
    _require_unique(bundle_frame, "image_id", f"{experiment_id}:declared_bundle_frame_inventory")
    frame_payload = [
        "image_id",
        "elapsed_time_s",
        "z_index",
        "image_product_type",
        "projection_method",
        "image_micrometers_per_pixel",
        "source_scope",
        "image_kind",
        "calibration_status",
        "scale_estimation_status",
        "calibration_method",
        "calibration_reference_version",
    ]
    frame_payload = [column for column in frame_payload if column in bundle_frame.columns]
    rename = {
        column: f"bundle_frame_{column}"
        for column in frame_payload
        if column != "image_id"
    }
    frame_piece = bundle_frame[frame_payload].rename(columns=rename)
    rows = rows.merge(frame_piece, on="image_id", how="left", validate="many_to_one")
    if rows["bundle_frame_elapsed_time_s"].isna().any():
        offenders = rows.loc[rows["bundle_frame_elapsed_time_s"].isna(), "snip_id"].head(10).tolist()
        raise ValueError(
            f"{experiment_id}: declared bundle frame inventory does not cover snips {offenders}"
        )

    _require_columns(
        bundle_masks,
        ["mask_id", "segmentation_backend", "segmentation_model_id"],
        f"{experiment_id}:declared_bundle_frame_masks",
    )
    _require_unique(bundle_masks, "mask_id", f"{experiment_id}:declared_bundle_frame_masks")
    rows = rows.merge(
        bundle_masks[["mask_id", "segmentation_backend", "segmentation_model_id"]],
        on="mask_id",
        how="left",
        validate="many_to_one",
    )
    if rows["segmentation_backend"].isna().any():
        offenders = rows.loc[rows["segmentation_backend"].isna(), "snip_id"].head(10).tolist()
        raise ValueError(f"{experiment_id}: declared bundle masks do not cover snips {offenders}")

    n_times = bundle_frame.groupby("well_id", sort=False)["time_index"].nunique(dropna=True)
    rows["acquisition_mode"] = rows["well_id"].map(
        lambda well_id: "snapshot" if n_times.get(well_id, 0) == 1 else "time_series"
    )
    rows["incubation_temperature_c"] = pd.to_numeric(rows.get("temperature"), errors="coerce")
    rows["elapsed_time_s"] = pd.to_numeric(rows["bundle_frame_elapsed_time_s"], errors="coerce")
    rows["source_scope_resolved"] = rows.get(
        "bundle_frame_source_scope", rows.get("source_scope", pd.Series(pd.NA, index=rows.index))
    )
    rows["calibration_status_resolved"] = rows.get(
        "bundle_frame_calibration_status",
        rows.get("calibration_status", pd.Series(pd.NA, index=rows.index)),
    )
    rows["calibration_method_resolved"] = rows.get(
        "bundle_frame_calibration_method",
        rows.get("calibration_method", pd.Series(pd.NA, index=rows.index)),
    )

    if "control_status" in plate.columns:
        rows = rows.merge(
            plate[["well_id", "control_status"]], on="well_id", how="left", validate="many_to_one"
        )
        rows["control_status_source"] = "plate_metadata.control_status"
    elif "is_control" in plate.columns:
        control = plate[["well_id", "is_control"]].copy()
        control["control_status"] = _coerce_bool(
            control["is_control"], f"{experiment_id}:plate_metadata.is_control"
        ).map({True: "control", False: "non_control"})
        rows = rows.merge(
            control[["well_id", "control_status"]], on="well_id", how="left", validate="many_to_one"
        )
        rows["control_status_source"] = "plate_metadata.is_control"
    else:
        rows["control_status"] = "unavailable"
        rows["control_status_source"] = "unavailable_no_explicit_source_column"

    rows["asset_key_snip_id"] = rows["snip_id"].astype(str)
    if "snip_product_key" in rows.columns:
        rows["asset_key_snip_product_key"] = rows["snip_product_key"]
    else:
        rows["asset_key_snip_product_key"] = pd.NA
    if "z_index" in snips.columns:
        rows["asset_key_z_index"] = snips["z_index"]
    else:
        rows["asset_key_z_index"] = pd.NA
    complete_asset_key = rows["asset_key_snip_product_key"].notna()
    rows["asset_key_status"] = np.where(
        complete_asset_key,
        "complete_source_columns",
        "unavailable_missing_snip_product_key",
    )
    rows["processed_snip_path_resolved"] = rows["processed_snip_path"].map(
        lambda value: str((pipeline_root / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    rows["embryo_mask_snip_path_resolved"] = rows["embryo_mask_snip_path"].map(
        lambda value: str((pipeline_root / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )

    source_records: list[dict[str, object]] = []
    loaded_counts = {name: len(frame) for name, frame in loaded.items()}
    for name, path in paths.items():
        used = name in required_names
        error = ""
        if not path.is_file():
            error = "canonical merged artifact absent"
        source_records.append(
            _record_source(
                path,
                experiment_id=experiment_id,
                source=name,
                authority="pipeline_paths_py_merged_template",
                used=used,
                row_count=loaded_counts.get(name),
                error=error,
            )
        )
    for source, path, frame in (
        ("declared_bundle_frame_inventory", bundle_frame_path, bundle_frame),
        ("declared_bundle_frame_masks", bundle_masks_path, bundle_masks),
        ("declared_bundle_plate_metadata", Path(str(manifest_row["plate_metadata_csv"])), None),
        ("declared_runtime_config", Path(str(manifest_row["runtime_config_yaml"])), None),
    ):
        source_records.append(
            _record_source(
                path,
                experiment_id=experiment_id,
                source=source,
                authority="experiment_manifest.csv",
                used=source in {"declared_bundle_frame_inventory", "declared_bundle_frame_masks"},
                row_count=None if frame is None else len(frame),
                error="" if path.is_file() else "declared bundle artifact absent",
            )
        )
    return LoadedExperiment(rows=rows, source_records=source_records)


def _validate_reference(reference: pd.DataFrame, label: str) -> None:
    required = ["stage_hpf", "p5", "p50", "p95", "n"]
    _require_columns(reference, required, label)
    if reference.empty:
        raise ValueError(f"{label}: reference is empty")
    numeric = reference[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: reference contains null/non-finite values")
    if not numeric["stage_hpf"].is_monotonic_increasing:
        raise ValueError(f"{label}: stage_hpf is not monotonic increasing")
    if not ((numeric["p5"] <= numeric["p50"]) & (numeric["p50"] <= numeric["p95"])).all():
        raise ValueError(f"{label}: percentile ordering p5 <= p50 <= p95 is violated")


def _reference_values(stage: pd.Series, reference: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stage_values = pd.to_numeric(stage, errors="coerce").to_numpy(dtype=float)
    ref_stage = reference["stage_hpf"].to_numpy(dtype=float)
    p5 = np.interp(stage_values, ref_stage, reference["p5"].to_numpy(dtype=float))
    p95 = np.interp(stage_values, ref_stage, reference["p95"].to_numpy(dtype=float))
    p5[~np.isfinite(stage_values)] = np.nan
    p95[~np.isfinite(stage_values)] = np.nan
    return p5, p95


def _add_reference_metrics(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    k_lower: float,
    k_upper: float,
    reference_version: str,
) -> pd.DataFrame:
    out = rows.copy()
    out["area_um2"] = pd.to_numeric(out["area_um2"], errors="coerce")
    out["predicted_stage_hpf"] = pd.to_numeric(out["predicted_stage_hpf"], errors="coerce")
    if out["area_um2"].isna().any() or (~np.isfinite(out["area_um2"])).any():
        offenders = out.loc[out["area_um2"].isna() | ~np.isfinite(out["area_um2"]), "snip_id"].head(10)
        raise ValueError(f"mask_geometry area is non-finite for snip IDs {offenders.tolist()}")
    p5, p95 = _reference_values(out["predicted_stage_hpf"], reference)
    out["reference_version"] = reference_version
    out["reference_p5_um2"] = p5
    out["reference_p95_um2"] = p95
    out["lower_threshold_um2"] = k_lower * p5
    out["upper_threshold_um2"] = k_upper * p95
    resolved = np.isfinite(out["predicted_stage_hpf"].to_numpy(dtype=float))
    out["recomputed_too_small"] = resolved & (
        out["area_um2"].to_numpy(dtype=float) < out["lower_threshold_um2"].to_numpy(dtype=float)
    )
    out["recomputed_too_large"] = resolved & (
        out["area_um2"].to_numpy(dtype=float) > out["upper_threshold_um2"].to_numpy(dtype=float)
    )
    out["recomputed_sa_outlier_flag"] = out["recomputed_too_small"] | out["recomputed_too_large"]
    out["recomputed_direction"] = np.select(
        [~resolved, out["recomputed_too_small"], out["recomputed_too_large"]],
        ["not_applicable_missing_stage", "too_small", "too_large"],
        default="within_band",
    )
    lower_margin = (out["area_um2"] - out["lower_threshold_um2"]) / out["lower_threshold_um2"]
    upper_margin = (out["upper_threshold_um2"] - out["area_um2"]) / out["upper_threshold_um2"]
    out["normalized_lower_margin"] = lower_margin
    out["normalized_upper_margin"] = upper_margin
    out["normalized_boundary_margin"] = np.minimum(lower_margin, upper_margin)
    out["lower_deficit_fraction"] = np.maximum(-lower_margin, 0.0)
    out["normalized_violation_depth"] = np.maximum(-out["normalized_boundary_margin"], 0.0)
    out["parity_match"] = out["sa_outlier_flag"].astype(bool) == out[
        "recomputed_sa_outlier_flag"
    ].astype(bool)
    out["stage_bin"] = pd.cut(
        out["predicted_stage_hpf"],
        bins=STAGE_BINS,
        labels=STAGE_LABELS,
        right=True,
        include_lowest=True,
    ).astype("string").fillna("missing")
    width = pd.to_numeric(out["width_um"], errors="coerce")
    perimeter = pd.to_numeric(out["perimeter_um"], errors="coerce")
    area = pd.to_numeric(out["area_um2"], errors="coerce")
    length = pd.to_numeric(out["length_um"], errors="coerce")
    valid_shape = (width > 0) & (perimeter > 0) & (area > 0)
    out["aspect_ratio"] = np.where(valid_shape, length / width, np.nan)
    out["circularity"] = np.where(valid_shape, 4.0 * math.pi * area / (perimeter**2), np.nan)
    out["shape_fields_valid"] = valid_shape
    out["shape_class"] = np.select(
        [
            ~valid_shape,
            (out["aspect_ratio"] >= 2.0) | (out["circularity"] < 0.35),
        ],
        ["unavailable", "elongated_or_thin"],
        default="compact",
    )
    return out


def _counts(rows: pd.DataFrame, mask: pd.Series | np.ndarray) -> tuple[int, int]:
    selected = rows.loc[np.asarray(mask, dtype=bool)]
    return len(selected), selected["physical_embryo_id"].astype(str).nunique()


def _failure_decomposition(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    surface_area_flag_columns = {"sa_outlier_flag", "recomputed_sa_outlier_flag"}
    other_flags = sorted(
        column
        for column in rows.columns
        if column.endswith("_flag") and column not in surface_area_flag_columns
    )
    any_other = rows[other_flags].any(axis=1) if other_flags else pd.Series(False, index=rows.index)
    reason = rows["qc_fail_reasons"].fillna("").astype(str)
    measures: list[tuple[str, pd.Series | np.ndarray]] = [
        ("all_rows", np.ones(len(rows), dtype=bool)),
        ("stored_sa_outlier", rows["sa_outlier_flag"]),
        ("recomputed_too_small", rows["recomputed_too_small"]),
        ("recomputed_too_large", rows["recomputed_too_large"]),
        ("surface_area_only_any_flag", rows["sa_outlier_flag"] & ~any_other),
        ("surface_area_only_exclusion_reason", reason.eq("sa_outlier_flag")),
        ("current_use_snip", rows["use_snip"]),
        ("recovered_if_surface_area_non_excluding", reason.eq("sa_outlier_flag")),
    ]
    for applicability in sorted(rows["surface_area_qc_applicability"].astype(str).unique()):
        measures.append(
            (
                f"surface_area_applicability={applicability}",
                rows["surface_area_qc_applicability"].astype(str).eq(applicability),
            )
        )
    decomposition = []
    for metric, mask in measures:
        row_count, embryo_count = _counts(rows, mask)
        decomposition.append(
            {
                "metric": metric,
                "row_count": row_count,
                "row_fraction": row_count / len(rows) if len(rows) else np.nan,
                "physical_embryo_count": embryo_count,
                "physical_embryo_fraction": embryo_count
                / rows["physical_embryo_id"].astype(str).nunique()
                if len(rows)
                else np.nan,
            }
        )
    cooccurrence = []
    sa = rows["sa_outlier_flag"].astype(bool)
    for column in other_flags:
        both = sa & rows[column].astype(bool)
        cooccurrence.append(
            {
                "other_qc_flag": column,
                "other_flag_rows": int(rows[column].sum()),
                "surface_area_flag_rows": int(sa.sum()),
                "cooccurrence_rows": int(both.sum()),
                "cooccurrence_physical_embryos": rows.loc[
                    both, "physical_embryo_id"
                ].astype(str).nunique(),
                "share_of_surface_area_flags": float(both.sum() / sa.sum()) if sa.any() else np.nan,
            }
        )
    embryo = (
        rows.groupby("physical_embryo_id", sort=False)
        .agg(
            experiment_id=("experiment_id", "first"),
            observation_count=("snip_id", "size"),
            any_sa_outlier=("sa_outlier_flag", "any"),
            any_too_small=("recomputed_too_small", "any"),
            any_too_large=("recomputed_too_large", "any"),
            all_currently_usable=("use_snip", "all"),
            any_currently_usable=("use_snip", "any"),
        )
        .reset_index()
    )
    return pd.DataFrame(decomposition), pd.DataFrame(cooccurrence), embryo


def _summary_by_group(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    work = rows.copy()
    work[column] = work[column].astype("string").fillna("<missing>")
    grouped = work.groupby(column, sort=False, dropna=False)
    return grouped.agg(
        row_count=("snip_id", "size"),
        physical_embryo_count=("physical_embryo_id", "nunique"),
        sa_outlier_rows=("sa_outlier_flag", "sum"),
        too_small_rows=("recomputed_too_small", "sum"),
        too_large_rows=("recomputed_too_large", "sum"),
        current_use_snip_rows=("use_snip", "sum"),
        mean_normalized_boundary_margin=("normalized_boundary_margin", "mean"),
    ).reset_index().rename(columns={column: "stratum_value"})


def _stratification(rows: pd.DataFrame) -> pd.DataFrame:
    requested = [
        "experiment_id",
        "stage_bin",
        "genotype",
        "chem_perturbation",
        "perturbation",
        "perturbation_domain",
        "control_status",
        "incubation_temperature_c",
        "source_scope_resolved",
        "calibration_status_resolved",
        "calibration_method_resolved",
        "segmentation_backend",
        "segmentation_model_id",
        "acquisition_mode",
        "surface_area_qc_applicability",
    ]
    pieces = []
    for column in requested:
        if column not in rows.columns:
            continue
        piece = _summary_by_group(rows, column)
        piece.insert(0, "stratum", column)
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    out["sa_outlier_fraction"] = out["sa_outlier_rows"] / out["row_count"]
    out["too_small_fraction"] = out["too_small_rows"] / out["row_count"]
    out["too_large_fraction"] = out["too_large_rows"] / out["row_count"]
    return out


def _run_lengths(values: Sequence[bool]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    for index, value in enumerate([*values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1, index - start))
            start = None
    return runs


def _track_persistence(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for physical_embryo_id, group in rows.groupby("physical_embryo_id", sort=False):
        ordered = group.sort_values(["time_index", "source_row_order"], kind="mergesort")
        if ordered["time_index"].duplicated().any():
            offenders = ordered.loc[ordered["time_index"].duplicated(keep=False), "snip_id"].tolist()
            raise ValueError(
                f"physical_embryo_id {physical_embryo_id!r} has duplicate time_index rows: {offenders}"
            )
        failures = ordered["recomputed_too_small"].astype(bool).tolist()
        runs = _run_lengths(failures)
        isolated_recoveries = 0
        for start, end, length in runs:
            if length != 1:
                continue
            recovered_before = start > 0 and not failures[start - 1]
            recovered_after = end + 1 < len(failures) and not failures[end + 1]
            isolated_recoveries += int(recovered_before or recovered_after)
        n_obs = len(ordered)
        n_low = int(sum(failures))
        max_run = max((length for _, _, length in runs), default=0)
        if n_obs == 1:
            track_class = (
                "single_observation_low_unassessable"
                if n_low
                else "single_observation_no_low_area"
            )
        elif max_run >= 2:
            track_class = "persistent_low_area"
        elif n_low:
            track_class = "isolated_single_frame_dip"
        else:
            track_class = "no_low_area"
        records.append(
            {
                "physical_embryo_id": str(physical_embryo_id),
                "experiment_id": str(ordered["experiment_id"].iloc[0]),
                "observation_count": n_obs,
                "too_small_count": n_low,
                "too_small_run_count": len(runs),
                "max_too_small_run_length": max_run,
                "isolated_failure_neighbor_recovery_count": isolated_recoveries,
                "max_lower_deficit_fraction": float(ordered["lower_deficit_fraction"].max()),
                "min_normalized_boundary_margin": float(
                    ordered["normalized_boundary_margin"].min()
                ),
                "track_persistence_class": track_class,
            }
        )
    return pd.DataFrame(records)


def _k_values(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0 or maximum < minimum:
        raise ValueError("invalid k_lower sweep bounds/step")
    count = int(round((maximum - minimum) / step))
    values = [round(minimum + index * step, 10) for index in range(count + 1)]
    if not math.isclose(values[-1], maximum, abs_tol=1e-9):
        raise ValueError("k_lower range is not divisible by --k-lower-step")
    return values


def _counterfactuals(
    rows: pd.DataFrame,
    *,
    k_values: Sequence[float],
    production_k_lower: float,
    k_upper: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_small = rows["area_um2"] < production_k_lower * rows["reference_p5_um2"]
    too_large = rows["area_um2"] > k_upper * rows["reference_p95_um2"]
    baseline = baseline_small | too_large
    summaries = []
    composition = []
    strata = [
        "experiment_id",
        "stage_bin",
        "genotype",
        "chem_perturbation",
        "perturbation_domain",
        "incubation_temperature_c",
        "source_scope_resolved",
        "calibration_status_resolved",
        "segmentation_backend",
        "acquisition_mode",
        "surface_area_qc_applicability",
    ]
    for k_lower in k_values:
        too_small = rows["area_um2"] < k_lower * rows["reference_p5_um2"]
        flag = too_small | too_large
        recovered = baseline & ~flag
        newly_flagged = ~baseline & flag
        flagged_rows, flagged_embryos = _counts(rows, flag)
        recovered_rows, recovered_embryos = _counts(rows, recovered)
        lost_rows, lost_embryos = _counts(rows, newly_flagged)
        summaries.append(
            {
                "k_lower": k_lower,
                "k_upper": k_upper,
                "flagged_rows": flagged_rows,
                "flagged_physical_embryos": flagged_embryos,
                "recovered_rows_vs_production": recovered_rows,
                "recovered_physical_embryos_vs_production": recovered_embryos,
                "newly_flagged_rows_vs_production": lost_rows,
                "newly_flagged_physical_embryos_vs_production": lost_embryos,
            }
        )
        for column in strata:
            if column not in rows.columns:
                continue
            values = rows[column].astype("string").fillna("<missing>")
            temp = pd.DataFrame(
                {
                    "stratum_value": values,
                    "flag": flag,
                    "recovered": recovered,
                    "physical_embryo_id": rows["physical_embryo_id"].astype(str),
                }
            )
            for value, group in temp.groupby("stratum_value", sort=False):
                composition.append(
                    {
                        "k_lower": k_lower,
                        "stratum": column,
                        "stratum_value": value,
                        "row_count": len(group),
                        "flagged_rows": int(group["flag"].sum()),
                        "recovered_rows_vs_production": int(group["recovered"].sum()),
                        "physical_embryo_count": group["physical_embryo_id"].nunique(),
                        "flagged_physical_embryos": group.loc[
                            group["flag"], "physical_embryo_id"
                        ].nunique(),
                    }
                )

    valid_shape = rows["shape_fields_valid"].astype(bool)
    policies = {
        "strict_exclusion": baseline,
        "diagnostic_only": pd.Series(False, index=rows.index),
        "candidate_aspect_rescue_v1": too_large
        | (baseline_small & (~valid_shape | (rows["aspect_ratio"] < 2.0))),
        "candidate_aspect_circularity_v1": too_large
        | (
            baseline_small
            & (
                ~valid_shape
                | (
                    (rows["aspect_ratio"] < 2.0)
                    & (rows["circularity"] >= 0.35)
                )
            )
        ),
    }
    policy_summary = []
    policy_composition = []
    for policy, exclude in policies.items():
        row_count, embryo_count = _counts(rows, exclude)
        policy_summary.append(
            {
                "policy": policy,
                "excluded_rows": row_count,
                "excluded_physical_embryos": embryo_count,
                "retained_rows": len(rows) - row_count,
                "shape_rule_definition": (
                    "none"
                    if policy in {"strict_exclusion", "diagnostic_only"}
                    else (
                        "too_large OR (too_small AND (invalid shape OR aspect_ratio < 2))"
                        if policy == "candidate_aspect_rescue_v1"
                        else "too_large OR (too_small AND (invalid shape OR "
                        "(aspect_ratio < 2 AND circularity >= 0.35)))"
                    )
                ),
            }
        )
        for column in strata:
            if column not in rows.columns:
                continue
            values = rows[column].astype("string").fillna("<missing>")
            temp = pd.DataFrame(
                {
                    "stratum_value": values,
                    "exclude": exclude,
                    "physical_embryo_id": rows["physical_embryo_id"].astype(str),
                }
            )
            for value, group in temp.groupby("stratum_value", sort=False):
                policy_composition.append(
                    {
                        "policy": policy,
                        "stratum": column,
                        "stratum_value": value,
                        "row_count": len(group),
                        "excluded_rows": int(group["exclude"].sum()),
                        "physical_embryo_count": group["physical_embryo_id"].nunique(),
                        "excluded_physical_embryos": group.loc[
                            group["exclude"], "physical_embryo_id"
                        ].nunique(),
                    }
                )
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(composition),
        pd.DataFrame(policy_summary),
        pd.DataFrame(policy_composition),
    )


def _stage_sensitivity(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    offsets: Sequence[float],
    k_lower: float,
    k_upper: float,
) -> pd.DataFrame:
    baseline = rows["recomputed_sa_outlier_flag"].astype(bool).to_numpy()
    records = []
    axes: list[tuple[str, pd.Series]] = [
        (f"predicted_stage_hpf_offset_{offset:+g}", rows["predicted_stage_hpf"] + offset)
        for offset in offsets
    ]
    if "stage_hpf" in rows.columns and pd.to_numeric(rows["stage_hpf"], errors="coerce").notna().any():
        axes.append(("plate_metadata.stage_hpf", pd.to_numeric(rows["stage_hpf"], errors="coerce")))
    for axis_name, stage_values in axes:
        p5, p95 = _reference_values(stage_values, reference)
        valid = np.isfinite(pd.to_numeric(stage_values, errors="coerce").to_numpy(dtype=float))
        flag = valid & (
            (rows["area_um2"].to_numpy(dtype=float) < k_lower * p5)
            | (rows["area_um2"].to_numpy(dtype=float) > k_upper * p95)
        )
        changed = flag != baseline
        for experiment_id, index in rows.groupby("experiment_id", sort=False).groups.items():
            idx = np.asarray(list(index), dtype=int)
            records.append(
                {
                    "stage_axis": axis_name,
                    "experiment_id": str(experiment_id),
                    "row_count": len(idx),
                    "flagged_rows": int(flag[idx].sum()),
                    "changed_rows_vs_production_axis": int(changed[idx].sum()),
                    "changed_fraction_vs_production_axis": float(changed[idx].mean()),
                }
            )
        records.append(
            {
                "stage_axis": axis_name,
                "experiment_id": "__all__",
                "row_count": len(rows),
                "flagged_rows": int(flag.sum()),
                "changed_rows_vs_production_axis": int(changed.sum()),
                "changed_fraction_vs_production_axis": float(changed.mean()),
            }
        )
    return pd.DataFrame(records)


def _boundary_review(
    rows: pd.DataFrame,
    track: pd.DataFrame,
    *,
    review_size: int,
    seed: int,
) -> pd.DataFrame:
    work = rows.merge(
        track[["physical_embryo_id", "track_persistence_class"]],
        on="physical_embryo_id",
        how="left",
        validate="many_to_one",
    ).copy()
    lower_distance = work["normalized_lower_margin"].abs()
    upper_distance = work["normalized_upper_margin"].abs()
    work["review_boundary"] = np.where(lower_distance <= upper_distance, "lower", "upper")
    selected_threshold = np.where(
        work["review_boundary"].eq("lower"),
        work["lower_threshold_um2"],
        work["upper_threshold_um2"],
    )
    work["review_boundary_relation"] = np.where(
        work["area_um2"] < selected_threshold,
        "below_threshold",
        "above_threshold",
    )
    work["review_abs_normalized_distance"] = np.where(
        work["review_boundary"].eq("lower"), lower_distance, upper_distance
    )
    work["deterministic_tiebreak"] = work["snip_id"].astype(str).map(
        lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
    )
    strata = [
        "review_boundary",
        "review_boundary_relation",
        "stage_bin",
        "perturbation_domain",
        "shape_class",
        "track_persistence_class",
    ]
    for column in strata:
        if column not in work.columns:
            work[column] = "unavailable"
        work[column] = work[column].astype("string").fillna("<missing>")
    work = work.sort_values(
        [*strata, "review_abs_normalized_distance", "deterministic_tiebreak"],
        kind="mergesort",
    )
    work["within_stratum_rank"] = work.groupby(strata, sort=False).cumcount()
    work = work.sort_values(
        ["within_stratum_rank", *strata, "review_abs_normalized_distance", "deterministic_tiebreak"],
        kind="mergesort",
    ).head(min(review_size, len(work)))
    work.insert(0, "review_order", np.arange(1, len(work) + 1, dtype=int))
    columns = [
        "review_order",
        "snip_id",
        "physical_embryo_id",
        "experiment_id",
        "well_id",
        "time_index",
        "asset_key_snip_id",
        "asset_key_snip_product_key",
        "asset_key_z_index",
        "asset_key_status",
        "processed_snip_path_resolved",
        "embryo_mask_snip_path_resolved",
        "predicted_stage_hpf",
        "stage_bin",
        "area_um2",
        "lower_threshold_um2",
        "upper_threshold_um2",
        "recomputed_direction",
        "review_boundary",
        "review_boundary_relation",
        "review_abs_normalized_distance",
        "genotype",
        "chem_perturbation",
        "perturbation",
        "perturbation_domain",
        "control_status",
        "control_status_source",
        "incubation_temperature_c",
        "shape_class",
        "aspect_ratio",
        "circularity",
        "track_persistence_class",
        "surface_area_qc_applicability",
        "sa_outlier_flag",
        "qc_fail_reasons",
        "segmentation_backend",
        "segmentation_model_id",
        "source_scope_resolved",
        "calibration_status_resolved",
        "calibration_method_resolved",
        "acquisition_mode",
        "within_stratum_rank",
    ]
    return work[[column for column in columns if column in work.columns]].reset_index(drop=True)


def _stage_seam(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "well_id",
        "physical_embryo_id",
        "embryo_id",
        "snip_id",
        "time_index",
        "area_um2",
        "perimeter_um",
        "length_um",
        "width_um",
        "aspect_ratio",
        "circularity",
        "predicted_stage_hpf",
        "stage_prediction_status",
        "model_version",
        "stage_hpf",
        "start_age_hpf",
        "elapsed_time_s",
        "incubation_temperature_c",
        "reference_version",
        "reference_p5_um2",
        "reference_p95_um2",
        "lower_threshold_um2",
        "upper_threshold_um2",
        "recomputed_too_small",
        "recomputed_too_large",
        "recomputed_sa_outlier_flag",
        "sa_outlier_flag",
        "surface_area_qc_applicability",
        "normalized_boundary_margin",
        "lower_deficit_fraction",
        "source_scope_resolved",
        "acquisition_mode",
        "genotype",
        "chem_perturbation",
        "perturbation",
        "perturbation_domain",
        "control_status",
        "calibration_status_resolved",
        "calibration_method_resolved",
        "segmentation_backend",
        "segmentation_model_id",
        "asset_key_snip_id",
        "asset_key_snip_product_key",
        "asset_key_z_index",
        "asset_key_status",
        "processed_snip_path_resolved",
        "embryo_mask_snip_path_resolved",
    ]
    return rows[[column for column in columns if column in rows.columns]].copy()


def _plot_overview(rows: pd.DataFrame, output: Path) -> None:
    colors = {"too_small": "#377eb8", "too_large": "#e41a1c", "within_band": "#999999"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for direction, group in rows.groupby("recomputed_direction", sort=False):
        ax.scatter(
            group["predicted_stage_hpf"],
            group["area_um2"],
            s=10,
            alpha=0.55,
            label=direction,
            color=colors.get(direction, "#984ea3"),
        )
    ordered = rows.sort_values("predicted_stage_hpf", kind="mergesort")
    ax.plot(ordered["predicted_stage_hpf"], ordered["lower_threshold_um2"], color="#377eb8", lw=1)
    ax.plot(ordered["predicted_stage_hpf"], ordered["upper_threshold_um2"], color="#e41a1c", lw=1)
    ax.set_xlabel("predicted_stage_hpf (nominal clock-stage axis)")
    ax.set_ylabel("area_um2")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Surface-area QC audit (read-only recomputation)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_counterfactual(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(summary["k_lower"], summary["flagged_rows"], marker="o", label="flagged rows")
    ax.plot(
        summary["k_lower"],
        summary["recovered_rows_vs_production"],
        marker="o",
        label="recovered vs k_lower=0.90",
    )
    ax.set_xlabel("k_lower")
    ax.set_ylabel("row count")
    ax.legend(frameon=False)
    ax.set_title("Offline lower-bound counterfactual")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _write_csv(frame: pd.DataFrame, output_dir: Path, filename: str) -> None:
    frame.to_csv(output_dir / filename, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    experiments = _read_experiments(args.experiment_list, args.experiment_limit)
    manifest = _manifest_rows(args.experiment_manifest, experiments)
    reference = _read_table(args.reference_csv, "surface_area_reference")
    _validate_reference(reference, "surface_area_reference")
    offsets = [float(value.strip()) for value in args.stage_offsets_hpf.split(",") if value.strip()]

    output_dir = args.output_dir.resolve()
    pipeline_root = args.pipeline_output_root.resolve()
    forbidden_roots = [pipeline_root, args.experiment_manifest.resolve().parent]
    if any(output_dir == root or root in output_dir.parents for root in forbidden_roots):
        raise ValueError(
            f"--output-dir {output_dir} must not be inside a production/bundle source root"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    source_records = [
        _record_source(
            args.experiment_list,
            experiment_id=None,
            source="experiment_list",
            authority="caller_explicit_ordered_authority",
            used=True,
            row_count=len(_read_experiments(args.experiment_list, None)),
        ),
        _record_source(
            args.experiment_manifest,
            experiment_id=None,
            source="experiment_manifest",
            authority="caller_declared_bundle_manifest",
            used=True,
            row_count=len(_read_table(args.experiment_manifest, "experiment_manifest")),
        ),
        _record_source(
            args.reference_csv,
            experiment_id=None,
            source="surface_area_reference",
            authority="packaged_reference_v1",
            used=True,
            row_count=len(reference),
        ),
    ]
    for experiment_order, experiment_id in enumerate(experiments):
        manifest_row = manifest.loc[manifest["experiment_id"].eq(experiment_id)].iloc[0]
        loaded = _load_experiment(experiment_id, manifest_row, pipeline_root)
        frame = loaded.rows.copy()
        frame["experiment_order"] = experiment_order
        frames.append(frame)
        source_records.extend(loaded.source_records)
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.sort_values(["experiment_order", "source_row_order"], kind="mergesort").reset_index(
        drop=True
    )
    rows = _add_reference_metrics(
        rows,
        reference,
        k_lower=args.production_k_lower,
        k_upper=args.k_upper,
        reference_version=args.reference_version,
    )

    parity = rows.loc[
        ~rows["parity_match"],
        [
            "experiment_id",
            "snip_id",
            "physical_embryo_id",
            "area_um2",
            "predicted_stage_hpf",
            "model_version",
            "reference_version",
            "sa_outlier_flag",
            "recomputed_sa_outlier_flag",
            "recomputed_direction",
            "normalized_boundary_margin",
        ],
    ].copy()
    decomposition, cooccurrence, embryo_outcomes = _failure_decomposition(rows)
    stratification = _stratification(rows)
    track = _track_persistence(rows)
    k_values = _k_values(args.k_lower_min, args.k_lower_max, args.k_lower_step)
    counterfactual, counterfactual_strata, policies, policy_strata = _counterfactuals(
        rows,
        k_values=k_values,
        production_k_lower=args.production_k_lower,
        k_upper=args.k_upper,
    )
    stage_sensitivity = _stage_sensitivity(
        rows,
        reference,
        offsets=offsets,
        k_lower=args.production_k_lower,
        k_upper=args.k_upper,
    )
    review = _boundary_review(rows, track, review_size=args.review_size, seed=args.seed)
    stage_seam = _stage_seam(rows)

    selected_path = output_dir / "selected_experiments.txt"
    selected_path.write_text("\n".join(experiments) + "\n")
    source_inventory = pd.DataFrame(source_records)
    _write_csv(source_inventory, output_dir, "source_inventory.csv")
    _write_csv(rows, output_dir, "surface_area_intermediate_rows.csv")
    _write_csv(parity, output_dir, "parity_mismatches.csv")
    _write_csv(decomposition, output_dir, "failure_decomposition.csv")
    _write_csv(cooccurrence, output_dir, "qc_flag_cooccurrence.csv")
    _write_csv(embryo_outcomes, output_dir, "physical_embryo_outcomes.csv")
    _write_csv(stratification, output_dir, "stratification.csv")
    _write_csv(track, output_dir, "track_persistence.csv")
    _write_csv(counterfactual, output_dir, "k_lower_counterfactual.csv")
    _write_csv(counterfactual_strata, output_dir, "k_lower_counterfactual_stratification.csv")
    _write_csv(policies, output_dir, "policy_counterfactual.csv")
    _write_csv(policy_strata, output_dir, "policy_counterfactual_stratification.csv")
    _write_csv(review, output_dir, "boundary_review_set.csv")
    _write_csv(stage_seam, output_dir, "stage_axis_sensitivity_input.csv")
    _write_csv(stage_sensitivity, output_dir, "stage_axis_sensitivity_summary.csv")
    _plot_overview(rows, output_dir / "surface_area_boundary_overview.png")
    _plot_counterfactual(counterfactual, output_dir / "k_lower_counterfactual.png")

    source_missing = source_inventory.loc[
        ~source_inventory["exists"].astype(bool), ["experiment_id", "source", "path", "error"]
    ]
    _write_csv(source_missing, output_dir, "missing_artifacts.csv")
    run_summary = {
        "measurement_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "git_revision": _git_revision(),
        "experiment_authority": str(args.experiment_list.resolve()),
        "experiment_authority_sha256": _sha256(args.experiment_list),
        "selected_experiments": experiments,
        "selection_rule": f"first {len(experiments)} non-empty ordered entries",
        "experiment_manifest": str(args.experiment_manifest.resolve()),
        "experiment_manifest_sha256": _sha256(args.experiment_manifest),
        "pipeline_output_root": str(pipeline_root),
        "reference_csv": str(args.reference_csv.resolve()),
        "reference_sha256": _sha256(args.reference_csv),
        "reference_version": args.reference_version,
        "production_k_lower": args.production_k_lower,
        "k_upper": args.k_upper,
        "row_count": len(rows),
        "physical_embryo_count": rows["physical_embryo_id"].astype(str).nunique(),
        "parity_mismatch_count": len(parity),
        "stored_surface_area_flag_count": int(rows["sa_outlier_flag"].sum()),
        "too_small_count": int(rows["recomputed_too_small"].sum()),
        "too_large_count": int(rows["recomputed_too_large"].sum()),
        "current_use_snip_count": int(rows["use_snip"].sum()),
        "boundary_review_count": len(review),
        "complete_asset_key_count": int(rows["asset_key_status"].eq("complete_source_columns").sum()),
        "explicit_control_status_count": int(
            (~rows["control_status_source"].eq("unavailable_no_explicit_source_column")).sum()
        ),
        "canonical_missing_artifact_count": int((~source_inventory["exists"].astype(bool)).sum()),
        "stage_sensitivity_offsets_hpf": offsets,
        "shape_candidate_note": (
            "Offline heuristic only: aspect>=2 or circularity<0.35 is treated as elongated/thin; "
            "no visual usability labels were assigned."
        ),
        "boundary_selection_note": (
            "Round-robin by boundary, side, stage bin, perturbation domain, shape class, and track "
            "class; nearest normalized boundary distance first; SHA-256(seed|snip_id) tie-break."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
