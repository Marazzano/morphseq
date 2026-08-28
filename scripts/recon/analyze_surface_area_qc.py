#!/usr/bin/env python3
"""Read-only surface-area QC parity, decomposition, and counterfactual study.

The script never discovers experiments or samples by globbing.  Experiments come from an
explicit ordered file, and every artifact path is resolved from the pipeline path declarations
recorded in ``pipeline_orchestrator/orchestration/paths.py``.  Counterfactuals are written only
to the requested report directory; pipeline artifacts are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# These are the merged/experiment path declarations in
# src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:299-308,340-349,363-372,
# 411-427,559-568,648-657,679-688,710-719,753-778,799-808,864-873.  Keeping the
# templates in one explicit table makes this standalone study runnable in morphseq-env even when
# the repository's src-layout package is not installed.  The source inventory records every
# resolved path, including missing files.
ARTIFACT_TEMPLATES: dict[str, str] = {
    "frame_inventory": "acquisition/{experiment_id}/frame_inventory/{experiment_id}_frame_inventory.csv",
    "plate_metadata": "acquisition/{experiment_id}/ingest_metadata/plate_metadata.csv",
    "frame_masks": "object_extraction/{experiment_id}/frame_masks/{experiment_id}_frame_masks.csv",
    "physical_embryo_registry": (
        "object_extraction/{experiment_id}/physical_embryo_registry/"
        "{experiment_id}_physical_embryo_registry.csv"
    ),
    "snip_inventory": "object_extraction/{experiment_id}/snips/{experiment_id}_snip_inventory.csv",
    "mask_geometry": "feature_extraction/{experiment_id}/mask_geometry/{experiment_id}_mask_geometry.csv",
    "stage_predictions": (
        "feature_extraction/{experiment_id}/stage_predictions/"
        "{experiment_id}_stage_predictions.csv"
    ),
    "surface_area_qc": (
        "quality_control/{experiment_id}/surface_area_qc/{experiment_id}_surface_area_qc.csv"
    ),
    "death_detection_qc": (
        "quality_control/{experiment_id}/death_detection/{experiment_id}_death_detection_qc.csv"
    ),
    "mask_quality_qc": (
        "quality_control/{experiment_id}/mask_quality_qc/{experiment_id}_mask_quality_qc.csv"
    ),
    "focus_qc": "quality_control/{experiment_id}/focus_qc/{experiment_id}_focus_qc.csv",
    "motion_blur_qc": (
        "quality_control/{experiment_id}/motion_blur_qc/{experiment_id}_motion_blur_qc.csv"
    ),
    "snip_qc": "quality_control/{experiment_id}/snip_qc/{experiment_id}_snip_qc.parquet",
}

CORE_REQUIRED_SOURCES: tuple[str, ...] = (
    "snip_inventory",
    "mask_geometry",
    "stage_predictions",
    "snip_qc",
)

QC_FLAG_COLUMNS: tuple[str, ...] = (
    "persistence_dead_flag",
    "viability_dead_flag",
    "focus_flag",
    "discontinuous_mask_flag",
    "edge_flag",
    "overlapping_mask_flag",
    "motion_blur_flag",
    "sa_outlier_flag",
)

QC_SOURCE_FLAGS: dict[str, tuple[str, ...]] = {
    "death_detection_qc": ("persistence_dead_flag", "viability_dead_flag"),
    "mask_quality_qc": (
        "discontinuous_mask_flag",
        "edge_flag",
        "overlapping_mask_flag",
    ),
    "focus_qc": ("focus_flag",),
    "motion_blur_qc": ("motion_blur_flag",),
}

STAGE_BINS: tuple[float, ...] = (-np.inf, 18.0, 24.0, 36.0, 48.0, 72.0, np.inf)
STAGE_LABELS: tuple[str, ...] = (
    "<18",
    "18-<24",
    "24-<36",
    "36-<48",
    "48-<72",
    ">=72",
)
K_LOWER_SWEEP: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90)
NA_LABEL = "unavailable"


@dataclass(frozen=True)
class StudyConfig:
    output_root: Path
    experiments_file: Path
    experiment_column: str
    reference_csv: Path
    report_dir: Path
    k_lower: float
    k_upper: float
    boundary_size: int
    seed: int
    write_contact_sheets: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce stored surface-area QC flags from explicit pipeline sources, decompose "
            "failures, and write offline-only diagnostic counterfactuals."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Declared pipeline output root.")
    parser.add_argument(
        "--experiments-file",
        type=Path,
        required=True,
        help="Explicit ordered CSV or newline-delimited experiment list; no discovery is performed.",
    )
    parser.add_argument(
        "--experiment-column",
        default="experiment_id",
        help="Experiment ID column for a CSV experiment list (default: experiment_id).",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
        help="Explicit surface-area reference CSV with stage_hpf,p5,p50,p95,n.",
    )
    parser.add_argument("--report-dir", type=Path, required=True, help="Derived-output directory.")
    parser.add_argument("--k-lower", type=float, default=0.90)
    parser.add_argument("--k-upper", type=float, default=1.40)
    parser.add_argument("--boundary-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--no-contact-sheets",
        action="store_true",
        help="Write the boundary review CSV but skip PNG contact sheets.",
    )
    return parser.parse_args(argv)


def ordered_experiments(path: Path, column: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Explicit experiment-list authority does not exist: {path}")
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, dtype=str)
        if column not in table.columns:
            raise ValueError(
                f"Experiment list {path} lacks column {column!r}; columns={list(table.columns)}"
            )
        values = table[column].dropna().astype(str).tolist()
    else:
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"Explicit experiment list {path} is empty.")
    duplicates = pd.Series(values)[pd.Series(values).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"Explicit experiment list {path} contains duplicate IDs {duplicates[:10]}; "
            "ordering authority must be unique."
        )
    return values


def artifact_paths(root: Path, experiment_id: str) -> dict[str, Path]:
    return {
        name: root / template.format(experiment_id=experiment_id)
        for name, template in ARTIFACT_TEMPLATES.items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_signature(columns: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()[:12]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def safe_bool(series: pd.Series, *, label: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{label}: boolean column has null values.")
        return series.astype(bool)
    parsed: list[bool] = []
    for value in series:
        if pd.isna(value):
            raise ValueError(f"{label}: boolean-like column has a null value.")
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            parsed.append(bool(value))
            continue
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError(f"{label}: unrecognized boolean value {value!r}.")
        parsed.append(mapping[key])
    return pd.Series(parsed, index=series.index, dtype=bool)


def require_unique(df: pd.DataFrame, key: str, label: str) -> None:
    if key not in df.columns:
        raise ValueError(f"{label}: missing required key {key!r}.")
    duplicated = df[key].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, key].astype(str).head(5).tolist()
        raise ValueError(f"{label}: duplicate {key} values; examples={examples}.")


def collapse_snip_inventory(df: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    required = [
        "snip_id",
        "physical_embryo_id",
        "well_id",
        "image_id",
        "time_index",
        "processed_snip_path",
        "is_valid_snip",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{experiment_id} snip_inventory missing required columns {missing}.")
    table = df.copy()
    table["snip_id"] = table["snip_id"].astype(str)
    parent_columns = [
        column
        for column in (
            "physical_embryo_id",
            "embryo_id",
            "experiment_id",
            "well_id",
            "image_id",
            "time_index",
            "channel_id",
            "is_valid_snip",
        )
        if column in table.columns
    ]
    conflicting: list[dict[str, Any]] = []
    for column in parent_columns:
        counts = table.groupby("snip_id", sort=False, dropna=False)[column].nunique(dropna=False)
        for snip_id in counts[counts > 1].index[:5]:
            conflicting.append({"snip_id": snip_id, "column": column})
    if conflicting:
        raise ValueError(
            f"{experiment_id} snip_inventory has conflicting observation identity across assets: "
            f"{conflicting[:5]}"
        )

    if not table["snip_id"].duplicated().any():
        result = table[["snip_id", *parent_columns]].copy()
        for column in (
            "snip_id",
            "physical_embryo_id",
            "embryo_id",
            "experiment_id",
            "well_id",
            "image_id",
            "channel_id",
        ):
            if column in result.columns:
                result[column] = result[column].astype(str)
        result["asset_count"] = 1
        result["processed_snip_path"] = table["processed_snip_path"].fillna("").astype(str)
        result["embryo_mask_snip_path"] = (
            table["embryo_mask_snip_path"].fillna("").astype(str)
            if "embryo_mask_snip_path" in table.columns
            else ""
        )
        result["review_path_status"] = np.where(
            result["processed_snip_path"].ne(""), "available", "path_missing"
        )
        if "snip_product_key" in table.columns:
            result["asset_key_status"] = "available"
            result["asset_keys_json"] = [
                json.dumps(
                    [
                        {
                            "snip_id": str(snip_id),
                            "snip_product_key": (
                                None if pd.isna(product) else str(product)
                            ),
                            "z_index": None if pd.isna(z_index) else int(z_index),
                        }
                    ],
                    separators=(",", ":"),
                )
                for snip_id, product, z_index in zip(
                    table["snip_id"],
                    table["snip_product_key"],
                    table.get("z_index", pd.Series(pd.NA, index=table.index)),
                )
            ]
        else:
            result["asset_key_status"] = "unavailable_source_schema"
            result["asset_keys_json"] = ""
        return result

    records: list[dict[str, Any]] = []
    for snip_id, group in table.groupby("snip_id", sort=False, dropna=False):
        first = group.iloc[0]
        row = {column: first[column] for column in parent_columns}
        row["snip_id"] = snip_id
        paths = group["processed_snip_path"].dropna().astype(str).unique().tolist()
        mask_paths = (
            group["embryo_mask_snip_path"].dropna().astype(str).unique().tolist()
            if "embryo_mask_snip_path" in group.columns
            else []
        )
        if "snip_product_key" in group.columns:
            asset_keys = []
            for _, asset in group.iterrows():
                z_value = asset.get("z_index")
                asset_keys.append(
                    {
                        "snip_id": snip_id,
                        "snip_product_key": (
                            None if pd.isna(asset.get("snip_product_key")) else str(asset["snip_product_key"])
                        ),
                        "z_index": None if pd.isna(z_value) else int(z_value),
                    }
                )
            row["asset_key_status"] = "available"
            row["asset_keys_json"] = json.dumps(asset_keys, separators=(",", ":"))
        else:
            row["asset_key_status"] = "unavailable_source_schema"
            row["asset_keys_json"] = ""
        row["asset_count"] = len(group)
        row["processed_snip_path"] = paths[0] if len(paths) == 1 else ""
        row["embryo_mask_snip_path"] = mask_paths[0] if len(mask_paths) == 1 else ""
        row["review_path_status"] = "available" if len(paths) == 1 else "ambiguous_multiple_assets"
        records.append(row)
    return pd.DataFrame.from_records(records)


def reference_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = ["stage_hpf", "p5", "p50", "p95", "n"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Reference {path} missing required columns {missing}.")
    numeric = table[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric["stage_hpf"].is_monotonic_increasing:
        raise ValueError(f"Reference {path} is non-numeric, null, or not stage-sorted.")
    if not ((numeric["p5"] <= numeric["p50"]) & (numeric["p50"] <= numeric["p95"])).all():
        raise ValueError(f"Reference {path} violates p5 <= p50 <= p95.")
    return numeric


def join_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    require_unique(right, "snip_id", label)
    available = [column for column in columns if column in right.columns]
    piece = right[["snip_id", *available]].copy()
    piece["snip_id"] = piece["snip_id"].astype(str)
    overlapping = [column for column in available if column in left.columns]
    if overlapping:
        comparison = left[["snip_id", *overlapping]].copy()
        comparison["snip_id"] = comparison["snip_id"].astype(str)
        comparison = comparison.merge(
            piece[["snip_id", *overlapping]],
            on="snip_id",
            how="left",
            validate="one_to_one",
            suffixes=("_left", "_right"),
        )
        for column in overlapping:
            left_value = comparison[f"{column}_left"].astype("string").fillna("<NA>")
            right_value = comparison[f"{column}_right"].astype("string").fillna("<NA>")
            mismatch = left_value.ne(right_value)
            if mismatch.any():
                offenders = comparison.loc[mismatch, "snip_id"].head(5).tolist()
                raise ValueError(
                    f"{label}: repeated column {column!r} disagrees with the observation spine "
                    f"for snip IDs {offenders}."
                )
        piece = piece.drop(columns=overlapping)
    if len(piece.columns) == 1:
        return left
    return left.merge(piece, on="snip_id", how="left", validate="one_to_one")


def add_plate_metadata(rows: pd.DataFrame, plate: pd.DataFrame | None) -> pd.DataFrame:
    out = rows.copy()
    wanted = ("genotype", "chem_perturbation", "temperature", "strain", "medium")
    if plate is None or "well_id" not in plate.columns:
        for column in wanted:
            out[column] = pd.NA
        out["control_status_raw"] = NA_LABEL
        out["control_status_source_column"] = NA_LABEL
        return out
    plate = plate.copy()
    plate["well_id"] = plate["well_id"].astype(str)
    require_unique(plate, "well_id", "plate_metadata")
    available = [column for column in wanted if column in plate.columns]
    control_columns = [
        column
        for column in ("control_status", "is_control", "control_flag")
        if column in plate.columns
    ]
    if len(control_columns) > 1:
        raise ValueError(f"plate_metadata has ambiguous control-status columns {control_columns}.")
    keep = ["well_id", *available, *control_columns]
    out = out.merge(plate[keep], on="well_id", how="left", validate="many_to_one")
    for column in wanted:
        if column not in out.columns:
            out[column] = pd.NA
    if control_columns:
        out["control_status_raw"] = out[control_columns[0]].astype("string").fillna(NA_LABEL)
        out["control_status_source_column"] = control_columns[0]
        out = out.drop(columns=control_columns)
    else:
        out["control_status_raw"] = NA_LABEL
        out["control_status_source_column"] = NA_LABEL
    return out


def unique_by_key(
    table: pd.DataFrame,
    *,
    key: str,
    value_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    if key not in table.columns:
        return pd.DataFrame(columns=[key, *value_columns])
    available = [column for column in value_columns if column in table.columns]
    subset = table[[key, *available]].copy()
    subset[key] = subset[key].astype(str)
    records: list[dict[str, Any]] = []
    for value, group in subset.groupby(key, sort=False, dropna=False):
        row: dict[str, Any] = {key: value}
        for column in available:
            values = group[column].dropna().unique().tolist()
            if len(values) > 1:
                row[column] = f"conflict:{json.dumps([str(item) for item in values[:8]])}"
            elif values:
                row[column] = values[0]
            else:
                row[column] = pd.NA
        records.append(row)
    return pd.DataFrame.from_records(records, columns=[key, *available])


def add_frame_context(rows: pd.DataFrame, frame: pd.DataFrame | None) -> pd.DataFrame:
    out = rows.copy()
    requested = (
        "source_scope",
        "image_kind",
        "calibration_status",
        "calibration_method",
        "image_micrometers_per_pixel",
        "raw_micrometers_per_pixel",
        "downsample_method",
        "n_sources",
        "acquisition_mode",
    )
    if frame is None:
        for column in requested:
            out[column] = pd.NA
        out["frame_context_status"] = "artifact_missing"
        return out
    lookup = unique_by_key(frame, key="image_id", value_columns=requested, label="frame_inventory")
    out = out.merge(lookup, on="image_id", how="left", validate="many_to_one")
    for column in requested:
        if column not in out.columns:
            out[column] = pd.NA
    declared_context = all(
        column in frame.columns for column in ("source_scope", "image_kind", "calibration_status")
    )
    out["frame_context_status"] = (
        "declared_modality_columns" if declared_context else "legacy_schema_missing_modality_columns"
    )
    return out


def add_segmentation_context(rows: pd.DataFrame, masks: pd.DataFrame | None) -> pd.DataFrame:
    out = rows.copy()
    columns = (
        "segmentation_backend",
        "segmentation_model_id",
        "tracking_backend",
        "tracking_model_id",
        "track_id_source",
    )
    if masks is None:
        for column in columns:
            out[column] = pd.NA
        out["segmentation_context_status"] = "artifact_missing"
        return out
    key = "mask_id" if "mask_id" in masks.columns and "mask_id" in out.columns else "image_id"
    lookup = unique_by_key(masks, key=key, value_columns=columns, label="frame_masks")
    out = out.merge(lookup, on=key, how="left", validate="many_to_one")
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    out["segmentation_context_status"] = "available"
    return out


def load_qc_source_flags(paths: dict[str, Path]) -> pd.DataFrame | None:
    pieces: list[pd.DataFrame] = []
    for source, flags in QC_SOURCE_FLAGS.items():
        path = paths[source]
        if not path.is_file():
            continue
        table = read_table(path)
        available = [column for column in flags if column in table.columns]
        applicability = [
            column
            for column in table.columns
            if column.endswith("_qc_applicability")
        ]
        if not available or "snip_id" not in table.columns:
            continue
        require_unique(table, "snip_id", f"{source}:{path}")
        piece = table[["snip_id", *available, *applicability]].copy()
        piece["snip_id"] = piece["snip_id"].astype(str)
        for column in available:
            piece[column] = safe_bool(piece[column], label=f"{source}:{column}")
        pieces.append(piece)
    if not pieces:
        return None
    merged = pieces[0]
    for piece in pieces[1:]:
        overlapping = (set(merged.columns) & set(piece.columns)) - {"snip_id"}
        if overlapping:
            piece = piece.drop(columns=sorted(overlapping))
        merged = merged.merge(piece, on="snip_id", how="outer", validate="one_to_one")
    return merged


def current_applicability(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(pd.NA, index=rows.index, dtype="string")
    sources = pd.Series(pd.NA, index=rows.index, dtype="string")
    stored = rows.get("surface_area_qc_applicability")
    if stored is not None:
        available = stored.notna()
        values.loc[available] = stored.loc[available].astype(str)
        sources.loc[available] = "stored"
    unresolved = values.isna()
    missing_stage = unresolved & rows["predicted_stage_hpf"].isna()
    values.loc[missing_stage] = "not_applicable"
    sources.loc[missing_stage] = "current_missing_stage_policy"
    unresolved = values.isna()
    declared = unresolved & rows["frame_context_status"].eq("declared_modality_columns")
    values.loc[declared] = np.where(
        rows.loc[declared, "image_kind"].astype(str).eq("single_z"),
        "diagnostic_only",
        "exclusion",
    )
    sources.loc[declared] = "current_declared_frame_modality"
    unresolved = values.isna()
    values.loc[unresolved] = "exclusion"
    sources.loc[unresolved] = "current_legacy_frame_fallback"
    return values, sources


def prepare_experiment(
    experiment_id: str,
    paths: dict[str, Path],
    reference: pd.DataFrame,
    *,
    k_lower: float,
    k_upper: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    inventory = collapse_snip_inventory(read_table(paths["snip_inventory"]), experiment_id)
    geometry = read_table(paths["mask_geometry"])
    stage = read_table(paths["stage_predictions"])
    snip_qc = read_table(paths["snip_qc"])
    require_unique(geometry, "snip_id", f"{experiment_id} mask_geometry")
    require_unique(stage, "snip_id", f"{experiment_id} stage_predictions")
    require_unique(snip_qc, "snip_id", f"{experiment_id} snip_qc")

    rows = inventory.copy()
    if "mask_id" not in rows.columns:
        original_inventory = read_table(paths["snip_inventory"])
        if "mask_id" in original_inventory.columns:
            mask_lookup = unique_by_key(
                original_inventory, key="snip_id", value_columns=("mask_id",), label="snip_inventory"
            )
            rows = rows.merge(mask_lookup, on="snip_id", how="left", validate="one_to_one")
    rows = join_one_to_one(
        rows,
        geometry,
        columns=("area_um2", "perimeter_um", "length_um", "width_um"),
        label=f"{experiment_id} mask_geometry",
    )
    rows = join_one_to_one(
        rows,
        stage,
        columns=("predicted_stage_hpf", "model_version", "stage_prediction_status"),
        label=f"{experiment_id} stage_predictions",
    )

    qc_columns = [column for column in snip_qc.columns if column != "snip_id"]
    rows = join_one_to_one(rows, snip_qc, columns=qc_columns, label=f"{experiment_id} snip_qc")
    direct_flags = load_qc_source_flags(paths)
    if direct_flags is not None:
        for column in [column for column in direct_flags.columns if column != "snip_id"]:
            if column in rows.columns:
                continue
            rows = join_one_to_one(
                rows,
                direct_flags,
                columns=(column,),
                label=f"{experiment_id} direct QC source",
            )

    if paths["surface_area_qc"].is_file():
        stored_surface = read_table(paths["surface_area_qc"])
        require_unique(stored_surface, "snip_id", f"{experiment_id} surface_area_qc")
        surface_columns = [
            column
            for column in ("sa_outlier_flag", "surface_area_qc_applicability")
            if column in stored_surface.columns
        ]
        renamed = stored_surface[["snip_id", *surface_columns]].copy()
        renamed = renamed.rename(
            columns={
                "sa_outlier_flag": "stored_surface_sa_outlier_flag",
                "surface_area_qc_applicability": "stored_surface_applicability",
            }
        )
        renamed["snip_id"] = renamed["snip_id"].astype(str)
        rows = rows.merge(renamed, on="snip_id", how="left", validate="one_to_one")
        stored_source = "surface_area_qc"
        stored_flag_column = "stored_surface_sa_outlier_flag"
    elif "sa_outlier_flag" in snip_qc.columns:
        stored_source = "snip_qc_carried_flag"
        stored_flag_column = "sa_outlier_flag"
    else:
        raise ValueError(
            f"{experiment_id}: neither {paths['surface_area_qc']} nor snip_qc sa_outlier_flag "
            "provides a stored surface-area verdict."
        )

    if paths["plate_metadata"].is_file():
        rows = add_plate_metadata(rows, read_table(paths["plate_metadata"]))
    else:
        rows = add_plate_metadata(rows, None)
    frame = read_table(paths["frame_inventory"]) if paths["frame_inventory"].is_file() else None
    rows = add_frame_context(rows, frame)
    masks = read_table(paths["frame_masks"]) if paths["frame_masks"].is_file() else None
    rows = add_segmentation_context(rows, masks)

    for column in ("area_um2", "perimeter_um", "length_um", "width_um", "predicted_stage_hpf"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    if rows["area_um2"].isna().any():
        bad = rows.loc[rows["area_um2"].isna(), "snip_id"].head(5).tolist()
        raise ValueError(f"{experiment_id}: non-finite/missing area_um2 for snip IDs {bad}.")

    stage_values = rows["predicted_stage_hpf"].to_numpy(dtype=float)
    finite_stage = np.isfinite(stage_values)
    ref_stage = reference["stage_hpf"].to_numpy(dtype=float)
    ref_p5 = np.full(len(rows), np.nan)
    ref_p50 = np.full(len(rows), np.nan)
    ref_p95 = np.full(len(rows), np.nan)
    ref_p5[finite_stage] = np.interp(stage_values[finite_stage], ref_stage, reference["p5"])
    ref_p50[finite_stage] = np.interp(stage_values[finite_stage], ref_stage, reference["p50"])
    ref_p95[finite_stage] = np.interp(stage_values[finite_stage], ref_stage, reference["p95"])
    rows["ref_p5"] = ref_p5
    rows["ref_p50"] = ref_p50
    rows["ref_p95"] = ref_p95
    rows["lower_bound_um2"] = k_lower * rows["ref_p5"]
    rows["upper_bound_um2"] = k_upper * rows["ref_p95"]
    rows["recomputed_too_small"] = finite_stage & (
        rows["area_um2"].to_numpy() < rows["lower_bound_um2"].to_numpy()
    )
    rows["recomputed_too_large"] = finite_stage & (
        rows["area_um2"].to_numpy() > rows["upper_bound_um2"].to_numpy()
    )
    rows["recomputed_sa_outlier_flag"] = (
        rows["recomputed_too_small"] | rows["recomputed_too_large"]
    )
    rows["stored_sa_outlier_flag"] = safe_bool(
        rows[stored_flag_column], label=f"{experiment_id}:{stored_flag_column}"
    )
    rows["stored_flag_source"] = stored_source

    if "stored_surface_applicability" in rows.columns:
        rows["surface_area_qc_applicability"] = rows["stored_surface_applicability"].combine_first(
            rows.get("surface_area_qc_applicability", pd.Series(pd.NA, index=rows.index))
        )
    applicability, applicability_source = current_applicability(rows)
    rows["resolved_surface_area_qc_applicability"] = applicability
    rows["applicability_source"] = applicability_source
    rows["parity_match"] = rows["stored_sa_outlier_flag"] == rows["recomputed_sa_outlier_flag"]
    rows["lower_normalized_distance"] = (
        rows["area_um2"] - rows["lower_bound_um2"]
    ) / rows["lower_bound_um2"]
    rows["upper_normalized_distance"] = (
        rows["upper_bound_um2"] - rows["area_um2"]
    ) / rows["upper_bound_um2"]
    rows["nearest_normalized_margin"] = rows[
        ["lower_normalized_distance", "upper_normalized_distance"]
    ].min(axis=1)

    valid_shape = (
        rows["length_um"].gt(0)
        & rows["width_um"].gt(0)
        & rows["perimeter_um"].gt(0)
        & rows["area_um2"].gt(0)
    )
    rows["aspect_ratio"] = np.where(valid_shape, rows["length_um"] / rows["width_um"], np.nan)
    rows["circularity"] = np.where(
        valid_shape,
        4.0 * math.pi * rows["area_um2"] / np.square(rows["perimeter_um"]),
        np.nan,
    )
    rows["shape_measurement_status"] = np.where(valid_shape, "available", "invalid_geometry")
    rows["shape_class"] = np.select(
        [rows["aspect_ratio"].ge(2.0), rows["aspect_ratio"].lt(2.0)],
        ["elongated_or_thin", "compact"],
        default=NA_LABEL,
    )
    rows["stage_bin"] = pd.cut(
        rows["predicted_stage_hpf"], bins=STAGE_BINS, labels=STAGE_LABELS, right=False
    ).astype("string").fillna(NA_LABEL)
    rows["track_observation_mode"] = np.where(
        rows.groupby("physical_embryo_id")["snip_id"].transform("size").gt(1),
        "multi_observation_track",
        "single_observation_track",
    )
    if "acquisition_mode" not in rows or rows["acquisition_mode"].isna().all():
        rows["acquisition_mode"] = NA_LABEL
    else:
        rows["acquisition_mode"] = rows["acquisition_mode"].astype("string").fillna(NA_LABEL)

    if "qc_fail_reasons" not in rows.columns:
        rows["qc_fail_reasons"] = ""
    rows["qc_fail_reasons"] = rows["qc_fail_reasons"].fillna("").astype(str)
    rows["is_valid_snip"] = safe_bool(
        rows["is_valid_snip"], label=f"{experiment_id}:is_valid_snip"
    )
    rows["other_qc_fail_reasons"] = rows["qc_fail_reasons"].map(
        lambda value: "|".join(
            reason for reason in value.split("|") if reason and reason != "sa_outlier_flag"
        )
    )
    rows["other_qc_failure"] = rows["other_qc_fail_reasons"].ne("")
    if "use_snip" in rows.columns:
        rows["use_snip"] = safe_bool(rows["use_snip"], label=f"{experiment_id}:use_snip")
    else:
        rows["use_snip"] = rows["qc_fail_reasons"].eq("")
    rows["isolated_sa_only_failure"] = rows["qc_fail_reasons"].eq("sa_outlier_flag")

    for column in QC_FLAG_COLUMNS:
        if column not in rows.columns:
            rows[column] = pd.NA
        elif column != "sa_outlier_flag":
            non_null = rows[column].notna()
            if non_null.any():
                parsed = safe_bool(rows.loc[non_null, column], label=f"{experiment_id}:{column}")
                rows.loc[non_null, column] = parsed.to_numpy()

    summary = {
        "experiment_id": experiment_id,
        "rows": len(rows),
        "physical_embryos": rows["physical_embryo_id"].nunique(),
        "stored_flag_source": stored_source,
        "stored_flag_rows": int(rows["stored_sa_outlier_flag"].notna().sum()),
        "parity_matches": int(rows["parity_match"].sum()),
        "parity_mismatches": int((~rows["parity_match"]).sum()),
        "too_small_rows": int(rows["recomputed_too_small"].sum()),
        "too_large_rows": int(rows["recomputed_too_large"].sum()),
        "missing_stage_rows": int(rows["predicted_stage_hpf"].isna().sum()),
        "surface_applicability_schema": (
            "stored" if (rows["applicability_source"] == "stored").all() else "mixed_or_recomputed"
        ),
    }
    return rows, summary


def annotate_track_persistence(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = rows.copy()
    out["too_small_run_length"] = 0
    out["too_small_track_class"] = "not_too_small"
    track_records: list[dict[str, Any]] = []
    for physical_embryo_id, group in out.groupby("physical_embryo_id", sort=False):
        ordered = group.sort_values(["time_index", "snip_id"], kind="stable")
        time_values = pd.to_numeric(ordered["time_index"], errors="coerce").to_numpy()
        failures = ordered["recomputed_too_small"].to_numpy(dtype=bool)
        runs: list[list[int]] = []
        current: list[int] = []
        for offset, failed in enumerate(failures):
            consecutive = (
                bool(current)
                and np.isfinite(time_values[offset])
                and np.isfinite(time_values[current[-1]])
                and time_values[offset] == time_values[current[-1]] + 1
            )
            if failed:
                if current and not consecutive:
                    runs.append(current)
                    current = []
                current.append(offset)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)

        isolated = 0
        persistent = 0
        neighbor_recovery = 0
        run_lengths: list[int] = []
        for run in runs:
            length = len(run)
            run_lengths.append(length)
            indices = ordered.index[run]
            out.loc[indices, "too_small_run_length"] = length
            if length >= 2:
                out.loc[indices, "too_small_track_class"] = "persistent_run"
                persistent += length
            else:
                position = run[0]
                before_ok = (
                    position > 0
                    and not failures[position - 1]
                    and time_values[position] == time_values[position - 1] + 1
                )
                after_ok = (
                    position + 1 < len(failures)
                    and not failures[position + 1]
                    and time_values[position + 1] == time_values[position] + 1
                )
                if before_ok and after_ok:
                    out.loc[indices, "too_small_track_class"] = "isolated_dip_with_recovery"
                    isolated += 1
                    neighbor_recovery += 1
                else:
                    out.loc[indices, "too_small_track_class"] = "single_failure_edge_or_gap"
                    isolated += 1
        failed_rows = ordered.loc[failures]
        track_records.append(
            {
                "experiment_id": str(ordered["experiment_id"].iloc[0]),
                "physical_embryo_id": str(physical_embryo_id),
                "n_rows": len(ordered),
                "n_too_small_rows": int(failures.sum()),
                "too_small_fraction": float(failures.mean()) if len(failures) else np.nan,
                "n_too_small_runs": len(runs),
                "max_consecutive_run": max(run_lengths, default=0),
                "isolated_or_edge_rows": isolated,
                "persistent_run_rows": persistent,
                "neighbor_recovery_rows": neighbor_recovery,
                "median_lower_normalized_distance_when_failed": (
                    float(failed_rows["lower_normalized_distance"].median())
                    if len(failed_rows)
                    else np.nan
                ),
                "minimum_lower_normalized_distance": (
                    float(failed_rows["lower_normalized_distance"].min())
                    if len(failed_rows)
                    else np.nan
                ),
            }
        )
    return out, pd.DataFrame.from_records(track_records)


def summarize_strata(rows: pd.DataFrame) -> pd.DataFrame:
    dimensions = {
        "experiment": "experiment_id",
        "nominal_stage_bin": "stage_bin",
        "genotype_raw": "genotype",
        "chem_perturbation_raw": "chem_perturbation",
        "control_status_raw": "control_status_raw",
        "incubation_temperature_raw": "temperature",
        "source_scope": "source_scope",
        "calibration_status": "calibration_status",
        "calibration_method": "calibration_method",
        "image_micrometers_per_pixel": "image_micrometers_per_pixel",
        "segmentation_backend": "segmentation_backend",
        "segmentation_model": "segmentation_model_id",
        "snapshot_time_series_declared": "acquisition_mode",
        "observed_track_cardinality": "track_observation_mode",
        "surface_qc_applicability": "resolved_surface_area_qc_applicability",
        "shape_class_diagnostic": "shape_class",
    }
    records: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        values = rows[column].astype("string").fillna(NA_LABEL)
        for stratum, index in values.groupby(values, sort=True).groups.items():
            group = rows.loc[index]
            records.append(
                {
                    "dimension": dimension,
                    "stratum": str(stratum),
                    "rows": len(group),
                    "physical_embryos": group["physical_embryo_id"].nunique(),
                    "too_small_rows": int(group["recomputed_too_small"].sum()),
                    "too_large_rows": int(group["recomputed_too_large"].sum()),
                    "sa_outlier_rows": int(group["recomputed_sa_outlier_flag"].sum()),
                    "isolated_sa_only_rows": int(group["isolated_sa_only_failure"].sum()),
                    "current_use_snip_rows": int(group["use_snip"].sum()),
                    "sa_outlier_rate": float(group["recomputed_sa_outlier_flag"].mean()),
                    "too_small_rate": float(group["recomputed_too_small"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def qc_cooccurrence(rows: pd.DataFrame) -> pd.DataFrame:
    sa = rows["recomputed_sa_outlier_flag"].astype(bool)
    records: list[dict[str, Any]] = []
    for flag in QC_FLAG_COLUMNS:
        if flag == "sa_outlier_flag":
            continue
        available = rows[flag].notna()
        if not available.any():
            records.append(
                {
                    "qc_flag": flag,
                    "available_rows": 0,
                    "sa_fail_rows": int(sa.sum()),
                    "cooccurring_rows": 0,
                    "cooccurrence_among_sa_fail": np.nan,
                    "availability_status": "unavailable",
                }
            )
            continue
        parsed = rows.loc[available, flag].astype(bool)
        cooccurring = sa.loc[available] & parsed
        sa_available = int(sa.loc[available].sum())
        records.append(
            {
                "qc_flag": flag,
                "available_rows": int(available.sum()),
                "sa_fail_rows": sa_available,
                "cooccurring_rows": int(cooccurring.sum()),
                "cooccurrence_among_sa_fail": (
                    float(cooccurring.sum() / sa_available) if sa_available else np.nan
                ),
                "availability_status": "available",
            }
        )
    return pd.DataFrame.from_records(records)


def policy_flags(rows: pd.DataFrame) -> dict[str, pd.Series]:
    strict = rows["recomputed_sa_outlier_flag"].astype(bool)
    valid_aspect = rows["aspect_ratio"].notna()
    valid_circularity = rows["circularity"].between(0.0, 1.05, inclusive="both")
    too_small = rows["recomputed_too_small"].astype(bool)
    too_large = rows["recomputed_too_large"].astype(bool)
    # Aspect < 2 is the explicitly documented diagnostic proxy in
    # surface_area_qc_pose_confound.md:27-45.  Circularity >= 0.5 is intentionally labeled an
    # unvalidated diagnostic cutoff, not a proposed production threshold.  Missing geometry retains
    # the strict verdict in both candidates.
    aspect_compact = too_large | (too_small & rows["aspect_ratio"].lt(2.0))
    aspect_compact = aspect_compact.where(valid_aspect, strict)
    compact_round = too_large | (
        too_small & rows["aspect_ratio"].lt(2.0) & rows["circularity"].ge(0.5)
    )
    compact_round = compact_round.where(valid_aspect & valid_circularity, strict)
    return {
        "strict_exclusion_k0.90": strict,
        "lower_diagnostic_upper_strict": too_large,
        "all_surface_area_diagnostic_only": pd.Series(False, index=rows.index),
        "isolated_recovered_dips_diagnostic": too_large
        | (
            too_small
            & rows["too_small_track_class"].ne("isolated_dip_with_recovery")
        ),
        "aspect_compact_lt2_diagnostic": aspect_compact.astype(bool),
        "compact_lt2_circularity_ge0.5_diagnostic": compact_round.astype(bool),
    }


def cohort_policy_summary(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_summary: list[dict[str, Any]] = []
    strata_summary: list[dict[str, Any]] = []
    flags = policy_flags(rows)
    strict_eligible = rows["is_valid_snip"] & ~(
        rows["other_qc_failure"] | flags["strict_exclusion_k0.90"]
    )
    strict_group_eligible = strict_eligible.groupby(rows["physical_embryo_id"]).any()
    for policy, surface_failure in flags.items():
        eligible = rows["is_valid_snip"] & ~(rows["other_qc_failure"] | surface_failure)
        group_eligible = eligible.groupby(rows["physical_embryo_id"]).any()
        policy_summary.append(
            {
                "policy": policy,
                "eligible_rows": int(eligible.sum()),
                "ineligible_rows": int((~eligible).sum()),
                "eligible_physical_embryos": int(group_eligible.sum()),
                "ineligible_physical_embryos": int((~group_eligible).sum()),
                "row_recovery_vs_strict": int((eligible & ~strict_eligible).sum()),
                "row_loss_vs_strict": int((strict_eligible & ~eligible).sum()),
                "physical_embryo_recovery_vs_strict": int(
                    (group_eligible & ~strict_group_eligible).sum()
                ),
                "physical_embryo_loss_vs_strict": int(
                    (strict_group_eligible & ~group_eligible).sum()
                ),
            }
        )
        for dimension, column in (
            ("experiment", "experiment_id"),
            ("nominal_stage_bin", "stage_bin"),
            ("genotype_raw", "genotype"),
            ("chem_perturbation_raw", "chem_perturbation"),
            ("temperature_raw", "temperature"),
            ("segmentation_backend", "segmentation_backend"),
            ("applicability", "resolved_surface_area_qc_applicability"),
        ):
            values = rows[column].astype("string").fillna(NA_LABEL)
            for stratum, index in values.groupby(values, sort=True).groups.items():
                local_eligible = eligible.loc[index]
                local_strict = strict_eligible.loc[index]
                strata_summary.append(
                    {
                        "policy": policy,
                        "dimension": dimension,
                        "stratum": str(stratum),
                        "rows": len(index),
                        "eligible_rows": int(local_eligible.sum()),
                        "row_recovery_vs_strict": int((local_eligible & ~local_strict).sum()),
                        "row_loss_vs_strict": int((local_strict & ~local_eligible).sum()),
                    }
                )
    return pd.DataFrame.from_records(policy_summary), pd.DataFrame.from_records(strata_summary)


def threshold_sweep(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_strict = rows["is_valid_snip"] & ~(
        rows["other_qc_failure"] | rows["recomputed_sa_outlier_flag"]
    )
    reference_groups = reference_strict.groupby(rows["physical_embryo_id"]).any()
    summary: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    for k_lower in K_LOWER_SWEEP:
        too_small = rows["predicted_stage_hpf"].notna() & (
            rows["area_um2"] < k_lower * rows["ref_p5"]
        )
        surface_failure = too_small | rows["recomputed_too_large"]
        eligible = rows["is_valid_snip"] & ~(rows["other_qc_failure"] | surface_failure)
        group_eligible = eligible.groupby(rows["physical_embryo_id"]).any()
        failed_groups = surface_failure.groupby(rows["physical_embryo_id"]).any()
        summary.append(
            {
                "k_lower": k_lower,
                "k_upper_fixed": 1.40,
                "too_small_rows": int(too_small.sum()),
                "too_large_rows": int(rows["recomputed_too_large"].sum()),
                "surface_failed_rows": int(surface_failure.sum()),
                "surface_failed_physical_embryos": int(failed_groups.sum()),
                "eligible_rows": int(eligible.sum()),
                "eligible_physical_embryos": int(group_eligible.sum()),
                "row_recovery_vs_k0.90": int((eligible & ~reference_strict).sum()),
                "row_loss_vs_k0.90": int((reference_strict & ~eligible).sum()),
                "physical_embryo_recovery_vs_k0.90": int(
                    (group_eligible & ~reference_groups).sum()
                ),
                "physical_embryo_loss_vs_k0.90": int((reference_groups & ~group_eligible).sum()),
            }
        )
        for dimension, column in (
            ("experiment", "experiment_id"),
            ("nominal_stage_bin", "stage_bin"),
            ("genotype_raw", "genotype"),
            ("chem_perturbation_raw", "chem_perturbation"),
            ("temperature_raw", "temperature"),
            ("source_scope", "source_scope"),
            ("calibration_status", "calibration_status"),
            ("segmentation_backend", "segmentation_backend"),
            ("snapshot_time_series_declared", "acquisition_mode"),
            ("applicability", "resolved_surface_area_qc_applicability"),
        ):
            values = rows[column].astype("string").fillna(NA_LABEL)
            for stratum, index in values.groupby(values, sort=True).groups.items():
                composition.append(
                    {
                        "k_lower": k_lower,
                        "dimension": dimension,
                        "stratum": str(stratum),
                        "rows": len(index),
                        "too_small_rows": int(too_small.loc[index].sum()),
                        "eligible_rows": int(eligible.loc[index].sum()),
                        "row_recovery_vs_k0.90": int(
                            (eligible.loc[index] & ~reference_strict.loc[index]).sum()
                        ),
                    }
                )
    return pd.DataFrame.from_records(summary), pd.DataFrame.from_records(composition)


def deterministic_key(seed: int, *values: Any) -> str:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def round_robin_boundary(rows: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    candidates = rows.loc[
        rows["is_valid_snip"]
        & rows["predicted_stage_hpf"].notna()
        & ~rows["recomputed_too_large"]
        & rows["review_path_status"].eq("available")
    ].copy()
    candidates["boundary_side"] = np.where(
        candidates["recomputed_too_small"], "below_lower", "above_lower"
    )
    candidates["boundary_absolute_distance"] = candidates["lower_normalized_distance"].abs()
    candidates["persistence_review_group"] = candidates["too_small_track_class"].replace(
        {"not_too_small": "nonfailure_neighbor"}
    )
    candidates["biology_raw"] = (
        "genotype="
        + candidates["genotype"].astype("string").fillna(NA_LABEL)
        + "|chem_perturbation="
        + candidates["chem_perturbation"].astype("string").fillna(NA_LABEL)
    )
    candidates["tie_break"] = [
        deterministic_key(seed, experiment, snip)
        for experiment, snip in zip(candidates["experiment_id"], candidates["snip_id"])
    ]
    candidates = candidates.sort_values(
        ["boundary_absolute_distance", "tie_break"], kind="stable"
    )
    per_side_target = {
        "below_lower": target // 2,
        "above_lower": target - target // 2,
    }
    selected_indices: list[int] = []
    strata_columns = [
        "stage_bin",
        "shape_class",
        "persistence_review_group",
        "biology_raw",
        "experiment_id",
    ]
    for side, side_target in per_side_target.items():
        side_rows = candidates.loc[candidates["boundary_side"].eq(side)]
        queues = {
            key: list(group.index)
            for key, group in side_rows.groupby(strata_columns, sort=True, dropna=False)
        }
        keys = sorted(queues, key=lambda key: deterministic_key(seed, side, *key))
        while len([index for index in selected_indices if candidates.loc[index, "boundary_side"] == side]) < side_target:
            progressed = False
            for key in keys:
                if not queues[key]:
                    continue
                selected_indices.append(queues[key].pop(0))
                progressed = True
                side_count = sum(candidates.loc[index, "boundary_side"] == side for index in selected_indices)
                if side_count >= side_target:
                    break
            if not progressed:
                break
    selected = candidates.loc[selected_indices].copy()
    selected = selected.sort_values(
        ["boundary_side", "boundary_absolute_distance", "tie_break"], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "review_id", [f"SA-{index:03d}" for index in range(1, len(selected) + 1)])
    selected["usable_embryo_label"] = ""
    selected["mask_quality_label"] = ""
    selected["pose_label"] = ""
    selected["reviewer_notes"] = ""
    keep = [
        "review_id",
        "experiment_id",
        "snip_id",
        "physical_embryo_id",
        "time_index",
        "is_valid_snip",
        "asset_key_status",
        "asset_keys_json",
        "processed_snip_path",
        "embryo_mask_snip_path",
        "boundary_side",
        "boundary_absolute_distance",
        "lower_normalized_distance",
        "area_um2",
        "lower_bound_um2",
        "predicted_stage_hpf",
        "stage_bin",
        "genotype",
        "chem_perturbation",
        "control_status_raw",
        "temperature",
        "shape_class",
        "aspect_ratio",
        "circularity",
        "too_small_track_class",
        "too_small_run_length",
        "resolved_surface_area_qc_applicability",
        "qc_fail_reasons",
        "usable_embryo_label",
        "mask_quality_label",
        "pose_label",
        "reviewer_notes",
    ]
    return selected[[column for column in keep if column in selected.columns]]


def resolve_data_path(output_root: Path, value: Any) -> Path | None:
    if pd.isna(value) or str(value) == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else output_root / path


def contact_sheets(boundary: pd.DataFrame, output_root: Path, report_dir: Path) -> list[Path]:
    sheets_dir = report_dir / "boundary_contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    rows_per_page = 20
    thumb_height = 230
    thumb_width = 104
    tile_width = 2 * thumb_width + 16
    tile_height = thumb_height + 52
    columns = 4
    font = ImageFont.load_default()
    for page_start in range(0, len(boundary), rows_per_page):
        page = boundary.iloc[page_start : page_start + rows_per_page]
        sheet = Image.new("RGB", (columns * tile_width, 5 * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, (_, row) in enumerate(page.iterrows()):
            x = (offset % columns) * tile_width
            y = (offset // columns) * tile_height
            image_path = resolve_data_path(output_root, row.get("processed_snip_path"))
            mask_path = resolve_data_path(output_root, row.get("embryo_mask_snip_path"))
            try:
                if image_path is None or not image_path.is_file():
                    raise FileNotFoundError(str(image_path))
                image = Image.open(image_path).convert("L")
                image.thumbnail((thumb_width, thumb_height), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (thumb_width, thumb_height), "black")
                canvas.paste(image.convert("RGB"), ((thumb_width - image.width) // 2, 0))
                sheet.paste(canvas, (x, y))
                overlay = canvas.copy()
                if mask_path is not None and mask_path.is_file():
                    mask = Image.open(mask_path).convert("L")
                    mask.thumbnail((thumb_width, thumb_height), Image.Resampling.NEAREST)
                    mask_canvas = Image.new("L", (thumb_width, thumb_height), 0)
                    mask_canvas.paste(mask, ((thumb_width - mask.width) // 2, 0))
                    red = Image.new("RGB", overlay.size, (255, 0, 0))
                    overlay = Image.blend(overlay, Image.composite(red, overlay, mask_canvas), 0.35)
                sheet.paste(overlay, (x + thumb_width + 4, y))
            except Exception:
                draw.rectangle((x, y, x + 2 * thumb_width, y + thumb_height), outline="red", width=2)
                draw.text((x + 4, y + 4), "asset unavailable", fill="red", font=font)
            label = (
                f"{row['review_id']} {row['boundary_side']}\n"
                f"stage={row['stage_bin']} shape={row['shape_class']}\n"
                f"track={row['too_small_track_class']}"
            )
            draw.multiline_text((x + 2, y + thumb_height + 2), label, fill="black", font=font, spacing=1)
        page_number = page_start // rows_per_page + 1
        output = sheets_dir / f"boundary_contact_sheet_{page_number:02d}.png"
        sheet.save(output, optimize=True)
        outputs.append(output)
    return outputs


def write_figures(rows: pd.DataFrame, sweep: pd.DataFrame, track: pd.DataFrame, report_dir: Path) -> None:
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    by_stage = (
        rows.groupby("stage_bin", observed=False)[["recomputed_too_small", "recomputed_too_large"]]
        .sum()
        .reindex(STAGE_LABELS)
    )
    ax = by_stage.plot(kind="bar", figsize=(9, 5), color=["#4477AA", "#CC6677"])
    ax.set_xlabel("Nominal predicted_stage_hpf bin")
    ax.set_ylabel("Failed rows")
    ax.set_title("Surface-area failure direction by nominal stage")
    plt.tight_layout()
    plt.savefig(figures / "failure_direction_by_nominal_stage.png", dpi=160)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep["k_lower"], sweep["eligible_rows"], marker="o", label="eligible rows")
    ax1.set_xlabel("Offline k_lower")
    ax1.set_ylabel("Rows eligible after all QC")
    ax2 = ax1.twinx()
    ax2.plot(
        sweep["k_lower"],
        sweep["eligible_physical_embryos"],
        marker="s",
        color="#CC6677",
        label="eligible physical embryos",
    )
    ax2.set_ylabel("Physical embryos with >=1 eligible row")
    ax1.set_title("Offline lower-bound sweep; k_upper fixed at 1.4")
    fig.tight_layout()
    fig.savefig(figures / "threshold_sweep.png", dpi=160)
    plt.close(fig)

    failed_tracks = track.loc[track["n_too_small_rows"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        failed_tracks["max_consecutive_run"],
        bins=np.arange(0.5, max(2.5, failed_tracks["max_consecutive_run"].max() + 1.5)),
        color="#228833",
    )
    ax.set_xlabel("Maximum consecutive too-small run length")
    ax.set_ylabel("Physical embryos")
    ax.set_title("Persistence of lower-bound failures")
    fig.tight_layout()
    fig.savefig(figures / "too_small_run_lengths.png", dpi=160)
    plt.close(fig)


def source_inventory(
    experiments: Sequence[str], output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for order, experiment_id in enumerate(experiments):
        paths = artifact_paths(output_root, experiment_id)
        present = {name: path.is_file() for name, path in paths.items()}
        for name, path in paths.items():
            record: dict[str, Any] = {
                "experiment_order": order,
                "experiment_id": experiment_id,
                "source": name,
                "path": str(path),
                "present": present[name],
                "size_bytes": path.stat().st_size if present[name] else pd.NA,
                "mtime_utc": (
                    pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
                    if present[name]
                    else pd.NA
                ),
                "sha256": sha256_file(path) if present[name] else pd.NA,
            }
            if present[name]:
                try:
                    sample = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    record["rows"] = len(sample)
                    record["columns_json"] = json.dumps(list(sample.columns), separators=(",", ":"))
                    record["schema_signature"] = schema_signature(sample.columns)
                except Exception as error:
                    record["read_error"] = f"{type(error).__name__}: {error}"
            records.append(record)
        stored_available = present["surface_area_qc"] or present["snip_qc"]
        core_ready = all(present[source] for source in CORE_REQUIRED_SOURCES)
        coverage.append(
            {
                "experiment_order": order,
                "experiment_id": experiment_id,
                **{f"{name}_present": status for name, status in present.items()},
                "core_inputs_present": core_ready,
                "stored_sa_source_present": stored_available,
                "evaluable": core_ready and stored_available,
            }
        )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(coverage)


def metadata(config: StudyConfig, experiments: Sequence[str], runtime_s: float) -> dict[str, Any]:
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        git_head = NA_LABEL
        git_branch = NA_LABEL
    authority_bytes = config.experiments_file.read_bytes()
    return {
        "measurement_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "runtime_seconds": runtime_s,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": git_head,
        "git_branch": git_branch,
        "output_root": str(config.output_root),
        "experiments_file": str(config.experiments_file),
        "experiments_file_sha256": hashlib.sha256(authority_bytes).hexdigest(),
        "experiment_column": config.experiment_column,
        "ordered_experiment_count": len(experiments),
        "reference_csv": str(config.reference_csv),
        "reference_sha256": sha256_file(config.reference_csv),
        "report_dir": str(config.report_dir),
        "k_lower": config.k_lower,
        "k_upper": config.k_upper,
        "k_lower_sweep": list(K_LOWER_SWEEP),
        "boundary_size_requested": config.boundary_size,
        "seed": config.seed,
        "predicted_stage_interpretation": "nominal_pipeline_axis_not_validated_biological_truth",
        "counterfactual_scope": "offline_derived_outputs_only",
    }


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)


def run(config: StudyConfig) -> None:
    started = time.monotonic()
    config.report_dir.mkdir(parents=True, exist_ok=True)
    experiments = ordered_experiments(config.experiments_file, config.experiment_column)
    reference = reference_table(config.reference_csv)
    sources, coverage = source_inventory(experiments, config.output_root)
    write_csv(sources, config.report_dir / "source_inventory.csv")
    write_csv(coverage, config.report_dir / "experiment_coverage.csv")
    write_csv(
        pd.DataFrame(
            {"experiment_order": range(len(experiments)), "experiment_id": experiments}
        ),
        config.report_dir / "ordered_experiments.csv",
    )

    frames: list[pd.DataFrame] = []
    parity_records: list[dict[str, Any]] = []
    for row in coverage.loc[coverage["evaluable"]].itertuples(index=False):
        paths = artifact_paths(config.output_root, row.experiment_id)
        frame, summary = prepare_experiment(
            row.experiment_id,
            paths,
            reference,
            k_lower=config.k_lower,
            k_upper=config.k_upper,
        )
        frames.append(frame)
        parity_records.append(summary)
        print(
            f"{row.experiment_id}: rows={len(frame):,} "
            f"parity_mismatches={summary['parity_mismatches']:,}"
        )
    if not frames:
        raise RuntimeError("No evaluable experiments had all explicit minimum sources.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        rows = pd.concat(frames, ignore_index=True, sort=False)
    rows, track = annotate_track_persistence(rows)

    parity = pd.DataFrame.from_records(parity_records)
    mismatches = rows.loc[
        ~rows["parity_match"],
        [
            "experiment_id",
            "snip_id",
            "stored_flag_source",
            "stored_sa_outlier_flag",
            "recomputed_sa_outlier_flag",
            "recomputed_too_small",
            "recomputed_too_large",
            "area_um2",
            "predicted_stage_hpf",
            "ref_p5",
            "ref_p95",
            "lower_bound_um2",
            "upper_bound_um2",
        ],
    ]
    write_csv(parity, config.report_dir / "parity_summary.csv")
    write_csv(mismatches, config.report_dir / "parity_mismatches.csv")
    write_csv(track, config.report_dir / "physical_embryo_track_persistence.csv")
    write_csv(summarize_strata(rows), config.report_dir / "stratified_failure_summary.csv")
    write_csv(qc_cooccurrence(rows), config.report_dir / "qc_flag_cooccurrence.csv")
    materialization_use = (
        rows.groupby(["is_valid_snip", "use_snip"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    write_csv(
        materialization_use,
        config.report_dir / "materialization_use_crosstab.csv",
    )
    invalid_by_experiment = (
        rows.loc[~rows["is_valid_snip"]]
        .groupby("experiment_id", sort=True)
        .agg(rows=("snip_id", "size"), use_snip_rows=("use_snip", "sum"))
        .reset_index()
    )
    write_csv(
        invalid_by_experiment,
        config.report_dir / "invalid_materializations_by_experiment.csv",
    )

    policies, policy_strata = cohort_policy_summary(rows)
    sweep, sweep_strata = threshold_sweep(rows)
    write_csv(policies, config.report_dir / "counterfactual_policy_summary.csv")
    write_csv(policy_strata, config.report_dir / "counterfactual_policy_strata.csv")
    write_csv(sweep, config.report_dir / "threshold_sweep.csv")
    write_csv(sweep_strata, config.report_dir / "threshold_sweep_strata.csv")

    boundary = round_robin_boundary(rows, config.boundary_size, config.seed)
    write_csv(boundary, config.report_dir / "boundary_review_set.csv")
    sheets = (
        contact_sheets(boundary, config.output_root, config.report_dir)
        if config.write_contact_sheets
        else []
    )

    sensitivity_columns = [
        "experiment_id",
        "snip_id",
        "physical_embryo_id",
        "well_id",
        "time_index",
        "is_valid_snip",
        "area_um2",
        "predicted_stage_hpf",
        "model_version",
        "stage_prediction_status",
        "stage_bin",
        "ref_p5",
        "ref_p50",
        "ref_p95",
        "lower_bound_um2",
        "upper_bound_um2",
        "stored_sa_outlier_flag",
        "recomputed_sa_outlier_flag",
        "recomputed_too_small",
        "recomputed_too_large",
        "lower_normalized_distance",
        "upper_normalized_distance",
        "resolved_surface_area_qc_applicability",
        "applicability_source",
        "track_observation_mode",
        "too_small_track_class",
        "too_small_run_length",
        "genotype",
        "chem_perturbation",
        "temperature",
        "source_scope",
        "calibration_status",
        "segmentation_backend",
        "segmentation_model_id",
    ]
    sensitivity = rows[[column for column in sensitivity_columns if column in rows.columns]].copy()
    for column in (
        "experiment_id",
        "snip_id",
        "physical_embryo_id",
        "well_id",
        "model_version",
        "stage_prediction_status",
        "stage_bin",
        "resolved_surface_area_qc_applicability",
        "applicability_source",
        "track_observation_mode",
        "too_small_track_class",
        "genotype",
        "chem_perturbation",
        "source_scope",
        "calibration_status",
        "segmentation_backend",
        "segmentation_model_id",
    ):
        if column in sensitivity.columns:
            sensitivity[column] = sensitivity[column].astype("string")
    sensitivity.to_parquet(config.report_dir / "stage_axis_sensitivity_input.parquet", index=False)
    write_figures(rows, sweep, track, config.report_dir)

    failure_summary = pd.DataFrame(
        [
            {
                "rows": len(rows),
                "physical_embryos": rows["physical_embryo_id"].nunique(),
                "too_small_rows": int(rows["recomputed_too_small"].sum()),
                "too_large_rows": int(rows["recomputed_too_large"].sum()),
                "surface_outlier_rows": int(rows["recomputed_sa_outlier_flag"].sum()),
                "surface_outlier_physical_embryos": int(
                    rows.groupby("physical_embryo_id")["recomputed_sa_outlier_flag"].any().sum()
                ),
                "isolated_surface_only_rows": int(rows["isolated_sa_only_failure"].sum()),
                "valid_materialized_rows": int(rows["is_valid_snip"].sum()),
                "invalid_materialized_rows": int((~rows["is_valid_snip"]).sum()),
                "current_use_snip_rows": int(rows["use_snip"].sum()),
                "parity_mismatches": int((~rows["parity_match"]).sum()),
                "boundary_review_rows": len(boundary),
                "contact_sheet_pages": len(sheets),
            }
        ]
    )
    write_csv(failure_summary, config.report_dir / "overall_failure_summary.csv")
    elapsed = time.monotonic() - started
    (config.report_dir / "study_metadata.json").write_text(
        json.dumps(metadata(config, experiments, elapsed), indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote study outputs to {config.report_dir} in {elapsed:.1f} s")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = StudyConfig(
        output_root=args.output_root.resolve(),
        experiments_file=args.experiments_file.resolve(),
        experiment_column=args.experiment_column,
        reference_csv=args.reference_csv.resolve(),
        report_dir=args.report_dir.resolve(),
        k_lower=float(args.k_lower),
        k_upper=float(args.k_upper),
        boundary_size=int(args.boundary_size),
        seed=int(args.seed),
        write_contact_sheets=not args.no_contact_sheets,
    )
    if config.k_lower <= 0 or config.k_upper <= 0:
        raise ValueError("k_lower and k_upper must be positive.")
    if config.boundary_size <= 0:
        raise ValueError("boundary_size must be positive.")
    run(config)


if __name__ == "__main__":
    main()
