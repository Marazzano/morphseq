"""Materialize reconciled SeaHub detections as canonical one-embryo wells.

This module owns the boundary between the SeaHub source corpus and morphseq's
drop-in front end.  A SeaHub FOV is never treated as a well: detection must
produce exactly eight positions, and each detected embryo is assigned its own
canonical well in a deterministic operational shard.

The shard is only a batching container.  Stable biological provenance is carried
by ``source_embryo_id`` and the source metadata copied onto every plate row.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageOps

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    validate_frame_inventory_identity_contract,
)
from data_pipeline.acquisition.metadata_ingest.frame_inventory.frame_inventory_validation import (
    validate_frame_inventory,
)
from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract import (
    validate_plate_metadata,
)
from data_pipeline.shared.identifiers import (
    build_image_id,
    build_well_id,
    normalize_well_index,
)
from data_pipeline.shared.identifiers.constructors import sanitize_experiment_id

from .reconciliation import PASS_THROUGH_FAILURE_STATUSES, apply_inclusion_policy

EXPECTED_EMBRYOS_PER_FOV = 8
WELLS_PER_SHARD = 96
_BBOX_COLUMNS = ("crop_x1_px", "crop_y1_px", "crop_x2_px", "crop_y2_px")
_DETECTION_CARRY_COLUMNS = (
    "embryo_position",
    *_BBOX_COLUMNS,
    "detection_confidence",
    "detection_phrase",
    "segmentation_qc_status",
    "box_threshold",
    "text_threshold",
    "raw_detection_count",
    "nms_detection_count",
    "selected_detection_count",
)


@dataclass(frozen=True)
class SeaHubIntegrationConfig:
    """Frozen operational choices for a SeaHub integration bundle."""

    operational_date: str = "20260723"
    micrometers_per_pixel: float = 7.8
    calibration_status: str = "placeholder"
    jpeg_quality: int = 95
    canvas_rounding_px: int = 32
    canvas_fill_value: int = 0
    flip_polarity: bool = True
    overwrite_images: bool = False
    temperature_c: float = 28.5
    medium: str = "E3"

    def __post_init__(self) -> None:
        if re.fullmatch(r"\d{8}", str(self.operational_date)) is None:
            raise ValueError("operational_date must be an 8-digit YYYYMMDD token.")
        if float(self.micrometers_per_pixel) <= 0:
            raise ValueError("micrometers_per_pixel must be > 0.")
        if self.calibration_status != "placeholder":
            raise ValueError(
                "SeaHub currently has no measured calibration; calibration_status "
                "must remain 'placeholder' until a calibration is supplied."
            )
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100].")
        if int(self.canvas_rounding_px) < 1:
            raise ValueError("canvas_rounding_px must be >= 1.")
        if not 0 <= int(self.canvas_fill_value) <= 255:
            raise ValueError("canvas_fill_value must be in [0, 255].")


@dataclass(frozen=True)
class SeaHubBundleResult:
    """Tables and paths emitted by :func:`build_seahub_dropin_bundle`."""

    embryo_ingest: pd.DataFrame
    well_provenance: pd.DataFrame
    detection_failures: pd.DataFrame
    dropped_fovs: pd.DataFrame
    experiment_manifest: pd.DataFrame
    canvas_width_px: int | None
    canvas_height_px: int | None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in {"", "nan", "none", "#no match"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _source_fov_ids(df: pd.DataFrame, *, label: str) -> pd.Series:
    if "source_fov_id" in df.columns:
        source = df["source_fov_id"]
    elif "image_id" in df.columns:
        source = df["image_id"]
    else:
        raise ValueError(
            f"{label} must contain source_fov_id or the legacy source image_id."
        )
    if source.isna().any() or source.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} has null/empty source FOV identities.")
    return source.astype(str)


def _failure_record(
    fov: pd.Series,
    *,
    reason: str,
    detection_count: int,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "source_fov_id": str(fov["source_fov_id"]),
        "source_experiment_id": fov.get("experiment_id"),
        "relative_path": fov.get("relative_path"),
        "filename": fov.get("filename"),
        "detection_count": int(detection_count),
        "failure_reason": reason,
        "failure_detail": detail,
    }


def build_embryo_ingest(
    reconciled_fovs: pd.DataFrame,
    detection_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detected embryo rows and an explicit FOV-level detection failure table.

    Reconciliation policy is evaluated before detection.  Included FOVs must have
    exactly one row for each position 1..8 and a passing detection status; otherwise
    the complete FOV is withheld so no partial pseudo-wells can leak downstream.
    """
    policy = (
        reconciled_fovs.copy()
        if {"include_for_seahub", "exclusion_reason"}.issubset(reconciled_fovs.columns)
        else apply_inclusion_policy(reconciled_fovs)
    )
    included = policy[policy["include_for_seahub"].astype(bool)].copy()
    included["source_fov_id"] = _source_fov_ids(
        included, label="included reconciled_fovs"
    )
    duplicate_fovs = included["source_fov_id"].duplicated(keep=False)
    if duplicate_fovs.any():
        offenders = sorted(
            included.loc[duplicate_fovs, "source_fov_id"].unique()
        )[:5]
        raise ValueError(
            "included reconciled_fovs must contain one row per source FOV; "
            f"duplicate IDs include {offenders}. Inclusion policy should have "
            "classified these rows as contract-invalid."
        )

    detections = detection_manifest.copy()
    detections["source_fov_id"] = _source_fov_ids(
        detections, label="detection_manifest"
    )
    missing_bbox = [column for column in _BBOX_COLUMNS if column not in detections.columns]
    if missing_bbox:
        raise ValueError(
            f"detection_manifest is missing crop box column(s): {missing_bbox}."
        )
    if "embryo_position" not in detections.columns:
        raise ValueError("detection_manifest is missing embryo_position.")

    detection_groups = {
        str(source_fov_id): group.copy()
        for source_fov_id, group in detections.groupby("source_fov_id", sort=False)
    }
    embryos: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for _, fov in included.iterrows():
        source_fov_id = str(fov["source_fov_id"])
        group = detection_groups.get(source_fov_id)
        if group is None:
            failures.append(
                _failure_record(
                    fov,
                    reason="missing_detection",
                    detection_count=0,
                )
            )
            continue

        positions = pd.to_numeric(group["embryo_position"], errors="coerce")
        boxes = group[list(_BBOX_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        expected_positions = set(range(1, EXPECTED_EMBRYOS_PER_FOV + 1))
        actual_positions = set(positions.dropna().astype(int))
        detail: str | None = None
        reason: str | None = None
        if len(group) != EXPECTED_EMBRYOS_PER_FOV:
            reason = "detection_count_not_8"
            detail = f"found {len(group)} detection rows"
        elif (
            positions.isna().any()
            or not positions.mod(1).eq(0).all()
            or actual_positions != expected_positions
        ):
            reason = "invalid_embryo_positions"
            detail = f"positions={positions.tolist()}"
        elif boxes.isna().any().any():
            reason = "invalid_detection_box"
            detail = "one or more crop coordinates are null/non-numeric"
        elif (
            boxes["crop_x2_px"].le(boxes["crop_x1_px"]).any()
            or boxes["crop_y2_px"].le(boxes["crop_y1_px"]).any()
            or boxes["crop_x1_px"].lt(0).any()
            or boxes["crop_y1_px"].lt(0).any()
            or (
                not _is_missing(fov.get("width_px"))
                and boxes["crop_x2_px"].gt(float(fov["width_px"])).any()
            )
            or (
                not _is_missing(fov.get("height_px"))
                and boxes["crop_y2_px"].gt(float(fov["height_px"])).any()
            )
        ):
            reason = "invalid_detection_box"
            detail = "one or more crop boxes are empty or outside the source frame"
        elif "segmentation_qc_status" in group.columns and not group[
            "segmentation_qc_status"
        ].astype(str).str.casefold().eq("pass").all():
            reason = "detection_qc_failed"
            detail = ",".join(
                sorted(group["segmentation_qc_status"].dropna().astype(str).unique())
            )
        elif "selected_detection_count" in group.columns:
            selected = pd.to_numeric(
                group["selected_detection_count"], errors="coerce"
            )
            if selected.isna().any() or not selected.eq(EXPECTED_EMBRYOS_PER_FOV).all():
                reason = "selected_detection_count_not_8"
                detail = f"values={sorted(selected.dropna().unique().tolist())}"

        if reason is not None:
            failures.append(
                _failure_record(
                    fov,
                    reason=reason,
                    detection_count=len(group),
                    detail=detail,
                )
            )
            continue

        fov_payload = fov.to_dict()
        source_image_path = fov_payload.pop("image_path", None)
        fov_payload.pop("image_id", None)
        fov_payload["source_image_path"] = source_image_path
        fov_payload["source_fov_id"] = source_fov_id
        fov_payload["source_experiment_id"] = fov_payload.get("experiment_id")
        for _, detection in group.sort_values("embryo_position").iterrows():
            record = dict(fov_payload)
            for column in _DETECTION_CARRY_COLUMNS:
                if column in detection.index:
                    record[column] = detection[column]
            position = int(detection["embryo_position"])
            record["embryo_position"] = position
            record["source_embryo_id"] = (
                f"seahub_{source_fov_id}_p{position:02d}"
            )
            record["reconciliation_failure_passed_through"] = str(
                record.get("metadata_match_status", "")
            ) in PASS_THROUGH_FAILURE_STATUSES
            embryos.append(record)

    embryo_df = pd.DataFrame.from_records(embryos)
    failure_df = pd.DataFrame.from_records(
        failures,
        columns=[
            "source_fov_id",
            "source_experiment_id",
            "relative_path",
            "filename",
            "detection_count",
            "failure_reason",
            "failure_detail",
        ],
    )
    if not embryo_df.empty and embryo_df["source_embryo_id"].duplicated().any():
        raise ValueError("source_embryo_id must be globally unique.")
    return embryo_df, failure_df


def _well_index_for_offset(offset: int) -> str:
    row = chr(ord("A") + int(offset) // 12)
    column = int(offset) % 12 + 1
    return normalize_well_index(row, column)


def assign_operational_identity(
    embryo_ingest: pd.DataFrame,
    *,
    config: SeaHubIntegrationConfig,
) -> pd.DataFrame:
    """Pack embryos deterministically into neutral operational shards."""
    if embryo_ingest.empty:
        return embryo_ingest.copy()
    required = {
        "source_experiment_id",
        "source_fov_id",
        "source_embryo_id",
        "embryo_position",
    }
    missing = sorted(required - set(embryo_ingest.columns))
    if missing:
        raise ValueError(f"embryo_ingest is missing identity source columns {missing}.")

    output_groups: list[pd.DataFrame] = []
    for source_experiment, group in embryo_ingest.groupby(
        "source_experiment_id", sort=True
    ):
        ordered = group.copy()
        ordered["_stage_sort"] = pd.to_numeric(
            ordered.get("stage_hpf"), errors="coerce"
        ).fillna(float("inf"))
        ordered["_condition_sort"] = (
            ordered.get("perturbation_key", pd.Series(index=ordered.index, dtype=object))
            .fillna("")
            .astype(str)
        )
        ordered["_fov_sort"] = (
            ordered.get("fov_label", ordered["source_fov_id"])
            .fillna("")
            .astype(str)
        )
        ordered = ordered.sort_values(
            [
                "_stage_sort",
                "_condition_sort",
                "_fov_sort",
                "source_fov_id",
                "embryo_position",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        canonical_rows: list[dict[str, Any]] = []
        for offset, row in ordered.iterrows():
            shard_index = offset // WELLS_PER_SHARD + 1
            shard_offset = offset % WELLS_PER_SHARD
            experiment_id = sanitize_experiment_id(
                f"{config.operational_date}_seahub_{source_experiment}"
                f"_shard{shard_index:03d}"
            )
            well_index = _well_index_for_offset(shard_offset)
            well_id = build_well_id(experiment_id, well_index)
            image_id = build_image_id(well_id, "BF", 0)
            record = row.drop(
                labels=[
                    "_stage_sort",
                    "_condition_sort",
                    "_fov_sort",
                ]
            ).to_dict()
            record.update(
                {
                    "experiment_id": experiment_id,
                    "operational_shard_index": shard_index,
                    "well_index": well_index,
                    "well_id": well_id,
                    "channel_id": "BF",
                    "time_index": 0,
                    "image_id": image_id,
                }
            )
            canonical_rows.append(record)
        output_groups.append(pd.DataFrame.from_records(canonical_rows))

    output = pd.concat(output_groups, ignore_index=True)
    for column in ("source_embryo_id", "well_id", "image_id"):
        if output[column].duplicated().any():
            raise ValueError(f"assigned identity column {column!r} is not unique.")
    return output


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(int(value) / int(multiple)) * int(multiple))


def _canvas_shape(
    embryos: pd.DataFrame, *, rounding_px: int
) -> tuple[int, int]:
    widths = pd.to_numeric(embryos["crop_x2_px"]) - pd.to_numeric(
        embryos["crop_x1_px"]
    )
    heights = pd.to_numeric(embryos["crop_y2_px"]) - pd.to_numeric(
        embryos["crop_y1_px"]
    )
    if (widths <= 0).any() or (heights <= 0).any():
        raise ValueError("All SeaHub crop boxes must have positive width and height.")
    return (
        _round_up(int(widths.max()), rounding_px),
        _round_up(int(heights.max()), rounding_px),
    )


def _relative_materialized_path(row: pd.Series) -> Path:
    return (
        Path("images")
        / str(row["well_id"])
        / "BF"
        / "projection"
        / "focus_stack"
        / f"{row['image_id']}.jpg"
    )


def _materialize_experiment_images(
    embryos: pd.DataFrame,
    *,
    experiment_root: Path,
    canvas_width_px: int,
    canvas_height_px: int,
    config: SeaHubIntegrationConfig,
) -> None:
    """Crop each source FOV once, then write its eight one-embryo frames."""
    for source_image_path, group in embryos.groupby(
        "source_image_path", sort=False, dropna=False
    ):
        if _is_missing(source_image_path):
            raise ValueError(
                f"source_image_path is missing for FOV {group.iloc[0]['source_fov_id']}."
            )
        source_path = Path(str(source_image_path))
        if not source_path.is_file():
            raise FileNotFoundError(f"SeaHub source image does not exist: {source_path}")
        with Image.open(source_path) as source:
            grayscale = source.convert("L")
            for _, row in group.iterrows():
                x1, y1, x2, y2 = (int(row[column]) for column in _BBOX_COLUMNS)
                if x1 < 0 or y1 < 0 or x2 > grayscale.width or y2 > grayscale.height:
                    raise ValueError(
                        f"Crop for {row['source_embryo_id']} falls outside source image "
                        f"{source_path}: {(x1, y1, x2, y2)} vs "
                        f"{grayscale.width}x{grayscale.height}."
                    )
                crop = grayscale.crop((x1, y1, x2, y2))
                if config.flip_polarity:
                    crop = ImageOps.invert(crop)
                canvas = Image.new(
                    "L",
                    (canvas_width_px, canvas_height_px),
                    color=int(config.canvas_fill_value),
                )
                left = (canvas_width_px - crop.width) // 2
                top = (canvas_height_px - crop.height) // 2
                canvas.paste(crop, (left, top))
                output_path = experiment_root / _relative_materialized_path(row)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists() and not config.overwrite_images:
                    with Image.open(output_path) as existing:
                        if existing.mode != "L" or existing.size != (
                            canvas_width_px,
                            canvas_height_px,
                        ):
                            raise ValueError(
                                f"Existing materialized image {output_path} does not "
                                "match the expected grayscale canvas "
                                f"{canvas_width_px}x{canvas_height_px}. Rerun with "
                                "overwrite_images=True after reviewing the stale file."
                            )
                    continue
                canvas.save(
                    output_path,
                    format="JPEG",
                    quality=int(config.jpeg_quality),
                )


def _frame_inventory(
    embryos: pd.DataFrame,
    *,
    canvas_width_px: int,
    canvas_height_px: int,
    config: SeaHubIntegrationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, embryo in embryos.iterrows():
        rows.append(
            {
                "experiment_id": embryo["experiment_id"],
                "well_index": embryo["well_index"],
                "well_id": embryo["well_id"],
                "channel_id": "BF",
                "time_index": 0,
                "z_index": pd.NA,
                "image_id": embryo["image_id"],
                "image_product_type": "projection",
                "projection_method": "focus_stack",
                "elapsed_time_s": 0.0,
                "acquisition_time_s": 0.0,
                "image_path": str(_relative_materialized_path(embryo)),
                "image_micrometers_per_pixel": float(
                    config.micrometers_per_pixel
                ),
                "image_width_px": int(canvas_width_px),
                "image_height_px": int(canvas_height_px),
                "orientation": "none",
                "image_file_format": "jpg",
                "pixel_dtype": "uint8",
                "downsample_factor": 1.0,
                "downsample_method": "none",
                "jpeg_quality": int(config.jpeg_quality),
                "flip_polarity": bool(config.flip_polarity),
                "source_scope": "seahub",
                "image_kind": "single_z",
                "z_position": pd.NA,
                "calibration_status": config.calibration_status,
                "canvas_fill_value": int(config.canvas_fill_value),
            }
        )
    return pd.DataFrame.from_records(rows)


def _first_present(row: pd.Series, *columns: str) -> Any:
    for column in columns:
        if column in row.index and not _is_missing(row[column]):
            return row[column]
    return pd.NA


def _plate_metadata(
    embryos: pd.DataFrame,
    *,
    config: SeaHubIntegrationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, embryo in embryos.iterrows():
        domain = str(embryo.get("perturbation_domain", "")).casefold()
        perturbation = embryo.get("perturbation_parsed")
        is_chemical = domain == "chemical" or str(
            embryo["source_experiment_id"]
        ).upper().startswith("CHEM")
        collection_name = _first_present(
            embryo, "metadata_collection_name", "collection_name"
        )
        row = {
            "experiment_id": embryo["experiment_id"],
            "well_id": embryo["well_id"],
            "well_index": embryo["well_index"],
            "genotype": "ctrl" if is_chemical else perturbation,
            "chem_perturbation": perturbation if is_chemical else pd.NA,
            "start_age_hpf": embryo.get("stage_hpf"),
            "temperature": float(config.temperature_c),
            "medium": config.medium,
            "stage_hpf": embryo.get("stage_hpf"),
            "stage_source_label": embryo.get("stage_source_label"),
            "stage_normalized_label": embryo.get("stage_normalized_label"),
            "stage_match_method": embryo.get("stage_match_method"),
            "stage_match_delta_hpf": embryo.get("stage_match_delta_hpf"),
            "stage_addition_hpf": embryo.get("stage_addition_hpf"),
            "stage_addition_source_label": embryo.get(
                "stage_addition_source_label"
            ),
            "perturbation": perturbation,
            "perturbation_key": embryo.get("perturbation_key"),
            "perturbation_domain": embryo.get("perturbation_domain"),
            "embryos_per_well": 1,
            "source_scope": "seahub",
            "source_experiment_id": embryo["source_experiment_id"],
            "source_fov_id": embryo["source_fov_id"],
            "source_embryo_id": embryo["source_embryo_id"],
            "fov_label": embryo.get("fov_label"),
            "fov_position": int(embryo["embryo_position"]),
            "source_filename": embryo.get("filename"),
            "source_relative_path": embryo.get("relative_path"),
            "image_kind": "single_z",
            "z_position": pd.NA,
            "micrometers_per_pixel": float(config.micrometers_per_pixel),
            "calibration_status": config.calibration_status,
            "calibration_issue": "SeaHub pixel calibration is unverified; revisit 7.8 um/px.",
            "operational_integration_date": config.operational_date,
            "collection_date": _first_present(
                embryo, "metadata_collection_date", "collection_date"
            ),
            "collection_name": collection_name,
            "collection_batch": _first_present(
                embryo, "metadata_collection_batch", "collection_batch"
            ),
            "collection_expt": _first_present(
                embryo, "metadata_expt", "collection_expt"
            ),
            "collection_ancestors": _first_present(
                embryo, "metadata_ancestors", "collection_ancestors"
            ),
            "has_seq_link": not _is_missing(collection_name),
            "metadata_match_status": embryo.get("metadata_match_status"),
            "metadata_match_score": embryo.get("metadata_match_score"),
            "metadata_candidate_count": embryo.get("metadata_candidate_count"),
            "metadata_row_number": embryo.get("metadata_row_number"),
            "metadata_experiment_original": embryo.get(
                "metadata_experiment_original"
            ),
            "metadata_experiment_effective": embryo.get(
                "metadata_experiment_effective"
            ),
            "metadata_experiment_corrected": embryo.get(
                "metadata_experiment_corrected", False
            ),
            "metadata_experiment_correction_source": embryo.get(
                "metadata_experiment_correction_source"
            ),
            "reconciliation_failure_passed_through": embryo.get(
                "reconciliation_failure_passed_through", False
            ),
        }
        rows.append(row)
    plate = pd.DataFrame.from_records(rows)
    validate_plate_metadata(plate, scope_label="seahub_plate_metadata")
    return plate


def _runtime_overlay(
    *,
    experiment_id: str,
    frame_inventory_csv: Path,
    plate_metadata_csv: Path,
    image_root: Path,
) -> dict[str, Any]:
    return {
        "experiments": [str(experiment_id)],
        "microscope": "SeaHub",
        "front_end": {"mode": "dropin"},
        "dropin": {
            "frame_inventory_csv": str(frame_inventory_csv.resolve()),
            "plate_metadata_csv": str(plate_metadata_csv.resolve()),
            "image_root": str(image_root.resolve()),
        },
        "image_materialization": {
            "products": [
                {
                    "channel_id": "BF",
                    "image_product_type": "projection",
                    "projection_method": "focus_stack",
                }
            ]
        },
        # SeaHub shards contain up to 96 one-frame embryo wells. Keep the canonical
        # per-well products, but amortize model initialization across the shard.
        "frame_detections": {"use_model_server": True},
        "frame_masks": {"use_model_server": True},
        "unet_snip": {"use_model_server": True},
    }


def build_seahub_dropin_bundle(
    reconciled_fovs: pd.DataFrame,
    detection_manifest: pd.DataFrame,
    *,
    output_root: str | Path,
    config: SeaHubIntegrationConfig | None = None,
    materialize_images: bool = True,
) -> SeaHubBundleResult:
    """Build and validate a complete per-experiment SeaHub drop-in bundle."""
    config = config or SeaHubIntegrationConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    policy = apply_inclusion_policy(reconciled_fovs)
    dropped = policy[~policy["include_for_seahub"].astype(bool)].copy()
    embryo_ingest, detection_failures = build_embryo_ingest(
        policy, detection_manifest
    )
    assigned = assign_operational_identity(embryo_ingest, config=config)

    integration_dir = output_root / "integration"
    integration_dir.mkdir(parents=True, exist_ok=True)
    embryo_ingest.to_csv(integration_dir / "embryo_ingest.csv", index=False)
    detection_failures.to_csv(
        integration_dir / "detection_failures.csv", index=False
    )
    dropped.to_csv(integration_dir / "dropped_fovs.csv", index=False)

    if assigned.empty:
        empty_manifest = pd.DataFrame(
            columns=[
                "experiment_id",
                "frame_inventory_csv",
                "plate_metadata_csv",
                "image_root",
                "runtime_config_yaml",
                "embryo_count",
            ]
        )
        empty_manifest.to_csv(
            integration_dir / "experiment_manifest.csv", index=False
        )
        return SeaHubBundleResult(
            embryo_ingest=embryo_ingest,
            well_provenance=assigned,
            detection_failures=detection_failures,
            dropped_fovs=dropped,
            experiment_manifest=empty_manifest,
            canvas_width_px=None,
            canvas_height_px=None,
        )

    canvas_width_px, canvas_height_px = _canvas_shape(
        assigned, rounding_px=config.canvas_rounding_px
    )
    provenance_columns = [
        column
        for column in (
            "experiment_id",
            "well_index",
            "well_id",
            "image_id",
            "source_experiment_id",
            "source_fov_id",
            "source_embryo_id",
            "embryo_position",
            "source_image_path",
            "relative_path",
            "filename",
            *_BBOX_COLUMNS,
        )
        if column in assigned.columns
    ]
    well_provenance = assigned[provenance_columns].copy()
    well_provenance.to_csv(
        integration_dir / "well_provenance.csv", index=False
    )

    manifest_rows: list[dict[str, Any]] = []
    for experiment_id, experiment_rows in assigned.groupby(
        "experiment_id", sort=True
    ):
        experiment_root = output_root / "experiments" / str(experiment_id)
        experiment_root.mkdir(parents=True, exist_ok=True)
        if materialize_images:
            _materialize_experiment_images(
                experiment_rows,
                experiment_root=experiment_root,
                canvas_width_px=canvas_width_px,
                canvas_height_px=canvas_height_px,
                config=config,
            )

        frame_inventory = _frame_inventory(
            experiment_rows,
            canvas_width_px=canvas_width_px,
            canvas_height_px=canvas_height_px,
            config=config,
        )
        validate_frame_inventory_identity_contract(
            frame_inventory, scope_label="seahub_dropin"
        )
        frame_inventory_csv = experiment_root / "dropin_frame_inventory.csv"
        frame_inventory.to_csv(frame_inventory_csv, index=False)
        if materialize_images:
            validate_frame_inventory(
                frame_inventory_csv,
                frame_inventory_csv.with_suffix(".csv.validated"),
                image_root=experiment_root,
                check_sources=True,
                validation_scope="merged",
            )

        plate_metadata = _plate_metadata(experiment_rows, config=config)
        plate_metadata_csv = experiment_root / "plate_metadata.csv"
        plate_metadata.to_csv(plate_metadata_csv, index=False)
        plate_metadata_csv.with_suffix(".csv.validated").write_text(
            "validated\n", encoding="utf-8"
        )

        runtime_config_yaml = experiment_root / "runtime_config.yaml"
        runtime_config_yaml.write_text(
            yaml.safe_dump(
                _runtime_overlay(
                    experiment_id=str(experiment_id),
                    frame_inventory_csv=frame_inventory_csv,
                    plate_metadata_csv=plate_metadata_csv,
                    image_root=experiment_root,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "experiment_id": experiment_id,
                "frame_inventory_csv": str(frame_inventory_csv.resolve()),
                "plate_metadata_csv": str(plate_metadata_csv.resolve()),
                "image_root": str(experiment_root.resolve()),
                "runtime_config_yaml": str(runtime_config_yaml.resolve()),
                "embryo_count": len(experiment_rows),
            }
        )

    experiment_manifest = pd.DataFrame.from_records(manifest_rows)
    experiment_manifest.to_csv(
        integration_dir / "experiment_manifest.csv", index=False
    )
    (integration_dir / "experiments.txt").write_text(
        "".join(f"{value}\n" for value in experiment_manifest["experiment_id"]),
        encoding="utf-8",
    )
    return SeaHubBundleResult(
        embryo_ingest=embryo_ingest,
        well_provenance=well_provenance,
        detection_failures=detection_failures,
        dropped_fovs=dropped,
        experiment_manifest=experiment_manifest,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
    )


def materialize_planned_experiment(
    *,
    bundle_root: str | Path,
    experiment_id: str,
    overwrite_images: bool = False,
) -> Path:
    """Materialize and source-validate one shard from a prior plan-only bundle.

    This is the cluster-array seam: reconciliation, identity assignment, and the
    corpus-wide canvas are planned once; each array task then writes one shard
    without changing any identity or metadata.
    """
    bundle_root = Path(bundle_root)
    integration_dir = bundle_root / "integration"
    manifest_path = integration_dir / "experiment_manifest.csv"
    provenance_path = integration_dir / "well_provenance.csv"
    manifest = pd.read_csv(manifest_path)
    selected_manifest = manifest[
        manifest["experiment_id"].astype(str).eq(str(experiment_id))
    ]
    if len(selected_manifest) != 1:
        available = sorted(manifest["experiment_id"].astype(str).unique())
        raise ValueError(
            f"Expected one planned manifest row for {experiment_id!r}; found "
            f"{len(selected_manifest)}. Available={available[:10]}."
        )
    manifest_row = selected_manifest.iloc[0]
    frame_inventory_csv = Path(str(manifest_row["frame_inventory_csv"]))
    experiment_root = Path(str(manifest_row["image_root"]))
    frame_inventory = pd.read_csv(frame_inventory_csv)
    provenance = pd.read_csv(provenance_path)
    experiment_rows = provenance[
        provenance["experiment_id"].astype(str).eq(str(experiment_id))
    ].copy()
    if experiment_rows.empty:
        raise ValueError(
            f"well_provenance has no rows for planned experiment {experiment_id!r}."
        )

    width_values = frame_inventory["image_width_px"].dropna().astype(int).unique()
    height_values = frame_inventory["image_height_px"].dropna().astype(int).unique()
    if len(width_values) != 1 or len(height_values) != 1:
        raise ValueError(
            f"Planned frame inventory for {experiment_id!r} must declare one canvas "
            f"shape; widths={width_values.tolist()}, heights={height_values.tolist()}."
        )
    first_frame = frame_inventory.iloc[0]
    config = SeaHubIntegrationConfig(
        micrometers_per_pixel=float(
            first_frame["image_micrometers_per_pixel"]
        ),
        calibration_status=str(first_frame["calibration_status"]),
        jpeg_quality=int(first_frame["jpeg_quality"]),
        canvas_fill_value=int(first_frame.get("canvas_fill_value", 0)),
        flip_polarity=_as_bool(first_frame["flip_polarity"]),
        overwrite_images=bool(overwrite_images),
    )
    _materialize_experiment_images(
        experiment_rows,
        experiment_root=experiment_root,
        canvas_width_px=int(width_values[0]),
        canvas_height_px=int(height_values[0]),
        config=config,
    )
    output_flag = frame_inventory_csv.with_suffix(".csv.validated")
    validate_frame_inventory(
        frame_inventory_csv,
        output_flag,
        image_root=experiment_root,
        check_sources=True,
        validation_scope="merged",
    )
    return output_flag


__all__ = [
    "EXPECTED_EMBRYOS_PER_FOV",
    "SeaHubBundleResult",
    "SeaHubIntegrationConfig",
    "assign_operational_identity",
    "build_embryo_ingest",
    "build_seahub_dropin_bundle",
    "materialize_planned_experiment",
]
