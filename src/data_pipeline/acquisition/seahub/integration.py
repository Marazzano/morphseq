"""Materialize reconciled SeaHub detections as canonical one-embryo wells.

This module owns the boundary between the SeaHub source corpus and morphseq's
drop-in front end.  A SeaHub FOV is never treated as a well: detection must
produce exactly eight positions, and each detected embryo is assigned its own
canonical well in a deterministic operational shard.

The shard is only a batching container.  Stable biological provenance is carried
by ``source_embryo_id`` and the source metadata copied onto every plate row.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps
from scipy import ndimage

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    validate_frame_inventory_identity_contract,
)
from data_pipeline.acquisition.metadata_ingest.frame_inventory.frame_inventory_validation import (
    validate_frame_inventory,
)
from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract import (
    validate_plate_metadata,
)
from data_pipeline.object_extraction.segmentation.frame_masks_contract import (
    FRAME_MASKS_REQUIRED_COLUMNS,
    MAX_VALID_MASK_AREA_FRACTION,
)
from data_pipeline.object_extraction.segmentation.masks.mask_geometry import (
    mask_geometry,
)
from data_pipeline.object_extraction.segmentation.masks.mask_rle import (
    encode_binary_mask_rle,
)
from data_pipeline.object_extraction.segmentation.validate_frame_masks import (
    validate_frame_masks,
)
from data_pipeline.shared.identifiers import (
    build_image_id,
    build_mask_id,
    build_track_id,
    build_well_id,
    normalize_well_index,
)
from data_pipeline.shared.identifiers.constructors import sanitize_experiment_id

from .reconciliation import PASS_THROUGH_FAILURE_STATUSES, apply_inclusion_policy
from .scale_calibration import CALIBRATION_OUTPUT_COLUMNS

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
_ALLOWED_CALIBRATION_STATUSES = frozenset(
    {"placeholder", "calibrated"}
)
_CALIBRATION_MERGE_COLUMNS = tuple(
    column
    for column in CALIBRATION_OUTPUT_COLUMNS
    if column not in {"source_fov_id", "source_stage_value", "stage_hpf"}
)
_SOURCE_MASK_MANIFEST_COLUMNS = (
    "mask_path",
    "mask_score",
    "mask_area_px",
    "raw_mask_area_px",
    "component_count_raw",
    "raw_component_count",
    "component_selection_method",
    "removed_component_area_px",
    "holes_filled_px",
    "holes_filled_area_px",
    "cleaned_to_prompt_area_ratio",
    "mask_to_prompt_area_ratio",
    "mask_bbox_x1_px",
    "mask_bbox_y1_px",
    "mask_bbox_x2_px",
    "mask_bbox_y2_px",
)
_SOURCE_MASK_PROVENANCE_COLUMNS = tuple(
    f"source_{column}" for column in _SOURCE_MASK_MANIFEST_COLUMNS
)
_SOURCE_MASK_RLE_FORMAT = "morphseq_rle_v1"
_MIN_SOURCE_MASK_DETECTOR_CROP_OVERLAP = 0.50


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
        if self.calibration_status not in _ALLOWED_CALIBRATION_STATUSES:
            raise ValueError(
                "Unknown SeaHub calibration_status "
                f"{self.calibration_status!r}; expected one of "
                f"{sorted(_ALLOWED_CALIBRATION_STATUSES)}."
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
    fov_scale_calibration: pd.DataFrame
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


def _included_source_fov_table(policy: pd.DataFrame) -> pd.DataFrame:
    included = policy[policy["include_for_seahub"].astype(bool)].copy()
    included["source_fov_id"] = _source_fov_ids(
        included, label="included reconciled_fovs"
    )
    if included["source_fov_id"].duplicated().any():
        duplicates = sorted(
            included.loc[
                included["source_fov_id"].duplicated(keep=False),
                "source_fov_id",
            ].unique()
        )
        raise ValueError(
            "Included SeaHub source FOV identities must be unique before scale "
            f"calibration; duplicates include {duplicates[:5]}."
        )
    return included


def _validated_fov_scale_calibration(
    policy: pd.DataFrame,
    calibration: pd.DataFrame | None,
    *,
    config: SeaHubIntegrationConfig,
) -> pd.DataFrame:
    """Return one auditable scale row for every included source FOV.

    Direct API callers may omit calibration and retain the explicit legacy 7.8
    placeholder.  The production CLI requires a mask manifest and supplies the
    provisional mask-derived table, so production cannot silently take this path.
    """
    included = _included_source_fov_table(policy)
    included_ids = included["source_fov_id"].astype(str).tolist()
    if calibration is None:
        return pd.DataFrame(
            {
                "source_fov_id": included_ids,
                "stage_hpf": pd.to_numeric(
                    included.get("stage_hpf"), errors="coerce"
                ),
                "image_micrometers_per_pixel": float(
                    config.micrometers_per_pixel
                ),
                "calibration_status": config.calibration_status,
                "scale_estimation_status": "configured_placeholder",
                "calibration_method": "configured_constant",
                "calibration_issue": (
                    "SeaHub pixel calibration is unverified; revisit the configured "
                    f"{float(config.micrometers_per_pixel):g} um/px value."
                ),
            }
        )

    table = calibration.copy()
    table["source_fov_id"] = _source_fov_ids(
        table, label="fov_scale_calibration"
    )
    if table["source_fov_id"].duplicated().any():
        duplicates = sorted(
            table.loc[
                table["source_fov_id"].duplicated(keep=False), "source_fov_id"
            ].unique()
        )
        raise ValueError(
            "fov_scale_calibration must contain one row per source FOV; "
            f"duplicates include {duplicates[:5]}."
        )
    required = {
        "image_micrometers_per_pixel",
        "calibration_status",
        "scale_estimation_status",
        "calibration_method",
        "calibration_issue",
    }
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        raise ValueError(
            "fov_scale_calibration is missing required column(s): "
            f"{missing_columns}."
        )

    by_id = table.set_index("source_fov_id", drop=False)
    missing_ids = sorted(set(included_ids) - set(by_id.index.astype(str)))
    if missing_ids:
        raise ValueError(
            "fov_scale_calibration does not cover every included source FOV; "
            f"missing {len(missing_ids)}, examples={missing_ids[:5]}."
        )
    table = by_id.loc[included_ids].reset_index(drop=True)

    scale = pd.to_numeric(table["image_micrometers_per_pixel"], errors="coerce")
    if scale.isna().any() or not scale.map(math.isfinite).all() or scale.le(0).any():
        raise ValueError(
            "fov_scale_calibration image_micrometers_per_pixel must be finite and > 0."
        )
    table["image_micrometers_per_pixel"] = scale.astype(float)
    status = table["calibration_status"].fillna("").astype(str).str.strip()
    unknown_status = sorted(set(status) - _ALLOWED_CALIBRATION_STATUSES)
    if unknown_status:
        raise ValueError(
            "fov_scale_calibration contains unknown calibration_status value(s): "
            f"{unknown_status}."
        )

    # Catch a calibration table accidentally reused after stage reconciliation changed.
    if "stage_hpf" in table.columns and "stage_hpf" in included.columns:
        expected_stage = pd.to_numeric(included["stage_hpf"], errors="coerce").reset_index(
            drop=True
        )
        observed_stage = pd.to_numeric(table["stage_hpf"], errors="coerce").reset_index(
            drop=True
        )
        comparable = expected_stage.notna() & observed_stage.notna()
        mismatch = comparable & ~pd.Series(
            [
                math.isclose(float(expected), float(observed), abs_tol=1e-6)
                for expected, observed in zip(expected_stage, observed_stage)
            ]
        )
        if mismatch.any():
            offenders = table.loc[mismatch, "source_fov_id"].astype(str).tolist()
            raise ValueError(
                "fov_scale_calibration stage_hpf disagrees with reconciliation for "
                f"source FOV(s): {offenders[:5]}."
            )
    return table


def _attach_fov_scale(
    embryos: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast exactly one source-FOV scale unchanged to its eight embryos."""
    if embryos.empty:
        return embryos.copy()
    carry = [
        "source_fov_id",
        *[
            column
            for column in _CALIBRATION_MERGE_COLUMNS
            if column in calibration.columns
        ],
    ]
    merged = embryos.merge(
        calibration[carry],
        on="source_fov_id",
        how="left",
        validate="many_to_one",
        indicator="_scale_join",
    )
    if not merged["_scale_join"].eq("both").all():
        missing = sorted(
            merged.loc[
                ~merged["_scale_join"].eq("both"), "source_fov_id"
            ].astype(str).unique()
        )
        raise ValueError(
            "Scale calibration failed to join onto detected SeaHub embryos; "
            f"missing source FOVs include {missing[:5]}."
        )
    return merged.drop(columns="_scale_join")


def _attach_source_masks(
    embryos: pd.DataFrame,
    source_mask_manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach one persisted, cleaned source-FOV mask to every emitted embryo.

    The production CLI passes the same complete SAM2 manifest used for provisional
    scale calibration.  Keeping this argument optional preserves the legacy direct
    API for tests and non-production callers, while production fails closed if any
    detected embryo lacks an absolute persisted mask.
    """
    if source_mask_manifest is None or embryos.empty:
        return embryos.copy()

    masks = source_mask_manifest.copy()
    masks["source_fov_id"] = _source_fov_ids(
        masks, label="source_mask_manifest"
    )
    required = {
        "embryo_position",
        "mask_path",
        "mask_area_px",
        "mask_score",
        "cleaned_to_prompt_area_ratio",
        "mask_bbox_x1_px",
        "mask_bbox_y1_px",
        "mask_bbox_x2_px",
        "mask_bbox_y2_px",
    }
    missing = sorted(required - set(masks.columns))
    if missing:
        raise ValueError(
            "source_mask_manifest is missing required column(s): "
            f"{missing}. Production SeaHub materialization requires persisted "
            "source-FOV SAM2 masks, not an areas-only census."
        )

    positions = pd.to_numeric(masks["embryo_position"], errors="coerce")
    if positions.isna().any() or not positions.mod(1).eq(0).all():
        raise ValueError(
            "source_mask_manifest embryo_position values must be non-null integers."
        )
    masks["embryo_position"] = positions.astype(int)
    duplicate_key = masks.duplicated(
        subset=["source_fov_id", "embryo_position"], keep=False
    )
    if duplicate_key.any():
        offenders = (
            masks.loc[duplicate_key, ["source_fov_id", "embryo_position"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "source_mask_manifest must contain one mask per source embryo; "
            f"duplicate keys include {offenders}."
        )

    carry = [
        "source_fov_id",
        "embryo_position",
        *[
            column
            for column in _SOURCE_MASK_MANIFEST_COLUMNS
            if column in masks.columns
        ],
    ]
    mask_view = masks[carry].rename(
        columns={
            column: f"source_{column}"
            for column in _SOURCE_MASK_MANIFEST_COLUMNS
            if column in masks.columns
        }
    )
    joined = embryos.merge(
        mask_view,
        on=["source_fov_id", "embryo_position"],
        how="left",
        validate="one_to_one",
        indicator="_source_mask_join",
    )
    missing_join = ~joined["_source_mask_join"].eq("both")
    if missing_join.any():
        offenders = (
            joined.loc[
                missing_join, ["source_fov_id", "embryo_position"]
            ]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "source_mask_manifest does not cover every detected SeaHub embryo; "
            f"missing keys include {offenders}."
        )
    joined = joined.drop(columns="_source_mask_join")

    resolved_paths: list[str] = []
    for raw_path in joined["source_mask_path"]:
        if _is_missing(raw_path):
            raise ValueError("source_mask_manifest contains a null/empty mask_path.")
        mask_path = Path(str(raw_path)).expanduser()
        if not mask_path.is_absolute():
            raise ValueError(
                "SeaHub source mask paths must be absolute; got "
                f"{mask_path}."
            )
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"SeaHub source mask does not exist: {mask_path}"
            )
        resolved_paths.append(str(mask_path.resolve()))
    joined["source_mask_path"] = resolved_paths

    numeric_contract = {
        column: pd.to_numeric(joined[column], errors="coerce")
        for column in (
            "source_mask_area_px",
            "source_mask_score",
            "source_cleaned_to_prompt_area_ratio",
            "source_mask_bbox_x1_px",
            "source_mask_bbox_y1_px",
            "source_mask_bbox_x2_px",
            "source_mask_bbox_y2_px",
        )
    }
    if any(
        values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all()
        for values in numeric_contract.values()
    ):
        raise ValueError(
            "source_mask_manifest cleaned-mask area, score, ratio, and bounding "
            "boxes must be finite for every detected embryo."
        )
    if numeric_contract["source_mask_area_px"].le(0).any():
        raise ValueError("source_mask_manifest mask_area_px must be positive.")
    cleaned_ratio = numeric_contract["source_cleaned_to_prompt_area_ratio"]
    if cleaned_ratio.le(0).any() or cleaned_ratio.ge(0.95).any():
        raise ValueError(
            "source_mask_manifest cleaned_to_prompt_area_ratio must be > 0 and "
            "< 0.95; refusing an empty or likely prompt-box/full-inset mask."
        )
    if (
        numeric_contract["source_mask_bbox_x2_px"]
        .le(numeric_contract["source_mask_bbox_x1_px"])
        .any()
        or numeric_contract["source_mask_bbox_y2_px"]
        .le(numeric_contract["source_mask_bbox_y1_px"])
        .any()
    ):
        raise ValueError(
            "source_mask_manifest cleaned-mask bounding boxes must have positive "
            "width and height."
        )
    return joined


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
    x1 = pd.to_numeric(embryos["crop_x1_px"])
    y1 = pd.to_numeric(embryos["crop_y1_px"])
    x2 = pd.to_numeric(embryos["crop_x2_px"])
    y2 = pd.to_numeric(embryos["crop_y2_px"])
    source_mask_bbox_columns = {
        "source_mask_bbox_x1_px",
        "source_mask_bbox_y1_px",
        "source_mask_bbox_x2_px",
        "source_mask_bbox_y2_px",
    }
    if source_mask_bbox_columns.issubset(embryos.columns):
        mask_x1 = pd.to_numeric(
            embryos["source_mask_bbox_x1_px"], errors="coerce"
        )
        mask_y1 = pd.to_numeric(
            embryos["source_mask_bbox_y1_px"], errors="coerce"
        )
        mask_x2 = pd.to_numeric(
            embryos["source_mask_bbox_x2_px"], errors="coerce"
        )
        mask_y2 = pd.to_numeric(
            embryos["source_mask_bbox_y2_px"], errors="coerce"
        )
        if pd.concat([mask_x1, mask_y1, mask_x2, mask_y2], axis=1).isna().any().any():
            raise ValueError(
                "Source-mask bounding boxes must be finite for every SeaHub embryo."
            )
        x1 = pd.concat([x1, mask_x1], axis=1).min(axis=1)
        y1 = pd.concat([y1, mask_y1], axis=1).min(axis=1)
        x2 = pd.concat([x2, mask_x2], axis=1).max(axis=1)
        y2 = pd.concat([y2, mask_y2], axis=1).max(axis=1)
    widths = x2 - x1
    heights = y2 - y1
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


def _load_cleaned_source_mask(
    row: pd.Series,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Load and verify one authoritative full-FOV SeaHub mask."""
    mask_path = Path(str(row["source_mask_path"]))
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L")) > 0
    if mask.shape != expected_shape:
        raise ValueError(
            f"Source mask shape for {row['source_embryo_id']} is {mask.shape}; "
            f"expected source image shape {expected_shape}: {mask_path}."
        )
    if not mask.any():
        raise ValueError(
            f"Source mask for {row['source_embryo_id']} is empty: {mask_path}."
        )

    _, component_count = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if int(component_count) != 1:
        raise ValueError(
            f"Cleaned source mask for {row['source_embryo_id']} has "
            f"{int(component_count)} connected components; expected exactly one."
        )
    filled = ndimage.binary_fill_holes(mask)
    if not np.array_equal(filled, mask):
        raise ValueError(
            f"Cleaned source mask for {row['source_embryo_id']} still contains "
            "holes; source-mask cleanup must run before materialization."
        )

    declared_area = pd.to_numeric(
        pd.Series([row.get("source_mask_area_px")]), errors="coerce"
    ).iloc[0]
    if pd.notna(declared_area) and int(round(float(declared_area))) != int(mask.sum()):
        raise ValueError(
            f"Persisted source mask area for {row['source_embryo_id']} does not "
            f"match its manifest ({int(mask.sum())} vs {declared_area})."
        )
    mask_y, mask_x = np.where(mask)
    actual_bbox = (
        int(mask_x.min()),
        int(mask_y.min()),
        int(mask_x.max()) + 1,
        int(mask_y.max()) + 1,
    )
    declared_bbox = tuple(
        int(round(float(row[column])))
        for column in (
            "source_mask_bbox_x1_px",
            "source_mask_bbox_y1_px",
            "source_mask_bbox_x2_px",
            "source_mask_bbox_y2_px",
        )
    )
    if declared_bbox != actual_bbox:
        raise ValueError(
            f"Persisted source mask bounding box for {row['source_embryo_id']} "
            f"does not match its manifest ({actual_bbox} vs {declared_bbox})."
        )
    return mask


def _source_mask_frame_row(
    row: pd.Series,
    *,
    mask_canvas: np.ndarray,
    image_path: Path,
) -> dict[str, Any]:
    """Encode one authoritative source mask in the canonical frame-mask schema."""
    geometry = mask_geometry(mask_canvas)
    image_height_px, image_width_px = mask_canvas.shape
    area_fraction = float(geometry["area_px"]) / float(
        image_height_px * image_width_px
    )
    if area_fraction > MAX_VALID_MASK_AREA_FRACTION:
        raise ValueError(
            f"Source mask for {row['source_embryo_id']} covers "
            f"{area_fraction:.1%} of its materialized frame; refusing a likely "
            "rectangle/full-frame mask."
        )
    raw_score = pd.to_numeric(
        pd.Series([row.get("source_mask_score")]), errors="coerce"
    ).iloc[0]
    mask_confidence = float(raw_score) if pd.notna(raw_score) else 1.0
    image_id = str(row["image_id"])
    well_id = str(row["well_id"])
    return {
        "experiment_id": str(row["experiment_id"]),
        "well_id": well_id,
        "image_id": image_id,
        "time_index": int(row["time_index"]),
        "z_index": pd.NA,
        "channel_id": str(row.get("channel_id", "BF")),
        "image_path": str(image_path.resolve()),
        "image_width_px": int(image_width_px),
        "image_height_px": int(image_height_px),
        "prompt_detection_id": pd.NA,
        "sam2_object_id": 0,
        "mask_id": build_mask_id(image_id, 0),
        "track_id": build_track_id(well_id, 0),
        "mask_rle": json.dumps(encode_binary_mask_rle(mask_canvas)),
        "mask_rle_format": _SOURCE_MASK_RLE_FORMAT,
        **{column: float(value) for column, value in geometry.items()},
        "mask_confidence": mask_confidence,
        "is_valid_mask": True,
        "segmentation_backend": "sam2_image_precomputed",
        "segmentation_model_id": "sam2.1_hiera_large:seahub_source_fov",
        "tracking_backend": "seahub_single_frame",
        "track_id_source": "source_embryo_position",
    }


def _materialize_experiment_images(
    embryos: pd.DataFrame,
    *,
    experiment_root: Path,
    canvas_width_px: int,
    canvas_height_px: int,
    config: SeaHubIntegrationConfig,
) -> pd.DataFrame:
    """Write one-embryo frames and their authoritative source-SAM2 masks."""
    use_source_masks = "source_mask_path" in embryos.columns
    frame_mask_rows: list[dict[str, Any]] = []
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
            source_masks_by_row: dict[Any, np.ndarray] = {}
            all_source_embryos: np.ndarray | None = None
            if use_source_masks:
                for row_index, row in group.iterrows():
                    source_masks_by_row[row_index] = _load_cleaned_source_mask(
                        row,
                        expected_shape=(grayscale.height, grayscale.width),
                    )
                all_source_embryos = np.logical_or.reduce(
                    list(source_masks_by_row.values())
                )

            for row_index, row in group.iterrows():
                x1, y1, x2, y2 = (int(row[column]) for column in _BBOX_COLUMNS)
                if x1 < 0 or y1 < 0 or x2 > grayscale.width or y2 > grayscale.height:
                    raise ValueError(
                        f"Crop for {row['source_embryo_id']} falls outside source image "
                        f"{source_path}: {(x1, y1, x2, y2)} vs "
                        f"{grayscale.width}x{grayscale.height}."
                    )
                mask_canvas: np.ndarray | None = None
                if use_source_masks:
                    source_mask = source_masks_by_row[row_index]
                    detector_crop_overlap = int(source_mask[y1:y2, x1:x2].sum())
                    detector_crop_overlap_fraction = (
                        detector_crop_overlap / int(source_mask.sum())
                    )
                    if (
                        detector_crop_overlap_fraction
                        < _MIN_SOURCE_MASK_DETECTOR_CROP_OVERLAP
                    ):
                        raise ValueError(
                            f"Source mask for {row['source_embryo_id']} overlaps only "
                            f"{detector_crop_overlap_fraction:.1%} of its associated "
                            "detector crop; refusing a likely mislabeled/wrong-neighbor "
                            "mask."
                        )
                    mask_y, mask_x = np.where(source_mask)
                    mask_x1 = int(mask_x.min())
                    mask_y1 = int(mask_y.min())
                    mask_x2 = int(mask_x.max()) + 1
                    mask_y2 = int(mask_y.max()) + 1
                    # The detector crop supplies useful local background for
                    # normalization. Expand it only when necessary so the persisted
                    # primary mask is never silently clipped.
                    x1 = min(x1, mask_x1)
                    y1 = min(y1, mask_y1)
                    x2 = max(x2, mask_x2)
                    y2 = max(y2, mask_y2)
                    if (
                        x2 - x1 > canvas_width_px
                        or y2 - y1 > canvas_height_px
                    ):
                        raise ValueError(
                            f"Detector crop plus full source mask for "
                            f"{row['source_embryo_id']} is {(x2 - x1)}x{(y2 - y1)}, "
                            f"larger than the planned {canvas_width_px}x"
                            f"{canvas_height_px} corpus canvas."
                        )
                crop = grayscale.crop((x1, y1, x2, y2))
                if config.flip_polarity:
                    crop = ImageOps.invert(crop)
                if use_source_masks and all_source_embryos is not None:
                    # Retain genuine local background for SeaHub normalization,
                    # but blank any of the seven sibling embryos that happens to
                    # enter this detector crop. This prevents neighbor anatomy
                    # leaking through the downstream soft mask-edge taper.
                    sibling_pixels = (
                        all_source_embryos[y1:y2, x1:x2]
                        & ~source_mask[y1:y2, x1:x2]
                    )
                    if sibling_pixels.any():
                        crop_pixels = np.asarray(crop).copy()
                        crop_pixels[sibling_pixels] = int(
                            config.canvas_fill_value
                        )
                        crop = Image.fromarray(crop_pixels, mode="L")
                canvas = Image.new(
                    "L",
                    (canvas_width_px, canvas_height_px),
                    color=int(config.canvas_fill_value),
                )
                left = (canvas_width_px - crop.width) // 2
                top = (canvas_height_px - crop.height) // 2
                canvas.paste(crop, (left, top))
                if use_source_masks:
                    cropped_mask = source_mask[y1:y2, x1:x2]
                    mask_canvas = np.zeros(
                        (canvas_height_px, canvas_width_px), dtype=bool
                    )
                    mask_canvas[
                        top : top + cropped_mask.shape[0],
                        left : left + cropped_mask.shape[1],
                    ] = cropped_mask
                    if int(mask_canvas.sum()) != int(source_mask.sum()):
                        raise ValueError(
                            f"Materialization clipped the source mask for "
                            f"{row['source_embryo_id']}."
                        )
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
                else:
                    canvas.save(
                        output_path,
                        format="JPEG",
                        quality=int(config.jpeg_quality),
                    )
                if mask_canvas is not None:
                    frame_mask_rows.append(
                        _source_mask_frame_row(
                            row,
                            mask_canvas=mask_canvas,
                            image_path=output_path,
                        )
                    )
    return pd.DataFrame.from_records(
        frame_mask_rows, columns=FRAME_MASKS_REQUIRED_COLUMNS
    )


def _frame_inventory(
    embryos: pd.DataFrame,
    *,
    experiment_root: Path,
    canvas_width_px: int,
    canvas_height_px: int,
    config: SeaHubIntegrationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, embryo in embryos.iterrows():
        pixel_size = float(
            embryo.get(
                "image_micrometers_per_pixel", config.micrometers_per_pixel
            )
        )
        calibration_status = str(
            embryo.get("calibration_status", config.calibration_status)
        )
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
                # Downstream model services consume the recorded path directly and
                # do not consistently carry the drop-in image_root.  SeaHub therefore
                # makes this contract stronger than the generic frame inventory:
                # every materialized image path is absolute.
                "image_path": str(
                    (experiment_root / _relative_materialized_path(embryo)).resolve()
                ),
                "image_micrometers_per_pixel": pixel_size,
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
                "calibration_status": calibration_status,
                "scale_estimation_status": embryo.get(
                    "scale_estimation_status", "configured_placeholder"
                ),
                "calibration_method": embryo.get(
                    "calibration_method", "configured_constant"
                ),
                "calibration_reference_version": embryo.get(
                    "calibration_reference_version", pd.NA
                ),
                "canvas_fill_value": int(config.canvas_fill_value),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_precomputed_frame_masks(
    frame_masks: pd.DataFrame,
    *,
    frame_inventory: pd.DataFrame,
    output_csv: Path,
) -> None:
    """Validate and atomically persist a shard's authoritative frame masks."""
    if frame_masks.empty:
        raise ValueError(
            "Cannot write SeaHub precomputed frame masks: materialization returned "
            "no authoritative masks."
        )
    validate_frame_masks(frame_masks, frame_inventory)
    expected_ids = set(frame_inventory["image_id"].astype(str))
    observed_ids = set(frame_masks["image_id"].astype(str))
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(
            "SeaHub precomputed masks must contain exactly one row for every "
            f"materialized frame; missing={missing[:5]}, extra={extra[:5]}."
        )
    if len(frame_masks) != len(frame_inventory):
        raise ValueError(
            "SeaHub precomputed masks must contain exactly one authoritative mask "
            f"per frame; masks={len(frame_masks)}, frames={len(frame_inventory)}."
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    frame_masks.to_csv(temporary, index=False)
    temporary.replace(output_csv)


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
        pixel_size = float(
            embryo.get(
                "image_micrometers_per_pixel", config.micrometers_per_pixel
            )
        )
        calibration_status = str(
            embryo.get("calibration_status", config.calibration_status)
        )
        calibration_issue = embryo.get("calibration_issue")
        if _is_missing(calibration_issue):
            calibration_issue = (
                "Mask-derived SeaHub scale is provisional rather than direct "
                "physical metrology; revisit before absolute-size analysis."
                if str(
                    embryo.get("scale_estimation_status", "")
                ).startswith("mask_area_")
                else (
                    "SeaHub pixel calibration is unverified; revisit "
                    f"{pixel_size:g} um/px."
                )
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
            "micrometers_per_pixel": pixel_size,
            "calibration_status": calibration_status,
            "scale_estimation_status": embryo.get(
                "scale_estimation_status", "configured_placeholder"
            ),
            "calibration_method": embryo.get(
                "calibration_method", "configured_constant"
            ),
            "calibration_reference_version": embryo.get(
                "calibration_reference_version", pd.NA
            ),
            "calibration_issue": calibration_issue,
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
    precomputed_frame_masks_csv: Path | None = None,
) -> dict[str, Any]:
    frame_masks: dict[str, Any] = {"use_model_server": True}
    if precomputed_frame_masks_csv is not None:
        frame_masks.update(
            {
                "mode": "precomputed",
                "precomputed_csv": str(precomputed_frame_masks_csv.resolve()),
                # Keep GroundingDINO as a required, independently persisted audit.
                # Its boxes may never replace the authoritative source SAM2 mask.
                "require_detection_audit": True,
            }
        )
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
        "frame_masks": frame_masks,
        # Source SeaHub images have a distinct intensity distribution. Disable
        # CLAHE only in this runtime overlay; Keyence/YX1 retain the global default.
        "snip_processing": {"apply_clahe": False},
        "unet_snip": {"use_model_server": True},
    }


def build_seahub_dropin_bundle(
    reconciled_fovs: pd.DataFrame,
    detection_manifest: pd.DataFrame,
    *,
    output_root: str | Path,
    config: SeaHubIntegrationConfig | None = None,
    fov_scale_calibration: pd.DataFrame | None = None,
    source_mask_manifest: pd.DataFrame | None = None,
    materialize_images: bool = True,
) -> SeaHubBundleResult:
    """Build and validate a complete per-experiment SeaHub drop-in bundle."""
    config = config or SeaHubIntegrationConfig()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(
                f"SeaHub output root exists and is not a directory: {output_root}"
            )
        existing = sorted(output_root.iterdir())
        if existing:
            preview = ", ".join(path.name for path in existing[:5])
            raise FileExistsError(
                "SeaHub output root must be fresh and empty; refusing to reuse "
                f"{output_root}. Existing entries include: {preview}. Choose a new "
                "run/output root or explicitly quarantine the old run first."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    policy = apply_inclusion_policy(reconciled_fovs)
    dropped = policy[~policy["include_for_seahub"].astype(bool)].copy()
    scale_calibration = _validated_fov_scale_calibration(
        policy, fov_scale_calibration, config=config
    )
    embryo_ingest, detection_failures = build_embryo_ingest(
        policy, detection_manifest
    )
    embryo_ingest = _attach_fov_scale(embryo_ingest, scale_calibration)
    embryo_ingest = _attach_source_masks(embryo_ingest, source_mask_manifest)
    assigned = assign_operational_identity(embryo_ingest, config=config)

    integration_dir = output_root / "integration"
    integration_dir.mkdir(parents=True, exist_ok=True)
    policy.to_csv(integration_dir / "reconciled_fovs.csv", index=False)
    scale_calibration.to_csv(
        integration_dir / "fov_scale_calibration.csv", index=False
    )
    if source_mask_manifest is not None:
        source_mask_manifest.to_csv(
            integration_dir / "source_mask_manifest.csv", index=False
        )
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
                "precomputed_frame_masks_csv",
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
            fov_scale_calibration=scale_calibration,
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
            "channel_id",
            "time_index",
            "image_id",
            "source_experiment_id",
            "source_fov_id",
            "source_embryo_id",
            "embryo_position",
            "source_image_path",
            "relative_path",
            "filename",
            *_BBOX_COLUMNS,
            *_CALIBRATION_MERGE_COLUMNS,
            *_SOURCE_MASK_PROVENANCE_COLUMNS,
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
        precomputed_frame_masks_csv = (
            experiment_root / "dropin_frame_masks.csv"
            if "source_mask_path" in experiment_rows.columns
            else None
        )
        materialized_frame_masks = pd.DataFrame(
            columns=FRAME_MASKS_REQUIRED_COLUMNS
        )
        if materialize_images:
            materialized_frame_masks = _materialize_experiment_images(
                experiment_rows,
                experiment_root=experiment_root,
                canvas_width_px=canvas_width_px,
                canvas_height_px=canvas_height_px,
                config=config,
            )

        frame_inventory = _frame_inventory(
            experiment_rows,
            experiment_root=experiment_root,
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
            if precomputed_frame_masks_csv is not None:
                _write_precomputed_frame_masks(
                    materialized_frame_masks,
                    frame_inventory=frame_inventory,
                    output_csv=precomputed_frame_masks_csv,
                )
            validate_frame_inventory(
                frame_inventory_csv,
                frame_inventory_csv.with_suffix(".csv.validated"),
                # Absolute paths are deliberate for SeaHub.  Passing no image_root
                # makes the shared validator reject any accidental relative path.
                image_root=None,
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
                    precomputed_frame_masks_csv=precomputed_frame_masks_csv,
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
                "precomputed_frame_masks_csv": (
                    str(precomputed_frame_masks_csv.resolve())
                    if precomputed_frame_masks_csv is not None
                    else pd.NA
                ),
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
        fov_scale_calibration=scale_calibration,
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
    materialized_frame_masks = _materialize_experiment_images(
        experiment_rows,
        experiment_root=experiment_root,
        canvas_width_px=int(width_values[0]),
        canvas_height_px=int(height_values[0]),
        config=config,
    )
    raw_precomputed_path = manifest_row.get("precomputed_frame_masks_csv")
    if not _is_missing(raw_precomputed_path):
        precomputed_path = Path(str(raw_precomputed_path)).expanduser()
        if not precomputed_path.is_absolute():
            raise ValueError(
                "Planned SeaHub precomputed_frame_masks_csv must be absolute; "
                f"got {precomputed_path}."
            )
        precomputed_path = precomputed_path.resolve()
        expected_precomputed_path = (
            experiment_root / "dropin_frame_masks.csv"
        ).resolve()
        if precomputed_path != expected_precomputed_path:
            raise ValueError(
                "Planned SeaHub precomputed_frame_masks_csv points outside the "
                f"selected shard: {precomputed_path}; expected "
                f"{expected_precomputed_path}."
            )
        _write_precomputed_frame_masks(
            materialized_frame_masks,
            frame_inventory=frame_inventory,
            output_csv=precomputed_path,
        )
    output_flag = frame_inventory_csv.with_suffix(".csv.validated")
    validate_frame_inventory(
        frame_inventory_csv,
        output_flag,
        # Absolute paths are deliberate for SeaHub.  Passing no image_root makes
        # the shared validator reject stale relative-path inventories.
        image_root=None,
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
