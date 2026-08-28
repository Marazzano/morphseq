"""Source-contract checks for the pipeline-to-core manifest adapter.

The declarations in this module are transcribed from current writer/validator
symbols at the cited source locations.  Pipeline imports cannot currently be used
from a clean ``src.*`` installation because those modules import the top-level
``data_pipeline`` package; the adapter reports that packaging discrepancy rather
than manipulating ``sys.path`` or weakening these checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.core.data.manifest_types import SchemaIssue


# These are the current writer/validator authorities used below.  Keep the symbol
# names in the report so a future writer change can be reconciled explicitly.
SOURCE_SYMBOL_CITATIONS: dict[str, tuple[str, ...]] = {
    "snip_inventory": (
        "src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/"
        "snip_identity_contract.py:SNIP_INVENTORY_COLUMNS",
        "src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/"
        "snip_identity_contract.py:validate_snip_inventory_contract",
    ),
    "frame_inventory": (
        "src/data_pipeline/acquisition/image_materialization/frame_inventory_contract.py:"
        "REQUIRED_FRAME_INVENTORY_COLUMNS",
        "src/data_pipeline/acquisition/image_materialization/"
        "materialized_image_write_policy.py:MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS",
        "src/data_pipeline/acquisition/metadata_ingest/frame_inventory/"
        "frame_inventory_validation.py:validate_frame_inventory",
    ),
    "stage_predictions": (
        "src/data_pipeline/feature_extraction/stage_predictions/contract.py:"
        "STAGE_PREDICTION_TABLE_COLUMNS",
        "src/data_pipeline/feature_extraction/stage_predictions/contract.py:"
        "validate_stage_prediction_features",
    ),
    "snip_qc": (
        "src/data_pipeline/quality_control/snip_qc/contract.py:SNIP_QC_TABLE_COLUMNS",
        "src/data_pipeline/quality_control/snip_qc/contract.py:validate_snip_qc",
    ),
    "plate_metadata": (
        "src/data_pipeline/acquisition/metadata_ingest/plate/"
        "plate_metadata_contract.py:REQUIRED_PLATE_METADATA_COLUMNS",
        "src/data_pipeline/acquisition/metadata_ingest/plate/"
        "plate_metadata_contract.py:validate_plate_metadata",
    ),
    "collection_provenance": (
        "src/data_pipeline/acquisition/metadata_ingest/"
        "collection_provenance_contract.py:REQUIRED_COLLECTION_PROVENANCE_KEYS",
        "src/data_pipeline/acquisition/metadata_ingest/"
        "collection_provenance_contract.py:validate_collection_provenance",
    ),
    "acquisition_inventory": (
        "src/data_pipeline/feature_extraction/stage_predictions/compute.py:"
        "_source_ordinal_by_frame",
        "src/data_pipeline/feature_extraction/stage_predictions/compute.py:"
        "_start_age_hpf_for_snip",
    ),
    "path_authority": (
        "src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:artifact_path",
    ),
    "snip_path_authority": (
        "src/data_pipeline/object_extraction/snip_processing/io.py:resolve_from_root",
    ),
}


SNIP_IDENTITY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
    "image_id",
    "time_index",
    "channel_id",
)

SNIP_ADAPTER_REQUIRED_COLUMNS: tuple[str, ...] = SNIP_IDENTITY_COLUMNS + (
    "processed_snip_path",
    "snip_product_key",
    "is_valid_snip",
    "error_message",
)

# Current construction/provenance fields are required by the cited live writer
# contract. They are non-fatal at the adapter boundary so older artifacts can be
# inspected and reported as explicit current-writer schema variants; the
# key/path/validity subset above remains fatal.
SNIP_CURRENT_WRITER_ADDITIONAL_COLUMNS: tuple[str, ...] = (
    "mask_id",
    "track_id",
    "image_path",
    "legacy_flat_snip_path",
    "embryo_mask",
    "embryo_mask_snip_path",
    "crop_x_min_px",
    "crop_y_min_px",
    "crop_x_max_px",
    "crop_y_max_px",
    "crop_width_px",
    "crop_height_px",
    "source_micrometers_per_pixel",
    "snip_micrometers_per_pixel",
    "crop_x_min_um",
    "crop_y_min_um",
    "crop_x_max_um",
    "crop_y_max_um",
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
    "source_image_product_key",
    "output_grid_id",
    "pixel_dtype",
)

FRAME_ADAPTER_REQUIRED_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "image_id",
    "channel_id",
    "time_index",
    "z_index",
    "image_product_type",
    "projection_method",
    "elapsed_time_s",
)

FRAME_CURRENT_WRITER_COLUMNS: tuple[str, ...] = (
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
)
FRAME_VARIANT_COLUMNS: tuple[str, ...] = ("source_ordinal",)

STAGE_ADAPTER_REQUIRED_COLUMNS: tuple[str, ...] = SNIP_IDENTITY_COLUMNS + (
    "predicted_stage_hpf",
)
STAGE_CURRENT_WRITER_ADDITIONAL_COLUMNS: tuple[str, ...] = (
    "stage_prediction_status",
    "model_version",
)

SNIP_QC_REQUIRED_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
    "use_snip",
    "qc_fail_reasons",
)

PLATE_METADATA_REQUIRED_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "well_index",
    "genotype",
    "start_age_hpf",
    "temperature",
    "medium",
)
PLATE_METADATA_VARIANT_COLUMNS: tuple[str, ...] = (
    "strain",
    "chem_perturbation",
)

COLLECTION_PROVENANCE_REQUIRED_KEYS: tuple[str, ...] = (
    "experiment_id",
    "is_collection",
    "sources",
    "start_age_by_source_ordinal",
    "start_age_by_time_index",
)

ACQUISITION_INVENTORY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "well_id",
    "time_index",
    "source_ordinal",
)

QC_BASE_COLUMNS: frozenset[str] = frozenset(SNIP_QC_REQUIRED_COLUMNS)
QC_FLAG_SUFFIX = "_flag"


@dataclass(frozen=True)
class TableContract:
    source_name: str
    adapter_required_columns: tuple[str, ...]
    current_writer_columns: tuple[str, ...]
    variant_columns: tuple[str, ...] = ()


TABLE_CONTRACTS: dict[str, TableContract] = {
    "snip_inventory": TableContract(
        "snip_inventory",
        SNIP_ADAPTER_REQUIRED_COLUMNS,
        tuple(dict.fromkeys(SNIP_ADAPTER_REQUIRED_COLUMNS + SNIP_CURRENT_WRITER_ADDITIONAL_COLUMNS)),
    ),
    "frame_inventory": TableContract(
        "frame_inventory",
        FRAME_ADAPTER_REQUIRED_COLUMNS,
        FRAME_CURRENT_WRITER_COLUMNS,
        FRAME_VARIANT_COLUMNS,
    ),
    "stage_predictions": TableContract(
        "stage_predictions",
        STAGE_ADAPTER_REQUIRED_COLUMNS,
        STAGE_ADAPTER_REQUIRED_COLUMNS + STAGE_CURRENT_WRITER_ADDITIONAL_COLUMNS,
    ),
    "snip_qc": TableContract(
        "snip_qc", SNIP_QC_REQUIRED_COLUMNS, SNIP_QC_REQUIRED_COLUMNS
    ),
    "plate_metadata": TableContract(
        "plate_metadata",
        PLATE_METADATA_REQUIRED_COLUMNS,
        PLATE_METADATA_REQUIRED_COLUMNS,
        PLATE_METADATA_VARIANT_COLUMNS,
    ),
    "acquisition_inventory": TableContract(
        "acquisition_inventory",
        ACQUISITION_INVENTORY_REQUIRED_COLUMNS,
        ACQUISITION_INVENTORY_REQUIRED_COLUMNS,
    ),
}


def inspect_table_schema(
    table: pd.DataFrame,
    *,
    experiment_id: str,
    contract: TableContract,
) -> tuple[SchemaIssue, ...]:
    """Collect required and optional column differences without mutating data."""

    issues: list[SchemaIssue] = []
    missing_required = tuple(
        c for c in contract.adapter_required_columns if c not in table.columns
    )
    if missing_required:
        issues.append(
            SchemaIssue(
                experiment_id=experiment_id,
                source_name=contract.source_name,
                severity="warning",
                code="missing_required_columns",
                message=(
                    f"{experiment_id}: {contract.source_name} is missing required column(s) "
                    f"{list(missing_required)} from {SOURCE_SYMBOL_CITATIONS[contract.source_name]}."
                ),
                columns=missing_required,
            )
        )
    missing_current = tuple(
        c
        for c in contract.current_writer_columns
        if c not in table.columns and c not in contract.adapter_required_columns
    )
    if missing_current:
        issues.append(
            SchemaIssue(
                experiment_id=experiment_id,
                source_name=contract.source_name,
                severity="info",
                code="missing_current_writer_columns",
                message=(
                    f"{experiment_id}: {contract.source_name} omits current-writer column(s) "
                    f"{list(missing_current)} from {SOURCE_SYMBOL_CITATIONS[contract.source_name]}. "
                    "The adapter can inspect this schema variant because its fatal subset is "
                    "present; present columns remain source-authoritative."
                ),
                columns=missing_current,
            )
        )
    missing_variants = tuple(c for c in contract.variant_columns if c not in table.columns)
    if missing_variants:
        issues.append(
            SchemaIssue(
                experiment_id=experiment_id,
                source_name=contract.source_name,
                severity="info",
                code="missing_variant_columns",
                message=(
                    f"{experiment_id}: {contract.source_name} omits supported variant column(s) "
                    f"{list(missing_variants)}."
                ),
                columns=missing_variants,
            )
        )
    return tuple(issues)


def require_table_contract(
    table: pd.DataFrame,
    *,
    experiment_id: str,
    contract: TableContract,
) -> tuple[SchemaIssue, ...]:
    """Return optional differences or fail with every missing required column."""

    issues = inspect_table_schema(table, experiment_id=experiment_id, contract=contract)
    missing = [i for i in issues if i.code == "missing_required_columns"]
    if missing:
        raise ValueError(missing[0].message)
    return issues


def normalize_boolean_series(
    values: pd.Series,
    *,
    experiment_id: str,
    source_name: str,
    column: str,
    nullable: bool = False,
) -> pd.Series:
    """Parse real booleans and exact True/False tokens without ``bool(value)``."""

    def _parse(value: object) -> object:
        if pd.isna(value):
            if nullable:
                return pd.NA
            raise ValueError(
                f"{experiment_id}: {source_name}.{column} contains null; a non-null boolean is required."
            )
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            token = value.strip()
            if token == "True":
                return True
            if token == "False":
                return False
        raise ValueError(
            f"{experiment_id}: {source_name}.{column} has unparseable boolean value {value!r}; "
            "expected bool, 0/1, or exact 'True'/'False'."
        )

    dtype = "boolean" if nullable else bool
    return pd.Series((_parse(v) for v in values), index=values.index, dtype=dtype)


def require_non_null(table: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    for column in columns:
        null_mask = table[column].isna()
        if null_mask.any():
            indices = table.index[null_mask].tolist()[:5]
            raise ValueError(
                f"{label}: required column {column!r} has null values at source rows {indices}."
            )


def require_unique(table: pd.DataFrame, columns: list[str], *, label: str) -> None:
    """Require uniqueness with nullable values participating in the key."""

    duplicate_mask = table.duplicated(subset=columns, keep=False)
    if duplicate_mask.any():
        examples = table.loc[duplicate_mask, columns].head(5).to_dict("records")
        raise ValueError(f"{label}: duplicate key {columns}; examples: {examples}.")


def qc_flag_and_applicability_columns(table: pd.DataFrame) -> tuple[str, ...]:
    """Return every non-base QC decision/applicability column in source order."""

    return tuple(c for c in table.columns if c not in QC_BASE_COLUMNS)
