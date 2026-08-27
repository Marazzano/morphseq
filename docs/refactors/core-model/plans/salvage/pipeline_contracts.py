"""Schema declarations for pipeline artifacts consumed by the core manifest.

These declarations intentionally describe the compatibility boundary observed by core.  They do
not import :mod:`src.data_pipeline`; the manifest adapter is the only module allowed to do that.
Validation is diagnostic and returns a complete report instead of failing on the first schema
difference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd


PIPELINE_CONTRACT_VERSION = "morphseq.training_manifest.sources.v1"

IDENTITY_SPINE_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
)
FRAME_PROVENANCE_COLUMNS: tuple[str, ...] = ("image_id", "time_index", "channel_id")

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
QC_APPLICABILITY_COLUMNS: tuple[str, ...] = (
    "focus_qc_applicability",
    "motion_blur_qc_applicability",
    "surface_area_qc_applicability",
    "death_detection_qc_applicability",
)

# Reserved for a future pipeline schema.  AGENTS.md and MANIFEST_SCHEMA.md are binding: none of
# these fields exists in the measured cohort, so core must never derive, default, or impute them.
RESERVED_OPTICAL_COLUMNS: tuple[str, ...] = (
    "source_micrometers_per_pixel",
    "snip_micrometers_per_pixel",
    "microscope_id",
    "objective_magnification",
    "z_position",
    "numerical_aperture",
)


@dataclass(frozen=True)
class SourceContract:
    """A tolerant declaration for one experiment-level source artifact."""

    name: str
    version: str
    key_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    expected_dtypes: Mapping[str, str]
    writer_provenance: str
    validator_provenance: str


@dataclass(frozen=True)
class ContractValidationReport:
    """JSON-ready result of validating one dataframe against a source contract."""

    source: str
    contract_version: str
    row_count: int
    columns: tuple[str, ...]
    missing_required: tuple[str, ...]
    observed_optional: tuple[str, ...]
    missing_optional: tuple[str, ...]
    extra_columns: tuple[str, ...]
    dtype_issues: tuple[dict[str, str], ...]
    duplicate_key_rows: int

    @property
    def conforms(self) -> bool:
        return not self.missing_required and not self.dtype_issues and self.duplicate_key_rows == 0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["conforms"] = self.conforms
        return result


_STRING_ID_DTYPES = {column: "string" for column in (*IDENTITY_SPINE_COLUMNS, "image_id", "channel_id")}

# Source: src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/
#         snip_identity_contract.py::SNIP_INVENTORY_COLUMNS and
#         src/data_pipeline/object_extraction/snip_processing/entrypoints/
#         run_snip_processing.py::run_snip_processing.
# The current writer additionally declares the first two RESERVED_OPTICAL_COLUMNS.  All 133 real
# inventories predate those additions, so the compatibility contract records them as optional and
# reports their absence rather than fabricating values or rejecting the full cohort.
SNIP_INVENTORY_CONTRACT = SourceContract(
    name="snip_inventory",
    version=PIPELINE_CONTRACT_VERSION,
    key_columns=("snip_id",),
    required_columns=(
        *IDENTITY_SPINE_COLUMNS,
        *FRAME_PROVENANCE_COLUMNS,
        "mask_id",
        "track_id",
        "image_path",
        "processed_snip_path",
        "embryo_mask",
        "embryo_mask_snip_path",
        "crop_x_min_px",
        "crop_y_min_px",
        "crop_x_max_px",
        "crop_y_max_px",
        "crop_width_px",
        "crop_height_px",
        "is_valid_snip",
        "error_message",
    ),
    optional_columns=("source_micrometers_per_pixel", "snip_micrometers_per_pixel"),
    expected_dtypes={
        **_STRING_ID_DTYPES,
        "time_index": "integer",
        "processed_snip_path": "string",
        "embryo_mask_snip_path": "string",
        "is_valid_snip": "boolean",
        "source_micrometers_per_pixel": "numeric",
        "snip_micrometers_per_pixel": "numeric",
    },
    writer_provenance=(
        "src/data_pipeline/object_extraction/snip_processing/entrypoints/"
        "run_snip_processing.py::run_snip_processing"
    ),
    validator_provenance=(
        "src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/"
        "snip_identity_contract.py::validate_snip_inventory_contract"
    ),
)

# Source: src/data_pipeline/feature_extraction/stage_predictions/compute.py::
#         compute_stage_prediction_features and contract.py::validate_stage_prediction_features.
# stage_prediction_status is optional only to tolerate measured schema S02; its absence is
# materialised as stage_status='unavailable' by the adapter.
STAGE_PREDICTIONS_CONTRACT = SourceContract(
    name="stage_predictions",
    version=PIPELINE_CONTRACT_VERSION,
    key_columns=("snip_id",),
    required_columns=(
        *IDENTITY_SPINE_COLUMNS,
        *FRAME_PROVENANCE_COLUMNS,
        "predicted_stage_hpf",
        "model_version",
    ),
    optional_columns=("stage_prediction_status",),
    expected_dtypes={
        **_STRING_ID_DTYPES,
        "time_index": "integer",
        "predicted_stage_hpf": "numeric",
        "model_version": "string",
        "stage_prediction_status": "string",
    },
    writer_provenance=(
        "src/data_pipeline/feature_extraction/stage_predictions/compute.py::"
        "compute_stage_prediction_features"
    ),
    validator_provenance=(
        "src/data_pipeline/feature_extraction/stage_predictions/contract.py::"
        "validate_stage_prediction_features"
    ),
)

# Source: src/data_pipeline/quality_control/snip_qc/build.py::build_snip_qc_verdict and
#         contract.py::validate_snip_qc.  Source flags are opportunistic by the pipeline's own
# contract, which permits the measured minimal Q02 schema.
SNIP_QC_CONTRACT = SourceContract(
    name="snip_qc",
    version=PIPELINE_CONTRACT_VERSION,
    key_columns=("snip_id",),
    required_columns=(*IDENTITY_SPINE_COLUMNS, "use_snip", "qc_fail_reasons"),
    optional_columns=(*QC_FLAG_COLUMNS, *QC_APPLICABILITY_COLUMNS),
    expected_dtypes={
        **{column: "string" for column in IDENTITY_SPINE_COLUMNS},
        "use_snip": "boolean",
        "qc_fail_reasons": "string",
        **{column: "boolean" for column in QC_FLAG_COLUMNS},
        **{column: "string" for column in QC_APPLICABILITY_COLUMNS},
    },
    writer_provenance=(
        "src/data_pipeline/quality_control/snip_qc/build.py::build_snip_qc_verdict"
    ),
    validator_provenance=(
        "src/data_pipeline/quality_control/snip_qc/contract.py::validate_snip_qc"
    ),
)

# Source: src/data_pipeline/acquisition/metadata_ingest/plate/plate_processing.py and
#         plate_metadata_contract.py::validate_plate_metadata.  The validator guarantees only the
# seven required fields; 38 observed schemas contribute optional columns.
PLATE_METADATA_CONTRACT = SourceContract(
    name="plate_metadata",
    version=PIPELINE_CONTRACT_VERSION,
    key_columns=("well_id",),
    required_columns=(
        "experiment_id",
        "well_id",
        "well_index",
        "genotype",
        "start_age_hpf",
        "temperature",
        "medium",
    ),
    optional_columns=(
        "strain",
        "chem_perturbation",
        "embryos_per_well",
        "mold_type",
        "start_age_morph",
        "start_stage_hpf",
        "pair",
        "series_number_map",
        "genotype_map_orig",
        "orig_genotype",
        "sequenced",
        "qc",
        "morph_seq_qc",
        "image_notes",
        "image_to_hash_map",
        "image_to_hash_plate_num",
        "hash_to_image_map",
        "hash_plate_num",
        "tricane",
    ),
    expected_dtypes={
        "experiment_id": "string",
        "well_id": "string",
        "well_index": "string",
        "genotype": "string",
        "start_age_hpf": "numeric",
        "temperature": "numeric",
        "medium": "string",
    },
    writer_provenance=(
        "src/data_pipeline/acquisition/metadata_ingest/plate/"
        "plate_processing.py::assemble_plate_metadata"
    ),
    validator_provenance=(
        "src/data_pipeline/acquisition/metadata_ingest/plate/"
        "plate_metadata_contract.py::validate_plate_metadata"
    ),
)

SOURCE_CONTRACTS: Mapping[str, SourceContract] = {
    contract.name: contract
    for contract in (
        SNIP_INVENTORY_CONTRACT,
        STAGE_PREDICTIONS_CONTRACT,
        SNIP_QC_CONTRACT,
        PLATE_METADATA_CONTRACT,
    )
}


def _dtype_issue(series: pd.Series, expected: str) -> str | None:
    """Return a diagnostic when non-null values cannot satisfy ``expected``."""

    values = series.dropna()
    if values.empty:
        return None
    if expected == "string":
        if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
            return None
        return f"expected string-compatible values, observed dtype {series.dtype}"
    if expected in {"numeric", "integer"}:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            return f"contains values that are not {expected}-compatible (dtype {series.dtype})"
        if expected == "integer" and not np.equal(numeric, np.floor(numeric)).all():
            return f"contains non-integral values (dtype {series.dtype})"
        return None
    if expected == "boolean":
        allowed = {"true", "false", "1", "0"}
        observed = set(values.astype(str).str.strip().str.casefold())
        if not observed <= allowed:
            return f"contains non-boolean tokens {sorted(observed - allowed)[:5]}"
        return None
    return f"unknown declared dtype category {expected!r}"


def validate_source_contract(
    dataframe: pd.DataFrame,
    contract: SourceContract | str,
) -> ContractValidationReport:
    """Validate all observable differences and return them without raising."""

    if isinstance(contract, str):
        try:
            contract = SOURCE_CONTRACTS[contract]
        except KeyError:
            raise KeyError(
                f"Unknown source contract {contract!r}; expected one of {sorted(SOURCE_CONTRACTS)}"
            ) from None

    columns = tuple(str(column) for column in dataframe.columns)
    column_set = set(columns)
    required = set(contract.required_columns)
    optional = set(contract.optional_columns)
    dtype_issues: list[dict[str, str]] = []
    for column, expected in contract.expected_dtypes.items():
        if column not in column_set:
            continue
        issue = _dtype_issue(dataframe[column], expected)
        if issue is not None:
            dtype_issues.append(
                {
                    "column": column,
                    "expected": expected,
                    "observed": str(dataframe[column].dtype),
                    "message": issue,
                }
            )

    duplicate_rows = 0
    if all(column in column_set for column in contract.key_columns):
        duplicate_rows = int(
            dataframe.duplicated(subset=list(contract.key_columns), keep=False).sum()
        )

    return ContractValidationReport(
        source=contract.name,
        contract_version=contract.version,
        row_count=int(len(dataframe)),
        columns=columns,
        missing_required=tuple(column for column in contract.required_columns if column not in column_set),
        observed_optional=tuple(column for column in contract.optional_columns if column in column_set),
        missing_optional=tuple(column for column in contract.optional_columns if column not in column_set),
        extra_columns=tuple(column for column in columns if column not in required | optional),
        dtype_issues=tuple(dtype_issues),
        duplicate_key_rows=duplicate_rows,
    )

