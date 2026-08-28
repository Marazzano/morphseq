"""Pipeline contract/path adapter used only by :mod:`pipeline_manifest`.

No other core module should import ``data_pipeline``.  Imports are deliberately lazy: the
repository currently exposes ``src.data_pipeline`` while pipeline modules import the top-level
``data_pipeline`` package.  If the package is not installed correctly, the production boundary
fails with an actionable packaging error instead of suggesting ``PYTHONPATH`` or mutating
``sys.path``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.core.data.manifest_types import SourceArtifactRecord


class PipelineContractImportError(ImportError):
    """Raised when the pipeline's public helper package is not importable."""


def _pipeline_symbol(module: str, symbol: str) -> Any:
    try:
        imported = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise PipelineContractImportError(
            "The manifest adapter could not import the pipeline contract/path helpers from "
            f"{module!r}. Install the repository so the top-level 'data_pipeline' package is "
            "importable. Do not work around this with PYTHONPATH or sys.path manipulation."
        ) from exc
    try:
        return getattr(imported, symbol)
    except AttributeError as exc:
        raise PipelineContractImportError(
            f"Pipeline helper {module}.{symbol} does not exist in the installed pipeline revision."
        ) from exc


@dataclass(frozen=True)
class SourceContract:
    source_name: str
    adapter_required: tuple[str, ...]
    current_writer_columns: tuple[str, ...]
    writer_symbol: str
    boolean_columns: tuple[str, ...] = ()
    nullable_integer_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedExperimentPaths:
    experiment_id: str
    paths: Mapping[str, Path]


# The adapter-required subset is owned by MANIFEST_SCHEMA.md v2.0. Current-writer supersets are
# loaded from the named symbols below rather than copied here.
ADAPTER_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "snip_inventory": (
        "experiment_id",
        "well_id",
        "physical_embryo_id",
        "embryo_id",
        "snip_id",
        "image_id",
        "time_index",
        "channel_id",
        "snip_product_key",
        "processed_snip_path",
        "is_valid_snip",
    ),
    "frame_inventory": (
        "experiment_id",
        "well_id",
        "time_index",
        "channel_id",
        "z_index",
        "image_product_type",
        "projection_method",
        "elapsed_time_s",
    ),
    "stage_predictions": ("snip_id", "predicted_stage_hpf"),
    "snip_qc": ("snip_id", "use_snip", "qc_fail_reasons"),
    "plate_metadata": (
        "experiment_id",
        "well_id",
        "genotype",
        "start_age_hpf",
        "temperature",
        "medium",
    ),
    "collection_provenance": (
        "experiment_id",
        "is_collection",
        "sources",
        "start_age_by_source_ordinal",
        "start_age_by_time_index",
    ),
    "acquisition_inventory": ("well_id", "time_index", "source_ordinal"),
}


def current_writer_contracts() -> dict[str, SourceContract]:
    """Load column authorities from the live pipeline writer/validator symbols."""

    # Product-aware inventory authority:
    # physical_embryo_registry.snip_identity_contract.SNIP_INVENTORY_COLUMNS and the rendering
    # extension inventory_contract.SNIP_INVENTORY_WRITE_COLUMNS.
    snip_columns = tuple(
        _pipeline_symbol(
            "data_pipeline.object_extraction.snip_processing.inventory_contract",
            "SNIP_INVENTORY_WRITE_COLUMNS",
        )
    )
    frame_columns = tuple(
        _pipeline_symbol(
            "data_pipeline.acquisition.image_materialization.frame_inventory_contract",
            "REQUIRED_FRAME_INVENTORY_COLUMNS",
        )
    )
    frame_derived_columns = tuple(
        _pipeline_symbol(
            "data_pipeline.acquisition.image_materialization.frame_inventory_contract",
            "DERIVED_FRAME_INVENTORY_COLUMNS",
        )
    )
    stage_columns = tuple(
        _pipeline_symbol(
            "data_pipeline.feature_extraction.stage_predictions.contract",
            "STAGE_PREDICTION_TABLE_COLUMNS",
        )
    )
    qc_columns = tuple(
        _pipeline_symbol(
            "data_pipeline.quality_control.snip_qc.contract",
            "SNIP_QC_TABLE_COLUMNS",
        )
    )
    qc_flags = tuple(
        _pipeline_symbol(
            "data_pipeline.quality_control.snip_qc.contract",
            "SNIP_QC_EXCLUSION_FLAGS",
        )
    )
    qc_applicability = tuple(
        dict.fromkeys(
            _pipeline_symbol(
                "data_pipeline.quality_control.applicability",
                "FLAG_APPLICABILITY_COLUMNS",
            ).values()
        )
    )
    plate_columns = tuple(
        _pipeline_symbol(
            "data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract",
            "REQUIRED_PLATE_METADATA_COLUMNS",
        )
    )
    collection_keys = tuple(
        _pipeline_symbol(
            "data_pipeline.acquisition.metadata_ingest.collection_provenance_contract",
            "REQUIRED_COLLECTION_PROVENANCE_KEYS",
        )
    )

    return {
        "snip_inventory": SourceContract(
            "snip_inventory",
            ADAPTER_REQUIRED_COLUMNS["snip_inventory"],
            snip_columns,
            "data_pipeline.object_extraction.snip_processing.inventory_contract."
            "SNIP_INVENTORY_WRITE_COLUMNS",
            boolean_columns=("is_valid_snip", "flip_x"),
            nullable_integer_columns=("z_index",),
        ),
        "frame_inventory": SourceContract(
            "frame_inventory",
            ADAPTER_REQUIRED_COLUMNS["frame_inventory"],
            tuple(dict.fromkeys((*frame_columns, *frame_derived_columns))),
            "data_pipeline.acquisition.image_materialization.frame_inventory_contract."
            "REQUIRED_FRAME_INVENTORY_COLUMNS",
            nullable_integer_columns=("z_index",),
        ),
        "stage_predictions": SourceContract(
            "stage_predictions",
            ADAPTER_REQUIRED_COLUMNS["stage_predictions"],
            stage_columns,
            "data_pipeline.feature_extraction.stage_predictions.contract."
            "STAGE_PREDICTION_TABLE_COLUMNS",
        ),
        "snip_qc": SourceContract(
            "snip_qc",
            ADAPTER_REQUIRED_COLUMNS["snip_qc"],
            tuple(dict.fromkeys((*qc_columns, *qc_flags, *qc_applicability))),
            "data_pipeline.quality_control.snip_qc.contract.SNIP_QC_TABLE_COLUMNS + "
            "SNIP_QC_EXCLUSION_FLAGS",
            boolean_columns=("use_snip", *qc_flags),
        ),
        "plate_metadata": SourceContract(
            "plate_metadata",
            ADAPTER_REQUIRED_COLUMNS["plate_metadata"],
            plate_columns,
            "data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract."
            "REQUIRED_PLATE_METADATA_COLUMNS",
        ),
        "acquisition_inventory": SourceContract(
            "acquisition_inventory",
            ADAPTER_REQUIRED_COLUMNS["acquisition_inventory"],
            ADAPTER_REQUIRED_COLUMNS["acquisition_inventory"],
            "data_pipeline.acquisition.metadata_ingest.collection_acquisition_union."
            "union_collection_acquisition_inventories",
        ),
        "collection_provenance": SourceContract(
            "collection_provenance",
            ADAPTER_REQUIRED_COLUMNS["collection_provenance"],
            collection_keys,
            "data_pipeline.acquisition.metadata_ingest.collection_provenance_contract."
            "REQUIRED_COLLECTION_PROVENANCE_KEYS",
        ),
    }


def resolve_experiment_paths(
    *, output_root: Path, experiment_id: str, acquisition_scope: str | None = None
) -> ResolvedExperimentPaths:
    """Resolve one explicit experiment through the pipeline path authority."""

    artifact_path = _pipeline_symbol(
        "data_pipeline.pipeline_orchestrator.orchestration.paths", "artifact_path"
    )
    path_mode_merged = _pipeline_symbol(
        "data_pipeline.pipeline_orchestrator.orchestration.paths", "PATH_MODE_MERGED"
    )

    paths: dict[str, Path] = {
        "snip_inventory": Path(
            artifact_path(
                output_root,
                "snip_inventory",
                "snip_inventory",
                experiment_id,
                path_mode=path_mode_merged,
            )
        ),
        "frame_inventory": Path(
            artifact_path(
                output_root,
                "frame_inventory",
                "inventory",
                experiment_id,
                path_mode=path_mode_merged,
            )
        ),
        "stage_predictions": Path(
            artifact_path(
                output_root,
                "stage_predictions",
                "stage_predictions",
                experiment_id,
                path_mode=path_mode_merged,
            )
        ),
        "snip_qc": Path(
            artifact_path(
                output_root,
                "snip_qc",
                "verdict",
                experiment_id,
                path_mode=path_mode_merged,
            )
        ),
        "plate_metadata": Path(
            artifact_path(output_root, "ingest_plate_metadata", "csv", experiment_id)
        ),
        "collection_provenance": Path(
            artifact_path(
                output_root, "collection_provenance", "provenance", experiment_id
            )
        ),
    }
    if acquisition_scope is not None:
        paths["acquisition_inventory"] = Path(
            artifact_path(
                output_root,
                "ingest_scope_metadata",
                "acquisition_inventory",
                experiment_id,
                format_vars={"scope": acquisition_scope},
            )
        )
    return ResolvedExperimentPaths(experiment_id=experiment_id, paths=paths)


def resolve_pipeline_path(value: str, output_root: Path) -> Path:
    """Resolve a stored pipeline-relative path through its declared helper."""

    resolver = _pipeline_symbol(
        "data_pipeline.object_extraction.snip_processing.io", "resolve_from_root"
    )
    return Path(resolver(value, output_root=output_root))


def read_artifact_table(path: Path, *, source_name: str, experiment_id: str) -> pd.DataFrame:
    """Read CSV/Parquet and make a missing Parquet engine a named hard failure."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"experiment {experiment_id!r}: required {source_name} artifact is missing: {path}"
        )
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"experiment {experiment_id!r}: cannot read {source_name} Parquet artifact "
                f"{path}; install a pandas Parquet engine such as pyarrow. QC must not be skipped."
            ) from exc
        except ValueError as exc:
            if "engine" in str(exc).lower() or "pyarrow" in str(exc).lower():
                raise RuntimeError(
                    f"experiment {experiment_id!r}: cannot read {source_name} Parquet artifact "
                    f"{path}; install a pandas Parquet engine such as pyarrow. QC must not be skipped."
                ) from exc
            raise
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"experiment {experiment_id!r}: unsupported {source_name} table format at {path}; "
        "expected CSV or Parquet"
    )


def read_collection_provenance(path: Path, *, experiment_id: str) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"experiment {experiment_id!r}: collection provenance artifact is missing: {path}"
        )
    reader = _pipeline_symbol(
        "data_pipeline.acquisition.metadata_ingest.collection_provenance",
        "read_collection_provenance",
    )
    return reader(path)


def fingerprint_artifact(
    *,
    experiment_id: str,
    source_name: str,
    path: Path,
    required: bool,
    row_count: int | None,
    schema_version: str | None = None,
) -> SourceArtifactRecord:
    """Fingerprint a source table/JSON; image contents are intentionally not hashed."""

    path = Path(path)
    if not path.exists():
        return SourceArtifactRecord(
            experiment_id,
            source_name,
            path,
            False,
            required,
            None,
            None,
            row_count,
            None,
            schema_version,
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return SourceArtifactRecord(
        experiment_id=experiment_id,
        source_name=source_name,
        path=path,
        exists=True,
        required=required,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        row_count=row_count,
        sha256=digest.hexdigest(),
        schema_version=schema_version,
    )


CURRENT_WRITER_SYMBOLS: tuple[str, ...] = (
    "data_pipeline.object_extraction.snip_processing.inventory_contract."
    "SNIP_INVENTORY_WRITE_COLUMNS",
    "data_pipeline.object_extraction.snip_processing.inventory_contract."
    "validate_snip_inventory",
    "data_pipeline.acquisition.image_materialization.frame_inventory_contract."
    "REQUIRED_FRAME_INVENTORY_COLUMNS",
    "data_pipeline.acquisition.image_materialization.frame_inventory_contract."
    "validate_frame_inventory_identity_contract",
    "data_pipeline.feature_extraction.stage_predictions.contract."
    "STAGE_PREDICTION_TABLE_COLUMNS",
    "data_pipeline.feature_extraction.stage_predictions.contract."
    "validate_stage_prediction_features",
    "data_pipeline.quality_control.snip_qc.contract.SNIP_QC_TABLE_COLUMNS",
    "data_pipeline.quality_control.snip_qc.contract.SNIP_QC_EXCLUSION_FLAGS",
    "data_pipeline.quality_control.snip_qc.contract.validate_snip_qc",
    "data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract."
    "REQUIRED_PLATE_METADATA_COLUMNS",
    "data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract."
    "validate_plate_metadata",
    "data_pipeline.acquisition.metadata_ingest.collection_provenance_contract."
    "REQUIRED_COLLECTION_PROVENANCE_KEYS",
    "data_pipeline.acquisition.metadata_ingest.collection_provenance_contract."
    "validate_collection_provenance",
    "data_pipeline.acquisition.metadata_ingest.collection_acquisition_union."
    "union_collection_acquisition_inventories",
    "data_pipeline.pipeline_orchestrator.orchestration.paths.artifact_path",
    "data_pipeline.object_extraction.snip_processing.io.resolve_from_root",
)
