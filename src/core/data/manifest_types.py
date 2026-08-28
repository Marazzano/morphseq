"""Typed public boundary for the pipeline-backed core manifest adapter.

The tables remain pandas dataframes because they are an interchange boundary with the
pipeline.  Everything around them is named and typed so callers never have to remember a
positional tuple layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import pandas as pd


ZSelectionMode = Literal["projection_null", "all"]


@dataclass(frozen=True)
class AssetSelectionPolicy:
    """Explicit asset vocabulary and the narrow vanilla selector."""

    product_order: tuple[str, ...]
    vanilla_product_key: str
    z_mode: ZSelectionMode = "projection_null"

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_order", tuple(self.product_order))
        if not self.product_order:
            raise ValueError("asset policy requires a non-empty explicit product_order")
        if len(set(self.product_order)) != len(self.product_order):
            raise ValueError("asset policy product_order contains duplicate product keys")
        if self.vanilla_product_key not in self.product_order:
            raise ValueError(
                "asset policy vanilla_product_key must appear in the explicit product_order"
            )


@dataclass(frozen=True)
class ValidityPolicy:
    enabled: bool = True
    name: str = "require_materialized_snip_v1"


@dataclass(frozen=True)
class QCPolicy:
    """Named observation-QC predicate; scientific content is deliberately configurable."""

    enabled: bool
    name: str
    version: str
    accepted_statuses: tuple[str, ...] = ("evaluated",)
    require_use_snip: bool = True
    exclude_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_statuses", tuple(self.accepted_statuses))
        object.__setattr__(self, "exclude_flags", tuple(self.exclude_flags))
        if not self.name or not self.version:
            raise ValueError("QC policy requires non-empty name and version")
        if self.enabled and not self.accepted_statuses:
            raise ValueError("enabled QC policy requires explicit accepted_statuses")


@dataclass(frozen=True)
class StagePolicy:
    enabled: bool
    name: str
    accepted_statuses: tuple[str, ...] = ()
    require_value: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_statuses", tuple(self.accepted_statuses))
        if not self.name:
            raise ValueError("stage policy requires a non-empty name")
        if self.enabled and not self.accepted_statuses:
            raise ValueError("enabled stage policy requires explicit accepted_statuses")


@dataclass(frozen=True)
class CovariatePolicy:
    name: str
    required_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_columns", tuple(self.required_columns))
        if not self.name:
            raise ValueError("covariate policy requires a non-empty name")


@dataclass(frozen=True)
class SplitPolicy:
    enabled: bool
    ratios: tuple[tuple[str, float], ...] = (
        ("train", 0.8),
        ("eval", 0.1),
        ("test", 0.1),
    )
    test_experiments: tuple[str, ...] = ()
    hash_salt: str = "morphseq-core-v1"
    tolerance: float = 0.15
    required_splits: tuple[str, ...] = ("train", "eval", "test")

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratios", tuple(tuple(item) for item in self.ratios))
        object.__setattr__(self, "test_experiments", tuple(self.test_experiments))
        object.__setattr__(self, "required_splits", tuple(self.required_splits))
        names = [name for name, _ in self.ratios]
        if len(names) != len(set(names)):
            raise ValueError("split ratios contain duplicate split names")
        if any(ratio < 0 for _, ratio in self.ratios):
            raise ValueError("split ratios must be non-negative")
        if self.enabled and abs(sum(ratio for _, ratio in self.ratios) - 1.0) > 1e-12:
            raise ValueError("enabled split ratios must sum to 1")
        if self.tolerance < 0:
            raise ValueError("split tolerance must be non-negative")
        unknown_required = set(self.required_splits) - set(names)
        if unknown_required:
            raise ValueError(
                f"required_splits are absent from ratios: {sorted(unknown_required)}"
            )


@dataclass(frozen=True)
class MetricMappingPolicy:
    """Track-A mapping switch or unmistakable test stub, never inferred from strings."""

    enabled: bool
    name: str
    version: str
    scientific_policy: bool = False
    source_column: str | None = None
    mapping: tuple[tuple[str, str], ...] = ()
    constant_group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", tuple(tuple(item) for item in self.mapping))
        if not self.name or not self.version:
            raise ValueError("metric mapping policy requires non-empty name and version")
        if not self.enabled:
            return
        has_mapping = self.source_column is not None and bool(self.mapping)
        has_constant = self.constant_group is not None
        if has_mapping == has_constant:
            raise ValueError(
                "enabled metric mapping requires exactly one of explicit mapping or constant_group"
            )
        if has_constant and self.scientific_policy:
            raise ValueError("constant metric groups are test-only and cannot be scientific")
        if has_constant and not any(token in self.name.lower() for token in ("test", "dummy")):
            raise ValueError("constant metric stub name must contain 'test' or 'dummy'")


@dataclass(frozen=True)
class ManifestPolicy:
    """Fully explicit selector-ready adapter configuration."""

    pipeline_output_root: Path
    experiment_ids: tuple[str, ...]
    assets: AssetSelectionPolicy
    validity: ValidityPolicy
    qc: QCPolicy
    stage: StagePolicy
    covariates: CovariatePolicy
    splits: SplitPolicy
    metric_mapping: MetricMappingPolicy
    acquisition_scope_by_experiment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pipeline_output_root", Path(self.pipeline_output_root))
        object.__setattr__(self, "experiment_ids", tuple(self.experiment_ids))
        object.__setattr__(
            self,
            "acquisition_scope_by_experiment",
            tuple(tuple(item) for item in self.acquisition_scope_by_experiment),
        )
        if not self.experiment_ids:
            raise ValueError("manifest policy requires an explicit ordered experiment_ids list")
        if len(set(self.experiment_ids)) != len(self.experiment_ids):
            raise ValueError("manifest policy experiment_ids contains duplicates")
        unknown_tests = set(self.splits.test_experiments) - set(self.experiment_ids)
        if unknown_tests:
            raise ValueError(
                f"test_experiments are not in experiment_ids: {sorted(unknown_tests)}"
            )
        scopes = dict(self.acquisition_scope_by_experiment)
        if len(scopes) != len(self.acquisition_scope_by_experiment):
            raise ValueError("acquisition_scope_by_experiment contains duplicate experiments")
        unknown_scopes = set(scopes) - set(self.experiment_ids)
        if unknown_scopes:
            raise ValueError(
                "acquisition scopes name experiments outside experiment_ids: "
                f"{sorted(unknown_scopes)}"
            )

    @property
    def acquisition_scopes(self) -> dict[str, str]:
        return dict(self.acquisition_scope_by_experiment)


@dataclass(frozen=True)
class AssetKey:
    snip_id: str
    snip_product_key: str
    z_index: int | None


@dataclass(frozen=True)
class SourceArtifactRecord:
    experiment_id: str
    source_name: str
    path: Path
    exists: bool
    required: bool
    size_bytes: int | None
    mtime_ns: int | None
    row_count: int | None
    sha256: str | None
    schema_version: str | None = None


@dataclass(frozen=True)
class SchemaIssue:
    severity: Literal["info", "warning", "error"]
    experiment_id: str
    source_name: str
    code: str
    message: str
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSchemaReport:
    experiment_id: str
    source_name: str
    path: Path | None
    status: str
    columns: tuple[str, ...]
    dtypes: tuple[tuple[str, str], ...]
    missing_adapter_required: tuple[str, ...] = ()
    missing_current_writer: tuple[str, ...] = ()
    unexpected_columns: tuple[str, ...] = ()
    writer_symbol: str | None = None


@dataclass(frozen=True)
class JoinReport:
    experiment_id: str
    source_name: str
    left_grain: str
    source_grain: str
    missing_left_keys: tuple[str, ...]
    extra_source_keys: tuple[str, ...]


@dataclass(frozen=True)
class ValidationReport:
    schemas: tuple[SourceSchemaReport, ...] = ()
    joins: tuple[JoinReport, ...] = ()
    issues: tuple[SchemaIssue, ...] = ()

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class FilterCount:
    experiment_id: str
    filter_name: str
    rows_in: int
    rows_out: int
    reason: str

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out


@dataclass(frozen=True)
class CohortReport:
    policy_name: str
    counts: tuple[FilterCount, ...]


@dataclass(frozen=True)
class ExperimentTables:
    """One explicit experiment's already-read sources for the testable pure seam."""

    experiment_id: str
    snip_inventory: pd.DataFrame
    frame_inventory: pd.DataFrame
    plate_metadata: pd.DataFrame
    stage_predictions: pd.DataFrame | None = None
    snip_qc: pd.DataFrame | None = None
    collection_provenance: Mapping[str, Any] | None = None
    acquisition_inventory: pd.DataFrame | None = None
    source_paths: Mapping[str, Path] = field(default_factory=dict)
    schema_versions: Mapping[str, str] = field(default_factory=dict)


PathResolver = Callable[[str, Path], Path]


@dataclass(frozen=True)
class ManifestResult:
    observation_table: pd.DataFrame
    asset_table: pd.DataFrame
    selected_sample_view: pd.DataFrame
    observation_order: tuple[str, ...]
    asset_order: tuple[AssetKey, ...]
    source_inventory: tuple[SourceArtifactRecord, ...]
    validation_report: ValidationReport
    cohort_report: CohortReport
    split_assignments: pd.DataFrame
    policy: ManifestPolicy


def disabled_training_policies(
    *,
    root: Path,
    experiment_ids: Sequence[str],
    product_keys: Sequence[str],
    vanilla_product_key: str,
) -> ManifestPolicy:
    """Convenience for inference/preflight with all train-only switches disabled."""

    return ManifestPolicy(
        pipeline_output_root=root,
        experiment_ids=tuple(experiment_ids),
        assets=AssetSelectionPolicy(tuple(product_keys), vanilla_product_key),
        validity=ValidityPolicy(enabled=False, name="inference_validity_disabled"),
        qc=QCPolicy(False, "inference_qc_disabled", "v1"),
        stage=StagePolicy(False, "inference_stage_disabled"),
        covariates=CovariatePolicy("inference_covariates_optional"),
        splits=SplitPolicy(enabled=False, required_splits=()),
        metric_mapping=MetricMappingPolicy(False, "inference_metric_disabled", "v1"),
    )
