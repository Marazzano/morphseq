"""Typed public boundary for pipeline-backed core-model manifests.

The tables remain DataFrames because the adapter has to preserve source columns and
nullable values.  The surrounding dataclasses make the table grains, reports, and
policy explicit instead of returning a positional tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd


SplitName = Literal["train", "eval", "test"]
ZSelectionMode = Literal["projection", "single_plane", "all_planes"]


@dataclass(frozen=True)
class SourceArtifact:
    """One declared input artifact and its local fingerprint metadata."""

    experiment_id: str
    source_name: str
    path: Path
    schema_version: str | None
    size_bytes: int
    mtime_ns: int
    row_count: int | None
    sha256: str


@dataclass(frozen=True)
class SchemaIssue:
    """A collected non-fatal schema difference for one declared source."""

    experiment_id: str
    source_name: str
    severity: Literal["info", "warning"]
    code: str
    message: str
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaReport:
    """Structured contract result; identity/key failures raise before return."""

    issues: tuple[SchemaIssue, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class CohortFilterRecord:
    """Count produced by one named filter for one experiment."""

    experiment_id: str
    filter_name: str
    rows_in: int
    rows_out: int
    reason: str


@dataclass(frozen=True)
class CohortReport:
    """Ordered per-filter accounting plus adapter diagnostics."""

    filters: tuple[CohortFilterRecord, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QCPolicy:
    name: str
    version: str
    enabled: bool = True
    accepted_statuses: tuple[str, ...] = ("evaluated",)
    require_use_snip: bool | None = True
    required_flags_false: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagePolicy:
    enabled: bool = False
    required: bool = False
    accepted_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class CovariatePolicy:
    enabled: bool = True
    required_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitPolicy:
    enabled: bool = True
    train_fraction: float = 0.8
    eval_fraction: float = 0.1
    test_fraction: float = 0.1
    tolerance: float = 0.1
    salt: str = "morphseq-core-split-v1"
    explicit_test_experiments: tuple[str, ...] = ()
    required_splits: tuple[SplitName, ...] = ("train", "eval", "test")


@dataclass(frozen=True)
class MetricMappingPolicy:
    enabled: bool = False
    name: str | None = None
    scientific_policy: bool = False


@dataclass(frozen=True)
class ManifestPolicy:
    """Selector-ready, named manifest construction and cohort policy."""

    name: str
    version: str
    experiment_ids: tuple[str, ...]
    allowed_product_keys: tuple[str, ...]
    selected_product_key: str
    z_selection_mode: ZSelectionMode = "projection"
    require_valid_snip: bool = True
    qc: QCPolicy = field(
        default_factory=lambda: QCPolicy(name="disabled", version="1", enabled=False)
    )
    stage: StagePolicy = field(default_factory=StagePolicy)
    covariates: CovariatePolicy = field(default_factory=CovariatePolicy)
    splits: SplitPolicy = field(default_factory=SplitPolicy)
    metric_mapping: MetricMappingPolicy = field(default_factory=MetricMappingPolicy)


@dataclass(frozen=True)
class PipelineManifestResult:
    """Complete deterministic result returned by the pipeline manifest adapter.

    Table grains:

    * ``observation_table``: one row per ``snip_id``;
    * ``asset_table``: one row per ``(snip_id, snip_product_key, z_index)``;
    * ``resolved_sample_table``: one row per selected observation for vanilla mode;
    * ``split_assignments``: one row per ``physical_embryo_id`` when splits are enabled.
    """

    observation_table: pd.DataFrame
    asset_table: pd.DataFrame
    resolved_sample_table: pd.DataFrame
    source_inventory: tuple[SourceArtifact, ...]
    schema_report: SchemaReport
    cohort_report: CohortReport
    split_assignments: pd.DataFrame
    policy: ManifestPolicy

