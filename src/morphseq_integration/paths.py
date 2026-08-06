"""Where the pipeline tree lives — resolved explicitly, never hardcoded at a call site.

The predecessor notebooks each carried their own absolute ``root`` string pointing at a personal
Dropbox mount, which is why none of them ran anywhere but one laptop. Paths are resolved here once,
overridable by environment variable or constructor argument, and passed down.

Layout assumed (matches the live tree):

    <pipeline_root>/
        input/plate_metadata/{experiment_id}_well_metadata.xlsx
        output/acquisition/{experiment_id}/ingest_metadata/plate_metadata.csv
        output/analysis_ready/{experiment_id}/analysis_ready/{experiment_id}_analysis_ready.parquet
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PIPELINE_ROOT = "MORPHSEQ_PIPELINE_ROOT"

_DEFAULT_PIPELINE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline"
)


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved locations of the pipeline inputs and outputs this package reads."""

    pipeline_root: Path

    @property
    def plate_metadata_dir(self) -> Path:
        return self.pipeline_root / "input" / "plate_metadata"

    @property
    def acquisition_dir(self) -> Path:
        return self.pipeline_root / "output" / "acquisition"

    @property
    def analysis_ready_dir(self) -> Path:
        return self.pipeline_root / "output" / "analysis_ready"

    def plate_workbook(self, experiment_id: str) -> Path:
        return self.plate_metadata_dir / f"{experiment_id}_well_metadata.xlsx"

    def plate_metadata_csv(self, experiment_id: str) -> Path:
        return self.acquisition_dir / experiment_id / "ingest_metadata" / "plate_metadata.csv"

    def analysis_ready_parquet(self, experiment_id: str) -> Path:
        return (
            self.analysis_ready_dir
            / experiment_id
            / "analysis_ready"
            / f"{experiment_id}_analysis_ready.parquet"
        )


def default_paths(pipeline_root: str | Path | None = None) -> PipelinePaths:
    """Resolve pipeline paths from an explicit argument, then ``$MORPHSEQ_PIPELINE_ROOT``, then the
    known cluster location.

    Raises:
        FileNotFoundError: if the resolved root does not exist, naming which source supplied it so a
            stale environment variable is obvious.
    """
    if pipeline_root is not None:
        root, source = Path(pipeline_root), "the pipeline_root argument"
    elif os.environ.get(ENV_PIPELINE_ROOT):
        root, source = Path(os.environ[ENV_PIPELINE_ROOT]), f"${ENV_PIPELINE_ROOT}"
    else:
        root, source = _DEFAULT_PIPELINE_ROOT, "the built-in default"

    if not root.is_dir():
        raise FileNotFoundError(
            f"[morphseq_integration] pipeline root {root} (from {source}) does not exist. "
            f"Pass pipeline_root= explicitly or set ${ENV_PIPELINE_ROOT}."
        )
    return PipelinePaths(pipeline_root=root)
