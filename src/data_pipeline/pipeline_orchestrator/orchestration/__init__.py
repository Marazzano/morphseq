"""Orchestration kingdom: workflow/path helpers for the pipeline.

This subpackage holds the ORCHESTRATION layer (how the pipeline runs: paths, shards, well
selection) — distinct from the IDENTITY layer (``data_pipeline.shared.identifiers``, how objects
are named). Both the Snakefile and the Python entrypoints import from here so rule-output and
code-output paths stay identical.

Currently: ``paths.py`` (the pipeline-wide path registry). The well-runner (well selection,
WellRun, shard merge) lands here next.
"""

from __future__ import annotations

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    EXPERIMENT,
    PATH_MODE_EXPERIMENT,
    PATH_MODE_MERGED,
    PATH_MODE_PER_WELL,
    PER_WELL_THEN_MERGE,
    STAGES,
    artifact_path,
    provenance_path,
    stage_dir,
    validated_path,
)

__all__ = [
    "STAGES",
    "EXPERIMENT",
    "PER_WELL_THEN_MERGE",
    "PATH_MODE_EXPERIMENT",
    "PATH_MODE_PER_WELL",
    "PATH_MODE_MERGED",
    "stage_dir",
    "artifact_path",
    "validated_path",
    "provenance_path",
]
