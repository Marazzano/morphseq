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
    PER_WELL_DIRNAME,
    EXPERIMENT,
    PATH_MODE_EXPERIMENT,
    PATH_MODE_MERGED,
    PATH_MODE_PER_WELL,
    PER_WELL_THEN_MERGE,
    PIPELINE_STEPS,
    artifact_path,
    known_artifacts,
    known_steps,
    provenance_path,
    step_dir,
    validated_path,
)

__all__ = [
    "PIPELINE_STEPS",
    "EXPERIMENT",
    "PER_WELL_THEN_MERGE",
    "PER_WELL_DIRNAME",
    "PATH_MODE_EXPERIMENT",
    "PATH_MODE_PER_WELL",
    "PATH_MODE_MERGED",
    "step_dir",
    "artifact_path",
    "validated_path",
    "provenance_path",
    "known_steps",
    "known_artifacts",
]
