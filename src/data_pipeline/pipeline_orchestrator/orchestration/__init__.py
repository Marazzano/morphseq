"""Orchestration kingdom: workflow/path helpers for the pipeline.

This subpackage holds the ORCHESTRATION layer (how the pipeline runs: paths, shards, well
selection) — distinct from the IDENTITY layer (``data_pipeline.shared.identifiers``, how objects
are named). Both the Snakefile and the Python entrypoints import from here so rule-output and
code-output paths stay identical.

Currently: ``paths.py`` (the pipeline-wide path registry) and ``well_runner.py`` (well selection
+ shard merge helpers; WellRun lands here next).
"""

from __future__ import annotations

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PER_WELL_DIRNAME,
    EXPERIMENT,
    EXECUTION_PER_WELL,
    EXECUTION_RUN_BATCH,
    PATH_MODE_EXPERIMENT,
    PATH_MODE_MERGED,
    PATH_MODE_PER_WELL,
    PER_WELL_THEN_MERGE,
    PIPELINE_STEPS,
    artifact_path,
    execution_mode,
    known_artifacts,
    known_steps,
    per_well_step_dir,
    provenance_path,
    step_dir,
    validated_path,
)
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths,
    run_well_shard_paths,
    concat_well_shards_to_file,
    run_well_ids_for_experiment,
)

__all__ = [
    "PIPELINE_STEPS",
    "EXPERIMENT",
    "EXECUTION_PER_WELL",
    "EXECUTION_RUN_BATCH",
    "PER_WELL_THEN_MERGE",
    "PER_WELL_DIRNAME",
    "PATH_MODE_EXPERIMENT",
    "PATH_MODE_PER_WELL",
    "PATH_MODE_MERGED",
    "step_dir",
    "per_well_step_dir",
    "artifact_path",
    "validated_path",
    "provenance_path",
    "execution_mode",
    "known_steps",
    "known_artifacts",
    "run_well_ids_for_experiment",
    "collect_well_shard_paths",
    "run_well_shard_paths",
    "concat_well_shards_to_file",
]
