"""snip_qc inputs — small, explicit in-memory assembly of the QC flag columns to judge.

Resolves each source product's per-well artifact path through the public path registry (no second
QC source registry, no raw path strings), reads ONLY the requested flag columns, and merges them
one-to-one on ``snip_id``. Missing registry rows, missing artifacts, missing flag columns, duplicate
snip_id, null flags, and non-boolean flags all FAIL LOUD — MVP never treats a missing flag as a pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PATH_MODE_PER_WELL,
    artifact_path,
    known_artifacts,
)


def load_snip_qc_flag_inputs(
    *,
    output_root: Path,
    experiment_id: str,
    well_id: str,
    sources: list[tuple[str, str]],
    flag_columns: list[str],
) -> pd.DataFrame:
    """Return one row per snip_id with ``snip_id`` + the requested ``flag_columns``.

    ``sources`` is a list of (step, artifact) pairs. ``artifact`` may be ``None`` to infer the
    single registered artifact for a step (fails loud if the step has more than one). Each source's
    per-well shard is read; the union of their flag columns must cover ``flag_columns`` exactly.
    """
    merged: pd.DataFrame | None = None
    seen_flags: set[str] = set()

    for step, artifact in sources:
        artifact_key = artifact or _sole_artifact(step)
        path = artifact_path(
            output_root, step, artifact_key, experiment_id,
            path_mode=PATH_MODE_PER_WELL, well_id=well_id,
        )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"snip_qc: source artifact for step {step!r} not found at {path}. "
                "Every MVP exclusion flag source must be built before snip_qc."
            )
        df = pd.read_csv(path)
        if "snip_id" not in df.columns:
            raise ValueError(f"snip_qc: source {step!r} ({path}) has no snip_id column.")

        wanted = [c for c in flag_columns if c in df.columns]
        for col in wanted:
            if col in seen_flags:
                raise ValueError(f"snip_qc: flag column {col!r} supplied by more than one source.")
            seen_flags.add(col)
        cols = ["snip_id", *wanted]
        piece = _validate_flag_piece(df[cols], step, wanted)

        merged = piece if merged is None else _one_to_one_merge(merged, piece, step)

    if merged is None:
        raise ValueError("snip_qc: no sources supplied.")

    missing = [c for c in flag_columns if c not in seen_flags]
    if missing:
        raise ValueError(
            f"snip_qc: requested flag column(s) {missing} not provided by any source {sources}. "
            "A missing flag is NOT a pass — add the source product or remove the reason."
        )
    return merged[["snip_id", *flag_columns]]


def _sole_artifact(step: str) -> str:
    arts = known_artifacts(step)
    if len(arts) != 1:
        raise ValueError(
            f"snip_qc: step {step!r} has {len(arts)} registered artifacts {arts}; "
            "pass an explicit artifact key."
        )
    return arts[0]


def _validate_flag_piece(piece: pd.DataFrame, step: str, flag_cols: list[str]) -> pd.DataFrame:
    if piece["snip_id"].duplicated().any():
        dupes = piece["snip_id"][piece["snip_id"].duplicated()].unique().tolist()
        raise ValueError(f"snip_qc: source {step!r} has duplicate snip_id(s) {dupes[:5]}.")
    for col in flag_cols:
        if piece[col].isna().any():
            raise ValueError(f"snip_qc: source {step!r} flag {col!r} has null value(s).")
        if piece[col].dtype != bool:
            raise ValueError(
                f"snip_qc: source {step!r} flag {col!r} must be boolean, got {piece[col].dtype}."
            )
    return piece


def _one_to_one_merge(left: pd.DataFrame, right: pd.DataFrame, step: str) -> pd.DataFrame:
    if set(left["snip_id"]) != set(right["snip_id"]):
        raise ValueError(
            f"snip_qc: source {step!r} snip_id set differs from earlier sources; all flag sources "
            "must cover the same snip universe one-to-one."
        )
    return left.merge(right, on="snip_id", how="inner", validate="one_to_one")
