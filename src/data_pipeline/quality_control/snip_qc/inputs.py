"""snip_qc inputs — load and verify the resolved QC flag sources.

Pure load + verify: no path resolution, no orchestration imports. Receives
ResolvedFlagSource objects (already resolved by flag_input_resolver.py and
deserialized from the tracked resolved_sources JSON artifact). For each source:
  - checks the CSV exists
  - checks snip_id is present and unique
  - checks each promised flag column is present and boolean-like
  - coerces to pandas nullable boolean (fails loud on NA or unknown value)
Then merges all sources one-to-one on snip_id.

See: docs/refactors/streamline-snakemake/target/specs/quality_control/snip_qc_verdict_and_flag_resolver.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .flag_input_resolver import ResolvedFlagSource

# Accepted string representations of boolean values (after strip + lowercase).
_BOOL_MAP: dict[str, bool] = {"true": True, "false": False, "1": True, "0": False}


def load_snip_qc_flag_inputs(
    resolved_sources: tuple[ResolvedFlagSource, ...],
) -> pd.DataFrame:
    """Return one row per snip_id with snip_id + all requested flag columns.

    Validates each source CSV and coerces flag columns to nullable boolean.
    All errors are fatal — a missing flag is not a pass.
    """
    merged: pd.DataFrame | None = None

    for source in resolved_sources:
        piece = _load_and_verify_source(source)
        merged = piece if merged is None else _one_to_one_merge(merged, piece, source.step)

    if merged is None:
        raise ValueError("snip_qc inputs: no resolved sources supplied.")

    return merged


def _load_and_verify_source(source: ResolvedFlagSource) -> pd.DataFrame:
    """Load one source CSV and verify snip_id + promised flag columns."""
    if not source.path.exists():
        raise FileNotFoundError(
            f"snip_qc inputs: source {source.step!r} not found at {source.path}. "
            "Every flag source must be built and validated before snip_qc runs."
        )

    df = pd.read_csv(source.path)

    if "snip_id" not in df.columns:
        raise ValueError(
            f"snip_qc inputs: source {source.step!r} ({source.path}) has no snip_id column."
        )
    if df["snip_id"].duplicated().any():
        dupes = df["snip_id"][df["snip_id"].duplicated()].unique().tolist()
        raise ValueError(
            f"snip_qc inputs: source {source.step!r} has duplicate snip_id(s) {dupes[:5]}."
        )

    missing_cols = [c for c in source.flag_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"snip_qc inputs: source {source.step!r} is missing promised flag column(s) "
            f"{missing_cols}. The resolver promised these columns but the CSV does not have them. "
            f"Columns present: {sorted(df.columns)}."
        )

    cols = ["snip_id", *source.flag_columns]
    piece = df[cols].copy()
    for col in source.flag_columns:
        piece[col] = _coerce_boolean_flag(piece[col], step=source.step, col=col)

    return piece


def _coerce_boolean_flag(series: pd.Series, *, step: str, col: str) -> pd.array:
    """Coerce a series to pandas nullable boolean. Fails loud on NA or unknown value."""
    result: list[bool | None] = []
    for val in series:
        if pd.isna(val):
            raise ValueError(
                f"snip_qc inputs: source {step!r} flag {col!r} has null value(s). "
                "Every exclusion flag must be a non-null boolean decision."
            )
        if isinstance(val, bool) or (isinstance(val, int) and val in (0, 1)):
            result.append(bool(val))
        elif isinstance(val, str):
            normalized = val.strip().lower()
            if normalized not in _BOOL_MAP:
                raise ValueError(
                    f"snip_qc inputs: source {step!r} flag {col!r} has unrecognized value "
                    f"{val!r}. Accepted: true/false/1/0 (case-insensitive)."
                )
            result.append(_BOOL_MAP[normalized])
        else:
            raise ValueError(
                f"snip_qc inputs: source {step!r} flag {col!r} has unrecognized value "
                f"{val!r} (type {type(val).__name__}). Accepted: bool, int 0/1, str true/false/1/0."
            )
    return pd.array(result, dtype="boolean")


def _one_to_one_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    step: str,
) -> pd.DataFrame:
    left_ids = set(left["snip_id"].astype(str))
    right_ids = set(right["snip_id"].astype(str))
    if left_ids != right_ids:
        only_left = sorted(left_ids - right_ids)[:5]
        only_right = sorted(right_ids - left_ids)[:5]
        raise ValueError(
            f"snip_qc inputs: source {step!r} snip_id set differs from earlier sources. "
            "All flag sources must cover the same snip universe one-to-one. "
            f"Only in earlier: {only_left}. Only in {step!r}: {only_right}."
        )
    return left.merge(right, on="snip_id", how="inner", validate="one_to_one")
