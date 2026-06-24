"""surface_area reference contract — schema truth for the packaged percentile curve.

The reference is a static source asset (not a pipeline artifact), but it still has a
contract: the flag math depends on `p5`/`p95` and review tooling depends on `p50`/`n`,
so all five columns are required and validated before the curve is used. Ported from the
legacy `validate_sa_reference` body in `surface_area_outlier_detection.py`.
"""

from __future__ import annotations

import pandas as pd

SURFACE_AREA_REFERENCE_REQUIRED_COLUMNS: list[str] = ["stage_hpf", "p5", "p50", "p95", "n"]


def validate_surface_area_reference(
    df: pd.DataFrame, *, scope_label: str = "surface_area_reference"
) -> None:
    """Fail loud unless ``df`` is a valid surface-area reference table.

    Requires all five columns, a non-empty monotonic `stage_hpf` axis, and finite,
    non-negative, ordered percentiles (``p5 <= p50 <= p95``) on every row.
    """
    missing = [c for c in SURFACE_AREA_REFERENCE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Expected {SURFACE_AREA_REFERENCE_REQUIRED_COLUMNS}."
        )

    if len(df) == 0:
        raise ValueError(f"{scope_label}: reference is empty; need at least one stage bin.")

    stage = pd.to_numeric(df["stage_hpf"], errors="coerce")
    if stage.isna().any():
        raise ValueError(f"{scope_label}: stage_hpf has null/non-numeric value(s).")
    if not stage.is_monotonic_increasing:
        raise ValueError(
            f"{scope_label}: stage_hpf must be sorted ascending so np.interp is well-defined."
        )

    for col in ("p5", "p50", "p95"):
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{scope_label}: percentile column {col!r} has null/non-numeric value(s).")
        if (values < 0).any():
            raise ValueError(f"{scope_label}: percentile column {col!r} has negative value(s).")

    p5 = pd.to_numeric(df["p5"], errors="coerce")
    p50 = pd.to_numeric(df["p50"], errors="coerce")
    p95 = pd.to_numeric(df["p95"], errors="coerce")
    if not ((p5 <= p50) & (p50 <= p95)).all():
        bad = df.loc[~((p5 <= p50) & (p50 <= p95)), "stage_hpf"].head(5).tolist()
        raise ValueError(
            f"{scope_label}: percentiles must satisfy p5 <= p50 <= p95; violated at "
            f"stage_hpf {bad}."
        )
