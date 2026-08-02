"""death_detection contracts — TWO tables, TWO grains.

  - death_detection_qc: per ``snip_id`` — SNIP_ID_SPINE_COLUMNS + viability_dead_flag +
    persistence_dead_flag + applicability. Product-pure: no death-time/stage columns.
  - death_event: per ``physical_embryo_id`` (the animal — channel- and time-independent, so NO
    embryo_id) — PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + death_event_time_index + death_event_stage_hpf.

Each table validates its OWN grain (spine first via the shared validator at the matching grain
token), then its own columns. Spine sets are imported from the minting site, never re-typed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    PHYSICAL_EMBRYO_ID_SPINE_COLUMNS,
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

# ── per-snip death flag table ────────────────────────────────────────────────────────────────
DEATH_DETECTION_QC_FLAG_COLUMNS: tuple[str, ...] = (
    "viability_dead_flag",
    "persistence_dead_flag",
)
DEATH_DETECTION_QC_APPLICABILITY_COLUMN = "death_detection_qc_applicability"
DEATH_DETECTION_QC_PAYLOAD_COLUMNS: tuple[str, ...] = (
    *DEATH_DETECTION_QC_FLAG_COLUMNS,
    DEATH_DETECTION_QC_APPLICABILITY_COLUMN,
)
DEATH_DETECTION_QC_TABLE_COLUMNS: list[str] = list(SNIP_ID_SPINE_COLUMNS + DEATH_DETECTION_QC_PAYLOAD_COLUMNS)

# ── per-physical_embryo death_event table ──────────────────────────────────────────────────────
DEATH_EVENT_PAYLOAD_COLUMNS: tuple[str, ...] = ("death_event_time_index", "death_event_stage_hpf")
DEATH_EVENT_TABLE_COLUMNS: list[str] = list(PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + DEATH_EVENT_PAYLOAD_COLUMNS)


def validate_death_detection_qc(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "death_detection_qc",
) -> None:
    """Fail loud unless ``df`` is a valid per-snip death flag table (snip grain first, then flags)."""
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )
    _require_columns(df, DEATH_DETECTION_QC_TABLE_COLUMNS, scope_label)
    # An empty well (0 snips) has nothing to validate — a 0-row CSV round-trip always comes back
    # as `object` dtype (no True/False tokens to infer bool from), so the dtype check inside
    # _require_non_null_bool would otherwise reject a legitimately-empty, correctly-schemaed well.
    if df.empty:
        return
    for col in DEATH_DETECTION_QC_FLAG_COLUMNS:
        _require_non_null_bool(df, col, scope_label)

    from data_pipeline.quality_control.applicability import (
        ALLOWED_QC_APPLICABILITY,
        QC_APPLICABILITY_NOT_APPLICABLE,
    )

    applicability = df[DEATH_DETECTION_QC_APPLICABILITY_COLUMN].astype(str)
    unknown = sorted(set(applicability) - ALLOWED_QC_APPLICABILITY)
    if unknown:
        raise ValueError(
            f"{scope_label}: {DEATH_DETECTION_QC_APPLICABILITY_COLUMN} has unknown "
            f"value(s) {unknown}."
        )
    if applicability.eq(QC_APPLICABILITY_NOT_APPLICABLE).any():
        raise ValueError(
            f"{scope_label}: death evidence is computed from fraction_alive; use "
            "'diagnostic_only' when it must not exclude a snip."
        )


def validate_death_event(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "death_event",
) -> None:
    """Fail loud unless ``df`` is a valid death_event table (physical-embryo grain, no embryo_id)."""
    validate_snip_grain_identity_columns(
        df,
        grain="physical_embryo_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )
    _require_columns(df, DEATH_EVENT_TABLE_COLUMNS, scope_label)
    if "embryo_id" in df.columns:
        raise ValueError(
            f"{scope_label}: death_event is an animal-level table and must NOT carry embryo_id "
            "(that would over-specify it to a channel the animal does not have)."
        )
    death_time = pd.to_numeric(df["death_event_time_index"], errors="coerce")
    if death_time.isna().any():
        raise ValueError(
            f"{scope_label}: 'death_event_time_index' has null/non-numeric value(s)."
        )
    if not np.isfinite(death_time.to_numpy(dtype=float)).all():
        raise ValueError(f"{scope_label}: 'death_event_time_index' has non-finite value(s).")

    # Stage is an annotation. Missing start age or temperature deliberately produces a null stage
    # without discarding an otherwise valid death event.
    stage_raw = df["death_event_stage_hpf"]
    stage = pd.to_numeric(stage_raw, errors="coerce")
    if (stage_raw.notna() & stage.isna()).any():
        raise ValueError(
            f"{scope_label}: 'death_event_stage_hpf' has non-numeric value(s)."
        )
    finite_stage = stage.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_stage).all():
        raise ValueError(f"{scope_label}: 'death_event_stage_hpf' has non-finite value(s).")


def _require_columns(df: pd.DataFrame, required: list[str], scope_label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. Expected {required}."
        )


def _require_non_null_bool(df: pd.DataFrame, col: str, scope_label: str) -> None:
    if df[col].isna().any():
        raise ValueError(f"{scope_label}: flag column {col!r} has null value(s).")
    if df[col].dtype != bool:
        raise ValueError(f"{scope_label}: flag column {col!r} must be boolean dtype, got {df[col].dtype}.")
