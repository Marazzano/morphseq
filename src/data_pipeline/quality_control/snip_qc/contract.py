"""snip_qc contract — the final per-snip QC verdict table.

Grain: one row per ``snip_id``. The verdict carries the FULL snip spine (imported from the minting
site, never re-typed) plus ``use_snip`` and ``qc_fail_reasons``. snip_qc is the final operational QC
table and must not be weaker than its inputs — it is not a minimal-key exception.

``SNIP_QC_EXCLUSION_FLAGS`` is the flat list of source flag columns that count as exclusions. The flag
column names ARE the vocabulary — ``qc_fail_reasons`` stores the pipe-joined exclusion flag-column
names that fired (e.g. ``"edge_flag|focus_flag"``); empty string means the snip passed. There is no
rename layer: a flag column is added here only when its source product exists and emits that column —
else snip_qc fails loud on a missing column.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

SNIP_QC_PAYLOAD_COLUMNS: tuple[str, ...] = ("use_snip", "qc_fail_reasons")
SNIP_QC_TABLE_COLUMNS: list[str] = list(SNIP_ID_SPINE_COLUMNS + SNIP_QC_PAYLOAD_COLUMNS)

# Default exclusion flag columns. This is the MVP semantic contract: the flag names themselves
# are the qc_fail_reasons vocabulary. Config may override it for permissive/strict QC runs; both
# planning and runtime must use the same resolved policy (see flag_input_resolver.py).
SNIP_QC_EXCLUSION_FLAGS: tuple[str, ...] = (
    "viability_dead_flag",
    "persistence_dead_flag",
    "sa_outlier_flag",
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
    "focus_flag",
    "motion_blur_flag",
)


def validate_snip_qc(
    df: pd.DataFrame,
    *,
    source: str = "",
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "snip_qc",
) -> None:
    """Fail loud unless ``df`` is a valid snip_qc verdict table (spine first, then verdict)."""
    label = f"{scope_label}:{source}" if source else scope_label
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=label,
    )

    missing = [c for c in SNIP_QC_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required column(s): {', '.join(missing)}. Expected {SNIP_QC_TABLE_COLUMNS}."
        )

    # An empty well (0 snips) has nothing to validate — a 0-row CSV round-trip always comes back
    # as `object` dtype (no True/False tokens to infer bool from), so the dtype check below would
    # otherwise reject a legitimately-empty, correctly-schemaed well.
    if df.empty:
        return

    if df["use_snip"].isna().any():
        raise ValueError(f"{label}: use_snip has null value(s); it must be non-null boolean.")
    if df["use_snip"].dtype != bool:
        raise ValueError(f"{label}: use_snip must be boolean dtype, got {df['use_snip'].dtype}.")

    reasons = df["qc_fail_reasons"]
    if reasons.isna().any():
        raise ValueError(f"{label}: qc_fail_reasons has null value(s); use empty string for a pass.")
    if reasons.dtype != object:
        raise ValueError(f"{label}: qc_fail_reasons must be a string column.")

    known = set(SNIP_QC_EXCLUSION_FLAGS)
    for value in reasons:
        if value == "":
            continue
        bad = [r for r in str(value).split("|") if r not in known]
        if bad:
            raise ValueError(
                f"{label}: qc_fail_reasons contains unknown flag(s) {bad}. "
                f"Known exclusion flags: {sorted(known)}."
            )

    # use_snip is true IFF there are no fail reasons.
    derived_use = reasons == ""
    if not (df["use_snip"] == derived_use).all():
        raise ValueError(
            f"{label}: use_snip must be true iff qc_fail_reasons == '' (the verdict and the switch disagree)."
        )
