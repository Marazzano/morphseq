"""snip_qc contract — the final per-snip QC verdict table.

Grain: one row per ``snip_id``. The verdict carries the FULL snip spine (imported from the minting
site, never re-typed) plus ``use_snip`` and ``qc_fail_reasons``. snip_qc is the final operational QC
table and must not be weaker than its inputs — it is not a minimal-key exception.

``SNIP_QC_EXCLUSION_REASONS`` maps each verdict reason to the source flag column it reads. The MVP
map covers the three migrated/specced exclusion families: death (two modes), surface area, and mask
quality (three flags). Future hooks (focus/blur/metadata) are NOT added until their products exist
and emit the named flag column — else snip_qc fails loud on a missing column.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

SNIP_QC_PAYLOAD_COLUMNS: tuple[str, ...] = ("use_snip", "qc_fail_reasons")
SNIP_QC_TABLE_COLUMNS: list[str] = list(SNIP_ID_SPINE_COLUMNS + SNIP_QC_PAYLOAD_COLUMNS)

# reason name -> source flag column. MVP only (see module docstring).
SNIP_QC_EXCLUSION_REASONS: dict[str, str] = {
    "dead_viability": "viability_dead_flag",
    "dead_persistence": "persistence_dead_flag",
    "surface_area_outlier": "sa_outlier_flag",
    "edge": "edge_flag",
    "discontinuous_mask": "discontinuous_mask_flag",
    "overlapping_mask": "overlapping_mask_flag",
}
# Future hooks — add ONLY when the source product lands and emits the named flag column:
#   "missing_metadata" -> "metadata_missing_flag"   (after metadata_completeness_qc)
#   "focus"            -> "focus_flag"               (after focus_qc; z-stack, in dev)
#   "blur"             -> "blur_flag"                (after blur_qc; z-stack, in dev)


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

    if df["use_snip"].isna().any():
        raise ValueError(f"{label}: use_snip has null value(s); it must be non-null boolean.")
    if df["use_snip"].dtype != bool:
        raise ValueError(f"{label}: use_snip must be boolean dtype, got {df['use_snip'].dtype}.")

    reasons = df["qc_fail_reasons"]
    if reasons.isna().any():
        raise ValueError(f"{label}: qc_fail_reasons has null value(s); use empty string for a pass.")
    if reasons.dtype != object:
        raise ValueError(f"{label}: qc_fail_reasons must be a string column.")

    known = set(SNIP_QC_EXCLUSION_REASONS)
    for value in reasons:
        if value == "":
            continue
        bad = [r for r in str(value).split("|") if r not in known]
        if bad:
            raise ValueError(
                f"{label}: qc_fail_reasons contains unknown reason(s) {bad}. "
                f"Known reasons: {sorted(known)}."
            )

    # use_snip is true IFF there are no fail reasons.
    derived_use = reasons == ""
    if not (df["use_snip"] == derived_use).all():
        raise ValueError(
            f"{label}: use_snip must be true iff qc_fail_reasons == '' (the verdict and the switch disagree)."
        )
