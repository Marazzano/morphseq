"""Schema definition for quality control flags."""

from __future__ import annotations

SNIP_EXCLUSION_FLAGS = [
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
    "viability_flag",
    "dead_flag",
    "sa_outlier_flag",
    "focus_flag",
    "motion_flag",
]

# Informational (non-exclusion) flags. The yolk/bubble auxiliary-mask flags were retired with the
# full-frame auxiliary-mask path: they were an unimplemented stub (always False) and never fed
# use_snip. None currently defined; the list is kept so consolidate_qc's `*SNIP_INFORMATIONAL_FLAGS`
# splices remain valid (empty) and a future informational flag has an obvious home.
SNIP_INFORMATIONAL_FLAGS: list[str] = []

QC_FAIL_FLAGS = SNIP_EXCLUSION_FLAGS

QC_OUTPUT_COLUMNS = [
    "snip_id",
    "use_snip",
    *SNIP_EXCLUSION_FLAGS,
    *SNIP_INFORMATIONAL_FLAGS,
    "death_inflection_time_int",
    "death_predicted_stage_hpf",
]

REQUIRED_COLUMNS_QC = QC_OUTPUT_COLUMNS
