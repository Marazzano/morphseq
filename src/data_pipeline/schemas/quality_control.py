"""Schema definition for quality control flags.

LEGACY-RETIREMENT DEBT (feature_world.md): this is the OLD qc_flags.csv vocabulary
(viability_flag/dead_flag/motion_flag/death_inflection_time_int) for the parked
analysis_ready path. It is NOT the refactored snip_qc contract — the live snip_qc
verdict lives in quality_control/snip_qc/contract.py::SNIP_QC_EXCLUSION_FLAGS and does
NOT read this module. This schema only feeds the legacy quality_control/validators.py,
quality_control/io/{loaders,writers}.py, and schemas/analysis_ready.py -> analysis_ready/*
chain, none of which is wired into the live DAG. Delete this module (and that chain)
as part of the analysis_ready retirement, not the config/backends cleanup.
"""

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
