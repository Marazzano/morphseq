"""Shared vocabulary for modality-aware QC applicability."""

from __future__ import annotations

QC_APPLICABILITY_EXCLUSION = "exclusion"
QC_APPLICABILITY_DIAGNOSTIC_ONLY = "diagnostic_only"
QC_APPLICABILITY_NOT_APPLICABLE = "not_applicable"

ALLOWED_QC_APPLICABILITY: frozenset[str] = frozenset(
    {
        QC_APPLICABILITY_EXCLUSION,
        QC_APPLICABILITY_DIAGNOSTIC_ONLY,
        QC_APPLICABILITY_NOT_APPLICABLE,
    }
)

FLAG_APPLICABILITY_COLUMNS: dict[str, str] = {
    "viability_dead_flag": "death_detection_qc_applicability",
    "persistence_dead_flag": "death_detection_qc_applicability",
    "sa_outlier_flag": "surface_area_qc_applicability",
    "focus_flag": "focus_qc_applicability",
    "motion_blur_flag": "motion_blur_qc_applicability",
}


def applicability_column_for_flag(flag_column: str) -> str | None:
    return FLAG_APPLICABILITY_COLUMNS.get(str(flag_column))


__all__ = [
    "ALLOWED_QC_APPLICABILITY",
    "FLAG_APPLICABILITY_COLUMNS",
    "QC_APPLICABILITY_DIAGNOSTIC_ONLY",
    "QC_APPLICABILITY_EXCLUSION",
    "QC_APPLICABILITY_NOT_APPLICABLE",
    "applicability_column_for_flag",
]
