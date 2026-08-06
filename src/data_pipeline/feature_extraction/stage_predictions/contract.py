"""stage_prediction_features contract — developmental stage (hpf) per snip.

Grain: one row per ``snip_id``. ``predicted_stage_hpf`` is finite when the source well has a
resolved start age and nullable when reconciliation intentionally passed through an unresolved
stage. ``stage_prediction_status`` makes that null explicit. ``model_version`` is method
provenance (a string), not a measured feature, so it is not range-checked here.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import (
    SNIP_FEATURE_TABLE_SPINE_COLUMNS,
    validate_feature_table,
)

STAGE_PREDICTION_PAYLOAD_COLUMNS: tuple[str, ...] = ("predicted_stage_hpf",)
STAGE_PREDICTION_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "model_version",
    "stage_prediction_status",
)
ALLOWED_STAGE_PREDICTION_STATUSES: frozenset[str] = frozenset(
    {"predicted", "missing_start_age_hpf", "missing_temperature"}
)

STAGE_PREDICTION_TABLE_COLUMNS: list[str] = list(
    SNIP_FEATURE_TABLE_SPINE_COLUMNS + STAGE_PREDICTION_PAYLOAD_COLUMNS + STAGE_PREDICTION_PROVENANCE_COLUMNS
)


def validate_stage_prediction_features(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "stage_prediction_features",
) -> None:
    """Fail loud unless ``df`` is a valid stage_prediction_features table (spine first)."""
    validate_feature_table(
        df,
        required_columns=STAGE_PREDICTION_TABLE_COLUMNS,
        feature_columns=STAGE_PREDICTION_PAYLOAD_COLUMNS,
        nullable_feature_columns=STAGE_PREDICTION_PAYLOAD_COLUMNS,
        scope_label=scope_label,
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
    )
    if df["model_version"].isna().any():
        raise ValueError(f"{scope_label}: model_version provenance must be non-null.")
    statuses = df["stage_prediction_status"].astype(str)
    unknown = sorted(set(statuses) - ALLOWED_STAGE_PREDICTION_STATUSES)
    if unknown:
        raise ValueError(
            f"{scope_label}: stage_prediction_status has unknown value(s) {unknown}; "
            f"allowed={sorted(ALLOWED_STAGE_PREDICTION_STATUSES)}."
        )
    predicted = statuses.eq("predicted")
    if df.loc[predicted, "predicted_stage_hpf"].isna().any():
        raise ValueError(
            f"{scope_label}: status='predicted' requires a non-null predicted_stage_hpf."
        )
    if df.loc[~predicted, "predicted_stage_hpf"].notna().any():
        raise ValueError(
            f"{scope_label}: unresolved stage statuses require predicted_stage_hpf to be null."
        )
