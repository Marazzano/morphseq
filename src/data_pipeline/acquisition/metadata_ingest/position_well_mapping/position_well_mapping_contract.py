"""Contract for the canonical position-to-well identity bridge."""

from __future__ import annotations

import pandas as pd

from data_pipeline.shared.identifiers import build_well_id

REQUIRED_POSITION_WELL_MAPPING_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "position_index",
    "well_index",
    "well_id",
    "mapping_method",
)

RETIRED_POSITION_WELL_MAPPING_COLUMNS: tuple[str, ...] = ("series_number",)


def validate_position_well_mapping(
    df: pd.DataFrame,
    *,
    scope_label: str = "position_well_mapping",
) -> None:
    """Fail unless ``df`` follows the canonical position mapping contract."""
    retired = [col for col in RETIRED_POSITION_WELL_MAPPING_COLUMNS if col in df.columns]
    if retired:
        raise ValueError(
            f"[{scope_label}] retired column(s) present: {retired}. "
            "Regenerate position_well_mapping.csv; new code requires position_index."
        )

    missing = [col for col in REQUIRED_POSITION_WELL_MAPPING_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[{scope_label}] missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"[{scope_label}] mapping is empty.")

    checked = df.copy()
    checked["experiment_id"] = checked["experiment_id"].astype(str)
    checked["well_index"] = checked["well_index"].astype(str)
    checked["well_id"] = checked["well_id"].astype(str)
    checked["position_index"] = pd.to_numeric(
        checked["position_index"], errors="raise"
    ).astype(int)

    if checked["well_index"].isna().any() or (checked["well_index"].str.len() == 0).any():
        raise ValueError(f"[{scope_label}] well_index contains empty values.")

    expected_well_id = [
        build_well_id(experiment_id, well_index)
        for experiment_id, well_index in zip(checked["experiment_id"], checked["well_index"])
    ]
    bad_well_id = checked["well_id"] != expected_well_id
    if bad_well_id.any():
        sample = checked.loc[
            bad_well_id, ["experiment_id", "well_index", "well_id"]
        ].head(5).to_dict(orient="records")
        raise ValueError(
            f"[{scope_label}] well_id is inconsistent with experiment_id + well_index. "
            f"First offenders: {sample}"
        )

    duplicate_position = checked.duplicated(
        subset=["experiment_id", "position_index"], keep=False
    )
    if duplicate_position.any():
        sample = checked.loc[
            duplicate_position, ["experiment_id", "position_index", "well_index", "well_id"]
        ].head(10).to_dict(orient="records")
        raise ValueError(
            f"[{scope_label}] duplicate experiment_id + position_index rows. "
            f"First offenders: {sample}"
        )
