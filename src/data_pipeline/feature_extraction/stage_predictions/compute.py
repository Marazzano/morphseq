"""stage_predictions compute — Kimmel1995 stage (hpf) per snip from plate metadata + frame timing.

No mask reading: stage is predicted from start_age_hpf + temperature (plate_metadata, by well_id)
and elapsed_time_s (frame_inventory, by image_id). Reuses the legacy pure predict_stage_hpf.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import SNIP_FEATURE_TABLE_SPINE_COLUMNS
from data_pipeline.feature_extraction.stage_inference import predict_stage_hpf

from .contract import STAGE_PREDICTION_TABLE_COLUMNS

MODEL_VERSION = "kimmel1995_temp_rate_v1"

_TIME_COLUMNS: tuple[str, ...] = ("elapsed_time_s", "experiment_time_s", "time_s")


def _elapsed_time_s(frame_inventory_by_image: pd.DataFrame, image_id: str, snip_id: str) -> float:
    row = frame_inventory_by_image.loc[image_id]
    for col in _TIME_COLUMNS:
        if col in row.index and not pd.isna(row[col]):
            return float(row[col])
    raise ValueError(
        f"stage_predictions: frame_inventory row for image_id {image_id!r} (snip {snip_id!r}) has "
        f"no frame timing. Expected one of {_TIME_COLUMNS}."
    )


def compute_stage_prediction_features(
    snip_inventory_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    plate_metadata_df: pd.DataFrame,
    *,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Return one stage_prediction_features row per snip."""
    frame_inventory_by_image = frame_inventory_df.set_index("image_id")
    plate_by_well = plate_metadata_df.set_index("well_id")

    rows: list[dict] = []
    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])
        well_id = str(snip["well_id"])
        image_id = str(snip["image_id"])

        if well_id not in plate_by_well.index:
            raise ValueError(
                f"stage_predictions: well_id {well_id!r} (snip {snip_id!r}) not in plate_metadata. "
                "Every snip's well must have a plate_metadata row with start_age_hpf + temperature."
            )
        plate = plate_by_well.loc[well_id]
        for col in ("start_age_hpf", "temperature"):
            if col not in plate.index or pd.isna(plate[col]):
                raise ValueError(
                    f"stage_predictions: plate_metadata for well {well_id!r} is missing {col!r}."
                )

        elapsed = _elapsed_time_s(frame_inventory_by_image, image_id, snip_id)
        predicted = predict_stage_hpf(
            float(plate["start_age_hpf"]), elapsed, float(plate["temperature"])
        )

        row = {col: snip[col] for col in SNIP_FEATURE_TABLE_SPINE_COLUMNS}
        row["predicted_stage_hpf"] = predicted
        row["model_version"] = model_version
        rows.append(row)

    return pd.DataFrame(rows, columns=STAGE_PREDICTION_TABLE_COLUMNS)
