"""death_detection death_event — lead-time conversion + stage-at-death, per physical embryo.

The lead-time adjustment is HOURS-based (the spec's hard requirement): the raw inflection
``time_index`` is converted to elapsed hours via the frame-timing input, ``lead_time_hr`` is
subtracted in hours, and the result maps back to the called-death frame ``D`` — the latest frame
whose elapsed time is at or before the adjusted elapsed time (clamped to the animal's first frame).
``death_event_stage_hpf`` is calculated at ``D`` from the well's plate metadata and frame timing.
Stage is a well/time property, not a snip property: calculating it directly avoids requiring the
dying embryo to retain a valid snip at the called-death frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.feature_extraction.stage_inference import predict_stage_hpf

from .config import DeathDetectionConfig

_TIME = "time_index"


def called_death_time_index(
    animal_timing_df: pd.DataFrame,
    inflection_time_index: int,
    *,
    lead_time_hr: float,
) -> int:
    """Map the raw inflection frame to the lead-time-adjusted called-death frame ``D`` (hours-based).

    ``animal_timing_df`` carries ``time_index`` + ``elapsed_time_s`` for one animal (one row per
    time_index). Converts the inflection's elapsed time to hours, subtracts ``lead_time_hr``, and
    returns the latest frame whose elapsed time is <= the adjusted time (clamped to the earliest
    frame so D never precedes the animal's first observation).
    """
    timing = animal_timing_df.dropna(subset=["elapsed_time_s"]).sort_values(_TIME)
    if len(timing) == 0:
        raise ValueError("death_event: animal has no frame timing (elapsed_time_s) to convert lead time.")
    by_time = timing.set_index(_TIME)["elapsed_time_s"].astype(float)
    if inflection_time_index not in by_time.index:
        raise ValueError(
            f"death_event: inflection time_index {inflection_time_index} not in this animal's "
            "frame timing — the timing join is keyed (experiment_id, well_id, time_index)."
        )
    inflection_elapsed_s = float(by_time.loc[inflection_time_index])
    adjusted_elapsed_s = inflection_elapsed_s - lead_time_hr * 3600.0

    at_or_before = by_time[by_time <= adjusted_elapsed_s]
    if len(at_or_before) == 0:
        return int(by_time.index.min())  # adjusted time precedes the first frame -> clamp
    return int(at_or_before.index[-1])


def compute_death_event(
    persistence_deaths_df: pd.DataFrame,
    frame_timing_df: pd.DataFrame,
    plate_metadata_df: pd.DataFrame,
    *,
    config: DeathDetectionConfig,
) -> pd.DataFrame:
    """Return the per-physical_embryo death_event rows (one per persistence-dead animal).

    ``persistence_deaths_df``: columns experiment_id, well_id, physical_embryo_id, inflection_time_index.
    ``frame_timing_df``: experiment_id, well_id, time_index, elapsed_time_s.
    ``plate_metadata_df``: one row per well with start_age_hpf and temperature.
    Output columns: experiment_id, well_id, physical_embryo_id, death_event_time_index,
    death_event_stage_hpf.
    """
    rows: list[dict] = []
    for _, animal in persistence_deaths_df.iterrows():
        exp = animal["experiment_id"]
        well = animal["well_id"]
        phys = animal["physical_embryo_id"]
        inflection = int(animal["inflection_time_index"])

        animal_timing = frame_timing_df[
            (frame_timing_df["experiment_id"] == exp)
            & (frame_timing_df["well_id"] == well)
        ]
        death_d = called_death_time_index(animal_timing, inflection, lead_time_hr=config.lead_time_hr)
        stage_hpf = _stage_at_frame(frame_timing_df, plate_metadata_df, exp, well, death_d)
        rows.append(
            {
                "experiment_id": exp,
                "well_id": well,
                "physical_embryo_id": phys,
                "death_event_time_index": death_d,
                "death_event_stage_hpf": stage_hpf,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "experiment_id",
            "well_id",
            "physical_embryo_id",
            "death_event_time_index",
            "death_event_stage_hpf",
        ],
    )


def _stage_at_frame(
    frame_timing_df: pd.DataFrame,
    plate_metadata_df: pd.DataFrame,
    exp: str,
    well: str,
    death_d: int,
) -> float:
    """Calculate well-level developmental stage at the called-death frame.

    Missing age or temperature is an unresolved annotation, not a pipeline failure, and therefore
    returns NaN. Structural problems (missing/ambiguous well or timing rows) still fail loudly.
    """
    timing = frame_timing_df[
        (frame_timing_df["experiment_id"] == exp)
        & (frame_timing_df["well_id"] == well)
        & (frame_timing_df[_TIME] == death_d)
    ]
    if len(timing) != 1:
        raise ValueError(
            f"death_event: expected exactly one frame-timing row for well_id {well!r} at "
            f"time_index {death_d}, found {len(timing)}."
        )

    plate = plate_metadata_df[
        (plate_metadata_df["experiment_id"] == exp)
        & (plate_metadata_df["well_id"] == well)
    ]
    if len(plate) != 1:
        raise ValueError(
            f"death_event: expected exactly one plate_metadata row for well_id {well!r}, "
            f"found {len(plate)}."
        )
    plate_row = plate.iloc[0]
    if "start_age_hpf" not in plate_row.index or pd.isna(plate_row["start_age_hpf"]):
        return float("nan")
    if "temperature" not in plate_row.index or pd.isna(plate_row["temperature"]):
        return float("nan")
    return predict_stage_hpf(
        float(plate_row["start_age_hpf"]),
        float(timing["elapsed_time_s"].iloc[0]),
        float(plate_row["temperature"]),
    )
