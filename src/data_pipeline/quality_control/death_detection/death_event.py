"""death_detection death_event — lead-time conversion + stage-at-death, per physical embryo.

The lead-time adjustment is HOURS-based (the spec's hard requirement): the raw inflection
``time_index`` is converted to elapsed hours via the frame-timing input, ``lead_time_hr`` is
subtracted in hours, and the result maps back to the called-death frame ``D`` — the latest frame
whose elapsed time is at or before the adjusted elapsed time (clamped to the animal's first frame).
``death_event_stage_hpf`` is ``stage_predictions`` sampled at ``D`` — "stage at the inferred death
event," not a raw stage-model death prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    stage_predictions_df: pd.DataFrame,
    *,
    config: DeathDetectionConfig,
) -> pd.DataFrame:
    """Return the per-physical_embryo death_event rows (one per persistence-dead animal).

    ``persistence_deaths_df``: columns experiment_id, well_id, physical_embryo_id, inflection_time_index.
    ``frame_timing_df``: experiment_id, well_id, time_index, elapsed_time_s.
    ``stage_predictions_df``: experiment_id, well_id, physical_embryo_id, time_index, predicted_stage_hpf
      (stage carries the full snip spine upstream; we group it to the animal here).
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
        stage_hpf = _stage_at_frame(stage_predictions_df, exp, well, phys, death_d)
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


def _stage_at_frame(stage_predictions_df, exp, well, phys, death_d) -> float:
    """Sample predicted_stage_hpf for one animal at the called-death frame, failing loud if absent."""
    match = stage_predictions_df[
        (stage_predictions_df["experiment_id"] == exp)
        & (stage_predictions_df["well_id"] == well)
        & (stage_predictions_df["physical_embryo_id"] == phys)
        & (stage_predictions_df[_TIME] == death_d)
    ]
    if len(match) == 0:
        raise ValueError(
            f"death_event: no stage_predictions row for physical_embryo_id {phys!r} at "
            f"time_index {death_d} (the called-death frame). stage_predictions is required for "
            "death_event."
        )
    return float(match["predicted_stage_hpf"].iloc[0])
