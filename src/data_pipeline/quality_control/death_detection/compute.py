"""death_detection compute — the public orchestrator + the per-frame viability flag.

Two independent per-snip facts, kept separate (never pre-merged):
  - viability_dead_flag: per FRAME — this frame's own fraction_alive <= dead_fraction_threshold;
  - persistence_dead_flag: per ANIMAL — the fraction_alive trace inflects and stays down, the
    inflection is lead-time-adjusted to frame D (death_event.py), broadcast as time_index >= D.

Persistence groups by ``physical_embryo_id`` (the animal). Each animal must carry exactly ONE
fraction_alive trace (single viability channel, chosen upstream by fraction_alive) — fail loud
otherwise. This module returns the per-snip flag table AND the per-animal inflection list (the
input to death_event); grain_reconciliation lands them on the universe.
"""

from __future__ import annotations

import pandas as pd

from .config import DeathDetectionConfig
from .death_event import called_death_time_index
from .persistence import broadcast_persistence_dead_flag, detect_inflection_time_index

_TIME = "time_index"
_FRACTION = "fraction_alive"
_REQUIRED_FRACTION_ALIVE_COLUMNS = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
    "time_index",
    "fraction_alive",
)


def compute_viability_dead_flag(
    fraction_alive_df: pd.DataFrame, *, dead_fraction_threshold: float
) -> pd.Series:
    """Per-frame viability flag: True where this frame's own fraction_alive <= threshold.

    Null fraction_alive (empty mask, no defined viability) is NOT dead — it is False, not a death
    call. Index-aligned to ``fraction_alive_df``.
    """
    fraction = pd.to_numeric(fraction_alive_df[_FRACTION], errors="coerce")
    return (fraction <= dead_fraction_threshold).fillna(False).astype(bool)


def compute_death_detection_flags(
    fraction_alive_df: pd.DataFrame,
    frame_timing_df: pd.DataFrame,
    *,
    config: DeathDetectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-snip flag table, per-animal inflection table).

    Per-snip table: snip_id + viability_dead_flag + persistence_dead_flag (raw, pre-universe).
    Inflection table: experiment_id, well_id, physical_embryo_id, inflection_time_index — one row
    per persistence-dead animal (feeds death_event; the broadcast frame D is computed there).
    """
    missing = [c for c in _REQUIRED_FRACTION_ALIVE_COLUMNS if c not in fraction_alive_df.columns]
    if missing:
        raise ValueError(f"death_detection: fraction_alive input missing column(s) {missing}.")

    work = fraction_alive_df.copy()
    work["viability_dead_flag"] = compute_viability_dead_flag(
        work, dead_fraction_threshold=config.dead_fraction_threshold
    )
    work["persistence_dead_flag"] = False

    inflection_rows: list[dict] = []
    timing_by_well = frame_timing_df

    for phys, animal in work.groupby("physical_embryo_id", sort=False):
        _require_single_channel(animal, phys)
        inflection = detect_inflection_time_index(animal, config=config)
        if inflection is None:
            continue
        exp = animal["experiment_id"].iloc[0]
        well = animal["well_id"].iloc[0]
        animal_timing = timing_by_well[
            (timing_by_well["experiment_id"] == exp) & (timing_by_well["well_id"] == well)
        ]
        death_d = called_death_time_index(animal_timing, inflection, lead_time_hr=config.lead_time_hr)
        flag = broadcast_persistence_dead_flag(animal, death_d)
        work.loc[animal.index, "persistence_dead_flag"] = flag.to_numpy()
        inflection_rows.append(
            {
                "experiment_id": exp,
                "well_id": well,
                "physical_embryo_id": phys,
                "inflection_time_index": inflection,
            }
        )

    flags = work[["snip_id", "viability_dead_flag", "persistence_dead_flag"]].copy()
    flags["viability_dead_flag"] = flags["viability_dead_flag"].astype(bool)
    flags["persistence_dead_flag"] = flags["persistence_dead_flag"].astype(bool)

    inflections = pd.DataFrame(
        inflection_rows,
        columns=["experiment_id", "well_id", "physical_embryo_id", "inflection_time_index"],
    )
    return flags, inflections


def _require_single_channel(animal: pd.DataFrame, physical_embryo_id) -> None:
    """Fail loud if an animal carries more than one fraction_alive trace (one row per time_index)."""
    dup_times = animal[_TIME][animal[_TIME].duplicated()].unique().tolist()
    if dup_times:
        raise ValueError(
            f"death_detection: physical_embryo_id {physical_embryo_id!r} has more than one "
            f"fraction_alive value at time_index(es) {dup_times[:5]} — death is an animal-level fact "
            "and requires exactly ONE viability trace per animal. The single viability channel is "
            "chosen upstream by fraction_alive (viability_channel_id); death_detection does not "
            "combine channels."
        )
