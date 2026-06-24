"""death_detection persistence — the per-animal death-inflection algorithm.

Grouping is by ``physical_embryo_id`` (the animal — NOT ``embryo_id``, which is channel-specific
and would split one animal across channels). Within an animal we sort by ``time_index``, find a
sustained ``fraction_alive`` decline candidate, and validate that the embryo STAYS down after it
(post-inflection dead fraction >= persistence_threshold).

This module produces the RAW inflection ``time_index`` only. The lead-time-adjusted called-death
frame ``D`` (hours-based) is computed in ``death_event.py``; the snip broadcast uses ``D``.
Ported from the legacy ``core/death_detection.py`` math, retargeted to the spine axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import DeathDetectionConfig

_TIME = "time_index"
_FRACTION = "fraction_alive"


def find_inflection_candidates(
    physical_embryo_fraction_alive_df: pd.DataFrame, *, config: DeathDetectionConfig
) -> list[tuple[int, float]]:
    """Return (time_index, decline_rate) candidates where smoothed fraction_alive drops fast.

    The input is one animal's trace (already single-channel — one row per time_index). Rows with
    a null fraction_alive are dropped before differencing.
    """
    data = physical_embryo_fraction_alive_df.dropna(subset=[_FRACTION]).sort_values(_TIME)
    if len(data) < config.min_timepoints:
        return []
    times = data[_TIME].to_numpy()
    fractions = data[_FRACTION].to_numpy(dtype=float)
    if len(times) < 3 or np.all(np.isnan(fractions)):
        return []

    if len(fractions) >= 5:
        window = min(config.smoothing_window, len(fractions))
        if window % 2 == 0:
            window -= 1
        window = max(window, 3)
        smoothed = savgol_filter(fractions, window_length=window, polyorder=min(2, window - 1))
    else:
        smoothed = fractions

    dt = np.diff(times)
    rates = np.diff(smoothed) / dt
    candidates: list[tuple[int, float]] = []
    for index, rate in enumerate(rates):
        if rate < -config.decline_rate_threshold:
            candidates.append((int(times[index]), float(rate)))
    return candidates


def validate_death_persistence(
    physical_embryo_fraction_alive_df: pd.DataFrame,
    inflection_time_index: int,
    *,
    config: DeathDetectionConfig,
) -> bool:
    """True if the animal STAYS dead after ``inflection_time_index`` (persistence test)."""
    data = physical_embryo_fraction_alive_df.dropna(subset=[_FRACTION])
    post = data[data[_TIME] > inflection_time_index]
    if len(post) == 0:
        return False
    dead_count = int((post[_FRACTION].astype(float) <= config.dead_fraction_threshold).sum())
    return (dead_count / len(post)) >= config.persistence_threshold


def detect_inflection_time_index(
    physical_embryo_fraction_alive_df: pd.DataFrame, *, config: DeathDetectionConfig
) -> int | None:
    """Return the earliest persistent inflection ``time_index`` for one animal, or None.

    Walks decline candidates earliest-first; the first one that passes the persistence test wins.
    Non-persistent candidates (transient dips) are skipped, scanning later in the trace.
    """
    data = physical_embryo_fraction_alive_df.sort_values(_TIME).copy()
    remaining = data
    while len(remaining.dropna(subset=[_FRACTION])) >= config.min_timepoints:
        candidates = find_inflection_candidates(remaining, config=config)
        if not candidates:
            return None
        earliest_time, _ = candidates[0]
        if validate_death_persistence(data, earliest_time, config=config):
            return earliest_time
        remaining = remaining[remaining[_TIME] > earliest_time]
    return None


def broadcast_persistence_dead_flag(
    physical_embryo_fraction_alive_df: pd.DataFrame, called_death_time_index: int
) -> pd.Series:
    """Return a bool Series (index-aligned to the input) = (time_index >= D) for one animal."""
    return physical_embryo_fraction_alive_df[_TIME] >= called_death_time_index
