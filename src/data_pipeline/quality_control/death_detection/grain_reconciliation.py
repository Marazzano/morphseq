"""death_detection grain_reconciliation — land computed evidence on the two promised grains.

death_detection emits TWO validated output grains:
  1. snip_id grain          -> death_detection_qc (viability + persistence flags)
  2. physical_embryo_id grain -> death_event       (called-death time + stage)

The algorithm finds death (persistence.py); the event code names when it happened
(death_event.py); THIS module reconciles the results to the declared output grains — projecting
flags onto every snip in the universe and confirming the event table sits at physical-embryo grain.
Missing / duplicate / extra keys fail loud (a QC table must match its universe exactly).
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    PHYSICAL_EMBRYO_ID_SPINE_COLUMNS,
    SNIP_ID_SPINE_COLUMNS,
)

_SNIP_FLAG_COLUMNS = ("viability_dead_flag", "persistence_dead_flag")


def reconcile_death_flags_to_snip_grain(
    death_flags_df: pd.DataFrame, snip_universe_df: pd.DataFrame
) -> pd.DataFrame:
    """Project the per-snip death flags onto the full snip universe (full spine + two flags).

    ``death_flags_df`` carries ``snip_id`` + the two bool flags for the snips death detection saw.
    Every snip in the universe must be present exactly once; missing/dup/extra fail loud (no snip
    silently defaults). Returns one row per universe snip carrying SNIP_ID_SPINE_COLUMNS + flags.
    """
    universe = snip_universe_df.copy()
    _require_unique(universe, "snip_id", "death_detection universe")
    _require_unique(death_flags_df, "snip_id", "death_detection flags")

    universe_ids = set(universe["snip_id"].astype(str))
    flag_ids = set(death_flags_df["snip_id"].astype(str))
    missing = universe_ids - flag_ids
    extra = flag_ids - universe_ids
    if missing:
        raise ValueError(
            f"death_detection: {len(missing)} universe snip(s) have no computed flags, e.g. "
            f"{sorted(missing)[:5]}. Every snip must be decided (no silent default)."
        )
    if extra:
        raise ValueError(
            f"death_detection: {len(extra)} flagged snip(s) are not in the universe, e.g. "
            f"{sorted(extra)[:5]}."
        )

    flags_by_snip = death_flags_df.set_index(death_flags_df["snip_id"].astype(str))
    out = universe[list(SNIP_ID_SPINE_COLUMNS)].copy()
    snip_ids = out["snip_id"].astype(str)
    for col in _SNIP_FLAG_COLUMNS:
        out[col] = pd.array([bool(flags_by_snip.loc[s, col]) for s in snip_ids], dtype=bool)
    return out


def reconcile_death_events_to_physical_embryo_grain(
    death_events_df: pd.DataFrame, physical_embryo_universe_df: pd.DataFrame
) -> pd.DataFrame:
    """Confirm the death_event table sits at physical-embryo grain within the animal universe.

    One row per persistence-dead animal. Every event's physical_embryo_id must exist in the animal
    universe (no orphan events); duplicates fail loud. Returns the events carrying
    PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + the two event annotations (no embryo_id).
    """
    _require_unique(death_events_df, "physical_embryo_id", "death_event")
    known_animals = set(physical_embryo_universe_df["physical_embryo_id"].astype(str))
    orphans = set(death_events_df["physical_embryo_id"].astype(str)) - known_animals
    if orphans:
        raise ValueError(
            f"death_event: {len(orphans)} event(s) reference unknown physical_embryo_id(s), e.g. "
            f"{sorted(orphans)[:5]}."
        )
    columns = list(PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + ("death_event_time_index", "death_event_stage_hpf"))
    return death_events_df[columns].reset_index(drop=True)


def _require_unique(df: pd.DataFrame, key: str, label: str) -> None:
    if key not in df.columns:
        raise ValueError(f"{label}: missing key column {key!r}.")
    dupes = df[key][df[key].duplicated()].unique().tolist()
    if dupes:
        raise ValueError(f"{label}: duplicate {key} value(s) {dupes[:5]}; expected one row per {key}.")
