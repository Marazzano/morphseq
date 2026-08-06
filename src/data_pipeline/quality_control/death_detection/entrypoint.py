"""death_detection entrypoint — the thin filesystem adapter for BOTH output grains.

Loads fraction_alive (full snip spine + time_index + fraction_alive), frame timing (elapsed_time_s
per experiment/well/time_index, from frame_inventory), plate metadata (for well-level stage at
death), the snip_inventory universe, and the per-well registry. Computes the two flags + the
inflection list, builds the death_event table, reconciles both onto their grains, validates both
(registry as verifier), then writes BOTH per-well shards. No domain logic here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import compute_death_detection_flags
from .config import resolve_config
from .contract import validate_death_detection_qc, validate_death_event
from .death_event import compute_death_event
from .grain_reconciliation import (
    reconcile_death_events_to_physical_embryo_grain,
    reconcile_death_flags_to_snip_grain,
)

_FRAME_TIMING_COLUMNS = ("experiment_id", "well_id", "time_index", "elapsed_time_s")


def run_death_detection(
    *,
    fraction_alive_csv: Path,
    frame_inventory_csv: Path,
    plate_metadata_csv: Path,
    snip_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_qc_csv: Path,
    output_death_event_csv: Path,
    config_overrides: dict | None = None,
) -> None:
    config = resolve_config(config_overrides)

    fraction_alive = pd.read_csv(fraction_alive_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    plate_metadata = pd.read_csv(plate_metadata_csv)
    snip_inventory = pd.read_csv(snip_inventory_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    frame_timing = _frame_timing(frame_inventory)

    flags, inflections = compute_death_detection_flags(fraction_alive, frame_timing, config=config)

    # ── per-snip death flag table ──
    qc_df = reconcile_death_flags_to_snip_grain(
        flags, snip_inventory, frame_inventory_df=frame_inventory
    )
    validate_death_detection_qc(qc_df, physical_embryo_registry_df=registry, check_sources=True)
    _write_csv(qc_df, output_qc_csv)

    # ── per-physical_embryo death_event table ──
    death_events = compute_death_event(inflections, frame_timing, plate_metadata, config=config)
    physical_embryo_universe = (
        snip_inventory[["experiment_id", "well_id", "physical_embryo_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    death_event_df = reconcile_death_events_to_physical_embryo_grain(
        death_events, physical_embryo_universe
    )
    validate_death_event(death_event_df, physical_embryo_registry_df=registry, check_sources=True)
    _write_csv(death_event_df, output_death_event_csv)


def _frame_timing(frame_inventory_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _FRAME_TIMING_COLUMNS if c not in frame_inventory_df.columns]
    if missing:
        raise ValueError(
            f"death_detection: frame_inventory missing timing column(s) {missing}. "
            "lead_time_hr conversion requires elapsed_time_s keyed by (experiment_id, well_id, time_index)."
        )
    return frame_inventory_df[list(_FRAME_TIMING_COLUMNS)].drop_duplicates().reset_index(drop=True)


def _write_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
