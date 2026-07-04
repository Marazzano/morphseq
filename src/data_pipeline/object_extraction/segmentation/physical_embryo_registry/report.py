"""physical_embryo_registry report — embryos-per-well. TERMINAL leaf (report_world.md). THREE artifacts:
  1. distribution (renderer B): how many wells have 1/2/3/... embryos — cohort structure;
  2. 96-well plate heatmap: the SAME per-well count laid out in physical 8x12 plate geometry, so
     spatial patterns (edge effects, a dead column, an empty quadrant) are visible.
  3. embryos-per-well OVER TIME (well x time_index heatmap): a death PROXY — a well whose embryo
     count DROPS over time flags a problematic well before any death model runs. The registry table
     is animal-grain (no time_index), so this reads the per-frame presence from the sibling
     object_extraction artifact `frame_masks` (well_id, time_index, track_id, is_valid_mask) and
     joins it to the registry on track_id → physical_embryo_id. Both inputs are object_extraction
     artifacts, so the report stays a stage-local leaf (no cross-stage forward dependency).

Consumes the registry's own merged output (one row per physical_embryo_id) plus the merged
frame_masks table (per-frame presence). Consumed by nothing; imported by nothing but its own
tasks.py subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.viz.reporting import (
    plot_count_per_group,
    plot_plate_heatmap,
    plot_well_survival_over_time,
)


def _embryos_present_per_well_time(registry: pd.DataFrame, frame_masks: pd.DataFrame) -> pd.DataFrame:
    """Count distinct physical embryos present per (well_id, time_index) — the over-time cohort size.

    frame_masks carries the per-frame track presence (well_id, time_index, track_id, is_valid_mask);
    the registry maps track_id → physical_embryo_id (identity origination). Only valid masks count as
    "present". Returns long rows (well_id, time_index, n_embryos) ready for the shared heatmap helper.
    """
    track_to_animal = registry.set_index("track_id")["physical_embryo_id"]
    fm = frame_masks[frame_masks["is_valid_mask"].astype(bool)].copy()
    fm["physical_embryo_id"] = fm["track_id"].map(track_to_animal)
    fm = fm.dropna(subset=["physical_embryo_id"])
    return (
        fm.groupby(["well_id", "time_index"])["physical_embryo_id"]
        .nunique()
        .reset_index(name="n_embryos")
    )


def build_physical_embryo_registry_report(
    *,
    physical_embryo_registry_csv: Path,
    frame_masks_csv: Path,
    output_embryos_per_well_png: Path,
    output_embryos_per_well_plate_png: Path,
    output_embryos_per_well_over_time_png: Path,
) -> list[Path]:
    registry = pd.read_csv(physical_embryo_registry_csv)
    frame_masks = pd.read_csv(frame_masks_csv)

    for output_png in (
        output_embryos_per_well_png,
        output_embryos_per_well_plate_png,
        output_embryos_per_well_over_time_png,
    ):
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    distribution = plot_count_per_group(
        registry,
        group_col="well_id",
        title="physical_embryo_registry — embryos per well",
        xlabel="physical embryos per well",
        output_path=output_embryos_per_well_png,
    )

    per_well_count = registry.groupby("well_id").size().reset_index(name="n_embryos")
    plate = plot_plate_heatmap(
        per_well_count,
        "n_embryos",
        well_col="well_id",
        title="physical_embryo_registry — embryos per well (plate layout)",
        cbar_label="embryos in well",
        output_path=output_embryos_per_well_plate_png,
    )

    # Embryos present per well over time_index — a DEATH PROXY (drop = problematic well). Uses the
    # shared well x time heatmap so the plot matches death_detection / analysis_ready exactly.
    over_time = _embryos_present_per_well_time(registry, frame_masks)
    over_time_plot = plot_well_survival_over_time(
        over_time,
        time_col="time_index",
        value_col="n_embryos",
        value_label="embryos present",
        title="physical_embryo_registry — embryos present per well over time (death proxy)",
        output_path=output_embryos_per_well_over_time_png,
    )
    return [distribution, plate, over_time_plot]
