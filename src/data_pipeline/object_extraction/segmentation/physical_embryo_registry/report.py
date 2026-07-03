"""physical_embryo_registry report — embryos-per-well. TERMINAL leaf (report_world.md). TWO artifacts:
  1. distribution (renderer B): how many wells have 1/2/3/... embryos — cohort structure;
  2. 96-well plate heatmap: the SAME per-well count laid out in physical 8x12 plate geometry, so
     spatial patterns (edge effects, a dead column, an empty quadrant) are visible.

Consumes only the registry's own merged output (one row per physical_embryo_id). Consumed by
nothing; imported by nothing but its own tasks.py subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.viz.reporting import plot_count_per_group, plot_plate_heatmap


def build_physical_embryo_registry_report(
    *,
    physical_embryo_registry_csv: Path,
    output_embryos_per_well_png: Path,
    output_embryos_per_well_plate_png: Path,
) -> list[Path]:
    registry = pd.read_csv(physical_embryo_registry_csv)

    for output_png in (output_embryos_per_well_png, output_embryos_per_well_plate_png):
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
    return [distribution, plate]
