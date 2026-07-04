"""physical_embryo_registry report — embryos-per-well. TIER-1 / named ask. TWO artifacts:
  1. distribution (renderer B): how many wells have 1/2/3/... embryos — cohort structure;
  2. 96-well plate heatmap: the SAME per-well count laid out in physical 8x12 plate geometry, so
     spatial patterns (edge effects, a dead column, an empty quadrant) are visible.

Terminal report. Consumes only the registry's own merged output (one row per physical_embryo_id).
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.viz.reporting import plot_count_per_group, plot_plate_heatmap

from .._loaders import merged


def build(output_dir: Path) -> list[Path]:
    registry = merged("object_extraction", "physical_embryo_registry", "physical_embryo_registry")

    distribution = plot_count_per_group(
        registry,
        group_col="well_id",
        title="physical_embryo_registry — embryos per well",
        xlabel="physical embryos per well",
        output_path=output_dir / "embryos_per_well.png",
    )

    per_well_count = registry.groupby("well_id").size().reset_index(name="n_embryos")
    plate = plot_plate_heatmap(
        per_well_count,
        "n_embryos",
        well_col="well_id",
        title="physical_embryo_registry — embryos per well (plate layout)",
        cbar_label="embryos in well",
        output_path=output_dir / "embryos_per_well_plate.png",
    )
    return [distribution, plate]
