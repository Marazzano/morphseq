"""mask_geometry report — feature histogram grid (renderer D). TERMINAL leaf (report_world.md).

Consumes only mask_geometry's own merged output (+ snip_inventory for gallery image paths, which
mask_geometry already reads to key its snips). The general "look at the distribution of every
feature" primitive: one gridded PNG, one panel per payload column. No cutoffs here (geometry
features have no pass/fail gate of their own), so every panel is a plain distribution.

Consumed by nothing; imported by nothing but its own tasks.py subcommand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.object_extraction.snip_processing.io import resolve_snip_inventory_image_paths
from data_pipeline.viz.reporting import plot_histogram_grid, render_value_quartile_gallery

# The mask_geometry payload columns (measured geometry per snip). Kept explicit rather than
# "all numeric columns" so identity/frame columns (time_index etc.) are not plotted as features.
GEOMETRY_FEATURE_COLUMNS = [
    "area_um2", "perimeter_um", "length_um", "width_um", "centroid_x_um", "centroid_y_um",
]


def build_mask_geometry_report(
    *,
    mask_geometry_csv: Path,
    snip_inventory_csv: Path,
    output_root: Path,
    output_geometry_feature_grid_png: Path,
    output_area_um2_quartile_gallery_png: Path,
) -> list[Path]:
    geom = pd.read_csv(mask_geometry_csv)
    snip_inventory = pd.read_csv(snip_inventory_csv)

    for output_png in (output_geometry_feature_grid_png, output_area_um2_quartile_gallery_png):
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    grid = plot_histogram_grid(
        geom,
        GEOMETRY_FEATURE_COLUMNS,
        title="mask_geometry — feature distributions",
        output_path=output_geometry_feature_grid_png,
    )
    # No threshold here → plain VALUE quartiles (what does a small / mid / large embryo look like?),
    # not a pass/fail cutoff gallery. area_um2 is the most interpretable geometry axis for eyeballing.
    gallery = render_value_quartile_gallery(
        geom.merge(resolve_snip_inventory_image_paths(snip_inventory, output_root=Path(output_root)), on="snip_id", how="left"),
        "area_um2",
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="mask_geometry — area_um2 value quartiles",
        output_path=output_area_um2_quartile_gallery_png,
    )
    return [grid, gallery]
