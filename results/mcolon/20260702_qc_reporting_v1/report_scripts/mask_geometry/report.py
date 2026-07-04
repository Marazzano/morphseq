"""mask_geometry report — feature histogram grid (renderer D). TIER-1 / named ask.

Terminal report. Consumes only mask_geometry's own merged output. The general "look at the
distribution of every feature" primitive: one gridded PNG, one panel per payload column. No cutoffs
here (geometry features have no pass/fail gate of their own), so every panel is a plain distribution.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.viz.reporting import plot_histogram_grid, render_value_quartile_gallery

from .._loaders import merged, snip_image_paths

# The mask_geometry payload columns (measured geometry per snip). Kept explicit rather than
# "all numeric columns" so identity/frame columns (time_index etc.) are not plotted as features.
GEOMETRY_FEATURE_COLUMNS = [
    "area_um2", "perimeter_um", "length_um", "width_um", "centroid_x_um", "centroid_y_um",
]


def build(output_dir: Path) -> list[Path]:
    geom = merged("feature_extraction", "mask_geometry", "mask_geometry")

    grid = plot_histogram_grid(
        geom,
        GEOMETRY_FEATURE_COLUMNS,
        title="mask_geometry — feature distributions",
        output_path=output_dir / "geometry_feature_grid.png",
    )
    # No threshold here → plain VALUE quartiles (what does a small / mid / large embryo look like?),
    # not a pass/fail cutoff gallery. area_um2 is the most interpretable geometry axis for eyeballing.
    gallery = render_value_quartile_gallery(
        geom.merge(snip_image_paths(), on="snip_id", how="left"),
        "area_um2",
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="mask_geometry — area_um2 value quartiles",
        output_path=output_dir / "area_um2_quartile_gallery.png",
    )
    return [grid, gallery]
