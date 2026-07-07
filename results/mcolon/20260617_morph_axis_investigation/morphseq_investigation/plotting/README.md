# Plotting Primitives

This package holds the reusable plotting layer for the modal-organization work.
Keep figure scripts thin and push shared frame logic here.

## Core primitives

- `DistributionVisualSpec`: one distribution, its sampled points, optional labels, and optional canonical density grid.
- `DistributionOverlay`: a target/reference pair evaluated on one shared plotting frame.
- `build_distribution_overlay(...)`: the shared-grid primitive. Use this when two distributions need to be compared visually on the same axes.
- `plot_v0_distribution_qc_grid(...)`: the generic V0 figure renderer.

## Shared-grid rule

If a canonical density grid exists, reuse its coordinates. Otherwise derive one shared box from both point clouds, square it, and evaluate both densities on that frame.

Do not rebuild this logic inside figure scripts unless there is a strong reason. If a script needs target/reference comparison, call `build_distribution_overlay(...)`.

## Overlay hooks

`plot_v0_distribution_qc_grid(...)` accepts `row_overlays`, a mapping from row index to callback list.

Use this for optional annotations that should sit on top of the base renderer without changing the renderer itself.

Typical pattern:

1. Build a `DistributionVisualSpec` list.
2. Call `build_distribution_overlay(...)` or let `plot_v0_distribution_qc_grid(...)` build the shared KDE frame.
3. Add row overlays only for the rows that need extra annotations.

## Where to look

- `modal_distribution_plotting.py` contains the reusable plotting helpers.
- `valley_visualization.py` is an example consumer that now reuses the shared overlay primitive.
- `core/density_composition.py` defines `DensityGrid` and `CanonicalGrid`.

## Conventions

- Keep base plots annotation-light.
- Prefer shared density grids over ad hoc axis limits when comparing two distributions.
- Keep row-specific decorations outside the core renderer unless they are broadly reusable.
