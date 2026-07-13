# Plotting Primitives

This package holds the reusable plotting layer for the modal-organization work.
Keep figure scripts thin and push shared frame logic here.

## Core primitives

- `DistributionVisualSpec`: one distribution, its sampled points, optional labels, and optional canonical density grid.
- `DistributionOverlay`: a target/reference pair evaluated on one shared plotting frame.
- `build_distribution_overlay(...)`: the shared-grid primitive. Use this when two distributions need to be compared visually on the same axes.
- `render_distribution_qc_grid(...)`: the analysis-free renderer; pass precomputed peak and metric summaries.
- `plot_v0_distribution_qc_grid(...)`: the backward-compatible V0 adapter. It computes legacy summaries and delegates to the renderer.

## Analysis/rendering boundary

`v0_analysis.py` owns the legacy V0 metric and peak-count calculations. The
renderer only draws supplied points, density grids, and summary records. New
callers should calculate analysis explicitly and call
`render_distribution_qc_grid(...)`; existing V0 scripts may continue using
`plot_v0_distribution_qc_grid(...)` without changing output or imports.

## Frozen shared evaluation-grid policy

Pure plotting consumes prepared density fields on their frozen evaluation grid.
It never derives an evaluation grid from points, evaluates a KDE, re-evaluates
a density, or interpolates one field onto another. Target/reference overlays
must therefore arrive with identical evaluation coordinates;
`build_distribution_overlay(...)` validates that identity and raises on a
mismatch.

The historical `derive_shared_grid(...)` helper and
`plot_v0_distribution_qc_grid(...)` entry point are compatibility composition
wrappers outside the pure-renderer boundary. New analysis code must prepare the
shared grid and density fields upstream, then call the pure renderer.

## Overlay hooks

Both grid entry points accept `row_overlays`, a mapping from row index to callback list.

Use this for optional annotations that should sit on top of the base renderer without changing the renderer itself.

Typical pattern:

1. Build a `DistributionVisualSpec` list.
2. Prepare every density on one frozen shared evaluation grid upstream.
3. Compute any V0 analysis with `v0_analysis.py`, then call `render_distribution_qc_grid(...)` (or use the explicitly legacy composition wrapper).
4. Add row overlays only for the rows that need extra annotations.

## Where to look

- `modal_distribution_plotting.py` contains the shared models and low-level density/point primitives. It lazily preserves the historical import surface.
- `v0_qc.py` contains the V0 grid layout, pure renderer, and legacy adapter.
- `v0_analysis.py` contains the legacy V0 numerical summaries.
- `resolved_peaks.py` contains resolved-peak labels, basin contours, and overlays.
- `valley_visualization.py` is an example consumer that now reuses the shared overlay primitive.
- `core/density_composition.py` defines `DensityGrid` and `CanonicalGrid`.

## Conventions

- Keep base plots annotation-light.
- Prefer shared density grids over ad hoc axis limits when comparing two distributions.
- Keep row-specific decorations outside the core renderer unless they are broadly reusable.
- Import new code from the responsibility-specific module; imports from
  `modal_distribution_plotting.py` remain supported for existing consumers.
