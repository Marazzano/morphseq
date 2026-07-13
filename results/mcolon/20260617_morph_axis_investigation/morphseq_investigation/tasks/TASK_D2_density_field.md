# TASK D2 (deferred fast-follow) — 2-D density-field panel

**Status: NOT built. Stub only, noting the follow-up.**

## Why this is split out
TASK_D (`engine/plotting.py`) emits 1-D KDE-strip IR (`strip_trace`,
`overlay_strip_subplot`, `strip_grid_figure`, `hdr_band_trace`) entirely through
the EXISTING faceting-engine `TraceData`/`SubplotData` contract — no changes to
the faceting engine itself were needed for strips.

The 2-D dual-distribution overlay (two peak density fields, e.g. target vs
reference, plotted on one shared grid as a filled/contour field with basin
outlines) has **no home in the current faceting IR**. `SubplotData` enforces
exactly one geometry per subplot — `traces` XOR `heatmap` — and `HeatmapData`
(`src/analyze/viz/plotting/faceting_engine/ir.py`) is shaped for a categorical
row/col matrix (`values: (n_rows, n_cols)`, `row_labels`/`col_labels` as
strings), not a continuous 2-D density field with numeric axis coordinates,
overlaid contour/basin structure, or two co-registered fields on one panel.

## What TASK D2 needs to build
A `DensityFieldData` (or similarly named) IR extension **in the faceting engine
itself** (`src/analyze/viz/plotting/faceting_engine/ir.py`), analogous to
`HeatmapData` but for continuous field data:
- numeric `x_axis`/`y_axis` (feature-unit coordinates, from `Grid.axis_values`),
- one or more `field` arrays (density on that grid — a `DensityGrid.density`
  reshaped to `(len(axis_x), len(axis_y))`),
- optional overlay support for basin-label rasters / contour levels / HDR mask
  outlines,
- a render path in both `renderers/matplotlib.py` and `renderers/plotly.py`.

This is shared viz code (touches the faceting engine, not just
`morphseq_investigation/engine/plotting.py`) — a separate task, not smuggled
into TASK_D.

## Constraints for whoever picks this up
- Honor the repo viz coordinate contract (MEMORY.md "Matplotlib Visualization
  Contract" — no direct `ax.imshow`/`contour`/`contourf`; go through the
  contract wrappers where applicable, mind the mpl 3.10 `origin=` flip bug).
- Axes stay in FEATURE units (ontology §1b decision (b)) — a 2-D field IR must
  not introduce a whitened/canonical basis.
- Two fields sharing one `grid_id` (ontology §1b raster comparability) should
  render as directly overlaid/co-registered panels, not independently rescaled
  subplots.

## Source
`docs/PRIMITIVE_ONTOLOGY.md` §1b handoff lines ~126–127 ("Plotting — ... N-feature
× M grid via the faceting engine (Sub-spec D)" / 2-D field noted as the
remaining gap); `tasks/TASK_D_plotting.md` "Explicit fast-follow" section.
