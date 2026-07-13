# TASK D — Plotting via the faceting engine (do NOT fork it)

**Prereq: TASK_0 merged. Soft-dep on TASK_A** (you render Grids/DensityGrids — stub
against `Grid`/`DensityGrid` fixtures, integrate once A lands). **Independent of B
and C** — start as soon as TASK_0 is in.

## Source of truth
`docs/PRIMITIVE_ONTOLOGY.md` §1b "KDE strips fall out for free". Handoff step 3 /
plotting (lines 119–127). MEMORY.md "Matplotlib Visualization Contract" (obey it).

## The faceting IR you emit into (confirmed real, do not fork)
`src/analyze/viz/plotting/faceting_engine/ir.py`:
- `TraceData(x, y, style, label=..., render_as='line'|'band', band_lower/upper=...)`
- `SubplotData(traces=[...], heatmap=None, key=(row,col), title, x_label, y_label, xlim, ylim)`
  — **exactly one geometry per subplot: traces XOR heatmap** (`__post_init__`
  enforces it).
- `FacetSpec(wrap, sharex, sharey, row_order, col_order)`
- `FigureData(title, subplots=[...], row_labels, col_labels, heatmap_style, colorbar)`
Renderers: mpl + plotly via the engine's `render()`. **Do NOT reimplement rendering.**

## Scope
`engine/plotting.py` — pure IR emitters (they build `FigureData`, they don't draw):

### 1. KDE strip = 1-D Grid over one feature + DensityGrid → a `TraceData` curve
- `strip_trace(density_grid, style, label) -> TraceData` where
  `x = grid.axis_values[0]`, `y = density_grid.density`.
- **Overlay** = several SampleSets evaluated on the SAME 1-D `grid_id` → several
  `TraceData` in one `SubplotData` (directly comparable — same grid). Assert shared
  `grid_id` before overlaying; mismatched grid → raise (don't silently misalign).
- `unassigned` renders as a reserved visual category (`label="unassigned"`, a muted
  style) — never dropped, never counted as a mode.

### 2. N features × M comparisons → a `FigureData` grid
- N one-feature grids (rows) × M SampleSets/comparisons (traces or cols) → the
  faceted panel. **One row per `label_group` view.**
- **Strips get a wider aspect ratio** (set via the subplot/figure sizing the engine
  exposes; don't hand-draw axes).

### 3. Per-SampleSet HDR overlays (1-D)
- Shade the HDR mask region under/over the strip using `TraceData` band support
  (`render_as='band'`, `band_lower`/`band_upper`) or a filled trace — whatever the
  engine's contract supports. Honor the repo viz coordinate contract.

### Colors
Use `src/analyze/viz/styling/color_mapping_config.py` (`get_color_for_genotype`)
where genotype sets are plotted (MEMORY.md "Genotype Colors"). Don't hard-code.

## Explicit fast-follow — split OUT as TASK_D2 (deferred, do not build now)
The faceting IR has **no 2-D density-field panel**. The 2-D dual-distribution
overlay (two peak fields on one grid) needs a `DensityFieldData` IR extension in the
faceting engine itself. That is a separate task touching shared viz code — note it,
don't smuggle it in here. (Handoff lines 126–127.) 1-D strips are this task's job.

## Out of scope
Grid/density computation (A), labelers (B), comparison logic (C). You turn finished
objects into `FigureData`. No new rendering backend.

## Tests (green before each commit)
`tests/engine/test_plotting.py`:
- `strip_trace` maps `axis_values`/`density` onto `TraceData.x`/`.y`.
- Overlay on shared `grid_id` → one `SubplotData` with N traces; mismatched grid →
  raises.
- N×M → a `FigureData` whose subplot count == N×M (or wraps per `FacetSpec`).
- `unassigned` present as its own reserved trace with the muted style.
- Emit-only: assert `render()` accepts the `FigureData` without error (smoke), no
  golden-image assertions.

## Commit checkpoints
- `⟢ COMMIT 1` — `strip_trace` + shared-grid overlay + test green.
  `dist-engine(task-d): KDE strip + shared-grid overlay as faceting IR`
- `⟢ COMMIT 2` — N×M FigureData grid + HDR band overlay + test green.
  `dist-engine(task-d): N-feature × M-comparison faceted strips + HDR bands`
- Open PR. (Open a TASK_D2 stub file noting the 2-D DensityFieldData follow-up.)

## Definition of done
Strips/overlays/N×M grids emit valid faceting IR; shared-`grid_id` overlay enforced;
`unassigned` reserved; genotype colors from config; viz contract honored; 2-D field
deferred as TASK_D2; two checkpoint commits; PR open.
