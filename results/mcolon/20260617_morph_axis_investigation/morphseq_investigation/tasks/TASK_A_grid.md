# TASK A — Grid construction + build_grid

**Prereq: TASK_0 merged.** You import `Grid`, `DensityGrid`, `make_grid_id` from
`engine/`. You do NOT re-derive the id hash — TASK_0 locked it; you feed it the
produced `axis_values`.

## Source of truth
`docs/PRIMITIVE_ONTOLOGY.md` §1b + Invariant #4/#9. Handoff step 2 (lines 110–117).

## Scope
`engine/grid.py`:
- `build_grid(feature_names, pooled_values, method, params) -> Grid`
  - `pooled_values`: the POOLED target+reference feature values (the shared
    comparison grid is built from the pool — §1b, construction walk Step 3).
  - Methods (all keep axes IN FEATURE UNITS — **decision (b)**, whitening chooses
    bounds/spacing only, never a basis change):
    - `pooled_min_max` — bounds from pooled min/max per feature.
    - `pooled_quantile` — bounds from pooled quantiles (params: `q_low`, `q_high`).
    - `pooled_mad_scaled` — MAD picks spacing/bounds; **stored axis_values stay
      feature-unit** (do NOT emit whitened coordinates).
    - `fixed_bounds` — explicit bounds from params.
  - `params` also carries `resolution` (cells per axis). Normalize params
    deterministically before handing to `make_grid_id` (sorted keys, rounded floats).
  - Compute `axis_values` (per-axis cell coordinates, feature units), then call
    `make_grid_id(...)` with those actual coordinates. Set `fit_sample_ids` from
    the pooled sample ids (order-independent per TASK_0).
- `evaluate_density(grid, sample_values, bandwidth_spec) -> DensityGrid`
  - KDE the given samples ON the grid's `axis_values` → `DensityGrid` tagged with
    the SAME `grid_id`. This is the primitive both peak runs and KDE strips use.
  - Reuse the existing KDE machinery if one already exists in `core/` (check
    `core/density_composition.py`, `core/bandwidth_tuning.py`) rather than adding a
    second KDE path — match the module's idiom.
  - **Never interpolate a density across grids** (§1b). To compare on a shared
    grid, re-`evaluate_density` the samples on that grid.

The 1-D case is not special-cased: `Grid(feature_names=("PC1",), axis_values=(<1d>,))`
+ `evaluate_density` IS a KDE strip. TASK_D consumes exactly this.

## Contracts you must honor
- `same grid_id ⟺ same evaluation coordinates` (TASK_0 property) — two calls with
  the same pooled values + method + params produce the same `grid_id`.
- `DensityGrid.density.shape == tuple(len(a) for a in grid.axis_values)`.
- No coordinate frame, no inverse, no "canonical" anywhere.

## Out of scope
Peak finding, comparison, plotting. You produce grids and densities only.

## Tests (green before each commit)
`tests/engine/test_grid.py`:
- Each method builds a Grid with feature-unit axes of the requested resolution.
- `pooled_mad_scaled` axis_values are in feature units (not whitened) — assert the
  range matches the pooled feature range, not a z-scored range.
- Same pooled values in shuffled row order → identical `grid_id`.
- 1-D grid + `evaluate_density` → a strip DensityGrid whose density integrates ~1.
- 2-D grid density shape matches axis lengths.

## Commit checkpoints
- `⟢ COMMIT 1` — `build_grid` with all four methods + test green.
  `dist-engine(task-a): build_grid with feature-unit axes (4 methods)`
- `⟢ COMMIT 2` — `evaluate_density` (grid-shared KDE, 1-D + 2-D) + test green.
  `dist-engine(task-a): grid-shared KDE evaluation incl. 1-D strips`
- Open PR.

## Definition of done
`build_grid` + `evaluate_density` land; feature-unit axes proven; `grid_id`
determinism/order-independence proven; two checkpoint commits; PR open.
