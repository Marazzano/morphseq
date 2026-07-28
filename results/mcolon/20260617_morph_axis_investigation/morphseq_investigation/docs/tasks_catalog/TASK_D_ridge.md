# TASK D — Ridgeline verb: port overlaid / stacked / mirror onto the grid IR

Starts once TASK_C's `DistributionGrid` IR is stable (at C's COMMIT 1/2). You
promote the existing bespoke ridge prototype into a proper Tier-2 verb that
consumes the SAME `DistributionGrid` as `plot_1d_density_grid`.

## Source of truth
`docs/VISUALIZATION_TAXONOMY.md` — the ridgeline is its OWN verb (offset baselines
within one coordinate system), NOT a `layout=` switch on the strip verb. The
EXISTING `v0/rich_distribution_plot.py` has the working prototype:
`render_ridgeline(..., variant="overlaid"|"stacked"|"mirror")` — three variants,
target-vs-reference styling, dashed unfilled reference. That logic is CORRECT and
already reviewed against real figures; you are RE-HOMING it, not redesigning it.

## Scope — `engine/plotting.py` (or `engine/ridge.py` if cleaner)
```python
plot_1d_ridgeline(
    grid: DistributionGrid, *, variant: str = "overlaid",   # "overlaid" | "stacked" | "mirror"
    role_styles=..., role_palettes=..., output_path=...,
) -> Figure
```
- Consumes the SAME `DistributionGrid` IR that TASK_C produces (so BOTH build
  paths feed it — within-population AND cross-population ridges come for free).
- Port the three variants' offset math from `v0/rich_distribution_plot.py`
  verbatim in spirit:
  - `overlaid` — shared baseline per bin (read coincidence).
  - `stacked`  — reference on baseline, target lifted just above (read shape).
  - `mirror`   — target up, reference mirrored down (read divergence).
- Reference role → dashed line + dashed unfilled outline (port the existing
  fill_between edgecolor/linestyle logic).
- Time bins stack as vertically-offset rows within a facet column; earliest at
  bottom.
- This is a genuine ridgeline (offset baselines), distinct from the
  `plot_1d_density_grid` panel — keep the taxonomy honest.

## Preserved invariants
- Reads densities off the `DistributionGrid`; NEVER fits KDEs (that's the
  numerical stage in TASK_C).
- No new rendering contract violations.

## Develop against
The `DistributionGrid` fixtures from TASK_C, plus (once E is close) the real b9d2
grid. You do NOT need the catalog directly — you need a populated
`DistributionGrid`.

## Tests — `tests/engine/test_ridge.py`
- Each variant renders a `DistributionGrid` fixture to a figure without error.
- Reference curves are dashed; offsets increase per bin; earliest at bottom.
- Same IR that `plot_1d_density_grid` consumes also drives `plot_1d_ridgeline`
  (shared-IR proof — build one grid, render both verbs).

## Commit checkpoints
- `⟢ COMMIT 1` — `plot_1d_ridgeline` with the `overlaid` variant on the grid IR +
  test green.
  `catalog(task-d): plot_1d_ridgeline (overlaid) on the shared DistributionGrid IR`
- `⟢ COMMIT 2` — `stacked` + `mirror` variants + tests green; old bespoke
  `v0/rich_distribution_plot.py` ridge code retired or pointed at the new verb.
  `catalog(task-d): stacked + mirror ridge variants; retire bespoke prototype`
- Open PR.

## Definition of done
`plot_1d_ridgeline(grid, variant=...)` renders all three variants from the shared
`DistributionGrid` IR (both build paths feed it); reference dashed/unfilled;
bespoke prototype retired; two checkpoint commits; PR open.
