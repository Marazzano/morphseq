# TASK C — Plotting: typed FacetKey + both build paths + port plot_1d_density_grid

Parallel after TASK_0. You migrate the plotting layer onto the new objects:
Path A (within-population label groups) and Path B (cross-population comparisons),
sharing one IR and one renderer.

## Source of truth
`docs/DISTRIBUTION_CATALOG_API.md` — "Typed facet keys", "PATH A", "PATH B",
"CurveKey". `docs/VISUALIZATION_TAXONOMY.md` — Tier 1 primitives vs Tier 2 verbs.
The EXISTING `engine/plotting.py` already has: `strip_trace`, `hdr_band_trace`,
`overlay_strip_subplot` (Tier-1 primitives — KEEP), the current
`plot_1d_density_grid` + `materialize_distribution_marginals` +
`build_distribution_grid` (on the OLD `DistributionGrouping` — you RESHAPE these),
and `_resolve_colors` / role palettes (KEEP the styling logic).

## Scope — `engine/plotting.py`

### Typed FacetKey (from TASK_0 — import, don't redefine)
`CoordinateFacet(name)`, `LabelGroupFacet()`. Replace ALL `FacetCoordinate` enum
usage. `ComparisonMemberFacet` is DEFERRED — do not add it.

### Path A — within-population (RESHAPE the existing pipeline)
```python
build_1d_density_grid(
    groups: Sequence[DistributionLabelGroup], feature: str, *,
    facet_row: FacetKey = LabelGroupFacet(), facet_col: FacetKey = CoordinateFacet("time_bin"),
) -> DistributionGrid
plot_1d_density_grid(grid, *, role_styles=..., role_palettes=..., output_path=...) -> Figure
```
- Input is now `DistributionLabelGroup`s (NOT `DistributionGrouping`s). Each
  supplies its distribution + label group; curves in a cell = that label group's
  SampleSets.
- Keep the internal numerical stage (the ONLY KDE-fitting: shared grid per cell,
  per-SampleSet marginal) — reshape it to read SampleSets off a
  `DistributionLabelGroup` instead of the old grouping.
- **Cell grid = union bounds of the CURVES SELECTED for that cell** (NOT
  automatically all of the distribution's samples — if `unassigned` is dropped it
  must not stretch the grid).
- **One-label-group-per-cell invariant, structural**: raise
  `IncomparableDistributionsError` if a cell mixes `distribution_id`s (skip the
  check when LabelGroupFacet is an axis — then it's guaranteed). Port from the old
  same-`label_group_name` check.
- NO `group_by` / `overlay_across` params — the label group IS the grouping.
- Keep role→style (reference gray/dashed) and the crimson/gray auto-gradient from
  the existing `_resolve_colors`.

### Path B — cross-population comparison (NEW)
```python
build_1d_distribution_comparison(
    comparisons: DistributionComparisons, feature: str, *,
    label_group: str,                      # ONE shared label group on every member (like-with-like)
    facet_col: FacetKey = CoordinateFacet("time_bin"),
) -> DistributionGrid
```
- One cell per `DistributionComparison` (keyed by its `coordinates`).
- Curves = each member's SampleSets of `label_group`. **Shared grid = union bounds
  of EVERY selected curve across ALL members in the cell** (wildtype peaks + b9d2
  peaks).
- Curve identity is STRUCTURED, not concatenated strings:
  ```python
  @dataclass(frozen=True)
  class CurveKey:
      comparison_member: Hashable   # the across value, e.g. "wildtype"
      sample_set: str               # e.g. "peak_0"
  ```
  Display renders "wildtype · peak_0"; the model keeps the two axes separate.
- Per-member styling: members map to styles by their across value (e.g. a
  reference value → gray/dashed) — but the asymmetric per-member `label_groups=`
  mapping is DEFERRED; single `label_group=` only.
- Feeds the SAME `plot_1d_density_grid` renderer (same `DistributionGrid` IR).

### The shared IR (`DistributionGrid`) — keep, adjust curve payload
Both builders emit `DistributionGrid` (flat tuple of curves + facet keys). The
curve payload carries the density, the cell key, the sample_count, and — for
Path B — the `CurveKey`. Renderer reads it; never fits.

## Develop against fixtures while A/B land
You do NOT need the catalog or real peak detection. Build hand-made
`Distribution`s with a hand-attached label column (via TASK_0 `with_label`) and
`DistributionLabelGroup`s; for Path B, hand-build a `DistributionComparisons` with
two member distributions. Drop the fixtures at your integration checkpoint once A
(comparisons) and B (real `resolved_peak` columns) are green.

## Preserved invariants
- KDE fit ONLY in the numerical stage, per cell, on the shared grid. Renderer
  reads densities.
- No direct `ax.imshow/contour` etc. (Tier-1 primitives already honor the
  matplotlib contract).

## Tests — `tests/engine/test_plotting.py`
- `build_1d_density_grid` on fixtures: correct cells, curves = label group's
  SampleSets, cell grid from selected curves only (dropping a category shrinks
  bounds), one-label-group-per-cell invariant raises on a mixed cell.
- `build_1d_distribution_comparison` on a 2-member fixture: one cell per
  comparison, curves keyed by `CurveKey(member, sample_set)`, shared grid spans
  both members.
- `plot_1d_density_grid` renders both IRs to a figure without error; reference
  role → dashed/gray.
- FacetKey typed (a bad coordinate name is a construction error, not a late crash).

## Commit checkpoints
- `⟢ COMMIT 1` — FacetKey wired; `build_1d_density_grid` + `plot_1d_density_grid`
  reshaped onto `DistributionLabelGroup` + fixtures green.
  `catalog(task-c): Path A density grid on DistributionLabelGroup + typed FacetKey`
- `⟢ COMMIT 2` — `build_1d_distribution_comparison` + `CurveKey` + fixtures green.
  `catalog(task-c): Path B cross-population comparison grid + CurveKey`
- `⟢ COMMIT 3` — integration: swap fixtures for real catalog/detect_peaks outputs
  where available; invariant tests green.
  `catalog(task-c): integrate plotting with catalog + detect_peaks outputs`
- Open PR.

## Definition of done
Both build paths emit one `DistributionGrid`; `plot_1d_density_grid` renders it;
typed FacetKey replaces the enum; one-label-group-per-cell enforced;
`DistributionGrouping` no longer referenced in plotting; three checkpoint commits;
PR open. (Ridge verb is TASK_D.)
