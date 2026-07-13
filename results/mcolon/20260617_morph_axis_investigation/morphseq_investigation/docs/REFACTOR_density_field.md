# Refactor: collapse `DensityGrid` into a mandatory grid + field primitive

**Status:** doing this now, as a standalone refactor (not a separate PR).
Sequencing note: run it as ONE coherent pass across all consumers + tests, and
land it *between* scientific changes (not interleaved with a bandwidth-rule or
metric change) so the test suite cleanly isolates "did the architecture move a
number." This file is the design record / target model for that pass.

## Why (the architectural crack)

The current core primitive is:

```python
@dataclass(frozen=True)
class DensityGrid:
    xx: np.ndarray
    yy: np.ndarray
    density: np.ndarray
    grid: CanonicalGrid | None = None   # <-- optional. this is the bug.
```

A density surface is meaningless without the grid that defines its coordinates
and cell area, but this type makes that association **optional, duplicated, and
incoherent**:

- **Optional** — `grid=None` opts out of the invariant. This is exactly what
  crashed `compare_bandwidth_rules_readout.py`: `build_distribution_overlay(...)`
  called without a `canonical_grid` returns `DensityGrid(..., grid=None)`; the
  figure only reads `.xx/.yy/.density` so it never noticed, but the resolved-peak
  engine reads `.grid.cell_area` and hit `None.xx` three call-layers deep.
- **Duplicated** — `xx`/`yy` are stored *and* recomputable from `grid`. Two
  sources of truth for one coordinate frame; they can silently disagree.
- **Incoherent** — nothing forbids `DensityGrid(xx=A, yy=B, grid=some_other_C)`.
  The object can assert two different coordinate frames at once.

The robust fix is not "detect `None` sooner." It is to make `None` (and the
duplication) **unrepresentable**.

## Target model

One frame; fields evaluated on it. The field cannot exist without its grid.

```python
@dataclass(frozen=True)
class CanonicalGrid:            # keep — already the coordinate-system spec
    x_min: float; x_max: float
    y_min: float; y_max: float
    grid_size: int
    # xx, yy, dx, dy, cell_area are derived properties (already implemented)

@dataclass(frozen=True)
class DensityField:             # replaces DensityGrid
    grid: CanonicalGrid         # MANDATORY
    values: np.ndarray
    def __post_init__(self):
        if self.values.shape != self.grid.xx.shape:
            raise ValueError("values shape does not match grid")
```

Consumers converge on one primitive, no translation/promotion/optional field:

- plotting: `field.grid.xx`, `field.grid.yy`, `field.values`
- peak engine: `field.grid.cell_area`, `field.values`
- integration: `np.sum(field.values) * field.grid.cell_area`

## The comparison should own the shared grid

The scientific operation is comparing two distributions, so the shared grid is
constitutive of the comparison, not incidental metadata:

```python
@dataclass(frozen=True)
class DistributionComparison:
    grid: CanonicalGrid
    target_density: np.ndarray      # or DensityField
    reference_density: np.ndarray
    def __post_init__(self):
        # both arrays must live on self.grid
        ...
```

Correct data flow (build the grid ONCE, up front):

```
target_points ─┐
               ├─► derive_shared_grid ─► CanonicalGrid ─┬─► evaluate target  ─► DensityField
reference_pts ─┘                                        └─► evaluate reference ─► DensityField
                                                        │
                                                        └─► DistributionComparison
```

```python
grid = derive_shared_grid(target_points, reference_points, grid_size=GRID_SIZE)
target   = evaluate_density(target_points, grid)
reference = evaluate_density(reference_points, grid)
comparison = DistributionComparison(grid, target.values, reference.values)

plot_distribution_overlay(comparison)              # both on comparison.grid
resolve_peaks(density=comparison.target_density, grid=comparison.grid)
measure_valleys(comparison.target_density, comparison.reference_density, comparison.grid)
```

`derive_shared_grid` is the box-derivation currently buried *inside*
`build_distribution_overlay` and then thrown away (the overlay keeps `xx/yy` but
drops the `CanonicalGrid`). Promote it to a named function so every consumer
builds the same grid the same way, and `None` never enters the system.

## Naming

`CanonicalGrid` + `DensityGrid` read as two competing grid objects. One is a
grid; the other is a sampled scalar field. Rename to make the distinction
obvious:

- `CanonicalGrid` (keep) — where measurements live
- `DensityGrid` → `DensityField` — the measurements on that grid

("DensityGrid" invites exactly the conflation that caused the crash.)

## Scope / blast radius (measured 2026-07-07)

`DensityGrid` is a load-bearing core type, not local to one figure:

- **9 files** reference `DensityGrid`, **61** total mentions, **32** direct
  `.grid` accesses.
- Heaviest: `plotting/modal_distribution_plotting.py` (15),
  `core/peak_counting.py` (12), `core/bandwidth_tuning.py` (9),
  `core/resolved_peak_metrics.py` (6), `core/density_composition.py` (6),
  plus `v0/plot_modal_v0_bandwidth_comparison.py`, `core/resolved_peak_analysis.py`,
  and `tests/test_resolved_peak_metrics.py` (4 each).

This is why it earns its own PR: the 17 tests in
`tests/test_resolved_peak_metrics.py` are the safety net for the rename, and the
change must land coherently across all 9 files in one pass.

## Interim state (already shipped, this is the stopgap)

`compare_bandwidth_rules_readout.py` currently promotes the overlay box to a
`CanonicalGrid` at the call site:

```python
box = build_distribution_overlay(grp, wt, grid=GRID, kde=None).box
canonical_grid = CanonicalGrid(x_min=box[0], x_max=box[1],
                               y_min=box[2], y_max=box[3], grid_size=GRID)
```

That is defensive patching, not the fix. When this PR lands, delete that
promotion and the optional-`.grid` path entirely; the shared grid comes from
`derive_shared_grid` / `DistributionComparison` instead.

## Definition of done

1. `DensityField{grid, values}` with mandatory grid + shape-check `__post_init__`.
2. `DistributionComparison{grid, target_density, reference_density}` with an
   invariant check that both arrays live on `grid`.
3. `derive_shared_grid(...)` extracted and public; `build_distribution_overlay`
   reworked to `evaluate_distribution_comparison(...)`.
4. Optional-`.grid` path deleted; no `grid=None` anywhere.
5. Duplicated `xx`/`yy` on the field removed (read through `grid`).
6. Call-site promotion in `compare_bandwidth_rules_readout.py` removed.
7. All 9 consumers + the 17 tests updated and green.
