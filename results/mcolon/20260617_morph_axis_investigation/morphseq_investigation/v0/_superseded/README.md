# Superseded v0 scripts (retired — do NOT run)

These three scripts drove the **pre-migration** distribution engine
(`DistributionGrouping` / `FacetCoordinate` enum / the peer
`label_genotype` / `label_peak_finding` labelers) and are kept ONLY for
historical reference:

- `b9d2_worked_example.py` — the original TASK_E acceptance walk (carve →
  `label_genotype` → peak-on-pooled-grid → `compare_label_groups` →
  `strip_grid_figure`). Its `load_binned` upstream-binning logic was reused (in
  spirit) by the canonical example.
- `rich_distribution_plot.py` — the ridgeline prototype; its offset math +
  target/reference styling was re-homed onto the grid IR in
  `engine/ridge.py::plot_1d_ridgeline`.
- `rich_distribution_plot_v2.py` — the three-stage refactor onto the retired
  `materialize_distribution_marginals` / `build_distribution_grid` /
  `FacetCoordinate` API.

They will NOT import/run on `main`: `engine.labelers` no longer exposes
`label_genotype` (see `tests/engine/test_labelers.py`), and the
`DistributionGrouping` / `FacetCoordinate` public API was removed.

**Canonical replacement:** `../b9d2_catalog_example.py` — reproduces the whole
story (Path A 3-row density grid, Path B WT-vs-b9d2 comparison, stacked ridge,
1→2 peak-count emergence assertion) through the new `DistributionCatalog` stack.
