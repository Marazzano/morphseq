# TASK E — b9d2 catalog example (ACCEPTANCE TARGET, SERIAL, RUNS LAST)

Wires the whole stack: catalog construction → detect_peaks + labels → both plot
paths. **The regression oracle is the EXISTING b9d2 figure** — the peak /
phenotype / genotype rows over time must reproduce, target-vs-reference styling
intact. If the story changes, something upstream regressed.

## Source of truth
`docs/DISTRIBUTION_CATALOG_API.md` end-to-end examples (PATH A and PATH B). The
existing `v0/b9d2_worked_example.py` + `v0/rich_distribution_plot_v2.py` are the
BEHAVIOR you are reproducing through the new API (and then superseding). The
existing output `v0/outputs/b9d2_grid_total_length_um.png` is the visual oracle.

## Scope — `v0/b9d2_catalog_example.py` (new; supersedes the two old v0 scripts)

### The lean pipeline (this is the acceptance shape)
```python
catalog = DistributionCatalog.from_dataframe(
    df, sample_id_column="embryo_id", feature_columns=FEATURE_NAMES,
    label_columns=("genotype", "phenotype_clean"),
    split_columns=("time_bin",),                      # PATH A default: one dist per time bin
)
catalog = catalog.detect_peaks(features=FEATURE_NAMES, output_label="resolved_peak")  # all bins, no lambda

# PATH A — the 3-row figure (peak / phenotype / genotype over time)
groups = (
    *catalog.label_groups("resolved_peak",  display_name="Resolved peaks"),
    *catalog.label_groups("phenotype_clean", display_name="Phenotype"),
    *catalog.label_groups("genotype",        display_name="Genotype"),
)
grid = build_1d_density_grid(
    groups, feature="total_length_um",
    facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("time_bin"),
)
plot_1d_density_grid(grid, output_path="outputs/b9d2_grid_total_length_um.png")
```
- Repeat per feature (`total_length_um`, `baseline_deviation_normalized`).
- Genotype/phenotype target-vs-reference comes from the label group's SampleSets
  (b9d2 vs WT within one time-distribution — Path A, within-population).

### Also exercise PATH B (the split example — proves compare works)
Build a SECOND catalog split on `("time_bin", "genotype")`, `detect_peaks`, then:
```python
comparisons = catalog2.compare(across="genotype", values=("wildtype", "b9d2"))
grid_b = build_1d_distribution_comparison(
    comparisons, feature="total_length_um", label_group="resolved_peak",
    facet_col=CoordinateFacet("time_bin"),
)
plot_1d_density_grid(grid_b, output_path="outputs/b9d2_compare_total_length_um.png")
```
This proves the cross-population path end to end (separate WT/b9d2 distributions,
matched by time, peaks overlaid per cell).

### Also render a ridge (TASK_D) from the SAME Path-A grid
```python
plot_1d_ridgeline(grid, variant="stacked", output_path="outputs/b9d2_ridge_stacked_total_length_um.png")
```

## The acceptance check (must hold)
- **Peak counts per bin match the pre-migration engine**: 14hpf → 1 target peak,
  later bins → 2 (the emergence signal). Print them; assert
  `counts[0] <= 1 and max(counts) >= 2`.
- The Path-A figure has 3 rows (Resolved peaks / Phenotype / Genotype) × time-bin
  columns, reference gray/dashed, target crimson. Eyeball against the OLD
  `b9d2_grid_total_length_um.png`.
- The Path-B figure overlays WT vs b9d2 peaks per time cell.
- All preserved invariants intact (no KDE on Distribution, grid_id discipline,
  frozen objects).

## Cleanup (decide + do)
The old `v0/b9d2_worked_example.py`, `v0/rich_distribution_plot_v2.py`,
`v0/rich_distribution_plot.py` are superseded. Either DELETE them or move to
`v0/_superseded/` with a one-line pointer to `b9d2_catalog_example.py`. Do NOT
leave two live pipelines that can drift.

## Commit checkpoints
- `⟢ COMMIT 1` — Path A: catalog → detect_peaks → label_groups → density grid
  reproduces the 3-row figure; peak-count assertion green.
  `catalog(task-e): b9d2 Path-A density grid reproduces the pre-migration figure`
- `⟢ COMMIT 2` — Path B compare figure + a ridge variant render.
  `catalog(task-e): b9d2 Path-B comparison + ridgeline through the catalog`
- `⟢ COMMIT 3` — old v0 scripts retired; README/tasks pointer updated.
  `catalog(task-e): retire superseded v0 scripts; catalog example is canonical`
- Open PR.

## Definition of done
`v0/b9d2_catalog_example.py` runs clean, emits the Path-A 3-row figure (matching
the old oracle), a Path-B comparison figure, and a ridge; peak counts match;
superseded scripts retired; three checkpoint commits; PR open. This closes the
migration.
