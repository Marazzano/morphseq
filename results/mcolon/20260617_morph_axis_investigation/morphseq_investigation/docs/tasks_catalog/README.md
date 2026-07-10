# Distribution Catalog — parallel implementation tasks

These briefs decompose `docs/DISTRIBUTION_CATALOG_API.md` (the LOCKED spec) into
implementable units. This is a **migration** of the existing `engine/`, not a
greenfield build: a working `plot_1d_density_grid` pipeline exists today on the
OLD `DistributionGrouping` shape, and the b9d2 figures are the regression oracle.

## Read before you touch anything

1. `docs/DISTRIBUTION_CATALOG_API.md` — THE locked spec. Everything defers to it.
2. `docs/VISUALIZATION_TAXONOMY.md` — the three visualization tiers (primitives /
   faceted verbs / custom figures); keeps valley_visualization out of the engine.
3. The existing `engine/` (objects.py, labelers.py, grid.py, plotting.py) and the
   two current callers (`v0/b9d2_worked_example.py`, `v0/rich_distribution_plot_v2.py`)
   — you are RESHAPING these, so read what's there.
4. Your own task file below.

## What changes vs. what's preserved (read this — it's a migration)

**Reshaped:**
- `Distribution` → typed-column table (samples + feature cols + label groups +
  coordinates); gains `from_dataframe` / `with_labels` / `discover_modes` /
  `sample_sets` / `label_group`.
- `SampleSet` → DERIVED view (from a label group), never durably stored glue.
- Labelers → COLUMN WRITERS (return a Distribution with a new label column +
  provenance/geometry), not free `(LabelGroup, [SampleSet])` tuples.
- `plot_1d_density_grid` → consumes `DistributionLabelGroup`s (typed FacetKey),
  no `group_by`.

**New:**
- `DistributionCatalog` (from_dataframe/split/**pool_by**/compare/id-helpers).
- `DistributionComparison(s)`, `build_1d_distribution_comparison`.
- Typed `FacetKey` (`CoordinateFacet` / `LabelGroupFacet`).

**Deleted from the public API:** `DistributionGrouping`,
`MaterializedDistributionGrouping`.

**Preserved invariants (NON-NEGOTIABLE — do not regress):**
- KDE is never fit on a `Distribution`; densities are materialized per facet cell
  downstream (shared-grid rule).
- grid_id raster-comparability (same grid_id ⟺ same evaluation coordinates).
- Everything frozen; every op returns a NEW object (no in-place mutation).
- `discover_modes` peak geometry is EAGER.

## Dependency graph (READ THIS — it dictates who can start when)

```
        TASK_0_foundation           ← BLOCKING. Serial. Committed + reviewed
        (typed-column Distribution,   BEFORE any other task starts. Freezes the
         coordinates+labels, SampleSet Distribution/LabelGroup/SampleSet contract,
         as derived view, LabelColumn,  the id scheme, and the coordinate-vs-label
         DistributionLabelGroup, ids)   split everyone keys off.
             │
     ┌───────┼──────────────┐          A / B / C run IN PARALLEL after TASK_0,
     ▼       ▼              ▼           each imports the frozen objects.
  TASK_A   TASK_B         TASK_C
  catalog  labelers+      plotting
  +compare discover_modes (paths A&B +
  +pool_by                 FacetKey; port
                           plot_1d_density_grid)
     └───────┴──────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
     TASK_D           (TASK_D can start once TASK_C's IR is stable)
     ridge verb
     (port overlaid/
      stacked/mirror
      onto the grid IR)
                 │
                 ▼
        TASK_E_b9d2_catalog   ← integration. Serial. Runs LAST. Reproduces the
        (the acceptance target)  b9d2 figure through the NEW catalog pipeline.
```

- **TASK_0 is non-negotiably first and serial.** A/B/C each invent a different
  object shape if they start early → merge hell.
- A/B/C are parallel after TASK_0. TASK_B is the biggest (re-expresses live peak
  code as a column writer). TASK_C can develop against hand-built Distribution +
  label-column fixtures while B lands; TASK_A can stub `discover_modes` similarly.
- TASK_D (ridge) needs TASK_C's `DistributionGrid` IR stable; start it at C's
  integration checkpoint.
- TASK_E is the acceptance target; it wires catalog → compare/label_group → plot
  and must reproduce the existing b9d2 grid (the regression oracle).

## Commit discipline (REQUIRED — applies to you, the agent)

Commit at the `⟢ COMMIT` checkpoints in your task file, not just at the end.

- Branch per task off latest `main` containing TASK_0:
  `git checkout -b catalog/task-<X>-<shortname>`
- **Use an ISOLATED WORKTREE per task branch.** The last parallel build had git
  races from multiple agents on one branch; the orchestrator should give each
  agent its own worktree (`isolation: "worktree"`).
- Commit message format: `catalog(task-<X>): <what landed at this checkpoint>`
- End every commit body with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Do NOT squash away checkpoint commits — they are the audit trail.
- Tests green BEFORE each `⟢ COMMIT`:
  `PYTHONPATH=.:src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output python -m pytest <your test file>`
  (run from the `20260617_morph_axis_investigation` dir). Red tests at a
  checkpoint = defect.
- Open a PR at your final checkpoint; do not merge into other tasks' branches.

## Where the code lands (confirm against the spec before TASK_0 commits)

Existing package `morphseq_investigation/engine/`:
```
engine/
  objects.py        # TASK_0: Distribution(table) + LabelColumn + SampleSet(derived)
                    #         + DistributionLabelGroup + coordinates
  identifiers.py    # TASK_0: make_distribution_id (from coordinates), make_sample_set_id
  invariants.py     # TASK_0: coverage/assignment guards (kept from old engine)
  catalog.py        # TASK_A: DistributionCatalog, DistributionComparison(s),
                    #         from_dataframe/split/pool_by/compare/find_ids/resolve_id/
                    #         to_index_dataframe
  labelers.py       # TASK_B: column-writer labelers + discover_modes (eager geometry)
  grid.py           # (exists) build_grid/evaluate_density — TASK_C reuses, no change expected
  plotting.py       # TASK_C: FacetKey, build_1d_density_grid, build_1d_distribution_comparison,
                    #         plot_1d_density_grid, CurveKey ; TASK_D: ridge verb
tests/engine/       # mirror the above, one test module per source module
v0/
  b9d2_catalog_example.py   # TASK_E: the new acceptance caller (replaces the two old v0 scripts)
```
The old `v0/b9d2_worked_example.py` + `v0/rich_distribution_plot_v2.py` + the
bespoke `v0/rich_distribution_plot.py` are superseded by TASK_E; TASK_E decides
whether to delete or leave them stale (see its brief).

## Definition of done (the whole set)
TASK_E reproduces the b9d2 rows (peak / phenotype / genotype over time) through
the catalog pipeline, target-vs-reference styling intact, via BOTH paths
exercised (label_group for within-population, compare for a split example). All
preserved invariants hold. Deferred items (comparison_id, match_peaks strategies,
ComparisonMemberFacet, mix_densities) remain TODO notes, NOT built.
