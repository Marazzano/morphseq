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
  b9d2_catalog_example.py   # TASK_E: the CANONICAL acceptance caller (replaces the old v0 scripts)
  _superseded/              # TASK_E: the retired pre-migration scripts (do not run) + a pointer README
```
The old `v0/b9d2_worked_example.py` + `v0/rich_distribution_plot_v2.py` + the
bespoke `v0/rich_distribution_plot.py` are superseded by TASK_E and now live in
`v0/_superseded/` (they will not run on `main` — they import the removed
`label_genotype` / `DistributionGrouping` API). The single live pipeline is
`v0/b9d2_catalog_example.py`; see `v0/_superseded/README.md`.

## Definition of done (the whole set)
TASK_E reproduces the b9d2 rows (peak / phenotype / genotype over time) through
the catalog pipeline, target-vs-reference styling intact, via BOTH paths
exercised (label_group for within-population, compare for a split example). All
preserved invariants hold. Deferred items (comparison_id, match_peaks strategies,
ComparisonMemberFacet, mix_densities) remain TODO notes, NOT built.

---

# ⟢ ORCHESTRATION STATUS / HANDOFF (updated 2026-07-10)

**You are the orchestrator continuing this refactor.** Read this whole section
before doing anything. Do NOT re-run TASK_0/A/B — they are merged. Your job is to
finish TASK_C, then run D, then E, then merge the rest to `main`.

## What is DONE and MERGED to `main`
`main` tip after these merges is the branch containing:
- **TASK_0 foundation** (PR #12, merged). Frozen `engine/objects.py` (typed-column
  `Distribution`), `engine/facets.py`, `engine/identifiers.py`, `engine/invariants.py`.
  Merged foundation includes 4 post-brief reconciliations you must know about:
  1. **`detect_peaks` is the pinned name** (NOT `discover_modes`). Primitive
     `Distribution.detect_peaks(*, features, output_label="resolved_peak", spec=None)
     -> DistributionLabelGroup` (`.distribution` carries the new column); catalog
     convenience `catalog.detect_peaks(...)`. No `ModeDiscoveryResult`.
  2. **`Distribution.pooled_coordinates: tuple[str,...] = ()`** — `pool_by` writes the
     collapsed coordinate name(s) here; `compare()` refuses a pooled-away name as
     `across`/`match_on`. (`distribution_id` still derives from `coordinates` only.)
  3. **`CategoryShape(geometry, hdr, feature_profile)`** geometry read-back: 
     `Distribution.sample_sets()` UNPACKS `LabelProvenance.geometry[category]` into the
     SampleSet's typed slots — accepts a `CategoryShape`, a bare `SampleSetGeometry`,
     or `None`; anything else raises `TypeError`. **TASK_C reads `sample_sets()`
     directly and gets proper `SampleSetGeometry` on `.geometry` + `HDR` on `.hdr`
     — there is NO wrapper and NO `sample_sets_with_hdr` helper.**
- **TASK_A catalog** (PR #13, merged). `engine/catalog.py`: `DistributionCatalog`
  (`from_dataframe`/`to_index_dataframe`/`find_ids`/`resolve_id`/`pool_by`/`compare`/
  `with_labels`/`detect_peaks`/`map_distributions`/`label_groups`) +
  `DistributionComparison`/`DistributionComparisons`. **33 tests green.** THIS is the
  real `DistributionComparisons` shape TASK_C's Path B must consume — read
  `engine/catalog.py` for the exact fields (`.comparisons`, `.across`, `.values`,
  `.match_on`; each `DistributionComparison` has `.coordinates`, `.members: {across
  value -> Distribution}`). Reconcile TASK_C's hand-built fixture against it.
- **TASK_B labelers** (PR #14, merged). `engine/labelers.py`: `Distribution.detect_peaks`
  body (eager per-category geometry stored as `CategoryShape` in provenance; per-run
  robustness in `LabelProvenance.spec["artifacts"]`). Peak-count regression parity
  (14hpf→1, later→2) on a SYNTHETIC b9d2-like fixture. **111 engine tests green.**
  NOTE: the real b9d2 CSV (`reference_b9d2_clean.csv`) is gitignored/absent — TASK_E
  follow-up should wire it once available; the synthetic fixture stands for now.

Integration verified: `test_objects/identifiers/invariants/catalog` green on merged `main`.

## What is IN PROGRESS — TASK_C (RESUME THIS FIRST)
Branch `catalog/task-c-plotting` on origin, tip **`e6850d6f`** = **COMMIT 1 of 3 ONLY**.
Its worktree: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-a560382f7b5cbdd1f`.
The agent (id `a560382f7b5cbdd1f`) **FAILED on a session token limit** mid-COMMIT-2, not on a code problem.

**What C already built (COMMIT 1, in `engine/plotting.py`, 20 tests):**
- Typed `FacetKey` wired (replaced the old `FacetCoordinate` enum).
- **Path A** `build_1d_density_grid(groups, feature, *, facet_row, facet_col)`.
- `plot_1d_density_grid(grid, ...)` renderer.
- The shared IR — **`DistributionGrid`** (fields: `feature_name`, `row: FacetKey`,
  `col: FacetKey`, `curves: tuple[DistributionCurve,...]`) and **`DistributionCurve`**
  (fields: `cell: (row_value,col_value)`, `sample_set_name`, `style_group`,
  `grid: Grid`, `density: DensityGrid`, `sample_count`). **TASK_D consumes THIS IR.**
- `IncomparableDistributionsError` (one-label-group-per-cell guard).

**What C still OWES (its COMMIT 2 + 3 — you must finish these):**
- **COMMIT 2 — Path B:** `build_1d_distribution_comparison(comparisons, feature, *,
  label_group, facet_col)` + the structured **`CurveKey(comparison_member, sample_set)`**
  (never a concatenated string). Shared grid = union of every selected curve across ALL
  members in a cell. Consume TASK_A's REAL `DistributionComparisons` (see above), not a
  fixture. Feeds the same `plot_1d_density_grid` renderer.
- **COMMIT 3 — integration:** swap C's hand-built fixtures for real
  `catalog`/`detect_peaks` outputs (now available on `main`); invariant tests green.

**To resume C, do this:**
1. `cd` into C's worktree (path above). It is on `catalog/task-c-plotting` at `e6850d6f`,
   which sits on the OLD foundation `de3631d1` — **rebase it onto merged `main`**:
   `git fetch origin && git rebase origin/main` (resolve any `engine/` conflicts; take
   main's `objects.py`/`facets.py`/`catalog.py`/`labelers.py` — C only owns `plotting.py`
   + `tests/engine/test_plotting.py`).
2. Confirm C's Path A still green after rebase, then build COMMIT 2 (Path B) + COMMIT 3.
3. Commit messages per the TASK_C brief; body ends with the Co-Authored-By line.
4. Push, retarget/keep PR base = `main`, do NOT self-merge (hand merges to the human,
   who has been merging).

## What is NOT STARTED
- **TASK_D (ridge)** — `TASK_D_ridge.md`. Start once C's `DistributionGrid` IR is stable
  (i.e. after C's COMMIT 2). Consumes the `DistributionGrid`/`DistributionCurve` shape
  above. Branch off `main` (after C merges) or off C's branch if D must start before C
  merges — prefer waiting for C on `main`.
- **TASK_E (acceptance, SERIAL, LAST)** — `TASK_E_b9d2_catalog.md`. Runs on an
  integration state where A+B+C+D are all on `main`. Reproduces the b9d2 3-row figure;
  peak-count oracle 14hpf→1 / later→2; also exercises Path B compare + a ridge. Then the
  final cleanup of superseded `v0/` scripts. TASK_E follow-up: wire the real (gitignored)
  b9d2 CSV if it becomes available.

## Merge protocol (IMPORTANT)
- Everything branches off `main` now (TASK_0 is merged — no more "main+" base branches).
- PRs target `main`. If a PR was opened against `catalog/task-0-foundation`, retarget its
  base to `main` (`gh api -X PATCH repos/nlammers371/morphseq/pulls/<N> -f base=main`).
- The agent may not self-merge its own PRs (self-approval guard) — the HUMAN merges, or
  the orchestrator merges only with explicit human authorization. Verify `mergeable_state
  == clean` before asking for a merge.
- Run engine tests via:
  `cd .../20260617_morph_axis_investigation && PYTHONPATH=.:src:$PYTHONPATH conda run -n
  segmentation_grounded_sam --no-capture-output python -m pytest morphseq_investigation/tests/engine/<file> -q`
  (`test_labelers.py` peak regression is SLOW — run it alone with a long timeout.)

## Ordered next actions for the next orchestrator
1. Resume/finish **TASK_C** (rebase onto `main`, build Path B + integration). 
2. Launch **TASK_D** (ridge) on the stable IR.
3. Get C and D merged to `main` (human merges).
4. Run **TASK_E** last; reproduce the b9d2 figure + peak counts; retire superseded v0.
5. Final: confirm the whole set's Definition of done (above) holds on `main`.
