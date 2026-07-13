# Unified Peak Analysis Refactor Handoff

Status: paused after the first consolidation pass and adversarial completion
audit. The normal `Distribution`/`DistributionCatalog` path is unified and
tested, but the full implementation plan is **not complete**. Continue from the
remaining-work section rather than treating the green test suite as proof of
the complete design.

Source of truth:

- `N_DIMENSIONAL_PEAK_ANALYSIS_DESIGN.md`
- `UNIFIED_PEAK_ANALYSIS_IMPLEMENTATION_PLAN.md`

## Commits created

The work was intentionally split into intermediate commits:

1. `e647c803 refactor: establish unified peak analysis foundation`
2. `2b4ddc0f refactor: route catalog consumers through unified peaks`
3. `daf39c61 refactor: remove remaining parallel peak consumers`
4. `b1f7f3d3 feat: complete peak metric and valley visualization migration`

Unrelated data-pipeline changes were present before this work and were left
unstaged/unmodified by these commits.

## What is implemented

### Unified ontology and density lifecycle

- `Distribution.label_groups` is the sole durable label store.
- The retired `LabelColumn`, `LabelGroupArtifacts`, `LabelProvenance.geometry`,
  and `DistributionLabelGroup` surfaces are removed.
- Unified `LabelGroup` stores assignments, optional exact density, typed
  per-sample-set geometry, optional `PeakResolutionSummary`, and ordinary
  labeling provenance.
- `DensityEstimate`, `DensityEstimateSpec`, `Distribution.densities`,
  `shared_density_index`, `calc_density()`, and immutable `with_density()` are
  implemented.
- Constructors enforce assignment coverage, explicit unassigned membership,
  density identity/ordered-feature compatibility, resolved-density retention,
  geometry/count consistency, and deterministic group-local peak IDs.

### Authoritative voting and resolution path

- Canonical contracts exist in `core/peak_stability.py`:
  `PeakVotingSpec`, `PeakCountRobustnessPolicy`, and
  `PeakResolutionSummary`.
- Tied modes select the larger count, warn, and are non-robust.
- Insufficient valid draws retain the modal count and are non-robust.
- Zero valid draws raise.
- Modal zero materializes zero peaks and leaves every sample unassigned.
- Positive non-robust modal results materialize rather than disappearing.
- A precomputed density grid can enter the core resolver without refitting the
  full-data KDE; bootstrap draws still calculate draw-local KDEs.

### Single adapter and public APIs

- `engine/peak_adapter.py::label_group_from_resolved_peaks()` is the only
  catalog-ontology conversion boundary.
- It transfers assignments/unassigned samples, typed geometry, exact density,
  complete vote summary, and provenance without analytical recomputation.
- `Distribution.detect_peaks()` and `DistributionCatalog.detect_peaks()` route
  through the same labeler, resolver, and adapter.
- The public API is scalar-driven rather than requiring callers to instantiate
  configuration objects:

  ```python
  distribution.detect_peaks(
      output_label="resolved_peaks",
      density=None,
      density_spec=None,
      n_draws=80,
      sample_fraction=0.80,
      min_valid_draws=None,       # ceil(0.80 * n_draws)
      min_mode_frequency=0.80,
  )
  ```

- Detection does not mutate the density registry and retains the exact density
  used by the resulting label group.
- Catalog detection is delegation only; it contains no KDE, detector, basin,
  or HDR implementation.
- `DistributionCatalog.peak_counts()` emits structured coordinate columns plus
  resolved count, mean, variance, modal frequency, and robustness.

### Consumers and plotting

- Accessors, comparison summaries, catalog tables, ridge plots, density plots,
  and b9d2 acceptance paths use the unified label ontology.
- Engine plotting consumes an effective retained density and never fits a KDE.
- An existing N-D raster can be analytically marginalized to one feature for
  ridge rendering without refitting density.
- Provided labels have a density-free scatter path; density-dependent builders
  fail clearly when no effective density exists.
- Non-robust modal results remain visible with dotted/dashed/hollow styling.
- `plot_distr_metric_over_time()` consumes purpose-specific dataframes without
  filtering or analysis and preserves robust and non-robust rows.
- Modal/V0 renderers now consume precomputed density grids rather than fitting
  densities inside plotting.
- Structural regression guards live in
  `tests/engine/test_unified_peak_path_guards.py`.

## Verification completed

- Full investigation test suite after the final code changes:

  ```text
  181 passed in 35.44s
  ```

- Earlier intermediate full run before the time-series additions:

  ```text
  175 passed in 58.90s
  ```

- Real `valley_visualization.py` run completed successfully after migration.
  All votes had 80 valid draws with `min_valid_draws=64`.

- Regenerated primary artifact:

  ```text
  plots/b9d2_valley_MST_edge.png
  modified 2026-07-12 22:13:00 PDT
  size 446,834 bytes
  ```

- b9d2 MST-edge target trajectory at 14/18/24/30/48 hpf:

  ```text
  counts:      1, 1, 2, 1, 2
  robust:      T, T, F, F, T
  mode freq:  .96, .99, .69, .61, .80
  ```

- Other regenerated outputs:

  ```text
  plots/b9d2_valley_median_kNN.png
  plots/cep290_valley_MST_edge.png
  plots/cep290_valley_median_kNN.png
  ```

## Important remaining work

The adversarial audit found the following real gaps. Do not mark the plan
complete until these are resolved and re-audited.

### 1. Unify the scientific density grid and plotting grid model

#### Reviewed contract (2026-07-12)

The grid/`resize=True` policy is now decided:

- The semantic overlay boundary requires the same ordered feature set. A
  caller must not overlay distributions whose feature identities differ.
- A catalog-level convenience accepts a catalog plus distribution IDs (and,
  when relevant, one shared label-group name). The catalog already owns the
  sample IDs and feature values, so it can resolve the selected populations
  and derive one shared grid without callers supplying point arrays or bounds.
- The MVP preparation operation is pairwise but semantically neutral. A pair
  contains two distributions in deterministic input/comparison-value order; it
  does not yet call either member reference or target. Reference/target roles
  belong to a later directed comparison or presentation operation.
- The primary biological-comparison surface is directed: callers explicitly
  identify one reference value and one or more target values, and the
  comparison layer constructs the requested target/reference contrasts within
  each matched group. A second biological reference requires a second MVP call;
  references are never silently pooled. The API does not biologically infer
  roles from member order or names.
- This is one public catalog call, for example
  `catalog.compare(across=..., reference=..., targets=..., match_on=...,
  label_group=..., grid_size=...)`.
  Matching members and expanding directed contrasts may remain separate
  internal stages, but callers do not need a second `.contrasts(...)`,
  `.with_roles(...)`, `.prepare(...)`, or `.generate(...)` call immediately
  after `compare()`.
- `compare()` returns descriptive, aligned, plot-ready contrasts. It resolves
  the requested label group, derives pair-specific shared grids, evaluates the
  densities on those grids, and may compute lightweight descriptive overlap.
  It does not run reference-null tests or silently invoke peak-voting analysis.
- `compare_distributions(reference=..., targets=..., label_group=...,
  grid_size=...)` is the one authoritative descriptive implementation and is
  directly usable with raw `Distribution` objects. `DistributionCatalog.compare()`
  only matches catalog members, routes each matched group through that function,
  and attaches `across`/`match_on`/held-coordinate/member-value context to the
  final wrapper. The catalog must not duplicate grid, density, overlap, or
  descriptive-comparison logic.
- Expensive inference is the explicit next step:
  `tested = comparisons.test_nulls(n_draws=..., seed=...)`. This returns an
  immutable enriched result; it does not mutate the descriptive comparisons.
  Both forms are valid plotting inputs, and plotting distinguishes not tested,
  tested nonsignificant, and tested significant states.
- A neutral pair remains the internal reusable preparation primitive. Direct
  catalog-ID entry can resolve two distributions into that primitive, while a
  directed comparison wraps the same prepared data with explicit target and
  reference roles. `all_pairs()` may exist as an explicit exploratory utility,
  but it is not the default comparison workflow. Pairs never cross matched
  groups.
- Grid construction is one N-dimensional operation: pool the selected values
  and derive the same coordinate axis and resolution for every occurrence of
  each ordered feature. The 1-D and 2-D cases are not separate policies.
- If a semantic evaluation grid is supplied, its ordered features define the
  comparison axes. Without one, a sole feature may be inferred from a
  single-feature distribution; multi-feature distributions require explicit
  ordered `features` (including the two names for a 2-D comparison).
- Derived bounds port the Valley robust-bound/padding behavior per feature.
  Equal `grid_size` per axis makes a square raster; native axes with different
  units are not forced to have equal numeric spans, and comparison performs no
  implicit normalization.
- The low-level rasterization, grid, and overlap utilities are deliberately
  blind to feature names, distribution IDs, catalogs, and label groups. They
  operate only on numeric axes/arrays. Semantic compatibility is validated by
  the higher-level composition boundary before those reusable primitives are
  called.
- Matching label-group names identify the comparable population view across
  distribution IDs. Individual sample sets inside a label group do not get
  independently derived grids; all samples represented by the selected view,
  including unassigned samples, contribute to the common frame.
- Label groups provide assignments, geometry, counts, and overlays; they do not
  filter the complete distribution membership used for density evaluation.
- Density-spec equality is not required. Each distribution uses its applicable
  retained/default density definition on the shared coordinates, and the result
  preserves both specifications as provenance. Shared axes do not claim equal
  estimator configuration.
- `resize=True` means prepare the selected compatible distributions on one
  shared grid. Following `valley_visualization.py`, densities are evaluated
  directly on that grid from catalog-owned samples and the applicable density
  specification. It does **not** mean interpolate an already-calculated density
  raster.
- Existing retained densities are not mutated. Plotting is outside the catalog
  and outside the MVP except for basic rendering; it receives already-prepared
  aligned inputs and neither derives catalog membership nor fits a KDE.
- Raster overlap itself requires only numerically compatible axes and array
  shapes. It must not duplicate feature-name policy internally.
- Prepared pairs feed both downstream branches: symmetric metrics/basic plots
  may remain neutral, while biological comparison assigns reference/target
  roles at contrast construction. Annotated plotting consumes that directed
  result instead of independently inferring roles. The standard comparison
  workflow is role-aware even though its underlying grid preparation is not.

There are currently two density/grid representations and two higher-level
plotting stacks:

```text
core CanonicalGrid/DensityGrid/ResolvedPeakDistribution
    -> plotting/modal_distribution_plotting.py

engine Grid/DensityGrid/DensityEstimate/Distribution/LabelGroup
    -> engine/plotting.py
```

The intended consolidation should distinguish:

- **Scientific evaluation grid**: immutable coordinates on which a retained
  density was calculated; belongs to `DensityEstimate`/resolver evidence.
- **Shared evaluation grid**: one scientific grid derived before evaluation
  from the samples belonging to selected compatible distribution IDs. Each
  participating density is calculated directly on it, making the resulting
  rasters suitable for both comparison and plotting.

A plotting frame may reuse the shared evaluation grid, but it is not a second
density representation and plotting does not construct or resize density
rasters.

The remembered `resize=True` behavior is therefore a catalog/data-preparation
operation, not a plotting-raster interpolation operation. This preserves the
earlier ontology rule that densities on different grids are reconciled by
evaluating samples on a shared grid, while also preserving the newer rule that
plotting does not calculate KDEs.

Recommended direction:

1. Implement authoritative raw-object `compare_distributions()` and make the
   role-aware catalog comparison surface a thin matching/context wrapper around
   it. Keep an explicitly named exploratory all-pairs utility secondary.
2. Implement the underlying numeric shared-grid derivation as a reusable
   feature-blind primitive.
3. Add one preparation primitive that accepts two resolved `Distribution`
   objects, validates ordered-feature equality, and evaluates each density
   directly on their shared grid; do not interpolate retained rasters.
4. Preserve pair/group coordinates and member values so comparison and plotting
   never parse `distribution_id` values. Provide pure adapters from core and
   engine density representations without moving catalog or feature semantics
   into raster utilities.
5. Require exact shared-grid equality when scientific raster comparison, peak
   assignment, or basin geometry is being claimed.

Implementation requirements:

1. Shared-grid evaluations are owned by the immutable result returned from
   `compare()`; they do not replace or reselect source-distribution densities.
2. Missing requested label groups, references, targets, or compatible feature
   tuples fail immediately with distribution and matched-coordinate context.
3. Partial comparison collections are not MVP behavior: comparison is
   all-or-error.
4. `test_nulls()` returns an immutable enriched result and keeps not-tested,
   invalid, nonsignificant, and significant states distinct.
5. Implement and test 1-D and 2-D comparison preparation. Do not advertise
   general N-D density comparison until bandwidth support exists.

### 2. Split the plotting package and remove analysis from renderers

`plotting/modal_distribution_plotting.py` is still approximately 900 lines and
mixes models, frame logic, density/point primitives, resolved-peak overlays,
V0-specific layout, and V0 analysis summaries.

Suggested split:

```text
plotting/
    models.py
    density.py
    points.py
    resolved_peaks.py
    layouts/v0_qc.py
```

Move `compute_v0_metric_summary()`, `compute_v0_peak_count_summary()`, and raw
`peak_count_detail()` use out of plotting and into a V0 analysis/benchmark
module. Renderers must receive plot-ready summaries.

The current `plotting/README.md` is stale: it still says plotting may derive a
shared KDE frame/evaluate densities. Update it after the grid contract is
decided.

### 3. Retained-density reselection is incomplete

`with_density()` can append and select a newly supplied estimate, but there is
no API to reselect an estimate already retained in `Distribution.densities`.
Add an immutable selection operation (for example
`select_shared_density(index_or_identity)`) and tests proving A -> B -> A
selection leaves existing resolved label groups unchanged.

### 4. General density calculation is not truly N-dimensional

`engine/grid.evaluate_density()` is dimension-agnostic, but
`engine/density._bandwidth()` calls a bandwidth geometry helper restricted to
2-D points. Consequently `Distribution.calc_density()` fails for one feature
and for more than two features. Either implement N-D bandwidth selection for
the supported rule or explicitly narrow the authoritative contract; the design
currently points toward N-D density calculation even though the immediate peak
detector remains 2-D.

### 5. Public non-voting resolved paths remain

`resolve_points_with_analysis_spec()` and
`resolve_density_grid_with_analysis_spec()` can create a
`ResolvedPeakDistribution` without voting or the catalog adapter. Null and
detector-internal single-pass operations are legitimate, but these general
public names contradict “every supported resolution run votes.” Demote/rename
them as internal single-pass primitives, isolate null use, and migrate active
callers. Keep one public resolved-assignment entry point.

### 6. Basin-validation failure is inconsistent with the adapter invariant

For a positive modal count whose final basin validation fails,
`compute_resolved_peaks()` currently returns modal evidence for N peaks but an
empty foreground and `resolved_peak_count=None`. The adapter then cannot create
a `LabelGroup` satisfying `peak_count == sample_set_count == geometry_count`.
Choose and test an authoritative behavior. Likely options are:

- materialize the voted seeded basins and retain validation as separate
  evidence; or
- raise a clear analysis error before constructing a durable result.

Do not return a result that cannot cross the sole adapter.

### 7. Canonical evidence still has mutability and legacy aliases

- `PeakCountVote.peak_count_frequencies` is a mutable dictionary inside a
  frozen dataclass and lacks validation for negative counts/frequencies and
  invalid sample fractions.
- `PeakResolutionSummary` still exposes legacy alias properties such as
  `mode_peak_count`, `vote`, and `count_is_stable`.
- `PeakResolutionEvidence` redundantly stores target count/stability fields
  beside the canonical summary without equality guards.

Make vote evidence deeply immutable, validate it fully, remove aliases, and
make the canonical summary the only count/robustness authority.

### 8. Direct labeler configuration can conflict

The internal/publicly importable labeler accepts voting objects and an optional
`resolution_config`; when both are supplied, one set can be ignored. Narrow the
labeler boundary so it accepts one unambiguous configuration path. The scalar
composition belongs at the `Distribution` API boundary.

### 9. Supplied grid-coordinate fidelity

The bridge from engine density to the 2-D core resolver reconstructs a uniform,
square `CanonicalGrid` from endpoints and length. This can change nonuniform
axis coordinates and rejects nonsquare grids even though the engine `Grid`
allows them. Resolve this as part of the unified scientific-grid work.

### 10. Literal one-KDE definition needs a decision

Active code contains both the geometry-derived isotropic evaluator and the
SciPy `gaussian_kde` backend. Decide whether “one KDE implementation” means one
authoritative default path with explicitly supported alternate backend, or
literally one backend. Update the design/tests or remove the alternate path;
do not silently claim the literal definition is satisfied.

## Recommended next sequence

1. Review and freeze the scientific-grid/shared-plot-grid/`resize=True`
   contract.
2. Introduce the shared plot raster IR and split plotting without changing
   rendered output.
3. Move all V0 analysis/counting out of plotting.
4. Harden vote/summary evidence and resolve basin-failure semantics.
5. Demote public single-pass assignment APIs and narrow the labeler boundary.
6. Add density reselection and N-D bandwidth support.
7. Rerun the full 181-test suite, real b9d2 rendering, structural guards, and a
   requirement-by-requirement audit.

## Environment notes

- The requested `morphseq-env` Conda environment is not installed on this
  machine.
- Verification used the available
  `segmentation_grounded_sam` environment, typically through:

  ```bash
  /net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/conda run \
      -n segmentation_grounded_sam --no-capture-output python -m pytest ...
  ```

- The project-specific AGENTS rule requiring `morphseq-env` applies only under
  `dev/particle_prediction/`; none of this work touched that scoped directory.

## Current completion judgment

The primary catalog path is substantially consolidated and scientifically
exercised, but the full source-of-truth plan is not yet complete. Leave the
active goal open.
