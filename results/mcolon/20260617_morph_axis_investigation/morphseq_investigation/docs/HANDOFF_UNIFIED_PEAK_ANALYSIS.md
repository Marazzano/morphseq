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
- **Shared plotting grid**: display-only common frame used to compare multiple
  already-calculated rasters.

The remembered `resize=True` behavior needs an explicit reviewed contract. A
safe interpretation is display-only resampling of retained raster values onto
one shared plotting grid. It must never silently refit a KDE or pretend the
resized raster is the generating density. Note that earlier ontology documents
explicitly prohibited interpolating densities and required reevaluating samples
on a shared grid, while the new plan prohibits plotting from calculating KDEs.
This tension must be resolved deliberately before implementing `resize=True`.

Recommended direction:

1. Introduce one plot-input raster type/protocol carrying ordered feature names,
   axis coordinates, density values, and source density identity.
2. Provide pure adapters from core and engine density representations.
3. Put shared plotting-grid derivation and any approved raster resampling in a
   non-analytical plotting-IR module.
4. Keep original density identity and original grid attached; mark resampled
   values as display derivatives.
5. Require exact shared-grid equality when scientific raster comparison, peak
   assignment, or basin geometry is being claimed.

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
