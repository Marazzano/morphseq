# Unified Peak Analysis Implementation Plan

Status: active implementation plan.

Source of truth:
`N_DIMENSIONAL_PEAK_ANALYSIS_DESIGN.md`.

The older plans in `docs/tasks_catalog/` describe the catalog implementation
that currently exists. They remain useful historical context, but they are not
the implementation contract for this refactor. In particular, this plan
supersedes their use of the parallel catalog peak detector,
`LabelGroupArtifacts`, and duplicated label representations.

## Goal

Produce one supported peak-analysis path:

```text
DensityEstimate
    -> robust peak-count voting and resolved peaks
    -> one adapter
    -> LabelGroup and derived SampleSets
```

Density calculation, density registration, peak resolution, comparison, and
plotting remain separate operations.

## Phase 1: Freeze contracts and invariants

Add or refine the target value objects:

- `DensityEstimate` with distribution identity, ordered feature names, and spec.
- `PeakVotingSpec` with draw count, sample fraction, and minimum valid draws.
- `PeakCountRobustnessPolicy` with minimum modal frequency.
- `PeakResolutionSummary` retaining the complete vote, both configurations,
  resolved modal count, and robustness decision.
- Unified `LabelGroup` with assignments, optional density, typed geometry, and
  optional peak-resolution summary.
- `Distribution.densities` and `shared_density_index`.

Enforce these invariants in constructors and tests:

- Density distribution identity and ordered features must match.
- A shared-density index must be in range.
- Every sample is assigned once or explicitly unassigned.
- Throwaway membership never materializes as a `SampleSet`.
- A resolved-peak group must retain its exact generating density.
- `peak_count == sample_set_count == len(peak_sample_sets)` for resolved peaks.
- Peak IDs are deterministic and local to one label group.

Exit gate: focused object tests pass without changing peak detection behavior.

## Phase 2: Separate density lifecycle operations

Implement explicit density operations:

```python
density = distribution.calc_density(spec=None)
distribution = distribution.with_density(density, select_as_shared=True)
```

Required behavior:

- `calc_density` computes only; it does not register, select, or label.
- `with_density` registers an already-calculated density and may select it.
- Multiple estimates may coexist for one distribution.
- Selecting another shared density does not alter existing resolved-peak groups.
- Provided labels with `density=None` use the shared-density fallback.
- Density-dependent plotting raises when no effective density exists.
- Plotting never calculates a KDE.

Exit gate: density calculation, registration, selection, and fallback tests pass.

## Phase 3: Make the robust resolver authoritative

Update the existing refined resolver to use the explicit voting contracts and
materialize the modal count even when it is non-robust.

Required behavior:

- Every supported resolution run performs peak-count voting.
- A non-robust modal result still produces assignments and peak `SampleSet`s.
- A tied mode selects the larger count, warns, and sets `is_robust=False`.
- Fewer than `min_valid_draws` retains the mode but sets `is_robust=False`.
- Zero valid draws raises because no modal count exists.
- A modal count of zero leaves all samples unassigned and creates no peak sets.

Do not add compatibility behavior for the parallel catalog detector.

Exit gate: resolver tests cover robust, non-robust, tied, insufficient-draw,
zero-draw, and zero-peak cases.

## Phase 4: Implement the single adapter

Add one conversion boundary:

```python
label_group_from_resolved_peaks(
    distribution,
    resolved_peak_distribution,
    *,
    name,
    density,
) -> LabelGroup
```

The adapter transfers, without recomputation:

- sample assignments and explicit unassigned membership;
- per-peak geometry;
- the complete vote;
- voting spec and robustness policy;
- resolved modal count and robustness decision;
- exact generating density;
- labeling provenance.

It must not fit density, detect candidates, count peaks, reconstruct basins, or
interpret detector internals independently.

Exit gate: adapter-fidelity tests prove that all authoritative resolver fields
survive conversion and all count/assignment invariants hold.

## Phase 5: Route public APIs through the resolver

Implement the public peak API with mutually exclusive density inputs:

```python
distribution.detect_peaks(
    *,
    output_label,
    density=None,
    density_spec=None,
    voting_spec,
    robustness_policy,
)
```

Rules:

- `density` uses an existing estimate.
- `density_spec` calculates an analysis-local estimate.
- Supplying both raises.
- Supplying neither uses the distribution's shared density and raises if absent.
- The exact density is stored on the resulting label group.
- Peak detection never mutates the distribution density registry.
- Label-group name collisions raise.

Apply the same behavior across `DistributionCatalog.detect_peaks()` without
introducing a second analytical implementation.

Exit gate: single-distribution and catalog API tests pass through the same
resolver and adapter.

## Phase 6: Migrate consumers

Migrate consumers in this order:

1. Peak-count and geometry extraction.
2. `Distribution.sample_sets()` and label-group accessors.
3. Catalog peak-count tables.
4. Density and ridge plotting.
5. Valley visualization.
6. Comparison and time-series metric plotting.

Consumers use typed fields and these convenience properties:

```python
label_group.sample_set_count
label_group.peak_count
label_group.is_robust
```

Robustness plotting keeps every modal result visible: robust values use solid
dots and non-robust values use hollow dots or dashed styling. Filtering is an
explicit caller choice.

Exit gate: repository search finds no consumer reading peak evidence from nested
provenance or recounting raw detector candidates.

## Phase 7: Delete the parallel path and duplicated storage

After consumers have migrated, remove:

- the catalog-side independent grid and KDE path;
- the 2-D grid bridge used only by that path;
- catalog-side HDR and basin reconstruction;
- raw `len(peaks)` peak-count decisions;
- `LabelGroupArtifacts` as durable peak storage;
- `LabelProvenance.geometry` as hidden geometry storage;
- `LabelColumn` and the separate richer `LabelGroup` duplication;
- retired helpers and tests whose only purpose is old-path compatibility.

Add a regression guard demonstrating that resolved-peak assignments can only be
produced through the authoritative resolver and adapter.

Exit gate: repository search finds one KDE implementation, one robust peak
resolver, and one adapter into the catalog ontology.

## Phase 8: Scientific acceptance and cleanup

Run scientifically meaningful regression fixtures, including the b9d2
peak-count trajectory. These validate the authoritative resolver; they are not
parity tests against the deleted catalog detector.

Verify:

- robust and non-robust results remain visible and internally consistent;
- purpose-specific result tables retain structured catalog coordinates;
- existing resolved groups remain stable when shared-density selection changes;
- analysis-local densities remain plottable without registry insertion;
- density-free plots work for provided labels without a KDE;
- density-dependent plots fail clearly when no effective density exists.

Exit gate: focused tests, broader investigation-package tests, and biological
acceptance fixtures pass.

## Phase 9: Implement descriptive comparison preparation

Implement `compare_distributions()` as the single raw-object descriptive
comparison path. Route `DistributionCatalog.compare()` through it after catalog
matching and attach catalog context only in the wrapper.

Required behavior:

- one explicit reference and one or more explicit targets;
- caller-supplied semantic grid, or explicit ordered features when a
  multi-feature distribution needs a derived grid;
- direct sole-feature inference for a single-feature distribution;
- pair-specific shared grids using robust pooled bounds and equal resolution
  per axis without implicit normalization or forced equal native-unit spans;
- direct evaluation on the shared grid, never raster interpolation;
- no density-spec equality requirement, with both specs retained as provenance;
- label groups provide overlays and never filter density-fit membership;
- missing label groups/members or incompatible features fail immediately;
- all-or-error results; no partial comparison collection;
- immutable, plot-ready descriptive results that do not mutate retained source
  densities; and
- explicit immutable `.test_nulls()` enrichment that distinguishes not tested,
  invalid, nonsignificant, and significant states.

Support and test 1-D and 2-D comparison density preparation only. Keep general
N-D comparison density support deferred until N-D bandwidth selection exists.

Exit gate: raw-distribution and catalog entry points produce equivalent
descriptive results, plotting consumes untested and tested results without
analysis, and structural guards prove the catalog does not duplicate shared-grid,
density, overlap, or null-test implementations.

## Deferred comparison task: peak matching

Peak matching is not required to consolidate the resolver. After consolidation,
implement it in the comparison layer using the design's `PeakMatchingPolicy`,
`PeakMatch`, and `PeakMatchingResult` stub.

The initial policy is one-to-one nearest-center matching with an explicit
normalized-distance rejection threshold. The precise radius-based normalization
must be reviewed before implementation.

## Definition of done

- One supported KDE implementation.
- One voting-based resolved-peak implementation.
- One adapter into the unified `LabelGroup` representation.
- No catalog-side alternative peak counter or basin reconstruction.
- Density calculation, registry mutation, peak assignment, and plotting have
  distinct APIs and effects.
- Modal results materialize regardless of robustness and remain visibly marked.
- Peak count, materialized sample sets, assignments, geometry, and evidence are
  mutually consistent.
