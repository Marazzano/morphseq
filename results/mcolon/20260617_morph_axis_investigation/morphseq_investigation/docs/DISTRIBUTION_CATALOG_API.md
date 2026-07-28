# Distribution Catalog API

This document describes the supported catalog API after unified peak-analysis
consolidation. The design source of truth is
`N_DIMENSIONAL_PEAK_ANALYSIS_DESIGN.md`; the implementation sequence and exit
gates are in `UNIFIED_PEAK_ANALYSIS_IMPLEMENTATION_PLAN.md`.

Documents under `docs/tasks_catalog/` and `PRIMITIVE_ONTOLOGY.md` are historical
records of the retired catalog implementation, not compatibility contracts.

## Ontology

`Distribution` owns one population:

- `sample_ids`, ordered `feature_names`, and `feature_values`;
- structured distribution-level `coordinates`;
- `label_groups: Mapping[str, LabelGroup]`;
- zero or more retained `DensityEstimate`s and an optional shared selection.

`LabelGroup` is the only durable labeling representation. It owns complete
sample assignments, optional typed per-sample-set geometry, optional exact
generating density, optional `PeakResolutionSummary`, and typed labeling
provenance. Unassigned membership is explicit and never materializes as a
`SampleSet`.

`Distribution.sample_sets(label_name)` derives `SampleSet`s from a label group.
Consumers use typed fields and properties instead of recounting detector
candidates or reading nested provenance:

```python
group = distribution.get_label_group("resolved_peaks")
group.sample_set_count
group.peak_count
group.is_robust
```

## Construction and provided labels

```python
catalog = DistributionCatalog.from_dataframe(
    frame,
    sample_id_column="embryo_id",
    feature_columns=("PC1", "PC2"),
    label_columns=("genotype", "phenotype"),
    split_columns=("time_bin", "genotype"),
)
```

Construction creates distributions and provided-label groups but calculates no
density and detects no peaks. `catalog.with_labels(frame, names)` attaches more
provided labels by sample ID. Label-group name collisions raise.

`catalog.pool_by(coordinate)` concatenates sample rows across that coordinate,
preserves provided-label assignments sample by sample, and drops retained
density/peak products because they belong to the pre-pooling distributions.

## Density lifecycle

Calculation and registry selection are separate immutable operations:

```python
density = distribution.calc_density(spec=None)
distribution = distribution.with_density(density, select_as_shared=True)
```

`calc_density` only calculates. `with_density` registers an existing estimate.
Multiple estimates may coexist. Changing the shared selection does not change
the density retained by an already-resolved peak group.

Provided labels with `density=None` use the distribution's shared density for
density-dependent plotting. Density-free consumers do not require a density.
Plotting never calculates a KDE; an N-dimensional retained raster may be
analytically marginalized for a requested one-dimensional density view.

## Peak detection

The public distribution API uses scalar voting controls:

```python
resolved = distribution.detect_peaks(
    output_label="resolved_peaks",
    density=None,
    density_spec=None,
    n_draws=80,
    sample_fraction=0.80,
    min_valid_draws=None,
    min_mode_frequency=0.80,
)
```

`density` and `density_spec` are mutually exclusive. Supplying neither uses the
shared density and raises if none exists. `min_valid_draws=None` means
`ceil(0.80 * n_draws)`. Detection retains the exact generating density and does
not mutate the density registry.

`DistributionCatalog.detect_peaks` exposes the same scalar controls and maps the
distribution method without owning analytical behavior. A single explicit
`DensityEstimate` is accepted only for a one-distribution catalog. For multiple
distributions, pass a complete `distribution_id -> DensityEstimate` mapping, or
use `density_spec`/the per-distribution shared-density fallback.

All supported runs vote. Robust and non-robust modal counts both materialize as
assignments and derived peak `SampleSet`s. The robustness decision controls
presentation, not whether a modal result exists.

## Catalog access and result tables

```python
groups = catalog.label_groups("phenotype")
counts = catalog.peak_counts("resolved_peaks")
```

`label_groups` returns the unified `LabelGroup` objects present across catalog
members. Plot builders that need distribution context accept explicit
`(Distribution, LabelGroup)` pairs.

`peak_counts` returns one row per resolved distribution with `distribution_id`,
each catalog coordinate in its own structured column, and typed summary fields:

- `resolved_peak_count`;
- `mean_peak_count`;
- `peak_count_variance`;
- `mode_frequency`;
- `is_robust`.

## Comparisons and plotting

`catalog.compare(across, values=..., match_on=...)` matches distributions using
structured coordinates. Coordinates select populations; label groups partition
samples within a population. Neither distribution IDs nor category strings are
parsed to reconstruct coordinates.

```python
groups = tuple(
    (distribution, distribution.get_label_group("resolved_peaks"))
    for distribution in catalog.distributions
)
grid = build_1d_density_grid(groups, feature="PC1")
```

Density and ridge renderers consume prebuilt plotting data. Density-dependent
builders raise clearly when no effective density exists. Robust results use
solid styling and non-robust modal results remain visible with hollow or dashed
styling; filtering is an explicit caller choice.
