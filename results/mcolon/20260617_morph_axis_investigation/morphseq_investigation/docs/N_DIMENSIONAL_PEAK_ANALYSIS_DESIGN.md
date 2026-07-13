# One Peak-Analysis Path: MVP Design

Status: minimal working design. The immediate goal is to consolidate the two
current peak-finding routes. Generalizing the detector to n dimensions comes
after that consolidation.

## Existing public ontology

Keep the established catalog ontology:

```text
DistributionCatalog
    contains
Distribution
    has one or more
LabelGroup
    materializes
SampleSet(s)
```

A resolved-peak analysis is one way to label a `Distribution`. It does not
introduce a special public peak container.

Provided labels and inferred labels produce the same `LabelGroup` shape:

```text
provided assignments  ─┐
                       ├─> LabelGroup
unsupervised labeling ─┘
```

## Distribution

A `Distribution` contains the original population:

```python
@dataclass(frozen=True)
class Distribution:
    distribution_id: str
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: np.ndarray
    shared_density: DensityEstimate | None = None
    label_groups: Mapping[str, LabelGroup] = field(default_factory=dict)
```

The feature matrix has shape `(n_samples, n_features)`. Sample IDs preserve the
join between label assignments and feature values. Ordered feature-name equality
is the compatibility check for comparing peak measurements across
distributions.

The `Distribution` may have one `shared_density`. It describes the whole
distribution independently of any labeling. It may be calculated eagerly when
the distribution is constructed or lazily when an analysis or visualization
first needs it.

## DensityEstimate

`DensityEstimate` is separate from labels and peak geometry. It retains the KDE
and the specification used to fit it:

```python
@dataclass(frozen=True)
class DensityEstimate:
    feature_names: tuple[str, ...]
    spec: DensityEstimateSpec
    # fitted density representation
```

The current validated default uses the longest-non-outlier-MST-edge bandwidth
rule. Exact density fields and grid fields can be refined when this design is
implemented.

## LabelGroup

One `LabelGroup` is one labeling run over one `Distribution`:

```python
@dataclass(frozen=True)
class LabelGroup:
    name: str
    distribution_id: str
    assignments: Mapping[str, Hashable]

    shared_density: DensityEstimate | None = None
    specified_density: DensityEstimate | None = None

    sample_set_geometries: Mapping[Hashable, SampleSetGeometry] = field(
        default_factory=dict
    )
    peak_resolution_summary: PeakResolutionSummary | None = None
    labeling_provenance: LabelingProvenance | None = None
```

For the MVP, labeling provenance supports only the two implemented origins:

```python
LabelingMethod = Literal["provided", "resolved_peaks"]
```

Labels loaded from a dataframe use `provided`. Labels created by the robust
peak-analysis route use `resolved_peaks`.

`shared_density` is the density used by the label group. Normally it is the same
object as `Distribution.shared_density`. A labeling analysis may provide a
`specified_density`; in that case the label group's `shared_density` is that
specified estimate.

```python
label_group.shared_density is (
    label_group.specified_density
    if label_group.specified_density is not None
    else distribution.shared_density
)
```

The density models the full distribution. Individual `SampleSet`s do not refit
their own KDEs. They share the label group's density and differ in membership
and, when measured, geometry.

Label-group names must be unique within a `Distribution`. Running peak finding
again with a different density specification requires a different label-group
name; existing groups are never silently replaced.

## SampleSet

A `SampleSet` is a derived category of a `LabelGroup`:

```python
@dataclass(frozen=True)
class SampleSet:
    sample_set_id: str
    sample_set_name: str
    distribution_id: str
    sample_ids: tuple[str, ...]
    geometry: SampleSetGeometry | None = None
```

`sample_set_geometries` maps label values such as `peak_0` to the intrinsic
geometry of the corresponding `SampleSet`. Provided label groups normally leave
this mapping empty. Peak finding populates it.

Per-peak properties belong to `SampleSetGeometry`. They are distinct from the
evidence describing how the number of peaks was resolved.

For the MVP, refine the current `SampleSetGeometry` to:

```python
@dataclass(frozen=True)
class SampleSetGeometry:
    feature_names: tuple[str, ...]
    center: np.ndarray
    radius: float
    support_fraction: float
    density_within_r80: float
    cv_radius_from_center: float
```

The changes from the current implementation are:

- Add `support_fraction`, because peak mass/support is a core per-peak property
  used in comparisons.
- Rename the current ambiguous `r80` field to `density_within_r80`; the value is
  a density/concentration measurement within the R80 region, not the R80 radius.
- Remove `grid_id`. Intrinsic geometry is expressed in the ordered named feature
  units. The density estimate and its specification belong to the parent label
  group's `shared_density`; raster identity is not needed for peak-level
  comparison.

## PeakResolutionSummary

Peak finding adds one optional, typed run-level summary to its `LabelGroup`:

```python
@dataclass(frozen=True)
class PeakResolutionSummary:
    peak_count_vote: PeakCountVote
    resolved_peak_count: int
    robustness_threshold: float
    is_robust: bool
```

Mean peak count, variance, modal frequency, and other summaries are derived from
the complete `PeakCountVote`; they do not need to be stored twice.

`is_robust` means only that the modal peak-count frequency passes the configured
threshold. The threshold is chosen by the user and retained with the result.

An analysis always retains its modal count and produces assignments for that
count, even when `is_robust` is false. Robustness qualifies the result; it does
not suppress the analysis artifact.

The central invariant is:

```python
peak_resolution_summary.resolved_peak_count == len(peak_sample_sets)
```

## Internal resolved-peak result

The existing refined analysis may continue to use
`ResolvedPeakDistribution` internally:

```text
Distribution + shared/specified density
    -> robust peak analysis
    -> ResolvedPeakDistribution          # internal computation result
    -> LabelGroup + derived SampleSets   # durable catalog form
```

The durable conversion retains:

- sample assignments;
- per-peak geometry;
- the complete peak-count vote;
- the resolved modal count;
- the robustness threshold and result;
- the density specification through `shared_density`;
- ordinary labeling provenance.

Temporary detector mechanics such as basin-label rasters do not need to be
retained for the MVP.

## Immediate consolidation target

There must be one robust peak-analysis implementation. The catalog route must
use the existing refined route that measures peak-count robustness; it must not
maintain a parallel peak counter or reinterpret a raw candidate count.

```text
one KDE implementation
    -> one robust resolved-peak implementation
    -> one conversion into LabelGroup
    -> catalog comparison through SampleSets
```

Visualization may use `Distribution.shared_density` or a label group's
`shared_density`. It does not rediscover peaks.

## Deferred work

The following are deliberately outside the immediate MVP:

- n-dimensional detector mechanics;
- valley and inter-peak relationship metrics;
- alternate estimator backends;
- basin-raster retention;
- per-SampleSet KDE fitting;
- generalized robustness for non-peak labelers.

These can be added after the two existing peak-finding routes have been reduced
to one implementation and the class structure above is stable.

## Catalog density workflow

Catalog construction does not calculate density unless requested:

```python
catalog = DistributionCatalog.from_dataframe(
    df,
    sample_id_column="embryo_id",
    feature_columns=FEATURE_NAMES,
    split_columns=("genotype", "time_bin"),
)
```

Calculate the shared density for every distribution with:

```python
catalog = catalog.calc_shared_density(spec=None, replace=False)
```

`calc_shared_density` always uses the complete ordered feature set already on
each `Distribution`; it does not accept another `features` argument. When
`spec=None`, it uses the validated default density specification.

Density replacement follows explicit lifecycle rules:

- If an existing shared density has the same specification, reuse it.
- If an existing shared density has a different specification, raise.
- Recalculate only when the caller passes `replace=True`.
- Existing label groups retain the density object used to create them; replacing
  the distribution's default must not silently leave an existing analysis in an
  inconsistent state.

More precisely:

- A provided-label group can follow a replaced distribution density because its
  assignments did not depend on KDE fitting.
- A label group with `specified_density` remains unchanged.
- A resolved-peak label group inheriting the distribution's shared density must
  be rerun when that density changes. Its assignments, geometry, peak-count vote,
  and resolution summary all depend on the old density.

Therefore replacement must raise when dependent resolved-peak groups exist
unless the caller explicitly requests that those analyses be rerun:

```python
catalog.calc_shared_density(
    spec=new_spec,
    replace=True,
    rerun_dependent_analyses=True,
)
```

## Default and custom peak-analysis passes

The default peak-analysis pass uses each distribution's shared density:

```python
catalog = catalog.detect_peaks(
    output_label="resolved_peaks_default",
)
```

If the shared density has not yet been calculated, this operation may calculate
the validated default lazily.

A custom density analysis supplies a density specification and must use another
label-group name:

```python
custom_spec = DensityEstimateSpec(
    method="custom_estimator",
    # estimator-specific parameters
)

catalog = catalog.detect_peaks(
    output_label="resolved_peaks_custom",
    density_spec=custom_spec,
)
```

This produces a `specified_density` for each new label group. It does not replace
the distribution's shared density or mutate the default peak-analysis group.

```text
Distribution
    shared_density = validated default

LabelGroup "resolved_peaks_default"
    shared_density = Distribution.shared_density
    specified_density = None

LabelGroup "resolved_peaks_custom"
    shared_density = custom DensityEstimate
    specified_density = that same custom DensityEstimate
```

One label group has exactly one shared density. Multiple density specifications
therefore produce multiple uniquely named label groups. Name collisions raise;
they never silently overwrite an existing analysis.

## Distribution-level metric extraction

Peak-count results should be extractable as one row per distribution:

```python
counts = catalog.peak_counts("resolved_peaks_default")
```

The extracted data retains catalog coordinates and derived resolution metrics:

```text
distribution_id
genotype
time_bin
resolved_peak_count
mean_peak_count
peak_count_variance
mode_frequency
is_robust
```

The complete `PeakCountVote` remains on the label group's
`PeakResolutionSummary`. Means, variances, and modal frequencies are derived
from it for tabular analysis and plotting.

## Distribution comparisons

Reference and target are relationships, not intrinsic distribution labels.
Use the existing comparison layer:

```python
comparisons = catalog.compare(
    across="genotype",
    values=("wildtype", "control", "mutant"),
    match_on=("time_bin",),
)
```

Each `DistributionComparison` contains the distributions matched at one time
bin. A comparison view may then identify one or more members as references and
targets:

```python
comparison_view = comparisons.with_roles(
    references=("wildtype", "control"),
    targets=("mutant",),
)
```

Comparison analysis—not plotting—calculates relative metrics such as:

- target-minus-reference peak count;
- relative spread;
- difference in mean distance or compactness;
- other future peak-organization contrasts.

Multiple references may remain separate or be summarized according to an
explicit comparison policy. They are never silently pooled.

## Result-table identity and coordinates

Do not force all analysis outputs into generic `source_id`, `metric_name`, and
`metric_value` columns. Each analysis returns a purpose-specific dataframe with
meaningful metric columns.

Use the identifier matching the output grain:

- Distribution-grain output uses `distribution_id`.
- Comparison-grain output uses `comparison_id`, plus explicit
  `ref_distribution_id` and `target_distribution_id` where applicable.

No generic `source_id` is required.

The catalog already retains structured coordinates directly on each
`Distribution`:

```python
distribution.coordinates = {
    "genotype": "b9d2",
    "time_bin": 30,
    "experiment_id": "20260304",
}
```

`distribution_id` is deterministically rendered from this coordinate mapping.
The ID is useful for joins, caching, and provenance, but the readable part of the
ID is never parsed back into metadata. Result helpers copy the structured
coordinates into output rows.

Likewise, a `DistributionComparison` retains its held-constant coordinates and
member distributions. Add a deterministic `comparison_id` helper based on the
held-constant coordinates and member distribution IDs. The structured fields
remain authoritative; `comparison_id` is never parsed.

Examples of purpose-specific output schemas follow.

Peak counts:

```text
distribution_id
genotype
time_bin
resolved_peak_count
mean_peak_count
peak_count_variance
mode_frequency
is_robust
```

Feature comparison:

```text
comparison_id
ref_distribution_id
target_distribution_id
reference
target
time_bin
feature_name
ref_mean
target_mean
mean_difference
effect_size
p_value
adjusted_p_value
is_robust
```

Relative peak comparison:

```text
comparison_id
ref_distribution_id
target_distribution_id
reference
target
time_bin
ref_mean_peak_count
target_mean_peak_count
mean_peak_count_difference
```

## Distribution metrics over time

Use the concise plotting name:

```python
plot_distr_metric_over_time(
    metric_df,
    value="mean_peak_count",
    time="time_bin",
    group_by="genotype",
    style="is_robust",
)
```

The plot accepts a purpose-specific dataframe and the name of the value column
to render. It can therefore display absolute distribution metrics or relative
comparison metrics without knowing how those values were calculated. Catalog
coordinates supply grouping and faceting columns directly.

`is_robust` is a visual annotation, not normally the y-axis metric. The plotting
layer may encode it through point fill, opacity, outline, or background shading.

Plotting remains separate from `DensityEstimate`. Density and peak objects expose
data; plotting functions arrange many distributions, label groups, time bins,
and comparison series into figures. Visualization never refits a KDE or
rediscovers peaks.

## Deferred companion output: feature differences from reference over time

In addition to multivariate peak analysis, the catalog should eventually support
a univariate reference-comparison analysis over every supplied feature.

The analysis proceeds as follows:

1. Split the catalog by genotype and time bin.
2. Match every target-genotype distribution to its reference distribution at
   the same time bin.
3. For every raw feature independently, compare the target marginal distribution
   with the reference marginal distribution.
4. Emit one result for each target, reference, time bin, and feature.

The durable output is a tidy dataframe:

```text
comparison_id
ref_distribution_id
target_distribution_id
reference
target
time_bin
feature_name
effect_size
p_value
adjusted_p_value
is_robust
```

The intended visualization is a heatmap:

```text
x-axis: time_bin
y-axis: feature_name
facet: target genotype
cell annotation: is_robust target-vs-reference difference
```

This answers:

> At which developmental times does each raw feature show a robust marginal
> difference from its reference distribution?

This output complements, but does not duplicate, a classifier-weight heatmap:

- Classifier weights show which features a multivariate classifier uses.
- The reference-comparison heatmap shows which features are individually
  different from reference.
- A feature may have both classifier importance and a marginal shift.
- A suppressor or correlated feature may have high classifier importance but no
  marginal difference of its own.

This heatmap is a separate analysis output from the ridge plots. The heatmap
summarizes feature-by-time reference-comparison results across the catalog. Ridge
plots remain independent diagnostic visualizations of selected one-dimensional
feature distributions; they do not discover or recount peaks and are not part of
the heatmap-generation path.

The definition of `is_robust` remains intentionally deferred. Candidate meanings
include multiple-testing-adjusted significance, recurrence under bootstrap
resampling, replication across experiments, or an explicit combination of these
criteria. These must remain separate evidence fields until a reviewed robustness
policy is selected.

## Behavior that must not break

The refactor must preserve these existing catalog guarantees:

- `Distribution` stores ordered feature names and an aligned
  `(n_samples, n_features)` matrix.
- Sample IDs remain the unique join key between features, label assignments, and
  derived `SampleSet`s.
- Distribution coordinates remain structured values used by selection,
  faceting, matching, and output tables.
- `distribution_id` remains derived deterministically from coordinates and is
  never parsed back.
- Coordinates and label groups retain different meanings: coordinates identify
  which distribution exists; label groups partition samples within it.
- Reference and target remain comparison relationships, never intrinsic
  distribution labels.
- Label-group names remain unique within a distribution; analyses never silently
  overwrite one another.
- Provided and resolved-peak labels produce the same public `LabelGroup` shape.
- A resolved peak remains an ordinary derived `SampleSet`, not a special public
  peak container.
- Peak assignments, per-peak geometry, resolved count, and robustness evidence
  remain mutually consistent.
- Plotting reads structured columns and objects; it never parses identifiers,
  refits KDEs, or rediscovers peaks.

## Migration and removal of the parallel peak path

The catalog's current peak-labeling implementation is a parallel analytical
path. It must be removed after the catalog has been routed through the refined,
robust resolved-peak implementation. Do not retain it as a long-lived deprecated
alternative.

### 1. Establish parity gates

Before deletion, add tests that run the same distributions through the reviewed
robust route and the new catalog adapter. Require equality or documented numeric
tolerance for:

- the complete peak-count vote;
- resolved modal count;
- robustness threshold and decision;
- sample-to-peak assignments;
- number of derived peak `SampleSet`s;
- per-peak geometry.

Retain the existing biological regression cases, including the b9d2 peak-count
trajectory, as acceptance fixtures.

### 2. Add the single adapter

Implement one conversion boundary:

```python
label_group_from_resolved_peaks(
    distribution,
    resolved_peak_distribution,
    *,
    name,
    shared_density,
) -> LabelGroup
```

The adapter transfers assignments, `SampleSetGeometry`,
`PeakResolutionSummary`, and labeling provenance. It does not fit density,
detect candidates, count peaks, or reconstruct basins.

`PeakResolutionSummary` is populated directly from the existing
`ResolvedPeakDistribution` resolution evidence. It is not independently
recomputed by the catalog layer.

### 3. Route every public caller through one resolver

`Distribution.detect_peaks()` and `DistributionCatalog.detect_peaks()` must call
the refined robust resolver and then the adapter above. Valley visualization and
other analysis callers must consume the same resolved result rather than
maintaining their own interpretation of peak count.

### 4. Delete the catalog-side analytical machinery

Remove from the current catalog labeler:

- independent peak-analysis grid construction;
- the 2D-only grid-to-canonical-grid bridge;
- independent density evaluation for label-group artifacts;
- catalog-side HDR and basin reconstruction;
- any use of raw `len(peaks)` as an alternative peak-count decision;
- robustness and artifact payloads hidden inside generic provenance mappings.

Temporary basin rasters and detector details may remain inside the refined
resolver while it is running, but they are not part of the durable MVP catalog
result.

### 5. Consolidate stored label representations

After the unified `LabelGroup` works end to end, remove the duplication among:

- `LabelColumn` as the stored assignment object;
- the separate richer `LabelGroup` run-result;
- `LabelProvenance.geometry` as a hidden geometry store;
- `LabelGroupArtifacts` as the catalog's retained peak-analysis payload.

The durable representation becomes the single `LabelGroup` described in this
document. `SampleSet`s remain derived views of its assignments and
`sample_set_geometries`.

### 6. Migrate consumers before deleting compatibility code

Update consumers in this order:

1. Peak-count and geometry tests.
2. `Distribution` and `DistributionCatalog` labeling methods.
3. Peak-count table extraction.
4. Existing density and ridge plotting.
5. Valley visualization.
6. Comparison and time-series metric plotting.

At each step, consumers should read typed fields rather than detector internals
or nested provenance dictionaries.

### 7. Delete rather than preserve two modes

Once parity and consumer migration pass:

- delete the retired helpers and tests that encode their behavior;
- search the repository for direct calls to those helpers;
- migrate or remove every remaining caller;
- add a regression guard proving that only the refined resolver produces
  resolved-peak assignments.

Completion means there is one KDE implementation, one robust peak-resolution
implementation, and one adapter into the catalog ontology.
