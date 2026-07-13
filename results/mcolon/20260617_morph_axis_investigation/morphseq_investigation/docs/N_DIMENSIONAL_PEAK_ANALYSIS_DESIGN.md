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
    densities: tuple[DensityEstimate, ...] = ()
    shared_density_index: int | None = None
    label_groups: Mapping[str, LabelGroup] = field(default_factory=dict)
```

The feature matrix has shape `(n_samples, n_features)`. Sample IDs preserve the
join between label assignments and feature values. Ordered feature-name equality
is the compatibility check for comparing peak measurements across
distributions.

The `Distribution` may retain multiple density estimates for the same population
and ordered feature set. `shared_density_index` selects the estimate used as the
distribution-level default. It is an index rather than a separately stored
density object, so the shared density cannot diverge from `densities`.

```python
distribution.shared_density == (
    distribution.densities[distribution.shared_density_index]
    if distribution.shared_density_index is not None
    else distribution.densities[0]
    if distribution.densities
    else None
)
```

If densities exist and no explicit shared index has been selected, the first
density is the intelligent default. An explicit shared-density selection
overrides that fallback. Density estimation remains separate from peak
resolution: creating or selecting a density does not create peak assignments.

Construction and selection raise `ValueError` when `shared_density_index` is
negative or out of range, or when a retained density has a different
`distribution_id` or different ordered feature names. A resolved-peak label
group's density must satisfy the same compatibility checks.

## DensityEstimate

`DensityEstimate` is separate from labels and peak geometry. It retains the KDE
and the specification used to fit it:

```python
@dataclass(frozen=True)
class DensityEstimate:
    distribution_id: str
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

    density: DensityEstimate | None = None

    sample_set_geometries: Mapping[Hashable, SampleSetGeometry] = field(
        default_factory=dict
    )
    peak_resolution_summary: PeakResolutionSummary | None = None
    labeling_provenance: LabelingProvenance | None = None

    @property
    def sample_set_count(self) -> int: ...

    @property
    def peak_count(self) -> int | None: ...

    @property
    def is_robust(self) -> bool | None: ...
```

For the MVP, labeling provenance supports only the two implemented origins:

```python
LabelingMethod = Literal["provided", "resolved_peaks"]
```

Labels loaded from a dataframe use `provided`. Labels created by the robust
peak-analysis route use `resolved_peaks`.

`density` is an optional label-group override. A resolved-peak label group stores
the exact density used to produce its assignments. A provided-label group
normally leaves `density=None`, because its assignments were not produced by a
KDE and it can safely use the distribution's shared density for visualization.

```python
effective_density = (
    label_group.density
    if label_group.density is not None
    else distribution.shared_density
)
```

The fallback is present for provided labels and other density-independent label
groups. A resolved-peak label group must never silently fall back: it retains the
exact density that generated its assignments even when that density is also the
distribution's shared density.

If a provided-label group has no density and its distribution has no shared
density, density-free plots still work. Density-dependent plots raise a clear
error; plotting never calculates a temporary KDE.

The density models the full distribution. Individual `SampleSet`s do not refit
their own KDEs. They use the label group's density and differ in membership
and, when measured, geometry.

Label-group names must be unique within a `Distribution`. Running peak finding
again with a different density specification requires a different label-group
name; existing groups are never silently replaced.

`sample_set_count` is the number of materialized, non-throwaway `SampleSet`s and
is defined for every label group. `peak_count` and `is_robust` are convenience
properties that delegate to `PeakResolutionSummary`; they return `None` for
provided labels rather than treating arbitrary categories as peaks. The values
are not stored twice.

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
    r80_radial_concentration: float
    cv_radius_from_center: float
```

The changes from the current implementation are:

- Add `support_fraction`, because peak mass/support is a core per-peak property
  used in comparisons.
- Rename the current ambiguous `r80` field to `r80_radial_concentration`; the
  value is the peak support inside the R80 region divided by the area of the R80
  disk. It is a radial concentration measurement, not the R80 radius or a KDE
  density value.
- Remove `grid_id`. Intrinsic geometry is expressed in the ordered named feature
  units. The density estimate and its specification belong to the parent label
  group's effective density; raster identity is not needed for peak-level
  comparison.

## PeakResolutionSummary

Peak finding adds one optional, typed run-level summary to its `LabelGroup`:

```python
@dataclass(frozen=True)
class PeakResolutionSummary:
    peak_count_vote: PeakCountVote
    resolved_peak_count: int
    voting_spec: PeakVotingSpec
    robustness_policy: PeakCountRobustnessPolicy
    is_robust: bool
```

Voting sufficiency and robustness interpretation are separate configurations:

```python
@dataclass(frozen=True)
class PeakVotingSpec:
    n_draws: int
    sample_fraction: float
    min_valid_draws: int

@dataclass(frozen=True)
class PeakCountRobustnessPolicy:
    min_mode_frequency: float
```

Mean peak count, variance, modal frequency, and other summaries are derived from
the complete `PeakCountVote`; they do not need to be stored twice.

`PeakVotingSpec` defines how the vote is collected and how many valid draws are
required for sufficient evidence. `PeakCountRobustnessPolicy` interprets a
sufficient vote. `is_robust` is true only when the vote has at least
`min_valid_draws`, has no tied mode, and its modal frequency passes
`min_mode_frequency`. Both configurations are retained with the result.

The supported resolved-peak path always performs peak-count voting. The modal
count is always retained and used to produce assignments and `SampleSet`s, even
when `is_robust` is false. Robustness qualifies the result; it does not suppress
the analysis artifact. This is important for diagnosing failure cases and for
plotting them differently rather than making them disappear.

The central invariant is:

```python
label_group.peak_count == label_group.sample_set_count
label_group.peak_count == len(peak_sample_sets)
```

Throwaway membership is represented as unassigned samples, never as a
`SampleSet`. Every distribution sample appears exactly once in either one
materialized sample set or the explicit unassigned collection. Unknown sample
IDs, duplicate membership, and assignments without matching geometry raise.

Voting edge cases have deterministic behavior:

- A tied modal count selects the larger count, emits a warning, and forces
  `is_robust=False`.
- Too few valid draws retain and materialize the modal count but force
  `is_robust=False`.
- Zero valid draws raise an analysis error because no modal count exists.
- A modal count of zero creates no peak sample sets and leaves all samples
  unassigned.

Peak IDs are local to one label group and deterministic within that result.
`peak_0` in different distributions or label groups does not imply biological
correspondence; cross-distribution peak matching is a separate analysis.

## Internal resolved-peak result

The existing refined analysis may continue to use
`ResolvedPeakDistribution` internally:

```text
Distribution + supplied or calculated density
    -> robust peak analysis
    -> ResolvedPeakDistribution          # internal computation result
    -> LabelGroup + derived SampleSets   # durable catalog form
```

The durable conversion retains:

- sample assignments;
- per-peak geometry;
- the complete peak-count vote;
- the resolved modal count;
- the voting specification, robustness policy, and result;
- the exact density estimate;
- ordinary labeling provenance.

Temporary detector mechanics such as basin-label rasters do not need to be
retained for the MVP.

## Immediate consolidation target

There must be one robust peak-analysis implementation. The catalog route must
use the existing refined route that measures peak-count robustness; it must not
maintain a parallel peak counter or reinterpret a raw candidate count.

```text
one supported KDE implementation
    -> one robust resolved-peak implementation
    -> one conversion into LabelGroup
    -> catalog comparison through SampleSets
```

“One supported KDE implementation” is literal for the distribution engine and
resolved-peak resolver: bandwidth rules may choose different scalar sigmas,
but all active paths evaluate them with
`evaluate_isotropic_gaussian_kde_from_dist2`. `core/support_geometry.py` keeps
historical SciPy and adaptive estimators only to reproduce legacy research
sensitivity diagnostics; those are not distribution-engine backends or
supported `ResolvedPeakAnalysisSpec` values. Promoting an alternate estimator
requires a future design and calibration change.

Visualization uses a label group's effective density. For provided labels this
normally falls back to `Distribution.shared_density`; for resolved peaks it is
the exact stored `LabelGroup.density`. Visualization does not rediscover peaks.

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

Density calculation and registry mutation are separate operations:

```python
density = distribution.calc_density(spec=None)
distribution = distribution.with_density(
    density,
    select_as_shared=True,
)
```

`calc_density` uses the complete ordered feature set already on the
`Distribution`; it does not accept another `features` argument. When `spec=None`,
it uses the validated default density specification. It only returns a density;
it does not register it, select it as shared, or create labels.

`with_density` explicitly retains the estimate in `Distribution.densities` and
may select it through `shared_density_index`. Additional retained estimates may
coexist, but there is only one selected shared density at a time. Catalog-level
helpers may map these two explicit operations across distributions, but density
calculation, registration, and peak assignment remain distinct effects.

Density replacement follows explicit lifecycle rules:

- Never silently replace a retained density estimate with a different one.
- Adding a density to the registry is an explicit `with_density` operation.
- Changing `shared_density_index` changes only the distribution fallback used by
  density-independent label groups and future default analyses.
- Existing resolved-peak groups retain the exact density object used to create
  them and are never changed by shared-density selection.

More precisely:

- A provided-label group can follow a replaced distribution density because its
  assignments did not depend on KDE fitting.
- A resolved-peak label group retains its exact stored `density` and remains
  unchanged.

## Default and custom peak-analysis passes

The default peak-analysis pass uses each distribution's shared density:

```python
catalog = catalog.detect_peaks(
    output_label="resolved_peaks_default",
    voting_spec=voting_spec,
    robustness_policy=robustness_policy,
)
```

If neither `density` nor `density_spec` is supplied, this operation uses each
distribution's shared density and raises clearly when none exists.

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
    voting_spec=voting_spec,
    robustness_policy=robustness_policy,
)
```

`detect_peaks` accepts either `density` or `density_spec`, never both. A supplied
`density` is used directly. A supplied `density_spec` calculates an
analysis-local density and stores that exact density on the resulting label
group. In every case `detect_peaks` creates peak assignments but never adds,
removes, or selects entries in `Distribution.densities`.

If an analysis-local density should also be discoverable from the distribution,
the caller explicitly adds it afterward with `with_density`. Large batch
analyses can omit that step without filling the registry with hundreds of
estimates. Their label groups remain fully plottable because they retain their
exact generating density; registry-based discovery may warn that such a density
is analysis-local.

```text
Distribution
    densities = (validated default,)
    shared_density_index = 0

LabelGroup "resolved_peaks_default"
    density = Distribution.shared_density

LabelGroup "resolved_peaks_custom"
    density = analysis-local custom estimate
```

One resolved-peak label group has exactly one density. Multiple density
specifications therefore produce multiple uniquely named label groups. Name
collisions raise; they never silently overwrite an existing analysis.

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
Use one role-aware catalog call:

```python
comparisons = catalog.compare(
    across="genotype",
    reference="wildtype",
    targets=("mutant_a", "mutant_b"),
    match_on=("time_bin",),
    label_group="resolved_peaks_default",
    features=("total_length_um", "baseline_deviation_normalized"),
    grid_size=40,
)
```

Each `DistributionComparison` contains the distributions matched at one time
bin. Biological comparisons normally declare roles explicitly and construct
directed contrasts as part of that same public call. Internally, member matching
and contrast expansion may remain separate operations, but callers should not
have to materialize an un-oriented comparison collection and then immediately
assign roles or call a public preparation/generation step.

The returned comparisons are already descriptive, aligned, and plot-ready. The
same call resolves the requested label group, validates ordered-feature
compatibility, derives each target/reference pair's shared grid, evaluates both
densities directly on it, and computes lightweight descriptive quantities such
as raster overlap. It does not run reference-null tests or silently rerun peak
voting. If the requested peak label group is absent, comparison raises rather
than hiding an expensive resolution run.

The authoritative descriptive implementation is also directly callable with
raw distribution objects:

```python
comparisons = compare_distributions(
    reference=wildtype_distribution,
    targets=(mutant_a_distribution, mutant_b_distribution),
    label_group="resolved_peaks_default",
    grid_size=40,
)
```

`compare_distributions()` performs the feature/label validation, pair-specific
shared-grid derivation, aligned density evaluation, and lightweight descriptive
comparison work. It does not perform catalog matching, infer biological roles,
or run null tests. The caller explicitly supplies the reference and targets.

Grid/feature inputs obey these MVP rules:

- A caller may supply an existing evaluation grid. At the semantic composition
  boundary, that grid's ordered features determine the comparison features;
  the low-level numeric grid implementation remains feature-blind.
- Without a supplied grid, a single-feature distribution can infer its only
  feature. If the distributions contain multiple features, the caller supplies
  the ordered `features` to compare. A two-dimensional comparison therefore
  names its two ordered features explicitly.
- A derived grid pools target/reference values independently for each requested
  feature, uses the Valley robust-bound/padding policy, and uses the same number
  of coordinates per axis. "Square" means equal raster resolution per axis,
  not forced equal numeric spans for native features with different units.
- Comparison does not normalize features implicitly; normalization is upstream.
- The supported comparison-density MVP is one or two dimensions. The grid
  abstraction may be N-dimensional, but the API must not advertise general N-D
  density comparison until N-D bandwidth support is implemented and tested.

Density specifications need not be identical. A provided label group may have
no retained density, and independently prepared distributions may legitimately
carry different density specifications. Comparison evaluates each member on
the shared coordinate grid using its applicable retained/default density
definition and preserves that provenance. Shared axes make raster operations
mechanically possible; they do not assert that estimator specifications are
identical. No `density_spec` equality guard is imposed.

The requested label group supplies assignments, geometry, counts, and overlays;
it does not filter samples used for the distribution density. Density uses the
complete distribution membership, including samples unassigned in that label
group. Comparing an individual sample set is a separate future operation.

`DistributionCatalog.compare()` is orchestration around that same function. It
matches members from catalog coordinates, resolves the reference and targets in
each matched group, routes those raw `Distribution` objects through
`compare_distributions()` exactly once, and attaches the final catalog context
(`across`, `match_on`, held-constant coordinates, and member values) to the
returned wrappers. It must not contain a second grid, density, overlap, or
descriptive-comparison implementation.

Contrast construction occurs only within each matched comparison group and
never pairs members from different time bins. Every contrast records its target
and reference explicitly; roles are not inferred from member names or order.
The MVP accepts one explicit `reference` per call. Comparing against a second
reference is a second call. This avoids silently expanding, pooling, or
summarizing multiple biological references before such a policy is designed.

An explicitly named `all_pairs()` utility may still produce semantically neutral
unordered pairs for exploratory symmetric analysis. It is not the default
biological-comparison workflow, because most scientific questions already know
which members are references and targets and do not require every quadratic
pair.

Known distribution IDs are resolved by the catalog and are never parsed. Once
resolved, they follow the same `compare_distributions()` path as coordinate-
matched members. A dedicated ID-only convenience is deferred unless a concrete
caller needs it.

Internal pair preparation:

1. requires identical ordered feature names;
2. derives one N-dimensional grid from both distributions' feature values;
3. evaluates both densities directly on that grid rather than interpolating a
   retained raster;
4. optionally retrieves the same named label-group view from both members; and
5. preserves the matched coordinates and member values for downstream use.

Low-level shared-grid derivation, rasterization, and overlap are numeric,
feature-blind utilities. The higher-level preparation boundary validates feature
identity before calling them. Plotting does not derive the grid or calculate a
KDE.

Prepared neutral pairs are reusable inputs to symmetric analysis and
basic/custom plotting. The one-call catalog comparison surface attaches the
biological roles used by comparison analysis and annotated plotting.

Expensive inference is a separate, explicit immutable enrichment:

```python
tested_comparisons = comparisons.test_nulls(n_draws=200, seed=42)
```

`test_nulls()` returns comparisons with the same aligned descriptive inputs plus
null distributions, valid-draw evidence, p-values, and test provenance. The
original descriptive comparisons remain unchanged. Plotting accepts either
form. It must distinguish `not tested` from `tested, not significant` and
`tested, significant`; absent inference is never interpreted as a nonsignificant
result.

Comparison analysis—not plotting—calculates relative metrics such as:

- target-minus-reference peak count;
- relative spread;
- difference in mean distance or compactness;
- other future peak-organization contrasts.

A future multiple-reference API may keep references separate or summarize them
according to an explicit policy. References are never silently pooled.

Symmetric metrics such as overlap need no role assignment. Directional metrics
assign reference/target roles in the comparison layer. Scientifically annotated
plots consume those directed comparison results rather than independently
inferring roles; neutral pair plots merely preserve deterministic member order.

Shared-grid density evaluations produced by `compare()` are owned by the
immutable comparison result and do not silently replace retained distribution
densities.

MVP failure behavior is all-or-error. A missing requested label group, missing
reference or target in any matched group, or incompatible requested feature
tuple raises immediately with distribution and matched-coordinate context.
Partial comparison collections are deferred. `test_nulls()` is immutable
enrichment, and untested comparisons explicitly remain untested.

### Deferred peak-matching stub

Peak matching belongs to the comparison layer. It consumes resolved peak
`SampleSetGeometry` from two label groups; it does not refit density, rerun peak
resolution, or change the local peak IDs.

```python
matching = comparison.match_peaks(
    label_group="resolved_peaks_default",
    reference="wildtype",
    target="mutant",
    policy=PeakMatchingPolicy(...),
)
```

```python
@dataclass(frozen=True)
class PeakMatchingPolicy:
    method: Literal["nearest_center"] = "nearest_center"
    max_normalized_distance: float = 1.0
    # Exact distance normalization remains to be reviewed.

@dataclass(frozen=True)
class PeakMatch:
    reference_peak_id: str
    target_peak_id: str
    center_distance: float
    normalized_distance: float

@dataclass(frozen=True)
class PeakMatchingResult:
    comparison_id: str
    reference_distribution_id: str
    target_distribution_id: str
    reference_label_group: str
    target_label_group: str
    matches: tuple[PeakMatch, ...]
    unmatched_reference_peak_ids: tuple[str, ...]
    unmatched_target_peak_ids: tuple[str, ...]
    policy: PeakMatchingPolicy
```

The MVP policy is one-to-one nearest-center matching with rejection beyond
`max_normalized_distance`. Matching requires identical ordered feature names and
compatible units. The distance must be normalized by a simple peak-scale
quantity so that the rejection threshold is interpretable across distributions;
a target-peak radius or a symmetric function of the reference and target radii
are candidate definitions. The exact normalization is deliberately left as a
reviewed policy choice rather than fixed in this consolidation design.

Unmatched peaks remain explicit. Non-robust label groups may still be matched,
but the result retains the robustness of both inputs for downstream filtering or
display. Cross-distribution peak identity exists only through
`PeakMatchingResult`; matching never makes local identifiers such as `peak_0`
globally meaningful.

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
layer renders robust results with solid styling and non-robust results with
non-solid styling so that failed robustness cases remain visible rather than
being filtered out.

Plotting remains separate from `DensityEstimate`. Density and peak objects expose
data; plotting functions arrange many distributions, label groups, time bins,
and comparison series into figures. Visualization never refits a KDE or
rediscovers peaks.

Alternative density specifications may be explored without mutating the durable
distribution or pretending that existing peaks came from another KDE:

```python
plot_distribution_density(
    distribution,
    density=alternative_density,
)
```

This is a visualization-only selection of an already calculated density.
Plotting never fits a KDE. When plotting a resolved-peak label group, plotting
uses that group's stored `density` and does not accept an override. If an
alternative density produces a peak result worth retaining, it must be run as a
separately named peak analysis.

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

### 1. Establish adapter-fidelity gates

Treat the reviewed robust voting resolver as authoritative. Do not preserve or
require parity with the catalog's parallel detector. Add tests proving that the
catalog adapter transfers the resolver's result without changing:

- the complete peak-count vote;
- resolved modal count;
- voting specification, robustness policy, and decision;
- sample-to-peak assignments;
- number of derived peak `SampleSet`s;
- per-peak geometry.

Retain scientifically important biological regression cases, including the b9d2
peak-count trajectory, as acceptance fixtures. These fixtures validate the
authoritative resolver; they are not compatibility tests for the retired path.

### 2. Add the single adapter

Implement one conversion boundary:

```python
label_group_from_resolved_peaks(
    distribution,
    resolved_peak_distribution,
    *,
    name,
    density,
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

Once adapter-fidelity tests and consumer migration pass:

- delete the retired helpers and tests that encode their behavior;
- search the repository for direct calls to those helpers;
- migrate or remove every remaining caller;
- add a regression guard proving that only the refined resolver produces
  resolved-peak assignments.

Completion means there is one supported engine/resolver KDE implementation,
one robust peak-resolution implementation, and one adapter into the catalog
ontology. Historical research-diagnostic estimators are outside that API.
