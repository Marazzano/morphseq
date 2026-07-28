# Compositional Distribution Analysis Model

## Core diagnosis

The repository already contains many useful typed scientific result objects:

```text
DensityField
PeakDetectionResult
PeakMembership
ResolvedPeakDistribution
ResolvedPeakDistributionSummary
RepeatedDrawTable
EmpiricalNullResult
```

The problem is not that these types are missing.

The problem is that they currently behave like disconnected products:

```text
points
→ function
→ result object

points
→ another function
→ another result object

result object
→ script-specific table
```

The empirical distribution is repeatedly reconstructed from arrays rather than represented as the persistent subject of analysis.

As a result:

- density results do not have a canonical owner
- detector outputs are not explicitly tied to the density that produced them
- resolved peak geometry is not naturally attached to the distribution it describes
- summaries and null results drift upward into analysis and visualization scripts
- scientific claims are assembled procedurally rather than read from composed analysis objects

The required correction is to introduce two persistent scientific records:

```text
DistributionRecord
DistributionComparison
```

Analysis functions should enrich those records with typed scientific products.

The existing result types remain important, but they become values stored on the distribution or comparison they describe.

---

# 1. Ontological order

The scientific construction proceeds in this order:

```text
canonical coordinates
→ empirical points
→ density
→ detector evidence
→ support membership
→ resolved peak geometry
→ resolved peak distribution
→ scalar distribution summary
→ distribution comparison
→ repeated draws
→ empirical inference
```

More explicitly:

```text
CanonicalGrid
    ↓

DistributionRecord
    owns empirical points
    ↓

DensityField
    density evaluated over the canonical grid
    ↓

PeakDetectionResult
    detector candidates and accepted/rejected decisions
    ↓

PeakMembership
    support assigned to detector candidates
    ↓

ResolvedPeak
    accepted candidate plus canonical support geometry
    ↓

ResolvedPeakDistribution
    full peak interpretation of one empirical distribution
    ↓

ResolvedPeakDistributionSummary
    compact scalar description of that interpretation
    ↓

DistributionComparison
    analytical relationship between multiple distributions
    ↓

RepeatedDrawTable
    raw stability or null values
    ↓

EmpiricalNullResult
    reduced statistical inference
```

The important dependency boundaries are:

```text
density is not peak evidence

peak evidence is not membership

membership is not resolved geometry

resolved geometry is not a scalar summary

a comparison is not a null test
```

Each layer adds one kind of scientific meaning.

---

# 2. Public interface philosophy

The ontology should be compositional, but the public interface should feel like normal scientific Python.

Users should access products through explicit typed dictionaries:

```python
distribution.densities["primary"]
distribution.peak_detections["primary"]
distribution.peak_memberships["primary"]
distribution.resolved_peak_distributions["primary"]
distribution.resolved_peak_summaries["primary"]
```

These typed slots cover the resolved-peak workflow because it is the first
vertical slice. They should not make `DistributionRecord` a peak-only class.

For V0, do not add a generic product namespace. Keep the public surface to the
typed dictionaries required by the observed resolved-peak path.

A later version may need an explicit extension namespace for products that are
not part of the resolved-peak spine, such as support geometry, density geometry,
component geometry, PCA projections, or future morphology descriptors:

```python
distribution.extension_products["support_geometry"]["primary"]
distribution.extension_products["density_geometry"]["stage1"]
```

The typed dictionaries are ergonomic convenience for common product families. If
the generic namespace is added later, call it `extension_products` rather than
`products` so it clearly reads as an escape hatch. Typed slots should remain the
preferred API for stable product families.

For comparisons:

```python
comparison.observed_metrics["primary"]
```

Future comparison products may look like:

```python
comparison.null_draws["pooled_label"]
comparison.stability_draws["downsample_80pct"]
comparison.null_tests["pooled_label"]
```

This is clearer than a universal artifact path such as:

```python
comparison.artifacts["draws/null/pooled_label"]
```

Hierarchical artifact paths are ontologically neat, but ergonomically taxonomic. They encode the product type, namespace, and product identity into one string, forcing users to decode all three mentally.

The public object should expose the scientific categories directly.

## V0 scope note

The sections below describe the broader intended architecture. For the first
implementation pass, keep V0 deliberately lean. The only required spine is:

```text
DistributionRecord
DistributionComparison
typed dictionaries for the observed resolved-peak path
pure add_* enrichment functions
```

Everything else in this document -- extension namespaces, rich product metadata,
canonical dependency keys, generic inspection helpers, truth compatibility, and
generic repeated-draw architecture -- is aspirational design guidance. Keep it in
the spec for future/documentation intent, but do not implement it until the lean
observed path proves the abstraction earns its keep.

---

# 3. Geometric substrate

## 3.1 CanonicalGrid

`CanonicalGrid` defines the shared coordinate support used for density evaluation, truth integration, and geometric comparison.

```python
@dataclass(frozen=True)
class CanonicalGrid:
    grid_id: str

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray

    axis_names: tuple[str, str]
    axis_units: tuple[str | None, str | None]

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @cached_property
    def xx(self) -> np.ndarray:
        ...

    @cached_property
    def yy(self) -> np.ndarray:
        ...

    @cached_property
    def cell_area(self) -> float:
        ...
```

It owns:

```text
canonical coordinates
derived mesh arrays
derived cell area
axis names
axis units
grid identity
```

It does not own:

```text
sample points
density values
peak detections
memberships
resolved geometry
```

Conceptually:

```text
CanonicalGrid = coordinate support
```

---

# 4. DistributionRecord as the persistent scientific subject

The empirical distribution should be a first-class object.

```python
@dataclass(frozen=True)
class DistributionRecord:
    distribution_id: str
    points: np.ndarray
    canonical_grid: CanonicalGrid

    densities: Mapping[str, DensityField] = field(default_factory=dict)

    peak_detections: Mapping[
        str,
        PeakDetectionResult,
    ] = field(default_factory=dict)

    peak_memberships: Mapping[
        str,
        EmpiricalPeakMembership,
    ] = field(default_factory=dict)

    resolved_peak_distributions: Mapping[
        str,
        ResolvedPeakDistribution,
    ] = field(default_factory=dict)

    resolved_peak_summaries: Mapping[
        str,
        ResolvedPeakDistributionSummary,
    ] = field(default_factory=dict)

    metadata: Mapping[str, Any] = field(default_factory=dict)
```

The record owns:

```text
distribution identity
authoritative empirical points
canonical coordinate context
typed derived products
basic metadata
```

The points are the authoritative empirical data.

Everything else is derived.

Retained arrays should be copied and marked read-only. This applies at minimum
to `DistributionRecord.points`, `DensityField.values`, and peak-membership
assignment arrays. Frozen dataclasses prevent field reassignment, but they do
not prevent in-place NumPy mutation.

Records may be partially populated. A `DistributionRecord` containing only
points, a canonical grid, and metadata is valid. Each enrichment helper validates
only the dependencies required for the product it adds.

The record should not become an active mega-object with methods for every possible analysis. It is the state-bearing scientific subject, while pure functions perform the transformations.

Product insertion should still be controlled. Do not mutate or directly replace
the product dictionaries at call sites. Enrichment helpers should be the only
public way to add products:

```python
distribution = add_density(distribution, ...)
```

Every `add_*` helper should reject duplicate product names by default:

```python
distribution = add_density(
    distribution,
    name="primary",
    spec=kde_spec,
    replace=False,
)
```

If `replace=False` and `"primary"` already exists in the relevant product family,
raise a clear error. Silent replacement of scientific products is not allowed.
Explicit `replace=True` is permitted for interactive iteration, but it must update
the product value and metadata together.

Prefer:

```python
distribution = add_density(distribution, ...)
distribution = add_peak_detection(distribution, ...)
```

rather than:

```python
distribution.estimate_density(...)
distribution.detect_peaks(...)
distribution.bootstrap(...)
distribution.plot(...)
```

Small inspection helpers are acceptable because they expose state rather than
performing analysis:

```python
distribution.list_products()
```

This should return a tidy table with product family, product name, value type,
dependencies, and basic provenance.

For V0, `list_products()` is optional. It is useful, but not required for the
first observed-path refactor.

---

# 5. Density layer

## 5.1 DensityEstimateSpec

The configuration used to estimate density should be separate from the realized field.

```python
@dataclass(frozen=True)
class DensityEstimateSpec:
    bandwidth_rule: str
    bandwidth_multiplier: float
    bandwidth_value: float
    normalization: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Do not call this `DensitySpec`: that name is already used by the synthetic
density-composition layer for component recipes. This object describes a KDE or
empirical density-estimation procedure, not a generative density recipe.

It answers:

```text
How should density be estimated?
```

## 5.2 DensityField

```python
@dataclass(frozen=True)
class DensityField:
    grid: CanonicalGrid
    values: np.ndarray
    density_spec: DensityEstimateSpec
    source_distribution_id: str
```

Invariant:

```text
values.shape == grid.xx.shape == grid.yy.shape
```

It answers:

```text
What density value is realized at each canonical coordinate?
```

It contains no peak semantics.

Composition:

```text
DistributionRecord.points
+
CanonicalGrid
+
DensityEstimateSpec
→ DensityField
```

Storage:

```python
distribution.densities["primary"]
```

Construction:

```python
distribution = add_density(
    distribution,
    name="primary",
    spec=kde_spec,
)
```

The function should:

1. read `distribution.points`
2. read `distribution.canonical_grid`
3. compute a `DensityField`
4. store it in `distribution.densities[name]`
5. record provenance in `distribution.product_metadata`

---

# 6. Detector layer

`PeakDetectionResult` remains detector evidence.

It answers:

```text
What candidate peaks were found?
Where are their centers?
Which candidates were accepted?
Which were rejected?
What detector evidence supported those decisions?
```

Conceptually:

```text
DensityField
→ PeakDetectionResult
```

Storage:

```python
distribution.peak_detections["primary"]
```

Construction:

```python
distribution = add_peak_detection(
    distribution,
    name="primary",
    density_name="primary",
    detector_spec=detector_spec,
)
```

The function should:

1. retrieve `distribution.densities[density_name]`
2. run the detector
3. return a typed `PeakDetectionResult`
4. store it in `distribution.peak_detections[name]`
5. record that it depends on the named density product

The detector result does not yet know:

```text
which samples belong to each candidate
how much support each accepted peak owns
peak radius
radial CV
unresolved support fraction
```

Those belong to later layers.

---

# 7. Membership layer

Membership is the hinge between detector evidence and resolved geometry.

## 7.1 EmpiricalPeakMembership

```python
@dataclass(frozen=True)
class EmpiricalPeakMembership:
    distribution_id: str
    sample_peak_ids: np.ndarray
    assignment_rule: str
    candidate_peak_ids: tuple[int, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
```

Convention:

```text
sample_peak_ids[i] >= 0
    sample i is assigned to that accepted detector candidate

sample_peak_ids[i] == -1
    sample i belongs to rejected or unresolved support
```

For V0:

```text
1. assign each sample to all detector candidates
2. preserve the candidate identity
3. replace assignments to rejected candidates with -1
```

Do not remove rejected candidates before assignment.

Otherwise all support is absorbed into accepted peaks and the unresolved fraction disappears.

Composition:

```text
DistributionRecord.points
+
PeakDetectionResult
+
assignment rule
→ EmpiricalPeakMembership
```

Storage:

```python
distribution.peak_memberships["primary"]
```

Construction:

```python
distribution = add_peak_membership(
    distribution,
    name="primary",
    detection_name="primary",
    assignment_spec=assignment_spec,
)
```

Later assignment rules may include:

```text
nearest candidate center
watershed
maximum assignment distance
density floor
assignment confidence
```

The V0 rule can remain nearest candidate center.

---

# 8. Resolved peak geometry

## 8.1 PeakGeometry

```python
@dataclass(frozen=True)
class PeakGeometry:
    peak_id: int
    center_coordinate: tuple[float, float]
    total_support_fraction: float
    radius: float
    cv_radius_from_center: float
```

It answers:

```text
Where is the peak?
How much total support belongs to it?
How broad is it?
How radially heterogeneous is it?
```

## 8.2 ResolvedPeak

```python
@dataclass(frozen=True)
class ResolvedPeak:
    geometry: PeakGeometry
    source_type: Literal["empirical", "truth"]
    detector_detail: PeakCandidateDetail | None
    provenance: Mapping[str, Any] = field(default_factory=dict)
```

Composition:

```text
accepted detector candidate
+
assigned support
+
canonical geometry
→ ResolvedPeak
```

A detector candidate is evidence.

A resolved peak is a candidate whose support and geometry have been interpreted relative to a distribution.

---

# 9. ResolvedPeakDistribution

```python
@dataclass(frozen=True)
class ResolvedPeakDistribution:
    distribution_id: str
    source_type: Literal["empirical", "truth"]

    density_field: DensityField
    detection_result: PeakDetectionResult
    membership: EmpiricalPeakMembership

    peaks: tuple[ResolvedPeak, ...]

    provenance: Mapping[str, Any] = field(default_factory=dict)
```

This is the complete semantic peak interpretation of one empirical distribution.

Composition:

```text
DensityField
+
PeakDetectionResult
+
EmpiricalPeakMembership
+
resolved accepted peak geometries
→ ResolvedPeakDistribution
```

It answers:

```text
What accepted peaks characterize this distribution?
Which samples belong to each peak?
How much support remains unresolved?
What density and detector context produced this interpretation?
```

Storage:

```python
distribution.resolved_peak_distributions["primary"]
```

Construction:

```python
distribution = add_resolved_peak_distribution(
    distribution,
    name="primary",
    density_name="primary",
    detection_name="primary",
    membership_name="primary",
)
```

The function should:

1. retrieve the named density
2. retrieve the named detector result
3. retrieve the named membership
4. compute `PeakGeometry` for each accepted candidate
5. construct `ResolvedPeak` objects
6. construct the `ResolvedPeakDistribution`
7. store it under the requested name

---

# 10. Scalar summary layer

## 10.1 ResolvedPeakDistributionSummary

```python
@dataclass(frozen=True)
class ResolvedPeakDistributionSummary:
    distribution_id: str
    source_type: Literal["empirical", "truth"]

    number_of_peaks: int

    assigned_support_fraction: float
    unassigned_support_fraction: float

    across_peak_total_support_fraction_mean: float
    across_peak_total_support_fraction_skew: float

    across_peak_radius_mean: float
    across_peak_radius_skew: float

    across_peak_cv_radius_from_center_mean: float
    across_peak_distance_mean: float

    provenance: Mapping[str, Any] = field(default_factory=dict)
```

Composition:

```text
ResolvedPeakDistribution
→ ResolvedPeakDistributionSummary
```

Storage:

```python
distribution.resolved_peak_summaries["primary"]
```

Construction:

```python
distribution = add_resolved_peak_summary(
    distribution,
    name="primary",
    resolved_peak_name="primary",
)
```

The summary is the primary unit for:

```text
table export
metric extraction
null testing
cross-distribution comparison
```

It intentionally discards:

```text
sample-level assignments
density arrays
detector details
full peak geometry context
```

Those remain available through the rich resolved object.

---

# 11. Metric definitions

Metrics should be defined over the summary object.

```python
@dataclass(frozen=True)
class ResolvedPeakMetricDefinition:
    name: str
    minimum_peak_count: int
    default_alternative: Literal[
        "two-sided",
        "greater",
        "less",
    ] = "two-sided"
```

Example registry:

```python
RESOLVED_PEAK_METRICS = {
    "number_of_peaks": ResolvedPeakMetricDefinition(
        name="number_of_peaks",
        minimum_peak_count=0,
    ),
    "assigned_support_fraction": ResolvedPeakMetricDefinition(
        name="assigned_support_fraction",
        minimum_peak_count=0,
    ),
    "across_peak_radius_mean": ResolvedPeakMetricDefinition(
        name="across_peak_radius_mean",
        minimum_peak_count=1,
    ),
    "across_peak_radius_skew": ResolvedPeakMetricDefinition(
        name="across_peak_radius_skew",
        minimum_peak_count=3,
    ),
    "across_peak_distance_mean": ResolvedPeakMetricDefinition(
        name="across_peak_distance_mean",
        minimum_peak_count=2,
    ),
}
```

Composition:

```text
ResolvedPeakDistributionSummary
+
ResolvedPeakMetricDefinition
→ scalar metric value or NaN
```

Structural validity belongs to the metric definition, not to figure code.

---

# 12. DistributionComparison

A comparison should be the persistent subject of cross-distribution analysis.

Lean V0 shape:

```python
@dataclass(frozen=True)
class DistributionComparison:
    comparison_id: str

    members: Mapping[str, DistributionRecord]

    observed_metrics: Mapping[
        str,
        pd.DataFrame,
    ] = field(default_factory=dict)

    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Future additions after the observed path works:

```python
null_draws: Mapping[str, RepeatedDrawTable]
stability_draws: Mapping[str, RepeatedDrawTable]
null_tests: Mapping[str, pd.DataFrame]
product_metadata: Mapping[str, ProductMetadata]
```

Typical members:

```python
comparison.members["reference"]
comparison.members["target"]
```

Functions that require target-versus-reference semantics should not rely on
implicit role names. They should either validate that `"reference"` and
`"target"` exist, or accept the role names explicitly. Prefer explicit role
arguments for reusable functions:

```python
reference_role = "reference"
target_role = "target"
```

In V0, the comparison owns:

```text
analytical roles
comparison identity
observed metric contrasts
basic metadata
```

After the observed path works, it may also own:

```text
null draws
stability draws
null-test summaries
```

It should not merge or rewrite the underlying distributions.

Geometry-aware comparison functions must validate grid compatibility. If a
function compares density fields, peak geometry, memberships, or resolved peak
summaries derived from geometry, it should confirm that the relevant member
products were built on compatible canonical grids. Prefer comparing `grid_id`
plus coordinate arrays, not object identity alone.

Small inspection helpers are acceptable later, but are not part of V0:

```python
comparison.list_products()
```

This should return a tidy table covering observed metrics and, if the future
products exist, null draws, stability draws, null tests, and comparison-level
product metadata.

Composition:

```text
reference DistributionRecord
+
target DistributionRecord
→ DistributionComparison
```

Construction:

```python
comparison = DistributionComparison(
    comparison_id="b9d2_vs_wt",
    members={
        "reference": wt,
        "target": b9d2,
    },
)
```

---

# 13. Observed comparison metrics

Observed contrasts are derived from named summary products on each member.

```python
comparison = add_observed_metrics(
    comparison,
    name="primary",
    summary_name="primary",
    reference_role="reference",
    target_role="target",
    metric_registry=RESOLVED_PEAK_METRICS,
)
```

The function should read:

```python
comparison.members[reference_role].resolved_peak_summaries["primary"]
comparison.members[target_role].resolved_peak_summaries["primary"]
```

and emit a tidy table containing:

```text
metric_name
observed_reference_value
observed_target_value
observed_difference
alternative
is_valid
invalid_reason
```

with:

```text
observed_difference =
    observed_target_value
    -
    observed_reference_value
```

Storage:

```python
comparison.observed_metrics["primary"]
```

---

# 14. Repeated draws

## 14.1 RepeatedDrawTable

```python
@dataclass(frozen=True)
class RepeatedDrawTable:
    member_values: pd.DataFrame
    contrast_values: pd.DataFrame
```

`member_values`:

```text
draw_id
member_id
member_role
metric_name
metric_value
is_valid
invalid_reason
```

`contrast_values`:

```text
draw_id
contrast_name
reference_member_id
target_member_id
metric_name
contrast_value
is_valid
invalid_reason
```

The same structure can support:

```text
null draws
downsampling stability
bootstrap uncertainty
truth calibration
```

The dictionary in which the table is stored conveys the analytical family.

For example:

```python
comparison.null_draws["pooled_label"]
comparison.stability_draws["downsample_80pct"]
```

No deeply namespaced string key is required.

---

# 15. Null workflow

Construction:

```python
comparison = add_null_draws(
    comparison,
    name="pooled_label",
    reference_role="reference",
    target_role="target",
    analysis_spec=analysis_spec,
    null_spec=null_spec,
)
```

For every null draw:

```text
resampled reference points
resampled target points
    ↓
temporary DistributionRecord pair
    ↓
density
    ↓
peak detection
    ↓
membership
    ↓
resolved peak distributions
    ↓
scalar summaries
    ↓
member metric values
    ↓
contrast metric values
```

Ordinary runs should retain only the compact scalar values.

Temporary density fields, assignments, and resolved objects can be discarded unless debugging is enabled.

Storage:

```python
comparison.null_draws["pooled_label"]
```

---

# 16. Stability workflow

Stability analysis asks a different question from null inference.

```text
null inference:
    Do target and reference differ under the null model?

stability analysis:
    Does the exact result persist under sampling variation?
```

Construction:

```python
comparison = add_stability_draws(
    comparison,
    name="downsample_80pct",
    analysis_spec=analysis_spec,
    stability_spec=stability_spec,
)
```

Storage:

```python
comparison.stability_draws["downsample_80pct"]
```

The stability table can support:

```text
modal peak count
modal frequency
peak-count variance
probability the count changes
mean target-reference difference
distribution of support-balance metrics
```

Null and stability draws should not be collapsed into one undifferentiated dictionary.

---

# 17. Empirical null reduction

```python
@dataclass(frozen=True)
class EmpiricalNullResult:
    observed_value: float

    null_mean: float
    null_std: float
    null_median: float
    null_q025: float
    null_q975: float

    empirical_p_value: float
    standardized_effect: float

    n_null: int
    n_valid_null: int
    valid_null_fraction: float

    test_is_valid: bool
```

Construction:

```python
comparison = add_null_tests(
    comparison,
    name="pooled_label",
    observed_metrics_name="primary",
    null_draws_name="pooled_label",
)
```

The function reads:

```python
comparison.observed_metrics["primary"]
comparison.null_draws["pooled_label"]
```

and stores:

```python
comparison.null_tests["pooled_label"]
```

The empirical-null primitive itself remains generic:

```python
run_empirical_null_test(
    observed_value=observed_difference,
    null_values=null_differences,
    alternative="two-sided",
)
```

It knows nothing about:

```text
KDEs
peak detectors
membership rules
resolved distributions
```

It only knows:

```text
one observed scalar
one vector of null scalars
one inferential direction
```

---

# 18. Future provenance model

The universal artifact wrapper solved a real problem:

```text
parameters
dependencies
provenance
```

But wrapping every scientific value in `AnalysisArtifact` makes access cumbersome.
This section is future/intent documentation, not part of the lean V0
implementation. V0 should keep basic `metadata` on records and defer
product-level provenance until the observed path is working.

Instead, use lightweight parallel metadata.

```python
@dataclass(frozen=True)
class ProductMetadata:
    product_family: str
    product_name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
```

Example:

```python
distribution.product_metadata[product_key("density", "primary")]
distribution.product_metadata[product_key("peak_detection", "primary")]
distribution.product_metadata[product_key("peak_membership", "primary")]
distribution.product_metadata[product_key("resolved_peak_distribution", "primary")]
distribution.product_metadata[product_key("resolved_peak_summary", "primary")]
```

Comparison metadata:

```python
comparison.product_metadata[product_key("observed_metrics", "primary")]
comparison.product_metadata[product_key("null_draws", "pooled_label")]
comparison.product_metadata[product_key("stability_draws", "downsample_80pct")]
comparison.product_metadata[product_key("null_tests", "pooled_label")]
```

The product values remain easy to access:

```python
distribution.densities["primary"]
```

while provenance remains available:

```python
distribution.product_metadata[product_key("density", "primary")]
```

This is slightly repetitive, but transparent.

It avoids bringing the universal wrapper back through a side door, but it creates
one risk: metadata can drift from values. Therefore product keys must be
canonical and insertion must be helper-controlled.

Required key convention:

```text
<product_family>:<product_name>
```

Keys should be created by a typed helper, not hand-built strings:

```python
def product_key(product_family: str, product_name: str) -> str:
    ...
```

The helper should validate known product-family spellings where possible. This
prevents tiny fractures such as `"peak_detection"` versus
`"peak_detections"`.

Examples:

```text
density:primary
peak_detection:primary
resolved_peak_summary:primary
null_draws:pooled_label
```

Required insertion rule:

```text
No product should be inserted without matching ProductMetadata.
No ProductMetadata should exist for a missing product.
```

This should be enforced in `add_*` helpers and in lightweight validation methods
on `DistributionRecord` and `DistributionComparison`.

Reusable computation specifications remain separate from product metadata.
Objects such as `DensityEstimateSpec`, `ResolvedPeakAnalysisSpec`, detector
specs, assignment specs, and null specs describe how computation should run.
`ProductMetadata` records the realized parameters, dependencies, provenance, and
execution context for one stored product.

---

# 19. Compositional transformation contract

Each transformation should follow the same general shape:

```text
record
+
named dependencies
+
computation specification
+
output name
→ enriched record of the same type
```

Distribution-level examples:

```python
distribution = add_density(
    distribution,
    name="primary",
    spec=kde_spec,
)
```

```python
distribution = add_peak_detection(
    distribution,
    name="primary",
    density_name="primary",
    detector_spec=detector_spec,
)
```

```python
distribution = add_peak_membership(
    distribution,
    name="primary",
    detection_name="primary",
    assignment_spec=assignment_spec,
)
```

```python
distribution = add_resolved_peak_distribution(
    distribution,
    name="primary",
    density_name="primary",
    detection_name="primary",
    membership_name="primary",
)
```

```python
distribution = add_resolved_peak_summary(
    distribution,
    name="primary",
    resolved_peak_name="primary",
)
```

```python
distribution = add_custom_product(
    distribution,
    family="support_geometry",
    name="primary",
    value=support_bundle,
    metadata=metadata,
)
```

Comparison-level examples:

```python
comparison = add_observed_metrics(
    comparison,
    name="primary",
    summary_name="primary",
    reference_role="reference",
    target_role="target",
)
```

```python
comparison = add_null_draws(
    comparison,
    name="pooled_label",
    reference_role="reference",
    target_role="target",
    analysis_spec=analysis_spec,
    null_spec=null_spec,
)
```

```python
comparison = add_stability_draws(
    comparison,
    name="downsample_80pct",
    analysis_spec=analysis_spec,
    stability_spec=stability_spec,
)
```

```python
comparison = add_null_tests(
    comparison,
    name="pooled_label",
    observed_metrics_name="primary",
    null_draws_name="pooled_label",
)
```

Every function should:

1. retrieve named typed dependencies
2. compute one typed scientific product
3. return a new record with that product inserted
4. record lightweight provenance and dependency metadata

---

# 20. Full empirical composition

Observed path:

```text
CanonicalGrid
    ↓

DistributionRecord(reference points)
DistributionRecord(target points)
    ↓

densities["primary"]
    ↓

peak_detections["primary"]
    ↓

peak_memberships["primary"]
    ↓

resolved_peak_distributions["primary"]
    ↓

resolved_peak_summaries["primary"]
    ↓

DistributionComparison
    ↓

observed_metrics["primary"]
```

Null path:

```text
comparison members
+
null-generation specification
+
fixed analysis configuration
    ↓

resampled point pairs
    ↓

temporary density/detection/membership/resolution pipeline
    ↓

compact member and contrast values
    ↓

comparison.null_draws["pooled_label"]
    ↓

comparison.null_tests["pooled_label"]
```

Stability path:

```text
comparison members
+
downsampling specification
+
fixed analysis configuration
    ↓

repeated point subsets
    ↓

temporary density/detection/membership/resolution pipeline
    ↓

compact member and contrast values
    ↓

comparison.stability_draws["downsample_80pct"]
```

---

# 21. How a scientific claim is represented

Consider the claim:

> The target has more mode structure than the reference on average, but the exact count is unstable.

The first half is supported by:

```python
comparison.observed_metrics["primary"]
comparison.null_draws["pooled_label"]
comparison.null_tests["pooled_label"]
```

These provide:

```text
observed target peak count
observed reference peak count
observed difference
null difference distribution
empirical p-value
standardized effect
valid-null accounting
```

The second half is supported by:

```python
comparison.stability_draws["downsample_80pct"]
```

This provides:

```text
distribution of target peak counts
modal target peak count
modal frequency
probability the count changes
count variance
distribution of target-reference differences
```

The figure reads those products from the comparison object.

It does not generate the claim itself.

---

# 22. Ownership model

## CanonicalGrid owns

```text
canonical coordinates
mesh arrays
cell area
axis metadata
```

## DistributionRecord owns

```text
empirical points
distribution identity
distribution-level derived products
distribution-level product metadata
```

## DensityField owns

```text
density values
density specification
canonical grid
source distribution identity
```

## PeakDetectionResult owns

```text
candidate identities
candidate centers
acceptance decisions
detector evidence
```

## EmpiricalPeakMembership owns

```text
sample-to-candidate assignments
unresolved assignment marker
assignment rule
```

## ResolvedPeak owns

```text
accepted peak geometry
support fraction
radius
radial CV
connection to detector evidence
```

## ResolvedPeakDistribution owns

```text
complete peak interpretation of one distribution
density context
detector result
membership
resolved accepted peaks
```

## ResolvedPeakDistributionSummary owns

```text
compact scalar description of one resolved distribution
```

## DistributionComparison owns

```text
analytical member roles
observed contrasts
null draws
stability draws
null-test results
comparison-level provenance
```

## RepeatedDrawTable owns

```text
raw repeated member values
raw repeated contrast values
draw-level validity
```

## EmpiricalNullResult owns

```text
reduced null statistics
p-value
standardized effect
valid-null diagnostics
```

---

# 23. File organization

## Lean V0 files to add now

Add only the files needed for the observed resolved-peak vertical slice:

```text
morphseq_investigation/
    core/
        distribution_record.py
        distribution_comparison.py
        peak_membership.py
```

Responsibilities:

```text
distribution_record.py
    DistributionRecord
    add_density(...)
    add_peak_detection(...)
    add_peak_membership(...)
    add_resolved_peak_distribution(...)
    add_resolved_peak_summary(...)
    readonly-array helper(s)
    replace=False duplicate-name checks

distribution_comparison.py
    DistributionComparison
    add_observed_metrics(...)
    explicit reference_role / target_role handling
    minimal shared-grid compatibility validation

peak_membership.py
    EmpiricalPeakMembership
    sample-to-candidate assignment helpers
```

## Existing files to keep as owners

Do not move existing primitives during V0 unless a small import adjustment is
needed. Keep current ownership:

```text
density_composition.py
    CanonicalGrid
    current DensityGrid / future DensityField work
    synthetic DensitySpec / DensityRealization

peak_counting.py
    PeakCandidateDetail
    PeakDetectionResult
    detect_peaks(...)

resolved_peak_analysis.py
    ResolvedPeakAnalysisSpec
    resolve_points_with_analysis_spec(...)
    run_resolved_peak_permutation_comparison(...)
    existing orchestration helpers

resolved_peak_metrics.py
    PeakGeometry
    ResolvedPeak
    ResolvedPeakDistribution
    ResolvedPeakDistributionSummary
    ResolvedPeakMetricDefinition
    RESOLVED_PEAK_METRICS
    summarize_resolved_peak_distribution(...)
    EmpiricalNullResult / run_empirical_null_test(...)
```

## Do not add yet

```text
product_metadata.py
repeated_draws.py
empirical_null.py
analysis/resolved_peak_bootstrap_comparison.py
analysis/resolved_peak_downsampling_stability.py
analysis/truth_peak_calibration.py
```

Those modules belong to the future architecture once the observed-path record
model proves useful.

<!-- Future layout sketch retained for documentation intent:
```text
core/
    canonical_grid.py
    density_composition.py
    distribution_record.py
    distribution_comparison.py
    product_metadata.py
    peak_counting.py
    peak_membership.py
    resolved_peak_analysis.py
    resolved_peak_metrics.py
    repeated_draws.py
    empirical_null.py
```
-->

<!--
resolved_peak_analysis.py
    PeakGeometry
    ResolvedPeak
    ResolvedPeakDistribution
    resolution functions

resolved_peak_metrics.py
    ResolvedPeakDistributionSummary
    ResolvedPeakMetricDefinition
    metric registry
    summary functions

repeated_draws.py
    RepeatedDrawTable
    repeated-draw validation

empirical_null.py
    EmpiricalNullResult
    empirical-null reduction
```
-->

Analysis orchestration remains outside the foundational core:

```text
analysis/
    resolved_peak_bootstrap_comparison.py
    resolved_peak_downsampling_stability.py
    truth_peak_calibration.py
```

---

# 24. Lean V0 implementation plan for an agent

The implementation target is intentionally smaller than the full architecture in
this document. Everything not listed here is extra and remains in the spec only
for future/documentation-intent purposes.

## Goal

Make one existing observed-path analysis stop reconstructing arrays ad hoc and
instead pass `DistributionRecord` objects through the resolved-peak pipeline.

The target vertical slice is:

```text
points
→ DistributionRecord
→ density
→ peak detection
→ membership
→ resolved peak distribution
→ resolved peak summary
→ DistributionComparison
→ observed metrics
```

## Implement now

1. Add a minimal `DistributionRecord`.

```python
@dataclass(frozen=True)
class DistributionRecord:
    distribution_id: str
    points: np.ndarray
    canonical_grid: CanonicalGrid
    densities: Mapping[str, DensityField] = field(default_factory=dict)
    peak_detections: Mapping[str, PeakDetectionResult] = field(default_factory=dict)
    peak_memberships: Mapping[str, EmpiricalPeakMembership] = field(default_factory=dict)
    resolved_peak_distributions: Mapping[str, ResolvedPeakDistribution] = field(default_factory=dict)
    resolved_peak_summaries: Mapping[str, ResolvedPeakDistributionSummary] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

2. Allow partial records. A record with only `distribution_id`, `points`,
   `canonical_grid`, and `metadata` is valid.
3. Copy retained arrays and mark them read-only, at least for
   `DistributionRecord.points`, `DensityField.values`, and membership arrays.
4. Add `replace=False` to every `add_*` helper and reject duplicate product names
   by default.
5. Implement only these enrichment functions:

```python
add_density(...)
add_peak_detection(...)
add_peak_membership(...)
add_resolved_peak_distribution(...)
add_resolved_peak_summary(...)
```

6. Add a minimal `DistributionComparison`.

```python
@dataclass(frozen=True)
class DistributionComparison:
    comparison_id: str
    members: Mapping[str, DistributionRecord]
    observed_metrics: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

7. Implement:

```python
add_observed_metrics(
    comparison,
    name="primary",
    summary_name="primary",
    reference_role="reference",
    target_role="target",
    metric_registry=RESOLVED_PEAK_METRICS,
    replace=False,
)
```

8. In `add_observed_metrics(...)`, validate that required member roles exist and
   that the required summaries exist.
9. Add a minimal grid compatibility check for geometry-aware comparisons. This
   can be simple in V0: compare `canonical_grid` bounds/coordinates or `grid_id`
   plus coordinates before computing observed contrasts.
10. Convert exactly one existing observed-path caller, preferably the code path
    used by `valley_visualization.py`, to build `DistributionRecord`s and consume
    their typed products.
11. Keep existing scientific outputs numerically unchanged for the converted
    observed path.
12. Add or update tests for the converted vertical slice.

## Do not implement yet

These are intentionally deferred:

```text
extension_products
ProductMetadata
product_key(...)
list_products()
generic provenance/dependency validation
RepeatedDrawTable
null_draws
stability_draws
null_tests
truth memberships
truth peak resolution
custom product insertion
full DensityField/DensityGrid repo-wide migration
```

They stay in this document as the intended direction, not as the V0 scope.

## Follow-up after V0 works

Only after one observed-path caller is cleaner and numerically stable:

1. Add `RepeatedDrawTable`.
2. Move downsampling stability out of visualization code.
3. Add raw null-draw retention.
4. Add null-test summaries derived from retained draws.
5. Revisit `ProductMetadata`, `extension_products`, and inspection helpers if
   concrete debugging or provenance needs justify them.

---

# Architectural verdict

The central ontology is:

```text
support
→ density
→ evidence
→ membership
→ geometry
→ resolved distribution
→ summary
→ comparison
→ repeated draws
→ inference
```

The central software pattern is:

```text
persistent scientific record
+
pure enrichment function
→ persistent scientific record with one more typed product
```

The V0 public interface should remain dictionary-shaped:

```python
distribution.densities["primary"]
distribution.peak_detections["primary"]
distribution.peak_memberships["primary"]
distribution.resolved_peak_distributions["primary"]
distribution.resolved_peak_summaries["primary"]

comparison.observed_metrics["primary"]
```

Later, if repeated-draw products are added, the same dictionary-shaped style can
extend to:

```python
comparison.null_draws["pooled_label"]
comparison.stability_draws["downsample_80pct"]
comparison.null_tests["pooled_label"]
```

Keep the ontology compositional, but make the interface domain-specific and pleasantly ordinary.

The scientific classes define what each result means.

The dictionaries define where users find it.

The records define what it belongs to.

The enrichment functions define how the analysis composes.
