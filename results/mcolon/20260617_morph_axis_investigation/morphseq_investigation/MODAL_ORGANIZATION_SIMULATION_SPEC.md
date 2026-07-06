## Purpose

This spec turns the modal-organization benchmark plan into an executable
simulation contract.

The key rule is that all claims are relative. A distribution is not compact,
modal, separated, bridged, or fragmented in isolation. It is more or less compact,
more or less modal, more or less separated, and more or less bridged than another
distribution under a named metric.

The simulation suite should therefore evaluate each metric by all-by-all
comparisons across controlled synthetic distributions. For a given metric, the
benchmark should report which pairwise orderings are recovered, which are
ambiguous, and which fail.

The generator should therefore define density components first, compose them into
a true density landscape, and only then sample observations from that landscape.
Words like `compact`, `diffuse`, `low_bridge`, and `high_bridge` are not primary
definitions. They are labels derived from explicit component masses, density
profiles, measured density ratios, widths, and observation processes.

---

## Core Simulation Primitive: Component-Based Density Composition

Each synthetic distribution should be defined by a fixed probability-mass budget
distributed across density components.

The primitive specification is not a scenario label. The primitive specification
is:

```text
DensitySpec =
    canonical_grid
    components
    mass_budget
```

Allowed component types:

```text
mode
bridge
background
outlier
```

Each component has:

```text
component_id
component_type
mass_fraction
geometry_type
anchor_x
anchor_y
orientation_angle
local_width
anisotropy_ratio
density_profile
parameters
```

Geometry always includes location. The location stored in a component spec is an
anchor: the coordinate used to generate the component. It is not necessarily the
same as the component's mass centroid or the resolved peak location after all
components are composed.

```text
component_anchor =
  specified location used to generate the component

component_mass_centroid =
  centroid measured from the evaluated component density on the grid

resolved_peak_location =
  local maximum measured from the composed true density landscape
```

For a Gaussian mode, the anchor is usually the mean. For a bridge, the anchor may
be the midpoint between endpoints. For a spiral or path-like component, the
anchor is the origin of the path coordinate frame.

### Component Relationships

In V0, component relationships are only used for bridge components.

A bridge component must specify:

```text
connected_components:
  exactly two component IDs
```

Both IDs must refer to mode components. No other component type may specify
`connected_components`.

```text
ModeComponent:
  connected_components: not allowed / empty

BridgeComponent:
  connected_components: exactly two mode component IDs

BackgroundComponent:
  connected_components: not allowed / empty

OutlierComponent:
  connected_components: not allowed / empty
```

This keeps the density recipe simple: modes define intended high-density
regions; bridges define intended intermediate density between two modes;
background and outlier components are not connected to anything.

Example:

```text
mode_left:
  component_type: mode
  mass_fraction: 0.42
  anchor_x: -1.0
  anchor_y: 0.0

mode_right:
  component_type: mode
  mass_fraction: 0.42
  anchor_x: 1.0
  anchor_y: 0.0

bridge_left_right:
  component_type: bridge
  mass_fraction: 0.16
  connected_components: [mode_left, mode_right]
  geometry_type: capsule
  local_width: 0.20
```

Invalid in V0:

```text
mode_left:
  component_type: mode
  connected_components: [mode_right]

background:
  component_type: background
  connected_components: [mode_left, mode_right]

bridge_abc:
  component_type: bridge
  connected_components: [mode_a, mode_b, mode_c]
```

Branching structures can be represented later as multiple pairwise bridges:

```text
bridge_ab: [mode_a, mode_b]
bridge_ac: [mode_a, mode_c]
```

The mass budget must be explicit:

```text
sum(component.mass_fraction) = 1.0
```

Each component is evaluated on the canonical grid and normalized independently:

```text
sum(F_component * dx * dy) = 1
```

The true density landscape is then composed as:

```text
F_composed(x, y) =
    sum_c component_mass_c * F_component_c(x, y)
```

and validated so that:

```text
sum(F_composed * dx * dy) = 1
```

### Component Truth Versus Composed Density Truth

The component specification describes the intended recipe. The composed density
landscape describes the realized distribution.

These are not guaranteed to match one-to-one. For example:

```text
component truth:
  mode_left: 0.42
  mode_right: 0.42
  bridge_left_right: 0.16

composed density truth:
  either two peaks with a shallow saddle
  or one broad merged mode
```

Both outcomes are valid. The composed density truth is measured from the final
composed density landscape `F_composed`, not inferred from the component names.

The benchmark should therefore keep two truth layers:

```text
component truth = what was specified
composed density truth = measurements computed from F_composed
```

Component truth records:

```text
component_id
component_type
mass_fraction
geometry_type
anchor_x
anchor_y
orientation_angle
local_width
anisotropy_ratio
density_profile
parameters
component_mass_centroid_x
component_mass_centroid_y
```

Composed density truth records:

```text
resolved_peak_count
resolved_peak_locations
resolved_peak_masses
resolved_peak_heights
resolved_peak_basins
resolved_mode_compactness
resolved_saddle_densities
resolved_valley_depths
bridge_region_masses
resolved_relative_peak_separations
```

Scenario names and component labels are never treated as truth. Truth is measured
from the composed density landscape.

### MVP Truth Descriptors

The first resolved truth layer only needs a small mode passport.

For each resolved mode:

```text
ModeTruth =
    resolved_mode_id
    source_component_ids
    mass_fraction
    component_anchor_x
    component_anchor_y
    component_centroid_x
    component_centroid_y
    peak_location_x
    peak_location_y
    peak_height
    fixed_mass_area_50
    fixed_mass_area_80
    effective_radius_50
    effective_radius_80
    compactness_50
    compactness_80
    anisotropy_ratio
    density_profile
```

For each neighboring resolved mode pair:

```text
ValleyTruth =
    resolved_mode_i
    resolved_mode_j
    center_distance
    relative_peak_separation_50
    relative_peak_separation_80
    saddle_density
    valley_density_ratio
    valley_depth
    bridge_component_id
    bridge_component_mass_fraction
    bridge_region_density_ratio
    bridge_region_mass_fraction
```

This is enough to describe where modes are, how much mass they contain, how
compact they are, how tall they are, and how strongly neighboring modes are
connected or separated.

### Conceptual Order

The simulation architecture is:

```text
1. specify mode, bridge, background, and outlier components
2. allocate an explicit probability-mass budget
3. evaluate each component on the canonical grid
4. normalize each component independently
5. weight components by assigned mass
6. sum components to form F_composed
7. resolve realized peaks, basins, saddles, bridges, and compactness
8. sample finite observations from F_composed
9. estimate metrics from sampled observations
10. compare estimated metrics to composed density truth
```

---

## Implementation Scope

This document is the north-star simulation contract. It defines the full
conceptual target so the benchmark does not drift back into scenario-name
classification.

The immediate coding target should be much smaller: a V0 benchmark that tests
whether the current metrics recover the basic relative orderings implied by the
ontology.

V0 should not implement the full registry, full metric ontology, full
sample-size ladder, full observation-process suite, or full visualization
system. Those are later expansions after the minimal benchmark shows that the
core metrics behave sensibly.

### V0 Benchmark

Use only these distributions:

```text
one_peak_compact
one_peak_diffuse
one_peak_elongated
one_peak_spiral
spiral_beaded
two_peaks_high_bridge
two_peaks_low_bridge
two_peaks_no_bridge
three_peaks_compact
```

Use one sample size first, chosen to match the main biological use case.

Use only these metric families:

```text
concentration:
  hdr_concentration_auc

modal_organization:
  HDR component persistence / ge2 persistence

separation:
  valley_depth or merge density

resolution_warning:
  kNN radius variability or MST edge ratio
```

V0 should produce:

- one per-distribution metric table
- one pairwise ordering table
- three heatmaps: concentration, modal organization, separation

Before any metric tables are generated, V0 should render a visual QA grid:

```text
morphseq_investigation/v0/plot_modal_v0_distribution_qc.py
```

This figure should show the composed true density landscape, sampled point
cloud, and optional KDE estimate so generator problems can be distinguished from
sampling or estimator problems.

V0 per-distribution metric table:

```text
distribution_id
n
seed
hdr_concentration_auc
hdr_ge2_persistence
valley_depth
resolution_warning_metric
resolution_warning_flag
notes
```

V0 pairwise ordering table:

```text
metric_family
distribution_a
distribution_b
stat_a
stat_b
effect
expected_relation
observed_relation
pass_fail_ambiguous
notes
```

V0 should answer:

- Does compact rank more concentrated than diffuse?
- Does `two_peaks_low_bridge` rank more separated than `two_peaks_high_bridge`?
- Does `three_peaks_compact` rank more modal than a smooth spiral?
- Does `spiral_beaded` rank more modal than `one_peak_spiral`?
- Does the smooth spiral avoid a false multi-mode call?

V0 expected orderings:

```text
concentration:
  one_peak_compact > one_peak_diffuse

separation:
  two_peaks_no_bridge > two_peaks_low_bridge > two_peaks_high_bridge

modal_organization:
  three_peaks_compact > two_peaks_low_bridge > one_peak_compact
  spiral_beaded > one_peak_spiral
  one_peak_spiral ~= one_peak_compact
```

If these fail, debug the generator and metrics before expanding the benchmark.

Defer until after V0:

- full `MetricSpec` registry
- full true-density validation suite
- sample-size ladder
- complete shape ladder
- HDBSCAN/GMM comparisons
- full observation-process taxonomy
- full visualization panel system

---

## Existing Utilities To Reuse

The current investigation folder already has most of the pieces needed for this.

### Synthetic Distribution Utilities

Use `synthetic_scenarios.py` as the first generator registry.

Existing assets:

- `Scenario`
- `SCENARIOS`
- `SCENARIOS_BY_NAME`
- `wt_reference`
- generators for compact, broad, tail, crescent, spiral, outlier, two-cluster,
  three-cluster, weak-separation, small-middle, and hole cases

These are useful, but the next version should add explicit quantitative
configuration metadata so scenario names are not the source of truth.

### Support Metrics

Use `support_geometry.py` for support and bridge/separation witnesses.

Existing reusable metrics:

- `valley_depth`
- `hdr_concentration_auc`
- `hdr_mass_area_profile`
- `mst_max_edge`
- `fiedler_value`
- `conductance`
- `compute_support_geometry`
- `SupportGeometryBundle`
- `StatResult`
- `KDESpec`
- `normalize_shape`

Important behavior already implemented:

- Each metric is compared to a matched-N reference null.
- Support metrics are shape-normalized before evaluation.
- The metrics are reported in parallel, not hidden behind one fused score.
- Disagreements are retained for interpretation.

### Density Geometry Metrics

Use `density_geometry.py` for within-support descriptors.

Existing reusable metrics:

- variance
- IQR
- skewness
- kurtosis
- tail index
- entropy
- `compute_density_geometry`
- `DensityGeometry`
- `DescriptorResult`

Important behavior already implemented:

- Density descriptors are reported against a matched-N WT bootstrap null.
- Results include percentile relative to the reference null.

### Component Metrics

Use `component_geometry.py` after support separation has been established.

Existing reusable metrics:

- GMM+BIC component count
- HDBSCAN component count when available
- component sizes
- agreement flag between component estimators

### Visualization Utilities

Reuse the visualization style from:

- `valley_visualization.py`
- `connectedness_panel.py`
- `hdr_area_validation.py`

Useful established visual motifs:

- target KDE on the real coordinate frame
- matched reference/null strip showing observed value relative to null
- HDR mass contour for concentration
- valley contour only when the valley witness is significant
- raw point panels alongside density summaries
- per-metric vote/status table

The simulation benchmark should use these same visual ideas, but orient them
around controlled all-by-all metric validation.

---

## Benchmark Metadata Object

Each synthetic distribution should be represented by explicit metadata.

```text
DistributionSpec =
    id
    generator
    n
    seed
    density_spec
    component_truth
    composed_density_truth
    observation_process
    expected_relative_scores
```

The important additions are `density_spec`, `component_truth`,
`composed_density_truth`, and `expected_relative_scores`.

`density_spec` defines the component recipe and mass budget.
`component_truth` records the intended density components.
`composed_density_truth` records measurements computed from the final composed
density `F_composed`.
`expected_relative_scores` encodes expected directional relationships for each
metric family, not just scenario labels.

Example:

```text
id: one_peak_compact
expected_relative_scores:
  concentration:
    greater_than: [one_peak_diffuse, one_peak_uniform_broad]
    less_than: []
    tied_with: [one_peak_compact_shifted]
  peak_count:
    tied_with: [one_peak_diffuse, one_peak_arc, one_peak_spiral]
  valley_separation:
    less_than: [two_peaks_low_bridge, two_peaks_no_bridge]
```

This makes "compact" testable as a relative ordering.

---

## Density Composition Contract

The simulation should be component-first. Scenario names such as `compact`,
`diffuse`, `low_bridge`, or `high_bridge` are only aliases for measured
properties of the composed density landscape.

Numeric density parameters are primary. Qualitative labels are derived.

```text
component recipe -> composed true density -> measured true-density quantity -> derived label
```

For example, store `peak_width_ratio`, `peak_height_ratio`, and
`hdr_area_auc_true`; derive `compact_label` from those values. Do not store
`compact` or `diffuse` as the source of truth.

Each distribution must define a component-based `DensitySpec`:

```text
DensitySpec =
    canonical_grid
    components
    mass_budget
```

Minimal component fields:

```text
canonical_grid:
  dimension
  x_range
  y_range
  grid_size
  dx
  dy

component:
  component_id
  component_type         # mode, bridge, background, outlier
  mass_fraction
  geometry_type          # point, ellipse, strip, path, spiral, capsule, region
  anchor_x
  anchor_y
  orientation_angle
  local_width
  anisotropy_ratio
  density_profile        # gaussian, uniform, radial_decay, ridge, flat_core
  connected_components   # bridge only; exactly two mode component IDs
  parameters
```

All density-strength labels must map to dimensionless ratios computed from the
resolved true density before sampling.

### Canonical Density Ratios

For a neighboring peak pair `(i, j)`, define:

```text
peak_density_pair = min(local_max_density_i, local_max_density_j)
valley_density_ratio = saddle_or_min_bridge_density / peak_density_pair
bridge_component_mass_fraction = probability mass assigned to the bridge component in the recipe
bridge_region_density_ratio = mean(F_composed over bridge_region) / peak_density_pair
bridge_region_mass_fraction = sum(F_composed[cell] * dx * dy for cell in bridge_region)
inter_peak_distance_ratio = inter_peak_distance / mean_peak_width
bridge_width_ratio = bridge_width / mean_peak_width
saddle_density_ratio = saddle_density / peak_density_pair
peak_prominence_ratio = 1 - saddle_density_ratio
```

These ratios make "low", "medium", and "high" comparable across distributions.
The exact density units do not matter; the ratios are evaluated relative to the
neighboring peaks in the same density landscape.

`valley_density_ratio` is a pointwise or bottleneck statistic: it summarizes the
lowest-density saddle or constriction between peaks. Bridge-region quantities
summarize how much intermediate density lies inside a defined bridge region of
the composed density landscape.

Recipe-level bridge mass and measured bridge-region mass should be stored
separately. `bridge_component_mass_fraction` records the probability budget
assigned to the bridge component. `bridge_region_mass_fraction` records the
probability mass measured inside the bridge/intermediate region of `F_composed`.
These need not be identical because mode tails, background, and the bridge
component can all contribute density inside the bridge region.

For V0, do not infer arbitrary bridge regions between every pair of peaks. Use
the bridge component's geometry to define a bridge mask, usually a capsule or
path mask between the two connected mode anchors. Later versions may infer
bridge regions from basin boundaries or saddle merge events.

Recommended label mapping:

```text
no_bridge:
  bridge_region_density_ratio <= 0.01
  bridge_region_mass_fraction <= 0.01

low_bridge:
  0.01 < bridge_region_density_ratio <= 0.10
  0.01 < bridge_region_mass_fraction <= 0.05

moderate_bridge:
  0.10 < bridge_region_density_ratio <= 0.30
  0.05 < bridge_region_mass_fraction <= 0.15

high_bridge:
  0.30 < bridge_region_density_ratio <= 0.70
  0.15 < bridge_region_mass_fraction <= 0.30

fully_merged:
  bridge_region_density_ratio > 0.70
```

Bridge density alone does not define the true mode count. A high-bridge case can
still have two modes if a saddle exists. A fully merged case should have no
meaningful saddle and should be treated as one broad mode.

```text
two_peaks_high_bridge:
  true_density_peak_count = 2 if saddle_exists and saddle_density_ratio < 1

two_peaks_fully_merged:
  true_density_peak_count = 1 if no meaningful saddle exists
```

Within-peak concentration labels should also be numeric. Define a reference
local width `w_ref` in the canonical coordinate frame, then set:

```text
peak_width_ratio = peak_width / w_ref
peak_height_ratio = local_max_density / reference_local_max_density
```

Recommended concentration ladder:

```text
very_compact:
  peak_width_ratio = 0.50

compact:
  peak_width_ratio = 0.75

reference:
  peak_width_ratio = 1.00

diffuse:
  peak_width_ratio = 1.50

very_diffuse:
  peak_width_ratio = 2.00
```

For equal-mass peaks in the same dimension, narrower peaks should imply higher
peak density. The generator should compute or record the resulting
`peak_height_ratio` rather than rely on the label.

Valley separation should use the opposite orientation:

```text
valley_depth = 1 - valley_density_ratio

larger valley_depth = more separated
larger bridge_region_density_ratio = more connected
```

Peak significance should also be explicit:

```text
mass_significant_peak:
  peak_mass_fraction >= 0.05

weak_satellite_peak:
  0.01 <= peak_mass_fraction < 0.05

outlier_or_noise:
  peak_mass_fraction < 0.01
```

These cutoffs are benchmark conventions, not biological truths. They keep
expected pairwise relationships stable across seeds and sample sizes.

For flat or nearly flat density regions, local maxima should be filtered by peak
prominence and mass thresholds so that numerical grid noise is not counted as
multiple modes.

### True-Density Validation

Before sampling finite observations, every `DensitySpec` should be validated on
the analytic or grid-evaluated true density produced by component composition.

The validation step should compute:

```text
true_density_peak_count
true_saddle_exists
true_saddle_density_ratios
true_bridge_region_density_ratios
true_bridge_region_mass_fractions
true_hdr_area_auc
true_support_area
true_peak_prominence_ratios
```

The measured true-density values should be compared against the intended
component truth. Scenario labels such as `compact`, `diffuse`, `low_bridge`, and
`high_bridge` are accepted only if the measured density ratios match the
configured numeric ranges.

This prevents synthetic scenarios from silently violating their intended
geometry.

### Density Versus Observation

The true density and the observation process must remain separate.

Example:

```text
one_peak_spiral_ridge_decay + iid
one_peak_spiral_ridge_decay + missing_segment
one_peak_spiral_ridge_decay + underresolved
spiral_beaded + iid
```

The first three share the same underlying density but differ in observation
process. The last has a different density landscape and true modal organization.

This separation is essential. Missing segments, underresolution, and dropout can
create observed fragmentation without changing the underlying density.

Each distribution should therefore distinguish true structure from expected
finite-sample behavior:

```text
peak_count_true
expected_observed_peak_count
expected_observed_fragmentation
observation_artifact_expected
expected_resolution_quality
```

Example:

```text
one_peak_spiral_missing_segment:
  peak_count_true: 1
  expected_observed_peak_count: ambiguous_or_fragmented
  expected_observed_fragmentation: elevated
  observation_artifact_expected: yes
  expected_resolution_quality: poor
```

A metric may correctly detect observed fragmentation in this case, but the final
interpretation should flag the result as observation-limited rather than
confidently biological.

### Minimal V1 Component Description

The first implementation only needs enough structure to test the benchmark's
core claims. Each density recipe should therefore specify:

```text
canonical_grid
mode_components
bridge_components
background_components
component_mass_fractions
observation_process
```

These fields are sufficient to define:

- how much mass is assigned to intended modes and bridges
- how concentrated each peak is
- whether shape differs from modal organization
- how strongly neighboring peaks are separated or bridged
- whether apparent fragmentation comes from density or observation process

More detailed fields, such as local covariance along a path or heterogeneous
bridge width, can be added later only when a metric needs them.

---

## Metric Families And Relative Targets

Each metric family gets its own all-by-all comparison matrix.

Every metric should also be registered with an explicit orientation.

```text
MetricSpec =
    metric_name
    metric_family
    orientation
    larger_means
    effect_definition
    validity_scope
    known_failure_modes
```

Examples:

```text
hdr_concentration_auc:
  metric_family: concentration
  orientation: lower_is_more_concentrated
  effect_definition: stat_b - stat_a
  validity_scope: all distributions
  known_failure_modes: scale dependence, underresolution

valley_depth:
  metric_family: separation
  orientation: higher_is_more_separated
  effect_definition: stat_a - stat_b
  validity_scope: distributions with candidate peaks
  known_failure_modes: bandwidth oversmoothing, outliers, curved supports

bridge_density:
  metric_family: bridge_continuity
  orientation: higher_is_more_connected
  effect_definition: stat_a - stat_b
  validity_scope: distributions with candidate peak pairs
  known_failure_modes: KDE oversmoothing, underresolved bridges
```

This prevents accidental mixing of metric directions in all-by-all heatmaps.

### 1. Concentration

Question:

> Does distribution A pack the same mass into less area than distribution B?

Primary utilities:

- `hdr_concentration_auc`
- `hdr_mass_area_profile`
- density-geometry variance/IQR as secondary descriptors

Expected direction:

```text
one_peak_compact > one_peak_diffuse
one_peak_compact > one_peak_uniform_broad
two_peaks_low_bridge > two_peaks_high_bridge, if each peak is locally compact
small_middle > one_peak_diffuse, for dense-mass concentration
```

Expected ties or near-ties:

```text
one_peak_compact ~= one_peak_compact_shifted
one_peak_arc ~= one_peak_spiral, if local width and sampling density are matched
```

Failure modes to detect:

- A metric calls a broad one-peak distribution "more modal" merely because it is
  diffuse.
- A metric confuses translation with concentration.
- A metric confuses curve length with low concentration when local width is
  unchanged.

### 2. Peak Count / Modal Organization

Question:

> Does distribution A have more mass-significant observed peaks than distribution B?

Primary utilities:

- HDR component count from KDE super-level sets
- component persistence across HDR thresholds
- `component_geometry.py` component counts after support separation

Expected direction:

```text
two_peaks_low_bridge > one_peak_compact
three_peaks_low_bridge > two_peaks_low_bridge
spiral_beaded > spiral_ridge_decay
one_large_plus_satellite > one_peak_core_tail_monotone, if satellite is mass-significant
```

Expected ties or near-ties:

```text
one_peak_round ~= one_peak_elongated
one_peak_arc ~= one_peak_round
one_peak_spiral_ridge_decay ~= one_peak_round
```

Failure modes to detect:

- Curvature creates false modes.
- An elongated ridge creates false modes.
- A monotone tail is counted as a second mode.
- A rare but real middle mode is erased.

### 3. Valley Separation

Question:

> Are peaks in A separated by deeper valleys or weaker bridges than peaks in B?

Primary utilities:

- `valley_depth`
- merge/saddle density from `valley_detection_detail`
- `mst_max_edge`
- `fiedler_value`
- `conductance`

Expected direction:

```text
two_peaks_no_bridge > two_peaks_low_bridge > two_peaks_moderate_bridge > two_peaks_high_bridge
three_peaks_no_bridge > three_peaks_high_bridge
small_middle < two_peaks_no_bridge, for complete support disconnection
small_middle > two_peaks_high_bridge, for bottleneck/valley structure
```

Expected ties or near-ties:

```text
one_peak_compact ~= one_peak_diffuse, after shape normalization for support metrics
one_peak_arc ~= one_peak_spiral, if both are connected with similar local sampling
```

Failure modes to detect:

- Outliers cause a false broken-support call.
- A curved but connected manifold is treated as disconnected.
- KDE bandwidth smooths over a real weak middle mode.
- Graph metrics fire on thin manifolds without a density valley.

### 4. Bridge Density / Support Continuity

Question:

> Given multiple peaks, how much intermediate support connects them?

Primary utilities:

- `valley_depth`
- graph bottleneck metrics
- bridge mass estimated from density between peak basins
- visual KDE bridge panels

Expected direction:

```text
two_peaks_high_bridge > two_peaks_moderate_bridge > two_peaks_low_bridge > two_peaks_no_bridge
```

Note that this direction is opposite of valley separation. The benchmark should
store metric orientation explicitly.

```text
valley_separation: larger = more separated
bridge_density: larger = more connected
```

Failure modes to detect:

- A method reports high bridge density for a KDE artifact produced by oversmoothing.
- A method reports no bridge in an under-sampled but truly connected continuum
  without flagging resolution risk.

### 5. Shape / Anisotropy

Question:

> Does A differ from B in shape without necessarily differing in peak count?

Primary utilities:

- density-geometry variance/IQR on canonical projection
- covariance eigenvalue ratio
- internal distance distribution
- future shape descriptors added to `density_geometry.py`

Expected direction:

```text
one_peak_elongated > one_peak_round, for anisotropy
one_peak_arc > one_peak_round, for curvature/nonlinearity
one_peak_spiral > one_peak_arc, for curvature/path complexity
```

Expected ties or near-ties:

```text
one_peak_elongated ~= one_peak_round, for peak count
one_peak_arc ~= one_peak_round, for support disconnection
```

Failure modes to detect:

- A shape difference is misreported as modal separation.
- A compact curved manifold is misreported as multiple peaks.

### 6. Observation / Resolution Effects

Question:

> Does the method distinguish true organization from sampling artifacts?

Primary utilities:

- matched-N nulls in `compute_support_geometry`
- `ConfidenceReport`
- LOO stability in `phenotype_geometry.py`
- bootstrap variability
- visualization panels with raw points and null strips

Expected direction:

```text
one_peak_missing_segment > one_peak_well_sampled, for apparent separation
one_peak_underresolved > one_peak_well_sampled, for instability
one_peak_uneven_sampling > one_peak_iid, for apparent local density heterogeneity
```

But these should carry lower stability/confidence than true multimodal cases.

Failure modes to detect:

- The method makes a confident biological discreteness claim from underresolution.
- The method ignores instability across bootstrap, bandwidth, or leave-one-out.

---

## All-By-All Evaluation Design

For each metric, build a matrix over distributions.

```text
rows    = distribution A
columns = distribution B
entry   = effect(A, B, metric)
```

Each entry should contain:

```text
metric_name
distribution_a
distribution_b
n
seed
stat_a
stat_b
effect
effect_orientation
expected_relation
observed_relation
pass_fail_ambiguous
confidence_a
confidence_b
notes
```

Recommended effect definitions:

```text
raw_delta        = stat_a - stat_b
log_ratio        = log((stat_a + eps) / (stat_b + eps))
percentile_delta = percentile_a - percentile_b
paired_win_rate  = P_seed(stat_a > stat_b)
```

For a first implementation, use `paired_win_rate` and median `raw_delta`.

```text
pass:
  expected A > B and paired_win_rate >= 0.80

ambiguous:
  expected A > B and 0.60 <= paired_win_rate < 0.80
  expected A ~= B and abs(median_delta) is small

fail:
  expected A > B and paired_win_rate < 0.60
  expected A ~= B but the metric consistently separates them
```

The thresholds are benchmark diagnostics, not biology.

---

## Sample-Size Ladder

The benchmark should evaluate metrics across sample sizes relevant to real data.

Recommended initial ladder:

```text
n = 20, 40, 80, 160
```

The final summary should report not only whether a metric succeeds, but where
its sample-size failure boundary begins.

Example:

```text
valley_depth ranks low_bridge > high_bridge at N >= 80,
but is ambiguous at N = 20.
```

---

## Simulation Grid

The first complete grid should vary one property at a time.

### Concentration Ladder

```text
one_peak_very_compact
one_peak_compact
one_peak_reference
one_peak_diffuse
one_peak_very_diffuse
```

Controlled fields:

```text
peak_count = 1
geometry = round
within_peak_density_profile = radial_decay
valley_structure = none
observation_process = iid
```

Expected:

```text
concentration decreases monotonically
peak_count remains tied
valley_separation remains tied or near-tied after support normalization
```

### Shape Ladder

```text
one_peak_round
one_peak_elongated
one_peak_arc
one_peak_spiral
one_peak_branch
```

Controlled fields:

```text
peak_count = 1
local_width = matched
mass = matched
observation_process = iid
```

Expected:

```text
shape metrics change
peak_count remains tied
support_disconnection remains low
```

### Valley / Bridge Ladder

```text
two_peaks_no_bridge
two_peaks_low_bridge
two_peaks_moderate_bridge
two_peaks_high_bridge
two_peaks_fully_merged
```

Controlled fields:

```text
peak_width = matched
peak_mass_balance = matched
inter_peak_distance = matched
bridge_region_density_ratio = [0.00, 0.05, 0.20, 0.50, >0.70]
bridge_region_mass_fraction = [0.00, 0.03, 0.10, 0.20, >0.30]
saddle_exists = true until fully_merged
```

Expected:

```text
valley_separation decreases monotonically
bridge_density increases monotonically
true_density_peak_count remains 2 while a saddle exists
true_density_peak_count becomes 1 for fully_merged if no saddle exists
support_disconnected is yes only at no_bridge / low_bridge, depending on threshold
```

The bridge labels are aliases for the canonical density ratios above, not
free-text qualitative categories.

### Peak Count Ladder

```text
one_peak_round
two_peaks_low_bridge
three_peaks_low_bridge
many_weak_peaks
```

Controlled fields:

```text
peak_width = matched
valley_structure = low_bridge or no_bridge
total_n = matched
```

Expected:

```text
peak_count increases
concentration may change and must be reported separately
valley statistics should not be interpreted as peak count
```

### Same Geometry, Different Modal Organization

```text
spiral_ridge_decay
spiral_beaded_weak
spiral_beaded_strong
```

Controlled fields:

```text
geometry = spiral
path_length = matched
local_width = matched
bridge_region_density_ratio = high enough to remain connected unless testing bead gaps
```

Expected:

```text
modal organization increases across beading strength
shape complexity remains tied
support continuity remains connected unless bead gaps are truly empty
```

### Same Distribution, Different Observation Process

```text
one_peak_iid
one_peak_underresolved
one_peak_uneven_sampling
one_peak_missing_segment
one_peak_density_weighted_dropout
one_peak_outlier_contaminated
```

Controlled fields:

```text
true_density_spec = one_peak_round
```

Expected:

```text
apparent fragmentation may increase
confidence/stability should decrease
interpretation should flag observation process
```

---

## Output Tables

The benchmark should write four table types.

### 1. Distribution Table

One row per generated distribution instance.

```text
distribution_id
family
n
seed
density_spec_id
component_truth_id
true_density_peak_count
true_saddle_exists
true_saddle_density_ratios
true_peak_prominence_min
expected_observed_peak_count
expected_observed_fragmentation
observation_artifact_expected
expected_resolution_quality
peak_mass_balance
peak_width
geometry
within_peak_density_profile
local_width
anisotropy_ratio
inter_peak_distance_ratio
valley_density_ratio
valley_depth
bridge_region_density_ratio
bridge_component_mass_fraction
bridge_region_mass_fraction
background_density_ratio
outlier_mass_fraction
observation_process
```

### 2. Metric Table

One row per distribution, seed, and metric.

```text
distribution_id
n
seed
metric_name
metric_family
metric_orientation
metric_validity_scope
stat
reference_stat
percentile_vs_reference
pvalue_vs_reference
confidence
metric_failure_warning
```

### 3. Pairwise Metric Table

One row per pair of distributions and metric.

```text
metric_name
distribution_a
distribution_b
n
seed
stat_a
stat_b
effect
effect_orientation
expected_basis
observed_relation
expected_relation
pass_fail_ambiguous
```

### 4. Metric Validation Summary

One row per metric and controlled contrast.

```text
metric_name
contrast_family
n
n_seeds
n_pairs
pass_rate
ambiguous_rate
fail_rate
median_effect
paired_win_rate
failure_boundary_n
main_failure_mode
```

---

## Visualization Outputs

Each metric family should get an all-by-all visual summary.

### Pairwise Heatmap

Rows and columns are distributions. Cell color is the signed effect for one
metric.

Required annotations:

- expected `A > B`, `A < B`, or `A ~= B`
- pass/fail/ambiguous marker
- sample size and number of seeds in the title

### Metric Ladder Plot

For controlled monotonic ladders, show the metric distribution over seeds.

Examples:

- concentration ladder for `hdr_concentration_auc`
- bridge ladder for `valley_depth`
- shape ladder for anisotropy metrics

### Example Panel

For a small selected set of distributions, reuse the existing visual grammar:

- raw point cloud
- target KDE
- HDR mass contour
- valley contour when significant
- reference/null strip showing observed value relative to matched-N null

---

## Implementation Plan

### Phase 0: Minimal V0 Benchmark

Add a small visual QA script:

```text
morphseq_investigation/v0/plot_modal_v0_distribution_qc.py
```

Responsibilities:

- generate the nine V0 distributions
- render the pre-metric distribution QC grid

Then add the minimal metric runner:

```text
run_modal_v0_benchmark.py
```

Responsibilities:

- reuse the same nine V0 distributions from `morphseq_investigation/v0/modal_v0_distributions.py`
- compute the four V0 metric outputs
- write one per-distribution metric table
- write one pairwise ordering table
- plot concentration, modal organization, and separation heatmaps
- flag obvious resolution or observation artifacts

Do not build the full registry or validation framework in this phase. Use
explicit local configuration for the nine distributions and keep the output
tables compact.

### Phase 1: Spec-Driven Registry

After V0 succeeds, add a new module:

```text
modal_simulation_registry.py
```

Responsibilities:

- define `DistributionSpec`
- define `DensitySpec`
- define `MetricSpec`
- encode numeric mappings for compact/diffuse and bridge-density labels
- validate composed true densities before sampling
- wrap existing `synthetic_scenarios.py` generators
- add missing ladder generators
- store expected pairwise relationships by metric family

Required validation entry point:

```text
validate_density_spec(spec)
```

This should assert that intended ratios match measured true-density ratios within
tolerance, true peak count matches expected true peak count, saddle existence is
consistent with the intended bridge class, and derived labels match measured
numeric ranges.

### Phase 2: Metric Runner

Add:

```text
run_modal_metric_benchmark.py
```

Responsibilities:

- generate all distributions for each `n` and seed
- compute selected metrics using existing utilities
- write the distribution and metric tables
- preserve full null distributions only when needed for plots

### Phase 3: Pairwise Evaluator

Add:

```text
evaluate_modal_pairwise_orderings.py
```

Responsibilities:

- build all-by-all comparisons for each metric
- compare observed ordering against expected ordering
- compute pass/ambiguous/fail summaries
- identify common failure modes

### Phase 4: Visualization

Add:

```text
plot_modal_metric_benchmark.py
```

Responsibilities:

- pairwise heatmaps
- ladder plots
- example panels using the existing KDE/HDR/null-strip style

---

## Acceptance Criteria

The benchmark is useful when it can make statements like:

```text
hdr_concentration_auc correctly recovers 94% of expected concentration orderings
across the one-peak concentration ladder and does not separate translated controls.
It fails on underresolved samples, which are flagged by low stability.
```

```text
valley_depth correctly ranks the two-peak bridge ladder from no_bridge to
high_bridge in 88% of paired seeds, but becomes ambiguous for high_bridge versus
fully_merged at N=20.
```

```text
graph bottleneck metrics detect weak support bridges, but also fire on spirals and
outliers; this confirms they should remain corroborating metrics rather than sole
discreteness calls.
```

The final report should not say only that a method "works." It should say which
metric works for which relative property, at which sample size, and where its
failure boundary begins.

---

## Simulation Results Insights

Current V0 validation shows the composed-density layer is behaving as intended:

- `F_composed` is validated before sampling.
- The V0 truth peak counts match the expected recipe counts.
- Bridge labels are now tied to measured composed-density ratios rather than
  recipe names alone.

The useful signal is not perfect per-replicate agreement. The useful signal is
the power boundary:

- concentration metrics already separate compact vs diffuse cases cleanly
- bridge-related metrics are informative but sample-size sensitive
- graph bottleneck metrics remain corroborating signals rather than sole truth
  calls
- larger `n` improves ordering stability, which is what we want to quantify

This is the main takeaway for the simulation framework: the benchmark should
report where each metric starts to separate distributions reliably, and where
it remains ambiguous. That gives us a practical operating range before we move
to direct distribution-vs-distribution comparisons.

See the companion docs:

- [MODAL_ORGANIZATION_OPEN_QUESTIONS.md](MODAL_ORGANIZATION_OPEN_QUESTIONS.md)
- [MODAL_ORGANIZATION_RESOLVED_QUESTIONS.md](MODAL_ORGANIZATION_RESOLVED_QUESTIONS.md)
