## Resolved Peak Distribution V0

All peak geometry is measured in canonical coordinates. The resolved peak layer is
not another detector: it is the semantic bridge between detector evidence and
canonical per-peak/distribution measurements.

`PeakDetectionResult` remains the detector output. `ResolvedPeakDistribution`
wraps detector evidence with the density grid and memberships needed to recompute
the resolved geometry.

### MVP boundary

Build V0 in two slices.

MVP-A is the empirical comparison path. It is required for target-versus-reference
inference:

```text
resolve_empirical_peak_distribution(...)
summarize_resolved_peak_distribution(...)
ResolvedPeakRunContext
RESOLVED_PEAK_METRICS
run_empirical_null_test(...)
summary-row export
valid-null accounting
```

MVP-A supports the actual inference loop:

```text
resampled points
-> KDE
-> peak detection
-> resolved distribution
-> scalar summary
-> null difference
-> empirical significance
```

MVP-B is the truth calibration path. Add it after MVP-A is working:

```text
resolve_truth_peak_distribution(...)
truth-to-empirical bias summaries
simulation-seed calibration tables
```

Keep the object model truth-compatible from the start, but do not let truth
resolution block the first empirical-null implementation.

### Design verdict

Use a richer in-memory object than a frozen summary row.

`ResolvedPeakDistribution` should retain the canonical geometry context used to
resolve the peaks:

```text
density_grid
sample_points, for empirical distributions
sample_peak_ids, for empirical distributions
grid_peak_ids, for truth distributions
detection_result
resolved accepted peaks
```

`DensityGrid` already carries the canonical grid, `xx`, `yy`, and `density`, so
the canonical grid should not be duplicated as a separate field.

### Object model

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from .density_composition import DensityGrid
from .peak_counting import PeakCandidateDetail, PeakDetectionResult


@dataclass(frozen=True)
class PeakGeometry:
    peak_id: int
    center_coordinate: tuple[float, float]
    total_support_fraction: float
    radius: float
    cv_radius_from_center: float


@dataclass(frozen=True)
class ResolvedPeak:
    geometry: PeakGeometry
    source_type: Literal["truth", "empirical"]
    detector_detail: PeakCandidateDetail | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPeakDistribution:
    distribution_id: str
    source_type: Literal["truth", "empirical"]

    # Canonical density representation used to resolve the peaks.
    density_grid: DensityGrid

    # Detector result retained for auditability.
    detection_result: PeakDetectionResult

    # Resolved accepted peaks.
    peaks: tuple[ResolvedPeak, ...]

    # Present for empirical distributions.
    sample_points: np.ndarray | None = None
    sample_peak_ids: np.ndarray | None = None

    # Present for truth distributions; same shape as density_grid.density.
    grid_peak_ids: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        density_shape = np.asarray(self.density_grid.density).shape
        peak_ids = {peak.geometry.peak_id for peak in self.peaks}

        if self.source_type == "empirical":
            if self.sample_points is None or self.sample_peak_ids is None:
                raise ValueError(
                    "Empirical distributions require sample_points and sample_peak_ids."
                )
            if len(self.sample_points) != len(self.sample_peak_ids):
                raise ValueError(
                    "sample_points and sample_peak_ids must have equal length."
                )
            if self.grid_peak_ids is not None:
                raise ValueError("Empirical distributions must not provide grid_peak_ids.")
            member_ids = set(np.asarray(self.sample_peak_ids, dtype=int))

        elif self.source_type == "truth":
            if self.grid_peak_ids is None:
                raise ValueError("Truth distributions require grid_peak_ids.")
            if np.asarray(self.grid_peak_ids).shape != density_shape:
                raise ValueError("grid_peak_ids must match density_grid.density shape.")
            if self.sample_points is not None or self.sample_peak_ids is not None:
                raise ValueError("Truth distributions must not provide empirical memberships.")
            member_ids = set(np.asarray(self.grid_peak_ids, dtype=int).ravel())

        else:
            raise ValueError(f"Unknown source_type: {self.source_type!r}")

        nonnegative_member_ids = {int(member_id) for member_id in member_ids if int(member_id) >= 0}
        if not nonnegative_member_ids.issubset(peak_ids):
            missing = sorted(nonnegative_member_ids - peak_ids)
            raise ValueError(
                "Every nonnegative membership id must correspond to an accepted "
                f"ResolvedPeak; missing ids: {missing}"
            )
```

Do not add V0 subclasses such as `SimulatedPeak` or `CandidateEmpiricalPeak`.
Use `source_type` and `provenance` until behavior actually diverges.

Do not add a `PeakCollection` wrapper in V0. A tuple of `ResolvedPeak` objects is
enough unless repeated collection behavior appears.

### Derived properties

These values should be derived from stored memberships and peaks, not stored as
independent mutable summary fields:

```python
@property
def number_of_peaks(self) -> int:
    return len(self.peaks)


@property
def assigned_support_fraction(self) -> float:
    if self.source_type == "empirical":
        if self.sample_peak_ids is None:
            return np.nan
        if len(self.sample_peak_ids) == 0:
            return np.nan
        return float(np.mean(self.sample_peak_ids != -1))

    if self.grid_peak_ids is None:
        return np.nan
    density = np.asarray(self.density_grid.density, dtype=float)
    if self.density_grid.grid is None:
        weights = density
    else:
        weights = density * float(self.density_grid.grid.cell_area)
    total = float(np.sum(weights))
    if total <= 0:
        return np.nan
    return float(np.sum(weights[self.grid_peak_ids != -1]) / total)


@property
def unassigned_support_fraction(self) -> float:
    assigned = self.assigned_support_fraction
    if np.isnan(assigned):
        return np.nan
    return float(1.0 - assigned)


@property
def assigned_sample_fraction(self) -> float | None:
    if self.source_type != "empirical":
        return None
    return self.assigned_support_fraction


@property
def unassigned_sample_fraction(self) -> float | None:
    if self.source_type != "empirical":
        return None
    return self.unassigned_support_fraction
```

### Array immutability

`dataclass(frozen=True)` prevents field reassignment, but it does not make NumPy
arrays immutable. Builders should copy retained arrays and mark them read-only:

```python
def _readonly_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result
```

Apply this to:

```text
sample_points
sample_peak_ids
grid_peak_ids
```

This prevents later mutation from silently invalidating derived peak geometry.

### Empirical membership convention

For empirical distributions, store one assignment per sample:

```text
sample_peak_ids[i] = accepted candidate peak id
sample_peak_ids[i] = -1 for unassigned or rejected support
```

For V0, nearest-center assignment is acceptable, but samples must first be
assigned to all detector candidates before rejected support is masked:

```text
1. read all candidate peaks and their accepted status
2. assign each sample to its nearest candidate peak center
3. replace assignments to rejected candidates with -1
4. construct ResolvedPeak objects only for accepted candidates
5. compute geometry from samples retained by each accepted peak
```

This ordering is required. If rejected candidates are removed before assignment,
every sample is absorbed by the nearest accepted peak and residual support is
erased.

Interpretation:

```text
assigned_support_fraction =
  number of samples assigned to accepted peaks / total samples

unassigned_support_fraction =
  number of samples assigned to rejected candidates / total samples
```

This does not yet mean geometric assignment confidence. It means accepted-peak
support captured by the current detector and assignment rule.

In V0, unresolved support arises only through candidate rejection.
Nearest-center assignment does not independently reject distant, low-density, or
intermediate observations. Therefore `unassigned_support_fraction` should not be
interpreted as all mass outside peak regions.

Later assignment rules may add maximum distance, density floors, watershed
membership, or assignment confidence. Those are not required for V0.

### Truth membership convention

Truth distributions do not need `sample_points` or `sample_peak_ids`, but they
do need `grid_peak_ids`.

For truth distributions, store one assignment per canonical grid cell:

```text
grid_peak_ids[y, x] = accepted candidate peak id
grid_peak_ids[y, x] = -1 for cells assigned to rejected candidates or otherwise unresolved
```

Truth peak metrics should be computed from canonical grid cells:

```text
1. read all detected truth candidates and their accepted status
2. assign canonical grid cells to all candidate peak centers
3. replace assignments to rejected candidates with -1
4. construct ResolvedPeak objects only for accepted candidates
5. weight cells by density * cell_area
6. compute support, radius, and radial CV from accepted cells
```

For V0, use the same nearest-center partition principle as empirical assignment,
but applied to grid cells and weighted by `F_composed`.

This makes the two sides parallel:

```text
empirical:
  samples -> nearest candidate peak center
  samples assigned to rejected candidates -> unresolved support

truth:
  density-weighted grid cells -> nearest candidate peak center
  cells assigned to rejected candidates -> unresolved support
```

### Within-peak metrics

Stored once for each resolved accepted peak:

```text
peak_id
center_coordinate
total_support_fraction
radius
cv_radius_from_center
```

Definitions:

```text
peak_id
  detector-local candidate peak id; stable only within one detection result

center_coordinate
  canonical coordinate of the resolved peak center, using the detector's local
  maximum coordinate for V0

total_support_fraction
  empirical: fraction of all samples assigned to this accepted peak
  truth: integrated density mass over grid cells assigned to this accepted peak

radius
  empirical: R80 of assigned sample distances from the peak center
  truth: density-weighted R80 of assigned grid-cell distances from the peak center

cv_radius_from_center
  std(distance_to_center) / mean(distance_to_center)
```

Edge-case conventions:

```text
0 assigned samples/cells: NaN
1 assigned sample: NaN for cv_radius_from_center
mean radius <= epsilon: NaN for cv_radius_from_center
```

Use population-style standard deviation (`ddof=0`) for V0 radial CV.

### Density is omitted from V0

Do not include `within_peak_density` in the V0 object.

The current code exposes several non-equivalent density notions:

```text
peak height
mean KDE density over assigned samples
mean density over a grid basin
sample count divided by basin area
KDE/truth mass divided by basin area
```

These are different quantities. Adding `within_peak_density` before choosing a
region/area definition would make the resolved peak object less clear.

If density is added later, define it explicitly, for example:

```text
within_peak_density =
  within_peak_total_support_fraction / within_peak_area
```

That requires a trustworthy peak-region area definition.

### Across-peak summaries

Across-peak summaries should be derived by a summary function, not stored on the
distribution object:

```python
def summarize_resolved_peak_distribution(
    distribution: ResolvedPeakDistribution,
) -> ResolvedPeakDistributionSummary:
    ...
```

V0 summary fields:

```text
number_of_peaks
assigned_support_fraction
unassigned_support_fraction

across_peak_total_support_fraction_mean
across_peak_total_support_fraction_skew

across_peak_radius_mean
across_peak_radius_skew

across_peak_cv_radius_from_center_mean

across_peak_distance_mean
```

The primary MVP-A null-tested metric set is narrower:

```text
number_of_peaks
assigned_support_fraction
across_peak_total_support_fraction_skew
across_peak_radius_mean
across_peak_radius_skew
across_peak_cv_radius_from_center_mean
across_peak_distance_mean
```

`unassigned_support_fraction` and
`across_peak_total_support_fraction_mean` should remain export conveniences, not
primary null-tested metrics.

`across_peak_total_support_fraction_mean` is algebraically constrained:

```text
across_peak_total_support_fraction_mean =
  assigned_support_fraction / number_of_peaks
```

when support is defined over accepted resolved peaks. It may still be useful as
a compact export convenience, but it is not independent information.

### Summary edge-case conventions

Use `NaN` for undefined summaries. Do not return zero for non-computable cases.

```text
0 peaks:
  all across-peak summaries NaN

1 peak:
  means defined when the underlying per-peak values are finite
  distance summaries NaN
  skew summaries NaN

2 peaks:
  means defined when finite
  across_peak_distance_mean = the one pairwise distance
  skew summaries NaN

3 or more peaks:
  means, pairwise distance mean, and skew summaries defined when finite
```

### Summary and metric registry

The summary object should be compact and scalar. It is the primary unit consumed
by empirical null testing.

```python
@dataclass(frozen=True)
class ResolvedPeakDistributionSummary:
    distribution_id: str
    source_type: Literal["truth", "empirical"]
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

Use metric metadata rather than a registry of bare lambdas. Structural validity
belongs with the metric definition.

```python
@dataclass(frozen=True)
class ResolvedPeakMetricDefinition:
    name: str
    minimum_peak_count: int
    default_alternative: Literal["two-sided", "greater", "less"] = "two-sided"
```

MVP-A metric registry:

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
    "across_peak_total_support_fraction_skew": ResolvedPeakMetricDefinition(
        name="across_peak_total_support_fraction_skew",
        minimum_peak_count=3,
    ),
    "across_peak_radius_mean": ResolvedPeakMetricDefinition(
        name="across_peak_radius_mean",
        minimum_peak_count=1,
    ),
    "across_peak_radius_skew": ResolvedPeakMetricDefinition(
        name="across_peak_radius_skew",
        minimum_peak_count=3,
    ),
    "across_peak_cv_radius_from_center_mean": ResolvedPeakMetricDefinition(
        name="across_peak_cv_radius_from_center_mean",
        minimum_peak_count=1,
    ),
    "across_peak_distance_mean": ResolvedPeakMetricDefinition(
        name="across_peak_distance_mean",
        minimum_peak_count=2,
    ),
}
```

Metric extraction should return `NaN` when structural requirements are not met.
For example, skew metrics require at least three resolved peaks.

### Run context

Centralize run metadata in one small object and flatten it only at export time:

```python
@dataclass(frozen=True)
class ResolvedPeakRunContext:
    analysis_id: str
    scenario_id: str
    replicate_id: str
    seed: int
    n: int
    bandwidth_rule: str
    bandwidth_multiplier: float
    bandwidth_value: float
    peak_detector_method: str
    canonical_grid_id: str
    assignment_rule: str
```

This avoids passing a long list of loose provenance columns through every
function.

### Empirical null primitive

Centralize p-value calculation and null diagnostics in a generic statistical
primitive, not a peak-specific framework.

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


def run_empirical_null_test(
    *,
    observed_value: float,
    null_values: np.ndarray,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    min_valid_null_fraction: float = 0.8,
) -> EmpiricalNullResult:
    ...
```

Validity rules:

```text
observed_value is finite
n_valid_null is sufficient
valid_null_fraction >= min_valid_null_fraction
```

When the test is invalid:

```text
empirical_p_value = NaN
standardized_effect = NaN
test_is_valid = False
```

The null result must report both `n_null` and `n_valid_null`. A metric whose null
value is `NaN` in most draws should not receive a plausible-looking p-value.

Recommended V0 statistic convention:

```text
observed_difference =
  observed_target_value - observed_reference_value

null_difference =
  null_target_value - null_reference_value
```

Use a plus-one empirical p-value to avoid zero p-values:

```text
greater:
  (1 + count(null_difference >= observed_difference)) / (1 + n_valid_null)

less:
  (1 + count(null_difference <= observed_difference)) / (1 + n_valid_null)

two-sided:
  (1 + count(abs(null_difference - null_center)
             >= abs(observed_difference - null_center))) / (1 + n_valid_null)
```

For V0, use `null_center = median(valid_null_difference)` for the two-sided
test. Report `standardized_effect` as:

```text
(observed_difference - null_mean) / null_std
```

Return `NaN` when `null_std` is zero or non-finite.

### Builder API

The core V0 API should stay small. Use pure functions, not a stateful `.fit(...)`
orchestrator.

Resolved distribution builders require `candidate_details` for every candidate
used in membership assignment. Aggregate `peak_locations` alone are insufficient
because they do not preserve accepted/rejected candidate identity.

A builder helper should validate detector details before assignment:

```python
def _validated_candidate_details(
    detection_result: PeakDetectionResult,
) -> tuple[PeakCandidateDetail, ...]:
    ...
```

Validation requirements:

```text
candidate ids are unique
candidate centers are finite
accepted status is present
coordinates are canonical rather than raw grid indices
accepted count matches detection_result.accepted_peak_count
```

MVP-A required:

```python
def resolve_empirical_peak_distribution(
    *,
    distribution_id: str,
    density_grid: DensityGrid,
    sample_points: np.ndarray,
    detection_result: PeakDetectionResult,
) -> ResolvedPeakDistribution:
    ...
```

Responsibilities:

```text
1. read all candidate peaks and their accepted status
2. assign samples to all candidate peak centers
3. map assignments to rejected candidates to -1
4. compute PeakGeometry for accepted candidates
5. retain DensityGrid and PeakDetectionResult
```

MVP-B, add after the empirical path:

```python
def resolve_truth_peak_distribution(
    *,
    distribution_id: str,
    density_grid: DensityGrid,
    detection_result: PeakDetectionResult,
) -> ResolvedPeakDistribution:
    ...
```

Responsibilities:

```text
1. read all truth candidates and their accepted status
2. assign canonical grid cells to all candidate peak centers
3. map cells assigned to rejected candidates to -1
4. weight cells by density * cell_area
5. compute truth PeakGeometry for accepted candidates
6. retain DensityGrid and PeakDetectionResult
```

Shared summary primitive:

```python
def summarize_resolved_peak_distribution(
    distribution: ResolvedPeakDistribution,
) -> ResolvedPeakDistributionSummary:
    ...
```

Keep resampling orchestration outside the core primitives. Analysis runners such
as `run_resolved_peak_bootstrap_comparison(...)` can remain analysis code rather
than core abstractions.

The first empirical comparison runner can have a narrow public shape:

```python
def run_resolved_peak_bootstrap_comparison(
    *,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    analysis_spec: ResolvedPeakAnalysisSpec,
    context: ResolvedPeakRunContext,
) -> pd.DataFrame:
    ...
```

This runner should return compact metric/null-test rows, not full resolved
objects for every null draw. Keep it in analysis code until more than one
workflow needs the same orchestration.

### Null workflow

For MVP-A, use one fixed analysis configuration:

```text
fixed primary bandwidth rule and multiplier
fixed detector method
fixed detector parameters
fixed assignment rule
```

For each null draw, rerun the KDE, peak detection, resolution, and summary under
that fixed configuration. Do not rerun a broad bandwidth sweep inside each null
draw. Alternate bandwidths and detectors are sensitivity analyses, not additional
null dimensions.

Lean empirical comparison workflow:

```text
1. select one primary bandwidth configuration
2. select one primary peak detector
3. build observed reference and target summaries
4. generate null resample index pairs
5. build one KDE per null reference and target draw
6. detect and resolve peaks
7. emit scalar summary values only
8. compute null differences
9. discard temporary resolved objects unless debugging is enabled
```

The null draw table, if emitted, should be one row per draw and metric. Ordinary
runs should not retain hundreds of full `ResolvedPeakDistribution` objects in
memory.

Do not centralize every resampling strategy in MVP-A. Start with the resampling
mode needed by the first empirical comparison runner, then generalize.

Centralize now:

```text
metric metadata
metric extraction from ResolvedPeakDistributionSummary
empirical p-value calculation
null summary statistics
valid-null accounting
run-context flattening for exports
```

Do not centralize yet:

```text
all resampling strategies
all bandwidth-selection workflows
all simulation and real-data null generation modes
distance-matrix caches
matched-peak comparison
```

### Storage versus export

The in-memory object keeps arrays and geometry context:

```text
density_grid
sample_points
sample_peak_ids
grid_peak_ids
detection_result
peaks
```

Compact exported tables should contain only derived values:

```text
resolved_peak_table
resolved_peak_distribution_summary_table
resolved_peak_null_test_table
```

MVP-A requires the summary and null-test tables. The per-peak table is useful for
inspection and debugging but is not the primary inference unit.

Do not serialize full density grids or sample arrays into the summary table.

#### `resolved_peak_table`

One row per accepted resolved peak. This is for inspection, diagnostics, and
plotting, not the primary V0 inference unit.

```text
analysis_id
distribution_id
source_type
scenario_id
condition_label
replicate_id
seed
n
bandwidth_rule
bandwidth_multiplier
bandwidth_value
peak_detector_method
canonical_grid_id
assignment_rule
candidate_peak_id
center_x
center_y
total_support_fraction
radius
cv_radius_from_center
```

#### `resolved_peak_distribution_summary_table`

One row per resolved distribution. This is the primary input to empirical null
testing.

```text
analysis_id
distribution_id
source_type
scenario_id
condition_label
replicate_id
seed
n
bandwidth_rule
bandwidth_multiplier
bandwidth_value
peak_detector_method
canonical_grid_id
assignment_rule
number_of_peaks
assigned_support_fraction
unassigned_support_fraction
across_peak_total_support_fraction_mean
across_peak_total_support_fraction_skew
across_peak_radius_mean
across_peak_radius_skew
across_peak_cv_radius_from_center_mean
across_peak_distance_mean
```

#### `resolved_peak_null_test_table`

One row per comparison and metric. This table should include validity diagnostics
so structurally missing metrics cannot masquerade as significant results.

```text
analysis_id
comparison_id
reference_group_id
target_group_id
metric_name
observed_reference_value
observed_target_value
observed_difference
null_mean
null_std
null_median
null_q025
null_q975
empirical_p_value
standardized_effect
n_null
n_valid_null
valid_null_fraction
test_is_valid
alternative
null_generation_method
bandwidth_rule
bandwidth_multiplier
peak_detector_method
n
```

Multiple-testing correction is optional for V0. If added, append
`empirical_q_value` rather than changing the empirical p-value semantics.

### Implementation order

```text
1. build empirical resolved distributions
2. build scalar summaries
3. add metric registry with structural validity requirements
4. generalize empirical-null reduction from existing code
5. add the first matched-N/bootstrap empirical comparison runner
6. export summary and null-test tables
7. add truth resolver for calibration
8. profile runtime
9. add distance caching only where profiling identifies pressure
```

Distance-matrix caching is a useful optimization because
`precompute_squared_distances(...)` and the isotropic KDE evaluator already exist,
but it is not a correctness requirement for the first implementation.

### Legacy compatibility

Do not change the existing detector APIs for V0:

```text
PeakCandidateDetail
PeakDetectionResult
peak_count_detail
count_mass_significant_modes
detect_peaks
```

Existing benchmark plots and detector audits should continue to consume
`PeakDetectionResult`. The resolved distribution layer should be inserted as an
additional builder layer for consumers that need canonical per-peak geometry.
