# Resolved Questions

This file records decisions that are now fixed in the V0 simulation layer.

## Density Naming

- Use `F_composed` for the post-composition density landscape.
- Keep `true_grid` only as a compatibility alias where older code still expects
  it.

## Observation Sampling

- Sample observations from the composed density grid by default.
- Keep component-based sampling only as a diagnostic mode.

## Bridge Ontology

- Bridges connect exactly two mode components.
- Bridges connect boundary regions, not mode centers.
- Bridge validity is judged on the composed density, not just on recipe mass
  fractions.

## Truth Validation

- Validate the composed density before benchmarking metrics.
- Check total mass, resolved peak count, and bridge-region ratios at the truth
  level.
- Treat sample-level metric orderings as calibration results, not as truth
  validation failures.

## Peak Counting

- Keep the superlevel-cap sweep as the compatibility detector, but route peak
  counting through method-aware detectors so the same density can be audited
  multiple ways.
- The routed benchmark now reports `superlevel_cap_mass`,
  `hdr_component_persistence`, and `kde_peak_basins_sample_support` on the same
  KDE grid.
- The superlevel-cap detector still sweeps a super-level threshold downward and
  returns the first split into mass-significant components.
- For one-peak cases, valley depth is rendered as `N/A` in the metric row.

## V0 Benchmark Scope

- Keep V0 focused on the compact/diffuse/elongated/spiral/bridge ladders.
- Use the benchmark to identify the sample-size range where a metric becomes
  statistically useful.
- Do not treat weak bridge-order instability as a failure of the simulation
  itself if the composed truth passes validation.

## Bandwidth Calibration

- Prefer the local geometry-derived bandwidth regime for V0 calibration.
- `longest_non_outlier_MST_edge` is the current leading rule family, with
  `median_kNN_distance` as the conservative fallback when a simpler local scale
  is desired.
- The global radius rules are not the right default for this benchmark; they
  tend to over-smooth the modal structure and degrade valley preservation.
- The remaining weakness is peak-count robustness, not valley ordering.
- In the failing cases, the composed truth splits more clearly than the KDE
  estimate, and the detected peaks are typically closer together than the truth.
  That is the current evidence that KDE smoothing is suppressing the split.
- The next tuning pass should therefore minimize truth-vs-detected peak split
  error and related valley-difference errors while staying in the local regime.
- The detector audit now separates KDE geometry from peak-support semantics so
  the peak-count failure can be attributed to cap mass, HDR persistence, or
  sample support instead of being treated as one undifferentiated KDE issue.

## Settled Recommendations

- Use `longest_non_outlier_MST_edge` and `median_kNN_distance` as the primary
  bandwidth rules for V0.
- Treat `kde_peak_basins_sample_support` as the primary peak-count method, with
  `hdr_component_persistence` and `superlevel_cap_mass` reported alongside it.
- Use the detector trio to explain peak-region and valley-region statistics
  across distributions, not to resweep the detector choice itself.
- Why:
  - `longest_non_outlier_MST_edge` best tracks the local separation scale
    without over-smoothing the compact multi-peak cases.
  - `median_kNN_distance` is the safer conservative fallback when the local MST
    edge is noisy or too aggressive.
  - `kde_peak_basins_sample_support` answers the biological question most
    directly: how many observed points belong to each KDE-defined basin.
  - `hdr_component_persistence` checks whether a lobe survives across density
    thresholds, so it is a good robustness diagnostic.
  - `superlevel_cap_mass` is the legacy cap-mass baseline; it is useful for
    explaining failures, but too brittle to be the sole decision rule.
