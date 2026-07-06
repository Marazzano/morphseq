# Modal Organization Bandwidth Tuning Plan

## Purpose

Calibrate KDE bandwidth rules against the modal V0 synthetic anchors by
comparing sample-estimated geometry to `F_composed` truth.

The calibration target is not peak count alone. The target is preservation of:

- global valley geometry
- bridge-pair valley geometry
- bridge-region density
- bridge-ladder ordering

## Truth Reference

For each V0 anchor distribution, measure the truth quantities directly from
`F_composed` before sampling.

Keep these layers separate:

- `global_valley_density_ratio`
- `bridge_pair_valley_density_ratio`
- `bridge_region_density_ratio`
- `bridge_region_mass_fraction`

In V0, the pairwise valley measurement is taken from the first two mode
components in the recipe. That is the bridge-ladder anchor pair for the current
two-peak cases.

Peak count remains a secondary sanity check.

## Candidate Bandwidth Families

Sweep geometry-derived bandwidth rules on the sampled point cloud:

- `median_kNN_distance`
- `q90_kNN_distance`
- `median_MST_edge_length`
- `q90_MST_edge_length`
- `longest_non_outlier_MST_edge`
- `connectivity_90_radius`
- `global_R50`
- `global_R80`

Use a small multiplier ladder for each rule:

- `0.75`
- `1.00`
- `1.25`
- `1.50`

Treat `global_R50`, `global_R80`, and `connectivity_90_radius` as comparator
rules if they over-smooth multimodal cases.

## Benchmark Anchors

Run the sweep on:

- `one_peak_compact`
- `one_peak_diffuse`
- `one_peak_spiral`
- `two_peaks_no_bridge`
- `two_peaks_low_bridge`
- `two_peaks_high_bridge`
- `three_peaks_compact`

Use `three_peaks_compact` as the first sanity check for peak-count recovery, but
keep the bandwidth choice anchored on geometry preservation.

## Scoring

Primary scoring should rank candidates by:

1. bridge-pair valley ratio error
2. bridge-pair valley depth error
3. global valley ratio error
4. bridge ordering recovery
5. valley ordering recovery
6. peak-count sanity
7. false split / false merge rates

The bridge ladder should remain monotone across the no-bridge / low-bridge /
high-bridge cases.

## Outputs

Write the calibration results to a dedicated table set:

- truth table
- candidate table
- rule summary table
- selected-rule summary
- markdown report

These outputs should make it straightforward to rerun the V0 benchmark after any
future bandwidth or peak-calling change.
