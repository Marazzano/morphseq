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

- Use one empirical peak-count helper for the V0 visualizations and benchmark
  scripts.
- The helper sweeps a super-level threshold downward and returns the first split
  into mass-significant components.
- For one-peak cases, valley depth is rendered as `N/A` in the metric row.

## V0 Benchmark Scope

- Keep V0 focused on the compact/diffuse/elongated/spiral/bridge ladders.
- Use the benchmark to identify the sample-size range where a metric becomes
  statistically useful.
- Do not treat weak bridge-order instability as a failure of the simulation
  itself if the composed truth passes validation.

