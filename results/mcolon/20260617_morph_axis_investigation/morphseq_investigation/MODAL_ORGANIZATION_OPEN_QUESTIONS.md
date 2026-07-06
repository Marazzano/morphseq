# Open Questions

This file tracks simulation questions that are still under active calibration.

## Metric Power

- What sample size is needed for valley-based bridge ordering to stabilize across
  the V0 bridge ladder?
- At what `n` does `mst_max_edge` become a reliable bridge separator rather than a
  broad corroborating signal?
- Should `fiedler` remain a corroborating metric only, or does it become useful
  for any sub-region of the bridge ladder at larger sample sizes?
/
## Bridge Ladder

- Do the current low/high bridge V0 recipes remain in the intended density bands
  after further tuning?
- Should the bridge mask stay capsule-like, or do we want a narrower path mask
  once we begin validating saddle measurements more directly?
- Do we want a separate `fully_merged` V0 case, or is that better left to a later
  ladder once the bridge validation is fully stable?

## Peak Counting

- Should we add alternative peak-count conventions later, such as HDR-based peak
  counting or basin-based peak counting, for comparison against the empirical
  valley-sweep helper?
- Do we want peak-count confidence thresholds to vary by sample size, or keep one
  fixed rule for the whole V0 benchmark?

## Visualization

- Should the shared V0 benchmark plots include a dedicated power-summary panel
  for each metric family?
- Do we want a compact per-distribution failure note in the plot footer when a
  metric remains sample-sensitive?

## Bandwidth Tuning

- The immediate calibration target is geometry-derived KDE bandwidth tuning
  against the V0 anchors.
- Use `three_peaks_compact` as a sanity anchor for peak-count recovery, but do
  not choose the rule on peak count alone.
- Keep global valley geometry, bridge-pair valley geometry, and bridge-region
  density separate when comparing KDE estimates to `F_composed` truth.
- The next executable step is the bandwidth-tuning sweep described in
  `MODAL_ORGANIZATION_BANDWIDTH_TUNING_PLAN.md`.
