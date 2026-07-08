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

- Should the basin-support detector stay with nearest-peak assignment for V0, or
  should it move to watershed / gradient-ascent basins before we trust it as a
  primary diagnostic?
- Do we want peak-count confidence thresholds to vary by sample size, or keep one
  fixed rule for the whole V0 benchmark?
- Should HDR persistence remain a diagnostic-only route, or should it be allowed
  to vote on the final peak count once the audit table stabilizes?
- We should explicitly compare `hdr_component_persistence` versus
  `kde_peak_basins_sample_support` for robustness on anisotropic / elongated
  cases.
- If basin counts a peak but does not report a stable split value, that is a
  detector-semantics mismatch, not automatically a basin failure: basin is
  sample-support based, while split still comes from the superlevel geometry
  sweep.
- Multi-basin cases with no credible split should be flagged as `ambig` in the
  audit output until we decide whether they count as one broad mode or two weak
  modes.

## Visualization

- Should the shared V0 benchmark plots include a dedicated power-summary panel
  for each metric family?
- Do we want a compact per-distribution failure note in the plot footer when a
  metric remains sample-sensitive?

## Bandwidth Tuning

- The immediate calibration target is geometry-derived KDE bandwidth tuning
  against the V0 anchors.
- Local geometry-derived rules are currently the right regime; the main question
  is how to tighten peak-count recovery without giving up valley preservation.
- `three_peaks_compact` is the most important failure case right now: the current
  local rules preserve the bridge ladder well, but they still merge the compact
  three-peak truth down to two peaks.
- Keep global valley geometry, bridge-pair valley geometry, and bridge-region
  density separate when comparing KDE estimates to `F_composed` truth.
- The next executable step is to refine the KDE/local-scale side of the peak
  detector so that truth-vs-detected split values move closer together on
  compact multi-peak cases, not just on the bridge ladder.
- The routed detector audit now makes it possible to tell whether a failure is
  coming from cap mass, HDR persistence, or empirical basin support.
