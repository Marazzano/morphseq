# Stage 2a Notes

Date: 2026-07-08

This note captures the current Stage 2a state for
`COMPOSE_single_path_plan.md` Part 4, centered on
`ResolutionStrategy.MODE_VOTE_FULL_DATA` + `BootstrapRetention.SUMMARY_ONLY`.

## Current Spec Notes

- The bootstrap vote is retained as a full histogram-like object, not collapsed to a boolean.
- `compute_resolved_peaks(...)` now does the whole Stage 2a flow:
  - bootstrap vote via `resample.subsample`
  - mode count selection from the vote
  - consensus seed construction from transient per-draw candidate centers
  - one honest full-data density carve
  - final basin validation via `PeakAcceptancePolicy.validate_resolved_basins`
- The full-data KDE is not re-fit to force the voted count.
- Failure remains crisp:
  - if the final basins fail validation, foreground resolution is rejected
  - `resolved_peak_count` stays `None`
  - `peaks` is empty
  - sample assignments are `-1`
- `is_reliable` is the conjunction of vote stability and final basin validity.
- Stage 2a only implements `MODE_VOTE_FULL_DATA` + `SUMMARY_ONLY`.
- The null/permutation path still uses the single-pass resolve path and does not call `compute_resolved_peaks(...)`.
- The resampling engine is now the shared source of draws for both the vote and the null path, so the seed behavior is a clean re-baseline rather than a hand-rolled loop match.

## Current Code State

Implemented in the current working tree:

- `morphseq_investigation/core/peak_stability.py`
- `morphseq_investigation/core/_resample_adapters.py`
- `morphseq_investigation/core/resolved_peak_metrics.py`
- `morphseq_investigation/core/distribution_records.py`
- `valley_visualization.py`

The visualization script now reads:

- `resolved_peak_count`
- `is_reliable`
- `resolution_evidence.count_stability.mode_frequency`

instead of recomputing a local resampled mode count.

## Verification Status

What has been checked so far:

- `python` resolves to the active `segmentation_grounded_sam` environment.
- `git diff --check` is clean for the Stage 2a files touched in this pass.
- The null/permutation helper scales linearly in draw count on the smoke probe that was run:
  - 10 draws: 2.661 s
  - 20 draws: 4.797 s
  - 40 draws: 8.908 s

What was started but not fully finished when this note was written:

- full `valley_visualization.py` end-to-end execution on the real b9d2/cep290 CSVs
- the focused pytest run for the new Stage 2a files

## Deferred Improvements

These are reasonable follow-ups, not blockers for Stage 2a:

- replace the greedy consensus-seed construction with a more explicit, documented clustering rule if the current heuristic ever becomes hard to defend scientifically
- add Stage 2b / 2c retention products (`BootstrapPeakCandidate`, per-draw sample assignments) when that evidence becomes useful downstream
- revisit the `resolved_peak_id` numbering contract if a later consumer needs a strict 1-based convention
- benchmark the real-data visualization path and trim any avoidable overhead in the bootstrapped vote if the current runtime is too high for routine use
- add a dedicated regression note for the seed-engine clean break so future re-baselines are explicit

## Working Tree Note

The Stage 2a changes are still in the working tree and were intentionally left uncommitted until the current pass is complete.
