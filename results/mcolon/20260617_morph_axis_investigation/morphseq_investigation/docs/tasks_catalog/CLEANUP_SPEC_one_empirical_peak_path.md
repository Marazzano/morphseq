# CLEANUP SPEC — one empirical peak path; sample sets carry their own marginal

Status: SPEC ONLY (not implemented). Author handoff for a future session.

## The doctrine (this is the invariant everything else defers to)

**Empirical 2-D peak finding is THE method, forever.** It is the tuned,
vote-consensus finder (`core.compute_resolved_peaks` with the bootstrap vote:
`n_bootstrap_draws=80`, `count_stability_policy(min_mode_frequency=0.80)`). It is
run **once per distribution, in 2-D** (both features jointly). Its output is a set
of **SampleSets** — each discovered mode is simply *a label group = a set of
sample IDs*. Nothing else discovers peaks. There is no 1-D peak finder today.

**Plotting never re-discovers modes.** A ridge/marginal curve is only ever the
*projection* of an already-discovered SampleSet onto one feature axis. Quoting the
pre-refactor `v0/rich_distribution_plot.py` (the reviewed oracle):

> Modes are found in 2-D (both features jointly, in `run_bin`); here we only
> PROJECT each discovered group onto one feature axis and render its marginal.
> **We never re-discover modes in 1-D.**

## The bug this spec removes: TWO peak paths

Right now there are two readouts of "how many modes," and they disagree on the
real b9d2 data:

| bin   | valley viz (correct) | catalog `detect_peaks` (wrong) |
|-------|----------------------|--------------------------------|
| 14hpf | 1 (96% vote)         | 2                              |
| 18hpf | 1 (99%)              | 1                              |
| 24/26 | 2 (69%)              | 2                              |
| 30hpf | 1 (60%)              | 4                              |
| 48/50 | 2 (79%)              | 3                              |

### Why they differ (verified in code)

Both paths call the SAME `compute_resolved_peaks` with the SAME default
`PeakResolutionConfig` (the vote runs in both). The divergence is NOT the config
and NOT the bandwidth. It is two things:

1. **Different readout field.**
   - `core.compute_resolved_peaks` runs the vote, sets
     `target_peak_count = count_stability.mode_peak_count` (the vote-consensus
     count), then seeds+carves exactly that many peaks. `resolved_peak_count` is
     the vote-gated field; `resolved.peaks` is the carved set.
   - Valley viz reads `dist.resolved_peak_count` — the vote-gated count
     (`valley_visualization.py:250-251`).
   - Catalog `detect_peaks` reads `len(resolved.peaks)` and makes one SampleSet
     per carved peak (`engine/labelers.py:312`, `resolved_peak_count = len(peaks)`
     recorded as "DERIVED = len(sample_set_ids)"). It also does NOT propagate
     `is_reliable` into whether/how many SampleSets exist.

2. **Different inputs to the finder.** Valley viz calls the finder on a
   `derive_shared_grid(grp, wt)` over **normalized** shapes (`normalize_shape`),
   pooling target+WT for the grid. Catalog `detect_peaks` builds its grid from a
   single distribution's own `feature_column` values (grid_method
   `pooled_min_max`, resolution 61) — different points, different grid, therefore
   a different vote and a different carve. So even the "same field" would not
   match until the inputs are unified.

Net: the catalog path is effectively a second, differently-fed detection that
over-splits (4 modes at 30hpf is not biology; it is an un-consensused carve on a
different grid).

## The fix (one path)

**Goal: `catalog.detect_peaks` and valley viz must be the SAME empirical finder,
same inputs, same vote-gated readout — producing identical SampleSets.**

1. **Single finder entry point.** Route both callers through one function that
   takes points + the analysis spec + the (shared) grid and returns the
   vote-resolved `ResolvedPeakDistribution`. No caller re-implements grid
   construction or spec defaults.

2. **Vote-gated SampleSets.** The number of SampleSets `detect_peaks` emits MUST
   equal the vote-consensus count (`resolved_peak_count` / seeded carve), not the
   raw detector `len(peaks)`. When `is_reliable is False` / `resolved_peak_count
   is None` (no usable vote), emit the honest degenerate result (a single
   "unresolved" SampleSet or zero, per the plan's crisp-failure shape) — never a
   speculative multi-peak carve.

3. **Unify the grid/normalization.** Decide ONE grid contract for detection:
   the pooled target+WT `derive_shared_grid` on normalized shapes that valley viz
   uses is the reviewed one. `detect_peaks` should build detection input the same
   way (or call the same helper). If the catalog's per-distribution grid is kept
   for some reason, it must be proven to reproduce the valley-viz vote counts on
   b9d2/cep290 before it is trusted.

4. **Delete the second path.** After 1-3, there is exactly one place modes are
   found. `resolved_peak_count` (vote) is the count everywhere; `len(peaks)` is
   never used as "the number of modes." Add a regression test: the b9d2 per-bin
   SampleSet counts from `catalog.detect_peaks` equal the valley-viz
   vote-consensus counts (14→1, 18→1, 24→2, 30→1, 48→2 under MST-edge @ 0.75).

## Cleanliness item: each SampleSet carries its own marginal (trivial)

This is how the pre-refactor script behaved and should be restored as a first-
class property, not recomputed ad hoc by every plotter.

- A SampleSet is a set of sample IDs. Given a feature axis and a shared 1-D grid,
  its marginal is **trivially** the 1-D KDE of that sample set's own values on
  that axis (`v0/rich_distribution_plot.py::_build_cell` did exactly this:
  `evaluate_density(grid, vals, bandwidth=_silverman_bandwidth(vals))` per group).
- Make this a method/property on the SampleSet (or a tiny helper keyed by
  sample_set_id): `sample_set.marginal(feature, grid) -> DensityGrid`. Then the
  ridge verb and the density grid both just ASK each SampleSet for its marginal
  and render it — no plotter re-derives which points belong to which mode, and no
  plotter re-runs detection. This is the "each sample set can carry its own
  marginal for a given sample_set_id; it is trivial to compute" cleanliness the
  owner called out.
- Invariant preserved: the marginal is a *projection* of an
  already-discovered set. It is never a re-discovery. The 1-D KDE here is for
  drawing a smooth curve only; it has no say in mode count.

## Future work (explicitly OUT of scope for this cleanup)

Wiring the empirical finder to operate on a genuinely 1-D case. Even then it is
the SAME empirical finder on different samples / label groups — NOT a new method.
Do not build a 1-D peak finder as part of this cleanup; this spec only collapses
the current two 2-D readouts into one and makes marginals a SampleSet property.

## Acceptance for the cleanup

- `catalog.detect_peaks` b9d2 per-bin SampleSet counts == valley-viz vote counts.
- Exactly one code path constructs detection input and reads the vote count.
- `SampleSet.marginal(feature, grid)` exists; ridge + density-grid verbs consume
  it instead of re-deriving membership or re-detecting.
- Valley viz + ridge remain BOTH emitted for b9d2 (peak quality is judged on the
  valley viz; the ridge projects the same SampleSets).
