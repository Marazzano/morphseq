# Plan: Principled KDE bandwidth for `valley_depth` — diagnose, then adapt

## Context

`valley_depth` in `support_geometry.py` is a **density-level-set topology
statistic**: it builds a 2-D Gaussian KDE over a group's morph-axis point cloud,
sweeps a super-level density threshold down from the peak, and reports the fraction
of peak at which the super-level set first fractures into ≥2 mass-significant
connected components. Because the answer depends on the *topology* of the estimated
density, the KDE bandwidth is load-bearing.

Today `_kde_grid` (support_geometry.py:89-101) calls `gaussian_kde(pts.T)` with no
`bw_method`, so scipy uses Scott's rule — a **single global scalar** depending only
on n and covariance, blind to local cloud shape. This produces two opposite errors:
oversmoothing true gaps (`three_discrete`/`small_middle` under-fire even at
`VALLEY_SWEEP_STEPS=200`) and, if narrowed, manufacturing gaps in smooth curved
manifolds (`spiral`/`crescent`).

A prior experiment (documented in `STATUS_and_diagnosis.md` §6) hard-coded
`bw_method=0.5` and validated it through the real bootstrap-null machinery: it
recovered some missed multimodal cases **but collapsed the easy `two_discrete`
true-positive to the same ~12% significance rate as `spiral`/`crescent` false
positives.** Verdict: a single fixed scalar cannot satisfy both constraints. This is
the *same* global-vs-local bandwidth problem already SOLVED on the graph side —
`_knn_adjacency` (support_geometry.py:252-290) uses a per-point self-tuning
Zelnik-Manor bandwidth (σ_i = distance to the k-th nearest neighbor) with a comment
explaining exactly why local beat global. The KDE estimator is the one place that
never received that treatment.

**Intended outcome:** a KDE bandwidth strategy for `valley_depth` that resolves real
sparse multimodal structure *without* inventing gaps on curved manifolds — validated
the same way the rejected experiment was (full synthetic suite, bootstrap-null
p-values, checking BOTH false-positive controls AND the easy `two_discrete`
true-positive). The deeper goal is understanding the estimator's failure modes, not
just patching one case — so we measure bandwidth sensitivity *before* picking a fix.

This is deliberately staged: a **diagnostic phase** (cheap, reviewable, high
information-per-hour) lands and is inspected before any adaptive estimator is
written. The bandwidth-sensitivity heatmap tells us whether the kNN-adaptive family
is even the right answer before we invest in it.

## Approach

### Phase 1 — Make bandwidth explicit and pluggable (baseline-preserving)

Refactor `_kde_grid` so the estimator/bandwidth is a first-class parameter, with the
**current Scott behavior as the exact default** (no behavior change when unspecified).

- `results/mcolon/20260617_morph_axis_investigation/support_geometry.py`
  - Change `_kde_grid(points_2d, grid_size=GRID_SIZE)` →
    `_kde_grid(points_2d, grid_size=GRID_SIZE, *, kde=None)` where `kde` is an
    optional density-estimator spec. Default `None` reproduces
    `gaussian_kde(pts.T)` (Scott) **byte-for-byte** — this is the regression guard.
  - Introduce a small estimator abstraction (a dataclass or a
    `Callable[[pts], Callable[[grid_points], density]]`) with two initial
    implementations:
    `scipy_gaussian` (current, params: `bw_method`, `bw_scale`) and a stub slot for
    `knn_adaptive` (Phase 3). Keep it minimal — a spec object + a factory, NOT a
    class hierarchy.
  - Thread the spec through `valley_depth` and `valley_detection_detail` (both call
    `_kde_grid`) as an optional keyword, defaulting to `None`. **Both must stay in
    lockstep** — `valley_detection_detail` is the visualization's source of truth and
    must render whatever bandwidth the statistic used.
  - The `statistics` registry (`SUPPORT_STATISTICS`) and `compute_support_geometry`
    pass the spec through unchanged when default; add an optional
    `kde=` passthrough so the harness can inject a candidate for the whole bundle.

Guard: `git diff` shows no numeric change in existing outputs when `kde=None`.
Verify by re-running one real anchor (`run_real_anchors.py`) and diffing the
`valley_depth` column of `tables/phenotype_geometry_summary.csv` against HEAD.

**Watch out — the visualization has its OWN KDE copies that will silently diverge.**
`gaussian_kde` is constructed in several places besides `_kde_grid`, none of which
share bandwidth today:
- `valley_visualization.py:132,141,153,158` — two viz helpers build their own KDEs.
- `synthetic_valley_connectedness_grid.py` — `_kde_grid_padded` (:102-111) builds a
  *separate* padded KDE AND `_count_modes` (:114-126) re-runs the valley sweep at
  only **25 steps** (vs the statistic's 200) purely to place the significance ring.
- `distribution_shift.py:30,77,78` — 1-D Stage-1 KDEs (out of scope; leave alone).

When Phase 3 changes the estimator, these viz copies must be pointed at the same
spec (or the ring/contour will show a Scott-bandwidth density under a statistic
computed at a different bandwidth). Phase 1 should route the two valley-viz helpers
and `_kde_grid_padded` through the same estimator spec so they can't drift. This is
the same "single source of truth" invariant that motivated the shared
`VALLEY_SWEEP_STEPS` constant.

### Phase 2 — Bandwidth-sensitivity diagnostic (SHIP + REVIEW before Phase 3)

The single most informative artifact. Add a script that runs the **real p-value
machinery** (`compute_support_geometry`, not a reimplementation) across a grid of
scalar bandwidth multipliers for every synthetic scenario.

- New: `results/mcolon/20260617_morph_axis_investigation/bandwidth_sensitivity.py`
  - Bandwidth multipliers, e.g. `[0.35, 0.5, 0.7, 1.0, 1.4, 2.0]` (Scott = 1.0).
  - For each `(scenario, seed, bandwidth_scale)`: draw the cloud from
    `synthetic_scenarios.SCENARIOS`, run `compute_support_geometry` with the
    `scipy_gaussian` spec at that scale against the `wt_reference` null, record
    `valley_depth`, `pvalue`, `n_components_at_split`, component mass fractions.
  - Reuse `synthetic_scenarios.SCENARIOS` / `wt_reference` / `SAMPLE_SIZES`; reuse
    `compute_support_geometry`. Do NOT duplicate the null logic.
  - Emit `tables/bandwidth_sensitivity.csv` and a **scenario × bandwidth
    significance heatmap** (`plots/bandwidth_sensitivity.png`).
  - Read this as a *persistence-across-smoothing-scale* view: a real split should be
    significant across a broad bandwidth band; a sampling-noise split (spiral/
    crescent) should be significant only in a narrow too-narrow window. If
    `two_discrete` and `spiral` do NOT separate at any single scalar, that is the
    finding that justifies moving to local adaptivity — and if they DO cleanly
    separate at some scale, that reframes the whole problem.

**Decision gate (HARD STOP — confirmed):** after Phase 2, STOP and present the
heatmap + reading. Do not begin Phase 3 (the kNN-adaptive estimator) until the user
has looked at the bandwidth-sensitivity result and confirmed the family is right.
This gate exists because if `two_discrete` and `spiral` separate cleanly at some
scalar bandwidth, the whole approach changes — measure before building.

### Phase 3 — kNN sample-point adaptive KDE (first serious candidate)

Mirror the already-validated graph-side `_knn_adjacency` local-scaling directly onto
the KDE, as a sample-point (not balloon) estimator so the valley's own bandwidth
never becomes part of the question.

- `support_geometry.py`: implement the `knn_adaptive` estimator behind the Phase-1
  spec interface.
  - Per-point bandwidth `h_i = bw_scale * r_k(x_i)`, where `r_k` is the distance to
    point i's k-th nearest neighbor (reuse the same `pdist`/`squareform` +
    `np.sort` pattern already in `_knn_adjacency`).
  - **Clip** `h_i` to a quantile-based band (e.g. `[q10*min_factor, q90*max_factor]`)
    — load-bearing at small n, not polish: unclipped, a sparse bridge point gets a
    huge kernel that smears mass across the gap (the exact failure we're fixing).
  - Density at a grid point = `mean_i (1/h_i^2) K((x - x_i)/h_i)` with a Gaussian K.
  - Starting params `k≈10–15`, `min_factor≈0.5`, `max_factor≈2–3` — **tuned against
    the synthetic suite, never against real b9d2/cep290 first.**

### Phase 4 — Validate against the full gate (mandatory, same bar as the rejection)

Any candidate must clear the exact bar the `bw=0.5` experiment failed.

- Re-run `validate_framework.py` (the synthetic gate) with the candidate estimator.
- Re-run `synthetic_valley_connectedness_grid.py` — the one-figure all-12-scenario
  view — as the fast visual first-pass.
- Explicitly check BOTH:
  - false-positive controls stay non-significant: `spiral`, `crescent`, `outliers`,
    `unimodal_compact`, `variance_only`.
  - the easy true-positive `two_discrete` recovers to high significance (the case
    `bw=0.5` broke).
  - the originally-broken `three_discrete` / `small_middle` improve.
- Only if all three hold is the candidate adopted as the new default. If it doesn't,
  the bandwidth-sensitivity heatmap + this gate output tell us *why*, which is itself
  the deliverable.

### Deferred (named benchmarks, NOT first-round implementation)

- **Abramson pilot-density** — deferred *hard*: it widens kernels in low-density
  regions, i.e. fills the exact valleys the statistic detects. Structurally wrong for
  a valley statistic, not merely "cautious." Only revisit if kNN-adaptive fails.
- **Botev / diffusion KDE** — a MISE-optimal *global* selector; our failure is
  topological, not MISE. Keep as a harness benchmark column, not a first fix.
- **Multiscale persistence statistic** (`valley_depth` over bandwidth × threshold as
  a 2-D stability score) — the eventual robust upgrade if single-bandwidth adaptivity
  still proves fragile. Phase 2's heatmap is the prototype of its bandwidth axis.

## Critical files

- `results/mcolon/20260617_morph_axis_investigation/support_geometry.py` — `_kde_grid`
  (:89), `valley_depth` (:134), `valley_detection_detail` (:168),
  `compute_support_geometry` (:511); reuse `_knn_adjacency` (:252) local-scaling
  pattern.
- `results/mcolon/20260617_morph_axis_investigation/synthetic_scenarios.py` —
  `SCENARIOS`, `wt_reference`, `SAMPLE_SIZES` (reuse as-is).
- `results/mcolon/20260617_morph_axis_investigation/validate_framework.py` — synthetic
  gate (Phase 4).
- `results/mcolon/20260617_morph_axis_investigation/synthetic_valley_connectedness_grid.py`
  — visual first-pass (Phase 4).
- NEW: `bandwidth_sensitivity.py` (Phase 2).

## Verification

- **Phase 1 regression:** `run_real_anchors.py`, diff the `valley_depth` column of
  `tables/phenotype_geometry_summary.csv` vs HEAD — must be identical (`kde=None`
  default is byte-for-byte Scott).
- **Phase 2:** inspect `plots/bandwidth_sensitivity.png` — confirm expected pattern
  (real splits persist across bandwidth; curved manifolds significant only in narrow
  windows). This gates Phase 3.
- **Phase 4:** `validate_framework.py` passes; `synthetic_valley_connectedness_grid.py`
  shows `two_discrete` recovered, `spiral`/`crescent`/`outliers` still continuous,
  `three_discrete`/`small_middle` improved.
- Run everything with
  `conda run -n segmentation_grounded_sam --no-capture-output python <script>`.

## Notes on the merge

The `valley-viz-one-box-framing` merge is committed locally on `main` (8 commits
ahead of `origin/main`) but **not pushed** — keeping it local until the bandwidth
work has a clean story, per the "make it pluggable and diagnosed before pushing"
reasoning. Push is a separate, explicit decision.
