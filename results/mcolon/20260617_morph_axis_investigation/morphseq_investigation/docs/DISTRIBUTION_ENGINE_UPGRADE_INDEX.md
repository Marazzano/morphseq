# Distribution Comparison & Plotting Engine — Upgrade Index

Parent index for the "distribution comparisons + distribution plotting" engine
upgrade. The **why** lives in `DISTRIBUTION_COMPARISON_MOTIVATION.md` (formerly
`MODAL_ORGANIZATION_BENCHMARK_PLAN.md`). This index maps the vague upgrade
specs onto the existing primitives in `core/` and `plotting/`, then splits the
work into independently-shippable sub-specs.

Implementation briefs for the current parallel plan live in `../tasks/`
(`TASK_0_foundation.md`, `TASK_A_grid.md`, `TASK_B_labelers.md`,
`TASK_C_compare.md`, `TASK_D_plotting.md`, `TASK_E_b9d2_example.md`).

## What we already have (grounding)

- `core/distribution_records.py`
  - `CanonicalGrid` (via `density_composition`) — **2-D only**
    (`x_min/x_max/y_min/y_max/grid_size`), stores **canonical coords only**.
  - `DistributionRecord` = points + `DistributionAnalysisContext(grid, spec)`
    + `resolved_peaks | None` + `peak_stats | None` + `metadata`.
  - `DistributionComparison` = `{role: DistributionRecord}` members sharing ONE
    grid + `observed_metrics` tables.
  - `compute_resolved_peaks` → per-peak `PeakGeometry(peak_id, center, ...)`,
    `sample_peak_ids` (positional), basin rasters.
  - `compute_observed_metrics` → per-metric `reference / target / difference`
    rows. **No peak-to-peak matching** — compares scalar summaries only.
- `core/resolved_peak_metrics.py` — `ResolvedPeak`, `PeakGeometry` (has
  `peak_id`, `center_coordinate`, `radius`, `cv_radius_from_center`), summary.
- `plotting/modal_distribution_plotting.py` — `DistributionOverlay`,
  `build_distribution_overlay`, `plot_kde_field`, `plot_hdr_contour`,
  `plot_resolved_peak_overlay`, `plot_v0_distribution_qc_grid`.

## What the upgrade asks for (mapped)

| Ask (user) | Gap today | Sub-spec |
|---|---|---|
| Peak matching (target→closest null peak, or→largest WT peak) | no matcher; metrics are scalar-only | **A** |
| Variance in canonical coords BUT map back to real units | no real-coord storage, no transform | **B** |
| Track `sample_id` + `peak_id` coordinates through the pipeline | only positional `sample_peak_ids` | **B** |
| N-dimensional (3-D+) distribution objects | `CanonicalGrid` is 2-D | **C** |
| Overlay two Distribution objects fast | overlay exists 2-D only, no matcher hook | **A**/**D** |
| 1-D KDE strips (feature vs WT) | none | **D** |
| N features × M (target×ref) grid, via the faceting engine | no adapter to the shared engine | **D** |

> **Faceting note:** there is *no* `plot_distribution_feature_over_time`. The
> real, well-integrated faceting engine is
> `src/analyze/viz/plotting/faceting_engine/` (IR + matplotlib/plotly
> renderers, entry `render()`), already backing
> `plotting/feature_over_time.py` (`src/analyze/viz/README.md`). Distribution
> overlays/strips are emitted as that engine's IR and rendered through it —
> **not** a bespoke grid. Strips get a wider aspect ratio. See sub-spec D.

## The locked primitive (read this first)

The core primitive is defined in **`PRIMITIVE_ONTOLOGY.md`** (LOCKED). One line:

> A **Distribution** is an unlabeled bag of samples at one timepoint. Any number
> of **labelers** (genotype, peak-finding, DTW) partition it into **SampleSets**
> — the durable atoms carrying membership, shape, and provenance. **`label_groups`**
> is a lightweight dict indexing those SampleSets by view name (`genotype`,
> `peak_bwA`, `dtw`). A peak is not a special type — it's a SampleSet whose
> provenance says it came from mode-finding.

Three nouns, one durable object (`SampleSet`). Peers, not a hierarchy:
`label(distribution_id, method, params) → (label_group_name, [SampleSet])`.
`sample_id` identity (Sub-spec B) is the load-bearing contract. Binning (reduce
over `embryo_id` within the timepoint, reusing `bin_embryos_by_time`) is a
compare-time policy applied symmetrically — never baked into the object.

**Next conversation:** fit the existing peak machinery
(`compute_resolved_peaks` / bootstrap vote) into this as *the `peak_finding`
labeler*. Comparison + plotting layers are stubbed TBD in the ontology doc.

## Sub-specs (ordered by dependency)

> Note: A/C/D below were drafted against the *earlier* `PeakMembership` /
> `diff_vs_null` framing and use stale names. The locked primitive above
> supersedes that vocabulary; the sub-specs will be re-aligned to
> Distribution/SampleSet/label_groups once peak-fitting is settled. B (identity
> spine) is unaffected and still correct.

1. **`SUBSPEC_A_peak_matching.md`** — a `PeakMatcher` producing a
   `PeakMatchResult` (target peak_id ↔ reference peak_id + distance +
   match policy: `closest_null_peak` / `largest_reference_peak`). Feeds signed
   per-peak effect metrics. Pure addition on top of `compute_observed_metrics`.
2. **`SUBSPEC_B_coordinate_frames_and_identity.md`** — dual-coordinate
   `CanonicalGrid` (canonical for calc/plot + affine back to real units) and a
   stable `sample_id` spine so peak variance maps to real units. **Foundational
   — A and D consume its outputs.**
3. **`SUBSPEC_C_analysis_paths_and_membership.md`** — the pivot above:
   `PeakMembership` (`sample_id → peak_id`) as the DISCOVER/MEASURE interface;
   refactor the 2-D resolve to emit it; add the `per_feature_marginal`
   measurement path ("for a peak, its N features" in real units, on any feature
   joined by `sample_id`). N-feature storage; ≥3-D discovery raises. This is the
   "different analysis paths from one distribution object" spec.
4. **`SUBSPEC_D_comparison_plotting_engine.md`** — overlay engine (fast
   two-object overlay), 1-D KDE strips, and the N-feature × M-comparison grid
   built on the existing `plot_grid` / feature-over-time helper.

Each sub-spec is independently reviewable and states its own "done when." Do
**not** treat these as a single PR — B first (identity + real coords), then A
and C in parallel, then D.

## Sequencing note

B is load-bearing: the `sample_id` spine + real↔canonical transform are what
C's `PeakMembership` and `per_feature_marginal` join on, what A reports peak
shifts in, and what D's strip axes use. **Ship B, then C's `PeakMembership` +
`per_feature_marginal` (that's the "measure a peak's N features" thing you want
first), then A and D consume membership.** A and D both depend on C's membership
object, not just on B.
