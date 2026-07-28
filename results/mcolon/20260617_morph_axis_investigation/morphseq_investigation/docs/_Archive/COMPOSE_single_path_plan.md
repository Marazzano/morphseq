# Single compositional path — spec

Supersedes the V0 install (`Disturbion_Peak_broad_refactor.md`). Corrections:

1. Enrichment verbs are `compute_*`, not `add_*`.
2. **One** path from support to comparison. No parallel resolve engine, no
   parallel readout re-resolve.
3. Density is specified **once**: `ResolvedPeakAnalysisSpec` is the sole
   bandwidth authority; `KDESpec` is derived from it, never passed alongside.
4. The intermediate products (density, candidates, sample→peak assignment) are
   **fields on the resolved distribution**, not separate public compute steps.
5. The downsample **vote is retained whole** (not collapsed to a boolean), so
   mean/mode/variance of peak count are readable per distribution — this is what
   makes the bifurcation-over-time figure a free downstream read.
6. "Robust / stable" is a **swappable policy applied to the vote**, defined in
   one place; plotting never learns the definition, only reads the result.
7. **Resolution strategy and evidence retention are orthogonal axes**, not three
   strategies. MVP and graph resolve differently; "middle" only retains more.
   MVP and graph produce the SAME `ResolvedPeakDistribution` shape — optional
   fields fill in as you climb. No disposable approximation that infects the API.
8. **Honest counts:** `target_peak_count` (voted) ≠ `resolved_peak_count`
   (achieved) ≠ `resolution_succeeded`. We never weaken the filter to force N.
9. **The raw KDE is never bent.** One honest full-data density; the count comes
   from the vote; peaks are the raw density's mass split into N in-basins that
   clear the empirical-mass floor. A weak mode reads as a small support fraction.
10. **One `PeakAcceptancePolicy`**, behavioral identity — same implementation +
    config applied in every bootstrap draw and the final full-data resolution.
11. **Shared resampling engine.** All resampling — the bootstrap vote AND the
    null-test permutation — goes through `analyze.utils.resampling` (`resample.
    subsample` / `resample.permute_groups`), NOT a hand-rolled loop. Deterministic
    `SeedSequence`, streaming p-value, fork-safe. We do not reinvent draws.

Guiding invariant #1 — **the figure is a pure function of the resolved objects.**
Nothing in plotting recomputes a grid, KDE, detection, or resample.

Guiding invariant #2 — **one scientific implementation** for density, detection,
acceptance, assignment, and summary. This does NOT mean observed and null records
use identical resampling DEPTH — only that they call the same primitives. (See
§1.10 null protocol.)

**Identity vocabulary — one thing (a PEAK) at two lifecycle stages.** There is
one kind of object: a peak. It is either a CANDIDATE (detected, not yet confirmed)
or RESOLVED (confirmed final). That lifecycle stage — not where the id is stored —
is the naming axis. Never name an array after its container ("grid").

```
candidate_peak_id    a detected peak, pre-confirmation
resolved_peak_id     a peak confirmed as final
resolved_peak_ids    the set of final peak identities (a peak-id array; over grid
                     cells is just ONE view of it — not "grid_ids")

sample_candidate_peak_ids   each sample → the candidate peak it belongs to (−1 = none)
sample_resolved_peak_ids    each sample → the FINAL resolved peak it belongs to (−1 = none)
      a sample-assignment array answers two things at once: it is a sample→peak
      assignment, and the peak is a candidate or a resolved one.
```

A bootstrap-draw peak or a graph node is NOT a separate namespace — it is a
CANDIDATE peak observed in a context, on its way to possibly becoming resolved.
Its within-context locator is `(draw_index, candidate_index)`; its stage is
"candidate." Only confirmation promotes a candidate to a `resolved_peak_id`.

**Plain-language vocabulary (so a biologist reads the object correctly).** The
foreground says PEAK, not "mode." A resolved peak IS a mode of the distribution;
we use one word — **peak** — everywhere a biologist reads (fields, titles,
counts). "Mode" appears only in prose about the statistical concept.

```
peak           the biological object: a resolved cluster of embryos (a "mode")
basin          INTERNAL mechanism only — the grid region whose mass a peak owns
               (watershed catchment). Never a foreground/biologist-facing name;
               the biologist reads "peak", the code carves "basins".
support        the embryos (samples) assigned to a peak
reliable       trustworthy = the count was stable AND the peaks are valid
```

---

# Part 1 — ontology, to primitives

`+` = "made of", `→` = "produces". ◆ = pure data primitive (no upward dependency).

## 1.1 Substrate

```
◆ ResolvedPeakAnalysisSpec = bandwidth_rule + bandwidth_multiplier
                           + peak_detector_method + assignment_rule
                           + min_component_mass_frac + min_sample_fraction
                           + min_prominence_ratio + outlier_density_floor_fraction
      the SOLE density+detector authority. KDESpec is derived from it.

◆ support points = ndarray(N, 2)   raw empirical coordinates

◆ CanonicalGrid = x_min + x_max + y_min + y_max + grid_size
      derives (cached): xs, ys, xx, yy, cell_area
      pure coordinate frame. Owns NO points, NO density.
      COMPUTED from (reference + target points + spec); SHARED by both members.
```

## 1.2 Density — a field on the grid

```
DensityField = grid: CanonicalGrid + values: ndarray(grid.xx.shape)
             + source_distribution_id
      "what density sits at each canonical coordinate." No peak semantics.
```

## 1.3 Detector evidence + the one acceptance policy

```
◆ PeakCandidateDetail = candidate_index + (peak_x, peak_y) + peak_height
                      + prominence_ratio + basin_sample_fraction + basin_kde_mass
                      + component_mass + accepted(bool) + reject_reason
      candidate_index is DETECTION-LOCAL — no claim of persistence across draws.
PeakDetectionResult   = method_name + counts + tuple[PeakCandidateDetail]
      raw candidate evidence + accept/reject verdicts. Kept for audit.
```

**PeakAcceptancePolicy — one object, two staged methods (review #4).** "Filter"
undersold it: this object defines the scientific criteria under which something
is PROMOTED to a peak. Basin-dependent validation cannot happen before basin
assignment, so the object exposes two methods but ONE config, applied identically
in every bootstrap draw AND the final full-data resolution:

```
PeakAcceptancePolicy
   min_sample_fraction + min_prominence_ratio + min_component_mass_fraction
   .filter_detection_candidates(detection)      → candidate-local criteria
   .validate_resolved_basins(basins)            → the EMPIRICAL-MASS floor:
        a basin holding < min_component_mass_fraction of total empirical mass is
        discarded, NOT resurrected to reach a target count. (~0.10)
```

Invariant: one implementation + one configuration in bootstrap and full-data.
The two methods make explicit that the mass floor operates on post-assignment
basins, not raw candidates.

## 1.4 Resolved peak — one FINAL peak, in grid coordinates

`ResolvedPeakProfile` (review #11) honestly covers geometry + support descriptors
in one object rather than splitting into two; center/radius are geometry, the
rest are measurements on the final basin.

```
◆ ResolvedPeakProfile = resolved_peak_id + center(x,y)   ← a point ON the grid
               + radius + total_support_fraction
               + cv_radius_from_center + within_peak_r80_density
ResolvedPeak   = profile: ResolvedPeakProfile
               + source_candidate: PeakCandidateDetail | None   ← MVP fills; graph = None
               + bootstrap_presence_fraction: float | None      ← per-peak robustness;
                     None under MODE_VOTE (vote gives COUNT stability, not per-peak);
                     populated only by candidate-matching / graph strategies
      a candidate PROMOTED to a FINAL peak, measured against a distribution.
      source_candidate is nullable because graph peaks are consensus nodes with
      no single raw candidate; a ResolvedPeakProvenance union is the future shape.
```

## 1.5 Peak-count stability — the retained vote

Two policy families that vary INDEPENDENTLY are split (review #4): count-stability
interpretation vs graph construction. Raw vote evidence is separated from derived
interpretation (review #9).

```
◆ PeakCountStabilityPolicy = min_mode_frequency: float = 0.50   count stable if mode ≥ this
◆ PeakStabilityGraphSpec   = coverage_fraction: float = 0.80    (graph only; §1.5b)
                           + merge_distance_in_bandwidths: float = 1.0   (unit in the name)

◆ PeakCountVote   the RAW evidence (sparse — only OBSERVED counts, no zero entries)
      peak_count_frequencies: Mapping[int,int]   # {1:4, 2:76} — integer FREQUENCIES
      n_draws_requested + n_draws_valid + sample_fraction
      helpers: frequency_for(c)→int (0 if absent), probability_for(c)→float

PeakCountStability   the DERIVED interpretation (over VALID draws)
      vote: PeakCountVote
      mode_peak_count: int | None      # argmax
      mode_frequency: float            # mode's share ("76%")
      mean_peak_count: float           # drifts up as a cluster splits
      peak_count_variance: float       # spikes at transition, collapses after
      valid_draw_fraction: float       # invalid (crash/empty) = NON-vote, excluded
      count_is_stable: bool            # PeakCountStabilityPolicy: mode_frequency ≥ min
      mean/variance pure functions of the vote over VALID draws. An invalid draw
      is NOT a vote for "0 modes" — excluded, so variance stays trustworthy.
      NOTE: count_is_stable is ONLY the vote check; the full-data validity check
      (resolution_succeeded) is separate. is_reliable = both (§1.6).
```

## 1.5b Bootstrap evidence + the stability graph (retention-gated)

```
◆ BootstrapPeakCandidate = draw_index + candidate_index   ← BootstrapCandidateKey
                         + center + support_fraction + prominence_ratio
                         + component_mass_fraction | None
      key = (draw_index, candidate_index); cross-draw identity is CONSTRUCTED by
      matching, NEVER inherited.

PeakBootstrapTally = vote: PeakCountVote
                   + per_draw_candidates: tuple[tuple[BootstrapPeakCandidate]]  # CANDIDATES+
                   + per_draw_sample_assignments | None                         # SAMPLE_ASSIGNMENTS+

PeakNode (recursive)   a CONSENSUS CANDIDATE peak (many draws' candidates merged);
                       the graph is a tree, descended by RECURSING into a node
   node_id + center(consensus centroid)
   bootstrap_presence_fraction: fraction of draws this consensus candidate appears
   member_candidate_keys: the (draw,candidate) candidates merged in
                          (within merge_distance_in_bandwidths)
   children: tuple[PeakNode]   # sub-peaks it recursively splits into; empty = leaf
                               # base case: children within merge distance collapse

PeakStabilityGraph = roots: tuple[PeakNode]   # stable nodes at the resolved count
                   + count_regime             # counts kept (top coverage_fraction); tail dropped
      merging is candidate-key relabeling → node persistence.
      CAVEAT (review #5): per-CANDIDATE persistence is free from center-merge, but
      per-SAMPLE assignment confidence is NOT — it needs per-draw sample→candidate
      assignments (SAMPLE_ASSIGNMENTS retention). Candidate relabeling alone gives
      candidate persistence, not how consistently each sample belongs to a node.
```

## 1.5c Resolution strategy × retention (the durable seam)

MVP and graph genuinely RESOLVE differently. Retention only RETAINS more. Two
ORTHOGONAL axes, not three strategies:

```
ResolutionStrategy                 BootstrapRetention (what is PERSISTED)
   MODE_VOTE_FULL_DATA                SUMMARY_ONLY           just the vote
   STABILITY_GRAPH                    CANDIDATES             + per-draw candidates → candidate graph
                                      SAMPLE_ASSIGNMENTS     + per-draw sample→candidate → sample confidence
                                      FULL_DRAWS             + everything (debug)
```

**Retention governs PERSISTED evidence, NOT transient resolver inputs (review #2).**
Every strategy MAY collect draw-level candidates transiently during resolution
(e.g. MODE_VOTE needs consensus seed locations, §1.5d). `BootstrapRetention` only
decides what survives IN THE RESULT. So MODE_VOTE + SUMMARY_ONLY can still use
draw-consensus locations, then persist only the vote.

**The projection contract (why the seam is durable).** EVERY strategy must
project onto the same FOREGROUND four (§1.6): `peaks` (final identity),
`sample_resolved_peak_ids` (final assignment), `resolved_peak_count`,
`is_reliable`. Only the `evidence` sidecar differs — GRAPH adds a
`stability_graph`; the answer shape is identical.

```
today (MODE_VOTE): vote → mass-split → final filter → THE peaks   ┐ same
future (GRAPH):    recursive merge → stable nodes → THE peaks     ┘ foreground

PeakResolutionResult
  ── foreground (the contract every strategy satisfies) ──
   peaks + sample_resolved_peak_ids + resolved_peak_count + is_reliable
  ── evidence (strategy-specific background) ──
   target_peak_count + resolution_succeeded + count_is_stable
   + count_stability + consensus_seed_set + full_data_detection
   + bootstrap_tally (CANDIDATES+) + stability_graph (STABILITY_GRAPH only)
```

The figure, peak metrics, and comparisons are all written against the foreground,
so they never change when the strategy changes. The graph is a richer BACKGROUND
explanation of the SAME foreground answer.

## 1.5d Consensus seed set — the vote gives N, not locations (review #3)

The vote gives a COUNT. Locations come from a SEPARATE operation over the
(transiently collected) per-draw candidates — this must not be magical:

```
◆ PeakSeed = seed_index + center + supporting_draw_fraction
PeakSeedSet = target_peak_count + seeds: tuple[PeakSeed] + construction_method
```

**MVP resolve (MODE_VOTE_FULL_DATA) — keep the KDE honest, split its mass.**
Do NOT force the KDE to have N modes. Keep one honest full-data `density`. Take
the count from the vote. Build a `PeakSeedSet` of `target_peak_count` consensus
locations from the per-draw candidates, carve the `density`'s mass into in-basins
seeded there, then run `PeakAcceptancePolicy.validate_resolved_basins`. Nothing
re-fit, nothing bent. A weak mode reads honestly as a small support fraction.

Failure semantics (review #17) — crisp, no partial foreground:
```
one carved basin fails validate_resolved_basins ⇒
   target_peak_count = N            (evidence)
   resolution_succeeded = False     (evidence)
   resolved_peak_count = None       (foreground: NO resolved answer)
   peaks = ()                       (foreground)
   sample_resolved_peak_ids[:] = -1 (foreground)
   rejected basins kept in evidence for audit (NOT exposed as N−1 peaks)
```
We never weaken the policy to force N, and never expose N−1 foreground peaks
while claiming failure — consumers would use them.

## 1.6 Resolved distribution — THE ANSWER (final peak identity)

`ResolvedPeakDistribution` is not a stage in a chain. It is the **terminal
biological claim** — the thing biologists, the figure, and the comparison layer
all read. `peaks` and `sample_resolved_peak_ids` are the **FINAL peak identity
and assignment**, not provisional candidates. Everything that computed them is
demoted to a background `evidence` sidecar, reachable but out of the way.

**The biologist's four questions ARE the foreground — four fields, four reads:**

```
How many peaks does this distribution have here?  → resolved_peak_count
Which peaks / where are they?                      → peaks  (FINAL: center, radius, support)
Which sample belongs to which peak?                → sample_resolved_peak_ids  (FINAL)
Can I trust it?                                     → is_reliable
```

```
ResolvedPeakDistribution = distribution_id + source_type
  ── FOREGROUND: the answer (plotted, compared, reported) ──────────────
   + density: DensityField              the full-data KDE (geometric substrate; backdrop)
   + peaks: tuple[ResolvedPeak]         FINAL peak identities (center, radius, support)
   + sample_resolved_peak_ids: ndarray  each SAMPLE → its FINAL peak (−1 = none)
   + resolved_peak_field: ndarray       each grid CELL → its FINAL peak id or −1
                                        (one VIEW of resolved_peak_ids; draws mode loops)
   + resolved_peak_count: int | None    "this many peaks" (None if resolution failed)
   + is_reliable: bool                  "trust it / don't"
  ── BACKGROUND: how we got here (rarely opened; audit / drill-down) ───
   + resolution_evidence: PeakResolutionEvidence
```

Both arrays carry FINAL resolved-peak ids: `sample_resolved_peak_ids` over
samples, `resolved_peak_field` over grid cells (a spatial view of the same
identities, NOT a separate "grid id"). If a pre-final watershed labeling is ever
needed it is a distinct evidence field, not this one.

```
PeakResolutionEvidence = target_peak_count: int | None    # the vote's answer (may ≠ resolved)
                       + resolution_succeeded: bool         # final-validity gate outcome
                       + count_is_stable: bool              # the vote check alone
                       + count_stability: PeakCountStability
                       + consensus_seed_set: PeakSeedSet    # how N locations were built
                       + full_data_detection: PeakDetectionResult
                       + bootstrap_tally: PeakBootstrapTally | None   # CANDIDATES+
                       + stability_graph: PeakStabilityGraph | None   # STABILITY_GRAPH only
      candidates and how-they-were-chosen live HERE. Candidates never leak into
      the foreground, the figure, or the comparison.
```

**Two checks, one honest reliability flag (review #6).** `is_reliable` is a
foreground convenience derived from two DISTINCT statements kept in evidence:

```
count_is_stable       the bootstrap COUNT vote was stable   (mode_frequency ≥ min)
resolution_succeeded  the full-data basins are VALID peaks  (final acceptance gate)
is_reliable = count_is_stable AND resolution_succeeded
```

Keeping both in evidence distinguishes "vote unstable but basins valid" from
"vote stable but a basin failed" from "both failed" — all of which would
otherwise collapse to one false. The final acceptance gate runs the SAME
`PeakAcceptancePolicy.validate_resolved_basins` used in every draw.

**Draws seed placement; the density gives geometry.** The draws tell us WHERE
the N modes recur (→ `consensus_seed_set`); the honest full-data `density`
provides mass and shape; `sample_resolved_peak_ids` assigns each sample to its
basin. Draws never supply peak geometry — only the seeds.

## 1.7 Scalar summary — computed on the FINAL peaks (comparison currency)

The final resolved peak ids + `sample_resolved_peak_ids` are the **stable
platform** on which every peak- and distribution-level metric is computed. Each
support point is assigned to a final resolved peak id or to none (−1). Only once
that platform exists — peaks final, assignment fixed — do we compute metrics.
Candidates never enter here; there is ONE notion of "a peak" past resolve.

```
ResolvedPeakDistributionSummary = distribution_id + source_type
   + number_of_peaks + assigned/unassigned_support_fraction
   + across_peak_{total_support_fraction,radius}_{mean,skew}
   + across_peak_cv_radius_from_center_mean
   + across_peak_distance_mean + across_peak_r80_density_mean
      computed FROM the final peaks + assignment; the lossy boundary:
      drops arrays, keeps scalars. These scalars are what comparisons contrast.
```

## 1.8 Comparison + inference

```
DistributionComparison = comparison_id + members{role → DistributionRecord}
                       + observed_metrics + null_tests
◆ ResolvedDistributionMetricDefinition = name + minimum_peak_count + default_alternative
      the registry names the DISTRIBUTION it summarizes (review #19): assigned
      fraction, number_of_peaks, across-peak stats are distribution-level, not
      per-peak. Rename RESOLVED_PEAK_METRICS → RESOLVED_DISTRIBUTION_METRICS is a
      Stage-3 mechanical change (3 live call sites) — kept here as intent.
◆ EmpiricalNullResult = observed_value + null_{mean,std,median,q025,q975}
                      + empirical_p_value + standardized_effect
                      + n_null + n_valid_null + valid_null_fraction + test_is_valid
      generic: one observed scalar vs one vector of nulls. Domain-blind.
```

## 1.10 Null-resolution protocol — shared primitives, NOT shared depth (review #7)

"One compositional path" means one IMPLEMENTATION of density/detection/acceptance/
assignment/summary — it does NOT mean observed and null records resample equally.

```
NullResolutionMode
   SINGLE_PASS      per permuted sample: density → detect → acceptance → final
                    peaks → summary. NO internal bootstrap vote; peak count is
                    whatever the single full-data pass yields under the policy.
   FULL_BOOTSTRAP   per permuted sample: the entire vote+resolve (expensive).
```

Default = SINGLE_PASS (chosen for cost). This is a DISTINCT resolution mode, not
merely "resolve without evidence": it also omits the vote that sets the count.
State it explicitly so the null path's peak count is understood as single-pass,
not vote-derived. `compute_summary_for_null_draw(...)` delegates to the same
primitives as observed resolution.

## 1.11 Shared resampling engine (uses `analyze.utils.resampling`)

Both resampling jobs go through the existing unified engine — deterministic
`SeedSequence`, fork-safe, `_FAILED`-draw tracking — NOT hand-rolled loops. We do
NOT modify the shared engine (it has other migration consumers); we call it
through a thin, typed morphseq adapter (`_resample_adapters.py`) that owns the
engine's stringly-typed conventions and the result bridges.

```
the bootstrap vote  → adapter.bootstrap_peak_vote(points, config)
      internally: resample.subsample(frac=config.bootstrap_sample_fraction)
      statistic returns a DICT per draw (not just a scalar):
         {"count": resolved_peak_count, "centers": [...] }   # centers for PeakSeedSet
      → out.samples (raw per-draw dicts) → PeakCountVote  (NOT resample.aggregate)
      → out.n_failed maps to invalid draws → valid_draw_fraction (engine's _FAILED
        IS our "invalid draw = non-vote" — already built)

the null test       → adapter.permutation_null_test(reference, target, config)
      internally: resample.permute_groups(a="X1", b="X2")
      statistic resolves BOTH groups SINGLE_PASS (§1.10), returns a DICT of
         per-metric deltas (outputs=RESOLVED_DISTRIBUTION_METRICS) → one run(),
         one p-value per metric (not one loop per metric)
      → resample.aggregate → PermutationSummary → EmpiricalNullResult
        (bridge: PermutationSummary.to_permutation_result())
```

**Adapter contract (why it exists).** The raw engine hands the statistic a
`data` dict and injects a magic `"indices"` key (subsample) or requires `"X1"`/
`"X2"` names (groups) — a stringly-typed handshake an agent will fumble. The
adapter's two helpers own those conventions and hand the resolver typed,
already-subset arrays. Resolver code calls the adapter, NEVER the raw engine.

**Shared-context convention (avoid rebuilding the grid per draw).** The engine
calls the statistic fresh each iteration with no precompute slot; `_perturb`
copies the `data` dict but shared objects pass by reference. So put the
`DistributionAnalysisContext` (grid + spec, immutable) in `data["context"]` — it
is shared across all draws, NOT rebuilt. Only the drawn points change per draw.

`resample.aggregate` yields permutation p (with +1 smoothing, Phipson–Smyth) and
bootstrap CIs. This retires the bespoke permutation loop in
`resolved_peak_analysis.py` and the `_resampled_mode_count` loop in
`valley_visualization.py` — one engine for all draws (spec correction #11).

## 1.9 Dependency wall

```
CanonicalGrid           knows coordinates          NOT points/density
DensityField            knows grid+values          NOT peaks
PeakDetectionResult     knows density evidence     NOT sample assignment/geometry
ResolvedPeak            knows one peak's geometry   NOT the whole distribution
PeakCountStability      knows the vote             NOT how peaks were detected
PeakAcceptancePolicy    knows promote/reject rules applied IDENTICALLY in draws + full data
PeakStabilityGraph      knows cross-draw structure builds identity by MATCHING, not inherited keys
ResolvedPeakDistribution owns the interpretation   does NOT re-resolve; evidence is a sidecar
Summary                 knows scalars              NOT the arrays it came from
Comparison              knows roles+contrasts      NOT how to resolve a member
EmpiricalNullResult     knows scalar vs vector     NOT the science above it
```

---

# Part 2 — the constructed flow

`DistributionRecord` = persistent subject; `DistributionComparison` = persistent
contrast. Every `compute_*` returns an enriched record and delegates the math to
the single engine.

```
INPUT: reference points, target points, spec: ResolvedPeakAnalysisSpec
   │
① derive_shared_grid(ref_pts, tgt_pts, spec) → CanonicalGrid   (shared frame)
   │
② DistributionRecord(points, grid) × 2        (reference, target — raw subjects)
   │
③ compute_resolved_peaks(record, resolution_config)   [config bundles spec/strategy/…]
   │     a. resample.subsample vote → PeakCountVote (shared engine, §1.11)
   │     b. target_peak_count = mode of the vote (over VALID draws)
   │     c. PeakSeedSet: N consensus locations from transient per-draw candidates
   │     d. one honest full-data density; carve mass into N in-basins at seeds
   │     e. PeakAcceptancePolicy.validate_resolved_basins → resolution_succeeded;
   │        count_is_stable = mode_freq ≥ min ; is_reliable = both
   │     f. FOREGROUND: peaks + sample_resolved_peak_ids + resolved_peak_field
   │        + resolved_peak_count + is_reliable  |  BACKGROUND: resolution_evidence
   │     → record.resolved_peaks  (the ANSWER + evidence sidecar)
   │     (retention decides what evidence PERSISTS; the resolver may hold draw
   │      candidates transiently regardless — §1.5c)
   │
⑤ compute_peak_stats(record)                  → record.peak_stats  (scalars, FINAL peaks)
   │
⑥ DistributionComparison{reference, target}
   │     compute_observed_metrics(comparison)  → observed_metrics  (ONE registry)
   │     compute_null_test(comparison, config) → null_tests
   │        via resample.permute_groups; each draw = SINGLE_PASS resolve (§1.10)
   ▼
⑦ PLOT — pure reads; SINGLE SOURCE OF TRUTH = density + peaks + is_reliable:
     backdrop contours ← target.resolved_peaks.density.values
     FINAL peaks       ← target.resolved_peaks.peaks[i].profile (center, radius)
     FINAL assignment  ← target.resolved_peaks.sample_resolved_peak_ids (scatter)
     "N peaks" / draw  ← resolved_peaks.{resolved_peak_count, is_reliable}
     mode loops        ← resolved_peaks.resolved_peak_field
   presentation policy (review #18): the object exposes peaks + is_reliable;
   the FIGURE decides whether to draw unstable peaks (audit: dashed) or suppress
   them (summary). Plotting still performs NO scientific recompute.
   ── secondary (evidence sidecar) ──
     "(f%)" confidence     ← evidence.count_stability.mode_frequency
     target-vs-WT arrows   ← comparison.null_tests
     bifurcation-over-time ← evidence.count_stability.{mean_peak_count,mode_peak_count,peak_count_variance} per hpf
     drill-down (GRAPH)    ← evidence.stability_graph roots + children
```

## Verb set (5 total; 2 on a record)

```
derive_shared_grid(ref_pts, tgt_pts, spec)                     → CanonicalGrid
compute_resolved_peaks(record, resolution_config)              → resolved_peaks
compute_peak_stats(record)                                     → scalar summary
compute_observed_metrics(comparison)                           → observed deltas (one registry)
compute_null_test(comparison, resolution_config)               → permutation p (SINGLE_PASS)
```

`compute_resolved_peaks` folds the vote + stability + evidence in (it needs the
bootstrap draws anyway to pick the count). Kept separate from `compute_peak_stats`
so null draws take the cheap SINGLE_PASS path.

## Target API (review #14/#15 — recommended, resolve exact shape at Stage 0/1)

Config bundling avoids parameter confetti and mismatched-piece footguns:

```
DistributionAnalysisContext = grid: CanonicalGrid + spec: ResolvedPeakAnalysisSpec
    both records share ONE context → enforces shared-frame + sole-authority
    (prevents grid-built-with-spec-A / resolved-with-spec-B)

PeakResolutionConfig = strategy: ResolutionStrategy + retention: BootstrapRetention
                     + n_bootstrap_draws (NOT "K") + bootstrap_sample_fraction
                     + count_stability_policy + graph_spec | None

DistributionRecord = distribution_id + source_type + points + analysis_context
                   + resolved_peaks | None + peak_stats | None
compute_resolved_peaks(record, resolution_config)
```

These are API-shaping decisions best finalized against the real call sites at
Stage 0/1; recorded here as the target, not pre-wired into every diagram above.

---

# Part 3 — what each stage unlocks downstream

- **① Frame** — target/reference peaks comparable + overlay-able; built once.
- **③ Resolve** — the object plotting reads; density / peaks / assignment /
  resolved_peak_field one owner, cannot drift, nothing recomputed to draw a mode.
  Honest by construction: `target ≠ resolved ≠ succeeded`, and the density is
  never bent to hit the count. A weak mode reads as a small `support_fraction`.
  Three capabilities fall out of the retained vote/evidence: (a) *confidence* —
  `is_reliable` + `mode_frequency` answer "we're sure there are N peaks," honest
  "unstable" when the vote splits; (b) *bifurcation time-series* — the retained
  vote per hpf shows a blob splitting (variance spikes while the count is still 1,
  mean drifts up, variance collapses once two clusters resolve) — a figure the
  current code cannot make; (c) *drill-down* (STABILITY_GRAPH) — `stability_graph`
  shows WHICH peak is splitting, via bandwidth-scaled cross-draw merge, as reads.
- **⑤ Summary** — cheap scalar boundary so ⑥'s null test never drags full
  resolved objects through thousands of permutations.
- **⑥ Compare** — row-2 readout reads `null_tests`; no parallel re-resolve; one
  registry ⇒ arrows/tables/observed metrics agree by construction.

Three organizing properties: **one owner per fact** (nothing stored twice),
**compute once / read many** (every expensive thing at a named stage), **the
retained vote makes new science cheap** (bifurcation diagram + split-structure
drill-down + honest instability are all reads off fields we already have).

---

# Part 4 — build stages

- **Stage 0 — collapse density authority.** `ResolvedPeakAnalysisSpec` is the
  only bandwidth source; derive `KDESpec` from it in the engine; drop the
  separate `kde=` from `derive_shared_grid` and the record helpers.
- **Stage 1 — unify + rename + kill dead compute (no figure change).**
  `add_*`→`compute_*`; collapse the 4-step resolve into `compute_resolved_peaks`
  delegating to the one engine; assignment + `resolved_peak_field` become
  fields; delete the discarded `add_observed_metrics` call + `_ = observed_metrics`;
  fold `_resolved_peaks` in. Adopt `analyze.utils.resampling` for the null
  permutation (retire the bespoke loop in `resolved_peak_analysis.py`).
  VERIFY byte-identical PNGs + stdout. NOTE: the resampling engine is a
  `SeedSequence` clean break — if numbers shift, it's the documented seeding
  change, not a bug; pin the null seed and re-baseline once, deliberately.
- **Stage 2a — MODE_VOTE_FULL_DATA + SUMMARY_ONLY (robustness out of plotting).**
  `compute_resolved_peaks` runs the vote via `resample.subsample`, builds the
  `PeakSeedSet`, mass-splits the `density`, runs
  `PeakAcceptancePolicy.validate_resolved_basins` → `resolution_succeeded`,
  fills `count_stability`. `render_gene` stops resampling and reads fields.
  Honest failure semantics wired (§1.5d). SHIP THIS.
- **Stage 2b — MODE_VOTE + CANDIDATES (retention only).** Persist
  `per_draw_candidates` (`BootstrapPeakCandidate`, `(draw,candidate)` keys).
  Storage only; zero figure change; unblocks the candidate graph with no rerun.
- **Stage 2c — SAMPLE_ASSIGNMENTS retention.** Persist `per_draw_sample_assignments`
  (needed for sample-level confidence — NOT free from candidate merge, review #5).
- **Stage 3 — readout from the record.** Reconcile the 3 metric sets into
  `RESOLVED_DISTRIBUTION_METRICS` (rename from `RESOLVED_PEAK_METRICS`, 3 call
  sites); `compute_null_test` as a comparison product driving row-2; retire
  `compute_reference_readout`. VERIFY row-2 identical.
- **Stage 4 — bifurcation time-series figure.** New plot over hpf bins reading
  `count_stability.{mean_peak_count,mode_peak_count,peak_count_variance}`.
- **Stage 5 — STABILITY_GRAPH + CANDIDATES.** Process the retained tally into a
  recursive `PeakStabilityGraph` (`merge_distance_in_bandwidths` merge,
  `coverage_fraction` regime, split structure; sample-node confidence needs
  Stage 2c). New drill-down figure. Same foreground shape — optional fields fill in.

**Code inspection — RESOLVED (verified against peak_counting.py / resolved_peak_metrics.py):**
- ✅ Seeded-N carving EXISTS: `_assign_cells_to_peaks(grid, peak_locations)`
  (peak_counting.py:240) + `_assign_points_to_peaks(points, peak_locations)` (:256)
  carve grid + assign samples by nearest of N given locations. `PeakSeedSet`→carve
  is a thin wrapper over these, NOT new machinery.
- ✅ `min_component_mass_frac`/`min_sample_fraction` are POST-basin (:774–783), read
  per-basin `basin_fraction` → belong in `validate_resolved_basins`. CAVEAT: the
  filter is gated `if len(peak_locations) > 1` — single-peak distributions skip it.
  The new policy MUST preserve that guard or single-mode behavior changes.
- ✅ `source_distribution_id` is only on the NEW `distribution_records.DensityField`
  (2 uses), not legacy `DensityGrid` — safe to drop when nested (review #13 = drop).

**Deferred (unchanged):** ProductMetadata, product_key, list_products,
RepeatedDrawTable, stability_draws dict namespace, extension_products.
```

---

# Part 5 — migration / refactoring plan (file map + order)

## 5.1 Files ADDED

```
core/peak_acceptance.py     PeakAcceptancePolicy (2 methods, 1 config).
                            EXTRACT the accept/reject logic currently inline in
                            _detect_kde_peak_basins_sample_support (peak_counting.py
                            :774–783), preserving the `>1 peak` guard. Called by
                            both the bootstrap draws and full-data resolution.
core/peak_stability.py      PeakCountVote, PeakCountStability, PeakCountStabilityPolicy,
                            PeakSeed/PeakSeedSet, PeakBootstrapTally,
                            BootstrapPeakCandidate. (PeakStabilityGraph + PeakNode +
                            PeakStabilityGraphSpec land here at Stage 5.)
core/peak_resolution.py     ResolutionStrategy, BootstrapRetention, NullResolutionMode,
                            PeakResolutionConfig, PeakResolutionResult, and the
                            resolve engine (vote → seed → carve → validate). This is
                            the ONE engine both compute_resolved_peaks and the null
                            path call.
core/_resample_adapters.py  Thin typed wrapper over analyze.utils.resampling (§1.11):
                            bootstrap_peak_vote(points, config) → PeakCountVote and
                            permutation_null_test(reference, target, config) →
                            EmpiricalNullResult. Owns the "indices"/"X1" conventions,
                            the dict-statistic outputs, and the aggregate→result
                            bridges. Resolver calls THIS, never the raw engine.
                            LOCAL to morphseq_investigation for now; promote to a
                            shared location only if a second consumer appears.
```

## 5.2 Files MODIFIED

```
core/distribution_records.py   THE big rewrite. add_*→compute_*; DistributionRecord
                               gains analysis_context, resolved_peaks, peak_stats
                               (foreground/evidence split); DensityField drops
                               source_distribution_id when nested; delegate all math
                               to peak_resolution.py. (Decision 5.5: greenfield vs
                               in-place.)
core/resolved_peak_metrics.py  ResolvedPeakProfile (rename PeakGeometry + support
                               fields); ResolvedPeak gains source_candidate|None +
                               bootstrap_presence_fraction|None; sparse rename of
                               summary fields (mode_peak_count etc.); rename
                               RESOLVED_PEAK_METRICS → RESOLVED_DISTRIBUTION_METRICS
                               + ResolvedDistributionMetricDefinition.
core/resolved_peak_analysis.py adopt analyze.utils.resampling (retire the bespoke
                               permutation loop); add NullResolutionMode=SINGLE_PASS
                               path; keep as thin orchestration over peak_resolution.
core/peak_counting.py          expose _assign_cells_to_peaks/_assign_points_to_peaks
                               as public seeded-carve helpers; extract acceptance
                               logic to peak_acceptance.py (leave detect_peaks calling it).
core/__init__.py               re-export the new symbols; drop retired ones.
valley_visualization.py        read foreground fields (resolved_peak_field,
                               sample_resolved_peak_ids, is_reliable); DELETE
                               _resampled_mode_count + _resolved_peaks; build
                               DistributionComparison and read null_tests for row-2.
```

## 5.3 Files RETIRED

```
resolved_peak_reference_readout.py   its permutation re-resolve → comparison.null_tests.
                                     Keep render_readout_cell (presentation) if reused;
                                     delete compute_reference_readout.
valley_visualization.py::_resampled_mode_count, ::_resolved_peaks   (folded into engine)
```

## 5.4 LIVE peripheral consumers — DO NOT BREAK (confirm with user)

The resolve/permutation engine has consumers beyond the figure. An agent must
keep these green (or the user marks them dead):

```
run_resolved_peak_permutation_array_task.py   SGE array-job path (per-task draws)
merge_resolved_peak_permutation_outputs.py     reduces array outputs
run_resolved_peak_v0_smoke_test.py             the validation smoke test
compare_bandwidth_rules_readout.py             bandwidth-rule diagnostic
morphseq_investigation/tests/test_*.py         must be updated, not deleted
```

→ USER INPUT NEEDED: which of these are still live vs abandoned? The array-job
path (`run_..._array_task` / `merge_...`) shares `run_resolved_peak_permutation_
draws` / `reduce_permutation_null_test` — adopting the shared resample engine must
either preserve those entry points or migrate the array path too.

## 5.5 Open decisions for the user before handoff

1. **Greenfield vs in-place** for `distribution_records.py` — object shapes changed
   substantially (foreground/evidence split, ResolvedPeakProfile). LEAN: build the
   new model in the new modules, swap `core/__init__.py`, delete old. Confirm.
2. **Defaults:** `n_bootstrap_draws` (current 80) and `bootstrap_sample_fraction`
   (current 0.80). Keep, or raise draws for a steadier variance estimate?
3. **SeedSequence re-baseline gate (human):** adopting `analyze.utils.resampling`
   changes null values for a given seed (documented clean break). WHO checks the
   before/after are scientifically equivalent, against WHAT (smoke-test embryos?
   b9d2/cep290 figures?) — this is a human sign-off, not an agent gate.
4. **Array-job migration scope** (from 5.4): migrate now or leave on the legacy loop?

## 5.6 Verify gates per stage (agent must not skip)

```
Stage 0/1   byte-identical b9d2 + cep290 PNGs + stdout — EXCEPT the null path,
            which re-baselines once under the new seed engine (decision 5.5.3).
Stage 2a    resolved_peak_count / is_reliable match the OLD _resampled_mode_count
            outcome on b9d2 + cep290 at every hpf bin (same draws/frac/seed policy).
Stage 3     row-2 readout numerically identical (or re-baselined per 5.5.3).
Stage 4/5   new figures; no regression to Stages ≤3 outputs.
```
