# Phenotype Geometry — Status & Diagnosis (2026-07-03)

Where the implementation actually stands, what the figures do and don't show, and
**why b9d2 is not being called discrete** even though CE is a discrete phenotype
over developmental time.

---

> **Update 2026-07-04 — kNN bandwidth fix (see `FINDING_knn_bandwidth_instability.md`).**
> The Stage-2 graph statistics' fragility was traced to the **global** heat-kernel
> bandwidth in `_knn_adjacency` (unstable once a real gap opens), NOT the MAD whitening
> — isolated by a 2×2 sweep. Fixed to a per-point self-tuning bandwidth (Zelnik-Manor).
> `conductance` is now implemented + wired into the Stage-2 bundle (was deferred because
> it was unusable under the old bandwidth). Spot-checks pass; the full synthetic gate +
> b9d2/cep290 re-run is still TODO. This does not by itself resolve §4 (the AND-rule
> under-call); conductance is still a graph witness subject to the same veto.

> **Update 2026-07-05 — `valley_depth` sweep-resolution fix + bandwidth investigated
> (not applied; see §6).** Built `synthetic_valley_connectedness_grid.py`, rendering
> the valley-KDE visualization + connectedness vote table side by side for every
> `synthetic_scenarios.py` case at once — the density-vs-graph disagreement is now
> visible per scenario in one figure
> (`plots/synthetic_valley_connectedness_grid.png`). This surfaced a real bug:
> `three_discrete` and `small_middle` sometimes read `valley_p≈1.0` (should fire) even
> though they use the same separation as `two_discrete` (which correctly fires).
> Root cause: `valley_depth`'s super-level threshold sweep used only 25 fixed steps,
> which can step clean over a narrow split-window — the "all 3 modes separated
> simultaneously" band is narrower for 3 unequal-height modes than for 2 equal ones.
> **Fixed:** `VALLEY_SWEEP_STEPS` raised 25 → 200 (cheap; just re-labels the same
> 30×30 KDE grid more times). Verified this alone fixes some seeds. On other seeds
> the miss persists even at 200 steps — that residual is a **KDE bandwidth**
> (Scott's rule) problem, not a resolution problem; see §6 for the full
> investigation and why narrowing the bandwidth was tried and rejected.

## 0. Open questions (require further investigation)

- **The 30 hpf continuity dip.** b9d2 reads discrete at 24 hpf and 48 hpf but
  **continuous at 30 hpf under EVERY knob setting** — including "valley alone".
  At 30 hpf `valley_p = 1.0`, i.e. the density gap detector finds *no gap at all*,
  so this is not the rule suppressing a visible gap; the gap genuinely isn't there
  in that bin. Two competing readings, unresolved:
    1. **Transient biology** — CE/HTA morphology momentarily converges mid-transition
       (~30 hpf) before re-separating by 48 hpf. A fate split need not be monotonic.
    2. **Bin artifact** — the 30 hpf bin may be underpowered (small n), mis-staged
       (embryos really 24/34 smeared in), or the 4-hpf bin edge lands badly.
  Not yet distinguished. To settle: compare n_group at 30 hpf vs other bins, test
  bin-edge shifts / a rolling window, and eyeball the 30 hpf scatter directly.

- **Stage 1 "differs from WT" has NO plots yet.** See §3 — the foundational question
  (is a feature even different from wildtype?) is computed but never visualized. This
  should be generated before trusting any downstream discrete/continuous call.

---

## 1. Implementation state (what is built, on disk, in this directory)

The six-module framework from the design spec is **built and the synthetic gate
passes.** Nothing is committed yet — it is all uncommitted working tree.

| module | stage / job | status |
|--------|-------------|--------|
| `distribution_shift.py` | Stage 1 — differ from WT? (Wasserstein, JS) | built |
| `support_geometry.py`   | Stage 2 — where is probability? (valley, MST, Fiedler) | built |
| `density_geometry.py`   | Stage 3 — how distributed? (var, skew, kurt, IQR, tail, entropy) | built |
| `component_geometry.py` | Stage 4 — how many pieces? (HDBSCAN, GMM-BIC) | built |
| `confidence.py`         | cross-cutting per-statistic confidence | built |
| `phenotype_geometry.py` | orchestrator (owns the conditioning) | built |
| `synthetic_scenarios.py` + `validate_framework.py` | Phase-A synthetic gate | **passes** |
| `run_real_anchors.py`   | applies full tree to cep290/b9d2, 2 axes | runs; emits CSVs |
| `morph_axis_connectedness.py` | older valley-only panel script | runs; emits panels |

Outputs currently on disk:
- `tables/phenotype_geometry_summary.csv` (per gene × timebin × axis)
- `tables/phenotype_geometry_full_reference_summary.csv` (pooled over stage)
- `tables/bootstrap_nulls.npz`
- `plots/{b9d2,cep290}_connectedness_panel.png`, `connectedness_trajectory.png`,
  `phenotype_geometry_trajectory.png`, `framework_validation.png`

**Bottom line:** Phases A–F of the plan are essentially implemented. The remaining
work is not "build more stages" — it is **reconciling the framework's call with the
known biology (b9d2 = discrete).**

---

## 2. Two artifacts, and they disagree

There are two separate scripts computing "continuous vs. discrete," and they give
different answers for b9d2. This is the source of the confusion.

| script | group definition | b9d2 call |
|--------|------------------|-----------|
| `run_real_anchors.py` (the **new** 6-module framework) | homozygous-only, CE/HTA, 1st/99th-pct outlier clip | **continuous** everywhere |
| `morph_axis_connectedness.py` (the **old** valley panel) — *now edited* to pool all CE/HTA zygosities and drop outlier clipping | all labeled CE/HTA, no clip | **discrete at 24 & 48 hpf** |

The panel you have been looking at is the **old** script (already fixed). The new
framework's CSV still uses the homozygous-only filter (`{'HTA': 29, 'CE': 8}` in
every row), so it never sees the full CE arm.

---

## 3. What the figures actually show — and what they do NOT

### What the connectedness panel shows
Per hpf: the pooled mutant cloud, **MAD-whitened** (`normalize_shape`), colored by
phenotype label, with a `continuous`/`DISCRETE` badge and `valley / p / n`.

### What it does NOT show (the honest gaps)
0. **Stage 1 "differs from WT" has NO plot at all — the biggest missing piece.**
   `distribution_shift.py` has zero plotting code. The foundational question of the
   whole tree — *is a given feature even different from wildtype?* — surfaces only as
   a p-value in a CSV (`shift_wasserstein_p`, `shift_js_p`) and a coarse
   "wildtype-like" badge. There is **no figure** showing, per feature, how the mutant
   distribution sits against the WT distribution (and its matched-N null band). Every
   downstream discrete/continuous call gates on this step, yet a human cannot eyeball
   whether the gate fired correctly. **This must be generated before trusting any
   downstream result.** Minimum: per (gene × feature × hpf), overlaid mutant vs WT
   distributions with the shift statistic + null band annotated; ideally a
   feature-vs-hpf trajectory of the shift p-value with the "differs" threshold marked.
1. **The Stage-2 statistics over time are not plotted.** valley_depth, MST, and
   Fiedler each drive the call, but only `valley` appears (as one number). You
   cannot currently see MST or Fiedler, or their WT-null bands, across hpf.
   → *Fix requested: metric strip under each panel + a trajectory row.*
2. **We never tested that CE differs from HTA.** Stage 1 tests mutant-vs-**WT**,
   not CE-vs-HTA. The framework is deliberately label-free (it must *discover* the
   split from geometry), so "are these two features actually different?" is a
   question the current pipeline does not answer. → *Fix requested: a **separate**
   labeled CE-vs-HTA separation script, kept out of the unsupervised core.*

---

## 4. Why b9d2 is NOT detected as discrete — root cause (from the knob sweep)

Established by `diagnose_decision_surface.py`, which sweeps two system knobs
(normalize_shape whitening × corroboration rule) across hpf on the full labeled
CE/HTA population and records the call. Outputs:
`plots/decision_surface_heatmap.png`, `plots/decision_surface_perknob.png`,
`tables/decision_surface.csv`. **b9d2 is the probe** — known-discrete, so a setting
that calls it continuous at 24/48 hpf is under-calling.

The discrete call is made in `support_geometry.py::support_call`:

> **discrete iff `valley_depth` is significant AND a graph statistic (MST or
> Fiedler) corroborates** — with a small relaxation for marginal valleys backed by
> strong Fiedler.

### The finding: it is the corroboration RULE, not the whitening.

An earlier draft of this doc blamed MAD-whitening. **The sweep disproves that.**

**Whitening barely moves the call.** MAD / trimmed-core / raw give nearly identical
valley p-values at the biologically-relevant bins (24 hpf: v = 0.01 / 0.02 / 0.03;
48 hpf: v = 0.02 / 0.02 / 0.60 — raw only washes it out at 48). Under the current
rule, b9d2 is continuous under **all three** whitenings, at **every** hpf. (See the
flat lines in the KNOB-1 panel of `decision_surface_perknob.png`.)

**The two required witnesses contradict each other.** At the split stages:

| hpf | valley_p | mst_p | fiedler_p | current call |
|-----|----------|-------|-----------|--------------|
| 24  | **0.01** (gap!) | 0.90 | 1.00 | continuous |
| 48  | **0.02** (gap!) | 0.97 | 0.99 | continuous |

`valley_depth` **DOES see the gap** (significant). But `fiedler`/`mst` **refuse to
corroborate** (≈0.99 — "well-connected, no gap"). The current rule needs BOTH, so
the density-yes / graph-no disagreement resolves to *continuous*.

**Why they disagree — the real structure.** valley_depth is a *density* statistic
(is there an empty low-density channel?); Fiedler/MST are *connectivity* statistics
on a kNN graph (can you walk across?). b9d2's CE/HTA split is **asymmetric and
unequal-size** (≈38 CE vs 31 HTA, CE trailing toward HTA). That geometry has a
genuine density dip (valley fires) but retains short-hop graph bridges across the
thinning zone (Fiedler/MST stay connected).

**The design tension this exposes.** The AND-rule was deliberately built to suppress
the *graph-yes / density-no* false positives (crescent, spiral, outliers — where
graph stats fire without a real gap; see the synthetic HARD cases). b9d2 is the
mirror image the design did not anticipate: *density-yes / graph-no*. The same guard
that correctly rejects curved-connected manifolds also rejects b9d2's true split.
b9d2 and the false-positive cases sit on **opposite sides of the same valley-vs-graph
disagreement** — you cannot relax the rule to catch one without exposing the other,
absent a statistic that distinguishes "asymmetric density gap" from
"curved connected manifold" better than Fiedler/MST.

### Which knob recovers the known-discrete truth
Relaxing the rule from `valley & graph` → `valley alone` calls b9d2 **discrete at
exactly 24 and 48 hpf** (continuous at 14/18 before the split resolves, a dip at 30,
re-sharpening at 48) — under all three whitenings. That trace (green, KNOB-2 panel)
is the biological story: the fate split emerges, blurs, then sharpens. `valley |
graph` also fires but is over-eager early (14/18 hpf).

### On the group filter (separate, secondary lever)
The homozygous-only filter in `run_real_anchors.py` keeps only 8 of 38 CE embryos,
which independently weakens the arm. The sweep above already uses the **full** CE/HTA
population, so the continuous call there is *not* a filter artifact — it is purely the
AND-rule. The filter matters for the `run_real_anchors.py` CSV, not for this
diagnosis.

**Diagnosis:** the split is real and the density statistic sees it. b9d2 reads
continuous because the system *requires graph corroboration that this split geometry
structurally cannot provide*. This is a property of the decision rule, not of the
normalization and not of the biology.

---

## 5. Status of the diagnosis + where a change (if any) would go

Done (this is a diagnosis, not a fix — nothing in the core was changed):

1. ✅ **[this doc]** — state + root cause on current, unchanged code.
2. ✅ **Knob sweep** (`diagnose_decision_surface.py`) — mapped the decision surface;
   isolated the corroboration rule (not whitening, not the filter) as the lever that
   decides b9d2. Heatmap + per-knob small-multiples + CSV.

Still open / optional next steps (NOT yet done):

3. **Metric figure on the panel** — per-column valley/MST/Fiedler strips + trajectory
   row under the connectedness panels, so the valley-fires / graph-vetoes split is
   visible per stage (the sweep shows it in aggregate; the panel would show it
   per-embryo).
4. **Separate CE-vs-HTA script** — labeled two-sample separation over time, to
   quantify "at what level must we describe this distribution." Kept OUT of the
   unsupervised core. This is the ground-truth reference, not a tuning target.

**If a change is ever made** (not decided here): the target is the
`support_call` corroboration rule in `support_geometry.py`, NOT `normalize_shape`.
The real problem is that Fiedler/MST cannot distinguish b9d2's asymmetric density
gap from a curved-connected manifold. Two honest directions:
  - accept a `valley alone` (or valley-dominant) rule and accept the crescent/spiral
    false-positive exposure it reopens, OR
  - add a Stage-2 statistic that separates "asymmetric density gap" from "curved
    connected" better than graph connectivity does (the principled fix).

**Guardrail:** any rule change must be re-run through `validate_framework.py`. The
crescent/spiral/outlier synthetics exist precisely to catch a rule that starts
inventing gaps — b9d2 reading discrete is only trustworthy if those still read
continuous.

---

## 6. `valley_depth` sweep resolution + KDE bandwidth investigation (2026-07-05)

### What prompted this
`synthetic_valley_connectedness_grid.py` renders the valley-KDE view (real KDE,
significance ring) directly above the connectedness vote table, for every
`synthetic_scenarios.py` case in one figure — so the density-vs-graph disagreement
discussed in §4 (b9d2: density-yes/graph-no) and its mirror (spiral/crescent:
graph-yes/density-no) is visible per synthetic case at a glance, not just for the
two real genes. Run: `python synthetic_valley_connectedness_grid.py` →
`plots/synthetic_valley_connectedness_grid.png`.

This surfaced a real bug: `three_discrete` and `small_middle` sometimes read
`valley_p ≈ 1.0` (i.e. `valley_depth` obs = 0.0, no split detected at all) even
though they are built with the *same* blob separation (6.0) as `two_discrete`,
which reliably fires. This is independent of the b9d2 corroboration-rule question
in §4 — it's a bug in the density statistic itself, on the easy synthetic cases the
framework is supposed to nail.

### Root cause 1 (fixed): threshold-sweep resolution
`valley_depth` sweeps a super-level threshold down from the KDE peak on a **fixed
25-point grid** (`np.linspace(0.97, 0.02, 25)`), returning the first threshold where
the super-level set has ≥2 real (≥3%-mass) connected components. For 2 equal blobs
the "both separated" window is wide, so 25 coarse steps reliably land inside it. For
3 blobs of unequal apparent height (a denser center flanked by two outer blobs), the
window where *all three* are simultaneously separated AND each clears the mass floor
can be much narrower — confirmed directly: a finer 60-point manual scan found a real
split at `frac=0.841` that the 25-point production grid stepped over entirely (its
neighboring grid points both saw "no split").

**Fix applied:** `VALLEY_SWEEP_STEPS = 200` (was a bare `25` inlined twice, in
`valley_depth` and `valley_detection_detail` — now one shared constant so the
statistic and its visualization can't drift out of sync). Cheap: it's just
re-labeling the same 30×30 boolean grid more times, no new KDE evaluations.
Verified fix on the seed where the bug was first found (`valley_depth` went from
`0.0` to `0.846`, correctly matching a hand-checked finer scan).

A `_refine_valley_transition(...)` stub is left next to `VALLEY_SWEEP_STEPS` for a
bisection-based refinement, per discussion, but it is **not needed and not wired
in** — reasoning below.

**Why bisection was considered and not used:** bisection can only refine a
transition the coarse scan *already detected* (i.e. narrow down exactly where
between two known-different heights the split occurs) — it cannot discover a split
that the coarse scan stepped over so cleanly that both neighboring samples agree
("no split" on both sides). Our actual failure mode was the latter: the 25-point
scan's adjacent samples agreed with each other right across the real transition, so
there was nothing for bisection to refine. A flat, denser sweep was the correct fix
for *this* failure mode; bisection remains a plausible future optimization if a
denser flat sweep ever becomes too slow, not a substitute for a dense-enough initial
scan.

### Root cause 2 (investigated, NOT fixed — deeper issue): KDE bandwidth
Even at 200 steps, some random draws of `three_discrete` still read `valley_p ≈ 1.0`
(obs = 0.0). Diagnosis: `_kde_grid` calls `gaussian_kde(pts.T)` with **no
`bw_method`**, so scipy defaults to Scott's rule, whose bandwidth factor is
`n**(-1/6)` — a function of sample size ONLY, blind to whether the cloud is one
blob or three. On the specific failing draw, the default bandwidth (factor ≈0.541)
oversmooths the sparse middle cluster so no threshold ever shows all 3 blobs as
simultaneously real; a narrower override (`bw_method=0.5`, barely different from
default) does briefly resolve it — confirming the mechanism — but this is not
enough to ship as a fix; see below.

**Tested: does the bootstrap null absorb the narrower bandwidth's noise?**
Correctly framed per discussion: what matters isn't whether the *raw* `valley_depth`
number looks alarming on a single-blob control, it's whether it's *significant*
against a matched-N WT null computed under the *same* bandwidth. Ran
`compute_support_geometry` (not a reimplementation) with `_kde_grid` patched to
`bw_method=0.5`, 8 seeds × 80 resamples, `valley_depth` only, N=40:

| scenario | expected | frac of seeds significant (p<0.05) |
|---|---|---|
| unimodal_compact | connected | 0.00 |
| variance_only | connected | 0.00 |
| outliers | connected | 0.00 |
| spiral | connected | **0.12** |
| crescent | connected | **0.12** |
| two_discrete | discrete | **0.12** |
| three_discrete | discrete | 0.25 |
| weak_separation | discrete | 0.12 |
| small_middle | discrete | 0.12 |

**Verdict: rejected.** The null does absorb noise on the plain single-blob controls
(`unimodal_compact`/`variance_only` stay at 0.00 as they should — validating the
p-value machinery itself). But `bw_method=0.5` is not a viable fix for two reasons:
1. It reopens a real false-positive risk on `spiral`/`crescent` (12% each) — exactly
   the curved-manifold cases the density statistic is supposed to leave alone.
2. More decisively, it **breaks the easy true-positive case**: `two_discrete` (two
   cleanly separated equal blobs) is now only significant 12% of the time — no
   better than the spiral/crescent false-positive rate. The statistic has lost the
   power to distinguish an obvious split from a curved manifold. This is a
   regression, not a partial fix — narrowing bw was the wrong lever.

**Why this happened:** Scott's rule is a single global scalar that has to
simultaneously be narrow enough to resolve a thin/sparse middle cluster (favors
small bandwidth) and wide enough to never manufacture structure inside a smooth
curved manifold (favors large bandwidth). Those are competing constraints on one
knob; no fixed factor satisfies both. This is the same shape-of-the-problem as
`FINDING_knn_bandwidth_instability.md`'s global-vs-local sigma story for the graph
statistics, but on the KDE side, and no local/adaptive analogue has been tried yet.

**Status:** `VALLEY_SWEEP_STEPS=200` is shipped (real, isolated improvement, no
observed downside). The bandwidth is **left at scipy default** (not changed).
`three_discrete`/`small_middle` will still occasionally under-fire on certain random
draws until a shape-adaptive bandwidth (or some other principled fix) is found.

**Next step (not started):** research a bandwidth-selection method that adapts to
local structure rather than a single global `n**(-1/6)` scalar — e.g.
cross-validated bandwidth per cloud, adaptive/local bandwidth analogous to the
Zelnik-Manor per-point sigma already used in `_knn_adjacency`, or restricting a
narrower bandwidth to the mode-counting sweep only rather than the whole KDE. Any
candidate must be re-validated the same way as above (full `synthetic_scenarios.py`
sweep, multi-seed, checking BOTH false-positive scenarios AND the easy true-positive
`two_discrete` case — not just the originally-broken cases) before being adopted.

### New visualization
- `synthetic_valley_connectedness_grid.py` — one script, one figure, all 12
  synthetic scenarios: row 1 = real KDE + phenotype points + significance ring
  (reuses the same visual language as `valley_visualization.py`), row 2 = the
  connectedness vote table (reuses `connectedness_panel.py`'s per-metric
  DISCR/CONT convention) plus a density-vs-graph disagreement classification
  (`density-YES/graph-NO`, `graph-YES/density-NO`, `agree: discrete`,
  `agree: continuous`). Useful going forward as the first check after any change to
  `support_geometry.py` — it shows at a glance whether a change fixes the intended
  case without reopening a false positive elsewhere, across ALL scenarios at once
  rather than one at a time.
