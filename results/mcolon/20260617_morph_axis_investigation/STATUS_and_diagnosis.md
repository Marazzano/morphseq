# Phenotype Geometry — Status & Diagnosis (2026-07-03)

Where the implementation actually stands, what the figures do and don't show, and
**why b9d2 is not being called discrete** even though CE is a discrete phenotype
over developmental time.

---

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
