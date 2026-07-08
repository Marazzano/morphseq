# FINDING — the graph statistics' fragility is the kNN BANDWIDTH, not the whitening (2026-07-04)

**Status: verified in simulation, core fix applied, calls re-checked. To be presented
later as a graph (gap-size × conductance monotonicity sweep).**

This is a root-cause finding about *why the Stage-2 graph statistics (Fiedler, and the
newly added conductance) misbehave*. It is upstream of, and distinct from, the b9d2
corroboration-rule diagnosis in `STATUS_and_diagnosis.md §4`.

---

## TL;DR

- The kNN-graph statistics were built with a **single global heat-kernel bandwidth**
  (`sigma = median of all kNN distances`) in `support_geometry._knn_adjacency`.
- That global bandwidth is **unstable once a real gap opens between two modes** — the
  regime a *support-connectivity* probe must handle. It either over-smooths the two
  modes into one blob (spurious high connectivity / high conductance) or collapses the
  graph into disconnected pieces (degenerate Fiedler/conductance → `nan`).
- We suspected two culprits — **MAD whitening** (`normalize_shape`) vs. the
  **bandwidth**. A 2×2 sweep isolates it: **it is the bandwidth.** Whitening barely
  moves the result; flipping the bandwidth to a **local / self-tuning** sigma fixes it.
- **Fix 1 applied:** `_knn_adjacency` now uses a per-point self-tuning bandwidth
  (Zelnik-Manor & Perona 2004): `sigma_i = dist to i's k-th NN`, and
  `w_ij = exp(-d_ij^2 / (sigma_i * sigma_j))`.
- **Fix 2 applied:** the large-gap limit where the graph becomes truly *disconnected*
  was being scored **backwards** (as "connected"). `conductance()` now detects
  disconnected graphs (via connected-component count) and returns the maximally-broken
  value. This is the "residual nan" — it was the fix working (max-broken), mis-handled;
  see "The residual nan" section.
- This is why the earlier note deferred conductance ("MST + Fiedler already cover it")
  — conductance was **unusable under the global bandwidth**. With the local bandwidth +
  disconnected-case handling it discriminates correctly and is now wired into the
  Stage-2 bundle.

---

## Note on which quantity is tabulated

Two related numbers appear below; do not confuse them (an earlier draft did):

- **raw φ** = the classic conductance of the cut, `cut / min(vol)`. **Small φ = broken.**
  Used only in the diagnostic 2×2 sweep to see the bandwidth failure directly.
- **the shipped statistic** = `conductance()` returns the INVERTED form `1/(1+φ)`, so it
  points the same way as the other Stage-2 stats: **larger = more broken** (→ 1.0 when
  fully broken, → small when well-connected). This is what the WT-null p-values use.

## The decisive experiment (2×2: whitening × bandwidth) — RAW φ

Two Gaussian clusters (60 pts each, sd 0.4) at `(±sep, 0)`, sweeping `sep`, statistic =
**raw φ** of the Fiedler spectral cut. Correct behavior: **raw φ → 0 as the gap grows**
(thinner bridge = more broken). This sweep used a scratch inline φ (returns `nan` on a
degenerate/disconnected cut rather than short-circuiting) purely to expose the failure:

| gap (sep) | MAD + global | none + global | MAD + local | none + local |
|-----------|--------------|---------------|-------------|--------------|
| 0.5 | 0.0606 | 0.0422 | 0.0889 | 0.0463 |
| 1   | 0.0038 | 0.0003 | 0.0208 | 0.0020 |
| 2   | 0.0000 | 0.0000 | 0.0021 | 0.0000 |
| 3   | 0.0000 | **nan** | 0.0033 | 0.0000 |
| 4   | **0.5402** | **nan** | 0.0034 | 0.0000 |
| 6   | **nan** | **nan** | **nan** | 0.0000 |
| 9   | **0.6133** | 0.0000 | 0.0035 | 0.0000 |
| 14  | **0.5472** | **nan** | 0.0004 | 0.0000 |

Reading:
- **Global-bandwidth columns break** past a moderate gap — raw φ jumps back UP to ~0.5–0.6
  (reads "well connected" when the clusters are maximally separated) or degenerates. This
  is the exact non-monotonicity that made conductance look backwards in the first pass.
- **Local-bandwidth columns stay correct** — raw φ → 0 and *stays* there as the gap grows,
  **regardless of whitening**. `none + local` is textbook-clean.
- **Flipping whitening does NOT fix the global column; flipping bandwidth DOES.**
  ⇒ the fragility is the bandwidth, not the whitening.
- The lone `MAD + local` **nan** at gap=6 is NOT a bandwidth failure — it is the graph
  becoming genuinely **disconnected** (2 components, 60/60). See "The residual nan" below;
  the shipped function handles this correctly.

### Why global bandwidth fails
A global sigma is set by the *whole cloud's* typical spacing. When a genuine gap opens,
the median inter-point distance inflates, so the Gaussian weight no longer distinguishes
"within a mode" from "across the gap." Local self-tuning sigma adapts per point — each
point's neighborhood scale is set by its own neighbors — so within-mode stays tight and
the across-gap edges stay genuinely weak, independent of how far apart the modes are.

## The same sweep through the SHIPPED conductance() (inverted, local bandwidth, nan fixed)

Statistic = `conductance()` as wired (`1/(1+φ)`, **larger = more broken**), after both the
local-bandwidth fix AND the disconnected-graph short-circuit (see "The residual nan"):

| gap (sep) | MAD-whiten | no-whiten |
|-----------|-----------|-----------|
| 0.5 | 0.9183 | 0.9558 |
| 1   | 0.9797 | 0.9980 |
| 2   | 0.9979 | 1.0000 |
| 3   | 0.9967 | 1.0000 |
| 4   | 0.9966 | 1.0000 |
| 6   | 1.0000 | 1.0000 |
| 9   | 0.9965 | 1.0000 |
| 14  | 0.9996 | 1.0000 |

Monotone-ish and **stable — no nan, no blow-up, saturating at 1.0** (fully broken) as the
gap grows, under both whitenings. This is the corrected, shipped behavior.

---

## Do we even need the whitening? (secondary question — answered: no, but harmless)

Conductance **p-value** vs a matched-N WT null (WT = one connected blob), through the
fixed shipped `conductance()`:

| case | MAD whitening | no whitening |
|------|---------------|--------------|
| connected (WT-like) | p = 0.817 | p = 0.842 |
| clean split | p = 0.008 | **p = 0.008** |
| crescent (curved-connected) | p = 0.525 | p = 0.050 |

The call-driving split-vs-connected separation is essentially identical under both
whitenings (split p≈0.008 either way; connected p≈0.8). Note the crescent under
**no-whiten drops to p=0.050** — i.e. dropping the whitening makes conductance *more*
likely to fire on a curved-connected manifold, which is the exact false positive the
`support_call` valley-guard exists to catch (and does: the crescent still calls
`connected` because valley_p=1.00). So on this axis MAD-whiten is slightly SAFER for
conductance, the opposite of the tentative earlier read. Net: whitening choice does not
change the discrete/connected calls. Consistent with the decision-surface heatmap
(`plots/decision_surface_heatmap.png`): under the production rule the b9d2 call is the
same continuous result under MAD *and* raw. The whitening's one visible effect was
**saturating Fiedler at ~1.00** (see the `f1.00` cells in the MAD block of that heatmap
vs. the *alive* `f0.60–0.97` values in the raw block) — a downside, not a benefit.

**Decision:** the bandwidth was the real bug and is fixed. Whitening is left in place as
a knob (it doesn't change conclusions); dropping it is a lower-stakes, separate call.

---

## Two fixes applied + verification

**Fix 1 — bandwidth:** `support_geometry._knn_adjacency` — global median sigma → per-point
self-tuning sigma. Touches the production Fiedler path.

**Fix 2 — the residual nan / disconnected case:** `conductance()` now short-circuits when
the graph is disconnected (or the sign-split degenerates) to the **maximally-broken** value
`1.0`, instead of the old `0.0` (which wrongly read "fully connected"). See next section.

Re-checked after both fixes:

- **Gap sweep (shipped `conductance()`):** stable and monotone-ish, saturating at 1.0 as
  the gap grows, under both whitenings — **no nan, no blow-up** (table above).
- **Canonical shape calls (correct):** connected → `connected` (cond p=0.817); clean split
  → `discrete` (valley p=0.008, Fiedler p=0.008, conductance p=0.008, MST p=0.042);
  crescent → `connected` (Fiedler p=0.000 fires but valley p=1.00, so the false-positive
  guard correctly holds).

## The residual nan — what it actually was

Earlier I called it "a degenerate spectral bisection at that seed" and said the fix was
"better but not bulletproof." **That framing was wrong on two counts, now corrected:**

1. It was not a bandwidth failure at all. At `gap=6`, the two clusters are far enough
   apart that **no kNN edge bridges them** — the graph splits into **2 genuinely separate
   connected components** (verified: sizes 60/60, one per true cluster). This is the
   large-gap *limit*, i.e. the maximally-broken case — the fix working, not failing.

2. A disconnected graph has **as many zero Laplacian eigenvalues as components** (here
   the two smallest are both ≈0: `[-0.0, 0.0, 0.0057, 0.0064]`). Conductance uses the
   *2nd* eigenvector as the cut direction — valid only for a **connected** graph. With two
   components that "2nd" vector is an arbitrary component-indicator, constant over both
   comps, so its sign-split puts all 120 points on one side → the old code's
   `if side.all(): return 0.0` fired → the maximally-broken case was scored as
   **maximally connected** (backwards). (The scratch sweep's inline φ returned `nan` on
   the same degeneracy — same root cause, different sentinel.)

**Fix:** count connected components first; `>1` ⇒ return `1.0` (fully broken). Also map the
degenerate-sign-split and zero-volume fallbacks to `1.0` for consistency. Fiedler needed no
change — its raw value already collapses to 0 when disconnected, so `1/(1+0)=1.0` was
already correct there.

```
  gap moderate (sep=2)                 gap large (sep=6)
  1 connected component                2 separate components
  λ: 0, 0.002(Fiedler), ...            λ: 0, 0(!), 0.006, ...   ← two zeros
  Fiedler vec splits  L | R            "2nd" vec constant over BOTH comps
  cut ≈ 0  → raw φ ≈ 0 → stat→1        sign-split = all-one-side (degenerate)
  ✅ broken (via the cut)              OLD: return 0.0  ❌ "connected" (backwards)
                                       NEW: n_comp>1 → return 1.0 ✅ max broken
```

### Full re-run — DONE (2026-07-04)
- **Synthetic gate (`validate_framework.py`): PASSED.** All continuous-expected cases
  stay continuous — including crescent & spiral (the guard conductance could have broken).
  Discrete cases fire; the disconnected-graph handling now reports component counts
  (`two_discrete` → "2 components", `continuum_with_hole` → "5 components"). Confidence
  still rises with N. The 3 known hard/edge cases remain report-only.
- **Real anchors (`run_real_anchors.py`): ran end-to-end**, exit 0. `conductance_p` is now
  a column in both summary CSVs and is wired into `support_call`. **Calls unchanged:**
  cep290 continuous, b9d2 continuous everywhere.
- **Conductance does NOT manufacture a discrete call.** In `run_real_anchors.py`
  (homozygous-only, HTA:29/CE:8) `valley_p = 1.0` at every b9d2 bin — the density gate
  sees no gap, so no graph witness (conductance included) can corroborate. Conductance
  fires nowhere (b9d2 conductance_p ∈ [0.27, 0.99]). This is the honest outcome: the fix
  makes conductance *usable and correct*, not *biased toward discrete*.
- Reminder: the flat valley_p=1.0 here is the **homozygous-only** view. The pooled
  CE/HTA view (`morph_axis_connectedness.py`) is where valley fires (~0.01–0.02 at
  24/48 hpf) — that panel is where conductance's corroboration actually gets tested.

---

## Relationship to the b9d2 diagnosis

`STATUS_and_diagnosis.md §5` asked for "a Stage-2 statistic that separates an
*asymmetric density gap* from a *curved-connected manifold* better than Fiedler/MST."
Conductance (bottleneck **sharpness**, now stable) is a candidate — but note the b9d2
under-call there is attributed to the *AND corroboration rule*, and adding conductance
as a corroborating graph statistic does **not by itself** change that (it is still a
graph witness, subject to the same veto). Whether a stable conductance shifts any real
b9d2 bin is an empirical question for the full re-run, not yet answered.

---

## Provenance

- Definition confirmed against the standard graph-conductance/Cheeger-cut definition:
  `φ(S) = w(edges crossing S) / min(vol S, vol S̄)`, small φ = bottleneck.
  (Wikipedia "Conductance (graph)"; UVM MATH 395 spectral graph theory notes.)
- Local self-tuning bandwidth: Zelnik-Manor & Perona, "Self-Tuning Spectral
  Clustering," NIPS 2004.
- Sweeps run in env `segmentation_grounded_sam`; scratch, not committed as scripts.
