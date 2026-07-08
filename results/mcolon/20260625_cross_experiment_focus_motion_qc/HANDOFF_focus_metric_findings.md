# Focus QC metric findings — handoff (2026-06-30)

## FINAL LOCKED THRESHOLDS (2026-06-30)
Two independent QC gates on the eroded embryo INTERIOR. An embryo is flagged if EITHER:
- **Focus gate:  `interior_strong_edge_frac < 0.25`**  (blurred / structureless / ghost / dead)
- **Saturation gate:  `top_spread_p99_p90 < 4`**  (bright-tail crushed = over-saturated)

Population behavior (7,500-embryo cross-experiment sample): focus gate flags ~3.0%
(20251125 4.9%, skewed by thin larvae — accepted, "over-exclude and move on"); saturation
gate flags ~0.65%; combined OR ~3.1%, mostly independent (overlap=37). spread<4 chosen over
<5 because 4→5 adds ~0.3% that is almost entirely GOOD bright larvae in 20251125, not real
saturation. edge<0.25 chosen over 0.20 to be conservative on soft cases (costs ~3% incl.
some in-focus thin larvae). KNOWN: the saturation gate also flags bright DORSAL views —
dorsal cannot be separated from true saturation by intensity (see saturation section).
NEXT: wire these two gates into the pipeline; then motion QC.

## The question
Find an image metric that flags **out-of-focus embryos** in 2D snips, without being
confounded by brightness/contrast/stage. Cross-experiment: 20250305, 20251125,
20260206 (+ 20250912 ghosts, off-pipeline).

## What FAILED (and why) — do not re-walk these
Every **whole-embryo averaged** metric failed to separate blur from in-focus, because
they all measure *contrast amount*, which tracks stage/orientation, not focus:
1. `sobel_mean_emb_norm` (mean edge energy) — content-confounded.
2. `log_mean_emb_norm` (mean LoG) — content-confounded.
3. `sobel_fine_to_coarse_ratio` / `log_fine_to_coarse_ratio` (spectral tilt) — LoG ratio
   hijacked by saturation; Sobel ratio de-confounded but too weak to threshold.
4. `emb_iqr_local` (embryo intensity IQR) — a good DEAD/saturation flag, but NOT a blur
   detector: sorts embryo shape/stage (thin late larvae = low IQR but in-focus). IQR<50
   clips ~4% but mostly good thin larvae; only ~0.4% (IQR<30) is genuine junk.
5. `grad/laplacian` edge-width (v1) — artifact: explodes where laplacian→0
   (flat/saturated), so DEAD/SAT scored "most blurred". Bug, not signal.
6. Proper perpendicular edge-WIDTH (Marziliano, v2) — dominated by the embryo
   **silhouette boundary** (soft for everyone); internal sharpness drowned out. No
   separation, even vs gold in-focus.

Root cause across all six: **averaging a global statistic to detect a local/contrast-
orthogonal property.** Five+ independent methods landing in the same place is the result,
not bad luck.

## The reframe that worked
User supplied (a) **gold-standard in-focus anchors** and (b) the **20250912 A04 ghosts**
(t95,97,100,104,110). The ghosts revealed the *actual* dominant failure mode is NOT
"soft edges" — it's **structureless embryos**: big, bright, well-masked, but EMPTY inside
(out of focus so far the body has no internal texture). The A10 "blur" anchors turned out
to be borderline-soft / nearly in-focus — not real failures, which is why nothing flagged
them.

## What WORKS — two axes
Compute on the **eroded embryo INTERIOR** (erode ~12px to strip the silhouette boundary).
Script: `posthoc_interior_structure_blur.py`. Anchor metrics:
`tables/anchor_interior_structure_metrics.csv`.

### Axis 1 — whole-embryo (ghost) blur  ✅ VALIDATED
- **`interior_strong_edge_frac`** = fraction of interior with real gradient edges
  (grad>0.04). **Cut ~0.20** flags 4/5 ghosts with **zero false positives** vs 9 gold
  in-focus + 6 dorsal. Also catches DEAD (D03 = 0.192, low) — a dead embryo has no real
  internal edges. (Ghost t100 is the dissenter — it reads in-focus on EVERY metric and on
  prior z-stack rel_entropy too; likely less severe / different mask than tagged.)
- **`interior_std`** = interior intensity std. User's pick as best all-round for blur.
  NOTE: does **NOT** catch DEAD (D03 = 0.247, mid-pack with in-focus) — that's fine,
  because DEAD is caught by `interior_strong_edge_frac`. The two metrics are
  **complementary, not redundant**: they catch different failures, which is why we keep
  both. (A scatter of the two distributions is a TODO to make the complementarity visible.)
- `interior_grad_p90` tracks strong_edge_frac closely (redundant).

### Axis 2 — partial / regional blur  ⚠️ PROOF-OF-CONCEPT, blocked on curvature
- Split body into 5 bands head→tail along the **PCA long axis**; compute
  `interior_strong_edge_frac` per band; summarize `band_strong_min_over_med` (low = one
  region much worse) and `band_strong_cv`.
- **t216 (A10) confirms the concept perfectly**: bands `0.756;0.746;0.752;0.443;0.065` —
  head band collapses while tail stays sharp = exactly "head out of focus, tail sharp."
  (Note: t216 is ALSO a motion artifact; motion inflates edge lines, so it is a
  motion+partial-blur confound, labeled `motion_and_partial_blur`, not pure blur.)
- **NOT yet usable**: normal anatomy (smooth yolk band next to busy head) also produces a
  low band, so min/median fires on in-focus embryos too (e.g. H04 is gold-fine but bands
  diverge). **Fix = curvature / pca_spine arc-length bands** (compare each band to what
  that anatomical band *should* be, not to the embryo's own median). User has curvature
  coming; the band code has a documented hook to swap straight-axis projection for spine
  arc-length without touching the metric.

## Decision / current evaluation
Decile galleries (per-experiment, 20/decile) built for both metrics:
`posthoc_interior_structure_deciles_cross_experiment.py` →
`figures/interior_structure_deciles/` (reuses 7,500-embryo IQR sample). Scatter of the two
axes over the population: `figures/interior_structure/interior_std_vs_strong_edge_frac_scatter.png`.

**Operational conclusion — use an ABSOLUTE cut, NOT a decile cut:**
- Gate = **`interior_strong_edge_frac < ~0.20`**. Catches ghosts (4/5) AND dead (0.19), with
  minimal collateral. Validated on anchors AND the population: 0.20 sits just below the
  bottom edge of the healthy cloud in the scatter.
- Why not a decile cut: like all metrics in this family, strong_edge_frac PARTLY sorts
  stage/morphology (D1 = thin diagonal larvae that are IN FOCUS but low-mass; D3-D6 = round
  yolk-balls; D7-D10 = curled late embryos). So "drop the bottom decile" would discard good
  thin larvae. The absolute 0.20 cut corresponds to only the first ~2-3 cells of D1 (the
  genuinely structureless ones), sparing the thin-larva contamination above it.
- `interior_std` = complementary second axis (separates ghosts; does NOT catch dead). Keep
  for the scatter / defense-in-depth, not as a standalone gate.
- The two-axis scatter makes the complementarity explicit: dead = low-edge/mid-std corner,
  ghosts = low/low, in-focus = high/high, motion+partial (t216) = high/high (invisible to
  whole-embryo metrics → needs the band axis).

## Saturation / brightness QC (2026-06-30)
Separate axis from focus. Question: separate TRUE over-saturation from bright-but-fine
DORSAL views. Scripts: `posthoc_saturation_histogram_anchors.py` (masked interior
histograms), `posthoc_focus_vs_saturation_scatter.py` (decision scatter).

Findings (interior pixels, eroded mask):
- `frac_eq255` / 255-spike is NOT the signature — saturated A06 cases peak at ~248-250,
  NOT 255; dorsal-good actually has MORE mass at 255. Useless feature.
- The real "bright problem" discriminator is **`p99 - p90` (bright-tail spread)**:
  clean embryos = 18-28; ALL bright problems (saturated + dorsal + dead) = 1-3 (mass
  crushed into a ~2-3 level band near the ceiling). Cut **`p99-p90 < ~10`** cleanly peels
  the over-bright band off the well-exposed population. `interior_mean`/median brightness
  confirms it (the bottom band is uniformly bright).
- **HARD LIMIT**: true-saturation and dorsal-bright are INDISTINGUISHABLE by interior
  intensity stats — they occupy the same bottom-left corner in the focus-vs-spread scatter
  (`figures/interior_structure/focus_vs_saturation_scatter.png`). Histograms are
  near-identical. To keep dorsal while dropping saturated you'd need SPATIAL/pose info
  (orientation), not the histogram. That's a biology call: is a dorsal view usable
  downstream? The metric can flag "too bright" (incl. dorsal) but cannot rescue dorsal.
- Focus (edge_frac) and brightness (p99-p90) are ORTHOGONAL axes — some saturated embryos
  pass the focus cut. Gate on both independently. PREFER `p99-p90` over `interior_mean` as
  the brightness gate: in the familiar-style scatters
  (`figures/interior_structure/focus_vs_top_spread_familiar.png` and
  `..._interior_mean_familiar.png`, focus + saturation anchors over the 7,500 population),
  `interior_mean` is ANTI-correlated with edge_frac (bright⇒less structure, diagonal cloud),
  so a mean cut entangles the two QC decisions; `p99-p90` is more orthogonal. Tentative
  joint gate: `edge_frac > 0.25` AND `p99-p90 > ~10`. In BOTH plots the gold_dorsal_in_focus
  anchors slide into the bright-problem band — reconfirming dorsal can't be separated from
  saturation by intensity. Anchors: clean_reference (G04 t102,
  A10 t194, E04 t108), oversaturated (A06 t82/t92/t93), dorsal_bright_good (E10 t145),
  dead_saturated (D10 t143).

## Files
- `posthoc_interior_structure_blur.py` — anchor metrics, both axes (the core result).
- `posthoc_interior_structure_deciles_cross_experiment.py` — decile galleries.
- `posthoc_focus_anchor_registry_review.py` — anchor registry (24 anchors incl. gold sets,
  ghosts loaded by direct path in the interior script since 20250912 is off-pipeline).
- Dead-ends kept for the record: `posthoc_edge_width_blur_anchors.py`,
  `posthoc_sobel_feature_size_analysis.py`, `posthoc_iqr_decile_galleries_cross_experiment.py`.
