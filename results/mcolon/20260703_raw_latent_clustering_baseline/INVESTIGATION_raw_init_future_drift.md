# Investigation: why does raw z_mu_b UMAP init drift from future embryos while margin space doesn't?

## Case study embryo

`20260304_E03_e02` (genotype `pbx1b_crispant`), bin at 72hpf (raw_zmub) /
74hpf (margin, different bin offset). Missing 64hpf and 68hpf entirely
(12hpf gap instead of the usual 4hpf) -- confirmed present in both raw and
margin source data, so the gap itself is a real data fact, not introduced
by either pipeline.

## What's ruled out

- **Not a same-time-bin outlier globally.** Pooled across ALL embryos and
  ALL time bins in raw 80-dim z_mu_b space, this embryo's nearest neighbor
  distance (1.02) is well below the dataset's global mean pairwise distance
  (2.27) -- it is NOT a global outlier. Its 10 nearest neighbors in the
  pooled 80-dim space are almost all `pbx1b_pbx4_crispant` at OTHER time
  bins (48-124hpf), not `pbx1b_crispant` at 72hpf. See
  `6_outlier_proof_pca.py` / `figures/outlier_proof_pca3d.html`.
- **Not (only) explained by its own consecutive-frame jump.** Both raw_zmub
  and margin show their single largest self-trajectory jump exactly at the
  60->72 / 62->74 transition (the 12hpf gap), and margin's jump there is
  proportionally *larger*, not smaller, than raw's, once normalized to each
  arm's own typical jump scale. So "big jump across the missing-data gap"
  happens in both arms and is not the differentiator.

## What's NOT ruled out -- the actual signal

Measuring the target's distance from the BULK of embryos in future bins
(not its own past self, not same-bin neighbors) -- i.e. does the embryo's
72/74hpf position sit near where slightly-older embryos (t+1, t+2, t+3
bins ahead) already are -- normalized by each future bin's own internal
spread (nearest-dist / bin's mean pairwise distance, so scale differences
between raw_zmub's and margin's coordinate systems don't confound the
comparison):

| arm      | bin  | next bin ratio | +2 bins ratio | +3 bins ratio |
|----------|------|-----------------|----------------|-----------------|
| raw_zmub | t=72 | 0.06 (t=76)     | 0.61 (t=80)    | 0.43 (t=84)     |
| margin   | t=74 | 0.05 (t=78)     | 0.24 (t=82)    | 0.07 (t=86)     |

**raw_zmub**: ratio climbs sharply (0.06 -> 0.61 -> 0.43) -- well-integrated
one step ahead, then drifts into isolation from the bulk of the population
two and three bins ahead.

**margin**: ratio stays low throughout (0.05 -> 0.24 -> 0.07) -- consistently
well-integrated at every future distance checked, never drifts away.

This is the real, reproducible asymmetry: raw_zmub's local one-step bridge
looks fine, but the embryo's neighborhood becomes progressively inconsistent
with where the bulk of the population actually is a few steps further into
development. Margin space does not show this pattern.

## Open question (goal of this investigation)

Is the future-drift specific to raw_zmub **real in the underlying raw
80-dim feature space itself** (i.e. does this embryo's actual raw z_mu_b
vector at 72hpf sit far from the t=80/t=84 raw feature population too,
before any UMAP touches it), OR is it an artifact **introduced by the
per-bin UMAP fitting + one-step Procrustes chain** in `aligned_umap_init`
(`src/analyze/trajectory_condensation/init_embedding.py`)?

Relevant structural fact about `aligned_umap_init`: it fits each time bin's
UMAP independently from scratch, then rigidly rotates/translates it to
align ONLY with the immediately preceding bin (one-step lookback via
Procrustes on shared embryos). It never checks or enforces consistency two
or three bins ahead. If raw z_mu_b's per-bin UMAP fits are individually
noisier/less stable than margin's (plausible given raw is 80-dim vs.
margin's low-dim class-contrast features), a one-step-only alignment
chain would be exactly the mechanism that lets local (t -> t+1) coherence
look fine while multi-step (t -> t+2, t+3) coherence silently degrades,
since nothing in the chain ever re-checks against anything beyond the
immediate previous link.

Not yet distinguished which of the two (real raw-feature-space signal vs.
UMAP/alignment-chain artifact) is responsible. Next step: repeat the same
future-bin-distance-ratio measurement directly in the raw 80-dim z_mu_b
space (no UMAP at all) for this embryo, and see whether the same
0.06 -> 0.61 -> 0.43 drift pattern is already present there, or whether the
raw feature space shows a flat/low ratio like margin does (which would
point squarely at the UMAP/alignment step as the source of the artifact).

## Scripts/files referenced

- `6_outlier_proof_pca.py` -- pooled all-embryo, all-time-bin PCA outlier check
- `figures/outlier_proof_pca3d.html` -- interactive 3D PCA proof artifact
- `figures/condensed_raw_zmub/condensed_positions.npz` -- raw_zmub x0/positions
- `results/mcolon/20260407_pbx_analysis_cont/results/positioning/trajectory/combined_raw_condensation_5class_bin4_perm500/condensed_positions.npz` -- margin arm
- `src/analyze/trajectory_condensation/init_embedding.py` -- `aligned_umap_init`, one-step Procrustes chain mechanism

## Resolution

Computed by `7_raw_vs_umap_future_ratio.py` (single shared `future_bin_ratio` function, Euclidean, native per-bin membership). Table below: ratio at +1/+2/+3 future bins for `20260304_E03_e02`.

| space        | +1 bin | +2 bins | +3 bins |
|--------------|--------|---------|---------|
| raw80        | 0.98 (t=76) | 1.78 (t=80) | 1.53 (t=84) |
| x0_2d        | 0.06 (t=76) | 0.61 (t=80) | 0.43 (t=84) |
| positions_2d | 0.08 (t=76) | 0.30 (t=80) | 0.39 (t=84) |
| margin_2d    | 0.05 (t=78) | 0.24 (t=82) | 0.07 (t=86) |

Correctness gate [PASS]: `x0_2d` reproduces the climbing 2D raw-arm curve and `margin_2d` reproduces the flat margin curve — confirming the shared function faithfully reproduces the original (unsaved) investigation table, so the `raw80` number is trustworthy.

**Verdict: H1 -- drift is REAL in raw 80-dim feature space. The raw z_mu_b future-bin ratio climbs like the 2D raw arm, so the embryo genuinely diverges from the future population before any UMAP. The one-step Procrustes chain is NOT the cause.**

---

## Follow-up: does the margin (classification) representation PRESERVE the raw manifold, or REORGANIZE it?

Before interpreting the extra raw-space islands as timing/batch/phenotype/noise,
the earlier question: does margin space keep the same embryo neighbors as raw
z_mu_b, or reorder which embryos are considered similar? Computed by
`8_neighbor_preservation.py` in the ORIGINAL feature spaces (not UMAP), WITHIN
each time bin. Fair-compression control = first 10 PCs of raw80 (margin is 10-dim;
some neighbor loss is guaranteed by 80->10 compression, so the honest baseline is
an equally low-dim linear projection of raw itself).

Neighbor overlap fraction (pooled over 27 bins, n-weighted; join = 2570 embryo×bin
rows, 304 embryos):

| k  | raw80 vs margin10 | raw80 vs rawPC10 (control) | ratio margin/control |
|----|-------------------|----------------------------|----------------------|
| 5  | 0.206             | 0.424                      | 0.485                |
| 10 | 0.270             | 0.473                      | 0.570                |
| 15 | 0.320             | 0.506                      | 0.633                |
| 30 | 0.473             | 0.600                      | 0.788                |
| 50 | 0.656             | 0.718                      | 0.913                |

- Pairwise-distance Spearman (raw vs margin, per bin): mean **0.469** (0.11–0.79).
- Merging-vs-mixing, norm. H(margin_cluster | raw_cluster): mean **0.473**
  (0 = coherent merging, 1 = full mixing).

### Verdict: partial, scale- and stage-dependent reorganization (between P1 and P2)

1. **Not pure smoothing (rules out clean P1).** At k=10 margin keeps only 27% of
   raw neighbors vs 47% for a fair 10-dim PCA projection (ratio 0.57). Margin loses
   substantially MORE local neighbors than dimensional compression alone predicts —
   there is genuine classifier-specific reordering, not just softened valleys.

2. **Not pure scrambling either (rules out clean P2).** The ratio climbs with k
   (0.49 -> 0.91): coarse/global organization is largely preserved while FINE-local
   structure is what gets reorganized. This is the "poor locally, preserved coarsely"
   signature — margin smooths fine structure but retains broad organization.
   Consistent with the moderate mixing entropy (0.47) = neither coherent merging nor
   full mixing.

3. **Reorganization is worst at LATE stages.** Distance-rank agreement DEGRADES over
   developmental time (rho ~0.65 early -> ~0.15–0.35 late) and mixing entropy RISES
   late. The classifier reorders embryo similarity most exactly where late-stage
   phenotype structure is supposed to live. So margin-space late-stage groupings are
   the LEAST faithful to raw geometry — the place we'd most want to trust them.

### Consequence for interpretation
The extra raw-space islands are NOT simply modes that margin coherently merged. Margin
constructs a partly different local similarity geometry, increasingly so at later hpf.
"raw has more islands, margin has fewer blobs" is therefore a mix of (a) fair
dimensional compression, (b) classifier-specific fine-scale reordering, and (c) genuine
mode merging — and those must not be conflated. Whether the reordered fine axes are
nuisance vs biology is the next question; this only establishes the two spaces make
genuinely different similarity claims, most strongly late.

Outputs: `tables/neighbor_preservation_{by_bin,summary}.csv`,
`tables/pairwise_distance_spearman_by_bin.csv`, `tables/cluster_contingency_summary.csv`,
`figures/neighbor_preservation.png`.

---

## Genotype-stratified follow-up: is margin reorganization selective for crispants?

`9_genotype_stratified_preservation.py`. Controls = WT (`wik_ab`, recovered via
`BRIDGE_PLUS_WIK_AB_GENOTYPES`) + inj_ctrl, kept separate (nominally equivalent).
Estimand D_g(k,t)=O_g^PC10−O_g^margin (classifier loss beyond compression);
ΔD_g=D_g−D_controls (row-weighted WT+inj pool). Neighbor-composition uses
availability baselines + matched nulls so abundance/experiment can't fake biology.
**Replication gate (WT-EXCLUDED) reproduces script 8 exactly: pooled O_margin
0.270@k10 / 0.656@k50 → PASS.**

### Results (WT-included analysis mode, k=10 unless noted)

Enrichment nulls below are the **corrected eligible-pool nulls** (methods review):
for focal i the same-genotype / same-experiment baseline is the share within
C_i = bin \ ({i} ∪ N_i^raw) — the embryos actually eligible to *become* a gained
neighbor — NOT the whole-bin share. Conditioning on C_i moved R_gen slightly UP
(old whole-bin null gave 0.365–0.541; corrected gives 0.375–0.552), so the
same-genotype-gain finding is robust to the correction, not an artifact of it.

| genotype | D_g | ΔD_g | R_gen | R_exp | cond_exp_enr | Lost_gen | O_within(k3) |
|----------|-----|------|-------|-------|--------------|----------|--------------|
| wik_ab (WT, ctrl)   | +0.300 | +0.020 | 0.375 | 0.055 | −0.000 | −0.140 | 0.496 |
| inj_ctrl (ctrl)     | +0.262 | −0.017 | 0.471 | −0.004| +0.025 | −0.163 | 0.463 |
| pbx1b_crispant      | +0.236 | −0.026 | 0.489 | 0.020 | +0.031 | −0.176 | 0.399 |
| pbx4_crispant       | +0.208 | −0.055 | 0.492 | 0.064 | +0.062 | −0.163 | 0.388 |
| pbx1b_pbx4_crispant | +0.168 | −0.107 | 0.552 | 0.046 | +0.037 | −0.159 | 0.325 |

**What holds up:**
- **Controls agree** (correction 1): WT vs inj_ctrl O_margin nearly superimposed,
  |O_WT−O_inj| 0.037–0.075 across k (WT n/bin 8–16). ΔD baseline is sound.
- **Gained neighbors are genotype-aligned, not experiment-aligned** — and this
  survives the corrected null. Every group: R_gen ≈ 0.5, R_exp ≈ 0 (−0.004…+0.064),
  cond_exp_enrich ≈ 0.00–0.06. Both sides of the transform agree: **Lost_gen < 0
  at every genotype** (−0.14…−0.18) — the raw neighbors that margin *drops* are less
  same-genotype than the raw neighborhood, so margin discards cross-genotype raw ties
  and replaces them with same-genotype ones. This is the one strong, robust finding.
- **ΔD is inverted vs the naive hypothesis** (double −0.107, worsening late): margin
  reorganizes control neighborhoods MORE than crispant neighborhoods under D/ΔD.

**What the two extra diagnostics walk BACK (softening, per methods review):**
- **The late ΔD plunge to −0.35 is Mechanism B, not A** (panel e decomposition). It
  is NOT that the double crispant's margin preservation rises late. The four raw
  overlap curves at 80–116 hpf: controls' **O_PC10 climbs** to ~0.55–0.64 while the
  double's O_PC10 stays flat ~0.40–0.45; both O_margin curves stay flat. So D_controls
  widens late (controls develop a big compression-vs-classifier gap) and the double
  only looks good *relative* to that widening. We therefore do NOT claim margin
  increasingly distills the double-crispant phenotype late. (Panel f: n_double is
  thin, ~20, at the latest bins — treat late points cautiously.)
- **The full-population crispant advantage is genotype CONSOLIDATION, not internal
  fine-structure preservation** (within-group, Possibility A not B). With same-genotype
  candidates only (k=3), the double crispant preserves its internal ordering LEAST of
  all groups (O_within 0.325 vs controls 0.46–0.50, singles ~0.39). Margin pulls double
  crispants together as a block but does NOT honor their fine internal geometry — so we
  do NOT claim "margin preserves crispant fine structure."

### Verdict (softened)

The safe, supported statement:

> The margin representation does not disproportionately destroy crispant neighborhoods;
> relative to an equally-dimensional PCA control it changes CONTROL neighborhoods more
> than crispant ones, with WT and inj_ctrl behaving alike. Margin-**gained** neighbors
> are strongly enriched for matching genotype (R_gen ≈ 0.5) and not for matching
> experiment (R_exp ≈ 0), and this holds under a null restricted to the embryos actually
> eligible to be gained; margin-**lost** neighbors are correspondingly the cross-genotype
> raw ties. So the transform is genotype-aligned, not batch-aligned — this is the robust
> finding.

Two claims are explicitly NOT yet supported and were pulled:
1. **"Margin increasingly distills phenotype late"** — the late ΔD plunge is driven by
   the controls' PC10 curve rising, not the double's margin curve (Mechanism B).
2. **"Margin preserves crispant fine structure"** — full-pop preservation reflects
   genotype consolidation; within-genotype, the double crispant is preserved *least*.

For the original question (raw-space extra islands): margin is genotype-aligned and
does not selectively mangle perturbation structure, so the islands are unlikely to be
phenotype structure margin destroyed. But because margin consolidates by genotype rather
than preserving fine internal geometry, we cannot yet call the raw islands mere
control/batch fragmentation either — distinguishing sub-phenotype raw structure from
nuisance is the next step, and needs a within-genotype raw-island analysis.

Outputs: `tables/preservation_by_genotype_bin.csv`, `preservation_diff_in_diff.csv`,
`control_consistency.csv`, `neighbor_composition{,_availability}.csv`,
`within_group_preservation.csv`; `figures/genotype_stratified_preservation.png`
(6 panels: a preservation, b ΔD, c control consistency, d batch-vs-biology,
e ΔD decomposition, f support n).
