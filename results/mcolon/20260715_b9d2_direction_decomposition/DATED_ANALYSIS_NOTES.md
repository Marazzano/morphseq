# DATED ANALYSIS NOTES — 2026-07-15

b9d2 phenotype direction decomposition. Full arc + the projection-clustering
follow-ups. See `ANALYSIS_SUMMARY.md` for the detailed writeup; this is the dated
index of what was added this session.

## Core arc (scripts 1–8)
- `1_direction_fingerprints.py` → 3×2 diff-vs-classifier fingerprint (CE/HTA/pooled,
  dims on Y, time on X). Two phenotypes = two directions; classifier recovers both
  (CE→dim71, HTA→dim33); pooled = superposition. HTA invisible to raw mean-diff,
  found only by classifier.
- `2_dimension_concentration.py` → concentration curves + threshold table + scatter.
  By magnitude the signal looks 1–2 dim; by Cohen's d² it's ~40 dim. classifier² vs
  Cohen's d² essentially uncorrelated.
- `3_ablation_auroc.py` → k-fold CV AUROC over time, drop 71/33/both. Dropping both
  costs only ~0.016 AUROC → 71/33 are redundant convenient handles.
- `4_ablated_refit_fingerprints.py` → refit with 71/33 removed (drawn as dark
  streaks); classifier spreads broadly to other dims.
- `5_per_dim_variance.py` → **ROOT CAUSE**: 71/33 are the highest-variance dims
  (105×/8× median), high even in WT (constitutive, not mutation-driven). |coef|
  tracks WT variance. The "hot dims" story is largely a scaling artifact.
- `6_zscored_ndims_over_time.py` → after WT-z-scoring, both Cohen's d² and classifier²
  agree the phenotype is ~35–45 dim, stable over development.
- `7_coverage_response_curves.py` → 4 figs, dims-for-X%-coverage over time (10–90%).
  variance_raw: 1 dim for 90% (artifact); variance_zscored: ~50 dims. Smooth
  staircase, no magic minimal set.
- `8_direction_fingerprints_zscored.py` → the 3×2 fingerprint on z-normed coords,
  shared scale per column. CE≫HTA in mean-shift; classifier comparable across both.

**Headline:** corrected for variance scale, the b9d2 phenotype lives in ~35–45 dims,
not 2. Action: z-score dims before selection/clustering.

## Projection-clustering follow-ups (scripts 9–10)
- `9_direction_projection_clustering.py` → cluster embryos by projection onto
  phenotype directions. **Cond A** (1 pooled b9d2-vs-WT direction): CE/HTA don't
  cleanly separate — a 1-D severity smear (CE +1.49 > HTA +0.96 > WT, IQRs overlap).
  **Cond B** (2 split CE-vs-WT + HTA-vs-WT): clean orthogonal resolution
  (CE=(+3.34,+0.21), HTA=(+0.19,+1.17)). Slide point: must FIND the two directions
  to separate two phenotypes; a pooled direction fails.
  NOTE: feeding a 1-D feature into UMAP inflates it into a noisy 2-D blob — the
  honest view of the collapse is the projection scores / a 1-D scatter, not UMAP.
- `10_unknown_cluster_shuffle.py` → discovery stress test (unknown cross-time
  cluster identity). **C_identity_flip** (real CE/HTA modes, A/B naming flipped per
  time bin): degrades gracefully, CE≠HTA still visible (each bin still has the real
  split; per-bin flip partially averages the two real directions). **D_random_null**
  (random A/B partition): collapses to the diagonal (two identical severity axes).
  TAKEAWAY: finding the right split PER SLICE matters most; cross-time identity
  tracking matters less — coherence is robust to identity-shuffle.

## Data / reference
- Source: `results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv`
  (phenotype_clean ∈ {CE, HTA, wildtype}, zygosity, 80 z_mu_b, predicted_stage_hpf).
- qsubs: `run_b9d2_projection.qsub` (scripts 9), `run_b9d2_shuffle.qsub` (script 10).
- Figures/HTMLs/tables are gitignored; regenerate from the scripts.
