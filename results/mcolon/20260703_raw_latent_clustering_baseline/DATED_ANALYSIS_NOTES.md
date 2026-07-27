# DATED ANALYSIS NOTES

## 2026-07-15 — z-scoring & phenotype-direction projection vs batch effects

Follow-up to the SEAM investigation (see `SEAM_INVESTIGATION_SUMMARY.md`) and the
b9d2 direction-decomposition work (`../20260715_b9d2_direction_decomposition/`),
which found the raw z_mu_b latent is dominated by a few constitutive high-variance
nuisance dims (71/33 — highest variance even in controls). Hypothesis: these drive
the PBX batch effects (batch columns/islands/entry-seams), and the classifier
removed batch effects before because it downweights those directions. Tests whether
z-scoring and/or projecting onto phenotype directions reproduces that batch cleanup.

- `34_zscore_classifier_select_clustering.py` (`run_zscore_select.qsub`) — three
  conditions, all inj_ctrl-referenced z-scored per bin, same condensation pipeline
  as `1_cluster_raw_latent.py`, multiview HTML by experiment+genotype:
  `z_all` (80 dims), `z_clf90` (top-90% classifier coef² mass), `z_clf50` (top-50%).
  NOTE: after z-scoring, variance-based selection is undefined (all var=1) → selection
  uses classifier-coefficient ranking. **Z-scoring alone did NOT remove the batch
  effects** → motivated the projection approach below.

- `35_phenotype_direction_projection_clustering.py` (`run_pheno_direction.qsub`) —
  represent each embryo×bin by its PROJECTION onto classifier phenotype directions
  (batch-orthogonal by construction). 3 directions (pbx1b/pbx4/double vs inj_ctrl),
  fit per bin on z-scored dims; project = w_filtered·z (one number/direction/bin) →
  3-dim feature. Coefficient filtered per-bin-per-direction on the fly at
  all/clf90/clf50. Outputs `figures/pheno_direction/{all,clf90,clf50}/`. "First one
  worked" — projection mixes experiments while keeping genotype.

- `36_all_pairwise_direction_projection.py` (`run_pairwise_direction.qsub`) — same
  recipe with ALL 6 pairwise directions among the 4 PBX genotypes (C(4,2)=6) → 6-dim
  feature. Full pairwise resolution + batch removal. `figures/pairwise_direction/`.

- `37_pbx_direction_fingerprints_zscored.py` — PBX 3×2 diff-vs-classifier fingerprint
  on WT(inj_ctrl)-z-normed coords, shared scale per column, not ablated. Left column
  shows a clean severity dose-response: **double > pbx4 > pbx1b** in mean-shift; right
  column (classifier) spreads across many dims, comparable across crispants even where
  the mean-shift is faint (distributional signal). `figures/pbx_direction_fingerprints_zscored_3x2.png`.

## Notes
- Reference geometry / z-score control for PBX = `inj_ctrl`.
- PBX binned columns carry a `_binned` suffix (e.g. `z_mu_b_71_binned`).
- Figures/HTMLs/tables are gitignored; regenerate from the scripts. Each run saves
  `x0_init.npz` + `condensed_positions.npz` + `projection_scores.csv`.
- Condensation solver is single-threaded numpy; qsub slots only help UMAP/BLAS.
