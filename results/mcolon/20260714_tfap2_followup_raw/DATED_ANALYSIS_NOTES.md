# DATED ANALYSIS NOTES

## 2026-07-14 — raw-latent tfap2 followup (initial)
Cluster all 16 tfap2 genotypes over developmental time in the improved RAW z_mu_b
space (not the April margin space). Primary deliverable = interactive multi-view
time-slice HTML.
- `0_load_tfap2_latents.py` → bin raw z_mu_b by predicted_stage_hpf (4 hpf) →
  `tables/tfap2_binned_zmub.csv` (597 embryos, 16 genotypes, 80 dims, 39 bins).
- `1_condense_raw_tfap2.py` → aligned UMAP init → condensation on raw z_mu_b;
  ridge QC. **Seams self-closed**: ridge one-sided 0.66 (x0) → 0.15 (condensed),
  two-sided → 0.036. No bridge correction needed.
- `2_make_multiview.py` → `multiview_time_slice.html` (genotype / experiment /
  z_mu_b dim views).
- `3_dim_snip_galleries.py` → interpret latent dims by embryo snips (one gallery
  per 4 evenly-spaced timepoints). Findings: dim71 = body-axis elongation/
  progression; dim85 = curvature/dysmorphology; dim33 = hatching/axis-emergence;
  dim20 = image brightness (technical).
- `common.py`, `run_condense.qsub`, `ANALYSIS_SUMMARY.md`.

## 2026-07-15 — direction-projection clustering (b9d2 method applied to tfap2)
Following the b9d2 direction-decomposition work (see
`../20260715_b9d2_direction_decomposition/`), which showed the raw latent is
dominated by high-variance nuisance dims and that projecting onto classifier
phenotype directions is batch-orthogonal:
- `4_direction_projection_clustering.py` → z-score embeddings (inj_ctrl-referenced
  per bin), fit each of 15 genotypes-vs-inj_ctrl directions per bin, project each
  embryo onto all 15 → 15-dim feature → condense → multiview HTML (experiment +
  genotype). Tests whether the phenotype-direction projection gives clean genotype
  structure with experiment effects removed on the full tfap2 data.
  (15 vs-control directions chosen over C(16,2)=120 all-pairs: cleaner, better
  supported, mirrors the PBX 3-direction run that worked.)
- `run_direction_projection.qsub` — 24 GB (4 slots × 6 GB), submitted to cluster.
  Output: `figures/direction_projection/all/multiview_time_slice.html`.

## Notes
- Figures/HTMLs/tables (incl. the large binned CSV) are gitignored; regenerate from
  the scripts. Each run saves `x0_init.npz` + `condensed_positions.npz` +
  `projection_scores.csv` so it can be re-rendered without recomputing.
- Condensation solver is single-threaded numpy; qsub slots only help UMAP/BLAS.
