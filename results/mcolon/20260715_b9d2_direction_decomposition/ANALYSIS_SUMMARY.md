# b9d2 Phenotype Direction Decomposition — Analysis Summary

**Directory:** `results/mcolon/20260715_b9d2_direction_decomposition/`
**Date:** 2026-07-15
**Data:** `results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv`
(23,074 frames; `phenotype_clean ∈ {CE, HTA, wildtype}`, `zygosity`, 80 `z_mu_b` dims,
`predicted_stage_hpf`). WT = 40 embryos, CE = 38, HTA = 31, spanning 11–125 hpf.

---

## The question

b9d2 has **two labeled phenotypes** vs wildtype (CE and HTA). Two things to test:
1. Do the two phenotypes occupy **two distinct directions** in the raw `z_mu_b` latent?
2. Can a **classifier recover those directions on its own** (without us splitting the
   phenotype), so the method generalizes to phenotype *discovery* where labels are absent?

The motivation: the raw latent is dominated by nuisance variation (batch, stage, imaging).
We want to select the phenotype-relevant dimensions and cluster on that subspace — the
single-cell "highly variable genes" idea, but for morphology directions.

---

## What we found (in order)

### 1. Two phenotypes → two directions, and the classifier recovers both
`1_direction_fingerprints.py` → `figures/direction_diff_vs_classifier_3x2.png`
(rows CE/HTA/pooled; left = raw embedding difference from WT, right = signed classifier
coefficient; dims on Y in natural order 20..99, time on X).
- **CE → dim 71** (both the raw difference and the classifier agree — a mean-shift phenotype).
- **HTA → dim 33**, but **only the classifier catches it** — the raw mean-difference is
  nearly blank. HTA is a *distributional* shift that WT-subtraction misses.
- **Pooled = superposition** of both stripes.
Conclusion: two distinct directions; the classifier recovers both at the split; use
classifier coefficients (not raw differences) because HTA is invisible to subtraction.

### 2. But by significance, the phenotype is high-dimensional
`2_dimension_concentration.py` → `figures/concentration_curves.png`, `concentration_scatter.png`
- By **raw magnitude / classifier coef**, 1 dim carries 50–80% of the signal (that's 71/33) —
  looks 1–2 dimensional.
- By **Cohen's d²** (significance-weighted), you need **~40 dims for 90%**.
- **Cohen's d² vs classifier coef² is essentially uncorrelated** (CE ρ=0.07, pooled ρ=−0.22):
  many dims separate the phenotypes reliably but the classifier ignores them.

### 3. Ablation: 71/33 are redundant, not essential
`3_ablation_auroc.py` → `figures/ablation_auroc_over_time.png`
(pooled CE+HTA vs WT; k-fold CV AUROC over time via `run_classification`; 20 perms, quick pass).
- full mean AUROC 0.736; drop71 0.715; **drop33 0.738 (zero cost)**; drop_both 0.720.
- Dropping *both* hot dims costs only ~0.016 AUROC — the classifier re-recovers
  discriminability from the other 78 dims. 71/33 are convenient handles, not unique carriers.
`4_ablated_refit_fingerprints.py` → `figures/ablated_refit_diff_vs_classifier_3x2.png`:
with 71/33 removed the classifier spreads broadly (new tops CE→72/85/88, HTA→23/22/21);
no single dominant replacement stripe.

### 4. Root cause: 71/33 are just the highest-variance dims (a scaling artifact)
`5_per_dim_variance.py` → `figures/variance_heatmaps.png`, `variance_rank_pooled.png`,
`variance_vs_coef.png`
- **dim 71 = 105× the median dim variance (rank #1/80 in WT); dim 33 = 8× (rank #2).**
- They are just as high-variance **in wildtype** as in CE/HTA — constitutive to the latent,
  not caused by the mutation.
- **|classifier coef| tracks WT variance** (CE ρ=0.75, pooled ρ=0.48).
So the unstandardized latent lets these two dims swamp the geometry: the mean-difference,
the classifier coefficient, and the magnitude concentration curve all collapse onto them
**for scale reasons**. (Nuance: HTA gives dim 33 more coef than its variance predicts →
a genuine distributional signal there.)

### 5. Standardize → the real phenotype is ~35–45 dimensional
`6_zscored_ndims_over_time.py` → `figures/zscored_ndims_for_90pct_over_time.png`
(WT-per-bin z-scored dims; Cohen's d² solid, classifier coef² refit on standardized inputs dashed).
- Median dims for 90%: CE d²=46/clf=33, HTA d²=39/clf=34, pooled d²=48/clf=36.
- Both weightings agree the phenotype lives in **~35–45 dims**, stable across 8–124 hpf.
The "71/33 = phenotype" picture was the nuisance high-variance dims dominating the
unstandardized space — exactly why the real directions were previously unfindable.

### 6. Coverage response curves (dims for X% of signal, over time)
`7_coverage_response_curves.py` → four figures, each with 3 genotype panels, X = time,
Y = # dims, one colored line per coverage level {10,20,…,90}%:
- `coverage_variance_raw.png` — after ~16 hpf **even the 90% line sits at ~1 dim** (dim 71
  alone holds ~90% of all variance). The artifact in one picture.
- `coverage_variance_zscored.png` — a proper staircase: 10%≈2–3 dims, 50%≈10–18, 90%≈46–55.
- `coverage_classifier_raw.png` — the classifier inherits the bias (CE has dead-zones at 1 dim).
- `coverage_classifier_zscored.png` — smooth staircase, all genotypes alike: 10%≈1–2, 50%≈8–9,
  90%≈33–36.

| metric | median dims for 90% |
|---|---|
| variance, **raw** | **1–8** ← artifact (pooled = 1) |
| variance, z-scored | 46–55 |
| classifier, raw | 28–33 (CE dead-zones at 1 for low %) |
| classifier, z-scored | 33–36 |

"Are there a couple more really important ones?" — the 10% line sits at ~1–3 dims, so a
small handful carry a modestly outsized share, but it is a **smooth staircase, not a cliff**:
there is no tiny magic set; the phenotype is genuinely distributed.

---

## Bottom line

- The two b9d2 phenotypes have **two distinct, recoverable directions**, and the classifier
  finds both without being told there are two.
- The apparent 1–2-dimensionality (dims 71/33) is a **scaling artifact**: those are simply the
  two highest-variance dims in the raw latent, high even in wildtype.
- Corrected for scale, the phenotype lives in **~35–45 dimensions** — a smooth distribution,
  not a handful of magic dims.

**Action for downstream clustering (e.g. tfap2):** z-score / standardize each dim before
selecting the phenotype subspace or clustering, so dimension importance reflects phenotypic
signal rather than raw variance scale.

---

## Scripts & outputs

| Script | Output |
|---|---|
| `1_direction_fingerprints.py` | `figures/direction_diff_vs_classifier_3x2.png`; `tables/{classifier_signed_coef_long,embedding_diff_long}.csv` |
| `2_dimension_concentration.py` | `figures/{concentration_curves,concentration_scatter,ndims_for_90pct_over_time}.png`; `tables/{threshold_ndims,per_dim_contributions}.csv` |
| `3_ablation_auroc.py` | `figures/ablation_auroc_over_time.png`; `tables/ablation_scores.csv` |
| `4_ablated_refit_fingerprints.py` | `figures/ablated_refit_diff_vs_classifier_3x2.png`; `tables/ablated_refit_signed_coef_long.csv` |
| `5_per_dim_variance.py` | `figures/{variance_heatmaps,variance_rank_pooled,variance_vs_coef}.png`; `tables/per_dim_variance_long.csv` |
| `6_zscored_ndims_over_time.py` | `figures/zscored_ndims_for_90pct_over_time.png`; `tables/zscored_ndims_over_time.csv` |
| `7_coverage_response_curves.py` | `figures/coverage_{variance_raw,variance_zscored,classifier_raw,classifier_zscored}.png`; `tables/coverage_response_long.csv` |

Reused machinery: `analyze.classification.directions.extract.extract_classifier_directions`
(signed unit direction vectors), `analyze.classification.run_classification` +
`analyze.classification.viz.plot_aurocs_over_time` (k-fold CV AUROC over time). Heatmap layout
follows `20260703_raw_latent_clustering_baseline/13_per_dim_difference.py`
(`fig_raw_delta_vs_classifier`). All Python via
`conda run -n segmentation_grounded_sam --no-capture-output python ...`.
