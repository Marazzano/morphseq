# GENE7 — functional form of every regression we have run

One page, so the models can be compared without reading four scripts. Everything below is fit
**per contrast** (`target × temperature × timepoint`), never pooled across contrasts.

Notation, all per embryo *i* and cell type *k*:

| symbol | meaning |
|---|---|
| `y_ik` | cell count for type *k* in embryo *i* |
| `N_i` | total cells recovered from embryo *i* |
| `b_i` | binary crispant indicator, 1 = crispant, 0 = matched control |
| `s_i` | signed distance from the shrunken-LDA hyperplane — the supervised morphology score |
| `s_i^w` | `s_i` minus **its own group's** mean → orthogonal to `b_i` by construction |
| `h_i` | hinge, `max(s_i − c, 0)` with `c` = control mean; all controls collapse to 0 |
| `p_i` | pooled PC1 (unsupervised, labels never used) |
| `w_i` | within-crispant PC1 (unsupervised, crispants only) |

All continuous predictors are z-scored within contrast. Every model carries the same offset
`log N_i` and is tested with `glmQLFTest`; BH is applied **within contrast**, across cell types.

---

## The supervised arms — `fit_edger_contrasts.R`

Each of arms 1–3 is a separate single-predictor fit; the tested coefficient is always coef 2.

| # | arm | linear predictor | tested |
|---|---|---|---|
| 1 | `binary_z` | `log μ_ik = log N_i + α_k + β_k · b_i` | `β_k` — the classical contrast, the baseline to beat |
| 2 | `s_z` | `log μ_ik = log N_i + α_k + β_k · s_i` | `β_k` — graded morphology, **replacing** the label |
| 3 | `hinge_z` | `log μ_ik = log N_i + α_k + β_k · h_i` | `β_k` — the mosaic-F0 shape: controls flat, escapers flat, severity graded above `c` |

**Arm 4 — the conditional model (the one §4–§5 rest on).** One fit, four columns:

```
log μ_ik = log N_i + α_k + β_k·b_i + γ_k^ctrl·s_i^w·1[control] + γ_k^crisp·s_i^w·1[crispant]
```

Tested: `γ_k^crisp` (coef 4) is the headline — *does morphological severity predict composition
**within** the crispants, after the group difference is already accounted for?* `γ_k^ctrl` (coef 3) is
the matched negative control: the same question asked of embryos that were never injected, so it
measures what the slope machinery yields from noise alone.

`β_k` here (coef 2) is fitted but **discarded by the pipeline**; `check_binary_coefficient.R` exists
to recover it, because the difference between this `β_k` and arm 1's is the "indirect" gain.

**Arms 5–7 — the escaper decomposition.** Two-group fits on subsets, same machinery:

| # | arm | subset | indicator |
|---|---|---|---|
| 5 | `escaper_vs_control` | controls + escapers | 1 = escaper |
| 6 | `severe_vs_control` | controls + severe | 1 = severe |
| 7 | `severe_vs_escaper` | crispants only | 1 = severe |

Arm 7 is the clean one: both groups were injected, so injection efficiency, handling and batch
cancel. It is also the dichotomised counterpart of `γ_k^crisp` on the same embryos, so the pair
measures what binarising the gradient costs.

## The audit models — `check_binary_coefficient.R`

Built only to attribute gains. M3 and M4 add exactly **one** column each, so they cost identical
degrees of freedom and differ only in which group carries the gradient.

| model | design | read |
|---|---|---|
| M1 | `~ b` | baseline |
| M2 | `~ b + s^w·ctrl + s^w·crisp` | arm 4; here `β_k` is extracted |
| M3 | `~ b + s^w·ctrl` | matched negative control |
| M4 | `~ b + s^w·crisp` | the real thing |

## The label-free arms — `fit_edger_unsupervised.R`

| arm | linear predictor | embryos |
|---|---|---|
| `pooled_pc1` | `log μ_ik = log N_i + α_k + β_k · p_i` | all contrast members |
| `within_pc1` | `log μ_ik = log N_i + α_k + β_k · w_i` | **crispants only** (~11) |

Cell-type filtering and dispersion are still estimated on the binary design, deliberately: it fixes
which cell types are in play so the arms are comparable to the supervised ones. `within_pc1` runs on
a different embryo set from everything else — that is a real caveat, not a detail.

## The Hooke arm — `fit_hooke_models.R`

```
~ cohort_PC1 + cohort_PC2          (+ offset(log(Offset)) appended automatically)
```

One PLN regression per cohort, 48 cohorts, predictors being **that cohort's own two principal
axes**, centred on its own mean. Cross-cohort comparison happens at the level of coefficients only —
the axes are emphatically not the same vector between cohorts (measured subspace similarity sits
barely above the random-plane floor).

---

## edgeR here: structure and assumptions

**Structure.** A negative-binomial GLM with a log link, one fit per cell type, and `log N_i` as a
fixed **offset** rather than a covariate. Fixing the coefficient at 1 is what turns the model from
"abundance" into "composition": `β` is a log fold-change in *proportion*, not in raw count.

The pipeline holds three things constant across every arm so that arms are comparable rather than
merely each-defensible:

1. **`filterByExpr` on the binary design** — the tested cell-type set is identical across arms.
2. **`estimateDisp(..., robust = TRUE)` on the binary design, then reused** — `glmQLFit` is passed
   `dispersion = dge$tagwise.dispersion` in every fit, so no arm gets to re-shrink its own
   dispersion and look better for it.
3. **BH within contrast.**

**Quasi-likelihood.** `glmQLFit`/`glmQLFTest` puts a second, per-gene quasi-dispersion on top of the
NB dispersion and tests with an F statistic on the residual df. This is the machinery that makes
small-n honest — but it is also exactly why hit counts are not additive. Adding the two slope
columns to arm 4 absorbs within-group scatter, the QL dispersion falls, and the *binary* coefficient
sharpens even though nothing about the group difference changed. That is the "indirect" channel, and
it is generic to adding any informative covariate, which is why M3 exists.

**Assumptions worth stating.** Counts are NB given the offset; the mean–variance relationship is
shared across cell types up to the tagwise dispersion; embryos are independent; cell types are fit
**independently**, so nothing models the fact that proportions must sum to 1. That last one is the
substantive gap — a real expansion of one abundant type mechanically depresses the others, and this
model will happily report both as hits.

## How Hooke differs

| | edgeR (here) | Hooke / PLN |
|---|---|---|
| unit | one model **per cell type** | one model over the **full cell-type vector** |
| correlation between types | none — fit independently | latent multivariate Gaussian; correlations estimated |
| compositionality | handled only via the offset | same offset logic, but the joint latent absorbs the sum-to-one coupling |
| dispersion | NB tagwise + QL, empirical-Bayes shrunk | Poisson–lognormal; variance from the latent covariance |
| inference | F test on residual df | variational; SEs by bootstrap or `variational_var` |
| n per fit | 12–24 embryos | 9–12 embryos, ~226 cell types |
| cost | seconds | minutes to hours (the reduced `~1` fit runs a penalised 226×226 precision estimate) |

**The honest summary.** Hooke's model is the better-specified one — it is the one that actually
represents composition as a joint object instead of pretending 226 cell types are 226 independent
experiments. What it buys in specification it pays for in estimation: a full covariance from ~10
embryos is a hard problem, the variational approximation is doing real work, and runtime is
prohibitive at this scale. edgeR is used for the head-to-head arm comparisons precisely because it
is cheap enough to fit seven arms × 36 contrasts with a permutation null on top, and because holding
dispersion fixed across arms makes those arms comparable. The independence assumption is the price,
and it is the main reason the raw hit counts should be read as *relative* between arms rather than
as absolute counts of perturbed cell types.
