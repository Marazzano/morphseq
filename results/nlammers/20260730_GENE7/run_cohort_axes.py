"""Fit the GENE7-native 10D basis and per-cohort intra-cohort axes; cache for the notebook.

Writes to ``data/``:
    gene7_global_scores.csv     567 wells in the whitened 10D basis + cohort metadata
    gene7_global_variance.csv   the basis's explained-variance spectrum
    cohort_sizes.csv            48 cohorts x n_wells
    cohort_summary.csv          per-cohort variance ratios, bootstrap CIs, stability, null pctiles
    cohort_loadings.csv         long-form loading vectors (cohort x axis x global component)
    cohort_similarity.csv       pairwise top-2 subspace similarity
    cohort_null.csv             permutation-null draws
    image_strips.csv            embryos sampled along each cohort's top-2 axes, with image paths
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import cohort_axes as ca
import morph_pca_spline as mps
from morphseq_integration import build_master_table

OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

print("[1/5] loading GENE7 master table ...")
g7 = mps.gene7_latents_from_master(build_master_table().query("has_morph").copy())
print(f"      {len(g7)} wells with legacy morphology")

WHITEN = False  # see notebook: whitening inflates the ~degenerate tail and changes no conclusion
print(f"[2/5] fitting GENE7-native 10D basis (whiten={WHITEN}) ...")
basis = ca.fit_global_basis(g7, whiten=WHITEN)
print(basis.explained_variance.round(4).to_string(index=False))

print("[3/5] fitting per-cohort intra-cohort axes (48 cohorts) ...")
cohorts = ca.fit_all_cohorts(basis)
sizes = ca.cohort_table(basis)
print(f"      {len(cohorts)} cohorts, sizes {sizes.n_wells.min()}-{sizes.n_wells.max()}")

print("[4/5] permutation null + cross-cohort comparison ...")
null = ca.permutation_null(basis, cohort_size=12, n_draws=300)
summary = ca.cohort_summary(cohorts, null=null)
similarity = ca.cohort_similarity_matrix(cohorts, k=2)
loadings = ca.axis_loadings(cohorts)

print("[5/5] resolving image strips along the top-2 axes ...")
strips = pd.concat(
    [ca.axis_image_strip(c, axis=a, n_images=12) for c in cohorts for a in (1, 2)],
    ignore_index=True,
)
print(f"      {len(strips)} tiles, {strips.image_exists.sum()} resolved on disk")

basis.scores.to_csv(OUT / "gene7_global_scores.csv", index=False)
basis.explained_variance.to_csv(OUT / "gene7_global_variance.csv", index=False)
sizes.to_csv(OUT / "cohort_sizes.csv", index=False)
summary.to_csv(OUT / "cohort_summary.csv", index=False)
loadings.to_csv(OUT / "cohort_loadings.csv", index=False)
similarity.to_csv(OUT / "cohort_similarity.csv")
null.to_csv(OUT / "cohort_null.csv", index=False)
strips.to_csv(OUT / "image_strips.csv", index=False)
print(f"\nwrote artifacts to {OUT}")
