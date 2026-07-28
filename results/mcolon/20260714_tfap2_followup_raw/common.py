"""
common.py — shared constants for the raw-latent tfap2 follow-up.

This follow-up condenses ALL 16 tfap2 genotypes over developmental time directly
in the raw z_mu_b latent space (not the classifier margin space of the April run
at 20260413_tfap2_followup/). Batch-entry seams are surfaced as honest QC via
ridge_score, not forcibly closed.
"""
from __future__ import annotations

# The 5 experiments feeding the tfap2 aggregate parquet.
EXPERIMENT_IDS = ["20260213", "20260223", "20260224", "20260319", "20260320"]

# All 16 genotypes present in the aggregate (14 crispants + 2 controls).
TFAP2_GENOTYPES = [
    "inj_ctrl",
    "non_inj_ctrl",
    "tfap2a_crispant",
    "tfap2b_crispant",
    "tfap2c_crispant",
    "tfap2d_crispant",
    "tfap2e_crispant",
    "tfap2a_tfap2b_crispant",
    "tfap2a_tfap2c_crispant",
    "tfap2a_tfap2d_crispant",
    "tfap2a_tfap2e_crispant",
    "tfap2b_tfap2c_crispant",
    "tfap2b_tfap2d_crispant",
    "tfap2b_tfap2e_crispant",
    "tfap2c_tfap2d_crispant",
    "tfap2c_tfap2e_crispant",
]

# Time binning + solver reproducibility. bin_width=4.0 matches the PBX raw run
# (20260703_raw_latent_clustering_baseline) so the relative-forces profile derived
# from the PBX margin reference transfers on an apples-to-apples basis.
BIN_WIDTH = 4.0
RANDOM_STATE = 42

# Distinct, colorblind-friendly palette for the 5 experiments (viewer "experiment"
# view). Genotype colors come from build_genotype_color_lookup, not hardcoded here.
EXPERIMENT_COLOR_MAP = {
    "20260213": "#1B9E77",
    "20260223": "#D95F02",
    "20260224": "#7570B3",
    "20260319": "#E7298A",
    "20260320": "#66A61E",
}

# Continuous z_mu_b dim views for the viewer, by COLUMN SUFFIX (the latent-dim
# number as in columns z_mu_b_20..z_mu_b_99). 71 = genotype gradient, 33 =
# multivariate suppressor (from the classifier hot-dims work in
# 20260703_raw_latent_clustering_baseline scripts 10-12). Easy to extend.
DIM_SUFFIXES = [71, 33, 85]
