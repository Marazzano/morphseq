"""Diagnostics for the morphological-stage polynomial model (`mdl_stage_hpf`).

The morph-stage model is a degree-3 polynomial regression (sklearn
``Pipeline([PolynomialFeatures(3), LinearRegression()])``, 286 features) mapping
the 10 morph-VAE PCA coordinates -> ``predicted_stage_hpf`` (the Kimmel-1995
temperature clock: start_age + elapsed_h*(0.055*T - 0.57)).  It is trained on the
WT REFERENCE embryos only (T<34); hotfish is NOT in the training set (only in the
PCA basis).  See fit_morph_spline_v2.ipynb and src/.../stage_inference.py.

This module reconstructs that ref-only model and provides two families of
diagnostics, kept deliberately separate:

  (A) SUPPORT COVERAGE  -- where the reference training data actually lives, so
      we can tell dense (well-constrained) regions from sparse (extrapolated)
      ones.  Two complementary metrics:
        * local training density  : kNN distance to the reference cloud in PCA
        * leverage / Mahalanobis   : hat-value in the 286-dim polynomial feature
                                     space (the exact quantity governing linear-
                                     regression extrapolation variance), plus
                                     Mahalanobis distance in PCA space.

  (B) PREDICTION INSTABILITY -- how much mdl_stage_hpf wobbles, two ways:
        * bootstrap ensemble  : refit the ref-only poly on resampled EMBRYOS
                                (not rows) N times -> per-point SD of prediction.
        * analytic interval    : closed-form LinearRegression prediction variance
                                 in the 286-dim feature space.

  (C) LABEL VALIDITY -- the target is a temperature clock, not morphology truth,
      so even dense-support estimates inherit label error.  Helpers to surface
      where the clock and the morphology disagree.

Everything can be related to ``morph_dist_spline`` (deviation from the WT spline)
so reliability-vs-outlier-ness is directly readable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _default_data_root() -> Path:
    env = os.environ.get("MORPHSEQ_DATA_ROOT")
    if env:
        return Path(env)
    for c in [
        Path("/Users/nick/Projects/data/morphseq/results/20260528"),
        Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/results/20260528"),
    ]:
        if c.exists():
            return c
    return Path("/Users/nick/Projects/data/morphseq/results/20260528")


CACHE_DIR = _default_data_root()
FIG_BASE = Path(os.environ.get("MORPHSEQ_FIG_ROOT",
                               "/Users/nick/Projects/data/morphseq/results/20260805")) / "stage_reliability"

PCA_COLS = [f"PCA_{p:02}_bio" for p in range(10)]
STAGE_TARGET_COL = "predicted_stage_hpf"   # Kimmel-clock training target
MODEL_STAGE_COL = "mdl_stage_hpf"          # stored polynomial prediction
POLY_DEGREE = 3
TRAIN_TEMP_MAX = 34.0                      # reference training used T < 34

RNG_SEED = 20260805


def fig_dir() -> Path:
    FIG_BASE.mkdir(parents=True, exist_ok=True)
    return FIG_BASE


def savefig(fig, name: str) -> Path:
    out = fig_dir()
    for suffix in (".png", ".pdf"):
        fig.savefig(out / f"{name}{suffix}", dpi=200, bbox_inches="tight")
    return out / f"{name}.png"


# --------------------------------------------------------------------------- #
# Data + model reconstruction
# --------------------------------------------------------------------------- #

def load_tables():
    """Return (ref, hf, spline) PCA tables from the cache."""
    ref = pd.read_csv(CACHE_DIR / "ab_ref_pca_morph_df.csv")
    hf = pd.read_csv(CACHE_DIR / "hf_pca_morph_df.csv")
    spline_path = CACHE_DIR / "spline_morph_df.csv"
    spline = pd.read_csv(spline_path) if spline_path.exists() else None
    return ref, hf, spline


def attach_morph_dist_spline(hf: pd.DataFrame) -> pd.DataFrame:
    """Merge ``morph_dist_spline`` (deviation from the WT spline, the outlier
    axis) onto the hotfish table from joint_141_morph_seq.csv, keyed on snip_id.
    No-op if the joint table or column is unavailable."""
    joint_path = CACHE_DIR / "joint_141_morph_seq.csv"
    if not joint_path.exists():
        return hf
    joint = pd.read_csv(joint_path)
    if "morph_dist_spline" not in joint.columns or "snip_id" not in joint.columns:
        return hf
    keep = ["snip_id", "morph_dist_spline"]
    if "morph_branch_flag" in joint.columns:
        keep.append("morph_branch_flag")
    return hf.merge(joint[keep].drop_duplicates("snip_id"), on="snip_id", how="left")


def reference_training_frame(ref: pd.DataFrame) -> pd.DataFrame:
    """The exact rows the poly model was trained on: reference, T < 34, finite."""
    r = ref.loc[pd.to_numeric(ref["temperature"], errors="coerce") < TRAIN_TEMP_MAX].copy()
    r = r.dropna(subset=PCA_COLS + [STAGE_TARGET_COL])
    return r


def build_model(train_df: pd.DataFrame) -> Pipeline:
    """Fit the degree-3 ref-only pipeline (reconstruction of morph_stage_model)."""
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=True)),
        ("linear", LinearRegression()),
    ])
    model.fit(train_df[PCA_COLS].values, train_df[STAGE_TARGET_COL].values)
    return model


def verify_reconstruction(model: Pipeline, hf: pd.DataFrame) -> dict:
    """Sanity check: reconstructed prediction vs stored mdl_stage_hpf on hotfish."""
    pred = model.predict(hf[PCA_COLS].values)
    if MODEL_STAGE_COL in hf.columns:
        err = np.abs(pred - hf[MODEL_STAGE_COL].values)
        return {"mean_abs_err": float(np.nanmean(err)),
                "max_abs_err": float(np.nanmax(err))}
    return {"mean_abs_err": np.nan, "max_abs_err": np.nan}


# --------------------------------------------------------------------------- #
# (A) SUPPORT COVERAGE
# --------------------------------------------------------------------------- #

def knn_distance_to_reference(query_pca: np.ndarray, ref_pca: np.ndarray,
                              k: int = 15) -> np.ndarray:
    """Mean distance from each query point to its k nearest REFERENCE points
    (10-D PCA).  Large -> query sits in a region the training data barely
    covers.  Uses a cKDTree; distances are in PCA units."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_pca)
    # k+1 not needed for cross-set queries (query != ref points), but guard anyway
    dist, _ = tree.query(query_pca, k=k)
    if dist.ndim == 1:
        dist = dist[:, None]
    return dist.mean(axis=1)


def mahalanobis_to_reference(query_pca: np.ndarray, ref_pca: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each query point from the reference cloud
    (mean + covariance of the reference PCA coords).  Scale-free analogue of
    kNN density that accounts for the anisotropy of the training distribution."""
    mu = ref_pca.mean(axis=0)
    cov = np.cov(ref_pca, rowvar=False)
    inv = np.linalg.pinv(cov)
    d = query_pca - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def polynomial_leverage(model: Pipeline, train_df: pd.DataFrame,
                        query_df: pd.DataFrame) -> np.ndarray:
    """Regression leverage (hat value) of each query point in the 286-dim
    polynomial feature space: h(x) = phi(x)^T (Phi^T Phi)^-1 phi(x), where Phi is
    the training design matrix.  This is EXACTLY the term that scales the
    prediction variance of the linear layer -- large leverage => the fit is
    extrapolating and the estimate is untrustworthy.  For training points h in
    (0,1] and sum(h) = n_params; query points off-support can exceed 1."""
    poly = model.named_steps["poly"]
    phi_train = poly.transform(train_df[PCA_COLS].values)   # (n, 286)
    phi_query = poly.transform(query_df[PCA_COLS].values)   # (m, 286)
    # (Phi^T Phi)^-1 via pseudo-inverse for numerical safety
    gram_inv = np.linalg.pinv(phi_train.T @ phi_train)
    # h_i = phi_i^T gram_inv phi_i, computed row-wise
    return np.einsum("ij,jk,ik->i", phi_query, gram_inv, phi_query)


# --------------------------------------------------------------------------- #
# (B) PREDICTION INSTABILITY
# --------------------------------------------------------------------------- #

def bootstrap_prediction_ensemble(train_df: pd.DataFrame, query_df: pd.DataFrame,
                                  *, n_boot: int = 100,
                                  embryo_col: str = "embryo_id",
                                  seed: int = RNG_SEED) -> np.ndarray:
    """Refit the ref-only poly on N embryo-level bootstrap resamples of the
    training data; return an (n_boot, m) matrix of predictions for the query
    points.  Resampling whole EMBRYOS (all their timepoints) respects the
    non-independence of repeated frames.  Per-point SD across rows = empirical
    instability of mdl_stage_hpf."""
    rng = np.random.default_rng(seed)
    embryos = train_df[embryo_col].to_numpy()
    unique_embryos = np.unique(embryos)
    Xq = query_df[PCA_COLS].values
    preds = np.empty((n_boot, Xq.shape[0]), dtype=float)
    for b in range(n_boot):
        drawn = rng.choice(unique_embryos, size=unique_embryos.size, replace=True)
        # gather all rows for the drawn embryos (with multiplicity)
        idx = np.concatenate([np.flatnonzero(embryos == e) for e in drawn])
        boot = train_df.iloc[idx]
        model = build_model(boot)
        preds[b] = model.predict(Xq)
    return preds


def analytic_prediction_se(model: Pipeline, train_df: pd.DataFrame,
                           query_df: pd.DataFrame) -> np.ndarray:
    """Closed-form standard error of the mean prediction for the linear layer:
        se(x) = sigma * sqrt( phi(x)^T (Phi^T Phi)^-1 phi(x) )
    with sigma^2 = residual variance on the training set (n - p dof).  This is
    the analytic counterpart to the bootstrap SD -- exact for the linear layer,
    conditional on the degree-3 basis being correct."""
    poly = model.named_steps["poly"]
    lin = model.named_steps["linear"]
    phi_train = poly.transform(train_df[PCA_COLS].values)
    phi_query = poly.transform(query_df[PCA_COLS].values)
    resid = train_df[STAGE_TARGET_COL].values - model.predict(train_df[PCA_COLS].values)
    n, p = phi_train.shape
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    gram_inv = np.linalg.pinv(phi_train.T @ phi_train)
    leverage = np.einsum("ij,jk,ik->i", phi_query, gram_inv, phi_query)
    return np.sqrt(sigma2 * np.clip(leverage, 0, None))


# --------------------------------------------------------------------------- #
# (C) LABEL VALIDITY (target = Kimmel clock, not morphology truth)
# --------------------------------------------------------------------------- #

def kimmel_rate(temperature_c):
    """Kimmel-1995 temperature-scaled developmental rate (per hour)."""
    return 0.055 * pd.to_numeric(temperature_c, errors="coerce") - 0.57


def clock_vs_model_gap(df: pd.DataFrame, *, model_col=MODEL_STAGE_COL,
                       target_col=STAGE_TARGET_COL) -> pd.Series:
    """mdl_stage_hpf - predicted_stage_hpf: how far the morphology-read stage
    departs from the clock label.  For hotfish this is the morphological
    stage shift; large where the clock's constant-rate assumption breaks down
    (e.g. hot cohorts) OR where the poly is extrapolating."""
    return pd.to_numeric(df[model_col], errors="coerce") - pd.to_numeric(df[target_col], errors="coerce")


# --------------------------------------------------------------------------- #
# BIOLOGICAL STAGE VARIABILITY  (within-cohort spread, decomposed)
#
# Goal: separate REAL biological stage spread within each (temperature,
# timepoint) cohort from the estimation noise of the model.  Core identity:
#     Var(observed stage) ~= Var_bio + mean(per-embryo estimation Var)
# so  Var_bio ~= Var_observed - mean(bootstrap Var), floored at 0.
#
# Everything is driven by ONE refit ensemble (compute_refit_ensemble): an
# (n_boot, n_query) matrix of predictions under embryo-level bootstrap refits of
# the ref-only poly.  Per-embryo estimation variance comes from its columns; the
# SE of the corrected biological SD comes from resampling ensemble COLUMNS
# (embryos) in-memory -- no additional model refits needed.
#
# Three estimators are compared per cohort to detect abnormality leakage:
#   mdl_stage_hpf     (surface)     -- primary
#   spline_stage_hpf  (projection)  -- orthogonal spread definitionally excluded
#   nn_stage_hpf      (kNN)         -- nearest reference embryo
# --------------------------------------------------------------------------- #

COHORT_KEYS = ("temperature", "timepoint")


def compute_spline_stage(hf: pd.DataFrame, spline: pd.DataFrame) -> np.ndarray:
    """Euclidean-projection-to-spline stage: for each hotfish point, the stage
    of the nearest spline KNOT in 10-D PCA (argmin over knots).  Orthogonal
    (off-manifold) displacement is in the null space of this estimator, so its
    within-cohort SD is 'progression-only' variability."""
    from scipy.spatial import distance_matrix
    stage_col = "stage_hpf" if "stage_hpf" in spline.columns else MODEL_STAGE_COL
    knot_pca = spline[PCA_COLS].values
    dm = distance_matrix(hf[PCA_COLS].values, knot_pca)
    nn = np.argmin(dm, axis=1)
    return spline[stage_col].to_numpy()[nn]


def compute_refit_ensemble(train_df: pd.DataFrame, query_df: pd.DataFrame,
                           *, n_boot: int = 300, embryo_col: str = "embryo_id",
                           seed: int = RNG_SEED) -> np.ndarray:
    """The single (n_boot, n_query) ensemble everything downstream reuses.
    Embryo-level bootstrap of the training data (whole embryos), refit each
    time.  This is the ONLY place model refits happen."""
    return bootstrap_prediction_ensemble(
        train_df, query_df, n_boot=n_boot, embryo_col=embryo_col, seed=seed)


def cached_refit_ensemble(train_df: pd.DataFrame, query_df: pd.DataFrame,
                          *, n_boot: int = 300, embryo_col: str = "embryo_id",
                          seed: int = RNG_SEED, cache_name: str = "refit_ensemble") -> np.ndarray:
    """Same as compute_refit_ensemble but memoized to disk (the full 300-refit
    ensemble on ~11k x 286 costs ~5 min).  Cache key includes n_boot, seed, the
    query snip_ids and the training row count, so a stale cache never silently
    loads.  Delete the .npz to force a rebuild."""
    cache = fig_dir() / f"{cache_name}.npz"
    query_ids = query_df["snip_id"].to_numpy().astype(str) if "snip_id" in query_df else np.arange(len(query_df)).astype(str)
    key = np.array([n_boot, seed, len(train_df), len(query_df)])
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if (z["key"].tolist() == key.tolist()
                and z["query_ids"].shape == query_ids.shape
                and bool((z["query_ids"].astype(str) == query_ids).all())):
            return z["ensemble"]
    ens = compute_refit_ensemble(train_df, query_df, n_boot=n_boot,
                                 embryo_col=embryo_col, seed=seed)
    np.savez(cache, ensemble=ens, key=key, query_ids=query_ids)
    return ens


def _biological_sd_from_values(obs, est_var):
    """Given a cohort's observed stage values and per-embryo estimation
    variances, return (observed_sd, estimation_sd, biological_sd)."""
    obs = np.asarray(obs, dtype=float)
    if obs.size < 2:
        return np.nan, np.nan, np.nan
    var_obs = float(np.var(obs, ddof=1))
    var_est = float(np.mean(est_var)) if est_var is not None and len(est_var) else 0.0
    var_bio = max(var_obs - var_est, 0.0)
    return np.sqrt(var_obs), np.sqrt(var_est), np.sqrt(var_bio)


def cohort_stage_variability(
    query_df: pd.DataFrame, ensemble: np.ndarray, *,
    stage_col: str = MODEL_STAGE_COL, cohort_keys=COHORT_KEYS,
    support_mask: np.ndarray | None = None, min_n: int = 2,
    n_se_boot: int = 1000, seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Per-cohort variance decomposition of ``stage_col``.

    Var_observed -> (Var_bio + Var_estimation); returns observed / estimation /
    biological SD per cohort, plus a bootstrap SE on the biological SD obtained
    by resampling embryos WITHIN the cohort from the stored ensemble columns
    (no model refits).  ``support_mask`` (bool over query rows) restricts to
    well-supported embryos when given (the 'gated' version).
    """
    rng = np.random.default_rng(seed)
    df = query_df.reset_index(drop=True)
    # per-embryo estimation variance from the ensemble columns
    per_embryo_est_var = ensemble.var(axis=0, ddof=1)   # (n_query,)
    rows = []
    idx_all = np.arange(len(df))
    mask = np.ones(len(df), bool) if support_mask is None else np.asarray(support_mask, bool)
    for keys, sub in df.groupby(list(cohort_keys), sort=True):
        sel = sub.index.to_numpy()
        sel = sel[mask[sel]]
        n = sel.size
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(cohort_keys, keys))
        row["n"] = int(n)
        if n < min_n:
            row.update(observed_sd=np.nan, estimation_sd=np.nan,
                       biological_sd=np.nan, biological_sd_se=np.nan)
            rows.append(row)
            continue
        obs = pd.to_numeric(df.loc[sel, stage_col], errors="coerce").to_numpy()
        est_var = per_embryo_est_var[sel]
        o_sd, e_sd, b_sd = _biological_sd_from_values(obs, est_var)
        # SE of biological SD: resample embryos in this cohort, recompute
        boot_b = np.empty(n_se_boot)
        for i in range(n_se_boot):
            draw = rng.integers(0, n, n)
            boot_b[i] = _biological_sd_from_values(obs[draw], est_var[draw])[2]
        row.update(observed_sd=o_sd, estimation_sd=e_sd, biological_sd=b_sd,
                   biological_sd_se=float(np.nanstd(boot_b, ddof=1)))
        rows.append(row)
    return pd.DataFrame(rows)


def multi_estimator_cohort_sd(
    query_df: pd.DataFrame, *, estimator_cols, cohort_keys=COHORT_KEYS,
    support_mask: np.ndarray | None = None, min_n: int = 2,
) -> pd.DataFrame:
    """Plain within-cohort SD (no decomposition) for several stage estimators
    side by side -- the surface-vs-projection-vs-kNN agreement check.  Where the
    surface SD exceeds the projection SD, off-manifold variation is leaking into
    the surface stage in that cohort."""
    df = query_df.reset_index(drop=True)
    mask = np.ones(len(df), bool) if support_mask is None else np.asarray(support_mask, bool)
    rows = []
    for keys, sub in df.groupby(list(cohort_keys), sort=True):
        sel = sub.index.to_numpy()
        sel = sel[mask[sel]]
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(cohort_keys, keys))
        row["n"] = int(sel.size)
        for col in estimator_cols:
            vals = pd.to_numeric(df.loc[sel, col], errors="coerce").dropna()
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if vals.size >= min_n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
