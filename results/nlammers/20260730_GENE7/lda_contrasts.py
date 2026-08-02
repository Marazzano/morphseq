"""Supervised morphological contrast axes: crispant vs matched control.

The question
------------
The intra-cohort PCA in :mod:`cohort_axes` finds the directions along which a cohort varies most.
Those directions are real -- the image strips show interpretable phenotype -- but nothing forces them
to point at the *perturbation*. This module asks the supervised version instead: within a matched
(temperature, timepoint) stratum, what direction in morphology space best separates a crispant target
from its controls, and how far along that direction does each embryo sit?

That signed distance ``s`` is the intended replacement for the binary crispant/control indicator in
the downstream PLN regressions. The motivating hypothesis is mosaic-F0 heterogeneity: injected
embryos are a mixture of effectively-null and escaper animals, so a binary label misclassifies the
escapers and attenuates the contrast, while a graded severity score partially recovers the true dose.

Design
------
**Space.** The first ``N_SHARED_DIMS`` axes of the GENE7-native 10-component PCA over ``z_mu_b_*``,
unwhitened -- the same coordinate system the cohort-PC covariates use, so the two are comparable.
LDA is equivariant under invertible linear maps of the feature space, so whitening would not change
the direction; unwhitened keeps the units interpretable.

**Matching.** One contrast per (target, temperature, timepoint): 3 crispant targets x 4 temperatures
x 3 timepoints = 36. Controls are the ``Control`` embryos in the identical (temperature, timepoint)
cell, so stage and thermal history are matched by construction rather than modelled.

Note the controls are **shared across the three targets within a condition** -- each control group is
reused three times. The 36 contrasts are therefore not independent, which matters for BH correction
and for any pooled statement across targets.

**Estimator.** Shrunken LDA. Plain LDA takes ``w = S^-1 (mu1 - mu0)``; at 11-vs-12 in 5D the pooled
covariance ``S`` carries 15 free parameters on ~21 df, and the inverse amplifies its smallest
eigenvalues -- exactly the worst-estimated ones -- so ``w`` swings toward whichever direction happened
to look tight. Ledoit-Wolf shrinkage replaces ``S`` with ``(1-a) S + a (tr S / p) I``, choosing ``a``
in closed form to minimise expected squared error. It self-tunes to ``a ~ 0`` when the data do not
need help. Unpenalised *logistic* regression is not used: 24 points in 5D are all but certain to be
linearly separable, the MLE then diverges, and the resulting "distance" has arbitrary scale.

``priors=(0.5, 0.5)`` is imposed so the hyperplane sits at the equal-prior midpoint of the two
centroids and does not shift when one group has an extra embryo.

**Distance.** ``s = (w.x + b) / ||w||`` -- the signed Euclidean distance normal to the hyperplane,
positive toward the crispant side.

What gets assessed, and why each is a different question
--------------------------------------------------------
``hotelling_p``
    Is there *any* centroid separation? Classical parametric two-sample test in 5D.

``loo_auc`` / ``loo_auc_p``
    Does the direction *generalise*? Each embryo is scored by a discriminant fit on the other n-1.
    This is the one that matters for the intended use, because the score has to mean something for
    embryos the axis was not built around. A contrast can pass Hotelling and still fail here.

``boot_angle_p95``
    Is the direction *stable*? Bootstrap resamples, refit, angle to the full-data direction. A
    contrast can separate significantly and still have an essentially arbitrary axis. Interpretation
    needs the random-direction reference (``random_angle_p05``): two random unit vectors in 5D
    average 90 degrees apart, so the null spread is wide.

``angle_to_stage`` / ``angle_to_cohort_pc1``
    Interpretation, not significance. Morphology encodes developmental stage strongly, and controls
    at three timepoints give the stage direction directly. A discriminant near-parallel to it is a
    delay score, and any transcriptional hit will be a stage-composition shift wearing a costume.
    The angle to the unsupervised cohort PC1 says whether the perturbation direction is the dominant
    axis of within-cohort variation or a minor one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import cohort_axes as ca

N_SHARED_DIMS = 5
CONTROL_GROUP = "Control"
CONTRAST_KEYS: tuple[str, ...] = ("temperature", "timepoint_seq")

N_PERMUTATIONS = 1000
N_BOOTSTRAPS = 500
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Shrunken LDA
# ---------------------------------------------------------------------------


def ledoit_wolf_covariance(centered: np.ndarray) -> "tuple[np.ndarray, float]":
    """Ledoit-Wolf shrunk covariance of already-centered rows, toward a scaled identity.

    Inlined rather than called from ``sklearn.covariance`` because the permutation and bootstrap
    loops make ~10^6 of these calls on 24x5 inputs, where per-call framework overhead dominates the
    arithmetic. ``test_matches_sklearn`` pins it to the reference implementation.

    Returns:
        ``(covariance, shrinkage)`` with shrinkage in ``[0, 1]``; 0 is the plain empirical covariance
        and 1 is a pure scaled identity (which makes LDA the raw centroid-difference direction).
    """
    n, p = centered.shape
    emp = centered.T @ centered / n
    mu = np.trace(emp) / p

    # d^2: how far the empirical covariance sits from the shrinkage target.
    delta = ((emp - mu * np.eye(p)) ** 2).sum() / p

    # b_bar^2: the estimation error in `emp` itself. Uses the identity
    #   sum_k ||x_k x_k' - S||_F^2 = sum_k ||x_k||^4 - n ||S||_F^2
    # which avoids forming n separate outer products.
    fourth = ((centered ** 2).sum(axis=1) ** 2).sum()
    beta_bar = (fourth / n - (emp ** 2).sum()) / (n * p)

    if delta <= 0:
        return emp, 0.0
    shrinkage = float(np.clip(min(beta_bar, delta) / delta, 0.0, 1.0))
    return (1.0 - shrinkage) * emp + shrinkage * mu * np.eye(p), shrinkage


def fit_discriminant(
    features: np.ndarray, labels: np.ndarray
) -> "tuple[np.ndarray, float, float]":
    """Shrunken-LDA hyperplane separating ``labels == 1`` (crispant) from ``labels == 0``.

    Returns ``(w, b, shrinkage)``. Equal priors, so ``b`` places the plane at the centroid midpoint.

    This pools both classes' centered rows and shrinks the result **once**, which is the classical
    pooled-within-class covariance. ``sklearn``'s ``LinearDiscriminantAnalysis(shrinkage="auto")``
    instead shrinks each class separately and then prior-averages the two, so the two directions
    differ (~20 degrees on a 12-vs-12 example). Pooling first is the better choice here because it
    estimates one covariance from all ~22 rows rather than two from ~11 each -- the entire reason
    shrinkage is needed in the first place. ``ledoit_wolf_covariance`` itself matches sklearn exactly.
    """
    positive = features[labels == 1]
    negative = features[labels == 0]
    mean_positive = positive.mean(axis=0)
    mean_negative = negative.mean(axis=0)

    centered = np.vstack([positive - mean_positive, negative - mean_negative])
    covariance, shrinkage = ledoit_wolf_covariance(centered)

    difference = mean_positive - mean_negative
    weights = np.linalg.solve(covariance, difference)
    intercept = -float(weights @ (mean_positive + mean_negative) / 2.0)
    return weights, intercept, shrinkage


def signed_distance(features: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    """Signed Euclidean distance to the hyperplane, positive on the crispant side."""
    norm = np.linalg.norm(weights)
    if norm == 0:
        return np.zeros(len(features))
    return (features @ weights + intercept) / norm


def mahalanobis_separation(features: np.ndarray, labels: np.ndarray) -> float:
    """Shrunk Mahalanobis distance between the two class centroids."""
    positive = features[labels == 1]
    negative = features[labels == 0]
    difference = positive.mean(axis=0) - negative.mean(axis=0)
    centered = np.vstack([positive - positive.mean(axis=0), negative - negative.mean(axis=0)])
    covariance, _ = ledoit_wolf_covariance(centered)
    return float(np.sqrt(max(difference @ np.linalg.solve(covariance, difference), 0.0)))


def hotelling_t2(features: np.ndarray, labels: np.ndarray) -> "tuple[float, float, float]":
    """Classical two-sample Hotelling T^2 on the *unshrunk* pooled covariance.

    Returns ``(T2, F, p)``. Parametric and normality-dependent, so it is reported as a cross-check
    on the permutation test rather than as the primary evidence. Returns NaNs when the pooled
    covariance is singular or the df are exhausted.
    """
    positive = features[labels == 1]
    negative = features[labels == 0]
    n1, n0 = len(positive), len(negative)
    p = features.shape[1]
    df2 = n1 + n0 - p - 1
    if df2 <= 0:
        return np.nan, np.nan, np.nan

    difference = positive.mean(axis=0) - negative.mean(axis=0)
    pooled = (
        (n1 - 1) * np.cov(positive, rowvar=False) + (n0 - 1) * np.cov(negative, rowvar=False)
    ) / (n1 + n0 - 2)
    try:
        quadratic = float(difference @ np.linalg.solve(pooled, difference))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    t2 = (n1 * n0) / (n1 + n0) * quadratic
    f_statistic = t2 * df2 / (p * (n1 + n0 - 2))
    return t2, f_statistic, float(stats.f.sf(f_statistic, p, df2))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC for ``labels == 1`` against ``labels == 0``; ties averaged."""
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def leave_one_out_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Score every embryo with a discriminant fit on the other n-1.

    Each fold produces its own ``w``, so the raw distances live on slightly different scales. Each
    held-out score is therefore divided by the *training* fold's pooled within-class SD, putting all
    of them in training-SD units before they are pooled into one AUC.
    """
    out = np.empty(len(features))
    for index in range(len(features)):
        mask = np.ones(len(features), dtype=bool)
        mask[index] = False
        if len(np.unique(labels[mask])) < 2:
            out[index] = np.nan
            continue
        weights, intercept, _ = fit_discriminant(features[mask], labels[mask])
        train_scores = signed_distance(features[mask], weights, intercept)
        variances = [
            train_scores[labels[mask] == group].var(ddof=1)
            for group in (0, 1)
            if (labels[mask] == group).sum() > 1
        ]
        scale = float(np.sqrt(np.mean(variances))) if variances else 1.0
        value = signed_distance(features[index : index + 1], weights, intercept)[0]
        out[index] = value / scale if scale > 0 else value
    return out


# ---------------------------------------------------------------------------
# One contrast
# ---------------------------------------------------------------------------


@dataclass
class ContrastAxis:
    """A single crispant-vs-matched-control discriminant and everything used to judge it."""

    label: str
    target: str
    temperature: float
    timepoint: float
    n_crispant: int
    n_control: int

    direction: np.ndarray          # unit vector in the shared subspace
    intercept: float
    shrinkage: float
    scores: pd.DataFrame           # per-embryo s, s_loo, label, identity columns

    separation: float = np.nan     # shrunk Mahalanobis between centroids
    hotelling_f: float = np.nan
    hotelling_p: float = np.nan
    loo_auc: float = np.nan
    perm_p_auc: float = np.nan
    perm_p_separation: float = np.nan
    log_sd_ratio: float = np.nan   # log(SD of s in crispants / SD in controls)
    perm_p_dispersion: float = np.nan
    null_auc_p95: float = np.nan   # the bar the observed LOO-AUC actually had to clear
    boot_angles: np.ndarray | None = None
    boot_rank_rho: np.ndarray | None = None       # ordinal stability of the embryo ranking
    boot_top_retention: np.ndarray | None = None  # does the severe end stay the severe end?
    random_rank_rho: np.ndarray | None = None     # data-geometry floor for the two above
    random_top_retention: np.ndarray | None = None
    loo_rank_rho: float = np.nan
    angle_to_centroid: float = np.nan
    angle_to_stage: float = np.nan
    angle_to_cohort_pc1: float = np.nan
    angle_to_cohort_pc2: float = np.nan
    metadata: dict = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return self.n_crispant + self.n_control


def contrast_label(target: str, temperature: float, timepoint: float) -> str:
    """``('atf6', 35, 24)`` -> ``atf6 vs ctrl | 35C | 24hpf``."""
    return f"{target} vs ctrl | {int(float(temperature))}C | {int(float(timepoint))}hpf"


def shared_columns(frame: pd.DataFrame, n_dims: int = N_SHARED_DIMS) -> list[str]:
    return [column for column in ca.GLOBAL_COLUMNS if column in frame.columns][:n_dims]


def log_sd_ratio(scores: np.ndarray, labels: np.ndarray) -> float:
    """``log(SD of s among crispants / SD among controls)``.

    The mosaic-F0 premise made quantitative: injected clutches mixing effective nulls with escapers
    should be more dispersed than their controls along the axis that separates them, giving a
    positive value. Logged so that the null is symmetric about 0 and the permutation test is not
    biased by which group happens to sit in the denominator.
    """
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        return np.nan
    sd_positive = positive.std(ddof=1)
    sd_negative = negative.std(ddof=1)
    if sd_positive <= 0 or sd_negative <= 0:
        return np.nan
    return float(np.log(sd_positive / sd_negative))


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """Acute angle in degrees between two directions, sign-agnostic (0-90)."""
    a = np.asarray(first, float)
    b = np.asarray(second, float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return np.nan
    cosine = abs(float(a @ b) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def enumerate_contrasts(scores: pd.DataFrame) -> list[tuple]:
    """Every (target, temperature, timepoint) with both a crispant group and matched controls."""
    out = []
    for keys, stratum in scores.groupby(list(CONTRAST_KEYS), sort=True):
        temperature, timepoint = keys
        controls = stratum.loc[stratum["perturbation_group"] == CONTROL_GROUP]
        if controls.empty:
            continue
        for target in sorted(stratum["perturbation_group"].unique()):
            if target == CONTROL_GROUP:
                continue
            if (stratum["perturbation_group"] == target).sum() < 4:
                continue
            out.append((target, temperature, timepoint))
    return out


def fit_contrast(
    scores: pd.DataFrame,
    *,
    target: str,
    temperature: float,
    timepoint: float,
    n_dims: int = N_SHARED_DIMS,
    n_permutations: int = N_PERMUTATIONS,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = RANDOM_SEED,
) -> ContrastAxis:
    """Fit and fully characterise one crispant-vs-control discriminant."""
    columns = shared_columns(scores, n_dims)
    stratum = scores.loc[
        (scores["temperature"] == temperature) & (scores["timepoint_seq"] == timepoint)
    ]
    members = stratum.loc[
        stratum["perturbation_group"].isin([target, CONTROL_GROUP])
    ].reset_index(drop=True)

    features = members.loc[:, columns].to_numpy(float)
    labels = (members["perturbation_group"] == target).to_numpy().astype(int)

    weights, intercept, shrinkage = fit_discriminant(features, labels)
    distances = signed_distance(features, weights, intercept)
    loo = leave_one_out_scores(features, labels)

    frame = members.loc[
        :,
        [c for c in ("well_id", "physical_embryo_id", "experiment_id", "snip_id", "seq_sample_id",
                     "target", "perturbation_group") if c in members.columns],
    ].copy()
    frame["is_crispant"] = labels
    frame["s"] = distances
    # Standardised within contrast so coefficients are comparable across the 36 fits.
    frame["s_z"] = (distances - distances.mean()) / (distances.std(ddof=1) or 1.0)
    frame["s_loo"] = loo

    axis = ContrastAxis(
        label=contrast_label(target, temperature, timepoint),
        target=target,
        temperature=float(temperature),
        timepoint=float(timepoint),
        n_crispant=int(labels.sum()),
        n_control=int((labels == 0).sum()),
        direction=weights / (np.linalg.norm(weights) or 1.0),
        intercept=intercept,
        shrinkage=shrinkage,
        scores=frame,
    )

    axis.separation = mahalanobis_separation(features, labels)
    _, axis.hotelling_f, axis.hotelling_p = hotelling_t2(features, labels)
    axis.loo_auc = auc(loo, labels)
    axis.log_sd_ratio = log_sd_ratio(distances, labels)

    (axis.perm_p_auc, axis.perm_p_separation, axis.perm_p_dispersion,
     axis.null_auc_p95) = permutation_test(
        features, labels, observed_auc=axis.loo_auc, observed_separation=axis.separation,
        observed_log_sd_ratio=axis.log_sd_ratio,
        n_permutations=n_permutations, seed=seed,
    )
    (axis.boot_angles, axis.boot_rank_rho, axis.boot_top_retention,
     replicate_ranks) = bootstrap_axis(
        features, labels, reference=axis.direction, reference_scores=distances,
        n_bootstraps=n_bootstraps, seed=seed,
    )
    axis.metadata["bootstrap_ranks"] = replicate_ranks
    axis.random_rank_rho, axis.random_top_retention = random_rank_floor(features, seed=seed)
    axis.loo_rank_rho = loo_rank_agreement(distances, loo)

    # lambda = 1 comparison: if the covariance correction barely moves the direction, the whole
    # small-n covariance question is moot for this contrast.
    centroid_direction = features[labels == 1].mean(axis=0) - features[labels == 0].mean(axis=0)
    axis.angle_to_centroid = angle_between(axis.direction, centroid_direction)

    return axis


def permutation_test(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    observed_auc: float,
    observed_separation: float,
    observed_log_sd_ratio: float = np.nan,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> "tuple[float, float, float]":
    """Label-permutation p-values for LOO-AUC, centroid separation, and the dispersion ratio.

    Shuffling the crispant/control labels destroys any real class structure while preserving the
    morphology cloud exactly, so it needs no normality assumption -- which matters at 11 per group.
    The AUC null is recomputed with the *full* leave-one-out procedure each time, so it calibrates
    the same statistic that is reported rather than an in-sample stand-in.

    The dispersion null matters as much as the others and is easy to omit: because the discriminant
    is *refit* on every permuted labelling, this null answers "would an arbitrary 11-vs-12 split of
    these same embryos show one side more spread out than the other, once a discriminant is fit to
    it?" -- which is the objection the dispersion figure would otherwise invite.

    p-values use the ``(hits + 1) / (n + 1)`` convention, which cannot return 0.
    """
    rng = np.random.RandomState(seed)
    hits_auc = 0
    hits_separation = 0
    hits_dispersion = 0
    null_aucs = []
    for _ in range(n_permutations):
        permuted = rng.permutation(labels)
        if len(np.unique(permuted)) < 2:
            continue
        null_auc = auc(leave_one_out_scores(features, permuted), permuted)
        null_aucs.append(null_auc)
        if null_auc >= observed_auc:
            hits_auc += 1
        if mahalanobis_separation(features, permuted) >= observed_separation:
            hits_separation += 1
        if np.isfinite(observed_log_sd_ratio):
            weights, intercept, _ = fit_discriminant(features, permuted)
            null_ratio = log_sd_ratio(
                signed_distance(features, weights, intercept), permuted
            )
            if np.isfinite(null_ratio) and null_ratio >= observed_log_sd_ratio:
                hits_dispersion += 1
    return (
        (hits_auc + 1) / (n_permutations + 1),
        (hits_separation + 1) / (n_permutations + 1),
        (hits_dispersion + 1) / (n_permutations + 1) if np.isfinite(observed_log_sd_ratio) else np.nan,
        float(np.percentile(null_aucs, 95)) if null_aucs else np.nan,
    )


def bootstrap_axis(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    reference: np.ndarray,
    reference_scores: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = RANDOM_SEED,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Bootstrap the fit and measure instability two ways: angular and ordinal.

    Resampling is stratified within class so both groups are always represented.

    **Angular** -- the angle between each bootstrap direction and the full-data direction. This is
    the property of the *estimator*.

    **Ordinal** -- each bootstrap direction is used to re-score every original embryo, and the
    resulting ordering is Spearman-correlated with the full-data ordering. This is the property of
    the *output*, and it is the one with downstream consequences: a regression on ``s`` cares only
    about where the embryos land relative to one another, not about where the normal vector points.

    The two can diverge sharply, and the reason is geometric. The angle is measured in the ambient
    Euclidean metric, but the embryos do not fill that space -- the cloud is anisotropic, so a
    direction perturbation that lies mostly orthogonal to the cloud's spread barely moves any score.
    A 45-degree wobble can leave the ordering essentially untouched.

    That cuts both ways, which is why :func:`random_rank_floor` is not optional: in a cloud dominated
    by one direction, *any* direction produces a similar ordering, so a high rank correlation can be
    a property of the data geometry rather than evidence that the axis is determined.

    Returns:
        ``(angles_degrees, spearman_rho, top_tercile_retention, replicate_ranks)`` -- the third is,
        for each bootstrap, the fraction of the full-data top-tercile embryos that stay in the
        bootstrap's top tercile; the fourth is the ``(n_bootstraps, n_embryos)`` matrix of ranks,
        kept so the churn can be drawn rather than only summarised.
    """
    rng = np.random.RandomState(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)

    cut = np.quantile(reference_scores, 2.0 / 3.0)
    reference_top = reference_scores >= cut

    angles, rhos, retention, replicate_ranks = [], [], [], []
    for _ in range(n_bootstraps):
        picks = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        try:
            weights, intercept, _ = fit_discriminant(features[picks], labels[picks])
        except np.linalg.LinAlgError:
            continue
        angles.append(angle_between(weights, reference))

        replicate = signed_distance(features, weights, intercept)
        rhos.append(float(stats.spearmanr(replicate, reference_scores).statistic))
        replicate_top = replicate >= np.quantile(replicate, 2.0 / 3.0)
        retention.append(float((replicate_top & reference_top).sum() / max(reference_top.sum(), 1)))
        replicate_ranks.append(np.argsort(np.argsort(replicate)))

    return (
        np.asarray(angles, dtype=float),
        np.asarray(rhos, dtype=float),
        np.asarray(retention, dtype=float),
        np.asarray(replicate_ranks, dtype=int),
    )


def random_rank_floor(
    features: np.ndarray, *, n_draws: int = 400, seed: int = RANDOM_SEED
) -> "tuple[np.ndarray, np.ndarray]":
    """Ordering agreement between two *arbitrary* directions applied to these same embryos.

    The floor that every rank-stability number has to be read against, and it is emphatically not 0.
    If the embryo cloud is dominated by one or two directions -- which after a PCA it is, by
    construction -- then two unrelated directions both largely recover that dominant axis and their
    orderings agree substantially. Measuring rank stability without this reference would make a
    completely undetermined axis look reproducible.

    Returns ``(spearman_rho, top_tercile_retention)`` over random direction pairs, matching the
    statistics :func:`bootstrap_axis` returns.
    """
    rng = np.random.RandomState(seed)
    directions = rng.normal(size=(2 * n_draws, features.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    rhos, retention = [], []
    for index in range(n_draws):
        first = features @ directions[2 * index]
        second = features @ directions[2 * index + 1]
        rhos.append(abs(float(stats.spearmanr(first, second).statistic)))
        # Orientation of a random direction is arbitrary, so take the better-matching sign.
        top_first = first >= np.quantile(first, 2.0 / 3.0)
        best = 0.0
        for candidate in (second, -second):
            top = candidate >= np.quantile(candidate, 2.0 / 3.0)
            best = max(best, (top & top_first).sum() / max(top_first.sum(), 1))
        retention.append(float(best))
    return np.asarray(rhos), np.asarray(retention)


def loo_rank_agreement(scores: np.ndarray, loo_scores: np.ndarray) -> float:
    """Spearman between the in-sample ordering and the leave-one-out ordering.

    How much of the ordering survives each embryo not having contributed to its own placement. Note
    the folds share n-1 of n embryos, so this is optimistic in absolute terms -- it is useful as a
    *relative* measure across contrasts, not as a standalone pass/fail.
    """
    if len(scores) < 3 or not np.all(np.isfinite(loo_scores)):
        return np.nan
    return float(stats.spearmanr(scores, loo_scores).statistic)


def random_cosine_reference(
    n_dims: int = N_SHARED_DIMS, *, n_draws: int = 20000, seed: int = RANDOM_SEED
) -> np.ndarray:
    """``|cos|`` between pairs of random unit vectors in ``n_dims`` dimensions.

    The floor that every direction-similarity number has to be read against, and it is not 0. In
    ``p`` dimensions the density of ``cos`` is proportional to ``(1 - c^2)^((p-3)/2)``, giving
    ``E|cos| = 0.375`` at p=5 -- so two entirely unrelated 5D directions already "agree" to 0.375,
    and a measured similarity of 0.4 means nothing at all.
    """
    rng = np.random.RandomState(seed)
    a = rng.normal(size=(n_draws, n_dims))
    b = rng.normal(size=(n_draws, n_dims))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.abs((a * b).sum(axis=1))


def random_angle_reference(
    n_dims: int = N_SHARED_DIMS, *, n_draws: int = 20000, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Acute angles (degrees) between pairs of random unit vectors in ``n_dims`` dimensions.

    The reference every stability number has to be read against: in 5D the median acute angle
    between two arbitrary directions is ~70 degrees, not 90, so a bootstrap spread reaching
    "60 degrees" is far less reassuring than it sounds.
    """
    cosine = random_cosine_reference(n_dims, n_draws=n_draws, seed=seed)
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Reference directions for interpretation
# ---------------------------------------------------------------------------


def stage_axes(scores: pd.DataFrame, *, n_dims: int = N_SHARED_DIMS) -> dict[float, np.ndarray]:
    """The developmental-stage direction at each temperature, from controls only.

    Controls are sampled at three timepoints per temperature, so an OLS regression of the shared
    coordinates on collection hpf recovers the direction morphology moves as development proceeds --
    estimated entirely from unperturbed embryos, and therefore usable as an independent yardstick.
    """
    columns = shared_columns(scores, n_dims)
    out = {}
    controls = scores.loc[scores["perturbation_group"] == CONTROL_GROUP]
    for temperature, block in controls.groupby("temperature", sort=True):
        hours = pd.to_numeric(block["timepoint_seq"], errors="coerce").to_numpy(float)
        if len(np.unique(hours)) < 2:
            continue
        design = np.column_stack([np.ones(len(hours)), hours])
        coefficients, *_ = np.linalg.lstsq(design, block.loc[:, columns].to_numpy(float), rcond=None)
        direction = coefficients[1]
        norm = np.linalg.norm(direction)
        if norm > 0:
            out[float(temperature)] = direction / norm
    return out


def attach_reference_angles(
    axes: "list[ContrastAxis]",
    scores: pd.DataFrame,
    *,
    n_dims: int = N_SHARED_DIMS,
) -> None:
    """Fill in each contrast's angle to the stage axis and to its cohort's unsupervised PCs.

    Mutates ``axes`` in place. The cohort PCs are refit here from the crispant members alone, in the
    same subspace, so the comparison is like-for-like with the discriminant.
    """
    from sklearn.decomposition import PCA

    columns = shared_columns(scores, n_dims)
    stage = stage_axes(scores, n_dims=n_dims)

    for axis in axes:
        if axis.temperature in stage:
            axis.angle_to_stage = angle_between(axis.direction, stage[axis.temperature])

        members = scores.loc[
            (scores["perturbation_group"] == axis.target)
            & (scores["temperature"] == axis.temperature)
            & (scores["timepoint_seq"] == axis.timepoint)
        ]
        if len(members) <= 2:
            continue
        matrix = members.loc[:, columns].to_numpy(float)
        pca = PCA(n_components=min(2, min(matrix.shape) - 1), random_state=RANDOM_SEED).fit(matrix)
        axis.angle_to_cohort_pc1 = angle_between(axis.direction, pca.components_[0])
        if len(pca.components_) > 1:
            axis.angle_to_cohort_pc2 = angle_between(axis.direction, pca.components_[1])


# ---------------------------------------------------------------------------
# Batch driver and tabular outputs
# ---------------------------------------------------------------------------


def fit_all_contrasts(
    scores: pd.DataFrame,
    *,
    n_dims: int = N_SHARED_DIMS,
    n_permutations: int = N_PERMUTATIONS,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> list[ContrastAxis]:
    """Fit every matched contrast, then attach the interpretive angles."""
    axes = []
    for target, temperature, timepoint in enumerate_contrasts(scores):
        axis = fit_contrast(
            scores,
            target=target,
            temperature=temperature,
            timepoint=timepoint,
            n_dims=n_dims,
            n_permutations=n_permutations,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        axes.append(axis)
        if verbose:
            print(
                f"  {axis.label:<34s} n={axis.n_total:2d}  "
                f"AUC={axis.loo_auc:.3f} (p={axis.perm_p_auc:.3f})  "
                f"sep={axis.separation:.2f}  shrink={axis.shrinkage:.2f}"
            )
    attach_reference_angles(axes, scores, n_dims=n_dims)
    return axes


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values, NaN-safe."""
    values = np.asarray(p_values, float)
    finite = np.flatnonzero(np.isfinite(values))
    out = np.full(len(values), np.nan)
    if finite.size == 0:
        return out
    ordered = finite[np.argsort(values[finite])]
    n = len(ordered)
    adjusted = values[ordered] * n / np.arange(1, n + 1)
    out[ordered] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return out


def contrast_summary(axes: "list[ContrastAxis]", *, n_dims: int = N_SHARED_DIMS) -> pd.DataFrame:
    """One row per contrast: separation, generalisation, stability, and the confound angles."""
    reference = random_angle_reference(n_dims)
    random_p05 = float(np.percentile(reference, 5))
    random_median = float(np.median(reference))

    rows = []
    for axis in axes:
        boot = axis.boot_angles if axis.boot_angles is not None else np.array([np.nan])
        rank = axis.boot_rank_rho if axis.boot_rank_rho is not None else np.array([np.nan])
        retain = axis.boot_top_retention if axis.boot_top_retention is not None else np.array([np.nan])
        floor_rank = axis.random_rank_rho if axis.random_rank_rho is not None else np.array([np.nan])
        floor_retain = (
            axis.random_top_retention if axis.random_top_retention is not None else np.array([np.nan])
        )
        rows.append(
            {
                "contrast": axis.label,
                "target": axis.target,
                "temperature": axis.temperature,
                "timepoint": axis.timepoint,
                "n_crispant": axis.n_crispant,
                "n_control": axis.n_control,
                "shrinkage": axis.shrinkage,
                "separation": axis.separation,
                "hotelling_f": axis.hotelling_f,
                "hotelling_p": axis.hotelling_p,
                "loo_auc": axis.loo_auc,
                "perm_p_auc": axis.perm_p_auc,
                "perm_p_separation": axis.perm_p_separation,
                "log_sd_ratio": axis.log_sd_ratio,
                "perm_p_dispersion": axis.perm_p_dispersion,
                "null_auc_p95": axis.null_auc_p95,
                "boot_angle_median": float(np.nanmedian(boot)),
                "boot_angle_p95": float(np.nanpercentile(boot, 95)),
                "random_angle_p05": random_p05,
                "random_angle_median": random_median,
                # Ordinal stability: does the EMBRYO ORDERING survive resampling? Each is paired
                # with its own data-geometry floor, since an anisotropic cloud gives even arbitrary
                # directions substantial ordering agreement.
                "rank_rho_median": float(np.nanmedian(rank)),
                "rank_rho_p05": float(np.nanpercentile(rank, 5)),
                "random_rank_rho_median": float(np.nanmedian(floor_rank)),
                # One-sided p against the arbitrary-direction null: what fraction of random
                # direction pairs order these same embryos at least as consistently as the
                # bootstrap does. Compares like with like -- median against the full null
                # distribution -- rather than pitting opposite tails against each other.
                "rank_p_vs_floor": float(np.mean(floor_rank >= np.nanmedian(rank))),
                "top_retention_median": float(np.nanmedian(retain)),
                "random_top_retention_median": float(np.nanmedian(floor_retain)),
                "loo_rank_rho": axis.loo_rank_rho,
                "angle_to_centroid": axis.angle_to_centroid,
                "angle_to_stage": axis.angle_to_stage,
                "angle_to_cohort_pc1": axis.angle_to_cohort_pc1,
                "angle_to_cohort_pc2": axis.angle_to_cohort_pc2,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q_auc"] = benjamini_hochberg(frame["perm_p_auc"].to_numpy())
        frame["q_separation"] = benjamini_hochberg(frame["perm_p_separation"].to_numpy())
        frame["q_dispersion"] = benjamini_hochberg(frame["perm_p_dispersion"].to_numpy())
        # "Usable" = generalises AND is better-determined than an arbitrary direction would be.
        frame["usable"] = (frame["q_auc"] < 0.1) & (frame["boot_angle_p95"] < random_median)
        # The ordinal counterpart of the angular gate: the embryo ORDERING has to be more
        # reproducible than what this cloud's geometry hands you from an arbitrary direction.
        frame["q_rank"] = benjamini_hochberg(frame["rank_p_vs_floor"].to_numpy())
        frame["rank_stable"] = frame["q_rank"] < 0.1
        frame["usable_ordinal"] = (frame["q_auc"] < 0.1) & frame["rank_stable"]
        # Flag rather than exclude: a discriminant close to the control stage direction may be a
        # developmental-delay score, so any transcriptional hit would be a stage-composition shift.
        frame["stage_confounded"] = frame["angle_to_stage"] < 35.0
    return frame


def contrast_scores(axes: "list[ContrastAxis]") -> pd.DataFrame:
    """Long-form per-embryo scores across all contrasts -- the PLN regression input."""
    blocks = []
    for axis in axes:
        block = axis.scores.copy()
        block.insert(0, "contrast", axis.label)
        block.insert(1, "contrast_target", axis.target)
        block.insert(2, "temperature", axis.temperature)
        block.insert(3, "timepoint", axis.timepoint)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def contrast_directions(axes: "list[ContrastAxis]", *, n_dims: int = N_SHARED_DIMS) -> pd.DataFrame:
    """Long-form unit loading vectors over the shared dimensions."""
    rows = []
    for axis in axes:
        for index, loading in enumerate(axis.direction[:n_dims]):
            rows.append(
                {
                    "contrast": axis.label,
                    "target": axis.target,
                    "temperature": axis.temperature,
                    "timepoint": axis.timepoint,
                    "shared_dim": ca.GLOBAL_COLUMNS[index],
                    "loading": float(loading),
                }
            )
    return pd.DataFrame(rows)


def direction_similarity(axes: "list[ContrastAxis]") -> pd.DataFrame:
    """Pairwise ``|cos angle|`` between every pair of discriminant directions.

    Asks whether a target keeps the same morphological signature across temperatures and
    timepoints, and whether different targets are distinguishable from one another. In 5D the
    expected value for *unrelated* directions is 0.375, not 0 -- read the matrix against that floor.
    Use :func:`direction_similarity_summary` for the within-target vs between-target comparison,
    which is the question the matrix is really there to answer.
    """
    labels = [axis.label for axis in axes]
    matrix = np.eye(len(axes))
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            cosine = abs(float(axes[i].direction @ axes[j].direction))
            matrix[i, j] = matrix[j, i] = cosine
    return pd.DataFrame(matrix, index=labels, columns=labels)


def direction_similarity_summary(
    axes: "list[ContrastAxis]", *, n_dims: int = N_SHARED_DIMS, restrict_usable: pd.Series | None = None
) -> pd.DataFrame:
    """Within-target vs between-target direction agreement, against the random-direction floor.

    The matrix is hard to read by eye, and the eye is unreliable here because the floor is 0.375
    rather than 0. This reduces it to the comparison that matters: does a target's discriminant point
    the same way across temperatures and timepoints (which would mean a reproducible gene-specific
    morphological signature), and is that more than any two contrasts share by chance?

    ``restrict_usable`` optionally limits the comparison to contrasts whose axis passed the gates --
    directions from failing contrasts are fit to noise and would dilute both groups toward the floor.
    """
    floor = random_cosine_reference(n_dims)
    keep = axes
    if restrict_usable is not None:
        usable = set(restrict_usable.loc[restrict_usable].index)
        keep = [axis for axis in axes if axis.label in usable]

    within, between = [], []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            value = abs(float(keep[i].direction @ keep[j].direction))
            (within if keep[i].target == keep[j].target else between).append(value)

    rows = [
        {"comparison": "within target", "n_pairs": len(within), "mean_abs_cos": np.mean(within)},
        {"comparison": "between targets", "n_pairs": len(between), "mean_abs_cos": np.mean(between)},
        {"comparison": "random 5D floor", "n_pairs": len(floor), "mean_abs_cos": float(floor.mean())},
    ]
    frame = pd.DataFrame(rows)
    frame["mean_abs_cos"] = frame["mean_abs_cos"].astype(float)
    frame["n_contrasts"] = len(keep)
    if within and between:
        frame.loc[0, "p_vs_between"] = float(stats.mannwhitneyu(within, between,
                                                                alternative="greater")[1])
        frame.loc[0, "p_vs_random"] = float(stats.mannwhitneyu(within, floor,
                                                               alternative="greater")[1])
        frame.loc[1, "p_vs_random"] = float(stats.mannwhitneyu(between, floor,
                                                               alternative="greater")[1])
    return frame


# ---------------------------------------------------------------------------
# Image strips
# ---------------------------------------------------------------------------


def contrast_image_strip(axis: ContrastAxis, *, masked: bool = False) -> pd.DataFrame:
    """Every embryo in the contrast, ordered by ``s``, with its snip path resolved.

    All members are returned rather than an even sample: at n~23 the whole contrast fits on two rows,
    and seeing where the controls actually land in the ordering is the point of the figure.
    """
    ordered = axis.scores.sort_values("s").reset_index(drop=True)
    ordered["image_path"] = [
        str(
            ca.snip_image_path(
                row.well_id,
                row.physical_embryo_id,
                experiment_id=getattr(row, "experiment_id", None),
                masked=masked,
            )
        )
        for row in ordered.itertuples()
    ]
    ordered["image_exists"] = [Path(path).is_file() for path in ordered["image_path"]]
    ordered["contrast"] = axis.label
    return ordered
