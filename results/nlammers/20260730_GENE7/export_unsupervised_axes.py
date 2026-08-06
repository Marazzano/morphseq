"""Unsupervised morphology axes — how much could we find with NO label information?

Deliberately separate from the supervised pipeline (`export_edger_inputs.py` /
`fit_edger_contrasts.R`). Nothing here reads a crispant/control label except to *define which
embryos belong to a contrast*; the axes themselves are pure unsupervised PCA.

Two questions, one axis each:

``pooled_pc1``
    PC1 of the pooled control + crispant cloud for a contrast. If the perturbation is the dominant
    source of morphological variation in that cell, PC1 should recover it without ever being told
    which embryos were injected. If stage, batch or mounting dominates instead, PC1 will find that
    and the perturbation will be invisible.

``within_pc1``
    PC1 of the crispant embryos alone (n ~ 11). The label-free counterpart of the supervised
    within-crispant dose slope.

Both are computed in the same 5D shared subspace as everything else, so results are commensurable
with the supervised analysis.

Two properties of PC1 that shape how the results must be read
-------------------------------------------------------------
**Sign is arbitrary.** An eigenvector is defined up to a flip, so with no labels there is no way to
know which end is "more perturbed". The recovered *set* of cell types is unaffected (a sign flip
negates every coefficient and leaves every p-value alone), so recovery fractions are honestly
label-free. Direction agreement is not: it needs one bit of label information, applied downstream as
a single global flip per contrast and reported as such. ``_orient`` fixes a deterministic
convention here purely for reproducibility.

**``within_pc1`` is only defined for crispants**, so that regression runs on ~11 embryos with no
controls, whereas the binary reference used ~23. Recovery fractions between the two analyses are not
strictly comparable, and both references are reported downstream.

Significance of the axis itself is assessed by a column-shuffle null: permuting each shared
dimension independently across embryos destroys the covariance structure while preserving every
marginal, so the null answers "is PC1 concentrating more variance than an uncorrelated cloud with
these same spreads would?"

Writes to ``data/unsupervised/``:

    unsupervised_predictors.csv   one row per (contrast, embryo): the two PC1 scores
    unsupervised_axes.csv         per (contrast, axis): variance ratio, null p, stability, angles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cohort_axes as ca  # noqa: E402
import lda_contrasts as lc  # noqa: E402

DATA = HERE / "data"
OUT = DATA / "unsupervised"

N_PERMUTATIONS = 2000
N_BOOTSTRAPS = 500
SEED = 42


def _orient(component: np.ndarray, scores: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministic sign convention: largest-magnitude loading positive.

    This is bookkeeping, not information — it makes reruns reproducible. It cannot recover the
    biologically meaningful orientation, because nothing label-free can.
    """
    dominant = int(np.argmax(np.abs(component)))
    if component[dominant] < 0:
        return -component, -scores
    return component, scores


def _column_shuffle_null(matrix: np.ndarray, *, n_permutations: int, seed: int) -> np.ndarray:
    """Null distribution of PC1's explained-variance ratio under destroyed covariance.

    Each column is permuted independently, so every dimension keeps its marginal spread but the
    correlations between them are broken. PC1 of such a cloud still concentrates more than 1/p of
    the variance — at n~22 in 5D, appreciably more — which is exactly why the comparison has to be
    made against this null rather than against 20%.
    """
    rng = np.random.RandomState(seed)
    ratios = []
    for _ in range(n_permutations):
        shuffled = np.column_stack([rng.permutation(column) for column in matrix.T])
        pca = PCA(n_components=1).fit(shuffled)
        total = shuffled.var(axis=0, ddof=1).sum()
        ratios.append(pca.explained_variance_[0] / total if total > 0 else np.nan)
    return np.asarray(ratios, dtype=float)


def _bootstrap_stability(matrix: np.ndarray, reference: np.ndarray, *,
                         n_bootstraps: int, seed: int) -> np.ndarray:
    """Angles between bootstrap-refit PC1 directions and the full-data PC1."""
    rng = np.random.RandomState(seed)
    angles = []
    for _ in range(n_bootstraps):
        sample = matrix[rng.choice(len(matrix), size=len(matrix), replace=True)]
        try:
            pca = PCA(n_components=1).fit(sample)
        except Exception:
            continue
        angles.append(lc.angle_between(pca.components_[0], reference))
    return np.asarray(angles, dtype=float)


def _fit_axis(matrix: np.ndarray, *, n_permutations: int, n_bootstraps: int,
              seed: int) -> "tuple[np.ndarray, dict]":
    """PC1 scores plus its diagnostics."""
    pca = PCA(n_components=min(2, min(matrix.shape) - 1)).fit(matrix)
    component, scores = _orient(pca.components_[0], pca.transform(matrix)[:, 0])
    total = matrix.var(axis=0, ddof=1).sum()
    ratio = pca.explained_variance_[0] / total if total > 0 else np.nan

    null = _column_shuffle_null(matrix, n_permutations=n_permutations, seed=seed)
    angles = _bootstrap_stability(matrix, component, n_bootstraps=n_bootstraps, seed=seed)
    diagnostics = {
        "variance_ratio": float(ratio),
        "null_ratio_mean": float(np.nanmean(null)),
        "null_ratio_p95": float(np.nanpercentile(null, 95)),
        "p_variance": float((np.nansum(null >= ratio) + 1) / (len(null) + 1)),
        "boot_angle_median": float(np.nanmedian(angles)),
        "boot_angle_p95": float(np.nanpercentile(angles, 95)),
        "gap_to_pc2": float(
            pca.explained_variance_[0] / pca.explained_variance_[1]
        ) if len(pca.explained_variance_) > 1 and pca.explained_variance_[1] > 0 else np.nan,
    }
    return component, scores, diagnostics


def build(*, n_permutations: int = N_PERMUTATIONS, n_bootstraps: int = N_BOOTSTRAPS,
          seed: int = SEED) -> "tuple[pd.DataFrame, pd.DataFrame]":
    scores_frame = pd.read_csv(DATA / "gene7_global_scores.csv")
    covariates = pd.read_csv(DATA / "morph_covariates.csv")
    supervised = pd.read_csv(DATA / "lda" / "contrast_scores.csv")
    axis_quality = pd.read_csv(DATA / "lda" / "contrast_summary.csv")
    directions = pd.read_csv(DATA / "lda" / "contrast_directions.csv")

    columns = lc.shared_columns(scores_frame)
    bridge = covariates.loc[:, ["well_id", "sample"]].drop_duplicates()
    random_angles = lc.random_angle_reference(len(columns))
    random_median = float(np.median(random_angles))

    predictor_rows, axis_rows = [], []
    for (target, temperature, timepoint) in lc.enumerate_contrasts(scores_frame):
        label = lc.contrast_label(target, temperature, timepoint)
        stratum = scores_frame.loc[
            (scores_frame["temperature"] == temperature)
            & (scores_frame["timepoint_seq"] == timepoint)
        ]
        members = stratum.loc[
            stratum["perturbation_group"].isin([target, lc.CONTROL_GROUP])
        ].reset_index(drop=True)
        is_crispant = (members["perturbation_group"] == target).to_numpy().astype(int)
        matrix = members.loc[:, columns].to_numpy(float)

        supervised_direction = directions.loc[directions["contrast"] == label, "loading"].to_numpy()

        block = members.loc[:, ["well_id"]].copy()
        block["contrast"] = label
        block["contrast_target"] = target
        block["temperature"] = float(temperature)
        block["timepoint"] = float(timepoint)
        block["is_crispant"] = is_crispant

        # ---- pooled PC1: no labels used at all ----
        component, pooled_scores, diagnostics = _fit_axis(
            matrix, n_permutations=n_permutations, n_bootstraps=n_bootstraps, seed=seed
        )
        block["pooled_pc1"] = pooled_scores
        diagnostics.update({
            "contrast": label, "axis": "pooled_pc1", "target": target,
            "temperature": float(temperature), "timepoint": float(timepoint),
            "n_embryos": len(matrix), "random_angle_median": random_median,
            # How much of the supervised discriminant did an unsupervised axis stumble onto?
            "angle_to_lda": lc.angle_between(component, supervised_direction)
            if len(supervised_direction) == len(component) else np.nan,
        })
        axis_rows.append(diagnostics)

        # ---- within-crispant PC1: crispants only ----
        crispant_matrix = matrix[is_crispant == 1]
        block["within_pc1"] = np.nan
        if len(crispant_matrix) > 3:
            component_w, within_scores, diagnostics_w = _fit_axis(
                crispant_matrix, n_permutations=n_permutations, n_bootstraps=n_bootstraps,
                seed=seed,
            )
            block.loc[block["is_crispant"] == 1, "within_pc1"] = within_scores
            diagnostics_w.update({
                "contrast": label, "axis": "within_pc1", "target": target,
                "temperature": float(temperature), "timepoint": float(timepoint),
                "n_embryos": len(crispant_matrix), "random_angle_median": random_median,
                "angle_to_lda": lc.angle_between(component_w, supervised_direction)
                if len(supervised_direction) == len(component_w) else np.nan,
            })
            axis_rows.append(diagnostics_w)

        predictor_rows.append(block)

    predictors = pd.concat(predictor_rows, ignore_index=True).merge(bridge, on="well_id", how="left")
    predictors = predictors.loc[predictors["sample"].notna()].copy()

    # z-score within the set each axis is defined on, so coefficients are per-SD and comparable.
    for column in ("pooled_pc1", "within_pc1"):
        values = predictors.groupby("contrast")[column]
        predictors[f"{column}_z"] = (values.transform(lambda x: (x - x.mean()) / (x.std(ddof=1)
                                                                                 or 1.0)))

    # Carry the SUPERVISED axis quality alongside, so every unsupervised result can be conditioned
    # on whether that contrast had a detectable morphological phenotype in the first place.
    axes = pd.DataFrame(axis_rows).merge(
        axis_quality[["contrast", "loo_auc", "q_auc", "usable", "separation", "log_sd_ratio"]],
        on="contrast", how="left",
    )
    axes["axis_significant"] = axes["p_variance"] < 0.05
    axes["axis_stable"] = axes["boot_angle_p95"] < axes["random_angle_median"]
    return predictors, axes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=N_BOOTSTRAPS)
    arguments = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    predictors, axes = build(n_permutations=arguments.permutations,
                             n_bootstraps=arguments.bootstraps)
    predictors.to_csv(OUT / "unsupervised_predictors.csv", index=False)
    axes.to_csv(OUT / "unsupervised_axes.csv", index=False)

    print(f"wrote {len(predictors)} rows across {predictors['contrast'].nunique()} contrasts")
    for axis_name, block in axes.groupby("axis"):
        print(f"\n  {axis_name}  (n={len(block)} contrasts)")
        print(f"    PC1 variance ratio      : median {block['variance_ratio'].median():.3f} "
              f"(column-shuffle null {block['null_ratio_mean'].median():.3f})")
        print(f"    significant vs null     : {int(block['axis_significant'].sum())}/{len(block)}")
        print(f"    stable (better than random direction): "
              f"{int(block['axis_stable'].sum())}/{len(block)}")
        print(f"    angle to supervised LDA : median {block['angle_to_lda'].median():.0f} deg "
              f"(random would be {block['random_angle_median'].median():.0f})")
        overlap = block[block["usable"]]
        print(f"    of the {len(overlap)} contrasts with a USABLE supervised axis, "
              f"{int(overlap['axis_significant'].sum())} have a significant PC1")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
