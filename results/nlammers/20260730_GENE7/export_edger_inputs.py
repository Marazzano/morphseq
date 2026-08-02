"""Step 1 of the edgeR pipeline -- write the per-contrast predictors R will regress on.

Three arms per contrast, one predictor each, all with identical degrees of freedom so their fits are
directly comparable:

``binary``  the classical crispant/control indicator -- the baseline being beaten
``s``       signed distance normal to the shrunken-LDA hyperplane (the proposal)
``hinge``   ``max(s - c, 0)`` with ``c`` the control mean -- controls collapsed to one value, the
            crispant-side gradient retained

The hinge is the mosaic-F0 model written as a functional form: if escapers are genuinely
indistinguishable from wildtype then their spread along ``s`` is noise, and severity only means
something above threshold. ``c`` is the control mean rather than ``min(s)`` among crispants, which
would hang the entire predictor on a single embryo.

All three are z-scored **within contrast**. Coefficients are then per-SD-of-predictor and comparable
in magnitude across arms and contrasts. (The Mahalanobis statistic downstream is scale-invariant
anyway -- rescaling a predictor rescales beta and its standard error together -- but the raw
coefficients are easier to read this way, and it removes the obvious objection that ``s`` only wins
because it has more range than a 0/1 indicator.)

Writes to ``data/edger/``:

    contrast_predictors.csv    one row per (contrast, embryo): the three predictors + identity
    contrast_axis_quality.csv  each contrast's axis diagnostics, carried through for the notebook

Usage::

    python export_edger_inputs.py                 # all 36 contrasts
    python export_edger_inputs.py --usable-only   # only contrasts whose axis passed the gates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = HERE / "data"
OUT = DATA / "edger"

# Carried so the notebook can condition every regression result on how good the axis was.
AXIS_QUALITY_COLUMNS = (
    "contrast", "target", "temperature", "timepoint", "n_crispant", "n_control",
    "loo_auc", "perm_p_auc", "q_auc", "null_auc_p95", "separation", "hotelling_p",
    "boot_angle_p95", "random_angle_median", "rank_rho_median", "random_rank_rho_median",
    "q_rank", "top_retention_median", "log_sd_ratio", "angle_to_stage", "shrinkage",
    "usable", "usable_ordinal", "rank_stable", "stage_confounded",
)


def build_predictors(*, usable_only: bool = False) -> "tuple[pd.DataFrame, pd.DataFrame]":
    scores = pd.read_csv(DATA / "lda" / "contrast_scores.csv")
    summary = pd.read_csv(DATA / "lda" / "contrast_summary.csv")
    covariates = pd.read_csv(DATA / "morph_covariates.csv")

    # contrast_scores keys embryos by well_id; the CDS keys them by embryo_ID with the RT-block
    # suffix. morph_covariates carries both, so it is the bridge.
    bridge = covariates.loc[:, ["well_id", "sample"]].drop_duplicates()
    frame = scores.merge(bridge, on="well_id", how="left")
    missing = int(frame["sample"].isna().sum())
    if missing:
        print(f"  !! {missing} rows have no CDS sample id and will be dropped")
        frame = frame.loc[frame["sample"].notna()].copy()

    if usable_only:
        keep = set(summary.loc[summary["usable"], "contrast"])
        frame = frame.loc[frame["contrast"].isin(keep)].copy()

    blocks = []
    for contrast, block in frame.groupby("contrast", sort=True):
        block = block.copy()
        control_mean = block.loc[block["is_crispant"] == 0, "s"].mean()
        block["hinge"] = np.maximum(block["s"] - control_mean, 0.0)
        block["binary"] = block["is_crispant"].astype(float)
        block["hinge_threshold"] = control_mean

        # s split into the part that IS the group label and the part that is not. Subtracting each
        # embryo's own group mean leaves a column with zero mean inside both groups, hence exactly
        # orthogonal to the binary indicator. Without that centering the two predictors are
        # near-collinear -- s was built to separate the groups -- and both coefficients would carry
        # badly inflated standard errors.
        block["s_within"] = block["s"] - block.groupby("is_crispant")["s"].transform("mean")

        # Three-level split. Keeps every embryo, unlike dropping the wildtype-looking crispants,
        # which would delete observations conditional on a label-derived quantity and mechanically
        # widen the group gap.
        #
        # "Escaper" is defined as a crispant falling inside the ENTIRE observed control range, not
        # below the control mean. The mean is the right bend point for the hinge but the wrong
        # classifier: an embryo below the control mean is more control-like than the average
        # control, so that rule leaves a median of ONE escaper per contrast and nothing is
        # estimable. Even at this most permissive threshold the category stays thin, and for a
        # structural reason worth stating -- a contrast has a trustworthy axis precisely when the
        # groups separate cleanly, which is when few crispants land in control territory. Escaper-
        # rich and axis-trustworthy are in tension, so the three-level fit is only estimable on a
        # subset (see contrast_status.csv).
        control_ceiling = block.loc[block["is_crispant"] == 0, "s"].max()
        block["escaper_threshold"] = control_ceiling
        block["escaper_class"] = np.where(
            block["is_crispant"] == 0, "control",
            np.where(block["s"] < control_ceiling, "escaper", "severe"),
        )

        for column in ("binary", "s", "hinge", "s_within"):
            values = block[column].to_numpy(float)
            spread = values.std(ddof=1)
            block[f"{column}_z"] = (values - values.mean()) / (spread if spread > 0 else 1.0)
        # Re-centre within group after scaling: the global mean shift above would otherwise
        # reintroduce a tiny group component and break the orthogonality this column exists for.
        block["s_within_z"] = (
            block["s_within_z"] - block.groupby("is_crispant")["s_within_z"].transform("mean")
        )
        blocks.append(block)

    predictors = pd.concat(blocks, ignore_index=True)
    ordered = [
        "contrast", "contrast_target", "temperature", "timepoint", "sample", "well_id",
        "is_crispant", "escaper_class", "binary_z", "s_z", "hinge_z", "s_within_z",
        "s", "hinge", "s_within", "hinge_threshold", "s_loo",
    ]
    predictors = predictors.loc[:, [c for c in ordered if c in predictors.columns]]

    quality = summary.loc[:, [c for c in AXIS_QUALITY_COLUMNS if c in summary.columns]]
    quality = quality.loc[quality["contrast"].isin(set(predictors["contrast"]))]
    return predictors, quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usable-only", action="store_true",
                        help="restrict to contrasts whose axis cleared both gates")
    arguments = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    predictors, quality = build_predictors(usable_only=arguments.usable_only)

    predictors.to_csv(OUT / "contrast_predictors.csv", index=False)
    quality.to_csv(OUT / "contrast_axis_quality.csv", index=False)

    print(f"wrote {len(predictors)} rows across {predictors['contrast'].nunique()} contrasts")
    print(f"  embryos per contrast : {predictors.groupby('contrast').size().min()}"
          f"-{predictors.groupby('contrast').size().max()}")
    print(f"  distinct samples     : {predictors['sample'].nunique()}")
    print(f"  usable axes          : {int(quality['usable'].sum())} / {len(quality)}")

    # A degenerate hinge (all crispants below the control mean) would silently become a constant.
    flat = predictors.groupby("contrast")["hinge_z"].std()
    if (flat < 1e-9).any():
        print(f"  !! constant hinge in: {list(flat.index[flat < 1e-9])}")

    # Orthogonality is the whole point of s_within; assert it rather than trust it.
    worst = 0.0
    for _, block in predictors.groupby("contrast"):
        worst = max(worst, abs(float(np.corrcoef(block["binary_z"], block["s_within_z"])[0, 1])))
    print(f"  max |corr(binary_z, s_within_z)| across contrasts: {worst:.2e}")

    cells = predictors.groupby(["contrast", "escaper_class"]).size().unstack(fill_value=0)
    print("\nthree-level cell sizes (escaper split at the control mean):")
    print(cells.describe().loc[["min", "50%", "max"]].round(1).to_string())
    thin = cells.loc[(cells.get("escaper", 0) < 4) | (cells.get("severe", 0) < 4)]
    print(f"  contrasts with a cell below 4: {len(thin)} / {len(cells)} "
          f"-> R will mark these not-estimable for the three-level fit")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
