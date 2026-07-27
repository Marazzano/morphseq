"""Score every b9d2 embryo by P(homozygous) from a genotype-only classifier, then look at the
trajectories colored by that score.

The motivating problem: "non-penetrant homozygote" and "penetrant het" are impossible to define
by fiat -- any hard class boundary is arbitrary and pollutes the label set. So instead of
labeling penetrance, MEASURE it:

    train  : binary logistic, homozygous vs wildtype ONLY (hets excluded -- they are the
             ambiguous group we refuse to label).
    score  : out-of-fold P(homozygous) via random k-fold GROUPED BY EMBRYO (not by experiment),
             so every training embryo gets a score from a model that never saw it.
    apply  : hets (and unknowns) are scored by the full model.
    plot   : rows = curvature/length, cols = ZYGOSITY, lines colored by P(homozygous).

The model never sees a phenotype label, so P(homozygous) is a severity score that is
unsupervised with respect to phenotype. Read penetrance off the DISTRIBUTION instead of a
threshold:
    - a homozygote scoring LOW  P(homo) is a candidate non-penetrant mutant
    - a het scoring HIGH P(homo) is a candidate penetrant carrier
    - if the het column is bimodal, penetrant/non-penetrant carriers separate on their own

CAVEAT: het/unknown scores are NOT out-of-fold in the same sense -- those embryos were never in
any fold, so their scores are an extrapolation onto a boundary they did not help define. The
figure marks which embryos are out-of-fold vs applied.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/26_homo_vs_wt_probability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style,
    update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time  # noqa: E402

SOURCE = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
OUTPUT_DIR = RUN_DIR / "figures" / "homo_vs_wt_probability"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
LABEL_COL = "cluster_categories"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]

N_FOLDS = 5
RANDOM_STATE = 42

ZYG_ORDER = ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous", "b9d2_unknown"]
ZYG_LABEL = {
    "b9d2_wildtype": "wildtype",
    "b9d2_heterozygous": "heterozygous",
    "b9d2_homozygous": "homozygous",
    "b9d2_unknown": "unknown",
}

# P(homo) is continuous; the plotting engine maps DISCRETE values to colors, so bin the score
# and hand it an explicit lookup sampled from a perceptually-uniform colormap.
N_BINS = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(SOURCE, low_memory=False)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    feature_cols = sorted(
        [c for c in df.columns if c.startswith("z_mu_b_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    df = df.dropna(subset=[TIME_COL, *FEATURES, *feature_cols])
    return df, feature_cols


def score_embryos(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Out-of-fold P(homozygous) for homo/WT; full-model P(homozygous) for het/unknown.

    Scoring is per-ROW (per timepoint); the embryo-level score is the mean over its rows, so a
    trajectory gets one color.
    """
    train_mask = df["genotype"].isin(["b9d2_homozygous", "b9d2_wildtype"])
    train = df[train_mask]
    y = (train["genotype"] == "b9d2_homozygous").astype(int).to_numpy()
    X = train[feature_cols].to_numpy()
    groups = train[ID_COL].to_numpy()

    def _pipe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE),
        )

    # Random k-fold grouped by embryo: an embryo's timepoints never straddle a fold.
    oof = np.full(len(train), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, te_idx in cv.split(X, y, groups=groups):
        model = _pipe().fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]

    # Full model for the embryos that were never trainable (het / unknown).
    full = _pipe().fit(X, y)

    df = df.copy()
    df["p_homo"] = np.nan
    df.loc[train.index, "p_homo"] = oof
    other = df.index[~train_mask]
    if len(other):
        df.loc[other, "p_homo"] = full.predict_proba(df.loc[other, feature_cols].to_numpy())[:, 1]

    df["scoring"] = np.where(train_mask, "out-of-fold", "applied (not in training)")

    # Embryo-level score = mean over that embryo's timepoints.
    emb_score = df.groupby(ID_COL)["p_homo"].mean()
    df["p_homo_embryo"] = df[ID_COL].map(emb_score)
    return df


def add_color_bin(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """Bin the embryo-level score and build a viridis lookup keyed by bin label."""
    idx = np.clip(np.digitize(df["p_homo_embryo"], BIN_EDGES) - 1, 0, N_BINS - 1)
    labels = [f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i+1]:.1f}" for i in range(N_BINS)]
    df = df.copy()
    df["p_homo_bin"] = [labels[i] for i in idx]

    cmap = plt.get_cmap("viridis")
    lookup = {lab: matplotlib.colors.to_hex(cmap(i / (N_BINS - 1))) for i, lab in enumerate(labels)}
    return df, lookup, labels


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=300,
        width_per_col=330,
        min_width=1150,
        individual_alpha=0.55,
        individual_width=0.9,
        trend_width=0.0,          # trend medians would average across colors -- suppress
        axis_label_fontsize=12,
        legend_fontsize=9,
    )


def main() -> None:
    df, feature_cols = load()
    print(f"loaded {df[ID_COL].nunique()} embryos, {len(feature_cols)} latent features")
    print(df.drop_duplicates(ID_COL)["genotype"].value_counts().to_string())

    df = score_embryos(df, feature_cols)
    emb = df.drop_duplicates(ID_COL)

    print("\nmean P(homozygous) by genotype:")
    print(emb.groupby("genotype")["p_homo_embryo"].agg(["count", "mean", "std"]).round(3).to_string())

    print("\nP(homozygous) distribution by genotype (embryo-level):")
    for g in ZYG_ORDER:
        sub = emb[emb["genotype"] == g]["p_homo_embryo"]
        if len(sub) == 0:
            continue
        q = sub.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict()
        print(f"  {ZYG_LABEL[g]:14s} n={len(sub):3d}  deciles {q}")

    # Candidate non-penetrant homozygotes / penetrant hets, by score alone.
    homo = emb[emb["genotype"] == "b9d2_homozygous"]
    het = emb[emb["genotype"] == "b9d2_heterozygous"]
    print(f"\nhomozygotes with P(homo) < 0.5  (candidate NON-PENETRANT): "
          f"{(homo['p_homo_embryo'] < 0.5).sum()} / {len(homo)}")
    print(f"hets with P(homo) > 0.5  (candidate PENETRANT carriers): "
          f"{(het['p_homo_embryo'] > 0.5).sum()} / {len(het)}")

    df, lookup, labels = add_color_bin(df)
    df["zygosity"] = df["genotype"].map(ZYG_LABEL)
    cols = [ZYG_LABEL[g] for g in ZYG_ORDER if g in set(df["genotype"])]

    common = dict(
        features=FEATURES,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="p_homo_bin",
        color_lookup=lookup,
        facet_col="zygosity",
        layout=FacetSpec(col_order=cols, sharex=True, sharey=False),
        show_individual=True,
        show_trend=False,
        bin_width=3.0,
        title="b9d2 — trajectories colored by P(homozygous) from a genotype-only classifier",
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_out = OUTPUT_DIR / "p_homo_by_zygosity.html"
    plot_feature_over_time(df, backend="plotly", output_path=str(html_out), **common)
    print(f"\nSaved: {html_out.relative_to(RUN_DIR)}")

    fig = plot_feature_over_time(df, backend="matplotlib", **common)
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    png_out = OUTPUT_DIR / "p_homo_by_zygosity.png"
    fig.savefig(png_out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_out.relative_to(RUN_DIR)}")

    # Per-embryo scores, so specific candidates can be pulled and reviewed.
    roster = (emb[[ID_COL, "experiment_id", "genotype", LABEL_COL, "p_homo_embryo", "scoring"]]
              .sort_values(["genotype", "p_homo_embryo"]))
    roster_out = OUTPUT_DIR / "p_homo_scores.csv"
    roster.to_csv(roster_out, index=False)
    print(f"Saved: {roster_out.relative_to(RUN_DIR)}  ({len(roster)} embryos)")


if __name__ == "__main__":
    main()
