"""
10_classifier_weight_heatmap.py
--------------------------------
Quick first pass: for each perturbation vs inj_ctrl, what does the classifier
consider important, per raw z_mu_b dimension?

Scripts 8/9 showed the classifier-margin space substantially reorganizes the
raw 80-dim z_mu_b neighbor structure. The margin space itself is only ~10
scalar pairwise-classifier axes, i.e. a heavy compression of the raw manifold.
This script does NOT compare that reorganization to raw mean differences
(too noisy per-dim, not the point) -- it just shows, directly, which of the
80 raw dims the classifier weights heavily for each contrast. If the hot
columns are a small sparse subset, that's consistent with "classification
looks at a subset of the raw space's degrees of freedom" (leaving the rest
of the raw manifold's structure -- batch, stage, other biology -- on the
table, unseen by margin space).

Uses extract_classifier_directions() (no CV/AUROC needed, just the fitted
logistic-regression unit coefficient vectors).

Outputs:
  figures/classifier_weight_heatmap_by_bin.png   rows = contrast x time_bin
  figures/classifier_weight_heatmap_pooled.png   rows = contrast, cols snapped
                                                  to key stage checkpoints
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.classification.directions.extract import extract_classifier_directions

TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"
RAW_WT_CSV = TABLES / "pbx_binned_zmub_with_wt.csv"

CRISPANTS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
GENO_LABEL = {"wik_ab": "WT", "pbx1b_crispant": "pbx1b",
              "pbx4_crispant": "pbx4", "pbx1b_pbx4_crispant": "double"}

KEY_STAGES = (20.0, 30.0, 58.0, 72.0, 96.0)   # hpf checkpoints for pooled view

DIR_META_CSV = TABLES / "classifier_directions_metadata.csv"
DIR_VEC_NPZ = TABLES / "classifier_directions_vectors.npz"


def load_raw():
    df = pd.read_csv(RAW_WT_CSV, low_memory=False)
    z_cols = sorted([c for c in df.columns if "z_mu_b" in c],
                    key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")))
    return df, z_cols


def fit_directions(df, z_cols):
    """One binary comparison per crispant vs inj_ctrl, all time bins at once."""
    comparisons = [{"positive": g, "negative": "inj_ctrl"} for g in CRISPANTS]
    directions = extract_classifier_directions(
        df,
        class_col="genotype", id_col="embryo_id", time_col="time_bin",
        comparisons=comparisons,
        features={"emb": z_cols},
        bin_width=4.0,
        verbose=True,
    )
    return directions


def directions_to_frame(directions, z_cols):
    """Long frame: one row per (comparison_id, time_bin_center, dim) -> |weight|."""
    meta = directions.metadata.copy()
    rows = []
    for _, r in meta.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        for name, w in zip(names, vec):
            rows.append(dict(comparison_id=r["comparison_id"],
                              time_bin_center=r["time_bin_center"],
                              dim=name, abs_weight=abs(float(w))))
    return pd.DataFrame(rows)


def make_by_bin_heatmap(long_df, z_cols):
    long_df = long_df.copy()
    long_df["row_label"] = (long_df.comparison_id.str.replace("__vs__inj_ctrl", "", regex=False)
                             .map(lambda g: GENO_LABEL.get(g, g))
                             + " @ " + long_df.time_bin_center.astype(int).astype(str) + "hpf")
    row_order = (long_df[["comparison_id", "time_bin_center", "row_label"]]
                 .drop_duplicates()
                 .sort_values(["comparison_id", "time_bin_center"]))
    piv = long_df.pivot(index="row_label", columns="dim", values="abs_weight")
    piv = piv.reindex(index=row_order.row_label, columns=z_cols)

    fig, ax = plt.subplots(figsize=(14, max(4, 0.28 * len(piv))))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xticks(range(0, len(z_cols), 5))
    ax.set_xticklabels([z_cols[i] for i in range(0, len(z_cols), 5)],
                        rotation=90, fontsize=6)
    ax.set_xlabel("raw z_mu_b dimension (natural order)")
    ax.set_title("Classifier |weight| per raw dim, by contrast x time_bin\n"
                  "(each row: crispant vs inj_ctrl logistic-regression unit coefficients)")
    fig.colorbar(im, ax=ax, label="|unit coefficient|", shrink=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "classifier_weight_heatmap_by_bin.png", dpi=130)
    plt.close(fig)


def make_pooled_heatmap(long_df, z_cols):
    """Snap each row's time_bin_center to the nearest key stage checkpoint,
    average |weight| within (comparison, checkpoint)."""
    long_df = long_df.copy()
    bins = np.asarray(KEY_STAGES)
    long_df["checkpoint"] = bins[
        np.argmin(np.abs(long_df.time_bin_center.values[:, None] - bins[None, :]), axis=1)
    ]
    pooled = (long_df.groupby(["comparison_id", "checkpoint", "dim"])["abs_weight"]
              .mean().reset_index())
    pooled["row_label"] = (pooled.comparison_id.str.replace("__vs__inj_ctrl", "", regex=False)
                            .map(lambda g: GENO_LABEL.get(g, g))
                            + " @ " + pooled.checkpoint.astype(int).astype(str) + "hpf")
    row_order = (pooled[["comparison_id", "checkpoint", "row_label"]]
                 .drop_duplicates().sort_values(["comparison_id", "checkpoint"]))
    piv = pooled.pivot(index="row_label", columns="dim", values="abs_weight")
    piv = piv.reindex(index=row_order.row_label, columns=z_cols)

    fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * len(piv))))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xticks(range(0, len(z_cols), 5))
    ax.set_xticklabels([z_cols[i] for i in range(0, len(z_cols), 5)],
                        rotation=90, fontsize=6)
    ax.set_xlabel("raw z_mu_b dimension (natural order)")
    ax.set_title(f"Classifier |weight| per raw dim, pooled to key stages {KEY_STAGES}")
    fig.colorbar(im, ax=ax, label="|unit coefficient| (mean over bins near checkpoint)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "classifier_weight_heatmap_pooled.png", dpi=130)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    df, z_cols = load_raw()
    df = df[df.genotype.isin((*CRISPANTS, "inj_ctrl"))].copy()
    print(f"Fitting directions: {len(df)} rows, {df.embryo_id.nunique()} embryos, "
          f"{len(z_cols)} z_mu_b dims")

    directions = fit_directions(df, z_cols)
    long_df = directions_to_frame(directions, z_cols)
    long_df.to_csv(TABLES / "classifier_weight_long.csv", index=False)

    make_by_bin_heatmap(long_df, z_cols)
    print(f"Saved figure -> figures/classifier_weight_heatmap_by_bin.png")
    make_pooled_heatmap(long_df, z_cols)
    print(f"Saved figure -> figures/classifier_weight_heatmap_pooled.png")

    # quick sparsity readout
    print("\n=== Sparsity check: fraction of dims carrying 90% of |weight| mass ===")
    for cid, grp in long_df.groupby("comparison_id"):
        avg = grp.groupby("dim")["abs_weight"].mean().sort_values(ascending=False)
        cum = np.cumsum(avg.values) / avg.values.sum()
        n90 = int(np.searchsorted(cum, 0.9) + 1)
        print(f"  {cid:35s}  {n90:2d}/{len(avg)} dims -> 90% of mean |weight| mass")


if __name__ == "__main__":
    main()
