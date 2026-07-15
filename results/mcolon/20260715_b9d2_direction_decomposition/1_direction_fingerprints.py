"""
1_direction_fingerprints.py
---------------------------
b9d2 has TWO labeled phenotypes vs wildtype (CE, HTA). Hypothesis: the two
phenotypes live along two DIFFERENT directions in z_mu_b space. If we split the
phenotype into its classes, CE and HTA should light up different dims, and the
pooled signal should be their superposition.

Follows the heatmap layout of
20260703_raw_latent_clustering_baseline/13_per_dim_difference.py
(fig_raw_delta_vs_classifier):
  imshow, Y-axis = z_mu_b dims in natural index order (20..99),
          X-axis = time_bin (hpf), RdBu_r diverging centered at 0.

3 x 2 grid:
  rows        = CE (top), HTA (middle), pooled CE+HTA (bottom), each vs WT
  LEFT column = raw embedding difference from WT: mean(GRP) - mean(WT) per dim
  RIGHT column= signed classifier coefficient (GRP-vs-WT logistic unit coef)
So you can directly see what the classifier hones in on vs where the raw
embedding actually moves. Orange y-labels = classifier-hot dims 71 / 33.

Data: reference_b9d2_clean.csv (phenotype_clean in {CE, HTA, wildtype},
zygosity, 80 z_mu_b dims, predicted_stage_hpf).
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

FIGURES = _HERE / "figures"
TABLES = _HERE / "tables"
B9D2_CSV = (
    _REPO / "results" / "mcolon" / "20260607_sci_cilia_gene14_imaging_qc"
    / "tables" / "reference_b9d2_clean.csv"
)

TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0

# Row order top -> bottom: CE, HTA, pooled. All vs wildtype.
CONTRASTS = ("CE", "HTA", "pooled")
CONTRAST_TITLE = {"CE": "CE vs WT", "HTA": "HTA vs WT", "pooled": "pooled (CE+HTA) vs WT"}
HOT_DIMS = ("z_mu_b_71", "z_mu_b_33")


def dim_short(d: str) -> str:
    return d.replace("z_mu_b_", "").replace("_binned", "")


def load_raw():
    df = pd.read_csv(B9D2_CSV, low_memory=False)
    z_cols = sorted(
        [c for c in df.columns if c.startswith("z_mu_b")],
        key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")),
    )
    return df, z_cols


def _grp_mask(df: pd.DataFrame, contrast: str) -> pd.Series:
    if contrast == "pooled":
        return df["phenotype_clean"].isin(["CE", "HTA"])
    return df["phenotype_clean"] == contrast


def signed_coef_long(df, z_cols, contrast) -> pd.DataFrame:
    """One row per (time_bin, dim) -> signed classifier coef (GRP vs WT).
    time_bin reported as the raw bin (center - bin_width/2) so x reads in hpf."""
    wt = df[df["zygosity"] == "wildtype"].copy()
    grp = df[_grp_mask(df, contrast)].copy()
    wt["genotype"], grp["genotype"] = "WT", "GRP"
    data = pd.concat([wt, grp], ignore_index=True)

    directions = extract_classifier_directions(
        data, class_col="genotype", id_col="embryo_id", time_col=TIME_COL,
        comparisons=[{"positive": "GRP", "negative": "WT"}],
        features={"emb": z_cols}, bin_width=BIN_WIDTH,
        min_samples_per_group=2, min_samples_per_member=2, verbose=False,
    )
    rows = []
    for _, r in directions.metadata.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        tb = float(r["time_bin_center"]) - BIN_WIDTH / 2.0
        for name, w in zip(names, vec):
            rows.append(dict(contrast=contrast, time_bin=tb, dim=name, signed_coef=float(w)))
    return pd.DataFrame(rows)


def embedding_diff_long(df, z_cols, contrast) -> pd.DataFrame:
    """One row per (time_bin, dim) -> mean(GRP) - mean(WT), same binning as the fit.
    Per-embryo means first so frame-heavy embryos don't dominate."""
    d = df.copy()
    d["time_bin"] = (d[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    wt = d[d["zygosity"] == "wildtype"]
    grp = d[_grp_mask(d, contrast)]
    rows = []
    for tb in sorted(set(grp["time_bin"]) & set(wt["time_bin"])):
        g = grp[grp["time_bin"] == tb]
        w = wt[wt["time_bin"] == tb]
        if g["embryo_id"].nunique() < 2 or w["embryo_id"].nunique() < 2:
            continue
        gm = g.groupby("embryo_id")[z_cols].mean().mean()
        wm = w.groupby("embryo_id")[z_cols].mean().mean()
        for name in z_cols:
            rows.append(dict(contrast=contrast, time_bin=float(tb), dim=name,
                             embedding_diff=float(gm[name] - wm[name])))
    return pd.DataFrame(rows)


def _panel(ax, long_df, value_col, z_cols, time_bins, vmax, *, title, cbar_label, fig):
    piv = long_df.pivot(index="dim", columns="time_bin", values=value_col)
    piv = piv.reindex(index=z_cols, columns=time_bins)
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(z_cols)))
    ax.set_yticklabels([dim_short(c) for c in z_cols], fontsize=4.3)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    for i, c in enumerate(z_cols):
        if c in HOT_DIMS:
            ax.get_yticklabels()[i].set_color("darkorange")
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_fontsize(6.0)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_raw()
    print(f"Loaded {len(df)} frames, {len(z_cols)} z_mu_b dims")

    coef_frames, diff_frames = [], []
    for contrast in CONTRASTS:
        coef_frames.append(signed_coef_long(df, z_cols, contrast))
        diff_frames.append(embedding_diff_long(df, z_cols, contrast))
        print(f"  {contrast:7} fitted")
    coef_all = pd.concat(coef_frames, ignore_index=True)
    diff_all = pd.concat(diff_frames, ignore_index=True)
    coef_all.to_csv(TABLES / "classifier_signed_coef_long.csv", index=False)
    diff_all.to_csv(TABLES / "embedding_diff_long.csv", index=False)

    # shared time grid; independent color scales per column (diff vs coef have
    # different natural magnitudes) but shared across the 3 rows within a column
    time_bins = sorted(set(coef_all.time_bin) | set(diff_all.time_bin))
    diff_vmax = float(np.nanmax(np.abs(diff_all.embedding_diff.values)))
    coef_vmax = float(np.nanmax(np.abs(coef_all.signed_coef.values)))

    fig, axes = plt.subplots(len(CONTRASTS), 2, figsize=(22, 6.4 * len(CONTRASTS)),
                             gridspec_kw=dict(wspace=0.22))
    for r, contrast in enumerate(CONTRASTS):
        ax_diff, ax_coef = axes[r, 0], axes[r, 1]
        _panel(ax_diff, diff_all[diff_all.contrast == contrast], "embedding_diff",
               z_cols, time_bins, diff_vmax, fig=fig,
               title=f"{CONTRAST_TITLE[contrast]} — RAW embedding difference (mean GRP - mean WT)",
               cbar_label="mean(GRP) - mean(WT)\n(+ toward phenotype)")
        _panel(ax_coef, coef_all[coef_all.contrast == contrast], "signed_coef",
               z_cols, time_bins, coef_vmax, fig=fig,
               title=f"{CONTRAST_TITLE[contrast]} — CLASSIFIER signed coefficient",
               cbar_label="signed logistic coef\n(+ toward phenotype)")
        ax_diff.set_ylabel("z_mu_b dim (natural index order 20..99)", fontsize=10)
        for ax in (ax_diff, ax_coef):
            ax.set_xticks(range(len(time_bins)))
            ax.set_xticklabels([f"{int(t)}" for t in time_bins], fontsize=6.5, rotation=90)
        if r == len(CONTRASTS) - 1:
            ax_diff.set_xlabel("time_bin (hpf)")
            ax_coef.set_xlabel("time_bin (hpf)")

    fig.suptitle(
        "b9d2 phenotype directions — LEFT: raw embedding difference from WT   |   "
        "RIGHT: classifier signed coefficient\n"
        "rows = CE / HTA / pooled. CE hones in on dim 71 (in both); HTA hones in on dim 33 "
        "(classifier finds it even where the raw difference is faint). Orange y-labels = dims 71 / 33.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "direction_diff_vs_classifier_3x2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved figure -> {out}")

    print("\n=== Sparsity: dims carrying 90% of mean |signed coef| mass ===")
    for contrast, grp in coef_all.groupby("contrast"):
        avg = grp.groupby("dim")["signed_coef"].apply(lambda s: s.abs().mean()).sort_values(ascending=False)
        cum = np.cumsum(avg.values) / avg.values.sum()
        n90 = int(np.searchsorted(cum, 0.9) + 1)
        print(f"  {contrast:7}  {n90:2d}/{len(avg)} dims -> 90% mass   top: {', '.join(avg.index[:4])}")


if __name__ == "__main__":
    main()
