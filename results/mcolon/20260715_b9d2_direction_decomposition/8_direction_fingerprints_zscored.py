"""
8_direction_fingerprints_zscored.py
-----------------------------------
Same 3 x 2 fingerprint as 1_direction_fingerprints.py (rows CE/HTA/pooled;
LEFT = embedding difference from WT, RIGHT = signed classifier coefficient; dims
on Y natural order, time on X), but computed on WT-Z-NORMALIZED coordinates and
with a shared color scale per column so magnitude is comparable across rows.

Why z-normalize: on raw dims the high-variance nuisance dims (71/33) dominate the
embedding difference and swamp the shared scale. Z-scoring each dim by WT (per
time bin) puts every dim in WT-noise units, so the fingerprint shows the DISTRIBUTED
phenotype structure rather than one saturating stripe. NOT ablated — full dim set.

Shared scale: one vmax across the 3 rows within each column (diff column shares a
scale; classifier column shares its own), so a strong stripe reads as genuinely
stronger than a weak one within that column.

Output: figures/direction_fingerprints_zscored_3x2.png
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
CONTRASTS = ("CE", "HTA", "pooled")
CONTRAST_TITLE = {"CE": "CE vs WT", "HTA": "HTA vs WT", "pooled": "pooled (CE+HTA) vs WT"}
HOT_DIMS = ("z_mu_b_71", "z_mu_b_33")


def dim_short(d):
    return d.replace("z_mu_b_", "").replace("_binned", "")


def load_and_zscore():
    """Load b9d2; z-score every dim by WT mean/std PER TIME BIN (WT-referenced).
    Returns the z-normalized frame + z_cols."""
    df = pd.read_csv(B9D2_CSV, low_memory=False)
    z_cols = sorted(
        [c for c in df.columns if c.startswith("z_mu_b")],
        key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")),
    )
    df = df.copy()
    df["time_bin"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    out = df.copy()
    for tb, idx in df.groupby("time_bin").groups.items():
        sub = df.loc[idx]
        wt = sub[sub["zygosity"] == "wildtype"]
        if len(wt) < 2:
            mu, sd = sub[z_cols].mean(), sub[z_cols].std(ddof=1)
        else:
            mu, sd = wt[z_cols].mean(), wt[z_cols].std(ddof=1)
        sd = sd.replace(0, np.nan)
        out.loc[idx, z_cols] = (sub[z_cols] - mu) / sd
    out[z_cols] = out[z_cols].fillna(0.0)
    return out, z_cols


def _grp_mask(df, contrast):
    if contrast == "pooled":
        return df["phenotype_clean"].isin(["CE", "HTA"])
    return df["phenotype_clean"] == contrast


def signed_coef_long(df, z_cols, contrast):
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


def embedding_diff_long(df, z_cols, contrast):
    """mean(GRP) - mean(WT) per dim per bin, on the (already z-scored) data."""
    wt = df[df["zygosity"] == "wildtype"]
    grp = df[_grp_mask(df, contrast)]
    rows = []
    for tb in sorted(set(grp["time_bin"]) & set(wt["time_bin"])):
        g, w = grp[grp["time_bin"] == tb], wt[wt["time_bin"] == tb]
        if g["embryo_id"].nunique() < 2 or w["embryo_id"].nunique() < 2:
            continue
        gm = g.groupby("embryo_id")[z_cols].mean().mean()
        wm = w.groupby("embryo_id")[z_cols].mean().mean()
        for name in z_cols:
            rows.append(dict(contrast=contrast, time_bin=float(tb), dim=name,
                             embedding_diff=float(gm[name] - wm[name])))
    return pd.DataFrame(rows)


def _panel(ax, long_df, value_col, z_cols, time_bins, vmax, *, title, cbar_label, fig):
    piv = long_df.pivot(index="dim", columns="time_bin", values=value_col).reindex(
        index=z_cols, columns=time_bins)
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(z_cols)))
    ax.set_yticklabels([dim_short(c) for c in z_cols], fontsize=4.3)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    for i, c in enumerate(z_cols):
        if c in HOT_DIMS:
            lbl = ax.get_yticklabels()[i]
            lbl.set_color("darkorange"); lbl.set_fontweight("bold"); lbl.set_fontsize(6.0)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_and_zscore()
    print(f"z-normalized {len(df)} frames, {len(z_cols)} dims")

    coef_frames, diff_frames = [], []
    for contrast in CONTRASTS:
        coef_frames.append(signed_coef_long(df, z_cols, contrast))
        diff_frames.append(embedding_diff_long(df, z_cols, contrast))
    coef_all = pd.concat(coef_frames, ignore_index=True)
    diff_all = pd.concat(diff_frames, ignore_index=True)

    time_bins = sorted(set(coef_all.time_bin) | set(diff_all.time_bin))
    # shared scale PER COLUMN across the 3 rows
    diff_vmax = float(np.nanmax(np.abs(diff_all.embedding_diff.values)))
    coef_vmax = float(np.nanmax(np.abs(coef_all.signed_coef.values)))

    fig, axes = plt.subplots(len(CONTRASTS), 2, figsize=(22, 6.4 * len(CONTRASTS)),
                             gridspec_kw=dict(wspace=0.22))
    for r, contrast in enumerate(CONTRASTS):
        ax_diff, ax_coef = axes[r, 0], axes[r, 1]
        _panel(ax_diff, diff_all[diff_all.contrast == contrast], "embedding_diff",
               z_cols, time_bins, diff_vmax, fig=fig,
               title=f"{CONTRAST_TITLE[contrast]} — z-normed embedding difference (WT-noise units)",
               cbar_label="mean(GRP)-mean(WT), z-units\n(shared scale, all rows)")
        _panel(ax_coef, coef_all[coef_all.contrast == contrast], "signed_coef",
               z_cols, time_bins, coef_vmax, fig=fig,
               title=f"{CONTRAST_TITLE[contrast]} — classifier signed coef (z-normed inputs)",
               cbar_label="signed logistic coef\n(shared scale, all rows)")
        ax_diff.set_ylabel("z_mu_b dim (natural index order)", fontsize=10)
        for ax in (ax_diff, ax_coef):
            ax.set_xticks(range(len(time_bins)))
            ax.set_xticklabels([f"{int(t)}" for t in time_bins], fontsize=6.5, rotation=90)
        if r == len(CONTRASTS) - 1:
            ax_diff.set_xlabel("time_bin (hpf)"); ax_coef.set_xlabel("time_bin (hpf)")

    fig.suptitle(
        "b9d2 phenotype fingerprints on Z-NORMALIZED coords (not ablated) — shared scale per column\n"
        "LEFT: z-normed embedding difference from WT   |   RIGHT: classifier signed coef. "
        "Orange = dims 71/33. Magnitude now comparable across CE/HTA/pooled within each column.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "direction_fingerprints_zscored_3x2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
