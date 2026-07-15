"""
4_ablated_refit_fingerprints.py
-------------------------------
Follow-up to script 1 + the ablation in script 3: if dims 71 and 33 are just
convenient handles (ablation barely cost AUROC), then with 71/33 REMOVED from the
feature set, what does the classifier reach for next? Does it find a coherent
"second-choice" phenotype direction, or scatter across many dims?

Same 3 x 2 layout as 1_direction_fingerprints.py (rows CE / HTA / pooled;
LEFT = raw embedding difference from WT, RIGHT = signed classifier coefficient;
dims on Y natural order, time on X hpf), BUT the classifier is refit on the
78-dim set (71 and 33 dropped). The raw-difference column is unchanged (it is not
a fit) and shown for reference. Orange y-labels now mark the NEW top classifier
dims (data-driven), and 71/33 are drawn as grey struck-out rows to show they are
absent from the fit.

Data: reference_b9d2_clean.csv.
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
DROP_DIMS = ("z_mu_b_71", "z_mu_b_33")   # ablated from the classifier feature set
N_HIGHLIGHT = 4                           # how many new top dims to mark per contrast


def dim_short(d: str) -> str:
    return d.replace("z_mu_b_", "").replace("_binned", "")


def load_raw():
    df = pd.read_csv(B9D2_CSV, low_memory=False)
    z_cols = sorted(
        [c for c in df.columns if c.startswith("z_mu_b")],
        key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")),
    )
    return df, z_cols


def _grp_mask(df, contrast):
    if contrast == "pooled":
        return df["phenotype_clean"].isin(["CE", "HTA"])
    return df["phenotype_clean"] == contrast


def signed_coef_long(df, feat_cols, contrast):
    """Signed classifier coef per (time_bin, dim) fit on feat_cols (71/33 dropped)."""
    wt = df[df["zygosity"] == "wildtype"].copy()
    grp = df[_grp_mask(df, contrast)].copy()
    wt["genotype"], grp["genotype"] = "WT", "GRP"
    data = pd.concat([wt, grp], ignore_index=True)
    directions = extract_classifier_directions(
        data, class_col="genotype", id_col="embryo_id", time_col=TIME_COL,
        comparisons=[{"positive": "GRP", "negative": "WT"}],
        features={"emb": feat_cols}, bin_width=BIN_WIDTH,
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
    d = df.copy()
    d["time_bin"] = (d[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    wt = d[d["zygosity"] == "wildtype"]
    grp = d[_grp_mask(d, contrast)]
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


def _panel(ax, long_df, value_col, y_dims, time_bins, vmax, *, title, cbar_label,
           highlight, fig, mark_ablated=False):
    """Render one heatmap. Always drawn on the FULL 80-dim y-axis; if the fit
    excluded 71/33 they appear as explicit dark rows (mark_ablated=True) so the
    figure shows where they were removed rather than silently dropping them."""
    piv = long_df.pivot(index="dim", columns="time_bin", values=value_col)
    piv = piv.reindex(index=y_dims, columns=time_bins)
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Overlay ablated rows as solid dark-grey streaks across all time bins.
    if mark_ablated:
        for d in DROP_DIMS:
            if d in y_dims:
                yi = y_dims.index(d)
                ax.axhspan(yi - 0.5, yi + 0.5, color="#333333", zorder=5)

    ax.set_yticks(range(len(y_dims)))
    ax.set_yticklabels([dim_short(c) for c in y_dims], fontsize=4.3)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    for i, c in enumerate(y_dims):
        lbl = ax.get_yticklabels()[i]
        if c in DROP_DIMS:            # ablated: dark, bold "(ablated)"
            lbl.set_color("#333333")
            lbl.set_fontweight("bold")
        elif c in highlight:          # new top dims: orange bold
            lbl.set_color("darkorange")
            lbl.set_fontweight("bold")
            lbl.set_fontsize(6.0)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_raw()
    feat_cols = [c for c in z_cols if c not in DROP_DIMS]
    print(f"Loaded {len(df)} frames; classifier fit on {len(feat_cols)} dims "
          f"(dropped {list(DROP_DIMS)})")

    coef_frames, diff_frames, highlights = [], [], {}
    for contrast in CONTRASTS:
        cf = signed_coef_long(df, feat_cols, contrast)
        coef_frames.append(cf)
        diff_frames.append(embedding_diff_long(df, z_cols, contrast))
        # new top dims for this contrast = largest mean |coef| among the 78
        avg = cf.groupby("dim")["signed_coef"].apply(lambda s: s.abs().mean()).sort_values(ascending=False)
        highlights[contrast] = list(avg.index[:N_HIGHLIGHT])
        print(f"  {contrast:7} new top dims after ablation: {', '.join(dim_short(d) for d in highlights[contrast])}")
    coef_all = pd.concat(coef_frames, ignore_index=True)
    diff_all = pd.concat(diff_frames, ignore_index=True)
    coef_all.to_csv(TABLES / "ablated_refit_signed_coef_long.csv", index=False)

    time_bins = sorted(set(coef_all.time_bin) | set(diff_all.time_bin))
    diff_vmax = float(np.nanmax(np.abs(diff_all.embedding_diff.values)))
    coef_vmax = float(np.nanmax(np.abs(coef_all.signed_coef.values)))

    fig, axes = plt.subplots(len(CONTRASTS), 2, figsize=(22, 6.4 * len(CONTRASTS)),
                             gridspec_kw=dict(wspace=0.22))
    for r, contrast in enumerate(CONTRASTS):
        ax_diff, ax_coef = axes[r, 0], axes[r, 1]
        _panel(ax_diff, diff_all[diff_all.contrast == contrast], "embedding_diff",
               z_cols, time_bins, diff_vmax, fig=fig, highlight=highlights[contrast],
               title=f"{CONTRAST_TITLE[contrast]} — RAW embedding difference (unchanged, all 80 dims)",
               cbar_label="mean(GRP) - mean(WT)")
        _panel(ax_coef, coef_all[coef_all.contrast == contrast], "signed_coef",
               z_cols, time_bins, coef_vmax, fig=fig, highlight=highlights[contrast],
               title=f"{CONTRAST_TITLE[contrast]} — CLASSIFIER refit WITHOUT 71/33",
               cbar_label="signed logistic coef", mark_ablated=True)
        ax_diff.set_ylabel("z_mu_b dim (natural index order)", fontsize=10)
        for ax in (ax_diff, ax_coef):
            ax.set_xticks(range(len(time_bins)))
            ax.set_xticklabels([f"{int(t)}" for t in time_bins], fontsize=6.5, rotation=90)
        if r == len(CONTRASTS) - 1:
            ax_diff.set_xlabel("time_bin (hpf)")
            ax_coef.set_xlabel("time_bin (hpf)")

    fig.suptitle(
        "b9d2 — with dims 71 & 33 ABLATED, what does the classifier reach for next?\n"
        "LEFT: raw embedding difference (all 80 dims, reference)   |   RIGHT: classifier refit on 78 dims "
        "(71/33 removed). Orange = new top dims per row; grey italic = the removed 71/33.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "ablated_refit_diff_vs_classifier_3x2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved figure -> {out}")

    print("\n=== Sparsity after ablation: dims for 90% of mean |signed coef| mass ===")
    for contrast, grp in coef_all.groupby("contrast"):
        avg = grp.groupby("dim")["signed_coef"].apply(lambda s: s.abs().mean()).sort_values(ascending=False)
        cum = np.cumsum(avg.values) / avg.values.sum()
        n90 = int(np.searchsorted(cum, 0.9) + 1)
        print(f"  {contrast:7}  {n90:2d}/{len(avg)} dims -> 90% mass   top: {', '.join(avg.index[:5])}")


if __name__ == "__main__":
    main()
