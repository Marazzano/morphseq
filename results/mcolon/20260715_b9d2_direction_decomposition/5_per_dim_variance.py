"""
5_per_dim_variance.py
---------------------
Concern: do dims 71/33 dominate the phenotype signal simply because they carry
the largest raw VARIANCE (so mean-shifts and classifier coefficients land there
for scale reasons), rather than being phenotypically special? The VAE latent is
not standardized, so a high-variance dim can swamp the geometry.

Trivial check: per-dim variance in WT and in b9d2 (CE / HTA), across time.
  - If 71/33 are variance outliers, the "hot dims" story is partly a scaling
    artifact and we should standardize dims before selection/clustering.
  - If 71/33 have ordinary variance, their salience is genuine phenotype signal.

Computed on per-embryo mean vectors within each time bin (embryo = unit), so
frame count doesn't inflate variance.

Outputs:
  figures/variance_heatmaps.png       per-dim variance (dims x time) for WT, CE, HTA
  figures/variance_rank_pooled.png    dims ranked by pooled variance, 71/33 marked
  figures/variance_vs_coef.png        does |classifier coef| just track high variance?
  tables/per_dim_variance_long.csv    variance per (group, time_bin, dim)

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
from scipy import stats

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
GROUPS = ("WT", "CE", "HTA")
HOT_DIMS = ("z_mu_b_71", "z_mu_b_33")
GROUP_COLOR = {"WT": "#4d4d4d", "CE": "#1b7837", "HTA": "#762a83"}


def dim_short(d):
    return d.replace("z_mu_b_", "").replace("_binned", "")


def load_raw():
    df = pd.read_csv(B9D2_CSV, low_memory=False)
    z_cols = sorted(
        [c for c in df.columns if c.startswith("z_mu_b")],
        key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")),
    )
    return df, z_cols


def _group_df(df, group):
    if group == "WT":
        return df[df["zygosity"] == "wildtype"]
    return df[df["phenotype_clean"] == group]


def variance_long(df, z_cols):
    """Per-dim variance across embryos, per (group, time_bin). Per-embryo means
    first so frame count doesn't inflate the variance."""
    d = df.copy()
    d["time_bin"] = (d[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    rows = []
    for group in GROUPS:
        g = _group_df(d, group)
        for tb, sub in g.groupby("time_bin"):
            emb = sub.groupby("embryo_id")[z_cols].mean()   # embryo = unit
            if len(emb) < 3:
                continue
            v = emb.var(ddof=1)
            for name in z_cols:
                rows.append(dict(group=group, time_bin=float(tb), dim=name,
                                 variance=float(v[name])))
    return pd.DataFrame(rows)


def fig_variance_heatmaps(var_long, z_cols):
    time_bins = sorted(var_long.time_bin.unique())
    # shared log color scale across groups for comparability
    vmax = np.nanpercentile(var_long.variance.values, 99)
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(18, 8), sharey=True)
    for ax, group in zip(axes, GROUPS):
        piv = (var_long[var_long.group == group]
               .pivot(index="dim", columns="time_bin", values="variance")
               .reindex(index=z_cols, columns=time_bins))
        im = ax.imshow(piv.values, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(z_cols)))
        ylabels = ax.set_yticklabels([dim_short(c) for c in z_cols], fontsize=4.3)
        ax.set_xticks(range(0, len(time_bins), 3))
        ax.set_xticklabels([f"{int(time_bins[i])}" for i in range(0, len(time_bins), 3)],
                           fontsize=7, rotation=90)
        ax.set_title(f"{group}  (per-dim variance)", fontsize=12)
        ax.set_xlabel("time_bin (hpf)")
        if len(ylabels) == len(z_cols):   # only the first (non-shared) axis has labels
            for i, c in enumerate(z_cols):
                if c in HOT_DIMS:
                    ylabels[i].set_color("darkorange")
                    ylabels[i].set_fontweight("bold")
                    ylabels[i].set_fontsize(6.5)
        fig.colorbar(im, ax=ax, shrink=0.7, label="variance across embryos")
    axes[0].set_ylabel("z_mu_b dim (natural index order)")
    fig.suptitle("Per-dim variance over time — are dims 71 / 33 (orange) variance outliers?\n"
                 "(if they dominate the color scale, their salience is partly a scaling artifact)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "variance_heatmaps.png", dpi=130)
    plt.close(fig)


def fig_variance_rank(var_long, z_cols):
    """Pooled-over-time mean variance per dim, ranked, per group; 71/33 marked."""
    fig, ax = plt.subplots(figsize=(16, 6))
    pooled = var_long.groupby(["group", "dim"])["variance"].mean().reset_index()
    # rank dims by WT variance (the reference geometry)
    wt_order = (pooled[pooled.group == "WT"].sort_values("variance", ascending=False)["dim"].tolist())
    x = np.arange(len(wt_order))
    width = 0.26
    for k, group in enumerate(GROUPS):
        vals = (pooled[pooled.group == group].set_index("dim").reindex(wt_order)["variance"].values)
        ax.bar(x + (k - 1) * width, vals, width, color=GROUP_COLOR[group], label=group, alpha=0.85)
    ax.set_xticks(x)
    xlabels = ax.set_xticklabels([dim_short(d) for d in wt_order], fontsize=5, rotation=90)
    for i, d in enumerate(wt_order):
        if d in HOT_DIMS:
            xlabels[i].set_color("darkorange")
            xlabels[i].set_fontweight("bold")
            ax.axvspan(i - 0.5, i + 0.5, color="orange", alpha=0.12, zorder=0)
    ax.set_ylabel("mean variance across embryos (pooled over time)")
    ax.set_xlabel("z_mu_b dim (sorted by WT variance, descending)")
    ax.set_title("Per-dim variance ranked by WT — where do dims 71 / 33 sit?\n"
                 "(if far left = high-variance outliers; if mid/right = ordinary variance)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    # annotate the rank of 71/33 in WT
    for d in HOT_DIMS:
        if d in wt_order:
            r = wt_order.index(d)
            ax.annotate(f"{dim_short(d)}: WT var rank {r+1}/{len(wt_order)}",
                        xy=(r, pooled[(pooled.group=='WT')&(pooled.dim==d)]["variance"].values[0]),
                        xytext=(r + 3, ax.get_ylim()[1] * (0.8 - 0.1*HOT_DIMS.index(d))),
                        fontsize=9, color="darkorange", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="darkorange"))
    fig.tight_layout()
    fig.savefig(FIGURES / "variance_rank_pooled.png", dpi=140)
    plt.close(fig)
    return wt_order


def fig_variance_vs_coef(df, var_long, z_cols):
    """Does |classifier coef| just track high-variance dims? Pooled scatter per group."""
    # classifier |coef| pooled (mean over bins), pooled CE+HTA vs WT and each split
    def coef_by_dim(contrast):
        wt = df[df["zygosity"] == "wildtype"].copy()
        if contrast == "pooled":
            grp = df[df["phenotype_clean"].isin(["CE", "HTA"])].copy()
        else:
            grp = df[df["phenotype_clean"] == contrast].copy()
        wt["genotype"], grp["genotype"] = "WT", "GRP"
        data = pd.concat([wt, grp], ignore_index=True)
        directions = extract_classifier_directions(
            data, class_col="genotype", id_col="embryo_id", time_col=TIME_COL,
            comparisons=[{"positive": "GRP", "negative": "WT"}],
            features={"emb": z_cols}, bin_width=BIN_WIDTH,
            min_samples_per_group=2, min_samples_per_member=2, verbose=False,
        )
        acc = {c: [] for c in z_cols}
        for _, r in directions.metadata.iterrows():
            vec = directions.vectors[r["vector_id"]]
            names = directions.feature_names[r["feature_set"]]
            for name, w in zip(names, vec):
                acc[name].append(abs(float(w)))
        return {c: np.mean(acc[c]) if acc[c] else np.nan for c in z_cols}

    # variance reference = WT variance pooled over time
    wt_var = (var_long[var_long.group == "WT"].groupby("dim")["variance"].mean())

    contrasts = ("CE", "HTA", "pooled")
    fig, axes = plt.subplots(1, len(contrasts), figsize=(18, 5.5))
    for ax, contrast in zip(axes, contrasts):
        coefs = coef_by_dim(contrast)
        xs = np.array([wt_var.get(c, np.nan) for c in z_cols])
        ys = np.array([coefs[c] for c in z_cols])
        ax.scatter(xs, ys, s=20, color="#999", alpha=0.7, edgecolors="none")
        for c, x, y in zip(z_cols, xs, ys):
            if c in HOT_DIMS:
                ax.scatter(x, y, s=90, facecolors="none", edgecolors="darkorange", linewidths=2)
                ax.annotate(dim_short(c), (x, y), fontsize=9, color="darkorange",
                            fontweight="bold", xytext=(4, 4), textcoords="offset points")
        good = np.isfinite(xs) & np.isfinite(ys)
        if good.sum() > 3:
            rho, p = stats.spearmanr(xs[good], ys[good])
            ax.set_title(f"{contrast}: |coef| vs WT variance   ρ={rho:.2f} (p={p:.1e})", fontsize=10)
        ax.set_xlabel("WT variance (per dim, pooled over time)")
        ax.set_ylabel("mean |classifier coef|")
        ax.grid(alpha=0.2)
    fig.suptitle("Does the classifier just weight high-variance dims?\n"
                 "(if 71/33 are far RIGHT = high variance AND high coef, salience may be scale-driven; "
                 "orange = dims 71/33)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "variance_vs_coef.png", dpi=140)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_raw()
    print(f"Loaded {len(df)} frames, {len(z_cols)} z_mu_b dims")

    var_long = variance_long(df, z_cols)
    var_long.to_csv(TABLES / "per_dim_variance_long.csv", index=False)

    fig_variance_heatmaps(var_long, z_cols)
    print("Saved figures/variance_heatmaps.png")
    wt_order = fig_variance_rank(var_long, z_cols)
    print("Saved figures/variance_rank_pooled.png")
    fig_variance_vs_coef(df, var_long, z_cols)
    print("Saved figures/variance_vs_coef.png")

    # headline numbers: WT variance rank of 71/33, and their variance vs the median
    pooled_wt = var_long[var_long.group == "WT"].groupby("dim")["variance"].mean().sort_values(ascending=False)
    med = pooled_wt.median()
    print("\n=== WT per-dim variance (pooled over time) ===")
    print(f"  median dim variance: {med:.4f}")
    for d in HOT_DIMS:
        rank = list(pooled_wt.index).index(d) + 1
        print(f"  {dim_short(d):>4}: variance={pooled_wt[d]:.4f}  "
              f"(rank {rank}/{len(pooled_wt)},  {pooled_wt[d]/med:.1f}x median)")
    print(f"  top-5 WT-variance dims: {', '.join(dim_short(d) for d in pooled_wt.index[:5])}")


if __name__ == "__main__":
    main()
