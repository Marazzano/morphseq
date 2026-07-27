"""
2_dimension_concentration.py
----------------------------
How many z_mu_b dimensions actually account for the b9d2 phenotype signal, and
which ones? Sweep a coverage threshold (90% -> 10%) to see how few dims you need,
so we can pick a principled "descriptive dimension" cutoff.

Three per-dim weightings, each reduced to a concentration (Lorenz) curve —
sort dims by contribution descending, cumulative fraction of total:
  RAW magnitude  : delta_d^2, delta = mean(GRP) - mean(WT)   (pure geometric shift)
  EFFECT size    : cohens_d_d^2                                (significance-weighted, N-aware)
  CLASSIFIER     : coef_d^2, unit logistic direction          (fraction of decision magnitude)

Across the 3 x 2 comparison (rows CE / HTA / pooled; the "2" = raw vs classifier,
plus effect-size as the honest raw threshold-setter).

Deliverables:
  figures/concentration_curves.png    cumulative fraction vs #dims, per contrast
  figures/concentration_scatter.png   effect-size^2 & raw^2 vs classifier^2 per dim
                                       (tests: does meaningful variation track model magnitude?)
  figures/ndims_for_90pct_over_time.png  companion: N dims for 90% over developmental time
  tables/threshold_ndims.csv          {contrast x weighting} x {90..10%} -> N dims needed
  tables/per_dim_contributions.csv     per-dim raw/effect/classifier contributions (pooled)

Pooled across time (per-embryo means first so frame-heavy embryos don't dominate),
with a time-resolved companion for the 90% count only.

Data: reference_b9d2_clean.csv (phenotype_clean in {CE, HTA, wildtype}, zygosity,
80 z_mu_b dims, predicted_stage_hpf).
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
CONTRASTS = ("CE", "HTA", "pooled")
CONTRAST_COLOR = {"CE": "#1b7837", "HTA": "#762a83", "pooled": "#4d4d4d"}
HOT_DIMS = ("z_mu_b_71", "z_mu_b_33")
THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
WEIGHTINGS = ("raw", "effect", "classifier")


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


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d, group b vs group a (both are per-embryo mean vectors for one dim)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return np.nan
    return (b.mean() - a.mean()) / pooled_sd


def per_dim_contributions(df, z_cols, contrast):
    """Pooled per-dim weightings: raw delta, Cohen's d, and classifier |coef|.

    Raw + effect are computed on per-embryo means (embryo = unit). Classifier is
    the mean |signed coef| across all time bins (unit direction per bin)."""
    # per-embryo mean vectors within this contrast's window (all time pooled)
    wt_e = df[df["zygosity"] == "wildtype"].groupby("embryo_id")[z_cols].mean()
    grp_e = df[_grp_mask(df, contrast)].groupby("embryo_id")[z_cols].mean()

    delta = grp_e.mean().to_numpy() - wt_e.mean().to_numpy()
    cohend = np.array([_cohens_d(wt_e[c].to_numpy(), grp_e[c].to_numpy()) for c in z_cols])

    # classifier: fit pooled across bins, average |coef| per dim
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
    coef_acc = {c: [] for c in z_cols}
    for _, r in directions.metadata.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        for name, w in zip(names, vec):
            coef_acc[name].append(abs(float(w)))
    clf = np.array([np.mean(coef_acc[c]) if coef_acc[c] else np.nan for c in z_cols])

    return pd.DataFrame({
        "contrast": contrast, "dim": z_cols,
        "raw": delta, "effect": cohend, "classifier": clf,
    })


def concentration_curve(weights: np.ndarray) -> np.ndarray:
    """Cumulative fraction of squared-magnitude, dims sorted descending.
    Returns cumfrac of length n (cumfrac[k-1] = fraction captured by top-k dims)."""
    w2 = np.square(np.nan_to_num(weights, nan=0.0))
    order = np.argsort(w2)[::-1]
    cum = np.cumsum(w2[order])
    total = cum[-1]
    return cum / total if total > 0 else np.zeros_like(cum)


def ndims_for(cumfrac: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumfrac, threshold) + 1)


def fig_concentration_curves(contrib: pd.DataFrame, z_cols):
    n = len(z_cols)
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(16, 5), sharey=True)
    style = {"raw": ("-", "#4d4d4d"), "effect": ("--", "#e08214"),
             "classifier": ("-.", "#2166ac")}
    for ax, contrast in zip(axes, CONTRASTS):
        c = contrib[contrib.contrast == contrast]
        for w in WEIGHTINGS:
            cum = concentration_curve(c[w].to_numpy())
            ls, col = style[w]
            ax.plot(np.arange(1, n + 1), cum, ls, color=col, lw=2, label=w)
        for t in THRESHOLDS:
            ax.axhline(t, color="#ddd", lw=0.6, zorder=0)
        ax.set_title(f"{contrast} vs WT", fontsize=12)
        ax.set_xlabel("# dimensions (sorted by contribution)")
        ax.set_xlim(1, n)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("cumulative fraction of squared magnitude")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("How many z_mu_b dims capture the phenotype signal? "
                 "(raw Δ², Cohen's d², classifier coef²)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "concentration_curves.png", dpi=140)
    plt.close(fig)


def fig_scatter(contrib: pd.DataFrame, z_cols):
    """Per-dim: does meaningful variation (raw / effect) track classifier magnitude?"""
    fig, axes = plt.subplots(2, len(CONTRASTS), figsize=(16, 9))
    for j, contrast in enumerate(CONTRASTS):
        c = contrib[contrib.contrast == contrast].reset_index(drop=True)
        clf2 = np.square(c["classifier"].to_numpy())
        for i, (wname, wlabel) in enumerate([("effect", "Cohen's d²"), ("raw", "raw Δ²")]):
            ax = axes[i][j]
            w2 = np.square(c[wname].to_numpy())
            ax.scatter(clf2, w2, s=18, color="#999", alpha=0.7, edgecolors="none")
            for k, dim in enumerate(c["dim"]):
                if dim in HOT_DIMS:
                    ax.scatter(clf2[k], w2[k], s=90, facecolors="none",
                               edgecolors="darkorange", linewidths=2)
                    ax.annotate(dim_short(dim), (clf2[k], w2[k]), fontsize=9,
                                color="darkorange", fontweight="bold",
                                xytext=(4, 4), textcoords="offset points")
            good = np.isfinite(clf2) & np.isfinite(w2)
            if good.sum() > 3:
                rho, p = stats.spearmanr(clf2[good], w2[good])
                ax.set_title(f"{contrast}: {wlabel} vs classifier²   ρ={rho:.2f} (p={p:.1e})",
                             fontsize=10)
            ax.set_xlabel("classifier coef²")
            ax.set_ylabel(wlabel)
            ax.grid(alpha=0.2)
    fig.suptitle("Does a dim's meaningful variation track how much magnitude the classifier gives it?\n"
                 "(each point = one z_mu_b dim; orange = dims 71 / 33)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "concentration_scatter.png", dpi=140)
    plt.close(fig)


def fig_ndims_over_time(df, z_cols):
    """Companion: N dims for 90% coverage, per contrast, over developmental time.
    Uses effect-size (Cohen's d) within each bin (the significance-weighted count)."""
    d = df.copy()
    d["tb"] = (d[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    bins = sorted(d["tb"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    for contrast in CONTRASTS:
        xs, ys = [], []
        for tb in bins:
            sub = d[d["tb"] == tb]
            wt_e = sub[sub["zygosity"] == "wildtype"].groupby("embryo_id")[z_cols].mean()
            grp_e = sub[_grp_mask(sub, contrast)].groupby("embryo_id")[z_cols].mean()
            if len(wt_e) < 3 or len(grp_e) < 3:
                continue
            cohend = np.array([_cohens_d(wt_e[c].to_numpy(), grp_e[c].to_numpy()) for c in z_cols])
            cum = concentration_curve(cohend)
            xs.append(tb)
            ys.append(ndims_for(cum, 0.9))
        ax.plot(xs, ys, "-o", ms=3, color=CONTRAST_COLOR[contrast], label=contrast)
    ax.set_xlabel("time_bin (hpf)")
    ax.set_ylabel("# dims for 90% (Cohen's d²)")
    ax.set_title("Dimensions needed for 90% of the significance-weighted signal, over time\n"
                 "(early timepoints expected less coherent)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "ndims_for_90pct_over_time.png", dpi=140)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_raw()
    print(f"Loaded {len(df)} frames, {len(z_cols)} z_mu_b dims")

    contrib = pd.concat([per_dim_contributions(df, z_cols, c) for c in CONTRASTS],
                        ignore_index=True)
    contrib.to_csv(TABLES / "per_dim_contributions.csv", index=False)

    # threshold table: N dims to reach each coverage, per (contrast, weighting)
    rows = []
    for contrast in CONTRASTS:
        c = contrib[contrib.contrast == contrast]
        for w in WEIGHTINGS:
            cum = concentration_curve(c[w].to_numpy())
            row = {"contrast": contrast, "weighting": w}
            for t in THRESHOLDS:
                row[f"{int(t*100)}%"] = ndims_for(cum, t)
            # which dims make the top of the 90% set
            w2 = np.square(np.nan_to_num(c[w].to_numpy()))
            order = np.argsort(w2)[::-1]
            top = [dim_short(c["dim"].to_numpy()[i]) for i in order[:5]]
            row["top5_dims"] = ", ".join(top)
            rows.append(row)
    thresh = pd.DataFrame(rows)
    thresh.to_csv(TABLES / "threshold_ndims.csv", index=False)

    print("\n=== N dims needed for each coverage threshold (pooled across time) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(thresh.to_string(index=False))

    fig_concentration_curves(contrib, z_cols)
    print("\nSaved figures/concentration_curves.png")
    fig_scatter(contrib, z_cols)
    print("Saved figures/concentration_scatter.png")
    fig_ndims_over_time(df, z_cols)
    print("Saved figures/ndims_for_90pct_over_time.png")


if __name__ == "__main__":
    main()
