"""
7_coverage_response_curves.py
-----------------------------
Response curves: how many dimensions do you need to capture X% of the signal,
resolved over developmental time, per genotype? Reveals whether a *few* dims
carry a disproportionate chunk (the "couple really important ones") or whether
the signal is spread out.

Layout (every figure): 3 panels (CE / HTA / pooled), X = time (hpf),
Y = # dims needed, one colored line per coverage level {10,20,30,40,50,60,70}%.

FOUR figures (the point: scaling controls whether a few dims dominate):
  1. coverage_variance_raw.png        variance^2, RAW dims
  2. coverage_variance_zscored.png    variance^2, WT-per-bin z-scored dims
  3. coverage_classifier_zscored.png  classifier coef^2, refit per bin on z-scored inputs (fair)
  4. coverage_classifier_raw.png      classifier coef^2, refit per bin on RAW inputs (inherits artifact)

Read: if the 10-30% lines hug the bottom (~1-3 dims) the signal is dominated by a
few dims. On RAW data the low-% lines collapse onto the high-variance dims 71/33;
z-scoring lifts them off — that contrast is the whole point.

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
COVERAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LEVEL_CMAP = plt.cm.viridis
MIN_EMBRYOS = 3


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


def ndims_for(weights: np.ndarray, threshold: float) -> int:
    w2 = np.square(np.nan_to_num(weights, nan=0.0))
    order = np.argsort(w2)[::-1]
    cum = np.cumsum(w2[order])
    total = cum[-1]
    if total <= 0:
        return len(weights)
    return int(np.searchsorted(cum / total, threshold) + 1)


def _classifier_coef(zdf_or_rawdf, feat, tb):
    """Fit GRP-vs-WT at one bin, return signed coef aligned to feat (dict)."""
    d = zdf_or_rawdf.copy()
    d[TIME_COL] = float(tb)
    try:
        directions = extract_classifier_directions(
            d, class_col="genotype", id_col="embryo_id", time_col=TIME_COL,
            comparisons=[{"positive": "GRP", "negative": "WT"}],
            features={"emb": feat}, bin_width=BIN_WIDTH,
            min_samples_per_group=2, min_samples_per_member=2, verbose=False,
        )
        if directions.metadata.empty:
            return None
        r = directions.metadata.iloc[0]
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        return {n: float(w) for n, w in zip(names, vec)}
    except Exception:
        return None


def compute_over_time(df, z_cols):
    """For each (metric, contrast, time_bin): ndims for each coverage level."""
    df = df.copy()
    df["time_bin"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    rows = []
    for tb, bin_df in df.groupby("time_bin"):
        wt = bin_df[bin_df["zygosity"] == "wildtype"]
        wt_e = wt.groupby("embryo_id")[z_cols].mean()
        if len(wt_e) < MIN_EMBRYOS:
            continue
        wt_mean = wt_e.mean()
        wt_std = wt_e.std(ddof=1).replace(0, np.nan)

        for contrast in CONTRASTS:
            grp = bin_df[_grp_mask(bin_df, contrast)]
            grp_e = grp.groupby("embryo_id")[z_cols].mean()
            if len(grp_e) < MIN_EMBRYOS:
                continue

            # --- variance^2 weights: within-genotype per-dim variance ---
            var_raw = grp_e.var(ddof=1).to_numpy()
            grp_z = (grp_e - wt_mean) / wt_std
            var_z = grp_z.var(ddof=1).to_numpy()

            # --- classifier coef, raw inputs ---
            raw_fit_df = pd.concat([
                wt_e.assign(genotype="WT", embryo_id=wt_e.index),
                grp_e.assign(genotype="GRP", embryo_id=grp_e.index),
            ], ignore_index=True)
            cmap_raw = _classifier_coef(raw_fit_df, z_cols, tb)
            clf_raw = (np.array([cmap_raw.get(c, np.nan) for c in z_cols])
                       if cmap_raw else np.full(len(z_cols), np.nan))

            # --- classifier coef, z-scored inputs ---
            wt_z = (wt_e - wt_mean) / wt_std
            z_fit_df = pd.concat([
                wt_z.assign(genotype="WT", embryo_id=wt_z.index),
                grp_z.assign(genotype="GRP", embryo_id=grp_z.index),
            ], ignore_index=True).dropna(axis=1, how="any")
            feat_z = [c for c in z_cols if c in z_fit_df.columns]
            cmap_z = _classifier_coef(z_fit_df, feat_z, tb)
            clf_z = (np.array([cmap_z.get(c, np.nan) for c in z_cols])
                     if cmap_z else np.full(len(z_cols), np.nan))

            weight_sets = {
                "variance_raw": var_raw, "variance_zscored": var_z,
                "classifier_raw": clf_raw, "classifier_zscored": clf_z,
            }
            for metric, w in weight_sets.items():
                row = {"metric": metric, "contrast": contrast, "time_bin": float(tb)}
                for lvl in COVERAGE_LEVELS:
                    row[f"cov_{int(lvl*100)}"] = ndims_for(w, lvl)
                rows.append(row)
    return pd.DataFrame(rows)


def make_figure(long_df, metric, title, out_name):
    sub = long_df[long_df.metric == metric]
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(18, 5.5), sharey=True)
    colors = {lvl: LEVEL_CMAP(i / (len(COVERAGE_LEVELS) - 1))
              for i, lvl in enumerate(COVERAGE_LEVELS)}
    for ax, contrast in zip(axes, CONTRASTS):
        c = sub[sub.contrast == contrast].sort_values("time_bin")
        for lvl in COVERAGE_LEVELS:
            ax.plot(c.time_bin, c[f"cov_{int(lvl*100)}"], "-o", ms=3,
                    color=colors[lvl], label=f"{int(lvl*100)}%")
        ax.set_title(f"{contrast} vs WT", fontsize=12)
        ax.set_xlabel("time_bin (hpf)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("# dims needed")
    axes[-1].legend(title="coverage", fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / out_name, dpi=140)
    plt.close(fig)
    print(f"Saved figures/{out_name}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_raw()
    print(f"Loaded {len(df)} frames, {len(z_cols)} z_mu_b dims")

    long_df = compute_over_time(df, z_cols)
    long_df.to_csv(TABLES / "coverage_response_long.csv", index=False)

    make_figure(long_df, "variance_raw",
                "Dims for X% of per-dim VARIANCE over time — RAW dims\n"
                "(low-% lines collapse to ~1 dim = the high-variance 71/33 artifact)",
                "coverage_variance_raw.png")
    make_figure(long_df, "variance_zscored",
                "Dims for X% of per-dim VARIANCE over time — WT-z-scored dims\n"
                "(variance evened out; low-% lines lift off — the honest picture)",
                "coverage_variance_zscored.png")
    make_figure(long_df, "classifier_zscored",
                "Dims for X% of CLASSIFIER coef² over time — z-scored inputs (fair)\n"
                "(all dims on even footing; discriminative weight spread across many dims)",
                "coverage_classifier_zscored.png")
    make_figure(long_df, "classifier_raw",
                "Dims for X% of CLASSIFIER coef² over time — RAW inputs\n"
                "(classifier inherits the variance artifact; low-% lines collapse onto 71/33)",
                "coverage_classifier_raw.png")

    # headline: median dims for 10% and 50% per metric/contrast
    print("\n=== median # dims for 10% / 50% / 90% coverage ===")
    for metric in ("variance_raw", "variance_zscored", "classifier_raw", "classifier_zscored"):
        for contrast in CONTRASTS:
            c = long_df[(long_df.metric == metric) & (long_df.contrast == contrast)]
            if c.empty:
                continue
            print(f"  {metric:18} {contrast:7}  10%={c.cov_10.median():.0f}  "
                  f"50%={c.cov_50.median():.0f}  70%={c.cov_70.median():.0f}  "
                  f"90%={c.cov_90.median():.0f}")


if __name__ == "__main__":
    main()
