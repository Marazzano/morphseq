"""
6_zscored_ndims_over_time.py
----------------------------
Regenerate the "dimensions needed for 90% of the signal, over time" figure, but
on Z-SCORED dimensions so the constitutive high-variance nuisance dims (71 = 105x
median variance, 33 = 8x) stop dominating. Once every dim is on an even playing
field, we see how many dims ACTUALLY carry the phenotype.

Z-scoring: per dim, per time bin, using WT mean/std (WT = reference geometry, so a
dim's phenotype signal is measured in WT-noise units).

Two weightings on the same figure, both as squared-magnitude concentration curves:
  Cohen's d^2         : significance-weighted (scale-invariant; reference line)
  classifier coef^2   : logistic direction REFIT per bin on the z-scored features
                        (now fair to compare — standardized inputs put all dims on
                        even footing, so the coefficient reflects true discriminative
                        weight, not input scale)

Y = # dims for 90% of the squared signal; X = time; one line per
(contrast in {CE, HTA, pooled}) x (weighting).

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
CONTRAST_COLOR = {"CE": "#1b7837", "HTA": "#762a83", "pooled": "#4d4d4d"}
WEIGHT_STYLE = {"cohens_d": "-o", "classifier": "--s"}
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


def concentration_ndims(weights: np.ndarray, threshold: float = 0.9) -> int:
    w2 = np.square(np.nan_to_num(weights, nan=0.0))
    order = np.argsort(w2)[::-1]
    cum = np.cumsum(w2[order])
    total = cum[-1]
    if total <= 0:
        return len(weights)
    return int(np.searchsorted(cum / total, threshold) + 1)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df, z_cols = load_raw()
    df = df.copy()
    df["time_bin"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH
    print(f"Loaded {len(df)} frames, {len(z_cols)} z_mu_b dims")

    rows = []
    for tb, bin_df in df.groupby("time_bin"):
        wt = bin_df[bin_df["zygosity"] == "wildtype"]
        wt_e = wt.groupby("embryo_id")[z_cols].mean()          # embryo = unit
        if len(wt_e) < MIN_EMBRYOS:
            continue
        # WT reference scale for this bin (per dim)
        wt_mean = wt_e.mean()
        wt_std = wt_e.std(ddof=1).replace(0, np.nan)

        for contrast in CONTRASTS:
            grp = bin_df[_grp_mask(bin_df, contrast)]
            grp_e = grp.groupby("embryo_id")[z_cols].mean()
            if len(grp_e) < MIN_EMBRYOS:
                continue

            # ── Cohen's d per dim (scale-invariant; WT-noise units) ──
            # d = (mean_grp - mean_wt) / pooled_sd
            n1, n2 = len(wt_e), len(grp_e)
            v1 = wt_e.var(ddof=1)
            v2 = grp_e.var(ddof=1)
            pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
            cohend = ((grp_e.mean() - wt_mean) / pooled_sd.replace(0, np.nan)).to_numpy()

            # ── classifier refit on WT-z-scored features for this bin ──
            # standardize BOTH groups by WT mean/std, then fit GRP vs WT
            wt_z = (wt_e - wt_mean) / wt_std
            grp_z = (grp_e - wt_mean) / wt_std
            zdf = pd.concat([
                wt_z.assign(genotype="WT", embryo_id=wt_z.index),
                grp_z.assign(genotype="GRP", embryo_id=grp_z.index),
            ], ignore_index=True)
            zdf[TIME_COL] = float(tb)   # single-bin fit
            zdf = zdf.dropna(axis=1, how="any")   # drop dims with zero WT std this bin
            feat = [c for c in z_cols if c in zdf.columns]
            clf_coef = np.full(len(z_cols), np.nan)
            try:
                directions = extract_classifier_directions(
                    zdf, class_col="genotype", id_col="embryo_id", time_col=TIME_COL,
                    comparisons=[{"positive": "GRP", "negative": "WT"}],
                    features={"emb": feat}, bin_width=BIN_WIDTH,
                    min_samples_per_group=2, min_samples_per_member=2, verbose=False,
                )
                if not directions.metadata.empty:
                    r = directions.metadata.iloc[0]
                    vec = directions.vectors[r["vector_id"]]
                    names = directions.feature_names[r["feature_set"]]
                    cmap = {n: float(w) for n, w in zip(names, vec)}
                    clf_coef = np.array([cmap.get(c, np.nan) for c in z_cols])
            except Exception:
                pass

            rows.append(dict(
                time_bin=float(tb), contrast=contrast,
                ndims_cohens_d=concentration_ndims(cohend),
                ndims_classifier=concentration_ndims(clf_coef),
            ))

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "zscored_ndims_over_time.csv", index=False)

    # ── plot ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for contrast in CONTRASTS:
        c = out[out.contrast == contrast].sort_values("time_bin")
        ax.plot(c.time_bin, c.ndims_cohens_d, WEIGHT_STYLE["cohens_d"], ms=4,
                color=CONTRAST_COLOR[contrast], label=f"{contrast} · Cohen's d²")
        ax.plot(c.time_bin, c.ndims_classifier, WEIGHT_STYLE["classifier"], ms=4,
                color=CONTRAST_COLOR[contrast], alpha=0.6,
                label=f"{contrast} · classifier coef²")
    ax.set_xlabel("time_bin (hpf)")
    ax.set_ylabel("# dims for 90% of squared signal")
    ax.set_title("Dimensions needed for 90% of the phenotype signal, over time — "
                 "on WT-z-scored dims\n"
                 "(nuisance high-variance dims 71/33 no longer dominate; "
                 "solid = Cohen's d², dashed = classifier coef²)", fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png = FIGURES / "zscored_ndims_for_90pct_over_time.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_png}")

    print("\n=== median # dims for 90% (z-scored) ===")
    for contrast in CONTRASTS:
        c = out[out.contrast == contrast]
        print(f"  {contrast:7} Cohen's d²: {c.ndims_cohens_d.median():.0f}   "
              f"classifier²: {c.ndims_classifier.median():.0f}")


if __name__ == "__main__":
    main()
