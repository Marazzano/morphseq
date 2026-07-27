"""Fraction of embryos beyond the wildtype envelope, over time -- simplest 'differs from WT' + CI.

The single simplest interpretable statement of "this genotype's distribution differs from wildtype":
per time bin, set a threshold at the 95th percentile of WILDTYPE's own distance-from-wildtype, then
ask what FRACTION of each genotype exceeds it. Wildtype sits at ~5% by construction (the built-in
false-positive rate / null). A genotype whose fraction rises above that is diverging -- and the
fraction is literally a penetrance estimate. The confidence interval is a Wilson binomial interval
on that fraction (no permutations needed).

Computed for THREE distance metrics side by side, to compare how well each separates genotypes:
  euclidean   : ||x - mu_wt|| on per-bin z-normed coords (dims independent)
  znorm       : same coords, but distance = RMS of per-dim z-scores (scale-normalized magnitude)
  mahalanobis : sqrt((x-mu)^T Sigma_wt^-1 (x-mu)) -- scale AND correlations (Ledoit-Wolf)

The threshold is metric-specific and per-bin (95th pct of wildtype for THAT metric in THAT bin),
so every panel's wildtype line is ~5% and the genotype curves are directly comparable across metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
SOURCE = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.styling import GENOTYPE_SUFFIX_COLORS  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

TABLE_DIR = SOURCE / "tables"
OUTPUT_DIR = RUN_DIR / "figures" / "fraction_beyond_wt_threshold"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
GENE_TABLE = {"cep290": "reference_cep290_clean.csv", "b9d2": "reference_b9d2_clean.csv"}
GENE_ORDER = ["cep290", "b9d2"]
ZYG_ORDER = ["wildtype", "heterozygous", "homozygous"]
BASELINE = "wildtype"
BIN_WIDTH = 4.0
MAX_HPF = 48.0
MIN_WT_PER_BIN = 8
PCTILE = 95            # wildtype threshold percentile -> WT baseline ~ (100-PCTILE)%
METRICS = ["euclidean", "znorm", "mahalanobis"]
METRIC_LABEL = {"euclidean": "Euclidean", "znorm": "z-normed (RMS)",
                "mahalanobis": "Mahalanobis (whitened)"}

COLORS = {k: GENOTYPE_SUFFIX_COLORS[k] for k in ZYG_ORDER}


def _qc_mask(v: pd.Series) -> pd.Series:
    if v.dtype == bool:
        return v.fillna(False)
    return v.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_gene(gene: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(
        TABLE_DIR / GENE_TABLE[gene],
        usecols=lambda c: c in {ID_COL, TIME_COL, "zygosity", "use_embryo_flag"}
        or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    if "use_embryo_flag" in df.columns:
        df = df[_qc_mask(df["use_embryo_flag"])]
    feats = sorted((c for c in df.columns if c.startswith("z_mu_b_")),
                   key=lambda c: int(c.split("_")[-1]))
    df = df[df["zygosity"].isin(ZYG_ORDER)].copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *feats])
    df = df[df[TIME_COL] <= MAX_HPF]
    df["time_bin"] = np.floor(df[TIME_COL] / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2.0
    return df, feats


def fraction_beyond(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per (metric, bin, zygosity): fraction of embryos beyond the wildtype PCTILE threshold."""
    # collapse each embryo to one distance per bin (mean over its rows in that bin), per metric
    rows = []
    for time_bin, bin_df in df.groupby("time_bin"):
        wt = bin_df[bin_df["zygosity"] == BASELINE]
        if wt[ID_COL].nunique() < MIN_WT_PER_BIN:
            continue
        Xwt = wt[feats].to_numpy()
        mu = Xwt.mean(axis=0)
        sd = Xwt.std(axis=0, ddof=1)
        sd_safe = np.where(sd > 0, sd, np.nan)
        precision = np.linalg.pinv(LedoitWolf().fit(Xwt).covariance_)

        d = bin_df[feats].to_numpy() - mu
        dz = d / sd_safe
        keep = ~np.isnan(dz).any(axis=0)
        dist = {
            "euclidean": np.sqrt(np.nansum(dz[:, keep] ** 2, axis=1)),
            "znorm": np.sqrt(np.nanmean(dz[:, keep] ** 2, axis=1)),
            "mahalanobis": np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d).clip(min=0)),
        }
        emb = pd.DataFrame({ID_COL: bin_df[ID_COL].to_numpy(),
                            "zygosity": bin_df["zygosity"].to_numpy(), **dist})
        emb = emb.groupby([ID_COL, "zygosity"], as_index=False)[METRICS].mean()

        for metric in METRICS:
            thr = np.percentile(emb.loc[emb["zygosity"] == BASELINE, metric], PCTILE)
            for zyg in ZYG_ORDER:
                v = emb.loc[emb["zygosity"] == zyg, metric]
                n = len(v)
                if n == 0:
                    continue
                k = int((v > thr).sum())
                lo, hi = proportion_confint(k, n, method="wilson")
                rows.append({"time_bin": time_bin, "metric": metric, "zygosity": zyg,
                             "frac": k / n, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_gene = {g: fraction_beyond(*load_gene(g)) for g in GENE_ORDER}

    fig, axes = plt.subplots(len(METRICS), len(GENE_ORDER), figsize=(12, 10),
                             sharex=True, sharey=True, squeeze=False)
    for r, metric in enumerate(METRICS):
        for c, gene in enumerate(GENE_ORDER):
            ax = axes[r, c]
            d = per_gene[gene]
            d = d[d["metric"] == metric]
            for zyg in ZYG_ORDER:
                sub = d[d["zygosity"] == zyg].sort_values("time_bin")
                if sub.empty:
                    continue
                ax.fill_between(sub["time_bin"], sub["lo"], sub["hi"],
                                color=COLORS[zyg], alpha=0.18, lw=0)
                ax.plot(sub["time_bin"], sub["frac"], marker="o", ms=4, lw=2.2,
                        color=COLORS[zyg], label=zyg)
            ax.axhline(1 - PCTILE / 100, ls=":", color="gray", lw=1, alpha=0.7)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(gene, fontsize=13, fontweight="bold")
            if r == len(METRICS) - 1:
                ax.set_xlabel("Hours post fertilization (bin)")
            if c == 0:
                ax.set_ylabel(f"frac beyond WT {PCTILE}th\n({METRIC_LABEL[metric]})", fontsize=11)
            if r == 0 and c == len(GENE_ORDER) - 1:
                ax.legend(fontsize=9, frameon=True, loc="upper left")

    fig.suptitle(
        f"Fraction of embryos beyond wildtype {PCTILE}th-percentile distance (Wilson 95% CI)\n"
        "Euclidean vs z-normed vs Mahalanobis; dotted line = wildtype baseline "
        f"({100 - PCTILE}%)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUTPUT_DIR / "fraction_beyond_wt_threshold_3metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")
    pd.concat([d.assign(gene=g) for g, d in per_gene.items()], ignore_index=True).to_csv(
        OUTPUT_DIR / "fraction_beyond_wt_threshold.csv", index=False)


if __name__ == "__main__":
    main()
