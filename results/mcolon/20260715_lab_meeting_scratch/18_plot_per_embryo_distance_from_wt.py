"""Per-embryo distance from wildtype OVER TIME: Euclidean vs Mahalanobis, by zygosity.

Each embryo gets a distance-from-wildtype at every timepoint (a per-row feature), then the
trajectories are plotted with plot_feature_over_time -- same machinery as the resolve-the-mess
figures -- colored by zygosity. Unlike a group-level metric, individual trajectories reveal
penetrance: some homozygotes ride high (penetrant), others stay in the wildtype band (non-penetrant).

Two distances, as feature rows:
  euclidean    : ||x - mu_wt|| on per-bin z-normed coords (dim / wildtype SD; no big-scale dim
                 dominates). Treats dimensions as independent.
  mahalanobis  : sqrt((x-mu_wt)^T Sigma_wt^-1 (x-mu_wt)), Sigma from per-bin wildtype (Ledoit-Wolf).
                 Accounts for scale AND correlations.

Reference is per-TIME-BIN wildtype (mu, Sigma recomputed each bin), so distance = "far from
wildtype AT THIS STAGE". Bins with too few wildtype to estimate the covariance are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
SOURCE = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style, update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402
from analyze.viz.styling import GENOTYPE_SUFFIX_COLORS  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

TABLE_DIR = SOURCE / "tables"
OUTPUT_DIR = RUN_DIR / "figures" / "per_embryo_distance_from_wt"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
GENE_TABLE = {"cep290": "reference_cep290_clean.csv", "b9d2": "reference_b9d2_clean.csv"}
GENE_ORDER = ["cep290", "b9d2"]
ZYG_ORDER = ["wildtype", "heterozygous", "homozygous"]   # draw order: wt underneath
BASELINE = "wildtype"
BIN_WIDTH = 4.0
MAX_HPF = 48.0
MIN_WT_PER_BIN = 8            # wildtype needed to estimate mu + 80x80 covariance in a bin
# Plot log2 distances: the raw distances are heavily right-skewed (homozygous fan to ~45 while
# wt/het sit near ~5), so a linear axis crushes the low end where wt/het/non-penetrant separate.
# Min distances are ~1.8 / ~3.1 (never near 0), so plain log2 is safe -- no epsilon needed.
FEATURES = ["log2_euclidean_dist_wt", "log2_mahalanobis_dist_wt"]
FEATURE_LABEL = {"log2_euclidean_dist_wt": "log2 Euclidean distance\nfrom wildtype",
                 "log2_mahalanobis_dist_wt": "log2 Mahalanobis distance\nfrom wildtype"}

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


def add_distances(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Attach per-row euclidean + mahalanobis distance from per-bin wildtype."""
    out = []
    for _, bin_df in df.groupby("time_bin"):
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
        chunk = bin_df.copy()
        euclid = np.sqrt(np.nansum(dz[:, keep] ** 2, axis=1))
        mahal = np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d).clip(min=0))
        chunk["euclidean_dist_wt"] = euclid
        chunk["mahalanobis_dist_wt"] = mahal
        chunk["log2_euclidean_dist_wt"] = np.log2(euclid)
        chunk["log2_mahalanobis_dist_wt"] = np.log2(mahal)
        out.append(chunk[[ID_COL, TIME_COL, "zygosity",
                          "euclidean_dist_wt", "mahalanobis_dist_wt", *FEATURES]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for gene in GENE_ORDER:
        d = add_distances(*load_gene(gene))
        d["gene"] = gene
        frames.append(d)
        print(f"[{gene}] rows with distance: {len(d)} | embryos: {d[ID_COL].nunique()}")
    df = pd.concat(frames, ignore_index=True)

    preset = ColorPreset(colors=COLORS, order=ZYG_ORDER)   # wildtype drawn first (underneath)
    style = update_style(
        presentation_style(), height_per_row=330, width_per_col=380, min_width=1000,
        individual_alpha=0.16, individual_width=0.6, trend_width=3.6,
        axis_label_fontsize=13, legend_fontsize=12,
    )

    fig = plot_feature_over_time(
        df,
        features=FEATURES,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="zygosity",
        color_preset=preset,
        facet_col="gene",
        layout=FacetSpec(col_order=GENE_ORDER, sharex=True, sharey=False),
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title="Per-embryo distance from wildtype over time (Euclidean & Mahalanobis), by zygosity",
        style=style,
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    n_cols = len(GENE_ORDER)
    for i, ax in enumerate(fig.axes):
        row, col = divmod(i, n_cols)
        ax.set_xlabel("Hours post fertilization")
        if col == 0 and row < len(FEATURES):
            ax.set_ylabel(FEATURE_LABEL[FEATURES[row]])

    out = OUTPUT_DIR / "per_embryo_distance_over_time_euclidean_vs_mahalanobis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")
    df.to_csv(OUTPUT_DIR / "per_embryo_distance_over_time.csv", index=False)


if __name__ == "__main__":
    main()
