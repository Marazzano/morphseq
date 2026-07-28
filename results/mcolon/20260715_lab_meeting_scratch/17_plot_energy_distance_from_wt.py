"""Energy distance from wildtype per time bin, by zygosity -- principled distribution distance.

Replaces the hand-rolled centroid RMS (script 16) with ENERGY DISTANCE
(analyze.difference_detection.compute_energy_distance): a formal statistical distance between two
distributions that is zero iff they are identical and captures differences in mean AND spread AND
shape -- not just the gap between centroids.

    E(P,Q) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|      (X~P wildtype, Y~Q group; |.| Euclidean)

Two normalizations are computed side by side, to decide empirically which to keep:
  - raw    : energy distance on raw z_mu_b. Simplest to explain, BUT the 80 dims span a ~22x SD
             range (z_mu_b_71, _33 dominate), so raw energy is driven by those few big-scale dims.
  - z-norm : each dim divided by wildtype per-bin SD first, so every dim contributes on equal
             (wildtype-noise) footing before the Euclidean terms.
If the two panels agree in shape, the exploded dims are not distorting things and raw is fine to
present; if they disagree, the signal is concentrated in a few dims and z-norm is the honest view.

No permutation p-values yet (distance only) -- significance is a planned second pass via
permutation_test_energy.
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

from analyze.difference_detection.distance_metrics import compute_energy_distance  # noqa: E402
from analyze.viz.styling import GENOTYPE_SUFFIX_COLORS  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

TABLE_DIR = SOURCE / "tables"
OUTPUT_DIR = RUN_DIR / "figures" / "energy_distance_from_wt"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
GENE_TABLE = {"cep290": "reference_cep290_clean.csv", "b9d2": "reference_b9d2_clean.csv"}
GENE_ORDER = ["cep290", "b9d2"]
GROUPS = ["homozygous", "heterozygous"]
BASELINE = "wildtype"
BIN_WIDTH = 4.0
MAX_HPF = 48.0
MIN_PER_BIN = 4          # min embryos per group AND per wildtype ref in a bin
# raw   : no scaling (dominated by big-scale dims like z_mu_b_71/_33)
# znorm : per-dim / wildtype SD (equal per-dim footing, ignores correlations)
# whiten: Mahalanobis -- transform by wildtype covariance^-1/2 (scale AND correlations);
#         Ledoit-Wolf shrinkage keeps the 80x80 covariance invertible with few embryos/bin.
NORMS = ["raw", "znorm", "whiten"]

COLORS = {k: GENOTYPE_SUFFIX_COLORS[k] for k in GROUPS}


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
    df = df[df["zygosity"].isin([*GROUPS, BASELINE])].copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *feats])
    df = df[df[TIME_COL] <= MAX_HPF]
    df["time_bin"] = np.floor(df[TIME_COL] / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2.0
    return df, feats


def _whitener(Xwt: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk wildtype covariance^-1/2 (Mahalanobis transform matrix)."""
    cov = LedoitWolf().fit(Xwt).covariance_
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def energy_by_bin(df: pd.DataFrame, feats: list[str], norm: str) -> pd.DataFrame:
    """Energy distance of each group vs wildtype per bin, under raw / znorm / whiten transform."""
    rows = []
    for time_bin, bin_df in df.groupby("time_bin"):
        wt = bin_df[bin_df["zygosity"] == BASELINE]
        if wt[ID_COL].nunique() < MIN_PER_BIN:
            continue

        # Build the transform from wildtype in this bin, then apply it to every group.
        wt_center = wt[feats].mean()

        def transform(rows_df: pd.DataFrame) -> np.ndarray:
            centered = (rows_df[feats] - wt_center).to_numpy()
            if norm == "raw":
                return centered
            if norm == "znorm":
                sd = wt[feats].std(ddof=1).to_numpy()
                sd = np.where(sd > 0, sd, np.nan)
                out = centered / sd
                return out[:, ~np.isnan(out).any(axis=0)]
            # whiten (Mahalanobis)
            return centered @ W.T

        if norm == "whiten":
            W = _whitener(wt[feats].to_numpy())
        Xwt = transform(wt)
        for group in GROUPS:
            g = bin_df[bin_df["zygosity"] == group]
            if g[ID_COL].nunique() < MIN_PER_BIN:
                continue
            e = compute_energy_distance(Xwt, transform(g))
            rows.append({"time_bin": time_bin, "zygosity": group, "energy": e,
                         "n_group": g[ID_COL].nunique(), "n_wt": wt[ID_COL].nunique()})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for gene in GENE_ORDER:
        df, feats = load_gene(gene)
        for norm in NORMS:
            results[(gene, norm)] = energy_by_bin(df, feats, norm)
            print(f"[{gene}/{norm}] bins={results[(gene, norm)]['time_bin'].nunique()}")

    # rows = normalization, cols = gene
    fig, axes = plt.subplots(len(NORMS), len(GENE_ORDER), figsize=(11.5, 8),
                             sharex=True, squeeze=False)
    for r, norm in enumerate(NORMS):
        for c, gene in enumerate(GENE_ORDER):
            ax = axes[r, c]
            d = results[(gene, norm)]
            for zyg in GROUPS:
                sub = d[d["zygosity"] == zyg].sort_values("time_bin")
                if sub.empty:
                    continue
                ax.plot(sub["time_bin"], sub["energy"], marker="o", ms=4, lw=2.4,
                        color=COLORS[zyg], label=zyg)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(gene, fontsize=13, fontweight="bold")
            if r == len(NORMS) - 1:
                ax.set_xlabel("Hours post fertilization (bin)")
            if c == 0:
                ax.set_ylabel(f"Energy distance\n({norm})", fontsize=12)
            if r == 0 and c == len(GENE_ORDER) - 1:
                ax.legend(fontsize=10, frameon=True, loc="upper left")

    fig.suptitle(
        "Energy distance from wildtype by zygosity (raw vs z-normed vs whitened (Mahalanobis) z_mu_b)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUTPUT_DIR / "energy_distance_from_wt_raw_znorm_whiten.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")

    pd.concat([d.assign(gene=g, norm=n) for (g, n), d in results.items()],
              ignore_index=True).to_csv(
        OUTPUT_DIR / "energy_distance_from_wt.csv", index=False)


if __name__ == "__main__":
    main()
