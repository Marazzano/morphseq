"""Energy distance from wildtype with a wildtype-half-split null band -- simplest principled version.

One panel per (normalization, gene): energy distance of homozygous / heterozygous vs wildtype per
time bin, plus a NULL band from splitting wildtype in half and computing energy(half A, half B)
over many random splits. The wildtype-vs-wildtype band is "how much energy distance you get from
wildtype against itself" -- pure sampling noise. A genotype whose energy distance rises ABOVE the
band is meaningfully different from wildtype. The half-split IS the permutation null, drawn as a
band, so no separate p-value machinery is needed.

Energy distance (analyze.difference_detection): E(P,Q)=2E|X-Y| - E|X-X'| - E|Y-Y'|; zero iff the
two distributions are identical (captures mean + spread + shape, not just centroid gap).

Three normalizations as rows (they agreed on shape in script 17; shown together as robustness):
  raw    : raw z_mu_b (a few big-scale dims dominate)
  znorm  : each dim / wildtype per-bin SD (equal per-dim footing)
  whiten : per-bin wildtype covariance^-1/2 (scale AND correlations; Ledoit-Wolf)

Null band = empirical 2.5-97.5 percentile of the wildtype-vs-wildtype energy distances.
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
OUTPUT_DIR = RUN_DIR / "figures" / "energy_distance_halfsplit_null"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
GENE_TABLE = {"cep290": "reference_cep290_clean.csv", "b9d2": "reference_b9d2_clean.csv"}
GENE_ORDER = ["cep290", "b9d2"]
GROUPS = ["homozygous", "heterozygous"]
BASELINE = "wildtype"
BIN_WIDTH = 4.0
MAX_HPF = 48.0
MIN_PER_BIN = 8          # min wildtype per bin (so each half has >= MIN_PER_BIN/2)
N_SPLITS = 100           # wildtype half-splits for the null band
SEED = 0
NORMS = ["raw", "znorm", "whiten"]
NULL_LO, NULL_HI = 2.5, 97.5

COLORS = {k: GENOTYPE_SUFFIX_COLORS[k] for k in GROUPS}
NULL_COLOR = GENOTYPE_SUFFIX_COLORS[BASELINE]


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


def _make_transform(wt_feats: np.ndarray, norm: str):
    """Return f(X)->transformed X, built from THIS bin's wildtype."""
    mu = wt_feats.mean(axis=0)
    if norm == "raw":
        return lambda X: X - mu
    if norm == "znorm":
        sd = wt_feats.std(axis=0, ddof=1)
        keep = sd > 0
        return lambda X: ((X - mu) / np.where(sd > 0, sd, np.nan))[:, keep]
    # whiten
    cov = LedoitWolf().fit(wt_feats).covariance_
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, 1e-8, None))) @ vecs.T
    return lambda X: (X - mu) @ W.T


def energy_with_null(df: pd.DataFrame, feats: list[str], norm: str) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    rows = []
    for time_bin, bin_df in df.groupby("time_bin"):
        wt = bin_df[bin_df["zygosity"] == BASELINE]
        wt_ids = wt[ID_COL].drop_duplicates().to_numpy()
        if len(wt_ids) < MIN_PER_BIN:
            continue
        Xwt_raw = wt[feats].to_numpy()
        f = _make_transform(Xwt_raw, norm)
        Xwt = f(Xwt_raw)

        rec = {"time_bin": time_bin}
        # groups vs full wildtype
        for group in GROUPS:
            g = bin_df[bin_df["zygosity"] == group]
            if g[ID_COL].nunique() < MIN_PER_BIN:
                continue
            rec[group] = compute_energy_distance(Xwt, f(g[feats].to_numpy()))
        # wildtype-vs-wildtype null: split embryos A/B many times
        null = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(len(wt_ids))
            half = len(perm) // 2
            a = np.isin(wt[ID_COL].to_numpy(), wt_ids[perm[:half]])
            null.append(compute_energy_distance(Xwt[a], Xwt[~a]))
        rec["null_lo"] = np.percentile(null, NULL_LO)
        rec["null_hi"] = np.percentile(null, NULL_HI)
        rec["null_med"] = np.median(null)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for gene in GENE_ORDER:
        df, feats = load_gene(gene)
        for norm in NORMS:
            results[(gene, norm)] = energy_with_null(df, feats, norm)
            print(f"[{gene}/{norm}] bins={len(results[(gene, norm)])}")

    fig, axes = plt.subplots(len(NORMS), len(GENE_ORDER), figsize=(12, 10),
                             sharex=True, squeeze=False)
    for r, norm in enumerate(NORMS):
        for c, gene in enumerate(GENE_ORDER):
            ax = axes[r, c]
            d = results[(gene, norm)].sort_values("time_bin")
            if not d.empty:
                ax.fill_between(d["time_bin"], d["null_lo"], d["null_hi"],
                                color=NULL_COLOR, alpha=0.22, lw=0,
                                label="wildtype null (WT vs WT, 95%)")
                ax.plot(d["time_bin"], d["null_med"], color=NULL_COLOR, ls=":", lw=1.4, alpha=0.9)
                for group in GROUPS:
                    if group in d:
                        ax.plot(d["time_bin"], d[group], marker="o", ms=4, lw=2.4,
                                color=COLORS[group], label=group)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(gene, fontsize=13, fontweight="bold")
            if r == len(NORMS) - 1:
                ax.set_xlabel("Hours post fertilization (bin)")
            if c == 0:
                ax.set_ylabel(f"energy distance\n({norm})", fontsize=12)
            if r == 0 and c == len(GENE_ORDER) - 1:
                ax.legend(fontsize=9, frameon=True, loc="upper left")

    fig.suptitle(
        "Energy distance from wildtype vs wildtype-half-split null (95% band)\n"
        "raw / z-normed / whitened coords",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUTPUT_DIR / "energy_distance_halfsplit_null_3norms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")
    pd.concat([d.assign(gene=g, norm=n) for (g, n), d in results.items()],
              ignore_index=True).to_csv(OUTPUT_DIR / "energy_distance_halfsplit_null.csv",
                                        index=False)


if __name__ == "__main__":
    main()
