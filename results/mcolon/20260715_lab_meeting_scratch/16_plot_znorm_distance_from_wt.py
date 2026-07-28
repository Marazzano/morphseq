"""Z-normalized morphology distance from wildtype, per time bin, by zygosity, WITH a CV null band.

Fixes the weakness of a single-wildtype-mean distance: we can't tell whether a small group
distance (e.g. cep290 het) is real or just measurement noise. Here the wildtype itself defines
the null: split wildtype into k folds; each fold's distance from the OTHER wildtype folds gives
one null trajectory, and the k folds together give a null ERROR BAND -- the intrinsic variability
of "wildtype vs wildtype" in this measurement. A zygosity group is only meaningfully diverged if
it sits ABOVE that band.

Per time bin, per z_mu_b dim d, distance of a group G from a wildtype reference R:
    z_d = ( mean_d(G) - mean_d(R) ) / SD_d(R in that bin)
    distance(G) = sqrt( mean_d z_d^2 )        # RMS z, "R-noise units"

Null: for each of k wildtype folds f, R = wildtype \ f (the other folds), G = fold f. That is a
held-out-wildtype-vs-rest-wildtype distance. Band = mean +/- SD across the k folds.
Groups (homozygous, heterozygous): R = ALL wildtype, distance computed the same way.

Groups are zygosity only -- deliberately no phenotype labels. The question is purely whether
heterozygotes are distribution-level distinguishable from wildtype.
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

from analyze.viz.styling import GENOTYPE_SUFFIX_COLORS  # noqa: E402

TABLE_DIR = SOURCE / "tables"
OUTPUT_DIR = RUN_DIR / "figures" / "znorm_distance_from_wt"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
GENE_TABLE = {"cep290": "reference_cep290_clean.csv", "b9d2": "reference_b9d2_clean.csv"}
GENE_ORDER = ["cep290", "b9d2"]
GROUPS = ["homozygous", "heterozygous"]
BASELINE = "wildtype"
BIN_WIDTH = 4.0
MAX_HPF = 48.0
N_SHUFFLES = 200         # wildtype half-split resamples per bin -> error bands
MIN_REF_PER_BIN = 4      # min WT per HALF (so a bin needs >= 2*this wildtype to be scored)
MIN_GRP_PER_BIN = 3
SEED = 42

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


def _rms_z(group_rows: pd.DataFrame, ref_rows: pd.DataFrame, feats: list[str]) -> float | None:
    """RMS z-normed distance of group mean from reference, in reference-SD units (per this bin)."""
    if ref_rows[ID_COL].nunique() < MIN_REF_PER_BIN:
        return None
    ref_mean = ref_rows[feats].mean()
    ref_sd = ref_rows[feats].std(ddof=1).replace(0, np.nan)
    z = (group_rows[feats].mean() - ref_mean) / ref_sd
    return float(np.sqrt(np.nanmean(z.values ** 2)))


def distances(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-bin distance distributions for homo / het / wildtype-null via wildtype resampling.

    All variability here comes from resampling WILDTYPE (its mean + per-dim SD is the z-norm
    reference / denominator). Resampling the mutant groups adds only second-order noise, so we do
    NOT bootstrap them -- their band comes from the same wildtype reshuffles.

    Each of N_SHUFFLES iterations, per time bin:
      - randomly split wildtype into two halves A (reference) and B (probe)
      - reference = half A: gives the z-norm mean + SD
      - homozygous / heterozygous : distance( group , A )
      - wildtype_null             : distance( B , A )   # WT-vs-WT, same reference
    Aggregate across shuffles -> mean +/- SD band per series. Because every series uses the same
    per-shuffle reference A, they are directly comparable, and the whole spread is 'how much does
    the distance-from-wildtype wobble as wildtype resamples'.
    """
    rng = np.random.RandomState(SEED)
    rows = []
    for time_bin, bin_df in df.groupby("time_bin"):
        wt_ids = bin_df.loc[bin_df["zygosity"] == BASELINE, ID_COL].drop_duplicates().to_numpy()
        if len(wt_ids) < 2 * MIN_REF_PER_BIN:      # need enough WT to split into two usable halves
            continue
        groups_here = {g: bin_df[bin_df["zygosity"] == g]
                       for g in GROUPS
                       if bin_df.loc[bin_df["zygosity"] == g, ID_COL].nunique() >= MIN_GRP_PER_BIN}

        for shuffle in range(N_SHUFFLES):
            perm = rng.permutation(wt_ids)
            half = len(perm) // 2
            a_ids, b_ids = set(perm[:half]), set(perm[half:])
            ref = bin_df[bin_df[ID_COL].isin(a_ids)]      # reference half A
            probe = bin_df[bin_df[ID_COL].isin(b_ids)]    # WT probe half B
            for group, g in groups_here.items():
                dist = _rms_z(g, ref, feats)
                if dist is not None:
                    rows.append({"time_bin": time_bin, "series": group,
                                 "shuffle": shuffle, "distance": dist})
            dist = _rms_z(probe, ref, feats)
            if dist is not None:
                rows.append({"time_bin": time_bin, "series": "wildtype_null",
                             "shuffle": shuffle, "distance": dist})

    per_shuffle = pd.DataFrame(rows)
    if per_shuffle.empty:
        return per_shuffle
    band = (per_shuffle.groupby(["series", "time_bin"])["distance"]
            .agg(mean="mean", sd="std", n=("count")).reset_index())
    return band


SERIES_STYLE = {
    "homozygous":    dict(color=COLORS["homozygous"],   ls="-",  z=3),
    "heterozygous":  dict(color=COLORS["heterozygous"], ls="-",  z=3),
    "wildtype_null": dict(color=NULL_COLOR,             ls=":",  z=2),
}
SERIES_LABEL = {
    "homozygous": "homozygous", "heterozygous": "heterozygous",
    "wildtype_null": "wildtype null (WT vs WT)",
}
SERIES_ORDER = ["wildtype_null", "heterozygous", "homozygous"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_gene = {}
    for gene in GENE_ORDER:
        df, feats = load_gene(gene)
        per_gene[gene] = distances(df, feats)
        print(f"[{gene}] {len(feats)} dims | "
              f"series={sorted(per_gene[gene]['series'].unique())} | "
              f"bins={per_gene[gene]['time_bin'].nunique()}")

    fig, axes = plt.subplots(1, len(GENE_ORDER), figsize=(11.5, 4.6),
                             sharex=True, sharey=True, squeeze=False)
    for col, gene in enumerate(GENE_ORDER):
        ax = axes[0, col]
        band = per_gene[gene]
        for series in SERIES_ORDER:
            sub = band[band["series"] == series].sort_values("time_bin")
            if sub.empty:
                continue
            st = SERIES_STYLE[series]
            lo, hi = sub["mean"] - sub["sd"], sub["mean"] + sub["sd"]
            ax.fill_between(sub["time_bin"], lo, hi, color=st["color"], alpha=0.20,
                            lw=0, zorder=st["z"])
            ax.plot(sub["time_bin"], sub["mean"], marker="o", ms=4, lw=2.4,
                    color=st["color"], ls=st["ls"], zorder=st["z"] + 1,
                    label=SERIES_LABEL[series])
        ax.set_title(gene, fontsize=13, fontweight="bold")
        ax.set_xlabel("Hours post fertilization (bin)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel("RMS z-normed latent distance\nfrom wildtype (WT-noise units)")
        if col == len(GENE_ORDER) - 1:
            ax.legend(fontsize=9, frameon=True, loc="upper left",
                      title="mean ± 1 SD over 200 WT reshuffles")

    fig.suptitle(
        "Distance from wildtype, each series ± SD over 200 wildtype reshuffles (z-normed, 80-dim)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUTPUT_DIR / "znorm_distance_from_wt_by_zygosity_cvnull.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")

    pd.concat([b.assign(gene=g) for g, b in per_gene.items()],
              ignore_index=True).to_csv(
        OUTPUT_DIR / "znorm_distance_from_wt_cvnull.csv", index=False)


if __name__ == "__main__":
    main()
