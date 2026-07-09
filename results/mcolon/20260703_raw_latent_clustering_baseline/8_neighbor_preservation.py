"""
8_neighbor_preservation.py
--------------------------
Does the classification (margin) representation PRESERVE the raw z_mu_b
manifold, or REORGANIZE it?

This is the question that comes BEFORE interpreting raw-space islands as
timing / batch / phenotype / noise. Two possibilities:

  P1  Same organization, weaker valleys. Embryos keep their neighbors; margin
      space merely softens/compresses the modes seen in raw space.
  P2  Fundamental reorganization. Embryos that were neighbors in raw space are
      no longer neighbors in margin space -> margin builds a different geometry.

Primary diagnostic = local neighbor preservation, computed in the ORIGINAL
feature spaces (not UMAP), WITHIN each time bin (embryos at the same stage are
the meaningful comparison population):

    overlap_i = |N_i^raw  ∩  N_i^margin| / k

Spaces:
  raw80    : 80-dim z_mu_b               (tables/pbx_binned_zmub.csv)
  margin10 : 10-dim pairwise signed margins
             (.../combined_pairwise_5class_bin4_perm500/pairwise_raw_vectors.csv)
             -- the exact features that fed the margin condensation baseline.
  rawPCd   : first d=10 PCs of raw80     (FAIR-COMPRESSION CONTROL)

The rawPCd control is essential: margin10 is 10-dim while raw80 is 80-dim, so
SOME neighbor loss is guaranteed by dimensional compression alone. The honest
question is whether margin10 preserves FEWER neighbors than an equally
low-dimensional linear projection of raw itself. If margin ~ rawPCd, the
"reorganization" is mostly just compression. If margin << rawPCd, the classifier
is specifically reordering embryos.

Also computed:
  - Spearman rank agreement of within-bin pairwise distances (raw vs margin).
  - Raw-vs-margin per-bin KMeans contingency + merging-vs-mixing entropy.

Outputs:
  tables/neighbor_preservation_by_bin.csv        (per bin x k x space-pair)
  tables/neighbor_preservation_summary.csv       (pooled over bins, per k)
  tables/pairwise_distance_spearman_by_bin.csv
  tables/cluster_contingency_summary.csv
  figures/neighbor_preservation.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"

RAW_CSV = TABLES / "pbx_binned_zmub.csv"
MARGIN_CSV = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/"
    "20260407_pbx_analysis_cont/results/positioning/pairwise/"
    "combined_pairwise_5class_bin4_perm500/pairwise_raw_vectors.csv"
)

K_VALUES = (5, 10, 15, 30, 50)
PC_D = 10                 # rawPCd control dimensionality (= margin dim)
MIN_BIN_N = 20            # bins with fewer joint embryos are skipped (need k+ pts)
RANDOM_STATE = 0


# ── loading / joining ────────────────────────────────────────────────────────
def load_joined():
    """Return long frame with columns: embryo_id, time_bin, genotype,
    raw80 block (z_mu_b_*), margin10 block. Inner-joined on (embryo_id,time_bin)."""
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    mar = pd.read_csv(MARGIN_CSV, low_memory=False)

    z_cols = [c for c in raw.columns if "z_mu_b" in c]
    m_cols = [c for c in mar.columns if "__vs__" in c]

    raw_keep = raw[["embryo_id", "time_bin", "genotype", *z_cols]].copy()
    mar_keep = mar[["embryo_id", "time_bin", *m_cols]].copy()

    joined = raw_keep.merge(mar_keep, on=["embryo_id", "time_bin"], how="inner")
    return joined, z_cols, m_cols


def impute_margin(M):
    """Column-mean impute NaNs in a margin block (within the calling scope).
    Columns that are all-NaN in this scope become 0 (no information)."""
    M = M.copy()
    col_mean = np.nanmean(M, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    idx = np.where(~np.isfinite(M))
    M[idx] = np.take(col_mean, idx[1])
    return M


def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def pca_project(X, d):
    """First-d PC projection (centered). d capped at rank."""
    Xc = X - X.mean(axis=0)
    d = min(d, min(Xc.shape) - 1)
    if d < 1:
        return Xc[:, :1]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:d].T


# ── neighbor overlap ─────────────────────────────────────────────────────────
def knn_indices(X, k):
    """For each row, indices of its k nearest neighbors (excluding self)."""
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def mean_overlap(nn_a, nn_b, k):
    """Mean over rows of |A_i ∩ B_i| / k for the first-k neighbor sets."""
    n = nn_a.shape[0]
    ov = np.empty(n)
    for i in range(n):
        a = set(nn_a[i, :k].tolist())
        b = set(nn_b[i, :k].tolist())
        ov[i] = len(a & b) / k
    return ov.mean(), ov


# ── per-bin computation ──────────────────────────────────────────────────────
def analyze_bin(sub, z_cols, m_cols):
    """All diagnostics for one time bin. Returns dict of records."""
    n = len(sub)
    kmax = max(K_VALUES)
    if n < max(MIN_BIN_N, kmax + 1):
        return None

    raw80 = zscore(sub[z_cols].values.astype(float))
    margin = zscore(impute_margin(sub[m_cols].values.astype(float)))
    rawPCd = zscore(pca_project(sub[z_cols].values.astype(float), PC_D))

    nn_raw = knn_indices(raw80, kmax)
    nn_mar = knn_indices(margin, kmax)
    nn_pcd = knn_indices(rawPCd, kmax)

    recs = []
    for k in K_VALUES:
        m_rm, _ = mean_overlap(nn_raw, nn_mar, k)   # raw80 vs margin10 (the test)
        m_rp, _ = mean_overlap(nn_raw, nn_pcd, k)   # raw80 vs rawPC10 (control)
        recs.append(dict(pair="raw80_vs_margin10", k=k, overlap=m_rm, n=n))
        recs.append(dict(pair="raw80_vs_rawPC10", k=k, overlap=m_rp, n=n))

    # pairwise-distance rank agreement (raw vs margin), sampled upper triangle
    Dr = pdist(raw80)
    Dm = pdist(margin)
    rho = spearmanr(Dr, Dm).statistic if len(Dr) > 2 else np.nan

    # merging vs mixing: independent KMeans in each space, contingency entropy
    ncl = min(4, n // 10) if n >= 20 else 2
    contingency_stats = cluster_contingency(raw80, margin, ncl)

    return dict(records=recs, rho=rho, n=n, ncl=ncl, **contingency_stats)


def cluster_contingency(raw80, margin, ncl):
    """KMeans in each space independently; summarize the raw->margin transition.

    Returns mean conditional entropy H(margin_cluster | raw_cluster), normalized
    by log(ncl). Low  -> coherent MERGING (each raw cluster maps to ~one margin
    cluster). High -> MIXING (raw clusters scatter across margin clusters)."""
    if ncl < 2:
        return dict(norm_cond_entropy=np.nan)
    cr = KMeans(ncl, n_init=10, random_state=RANDOM_STATE).fit_predict(raw80)
    cm = KMeans(ncl, n_init=10, random_state=RANDOM_STATE).fit_predict(margin)
    ent = []
    weights = []
    for a in np.unique(cr):
        rows = cm[cr == a]
        counts = np.bincount(rows, minlength=ncl).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        h = -(p * np.log(p)).sum() / np.log(ncl)
        ent.append(h)
        weights.append(len(rows))
    weights = np.array(weights, float)
    return dict(norm_cond_entropy=float(np.average(ent, weights=weights)))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    joined, z_cols, m_cols = load_joined()
    print(f"Joined (embryo_id,time_bin) rows: {len(joined)}  "
          f"embryos: {joined.embryo_id.nunique()}  bins: {joined.time_bin.nunique()}")
    print(f"raw dim={len(z_cols)}  margin dim={len(m_cols)}  PC control dim={PC_D}\n")

    per_bin = []
    rho_rows = []
    cont_rows = []
    for b, sub in joined.groupby("time_bin"):
        out = analyze_bin(sub, z_cols, m_cols)
        if out is None:
            continue
        for r in out["records"]:
            per_bin.append(dict(time_bin=b, **r))
        rho_rows.append(dict(time_bin=b, n=out["n"], spearman_rho=out["rho"]))
        cont_rows.append(dict(time_bin=b, n=out["n"], ncl=out["ncl"],
                              norm_cond_entropy=out["norm_cond_entropy"]))

    pb = pd.DataFrame(per_bin)
    pb.to_csv(TABLES / "neighbor_preservation_by_bin.csv", index=False)

    # pooled summary (mean over bins, n-weighted), per k, per pair
    summ = (pb.groupby(["pair", "k"])
              .apply(lambda g: np.average(g.overlap, weights=g.n), include_groups=False)
              .rename("overlap_weighted").reset_index())
    summ.to_csv(TABLES / "neighbor_preservation_summary.csv", index=False)

    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(TABLES / "pairwise_distance_spearman_by_bin.csv", index=False)

    cont_df = pd.DataFrame(cont_rows)
    cont_df.to_csv(TABLES / "cluster_contingency_summary.csv", index=False)

    # ── report ────────────────────────────────────────────────────────────────
    print("=== Neighbor preservation (pooled over bins, n-weighted) ===")
    piv = summ.pivot(index="k", columns="pair", values="overlap_weighted")
    piv["ratio_margin_over_PCcontrol"] = (
        piv["raw80_vs_margin10"] / piv["raw80_vs_rawPC10"])
    print(piv.round(3).to_string())

    print(f"\n=== Pairwise-distance Spearman (raw vs margin), per bin ===")
    print(f"  mean rho = {rho_df.spearman_rho.mean():.3f}  "
          f"(min {rho_df.spearman_rho.min():.3f}, max {rho_df.spearman_rho.max():.3f})")

    print(f"\n=== Merging vs mixing (norm. conditional entropy H(margin|raw)) ===")
    print(f"  mean = {cont_df.norm_cond_entropy.mean():.3f}  "
          f"(0=coherent merging, 1=full mixing)")

    # ── verdict heuristic ─────────────────────────────────────────────────────
    k10 = piv.loc[10]
    margin_ov = k10["raw80_vs_margin10"]
    pc_ov = k10["raw80_vs_rawPC10"]
    ratio = margin_ov / pc_ov if pc_ov > 0 else np.nan
    print("\n=== VERDICT (k=10) ===")
    print(f"  margin preserves {margin_ov:.1%} of raw neighbors; "
          f"a fair {PC_D}-dim PCA projection preserves {pc_ov:.1%}.")
    if ratio >= 0.75:
        print(f"  -> ratio {ratio:.2f}: margin's neighbor loss is ~explained by "
              f"dimensional compression. Consistent with P1 (same organization, "
              f"weaker valleys); little classifier-specific reorganization.")
    elif ratio >= 0.45:
        print(f"  -> ratio {ratio:.2f}: margin loses MORE neighbors than compression "
              f"alone predicts. Partial classifier-specific reorganization.")
    else:
        print(f"  -> ratio {ratio:.2f}: margin loses FAR more neighbors than a fair "
              f"low-dim projection. Consistent with P2 (fundamental reorganization) — "
              f"the classifier constructs a different similarity geometry.")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for pair, style in [("raw80_vs_margin10", dict(color="#B2182B", marker="o")),
                        ("raw80_vs_rawPC10", dict(color="#2166AC", marker="s"))]:
        g = summ[summ.pair == pair].sort_values("k")
        ax.plot(g.k, g.overlap_weighted, label=pair, lw=2, **style)
    ax.set_xlabel("k (neighborhood size)")
    ax.set_ylabel("mean neighbor overlap fraction")
    ax.set_title("Local neighbor preservation\n(margin vs fair-compression control)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rho_df.time_bin, rho_df.spearman_rho, "o-", color="#333")
    ax.set_xlabel("time bin (hpf)")
    ax.set_ylabel("Spearman rho (pairwise dist)")
    ax.set_title("Global distance-rank agreement\nraw vs margin, per bin")
    ax.set_ylim(-0.1, 1)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(cont_df.time_bin, cont_df.norm_cond_entropy, "o-", color="#666")
    ax.axhline(0.0, ls="--", c="green", alpha=0.5)
    ax.set_xlabel("time bin (hpf)")
    ax.set_ylabel("norm. cond. entropy H(margin|raw)")
    ax.set_title("Merging (0) vs mixing (1)\nper bin")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "neighbor_preservation.png", dpi=130)
    print(f"\nSaved -> figures/neighbor_preservation.png")
    print(f"Saved -> tables/neighbor_preservation_by_bin.csv, "
          f"neighbor_preservation_summary.csv, pairwise_distance_spearman_by_bin.csv, "
          f"cluster_contingency_summary.csv")


if __name__ == "__main__":
    main()
