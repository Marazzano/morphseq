"""
17_knn_local_global_heatmap.py

Local -> global geometry heatmap.

For every embryo, sweep K (0..30). At each K, in the condensed 2D space and
WITHIN matched time bins, compute the difference between two pooled group means:

    delta(embryo, K) = || mean(focal embedding)  -  mean(K nearest neighbors) ||

  - focal pool = the focal embryo's own position (one point).
  - neighbor pool = its K nearest OTHER embryos at the same time bin.
  - K = 0  -> empty neighbor pool -> the focal embedding baseline (delta defined
    as the focal's distance from... nothing; we report 0 at K=0 as the baseline
    origin, then the delta grows as neighbors are pooled in).
  - per-bin value normalized by that bin's arm scale (mean pairwise distance),
    then averaged over the embryo's valid bins -> one cell.

As K grows, the neighbor pool widens from a local cluster toward the global
population, so reading a row left->right shows how each embryo's offset evolves
from LOCAL to GLOBAL geometry. An embryo that stays offset as K grows is
globally distinct; one whose delta collapses is only locally distinct.

Heatmap: rows = all embryos in sequential order (NO genotype grouping),
cols = K sweep, 4 panels = raw PRE | raw POST | margin PRE | margin POST,
one SHARED arm-normalized color scale across all four. E09 (outlier) and
G12 (control) are marked on the y-axis.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python 17_knn_local_global_heatmap.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

RAW_NPZ = os.path.join(HERE, "figures", "condensed_raw_zmub", "condensed_positions.npz")
MARGIN_NPZ = (
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/"
    "20260407_pbx_analysis_cont/results/positioning/trajectory/"
    "combined_raw_condensation_5class_bin4_perm500/condensed_positions.npz"
)

OUTLIER = "20251207_pbx_E09_e01"
CONTROL = "20251207_pbx_G12_e01"
K_SWEEP = list(range(0, 31))   # 0..30


def load_arm(path, name):
    d = np.load(path, allow_pickle=True)
    return {
        "name": name,
        "positions": d["positions"],   # (N,27,2) POST
        "x0": d["x0"],                 # (N,27,2) PRE
        "mask": d["mask"],             # (N,27)
        "embryo_ids": np.array([str(e) for e in d["embryo_ids"]]),
    }


def all_bin_scales(coords, mask):
    """Per-bin mean pairwise distance among valid embryos (arm scale)."""
    scales = np.full(mask.shape[1], np.nan)
    for b in range(mask.shape[1]):
        valid = np.where(mask[:, b])[0]
        if len(valid) < 2:
            continue
        P = coords[valid, b, :]
        diff = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        iu = np.triu_indices(len(valid), k=1)
        scales[b] = dist[iu].mean()
    return scales


def knn_group_delta(coords, mask, fi, scales, ks):
    """
    For focal index fi, return an array over `ks` of:
        mean_over_valid_bins( || focal_pos - mean(K nearest neighbor pos) || / arm_scale )
    K=0 -> 0.0 baseline (no neighbor pool yet).
    """
    out = np.full(len(ks), np.nan)
    # accumulate per-bin sorted neighbor positions once, reuse across K
    per_bin = []  # (focal_pos, sorted_neighbor_pos, scale)
    for b in range(mask.shape[1]):
        if not mask[fi, b]:
            continue
        s = scales[b]
        if not np.isfinite(s) or s == 0:
            continue
        others = np.where(mask[:, b])[0]
        others = others[others != fi]
        if len(others) == 0:
            continue
        fp = coords[fi, b, :]
        dd = np.sqrt(((coords[others, b, :] - fp) ** 2).sum(-1))
        order = np.argsort(dd)
        per_bin.append((fp, coords[others[order], b, :], s))

    if not per_bin:
        return out

    for ki, k in enumerate(ks):
        vals = []
        for fp, nbr_sorted, s in per_bin:
            if k == 0:
                vals.append(0.0)
                continue
            kk = min(k, len(nbr_sorted))
            if kk == 0:
                continue
            centroid = nbr_sorted[:kk].mean(axis=0)
            vals.append(np.linalg.norm(fp - centroid) / s)
        if vals:
            out[ki] = np.nanmean(vals)
    return out


def build_matrix(arm, coords_key):
    """(n_embryos, n_K) delta matrix for the given arm and state (x0 or positions)."""
    coords = arm[coords_key]
    mask = arm["mask"]
    scales = all_bin_scales(coords, mask)
    n = len(arm["embryo_ids"])
    M = np.full((n, len(K_SWEEP)), np.nan)
    for fi in range(n):
        if mask[fi].sum() == 0:
            continue
        M[fi] = knn_group_delta(coords, mask, fi, scales, K_SWEEP)
    return M


def main():
    print("Loading arms...")
    arms = {"raw": load_arm(RAW_NPZ, "raw"), "margin": load_arm(MARGIN_NPZ, "margin")}

    panels = [
        ("raw", "x0", "raw  PRE"),
        ("raw", "positions", "raw  POST"),
        ("margin", "x0", "margin  PRE"),
        ("margin", "positions", "margin  POST"),
    ]

    mats = {}
    for aname, ckey, title in panels:
        print(f"  building {title} ...")
        mats[(aname, ckey)] = build_matrix(arms[aname], ckey)

    # shared color scale across all 4 panels
    allvals = np.concatenate([m[np.isfinite(m)].ravel() for m in mats.values()])
    vmin, vmax = np.nanpercentile(allvals, 1), np.nanpercentile(allvals, 99)
    print(f"  shared color scale: vmin={vmin:.3f} vmax={vmax:.3f}")

    # marked embryos, per arm (row index differs only if id order differs; it doesn't here)
    def row_of(aname, emb):
        ids = list(arms[aname]["embryo_ids"])
        return ids.index(emb) if emb in ids else None

    fig, axes = plt.subplots(1, 4, figsize=(20, 9), squeeze=False, sharey=False)
    im = None
    for ci, (aname, ckey, title) in enumerate(panels):
        ax = axes[0][ci]
        M = mats[(aname, ckey)]
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                       extent=[K_SWEEP[0] - 0.5, K_SWEEP[-1] + 0.5, M.shape[0] - 0.5, -0.5],
                       interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("K (nearest neighbors pooled)")
        if ci == 0:
            ax.set_ylabel("embryo (sequential order)")
        # mark E09 / G12
        for emb, lbl, col in [(OUTLIER, "E09 (outlier)", "#00e5ff"),
                              (CONTROL, "G12 (control)", "#7CFC00")]:
            r = row_of(aname, emb)
            if r is not None:
                ax.axhline(r, color=col, lw=1.1, alpha=0.9)
                if ci == 3:
                    ax.text(K_SWEEP[-1] + 1, r, lbl, color=col, fontsize=8,
                            va="center", ha="left")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.06)
    cbar.set_label("|| mean(focal) - mean(K-NN) ||  (arm-normalized)")
    fig.suptitle(
        "Local -> global geometry: pooled-mean offset vs K, per embryo\n"
        "cell = ||mean(focal) - mean(K nearest neighbors)|| (arm-normalized, matched bins); "
        "rows = all embryos sequential; read left->right = local to global",
        fontsize=13, fontweight="bold")
    p = os.path.join(FIG, "knn_local_global_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")

    # tidy table (long) for reuse
    rows = []
    for (aname, ckey), M in mats.items():
        ids = arms[aname]["embryo_ids"]
        for fi in range(M.shape[0]):
            for ki, k in enumerate(K_SWEEP):
                if np.isfinite(M[fi, ki]):
                    rows.append({"arm": aname, "state": ckey, "embryo_id": ids[fi],
                                 "k": k, "delta": float(M[fi, ki])})
    df = pd.DataFrame(rows)
    tp = os.path.join(TAB, "knn_local_global_delta.csv")
    df.to_csv(tp, index=False)
    print(f"[saved] {tp}  ({len(df)} rows)")

    # quick E09 vs population readout at a few K, POST arms
    print("\nE09 vs population median delta (POST):")
    for aname in ("raw", "margin"):
        M = mats[(aname, "positions")]
        r = row_of(aname, OUTLIER)
        for k in (2, 10, 20, 30):
            ki = K_SWEEP.index(k)
            col = M[:, ki]
            e09 = M[r, ki]
            med = np.nanmedian(col)
            print(f"  {aname} K={k:>2}: E09={e09:.3f}  pop_median={med:.3f}  "
                  f"ratio={e09/med if med else np.nan:.2f}")


if __name__ == "__main__":
    main()
