"""
17_knn_pool_delta.py

Per-dimension pooled-neighborhood delta between two focal embryos, measured in
RAW z_mu_b space, across four arm-states, as an 8-panel figure (no overlays).

Focal pair:
  - 20251207_pbx_E09_e01  (outlier)
  - 20251207_pbx_G12_e01  (control)

For each K (0..30), each arm-state:
  Pool A = E09 + its K nearest neighbors (selected in that arm-state's condensed 2D)
  Pool B = G12 + its K nearest neighbors (same)
  Every pool member contributes its RAW z_mu_b (80-dim) vector.
  per-dim delta(K)[d] = mean_A(z_mu_b)[d] - mean_B(z_mu_b)[d]      (signed, in z_mu_b)

Only NEIGHBOR SELECTION differs by arm-state; the delta is always measured in
z_mu_b. So each heatmap shows WHICH z_mu_b dimensions carry the E09-vs-G12
difference and how each collapses as K grows -- and comparing arm-states shows
which dims margin's reorganization removes vs preserves.

Four arm-states = {raw, margin} x {PRE (x0), POST (positions)}.
Each arm-state gets TWO panels:
  (1) per-dim K&N difference heatmap  (rows = z_mu_b dims 20..99 natural order,
      cols = K, signed diverging, SHARED color scale across all 4 heatmaps)
  (2) condensed 2D slice at E09's ~72hpf bin, points colored by distance-to-E09,
      E09/G12 bolded.
=> 4 rows x 2 cols = 8 separate panels, nothing overlaid.

The margin npz uses time bins offset +2 from the raw npz / z_mu_b table; margin
(embryo,bin) is snapped to the embryo's nearest RAW bin for the z_mu_b lookup.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python 17_knn_pool_delta.py
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

RAW_NPZ = os.path.join(HERE, "figures", "condensed_raw_zmub", "condensed_positions.npz")
MARGIN_NPZ = (
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/"
    "20260407_pbx_analysis_cont/results/positioning/trajectory/"
    "combined_raw_condensation_5class_bin4_perm500/condensed_positions.npz"
)
ZTBL = os.path.join(TAB, "pbx_binned_zmub_with_wt.csv")

E09 = "20251207_pbx_E09_e01"
G12 = "20251207_pbx_G12_e01"
K_SWEEP = list(range(0, 31))


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {
        "positions": d["positions"],
        "x0": d["x0"],
        "mask": d["mask"],
        "time_values": d["time_values"],
        "embryo_ids": np.array([str(e) for e in d["embryo_ids"]]),
    }


def build_zlookup():
    tbl = pd.read_csv(ZTBL, low_memory=False)
    zc = [c for c in tbl.columns if c.startswith("z_mu_b")]      # z_mu_b_20_binned..99
    tbl["time_bin"] = tbl["time_bin"].astype(float)
    look = {}
    arr = tbl[["embryo_id", "time_bin"] + zc].to_numpy(dtype=object)
    for row in arr:
        look[(row[0], float(row[1]))] = np.asarray(row[2:], dtype=float)
    raw_bins = np.array(sorted(tbl["time_bin"].unique()))
    dim_labels = [c.replace("z_mu_b_", "").replace("_binned", "") for c in zc]  # "20".."99"
    return look, zc, raw_bins, dim_labels


def zvec(zlook, raw_bins, embryo_id, npz_bin):
    v = zlook.get((embryo_id, float(npz_bin)))
    if v is not None:
        return v
    snapped = float(raw_bins[np.argmin(np.abs(raw_bins - npz_bin))])
    return zlook.get((embryo_id, snapped))


def knn_order(arm, coords, focal, b):
    ids = list(arm["embryo_ids"])
    fi = ids.index(focal)
    if not arm["mask"][fi, b]:
        return fi, None
    others = np.where(arm["mask"][:, b])[0]
    others = others[others != fi]
    if len(others) == 0:
        return fi, np.array([], dtype=int)
    fp = coords[fi, b, :]
    dd = np.sqrt(((coords[others, b, :] - fp) ** 2).sum(-1))
    return fi, others[np.argsort(dd)]


def pool_zmean(arm, coords, focal, k, zlook, raw_bins):
    """Mean z_mu_b vector over {focal + k NN}, averaged across the focal's valid bins."""
    ids = list(arm["embryo_ids"])
    fi = ids.index(focal)
    tv = arm["time_values"]
    per_bin = []
    for b in range(arm["mask"].shape[1]):
        if not arm["mask"][fi, b]:
            continue
        _fi, order = knn_order(arm, coords, focal, b)
        if order is None:
            continue
        members = [focal] + [ids[j] for j in order[:k]]
        vecs = [zvec(zlook, raw_bins, m, tv[b]) for m in members]
        vecs = [v for v in vecs if v is not None]
        if vecs:
            per_bin.append(np.mean(vecs, axis=0))
    return np.mean(per_bin, axis=0) if per_bin else None


def dim_delta_matrix(arm, coords, zlook, raw_bins, n_dims):
    """(n_dims, n_K) signed per-dim delta = mean(E09 pool) - mean(G12 pool)."""
    M = np.full((n_dims, len(K_SWEEP)), np.nan)
    for ki, k in enumerate(K_SWEEP):
        mA = pool_zmean(arm, coords, E09, k, zlook, raw_bins)
        mB = pool_zmean(arm, coords, G12, k, zlook, raw_bins)
        if mA is not None and mB is not None:
            M[:, ki] = mA - mB
    return M


def pick_bin(arm, target=72.0):
    tv = arm["time_values"]
    fi = list(arm["embryo_ids"]).index(E09)
    valid = [b for b in range(len(tv)) if arm["mask"][fi, b]]
    return min(valid, key=lambda b: abs(tv[b] - target))


def main():
    raw = load_npz(RAW_NPZ)
    margin = load_npz(MARGIN_NPZ)
    zlook, zc, raw_bins, dim_labels = build_zlookup()
    n_dims = len(zc)
    print(f"z_mu_b: {n_dims} dims ({dim_labels[0]}..{dim_labels[-1]}), "
          f"raw bins {raw_bins.min()}..{raw_bins.max()}, "
          f"margin bins {margin['time_values'].min()}..{margin['time_values'].max()}")

    # 4 arm-states: (name, npz, coords_key)
    states = [
        ("raw  PRE", raw, "x0"),
        ("raw  POST", raw, "positions"),
        ("margin  PRE", margin, "x0"),
        ("margin  POST", margin, "positions"),
    ]

    # per-dim delta heatmaps
    mats = {}
    for name, arm, ckey in states:
        print(f"  building dim-delta {name} ...")
        mats[name] = dim_delta_matrix(arm, arm[ckey], zlook, raw_bins, n_dims)

    # shared diverging scale across all 4 heatmaps (symmetric about 0)
    allv = np.concatenate([m[np.isfinite(m)].ravel() for m in mats.values()])
    vlim = np.nanpercentile(np.abs(allv), 99)
    print(f"  shared diverging scale: +/-{vlim:.3f}")

    # long table
    rows = []
    for name in mats:
        for di, dl in enumerate(dim_labels):
            for ki, k in enumerate(K_SWEEP):
                if np.isfinite(mats[name][di, ki]):
                    rows.append({"arm_state": name, "dim": dl, "k": k,
                                 "delta_zmub": float(mats[name][di, ki])})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "knn_pool_dim_delta.csv"), index=False)

    # ---- 8-panel figure: 4 rows (arm-states) x 2 cols (heatmap | slice) ----
    fig, axes = plt.subplots(4, 2, figsize=(16, 22),
                             gridspec_kw={"width_ratios": [1.4, 1.0]})
    yt = np.arange(0, n_dims, 5)
    im = None
    for ri, (name, arm, ckey) in enumerate(states):
        # (1) per-dim heatmap
        axh = axes[ri][0]
        im = axh.imshow(mats[name], aspect="auto", cmap="RdBu_r",
                        vmin=-vlim, vmax=vlim, interpolation="nearest",
                        extent=[K_SWEEP[0] - 0.5, K_SWEEP[-1] + 0.5, n_dims - 0.5, -0.5])
        axh.set_title(f"{name}  |  per-dim z_mu_b delta  (E09 pool - G12 pool)", fontsize=11, fontweight="bold")
        axh.set_xlabel("K (nearest neighbors pooled)")
        axh.set_ylabel("z_mu_b dim")
        axh.set_yticks(yt)
        axh.set_yticklabels([dim_labels[i] for i in yt], fontsize=7)

        # (2) condensed 2D slice colored by distance-to-E09
        axs = axes[ri][1]
        coords = arm[ckey]
        b = pick_bin(arm)
        ids = list(arm["embryo_ids"])
        fi = ids.index(E09)
        valid = np.where(arm["mask"][:, b])[0]
        P = coords[valid, b, :]
        fp = coords[fi, b, :]
        dist = np.sqrt(((P - fp) ** 2).sum(-1))
        sc = axs.scatter(P[:, 0], P[:, 1], c=dist, cmap="viridis", s=20, edgecolor="none", alpha=0.85)
        axs.scatter(fp[0], fp[1], s=170, marker="*", color="#e41a1c",
                    edgecolor="black", linewidth=1.3, label="E09 (focal)", zorder=5)
        if G12 in ids and arm["mask"][ids.index(G12), b]:
            gp = coords[ids.index(G12), b, :]
            axs.scatter(gp[0], gp[1], s=140, marker="D", color="#377eb8",
                        edgecolor="black", linewidth=1.3, label="G12 (control)", zorder=5)
        axs.set_title(f"{name}  |  condensed 2D @ {arm['time_values'][b]:.0f}hpf\n(colored by dist to E09)", fontsize=10)
        axs.legend(fontsize=8, loc="best")
        fig.colorbar(sc, ax=axs, fraction=0.046, pad=0.04).set_label("dist to E09", fontsize=8)

    cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("signed z_mu_b delta  (red = E09 pool higher, blue = G12 pool higher)")
    fig.suptitle("Per-dimension z_mu_b pool-delta (E09 outlier vs G12 control) across 4 arm-states + condensed slices",
                 fontsize=14, fontweight="bold", y=0.995)
    p = os.path.join(FIG, "knn_pool_dim_delta.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
