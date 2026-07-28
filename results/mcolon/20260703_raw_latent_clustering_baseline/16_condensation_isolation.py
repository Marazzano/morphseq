"""
16_condensation_isolation.py

Hypothesis: condensation pushes an outlier embryo away from its neighbors.
The RAW-latent condensation arm correctly separates the outlier; the MARGIN
(classifier) arm AMPLIFIES that separation. A control embryo that sits within
the main distribution should stay put in both arms.

Contrast pair (tested separately):
  - 20251207_pbx_E09_e01 = THE OUTLIER  (expect distances to grow pre->post, more in margin)
  - 20251207_pbx_G12_e01 = THE CONTROL  (expect roughly flat pre->post in both arms)

Three steps:
  1. Isolation (distance-to-bulk + NN dist) over condensation iterations, 2x2
     {E09,G12} x {raw,margin}. Arm-normalized.
  2. K=5/15 neighbors at POST, genotype-colored + difference test (who / reorganized?).
  3. (HEADLINE) mean distance to K-NN, PRE (x0) vs POST (positions), both arms,
     arm-normalized: "raw separates, margin amplifies, control flat".

All distances are computed in the condensed 2D space, WITHIN matched time bins
(same-stage), always masking invalid (embryo,bin) cells, and normalized by each
arm's own per-bin mean pairwise distance so raw and margin are comparable.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python 16_condensation_isolation.py
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
FOCALS = {OUTLIER: "outlier", CONTROL: "control"}
KS = [5, 15]

# Local genotype color map (the shared config is suffix-based; these are full pbx strings)
GENO_COLORS = {
    "pbx1b_pbx4_crispant": "#B2182B",
    "pbx4_crispant": "#EF8A62",
    "pbx1b_crispant": "#F7B267",
    "inj_ctrl": "#2166AC",
    "wik_ab": "#67A9CF",
}
def geno_color(g):
    return GENO_COLORS.get(str(g), "#808080")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_arm(path, name):
    d = np.load(path, allow_pickle=True)
    arm = {
        "name": name,
        "positions": d["positions"],           # (N,27,2) POST
        "x0": d["x0"],                          # (N,27,2) PRE
        "mask": d["mask"],                      # (N,27)
        "time_values": d["time_values"],        # (27,)
        "embryo_ids": np.array([str(e) for e in d["embryo_ids"]]),
        "labels": np.array([str(l) for l in d["labels"]], dtype=object),
        "position_history": d["position_history"],  # (S,N,27,2)
        "snapshot_iters": d["snapshot_iters"],       # (S,)
    }
    arm["id_to_idx"] = {e: i for i, e in enumerate(arm["embryo_ids"])}
    return arm


def focal_idx(arm, embryo_id):
    if embryo_id not in arm["id_to_idx"]:
        raise ValueError(f"{embryo_id} not in {arm['name']} arm")
    return arm["id_to_idx"][embryo_id]


# ----------------------------------------------------------------------------
# Per-bin arm scale (mean pairwise distance among valid embryos at that bin)
# ----------------------------------------------------------------------------
def bin_scale(coords, mask, b):
    """Mean pairwise distance among valid embryos at time bin b, in `coords`."""
    valid = np.where(mask[:, b])[0]
    if len(valid) < 2:
        return np.nan
    P = coords[valid, b, :]                      # (m,2)
    diff = P[:, None, :] - P[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(valid), k=1)
    return dist[iu].mean()


def all_bin_scales(coords, mask):
    return np.array([bin_scale(coords, mask, b) for b in range(mask.shape[1])])


# ----------------------------------------------------------------------------
# Isolation metrics for a focal embryo in a given coordinate array
# ----------------------------------------------------------------------------
def focal_isolation(coords, mask, fi, scales, k=None):
    """
    For focal index fi in `coords` (N,27,2), at each of its valid bins compute:
      - dist_to_bulk : mean distance to all OTHER valid embryos at that bin
      - nn_dist      : distance to nearest OTHER valid embryo at that bin
      - mean_knn     : mean distance to its k nearest OTHER valid embryos (if k given)
    Each per-bin value normalized by that bin's arm scale, then averaged over the
    focal embryo's valid bins.
    Returns dict of averaged normalized metrics.
    """
    d2b, nn, knn = [], [], []
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
        p = coords[fi, b, :]
        dd = np.sqrt(((coords[others, b, :] - p) ** 2).sum(-1)) / s
        dd_sorted = np.sort(dd)
        d2b.append(dd.mean())
        nn.append(dd_sorted[0])
        if k is not None:
            kk = min(k, len(dd_sorted))
            knn.append(dd_sorted[:kk].mean())
    out = {
        "dist_to_bulk": np.nanmean(d2b) if d2b else np.nan,
        "nn_dist": np.nanmean(nn) if nn else np.nan,
    }
    if k is not None:
        out["mean_knn"] = np.nanmean(knn) if knn else np.nan
    return out


def neighbor_dist_reference(coords, mask, fi, k):
    """Raw difference test: focal embryo's mean normalized K-NN distance vs.
    the population baseline = mean/median of EVERY other embryo's mean K-NN
    distance, computed at the focal embryo's own valid bins (same-stage).
    One pass, no resampling. Returns dict with the focal value, the bulk
    baseline, and the raw difference / ratio (how many x the bulk).
    """
    scales = all_bin_scales(coords, mask)
    valid_bins = [b for b in range(mask.shape[1]) if mask[fi, b]
                  and np.isfinite(scales[b]) and scales[b] != 0]
    if not valid_bins:
        return dict(focal=np.nan, bulk_mean=np.nan, bulk_median=np.nan,
                    diff=np.nan, ratio=np.nan)

    def mean_knn_for(idx):
        vals = []
        for b in valid_bins:
            others = np.where(mask[:, b])[0]
            others = others[others != idx]
            if len(others) == 0:
                continue
            p = coords[idx, b, :]
            dd = np.sqrt(((coords[others, b, :] - p) ** 2).sum(-1)) / scales[b]
            vals.append(np.sort(dd)[:min(k, len(dd))].mean())
        return np.nanmean(vals) if vals else np.nan

    focal = mean_knn_for(fi)
    # every other embryo valid at ALL of the focal's bins (matched pool)
    cand = None
    for b in valid_bins:
        s = set(np.where(mask[:, b])[0].tolist())
        cand = s if cand is None else (cand & s)
    cand.discard(fi)
    cand = np.array(sorted(cand))
    others_vals = np.array([mean_knn_for(int(c)) for c in cand]) if cand.size else np.array([])
    others_vals = others_vals[np.isfinite(others_vals)]
    bulk_mean = float(others_vals.mean()) if others_vals.size else np.nan
    bulk_median = float(np.median(others_vals)) if others_vals.size else np.nan
    return dict(focal=focal, bulk_mean=bulk_mean, bulk_median=bulk_median,
                diff=focal - bulk_mean, ratio=focal / bulk_mean if bulk_mean else np.nan)


def neighbor_rows(coords, mask, fi, labels, k):
    """
    Collect the focal embryo's k nearest OTHER valid neighbors at EACH valid bin.
    Returns a list of dicts (bin, time, neighbor_embryo_idx, genotype, norm_dist).
    Uses arm scale per bin for norm_dist.
    """
    scales = all_bin_scales(coords, mask)
    rows = []
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
        p = coords[fi, b, :]
        dd = np.sqrt(((coords[others, b, :] - p) ** 2).sum(-1)) / s
        order = np.argsort(dd)
        for j in order[:min(k, len(order))]:
            oi = others[j]
            rows.append({
                "bin": b,
                "neighbor_idx": int(oi),
                "genotype": str(labels[oi]),
                "norm_dist": float(dd[j]),
            })
    return rows


# ----------------------------------------------------------------------------
# STEP 1 : isolation over condensation iterations (2x2)
# ----------------------------------------------------------------------------
def step1(arms):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    rows_out = []
    focal_list = [(OUTLIER, "outlier"), (CONTROL, "control")]
    arm_list = [("raw", arms["raw"]), ("margin", arms["margin"])]

    for r, (emb, role) in enumerate(focal_list):
        for c, (aname, arm) in enumerate(arm_list):
            ax = axes[r][c]
            fi = focal_idx(arm, emb)
            hist = arm["position_history"]      # (S,N,27,2)
            S = hist.shape[0]
            mask = arm["mask"]
            iters = arm["snapshot_iters"]
            d2b_curve, nn_curve = [], []
            for s_i in range(S):
                coords = hist[s_i]
                scales = all_bin_scales(coords, mask)
                iso = focal_isolation(coords, mask, fi, scales)
                d2b_curve.append(iso["dist_to_bulk"])
                nn_curve.append(iso["nn_dist"])
            # PRE (x0) and POST (positions) reference points
            pre_scales = all_bin_scales(arm["x0"], mask)
            post_scales = all_bin_scales(arm["positions"], mask)
            pre = focal_isolation(arm["x0"], mask, fi, pre_scales)
            post = focal_isolation(arm["positions"], mask, fi, post_scales)

            ax.plot(range(S), d2b_curve, "-o", ms=3, color="#B2182B", label="dist-to-bulk")
            ax.plot(range(S), nn_curve, "-s", ms=3, color="#2166AC", label="NN dist")
            ax.axhline(pre["dist_to_bulk"], ls=":", color="#B2182B", alpha=0.5)
            ax.axhline(post["dist_to_bulk"], ls="--", color="#B2182B", alpha=0.7)
            ax.set_title(f"{emb}\n[{role}]  |  {aname} arm", fontsize=10)
            ax.set_xlabel("condensation snapshot")
            ax.set_ylabel("normalized isolation")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

            rows_out.append({
                "embryo": emb, "role": role, "arm": aname,
                "d2b_pre": pre["dist_to_bulk"], "d2b_post": post["dist_to_bulk"],
                "nn_pre": pre["nn_dist"], "nn_post": post["nn_dist"],
                "d2b_delta": post["dist_to_bulk"] - pre["dist_to_bulk"],
                "nn_delta": post["nn_dist"] - pre["nn_dist"],
            })

    fig.suptitle(
        "Step 1: isolation over condensation iterations (arm-normalized)\n"
        "dotted = PRE (x0), dashed = POST (positions), dist-to-bulk in red",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "condensation_isolation_curves.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(TAB, "condensation_isolation.csv"), index=False)
    print(f"[step1] saved {p}")
    return df


# ----------------------------------------------------------------------------
# STEP 2 : K neighbors at POST, genotype composition + difference test
# ----------------------------------------------------------------------------
def step2(arms):
    arm_list = [("raw", arms["raw"]), ("margin", arms["margin"])]
    focal_list = [(OUTLIER, "outlier"), (CONTROL, "control")]
    rows_out = []
    # figure: rows = K (5,15), cols = arm; stacked-bar genotype composition, grouped by focal
    fig, axes = plt.subplots(len(KS), len(arm_list), figsize=(12, 8), squeeze=False)
    genos = list(GENO_COLORS.keys())

    for ri, k in enumerate(KS):
        for ci, (aname, arm) in enumerate(arm_list):
            ax = axes[ri][ci]
            bar_x = np.arange(len(focal_list))
            bottoms = {gx: np.zeros(len(focal_list)) for gx in [0, 1]}
            comp = np.zeros((len(focal_list), len(genos)))
            for fj, (emb, role) in enumerate(focal_list):
                fi = focal_idx(arm, emb)
                nrows = neighbor_rows(arm["positions"], arm["mask"], fi, arm["labels"], k)
                n_total = len(nrows)
                for nr in nrows:
                    gi = genos.index(nr["genotype"]) if nr["genotype"] in genos else None
                    if gi is not None:
                        comp[fj, gi] += 1
                    rows_out.append({
                        "focal": emb, "role": role, "arm": aname, "k": k,
                        "neighbor_idx": nr["neighbor_idx"], "genotype": nr["genotype"],
                        "bin": nr["bin"], "norm_dist": nr["norm_dist"],
                    })
                if n_total > 0:
                    comp[fj] = comp[fj] / n_total
            base = np.zeros(len(focal_list))
            for gi, g in enumerate(genos):
                ax.bar(bar_x, comp[:, gi], bottom=base, color=geno_color(g),
                       label=g if (ri == 0 and ci == 0) else None, width=0.6)
                base += comp[:, gi]
            ax.set_xticks(bar_x)
            ax.set_xticklabels([f"{e.split('_')[-2]}_{e.split('_')[-1]}\n[{r}]"
                                for e, r in focal_list], fontsize=8)
            ax.set_title(f"{aname} arm  |  K={k}", fontsize=10)
            ax.set_ylabel("neighbor genotype fraction")
            ax.set_ylim(0, 1)

    # Difference test: is each focal embryo's neighborhood genuinely more
    # isolated than a random matched embryo (drawn at its own bins)?
    diff_rows = []
    for k in KS:
        for aname, arm in arm_list:
            for emb, role in focal_list:
                fi = focal_idx(arm, emb)
                ref = neighbor_dist_reference(arm["positions"], arm["mask"], fi, k)
                diff_rows.append({
                    "focal": emb, "role": role, "arm": aname, "k": k,
                    "focal_mean_knn": ref["focal"],
                    "bulk_mean_knn": ref["bulk_mean"],
                    "bulk_median_knn": ref["bulk_median"],
                    "raw_diff": ref["diff"], "ratio_vs_bulk": ref["ratio"],
                })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(TAB, "neighbor_difference_test.csv"), index=False)
    print("[step2] difference test (POST): focal vs matched-bin bulk baseline")
    print(diff_df.to_string(index=False))

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Step 2: genotype composition of K-nearest neighbors (POST-condensation)\n"
                 "outlier vs control, raw vs margin", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(FIG, "neighbor_genotype_by_arm.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(TAB, "neighbor_sets.csv"), index=False)
    print(f"[step2] saved {p}")
    return df


# ----------------------------------------------------------------------------
# STEP 3 (HEADLINE) : mean K-NN distance, PRE vs POST, both arms
# ----------------------------------------------------------------------------
def step3(arms):
    arm_list = [("raw", arms["raw"]), ("margin", arms["margin"])]
    focal_list = [(OUTLIER, "outlier"), (CONTROL, "control")]
    rows_out = []
    for k in KS:
        for emb, role in focal_list:
            for aname, arm in arm_list:
                fi = focal_idx(arm, emb)
                pre_scales = all_bin_scales(arm["x0"], arm["mask"])
                post_scales = all_bin_scales(arm["positions"], arm["mask"])
                pre = focal_isolation(arm["x0"], arm["mask"], fi, pre_scales, k=k)
                post = focal_isolation(arm["positions"], arm["mask"], fi, post_scales, k=k)
                rows_out.append({
                    "embryo": emb, "role": role, "arm": aname, "k": k,
                    "mean_knn_pre": pre["mean_knn"], "mean_knn_post": post["mean_knn"],
                    "delta": post["mean_knn"] - pre["mean_knn"],
                })
    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(TAB, "knn_distance_pre_post.csv"), index=False)

    # Figure: slope plot pre->post. rows = K, cols = arm; two lines (outlier/control)
    fig, axes = plt.subplots(len(KS), len(arm_list), figsize=(12, 8), squeeze=False)
    role_color = {"outlier": "#B2182B", "control": "#2166AC"}
    for ri, k in enumerate(KS):
        for ci, (aname, _arm) in enumerate(arm_list):
            ax = axes[ri][ci]
            sub = df[(df.k == k) & (df.arm == aname)]
            for _, r in sub.iterrows():
                col = role_color[r["role"]]
                ax.plot([0, 1], [r["mean_knn_pre"], r["mean_knn_post"]],
                        "-o", color=col, ms=7, lw=2,
                        label=f"{r['role']} ({r['embryo'].split('_')[-2]})")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["PRE\n(x0)", "POST\n(positions)"])
            ax.set_title(f"{aname} arm  |  K={k}", fontsize=10)
            ax.set_ylabel("mean norm. dist to K-NN")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
    fig.suptitle("Step 3 (headline): mean distance to K-NN, PRE vs POST condensation (arm-normalized)\n"
                 "expect outlier grows (raw), grows MORE (margin); control flat both arms",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(FIG, "knn_distance_pre_post.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[step3] saved {p}")
    return df


# ----------------------------------------------------------------------------
# STEP 4 : K-sweep of the pre->post delta (where does margin flip amplify<->suppress?)
# ----------------------------------------------------------------------------
K_SWEEP = list(range(2, 31))


def step4(arms):
    """
    For K in 2..30, compute the pre->post condensation change in mean normalized
    K-NN distance (delta = post - pre) for E09 (outlier) and G12 (control), in raw
    and margin arms. This exposes the scale dependence: margin AMPLIFIES the
    outlier's separation at small K but SUPPRESSES it (stays flat while raw flings
    it out) at larger K. The crossover K is the finding.
    """
    arm_list = [("raw", arms["raw"]), ("margin", arms["margin"])]
    focal_list = [(OUTLIER, "outlier"), (CONTROL, "control")]
    rows_out = []
    for k in K_SWEEP:
        for emb, role in focal_list:
            for aname, arm in arm_list:
                fi = focal_idx(arm, emb)
                pre_scales = all_bin_scales(arm["x0"], arm["mask"])
                post_scales = all_bin_scales(arm["positions"], arm["mask"])
                pre = focal_isolation(arm["x0"], arm["mask"], fi, pre_scales, k=k)
                post = focal_isolation(arm["positions"], arm["mask"], fi, post_scales, k=k)
                rows_out.append({
                    "embryo": emb, "role": role, "arm": aname, "k": k,
                    "mean_knn_pre": pre["mean_knn"], "mean_knn_post": post["mean_knn"],
                    "delta": post["mean_knn"] - pre["mean_knn"],
                })
    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(TAB, "knn_delta_ksweep.csv"), index=False)

    # find crossover K for the outlier: smallest K where raw delta exceeds margin delta
    piv = (df[df.role == "outlier"]
           .pivot_table(index="k", columns="arm", values="delta"))
    crossover_k = None
    for k in K_SWEEP:
        if k in piv.index and piv.loc[k, "raw"] > piv.loc[k, "margin"]:
            crossover_k = k
            break

    # Figure: two panels (outlier, control); x=K, y=delta; raw vs margin lines.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), squeeze=False)
    arm_color = {"raw": "#1B7837", "margin": "#762A83"}
    for ci, (emb, role) in enumerate(focal_list):
        ax = axes[0][ci]
        sub = df[df.role == role]
        for aname in ("raw", "margin"):
            s = sub[sub.arm == aname].sort_values("k")
            ax.plot(s["k"], s["delta"], "-o", ms=4, color=arm_color[aname], label=f"{aname} arm")
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        if role == "outlier" and crossover_k is not None:
            ax.axvline(crossover_k, color="gray", ls=":", alpha=0.7)
            ax.annotate(f"crossover K={crossover_k}\n(raw isolation overtakes margin)",
                        xy=(crossover_k, 0), xytext=(crossover_k + 1, ax.get_ylim()[1] * 0.5),
                        fontsize=8, color="gray")
        ax.set_title(f"{role}  ({emb.split('_')[-2]})", fontsize=11)
        ax.set_xlabel("K (neighborhood size)")
        ax.set_ylabel("pre->post delta in mean norm. K-NN dist\n(+ = pushed away during condensation)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Step 4: K-sweep of the pre->post condensation delta\n"
        "outlier (left): margin amplifies at small K, raw overtakes / margin flattens at large K; "
        "control (right): both arms near zero",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(FIG, "knn_delta_ksweep.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[step4] saved {p}  (outlier crossover K = {crossover_k})")
    return df, crossover_k


def main():
    print("Loading arms...")
    arms = {"raw": load_arm(RAW_NPZ, "raw"), "margin": load_arm(MARGIN_NPZ, "margin")}
    for aname, arm in arms.items():
        for emb in FOCALS:
            fi = focal_idx(arm, emb)
            print(f"  {aname}: {emb} idx={fi} valid_bins={int(arm['mask'][fi].sum())} "
                  f"label={arm['labels'][fi]}")

    s1 = step1(arms)
    s2 = step2(arms)
    s3 = step3(arms)
    s4, crossover_k = step4(arms)

    print("\n=== STEP 1: isolation pre->post (delta = post - pre) ===")
    print(s1[["embryo", "role", "arm", "d2b_pre", "d2b_post", "d2b_delta",
              "nn_pre", "nn_post", "nn_delta"]].to_string(index=False))

    print("\n=== STEP 3 (HEADLINE): mean K-NN distance pre->post ===")
    print(s3[["role", "arm", "k", "mean_knn_pre", "mean_knn_post", "delta"]].to_string(index=False))

    print("\n=== STEP 2: neighbor genotype composition (POST) ===")
    comp = (s2.groupby(["role", "arm", "k", "genotype"]).size()
            .groupby(level=[0, 1, 2]).apply(lambda x: (x / x.sum()).round(2)))
    print(comp.to_string())

    print(f"\n=== STEP 4: K-sweep crossover (outlier) K = {crossover_k} ===")
    piv = (s4[s4.role == "outlier"].pivot_table(index="k", columns="arm", values="delta")
           .round(3))
    print("outlier pre->post delta by K (raw vs margin):")
    print(piv.to_string())


if __name__ == "__main__":
    main()
