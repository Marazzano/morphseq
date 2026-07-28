"""
18_dim_expression_slices.py

"Expression plots" on the condensed 2D space: color each condensed point by its
RAW z_mu_b value for a given latent dimension (like coloring a UMAP by gene
expression, but the 'gene' is a z_mu_b dim). Goal: see whether the condensed
space is organized BY these dimensions -- i.e. is E09's separation from G12
explained by an extreme value on dim 33 / 71 / 85?

For each dim in {z_mu_b_33, z_mu_b_71, z_mu_b_85}:
  4 condensed-2D slices = {raw, margin} x {PRE (x0), POST (positions)},
  points colored by that dim's raw value ("expression"), E09/G12 outlined.
Per-dim shared color scale across its 4 slices. 3 dims x 4 = 12 panels.

Slice bin = E09's bin nearest 72hpf in each arm. z_mu_b value looked up per
(embryo, bin); margin npz bins (+2 offset) snapped to nearest raw bin.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python 18_dim_expression_slices.py
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
DIMS = ["z_mu_b_33_binned", "z_mu_b_71_binned", "z_mu_b_85_binned"]

# Genotype colors: matches 1_cluster_raw_latent.py / 5_harmony_quad_html.py
# (condensed_harmony_theta4 panels) -- pbx1b=soft purple, pbx4=yellow, double=crimson.
GENO_COLORS = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd",       # soft purple
    "pbx4_crispant": "#F7B267",        # yellow
    "pbx1b_pbx4_crispant": "#B2182B",  # crimson
}
def geno_color(g):
    return GENO_COLORS.get(str(g), "#808080")


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {
        "positions": d["positions"], "x0": d["x0"], "mask": d["mask"],
        "time_values": d["time_values"],
        "embryo_ids": np.array([str(e) for e in d["embryo_ids"]]),
        "labels": np.array([str(l) for l in d["labels"]], dtype=object),
    }


def build_dim_lookup(dims):
    """(embryo_id, raw_bin) -> {dim: value}, plus sorted raw bins."""
    tbl = pd.read_csv(ZTBL, low_memory=False)
    tbl["time_bin"] = tbl["time_bin"].astype(float)
    look = {}
    sub = tbl[["embryo_id", "time_bin"] + dims].to_numpy(dtype=object)
    for row in sub:
        look[(row[0], float(row[1]))] = {d: float(row[2 + i]) for i, d in enumerate(dims)}
    raw_bins = np.array(sorted(tbl["time_bin"].unique()))
    return look, raw_bins


def dim_value(look, raw_bins, embryo_id, npz_bin, dim):
    v = look.get((embryo_id, float(npz_bin)))
    if v is None:
        snapped = float(raw_bins[np.argmin(np.abs(raw_bins - npz_bin))])
        v = look.get((embryo_id, snapped))
    return v[dim] if v is not None else np.nan


def pick_bin(arm, target=72.0):
    tv = arm["time_values"]
    fi = list(arm["embryo_ids"]).index(E09)
    valid = [b for b in range(len(tv)) if arm["mask"][fi, b]]
    return min(valid, key=lambda b: abs(tv[b] - target))


def main():
    raw = load_npz(RAW_NPZ)
    margin = load_npz(MARGIN_NPZ)
    look, raw_bins = build_dim_lookup(DIMS)

    states = [("raw PRE", raw, "x0"), ("raw POST", raw, "positions"),
              ("margin PRE", margin, "x0"), ("margin POST", margin, "positions")]

    # precompute per-state: coords, valid idx, dim values, focal positions
    slices = {}
    for name, arm, ckey in states:
        b = pick_bin(arm)
        coords = arm[ckey]
        ids = list(arm["embryo_ids"])
        valid = np.where(arm["mask"][:, b])[0]
        P = coords[valid, b, :]
        vals = {d: np.array([dim_value(look, raw_bins, ids[i], arm["time_values"][b], d)
                             for i in valid]) for d in DIMS}
        genos = arm["labels"][valid]
        foc = {}
        for f in (E09, G12):
            fi = ids.index(f)
            foc[f] = coords[fi, b, :] if arm["mask"][fi, b] else None
        slices[name] = dict(P=P, vals=vals, genos=genos, foc=foc, hpf=arm["time_values"][b])

    n_rows = 1 + len(DIMS)  # genotype row + one row per dim
    fig, axes = plt.subplots(n_rows, len(states), figsize=(21, 5 * n_rows), squeeze=False)

    # ---- row 0: genotype (standard colors) ----
    seen = {}
    for ci, (name, arm, ckey) in enumerate(states):
        ax = axes[0][ci]
        S = slices[name]
        cols = [geno_color(g) for g in S["genos"]]
        ax.scatter(S["P"][:, 0], S["P"][:, 1], c=cols, s=26, edgecolor="none", alpha=0.9)
        for g in S["genos"]:
            seen[g] = geno_color(g)
        for f, mk, lab in [(E09, "*", "E09"), (G12, "D", "G12")]:
            fp = S["foc"][f]
            if fp is not None:
                ax.scatter(fp[0], fp[1], marker=mk, s=320 if mk == "*" else 200,
                           facecolor="none", edgecolor="black", linewidth=2.2, zorder=5,
                           label=lab)
        ax.set_title(f"GENOTYPE  |  {name} @ {S['hpf']:.0f}hpf", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
    # genotype legend on the far-right genotype panel
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", ms=8, label=g)
               for g, c in GENO_COLORS.items() if g in seen]
    axes[0][-1].legend(handles=handles, fontsize=7, loc="center left",
                       bbox_to_anchor=(1.02, 0.5), title="genotype")

    for ri0, dim in enumerate(DIMS):
        ri = ri0 + 1
        # per-dim shared color scale across its 4 slices
        allv = np.concatenate([slices[n]["vals"][dim][np.isfinite(slices[n]["vals"][dim])]
                               for n, _, _ in states])
        vmin, vmax = np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)
        for ci, (name, arm, ckey) in enumerate(states):
            ax = axes[ri][ci]
            S = slices[name]
            sc = ax.scatter(S["P"][:, 0], S["P"][:, 1], c=S["vals"][dim], cmap="coolwarm",
                            vmin=vmin, vmax=vmax, s=26, edgecolor="none", alpha=0.9)
            for f, mk, lab in [(E09, "*", "E09"), (G12, "D", "G12")]:
                fp = S["foc"][f]
                if fp is not None:
                    fv = dim_value(look, raw_bins, f, S["hpf"], dim)
                    ax.scatter(fp[0], fp[1], c=[fv], cmap="coolwarm", vmin=vmin, vmax=vmax,
                               marker=mk, s=320 if mk == "*" else 200,
                               edgecolor="black", linewidth=2.0, zorder=5, label=f"{lab}={fv:.2f}")
            ax.set_title(f"{dim.replace('_binned','')}  |  {name} @ {S['hpf']:.0f}hpf", fontsize=10)
            ax.legend(fontsize=8, loc="best")
            if ci == len(states) - 1:
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(
                    dim.replace('_binned', '') + " value", fontsize=8)
    fig.suptitle("Expression plots: condensed 2D colored by raw z_mu_b value per dimension "
                 "(is the space organized by dims 33 / 71 / 85? does E09 sit at an extreme?)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(FIG, "dim_expression_slices.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")

    # readout: E09 & G12 value + population percentile per dim (raw POST)
    print("\nFocal dim values & percentile within raw-POST slice:")
    S = slices["raw POST"]
    for dim in DIMS:
        col = S["vals"][dim]; col = col[np.isfinite(col)]
        for f in (E09, G12):
            fv = dim_value(look, raw_bins, f, S["hpf"], dim)
            pct = (col < fv).mean() * 100 if col.size else np.nan
            print(f"  {dim.replace('_binned','')}  {f.split('_')[-2]}: value={fv:.3f}  pctile={pct:.0f}%")


if __name__ == "__main__":
    main()
