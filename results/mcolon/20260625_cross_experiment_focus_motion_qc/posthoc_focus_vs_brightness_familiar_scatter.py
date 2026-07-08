#!/usr/bin/env python
"""Familiar-style scatters: same X (interior_strong_edge_frac) + 0.25 focus line as the
interior_std scatter, with the known focus anchors AND the saturation anchors over the
7,500 population. Two separate figures, swapping only the Y axis:
  A) Y = top_spread_p99_p90  (bright-tail spread; LOW = saturated/crushed)
  B) Y = interior_mean       (overall interior brightness; HIGH = bright/saturated)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "interior_structure"

POP_CSV = TABLES / "interior_structure_cross_experiment_metrics.csv"
FOCUS_ANCHORS = TABLES / "anchor_interior_structure_metrics.csv"

IMG_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files")
XCOL = "interior_strong_edge_frac"
FOCUS_CUT = 0.25

# colors matching the existing anchor plots
FOCUS_COLORS = {
    "fail_blur_ghost": "#8b0000", "motion_and_partial_blur": "#e377c2",
    "fail_blur_anchor": "#d62728", "fail_blur_candidate": "#d62728", "fail_blur_caught": "#d62728",
    "dorsal_bright_review": "#ff7f0e", "pass_dorsal_bright_review": "#ff7f0e",
    "gold_dorsal_in_focus": "#1f77b4",
    "gold_in_focus": "#1a8820", "pass_in_focus": "#1a8820", "pass_anchor": "#1a8820",
    "pass_after_blur_anchor": "#1a8820", "pass_or_gray_anchor": "#7f7f7f",
    "dead_negative_control": "#000000", "saturation_fail_anchor": "#9467bd",
}

# saturation anchors (computed inline so they're guaranteed present), drawn as rings
SAT_ANCHORS = [
    ("20250305", "G04", 102, "SAT:clean_reference", "#1f77b4"),
    ("20251125", "A10", 194, "SAT:clean_reference", "#1f77b4"),
    ("20250305", "E04", 108, "SAT:clean_reference", "#1f77b4"),
    ("20260206", "D10", 143, "SAT:dead_saturated", "#000000"),
    ("20251125", "A06", 82,  "SAT:oversaturated",  "#d62728"),
    ("20251125", "A06", 92,  "SAT:oversaturated",  "#d62728"),
    ("20251125", "A06", 93,  "SAT:oversaturated",  "#d62728"),
    ("20251125", "E10", 145, "SAT:dorsal_bright_good", "#1a8820"),
]


def sat_xy(exp, well, t):
    from skimage import exposure, filters
    base = f"{exp}_{well}_ch00_t{t:04d}"
    ip = IMG_ROOT / f"raw_data_organized/{exp}/images/{exp}_{well}/{base}.jpg"
    mp = IMG_ROOT / f"exported_masks/{exp}/masks/{base}_masks_emnum_1.png"
    img = np.array(Image.open(ip).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(mp)).astype(bool)
    er = ndi.binary_erosion(mask, iterations=5); er = er if er.sum() > 200 else mask
    px = img[er].astype(float); p90, p99 = np.percentile(px, [90, 99])
    y, x = np.where(mask); pad = 32
    y0, y1 = max(y.min() - pad, 0), min(y.max() + pad + 1, img.shape[0])
    x0, x1 = max(x.min() - pad, 0), min(x.max() + pad + 1, img.shape[1])
    ic, mc = img[y0:y1, x0:x1].astype(np.float32), mask[y0:y1, x0:x1]
    dil = ndi.binary_dilation(mc, iterations=20)
    vals = ic[(mc | dil) & np.isfinite(ic)]; lo, hi = np.percentile(vals, [1, 99])
    i01 = exposure.rescale_intensity(ic, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32) if hi > lo else ic * 0
    inter = ndi.binary_erosion(mc, iterations=12); inter = inter if inter.sum() > 200 else mc
    grad = filters.sobel(ndi.gaussian_filter(i01, 1.0))
    return float(np.mean(grad[inter] > 0.04)), float(p99 - p90), float(np.median(px))


def make(ycol, ylabel, title, guides, out):
    pop = pd.read_csv(POP_CSV)
    pop = pop[pop["interior_error"].fillna("") == ""].dropna(subset=[XCOL, ycol])
    fa = pd.read_csv(FOCUS_ANCHORS)
    fa = fa[fa["struct_error"].fillna("") == ""].dropna(subset=[XCOL, ycol])

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.scatter(pop[XCOL], pop[ycol], s=5, c="#cccccc", alpha=0.4,
               label=f"all not-dead (n={len(pop)})", zorder=1)
    for lab, grp in fa.groupby("user_label"):
        ax.scatter(grp[XCOL], grp[ycol], s=70, c=FOCUS_COLORS.get(lab, "#444"),
                   edgecolor="black", linewidth=0.5, label=lab, zorder=3)
    # saturation anchors as bold rings on top
    seen = set()
    sat_idx = 1 if ycol == "top_spread_p99_p90" else 2
    for exp, well, t, lab, col in SAT_ANCHORS:
        try:
            vals = sat_xy(exp, well, t)
        except Exception:
            continue
        x, yv = vals[0], vals[sat_idx]
        lbl = lab if lab not in seen else None; seen.add(lab)
        ax.scatter([x], [yv], s=180, facecolor="none", edgecolor=col, linewidth=2.6,
                   marker="o", label=lbl, zorder=5)

    ax.axvline(FOCUS_CUT, color="#d62728", ls="--", lw=1.2, alpha=0.7)
    ax.text(FOCUS_CUT, ax.get_ylim()[1], " edge_frac=0.25 (focus cut)", color="#d62728",
            fontsize=7, va="top", rotation=90)
    for gy in guides:
        ax.axhline(gy, color="#333", ls=":", lw=1, alpha=0.5)
        ax.text(ax.get_xlim()[1], gy, f" {ycol}={gy}", color="#333", fontsize=7,
                ha="right", va="bottom")

    ax.set_xlabel("interior_strong_edge_frac  (focus/structure; LOW = blurred)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    make("top_spread_p99_p90",
         "top_spread p99-p90  (bright-tail spread; LOW = saturated)",
         "Focus (X) vs bright-tail spread (Y) — focus + saturation anchors over population",
         [10],
         FIGURES / "focus_vs_top_spread_familiar.png")
    make("interior_mean",
         "interior_mean intensity  (HIGH = bright/saturated)",
         "Focus (X) vs interior brightness (Y) — focus + saturation anchors over population",
         [210],
         FIGURES / "focus_vs_interior_mean_familiar.png")


if __name__ == "__main__":
    main()
