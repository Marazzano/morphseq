#!/usr/bin/env python
"""Decision scatter: focus axis (X) vs saturation axis (Y).

X = interior_strong_edge_frac  (low = blurred/structureless)
Y = frac_ge250 of interior      (high = saturated / bright-clipped)
Background = the 7,500 not-dead population; overlaid = the saturation anchors so
you can see where TRUE saturation vs DORSAL-bright vs CLEAN land relative to the cloud.

Goal: pick joint QC thresholds. A good embryo = high X, low Y (lower-right). Junk =
low X (left). Saturated = high Y (top). Dorsal-bright sits near saturated on Y (the
known hard case).
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
FIGURES.mkdir(parents=True, exist_ok=True)

POP_CSV = TABLES / "interior_structure_cross_experiment_metrics.csv"
OUT = FIGURES / "focus_vs_saturation_scatter.png"

IMG_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files")

XCOL = "interior_strong_edge_frac"
YCOL = "top_spread_p99_p90"
CCOL = "interior_mean"   # color = median/mean interior intensity

# (exp, well, t, label, color)
ANCHORS = [
    ("20250305", "G04", 102, "clean_reference", "#1f77b4"),
    ("20251125", "A10", 194, "clean_reference", "#1f77b4"),
    ("20250305", "E04", 108, "clean_reference", "#1f77b4"),
    ("20260206", "D10", 143, "dead_saturated", "#000000"),
    ("20251125", "A06", 82,  "oversaturated",  "#d62728"),
    ("20251125", "A06", 92,  "oversaturated",  "#d62728"),
    ("20251125", "A06", 93,  "oversaturated",  "#d62728"),
    ("20251125", "E10", 145, "dorsal_bright_good", "#1a8820"),
]


def anchor_xy(exp, well, t):
    base = f"{exp}_{well}_ch00_t{t:04d}"
    ip = IMG_ROOT / f"raw_data_organized/{exp}/images/{exp}_{well}/{base}.jpg"
    mp = IMG_ROOT / f"exported_masks/{exp}/masks/{base}_masks_emnum_1.png"
    img = np.array(Image.open(ip).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(mp)).astype(bool)
    interior = ndi.binary_erosion(mask, iterations=5)
    interior = interior if interior.sum() > 200 else mask
    px = img[interior].astype(float)
    p90, p99 = np.percentile(px, [90, 99])
    top_spread = float(p99 - p90)
    med = float(np.median(px))
    # strong-edge frac (same definition as the population metric)
    from skimage import exposure, filters
    from scipy import ndimage as ndi2
    y, x = np.where(mask)
    pad = 32
    y0, y1 = max(y.min() - pad, 0), min(y.max() + pad + 1, img.shape[0])
    x0, x1 = max(x.min() - pad, 0), min(x.max() + pad + 1, img.shape[1])
    ic, mc = img[y0:y1, x0:x1].astype(np.float32), mask[y0:y1, x0:x1]
    dil = ndi2.binary_dilation(mc, iterations=20)
    vals = ic[(mc | dil) & np.isfinite(ic)]
    lo, hi = np.percentile(vals, [1, 99])
    i01 = exposure.rescale_intensity(ic, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32) if hi > lo else ic * 0
    inter = ndi2.binary_erosion(mc, iterations=12)
    inter = inter if inter.sum() > 200 else mc
    grad = filters.sobel(ndi2.gaussian_filter(i01, 1.0))
    sef = float(np.mean(grad[inter] > 0.04))
    return sef, top_spread, med


def main():
    pop = pd.read_csv(POP_CSV)
    pop = pop[pop["interior_error"].fillna("") == ""].dropna(subset=[XCOL, YCOL, CCOL])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    sc = ax.scatter(pop[XCOL], pop[YCOL], c=pop[CCOL], s=8, cmap="viridis",
                    vmin=80, vmax=240, alpha=0.6, zorder=1)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("median interior intensity (bright = saturated/dorsal)")

    seen = set()
    for exp, well, t, label, color in ANCHORS:
        try:
            x, y, med = anchor_xy(exp, well, t)
        except Exception as e:  # noqa: BLE001
            print(f"skip {exp} {well} {t}: {e}")
            continue
        lbl = label if label not in seen else None
        seen.add(label)
        ax.scatter([x], [y], s=160, facecolor="none", edgecolor=color, linewidth=2.4,
                   label=lbl, zorder=4)
        ax.annotate(f"{label.split('_')[0]} {well} t{t}", (x, y), fontsize=6.5,
                    xytext=(5, 5), textcoords="offset points", zorder=5)

    ax.axvline(0.25, color="#d62728", ls="--", lw=1, alpha=0.6)
    ax.text(0.25, ax.get_ylim()[1], " edge_frac=0.25 (focus cut)", color="#d62728",
            fontsize=7, va="top", rotation=90)
    ax.axhline(10, color="#333", ls=":", lw=1, alpha=0.6)
    ax.text(ax.get_xlim()[1], 10, " p99-p90=10 (bright-tail crush)", color="#333",
            fontsize=7, ha="right", va="bottom")

    ax.set_xlabel("interior_strong_edge_frac  (focus/structure; LOW = blurred)")
    ax.set_ylabel("top_spread p99-p90  (bright-tail spread; LOW = saturated/crushed)")
    ax.set_title("Focus (X) vs bright-tail spread (Y), colored by median intensity\n"
                 "GOOD = high X / high Y / mid color.  SATURATED+DORSAL = low Y / bright color (bottom).")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    plt.close(fig)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
