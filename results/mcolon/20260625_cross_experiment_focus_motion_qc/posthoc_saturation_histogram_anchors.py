#!/usr/bin/env python
"""Saturation QC: masked intensity histograms to separate TRUE saturation from
bright-but-fine DORSAL views.

Physical distinction (both are "bright", only saturation is bad):
  - TRUE saturation -> pixels CLIP at the ceiling. Histogram has a SPIKE at 255,
    compressed/no spread near the top, information above the clip point destroyed.
  - DORSAL-bright    -> genuinely bright but NOT clipped. Histogram shifted high but
    still SPREAD with a smooth roll-off; little mass exactly at 255.

So the discriminator is CLIPPING, not brightness: frac_eq255, top-bin peakedness,
and the spread of the bright tail (p99-p90), measured on embryo-interior pixels.

One histogram row per anchor; features in the title.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "saturation_histograms"
FIGURES.mkdir(parents=True, exist_ok=True)

IMG_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files")
OUT_CSV = TABLES / "saturation_histogram_anchor_metrics.csv"
OUT_FIG = FIGURES / "saturation_anchor_histograms.png"

# (experiment, well, time, label, color)
# clean_reference = top interior_strong_edge_frac (~0.70+), moderate brightness, low sat:
# the "this is what a GOOD embryo histogram looks like" positive control.
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


def paths(exp, well, t):
    base = f"{exp}_{well}_ch00_t{t:04d}"
    return (IMG_ROOT / f"raw_data_organized/{exp}/images/{exp}_{well}/{base}.jpg",
            IMG_ROOT / f"exported_masks/{exp}/masks/{base}_masks_emnum_1.png")


def features(exp, well, t):
    ip, mp = paths(exp, well, t)
    img = np.array(Image.open(ip).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(mp)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        raise ValueError("bad mask")
    interior = ndi.binary_erosion(mask, iterations=5)
    if interior.sum() < 200:
        interior = mask
    px = img[interior].astype(np.float64)
    hist = np.bincount(px.astype(int), minlength=256).astype(float)
    hist_norm = hist / hist.sum()
    p90, p99 = np.percentile(px, [90, 99])
    # peakedness of the 255 wall vs the 240-254 neighborhood
    neigh = hist_norm[240:255].mean() + 1e-9
    feats = {
        "experiment_id": exp, "well_id": well, "time_index": t,
        "n_px": int(interior.sum()),
        "mean": float(px.mean()), "median": float(np.median(px)),
        "frac_ge245": float(np.mean(px >= 245)),
        "frac_ge250": float(np.mean(px >= 250)),
        "frac_eq255": float(np.mean(px >= 255)),
        "p90": float(p90), "p99": float(p99),
        "top_spread_p99_minus_p90": float(p99 - p90),
        "frac_200_250": float(np.mean((px >= 200) & (px < 250))),
        "top_bin_peakedness": float(hist_norm[255] / neigh),
    }
    return feats, px


def main():
    rows, pxs = [], []
    for exp, well, t, label, color in ANCHORS:
        f, px = features(exp, well, t)
        f["user_label"] = label
        f["color"] = color
        rows.append(f)
        pxs.append(px)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    show = ["user_label", "well_id", "time_index", "mean", "frac_eq255",
            "frac_ge250", "top_spread_p99_minus_p90", "frac_200_250", "top_bin_peakedness"]
    print(df[show].to_string(index=False))

    n = len(ANCHORS)
    fig, axes = plt.subplots(n, 1, figsize=(11, n * 2.3), squeeze=False)
    for i, ((exp, well, t, label, color), px) in enumerate(zip(ANCHORS, pxs)):
        ax = axes[i, 0]
        ax.hist(px, bins=256, range=(0, 255), color=color, alpha=0.8)
        for v, c in [(245, "#888"), (250, "#ff7f0e"), (255, "#d62728")]:
            ax.axvline(v, color=c, ls="--", lw=1)
        r = df.iloc[i]
        ax.set_title(
            f"{exp} {well} t{t}  [{label}]   "
            f"mean={r['mean']:.0f}  frac=255:{r['frac_eq255']:.3f}  "
            f">=250:{r['frac_ge250']:.3f}  p99-p90={r['top_spread_p99_minus_p90']:.0f}  "
            f"255-peakedness={r['top_bin_peakedness']:.1f}",
            fontsize=9)
        ax.set_xlim(0, 256)
        ax.set_ylabel("interior px count", fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("intensity (0-255); dashed = 245 / 250 / 255")
    fig.suptitle("Saturation QC: embryo-interior intensity histograms.  "
                 "TRUE saturation = spike at 255 + low p99-p90.  "
                 "DORSAL-bright = high but spread, smooth roll-off, little at 255.", y=0.997, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_FIG, dpi=170)
    plt.close(fig)
    print(f"\nSaved -> {OUT_FIG}")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
