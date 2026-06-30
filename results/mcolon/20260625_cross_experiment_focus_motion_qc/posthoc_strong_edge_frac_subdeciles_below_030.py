#!/usr/bin/env python
"""High-res subdecile galleries of interior_strong_edge_frac in the <=0.30 region.

Purpose: pick the operational cutoff. Zooms into the threshold zone (<=0.30) by
splitting it into 10 equal-count subdeciles POOLED ACROSS EXPERIMENTS (one global
absolute scale, since the goal is one absolute cut). One separate high-res PNG per
subdecile, 10 crops each, labeled value + experiment + well/time.

Reuses tables/interior_structure_cross_experiment_metrics.csv (no recompute).
"""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "strong_edge_frac_subdeciles_below030"
FIGURES.mkdir(parents=True, exist_ok=True)

METRICS_CSV = TABLES / "interior_structure_cross_experiment_metrics.csv"
OUT_MANIFEST = TABLES / "strong_edge_frac_subdecile_below030_samples.csv"

METRIC = "interior_strong_edge_frac"
REGION_MAX = 0.30
N_SUB = 10
N_PER = 10
THUMB = 320  # higher-res thumbnails than the decile overview


def crop(image_path, mask_path, pad=70):
    ip, mp = Path(str(image_path)), Path(str(mask_path))
    if not ip.exists() or not mp.exists():
        return None, None
    img = np.array(Image.open(ip).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(mp)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        return img, None
    ys, xs = np.where(mask)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, img.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, img.shape[1])
    cimg, cmask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    h, w = cimg.shape
    s = max(h, w) / THUMB
    if s > 1:
        cimg = np.array(Image.fromarray(cimg).resize((max(1, int(w / s)), max(1, int(h / s))), Image.BILINEAR))
        cmask = np.array(Image.fromarray(cmask).resize(cimg.shape[::-1], Image.NEAREST)).astype(bool)
    return cimg, cmask


def main():
    df = pd.read_csv(METRICS_CSV)
    df = df[df["interior_error"].fillna("") == ""].dropna(subset=[METRIC]).copy()
    df["experiment_id"] = df["experiment_id"].astype(str)

    region = df[df[METRIC] <= REGION_MAX].copy().sort_values(METRIC)
    print(f"{len(region)} embryos with {METRIC} <= {REGION_MAX} "
          f"({100*len(region)/len(df):.1f}% of {len(df)} sampled)", flush=True)
    region["subdecile"] = (pd.qcut(region[METRIC].rank(method="first"), N_SUB,
                                   labels=False, duplicates="drop").astype(int) + 1)

    manifests = []
    for sd in sorted(region["subdecile"].unique()):
        grp = region[region["subdecile"] == sd].sort_values(METRIC)
        lo, hi = grp[METRIC].min(), grp[METRIC].max()
        if len(grp) > N_PER:
            picks = grp.iloc[np.linspace(0, len(grp) - 1, N_PER).astype(int)]
        else:
            picks = grp
        picks = picks.copy()
        picks["pick_rank"] = np.arange(1, len(picks) + 1)
        manifests.append(picks)

        fig, axes = plt.subplots(2, 5, figsize=(20, 9), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, (_, row) in enumerate(picks.iterrows()):
            ax = axes[i // 5, i % 5]
            cimg, cmask = crop(row["image_path"], row["mask_path"])
            if cimg is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            else:
                ax.imshow(cimg, cmap="gray", vmin=0, vmax=255)
                if cmask is not None and cmask.any():
                    ax.contour(cmask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.6)
            ax.set_title(f"{row[METRIC]:.3f}  {row['experiment_id']} {row['well_id']} t{int(row['time_index'])}",
                         fontsize=9)
        fig.suptitle(f"strong_edge_frac SUBDECILE {sd}/{N_SUB}  (range {lo:.3f}-{hi:.3f})  "
                     f"of the <= {REGION_MAX} region, pooled across experiments", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = FIGURES / f"subdecile_{sd:02d}_range_{lo:.3f}_{hi:.3f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig); plt.close("all"); gc.collect()
        print(f"Saved -> {out}", flush=True)

    pd.concat(manifests, ignore_index=True).to_csv(OUT_MANIFEST, index=False)
    print(f"Saved manifest -> {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
