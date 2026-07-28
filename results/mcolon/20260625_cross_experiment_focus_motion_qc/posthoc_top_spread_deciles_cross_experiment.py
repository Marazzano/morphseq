#!/usr/bin/env python
"""Pooled (cross-experiment) decile galleries for top_spread_p99_p90 (saturation axis).

D1 = lowest spread (bright-tail crushed = most saturated) -> D10 = highest (best
exposed). Pooled across all experiments on one absolute scale (the user wants to
compare across, not within). 20/decile. Each crop annotated with its interior_mean
(brightness) so saturated-vs-just-bright can be judged within each spread band.

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
FIGURES = HERE / "figures" / "top_spread_deciles"
FIGURES.mkdir(parents=True, exist_ok=True)

METRICS_CSV = TABLES / "interior_structure_cross_experiment_metrics.csv"
OUT_MANIFEST = TABLES / "top_spread_decile_samples_pooled.csv"

METRIC = "top_spread_p99_p90"
N_DECILES = 10
N_PER = 20
THUMB = 240


def crop(image_path, mask_path, pad=60):
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
    df = df[df["interior_error"].fillna("") == ""].dropna(subset=[METRIC, "interior_mean"]).copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["decile"] = (pd.qcut(df[METRIC].rank(method="first"), N_DECILES, labels=False,
                            duplicates="drop").astype(int) + 1)

    manifests = []
    for decile in sorted(df["decile"].unique()):
        grp = df[df["decile"] == decile].sort_values(METRIC)
        lo, hi = grp[METRIC].min(), grp[METRIC].max()
        if len(grp) > N_PER:
            picks = grp.iloc[np.linspace(0, len(grp) - 1, N_PER).astype(int)]
        else:
            picks = grp
        picks = picks.copy(); picks["pick_rank"] = np.arange(1, len(picks) + 1)
        manifests.append(picks)

        fig, axes = plt.subplots(4, 5, figsize=(20, 16), squeeze=False)
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
                    ax.contour(cmask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.5)
            # annotate spread + brightness so saturated vs just-bright is visible
            ax.set_title(f"spread={row[METRIC]:.0f}  mean={row['interior_mean']:.0f}\n"
                         f"{row['experiment_id']} {row['well_id']} t{int(row['time_index'])}",
                         fontsize=8)
        fig.suptitle(f"top_spread_p99_p90 DECILE {decile}/{N_DECILES}  (range {lo:.0f}-{hi:.0f})  "
                     f"pooled across experiments  |  D1 = saturated/crushed -> D10 = well-exposed\n"
                     f"'mean' = interior brightness (high = bright/saturated)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = FIGURES / f"top_spread_decile_{decile:02d}_range_{lo:.0f}_{hi:.0f}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig); plt.close("all"); gc.collect()
        print(f"Saved -> {out}", flush=True)

    pd.concat(manifests, ignore_index=True).to_csv(OUT_MANIFEST, index=False)
    print(f"Saved manifest -> {OUT_MANIFEST}", flush=True)
    # quick reference: decile boundaries
    q = df.groupby("decile")[METRIC].agg(["min", "median", "max", "count"])
    print(q.to_string())


if __name__ == "__main__":
    main()
