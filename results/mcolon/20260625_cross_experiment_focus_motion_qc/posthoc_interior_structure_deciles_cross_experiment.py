#!/usr/bin/env python
"""Per-experiment decile galleries for the two interior-structure focus metrics.

Metrics (computed on the eroded embryo INTERIOR, silhouette boundary stripped):
  - interior_std            : interior intensity std. Best all-round; also low for
                              dead (flat interior). Low = blurred/dead.
  - interior_strong_edge_frac: fraction of interior that is a real edge. Cleanest on
                              the ghost-blur axis. Low = blurred.

Reuses the already-sampled 7,500 not-dead embryos (with paths) from the IQR work, so
no new sampling and directly comparable to the IQR decile galleries. D1 = lowest
(most blurred/dead) -> D10 = highest (sharpest), per experiment, 20 imgs/decile.
"""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "interior_structure_deciles"
FIGURES.mkdir(parents=True, exist_ok=True)

SAMPLE_CSV = TABLES / "iqr_cross_experiment_decile_metrics.csv"   # reuse the IQR sample
OUT_METRICS = TABLES / "interior_structure_cross_experiment_metrics.csv"

METRICS = [
    ("interior_std", "interior intensity std"),
    ("interior_strong_edge_frac", "interior strong-edge frac"),
]
N_DECILES = 10
N_PER_DECILE = 20


def bbox(mask, pad=32):
    ys, xs = np.where(mask)
    return (max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, mask.shape[0]),
            max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, mask.shape[1]))


def robust01(img, valid):
    vals = img[valid & np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32)


def interior_metrics(image_path, mask_path):
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(mask_path)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        raise ValueError("bad mask")
    y0, y1, x0, x1 = bbox(mask)
    img_c, mask_c = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    dil = ndi.binary_dilation(mask_c, iterations=20)
    img01 = robust01(img_c, mask_c | dil)
    interior = ndi.binary_erosion(mask_c, iterations=12)
    if interior.sum() < 200:
        interior = ndi.binary_erosion(mask_c, iterations=6)
    if interior.sum() < 200:
        interior = mask_c
    sm = ndi.gaussian_filter(img01, sigma=1.0)
    grad = filters.sobel(sm)
    g_in = grad[interior]
    return {
        "interior_std": float(np.std(img01[interior])),
        "interior_strong_edge_frac": float(np.mean(g_in > 0.04)),
    }


def compute() -> pd.DataFrame:
    if OUT_METRICS.exists():
        print(f"Loading existing -> {OUT_METRICS}", flush=True)
        return pd.read_csv(OUT_METRICS)
    df = pd.read_csv(SAMPLE_CSV)
    df = df.dropna(subset=["image_path", "mask_path"]).copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rec = row.to_dict()
        try:
            rec.update(interior_metrics(row["image_path"], row["mask_path"]))
            rec["interior_error"] = ""
        except Exception as exc:  # noqa: BLE001
            rec["interior_error"] = str(exc)
        rows.append(rec)
        if i % 500 == 0 or i == len(df):
            print(f"[interior] {i}/{len(df)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_METRICS, index=False)
    print(f"Saved -> {OUT_METRICS}", flush=True)
    return out


THUMB = 200  # downscale crops to keep matplotlib memory bounded


def crop_for_gallery(image_path, mask_path, pad=60):
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
    # downscale the largest side to THUMB to bound memory (galleries hold 200 crops)
    h, w = cimg.shape
    s = max(h, w) / THUMB
    if s > 1:
        pim = Image.fromarray(cimg).resize((max(1, int(w / s)), max(1, int(h / s))), Image.BILINEAR)
        pmk = Image.fromarray(cmask).resize((max(1, int(w / s)), max(1, int(h / s))), Image.NEAREST)
        cimg, cmask = np.array(pim), np.array(pmk).astype(bool)
    return cimg, cmask


def build_galleries(df: pd.DataFrame, metric: str, title: str) -> None:
    use = df[df["interior_error"].fillna("") == ""].dropna(subset=[metric]).copy()
    for exp in sorted(use["experiment_id"].unique()):
        sub = use[use["experiment_id"] == exp].copy()
        sub["decile"] = (pd.qcut(sub[metric].rank(method="first"), N_DECILES, labels=False,
                                 duplicates="drop").astype(int) + 1)
        fig, axes = plt.subplots(N_DECILES, N_PER_DECILE,
                                 figsize=(N_PER_DECILE * 1.5, N_DECILES * 1.6), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for r, decile in enumerate(sorted(sub["decile"].unique())):
            grp = sub[sub["decile"] == decile].sort_values(metric)
            if len(grp) > N_PER_DECILE:
                grp = grp.iloc[np.linspace(0, len(grp) - 1, N_PER_DECILE).astype(int)]
            for c, (_, row) in enumerate(grp.iterrows()):
                ax = axes[r, c]
                img, mask = crop_for_gallery(row["image_path"], row["mask_path"])
                if img is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6)
                else:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                    if mask is not None and mask.any():
                        ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.4)
                ax.set_title(f"{row[metric]:.3f}", fontsize=5)
            axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_ylabel(f"D{decile}", fontsize=9)
        fig.suptitle(f"{exp}: {title} deciles (D1 low/blurred -> D10 high/sharp), {N_PER_DECILE}/decile", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out = FIGURES / f"{exp}_{metric}_deciles.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        plt.close("all")
        gc.collect()
        print(f"Saved -> {out}", flush=True)


def main():
    df = compute()
    for metric, title in METRICS:
        build_galleries(df, metric, title)


if __name__ == "__main__":
    main()
