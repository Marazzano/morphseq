#!/usr/bin/env python
"""Stage 1 (v2): proper perpendicular edge-width blur metric on labeled anchors.

Fix vs v1: v1 used grad/|laplacian|, which explodes where the laplacian -> 0
(flat/saturated/noisy regions), so the DEAD and SATURATED anchors scored as
"most blurred" -- a metric artifact, not focus. This version measures real edge
WIDTH the textbook (Marziliano) way: at each strong edge pixel, walk along the
gradient direction in both directions until the intensity stops monotonically
changing, and COUNT the pixels of that transition. Sharp edge = narrow (1-2 px);
blurred edge = wide. Contrast cancels (it's a pixel count, not a magnitude).

Judgement: dead/saturated anchors are shown but EXCLUDED from the separation
verdict (no in-focus structure to measure). The real test is:
  do the blur fails have LARGER edge width than gold_in_focus + gold_dorsal?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "edge_width"
FIGURES.mkdir(parents=True, exist_ok=True)

ANCHORS = TABLES / "focus_anchor_registry_metrics.csv"
OUT_CSV = TABLES / "focus_anchor_edge_width_metrics.csv"
OUT_FIG = FIGURES / "anchor_edge_width_sorted.png"

# class -> (sort group, color, used_in_focus_verdict)
CLASS = {
    "fail_blur_anchor":          (1, "#d62728", True),
    "fail_blur_candidate":       (1, "#d62728", True),
    "fail_blur_caught":          (1, "#d62728", True),
    "dorsal_bright_review":      (2, "#ff7f0e", True),
    "pass_dorsal_bright_review": (2, "#ff7f0e", True),
    "gold_dorsal_in_focus":      (3, "#1f77b4", True),
    "gold_in_focus":             (4, "#1a8820", True),
    "pass_in_focus":             (4, "#1a8820", True),
    "pass_anchor":               (4, "#1a8820", True),
    "pass_after_blur_anchor":    (4, "#1a8820", True),
    "pass_or_gray_anchor":       (5, "#7f7f7f", False),
    "dead_negative_control":     (6, "#000000", False),
    "saturation_fail_anchor":    (6, "#9467bd", False),
}

EDGE_PCTL = 92      # measure width only at strongest 8% of edge pixels
MAX_WALK = 20       # cap transition length (px)
PATCH = 48


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


def perpendicular_widths(img01, emb):
    """Real edge widths: walk the intensity profile across each strong edge pixel.

    At each edge pixel, follow +/- gradient direction while intensity keeps moving
    the same way (monotonic ramp of the edge). Width = pixels from the local min to
    the local max of that ramp. Pure geometry -> contrast-invariant.
    """
    sm = ndi.gaussian_filter(img01, sigma=1.0)
    gx = ndi.sobel(sm, axis=1)
    gy = ndi.sobel(sm, axis=0)
    grad = np.hypot(gx, gy)
    g_in = grad[emb]
    if g_in.size == 0:
        return np.array([]), None
    thr = np.percentile(g_in, EDGE_PCTL)
    ys, xs = np.where(emb & (grad >= thr) & (grad > 1e-4))
    if ys.size == 0:
        return np.array([]), None

    h, w = img01.shape
    # subsample edge pixels for speed (cap ~3000)
    if ys.size > 3000:
        idx = np.random.default_rng(0).choice(ys.size, 3000, replace=False)
        ys, xs = ys[idx], xs[idx]

    widths = np.full(img01.shape, np.nan, dtype=np.float32)
    out = []
    for y, x in zip(ys, xs):
        dx, dy = gx[y, x], gy[y, x]
        n = np.hypot(dx, dy)
        if n < 1e-6:
            continue
        ux, uy = dx / n, dy / n
        # walk in +grad direction until intensity stops increasing
        up = 0
        py, px = float(y), float(x)
        prev = sm[y, x]
        for _ in range(MAX_WALK):
            py += uy; px += ux
            iy, ix = int(round(py)), int(round(px))
            if not (0 <= iy < h and 0 <= ix < w):
                break
            val = sm[iy, ix]
            if val <= prev:
                break
            prev = val; up += 1
        # walk in -grad direction until intensity stops decreasing
        down = 0
        py, px = float(y), float(x)
        prev = sm[y, x]
        for _ in range(MAX_WALK):
            py -= uy; px -= ux
            iy, ix = int(round(py)), int(round(px))
            if not (0 <= iy < h and 0 <= ix < w):
                break
            val = sm[iy, ix]
            if val >= prev:
                break
            prev = val; down += 1
        wd = up + down + 1
        out.append(wd)
        widths[y, x] = wd
    return np.array(out, dtype=float), widths


def metric_row(row):
    img = np.array(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(row["mask_path"])).astype(bool)
    if mask.shape != img.shape or not mask.any():
        return {"edge_error": "bad mask"}
    y0, y1, x0, x1 = bbox(mask)
    img_c, mask_c = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    dil = ndi.binary_dilation(mask_c, iterations=20)
    img01 = robust01(img_c, mask_c | dil)
    emb = ndi.binary_erosion(mask_c, iterations=3)
    if emb.sum() < 100:
        emb = mask_c
    widths, wmap = perpendicular_widths(img01, emb)
    if widths.size < 10:
        return {"edge_error": "too few edges"}

    # per-patch worst (catches partial blur)
    patch_meds = []
    if wmap is not None:
        h, w = wmap.shape
        for yy in range(0, h, PATCH):
            for xx in range(0, w, PATCH):
                blk = wmap[yy:yy + PATCH, xx:xx + PATCH]
                vals = blk[np.isfinite(blk)]
                if vals.size >= 8:
                    patch_meds.append(float(np.median(vals)))
    patch_meds = np.array(patch_meds)
    return {
        "edge_error": "",
        "ew_median": float(np.median(widths)),
        "ew_p75": float(np.percentile(widths, 75)),
        "ew_frac_wide": float(np.mean(widths >= 5)),   # fraction of edges blurred >=5px
        "ew_worst_patch": float(np.max(patch_meds)) if patch_meds.size else float("nan"),
        "n_edges": int(widths.size),
    }


def main():
    df = pd.read_csv(ANCHORS)
    df = df[df["metric_error"].fillna("") == ""].copy()
    recs = []
    for _, row in df.iterrows():
        rec = {k: row[k] for k in ["experiment_id", "well_id", "time_index", "user_label",
                                   "emb_iqr_local", "emb_frac_gt245"]}
        rec.update(metric_row(row))
        recs.append(rec)
    out = pd.DataFrame(recs)
    out["sort_key"] = out["user_label"].map(lambda c: CLASS.get(c, (9, "#444", False))[0])
    out["color"] = out["user_label"].map(lambda c: CLASS.get(c, (9, "#444", False))[1])
    out["in_verdict"] = out["user_label"].map(lambda c: CLASS.get(c, (9, "#444", False))[2])
    out = out.sort_values(["sort_key", "experiment_id", "well_id", "time_index"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    ok = out[out["edge_error"].fillna("") == ""]
    print(ok[["user_label", "ew_median", "ew_p75", "ew_frac_wide", "ew_worst_patch", "emb_iqr_local"]].to_string(index=False))

    # verdict summary: blur fails vs gold in-focus(+dorsal)
    v = ok[ok["in_verdict"]]
    blur = v[v["sort_key"] == 1]
    focus = v[v["sort_key"].isin([3, 4])]
    print("\n=== SEPARATION (edges, higher width = more blurred) ===")
    for m in ["ew_median", "ew_p75", "ew_frac_wide", "ew_worst_patch"]:
        print(f"{m}: blur-fail median={blur[m].median():.2f} | gold-focus median={focus[m].median():.2f} "
              f"| blur range=[{blur[m].min():.2f},{blur[m].max():.2f}] focus range=[{focus[m].min():.2f},{focus[m].max():.2f}]")

    plot = ok
    labels = [f"{r.experiment_id} {r.well_id} t{int(r.time_index)}\n{r.user_label}" for _, r in plot.iterrows()]
    y = np.arange(len(plot))
    metrics = [("emb_iqr_local", "IQR (contrast)"),
               ("ew_median", "edge width median"),
               ("ew_p75", "edge width p75"),
               ("ew_frac_wide", "frac edges >=5px wide"),
               ("ew_worst_patch", "WORST patch width")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), max(5, len(plot) * 0.42)), sharey=True)
    for ax, (m, title) in zip(axes, metrics, strict=True):
        ax.scatter(plot[m], y, c=plot["color"], s=46)
        ax.set_xlabel(m); ax.set_title(title); ax.grid(alpha=0.25)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=7); axes[0].invert_yaxis()
    fig.suptitle("Stage 1 v2: real edge-width blur. RED=blur-fail (want HIGH), "
                 "BLUE=gold dorsal in-focus, GREEN=gold/pass in-focus (want LOW). "
                 "purple/black=dead/sat (ignore for focus verdict)", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"\nSaved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
