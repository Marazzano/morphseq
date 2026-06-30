#!/usr/bin/env python
"""Interior-structure blur metrics: targets the GHOST failure mode.

Key realization (2026-06-30): the confirmed-bad 20250912 A04 ghosts are big,
bright, well-masked embryos that are EMPTY inside -- the embryo stands out from
background (high contrast) but has almost no internal texture because it's badly
out of focus. So the discriminator is NOT edge WIDTH and NOT embryo-vs-background
contrast; it's INTERNAL STRUCTURE PER UNIT AREA, measured on the embryo interior
with the silhouette boundary removed (hard erosion) so the body outline -- which
is wide/strong for everyone -- doesn't dominate.

The 20250912 ghosts are not in focus_metrics_by_experiment.csv, so they are loaded
by DIRECT PATH here and merged with the registry anchors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "interior_structure"
FIGURES.mkdir(parents=True, exist_ok=True)

ANCHORS = TABLES / "focus_anchor_registry_metrics.csv"
OUT_CSV = TABLES / "anchor_interior_structure_metrics.csv"
OUT_FIG = FIGURES / "anchor_interior_structure_sorted.png"

IMG_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files")

# Off-pipeline ghosts: confirmed badly out of focus, loaded by direct path.
GHOSTS = [
    ("20250912", "A04", 95), ("20250912", "A04", 97), ("20250912", "A04", 100),
    ("20250912", "A04", 104), ("20250912", "A04", 110),
]

# class -> (sort group, color, in_focus_verdict)
CLASS = {
    "fail_blur_ghost":           (0, "#8b0000", True),   # the 20250912 ghosts (worst)
    "motion_and_partial_blur":   (1, "#e377c2", True),   # A10 t216: head OOF, tail sharp (+motion)
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


def ghost_paths(exp, well, t):
    base = f"{exp}_{well}_ch00_t{t:04d}"
    img = IMG_ROOT / f"raw_data_organized/{exp}/images/{exp}_{well}/{base}.jpg"
    mask = IMG_ROOT / f"exported_masks/{exp}/masks/{base}_masks_emnum_1.png"
    return str(img), str(mask)


def bbox(mask, pad=32):
    ys, xs = np.where(mask)
    return (max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, mask.shape[0]),
            max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, mask.shape[1]))


N_BANDS = 5  # head->tail bands along the embryo long axis


def axis_bands(interior, n_bands=N_BANDS):
    """Split the interior into n_bands along the embryo long axis via PCA.

    Projects interior pixel coords onto the principal (longest) axis and bins into
    equal-arc bands head->tail. This is the MVP straight-axis version; when curvature
    / pca_spine arc-length is available, swap the projection `t` below for spine
    arc-length and the rest of the metric is unchanged.
    Returns an int label image (0..n_bands-1) over interior pixels, -1 elsewhere.
    """
    ys, xs = np.where(interior)
    if ys.size < n_bands * 20:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(0)
    # principal axis = top eigenvector of the coordinate covariance
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis = vt[0]
    t = pts @ axis  # projection along long axis
    # equal-count bins so each band has comparable pixel support (robust to shape)
    edges = np.quantile(t, np.linspace(0, 1, n_bands + 1))
    edges[0] -= 1e-6
    band_of_px = np.clip(np.digitize(t, edges[1:-1]), 0, n_bands - 1)
    labels = np.full(interior.shape, -1, dtype=np.int8)
    labels[ys, xs] = band_of_px
    return labels


def robust01(img, valid):
    vals = img[valid & np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32)


def metric_row(image_path, mask_path):
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(mask_path)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        return {"struct_error": "bad mask"}
    y0, y1, x0, x1 = bbox(mask)
    img_c, mask_c = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    dil = ndi.binary_dilation(mask_c, iterations=20)
    img01 = robust01(img_c, mask_c | dil)

    # INTERIOR only: erode hard to strip the silhouette boundary (the body outline
    # is wide/strong for everyone and otherwise dominates).
    interior = ndi.binary_erosion(mask_c, iterations=12)
    if interior.sum() < 200:
        interior = ndi.binary_erosion(mask_c, iterations=6)
    if interior.sum() < 200:
        interior = mask_c

    sm = ndi.gaussian_filter(img01, sigma=1.0)
    grad = filters.sobel(sm)
    lap = np.abs(ndi.gaussian_laplace(img01, sigma=1.5))

    g_in = grad[interior]
    l_in = lap[interior]
    px_in = img01[interior]

    # structure-per-area: how much real internal edge content exists in the body
    strong = float(np.mean(g_in > 0.04))            # fraction of interior that is a real edge
    edge_density = float(np.mean(g_in))             # mean interior gradient (texture amount)
    lap_density = float(np.mean(l_in))              # mean interior 2nd-deriv (fine detail)
    interior_std = float(np.std(px_in))             # interior intensity variation
    # p90 of interior gradient: a ghost has NO strong internal edges -> low p90
    grad_p90 = float(np.percentile(g_in, 90))

    # --- per-band (head->tail) structure for PARTIAL defocus ---
    # Partial blur = one anatomical region much less structured than the rest, even
    # though the whole-embryo level looks fine. Uniform blur (ghost) -> all bands LOW
    # but similar -> low variation; in-focus -> all bands high & similar -> low
    # variation; partial blur -> bands DIVERGE -> high variation + low min/median.
    band_strong_min_over_med = float("nan")
    band_strong_cv = float("nan")
    band_vals_str = ""
    labels = axis_bands(interior)
    if labels is not None:
        strong_map = (grad > 0.04)
        band_strong = []
        for b in range(N_BANDS):
            m = labels == b
            if m.sum() >= 30:
                band_strong.append(float(strong_map[m].mean()))
        if len(band_strong) >= 3:
            bs = np.array(band_strong)
            med = np.median(bs)
            band_strong_min_over_med = float(bs.min() / (med + 1e-9))
            band_strong_cv = float(bs.std() / (bs.mean() + 1e-9))
            band_vals_str = ";".join(f"{v:.3f}" for v in bs)

    return {
        "struct_error": "",
        "interior_edge_density": edge_density,
        "interior_strong_edge_frac": strong,
        "interior_lap_density": lap_density,
        "interior_grad_p90": grad_p90,
        "interior_std": interior_std,
        "interior_n_px": int(interior.sum()),
        "band_strong_min_over_med": band_strong_min_over_med,
        "band_strong_cv": band_strong_cv,
        "band_strong_vals": band_vals_str,
    }


def main():
    a = pd.read_csv(ANCHORS)
    a = a[a["metric_error"].fillna("") == ""].copy()
    recs = []
    for _, row in a.iterrows():
        rec = {"experiment_id": str(row["experiment_id"]), "well_id": row["well_id"],
               "time_index": int(row["time_index"]), "user_label": row["user_label"],
               "emb_iqr_local": row.get("emb_iqr_local", np.nan),
               "image_path": row["image_path"], "mask_path": row["mask_path"]}
        rec.update(metric_row(row["image_path"], row["mask_path"]))
        recs.append(rec)
    # add ghosts by direct path
    for exp, well, t in GHOSTS:
        ip, mp = ghost_paths(exp, well, t)
        rec = {"experiment_id": exp, "well_id": well, "time_index": t,
               "user_label": "fail_blur_ghost", "emb_iqr_local": np.nan,
               "image_path": ip, "mask_path": mp}
        rec.update(metric_row(ip, mp))
        recs.append(rec)

    out = pd.DataFrame(recs)
    out["sort_key"] = out["user_label"].map(lambda c: CLASS.get(c, (9, "#444", False))[0])
    out["color"] = out["user_label"].map(lambda c: CLASS.get(c, (9, "#444", False))[1])
    out["in_verdict"] = out["user_label"].map(lambda c: CLASS.get(c, (9, "#444", False))[2])
    out = out.sort_values(["sort_key", "experiment_id", "well_id", "time_index"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    ok = out[out["struct_error"].fillna("") == ""]
    cols = ["user_label", "interior_edge_density", "interior_strong_edge_frac",
            "interior_grad_p90", "interior_lap_density", "interior_std"]
    print(ok[cols].to_string(index=False))

    # verdict: blur (ghost + fail) LOW structure vs in-focus HIGH structure
    v = ok[ok["in_verdict"]]
    blur = v[v["sort_key"].isin([0, 1])]
    focus = v[v["sort_key"].isin([3, 4])]
    print("\n=== SEPARATION (lower interior structure = more blurred) ===")
    for m in ["interior_edge_density", "interior_strong_edge_frac", "interior_grad_p90", "interior_lap_density"]:
        print(f"{m}: blur median={blur[m].median():.4f} [{blur[m].min():.4f},{blur[m].max():.4f}] | "
              f"in-focus median={focus[m].median():.4f} [{focus[m].min():.4f},{focus[m].max():.4f}]")
    print("\n=== ghosts specifically ===")
    g = ok[ok["sort_key"] == 0]
    print(g[["time_index", "interior_edge_density", "interior_strong_edge_frac", "interior_grad_p90"]].to_string(index=False))

    print("\n=== PER-BAND (partial defocus axis): low min/median or high CV = one bad region ===")
    bcols = ["user_label", "band_strong_min_over_med", "band_strong_cv", "band_strong_vals"]
    print(ok[bcols].to_string(index=False))

    labels = [f"{r.experiment_id} {r.well_id} t{int(r.time_index)}\n{r.user_label}" for _, r in ok.iterrows()]
    y = np.arange(len(ok))
    metrics = [("interior_strong_edge_frac", "WHOLE: strong-edge frac (ghost axis)"),
               ("interior_grad_p90", "WHOLE: gradient p90"),
               ("band_strong_min_over_med", "BAND: min/median (partial: LOW=bad)"),
               ("band_strong_cv", "BAND: CV (partial: HIGH=bad)"),
               ("interior_std", "interior intensity std")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), max(6, len(ok) * 0.40)), sharey=True)
    for ax, (m, title) in zip(axes, metrics, strict=True):
        ax.scatter(ok[m], y, c=ok["color"], s=46)
        ax.set_xlabel(m); ax.set_title(title); ax.grid(alpha=0.25)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=7); axes[0].invert_yaxis()
    fig.suptitle("Interior structure (LOW = blurred). darkred=ghosts, red=A10 blur, blue=gold dorsal, "
                 "green=gold in-focus. Do blurs sit LOW and in-focus HIGH?", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"\nSaved -> {OUT_FIG}")

    # --- scatter: the two complementary axes, anchors over the 7500 population ---
    scatter_fig = FIGURES / "interior_std_vs_strong_edge_frac_scatter.png"
    bg_csv = TABLES / "interior_structure_cross_experiment_metrics.csv"
    fig, ax = plt.subplots(figsize=(8.5, 7))
    if bg_csv.exists():
        bg = pd.read_csv(bg_csv)
        bg = bg[bg["interior_error"].fillna("") == ""]
        ax.scatter(bg["interior_std"], bg["interior_strong_edge_frac"],
                   s=5, c="#cccccc", alpha=0.4, label=f"all not-dead (n={len(bg)})", zorder=1)
    for lab, grp in ok.groupby("user_label"):
        ax.scatter(grp["interior_std"], grp["interior_strong_edge_frac"],
                   s=70, c=grp["color"].iloc[0], edgecolor="black", linewidth=0.5,
                   label=lab, zorder=3)
    ax.axhline(0.20, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax.text(ax.get_xlim()[1], 0.205, " strong_edge_frac=0.20 (ghost cut)",
            color="#d62728", fontsize=7, va="bottom", ha="right")
    ax.set_xlabel("interior_std  (catches ghost blur; NOT dead)")
    ax.set_ylabel("interior_strong_edge_frac  (catches ghost blur AND dead)")
    ax.set_title("Two complementary focus axes — anchors over the not-dead population\n"
                 "dead sits LOW-edge / MID-std; ghosts LOW/LOW; in-focus HIGH/HIGH")
    ax.legend(fontsize=6, loc="lower right", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(scatter_fig, dpi=170)
    plt.close(fig)
    print(f"Saved -> {scatter_fig}")


if __name__ == "__main__":
    main()
