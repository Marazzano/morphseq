#!/usr/bin/env python
"""Anchor registry for focus/saturation/feature-size QC review."""

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
FIGURES = HERE / "figures" / "focus_anchor_registry"
FIGURES.mkdir(parents=True, exist_ok=True)

FOCUS_CSV = TABLES / "focus_metrics_by_experiment.csv"

OUT_CSV = TABLES / "focus_anchor_registry_metrics.csv"
OUT_FIG = FIGURES / "focus_anchor_registry_diagnostic_panel.png"
OUT_SCALE_FIG = FIGURES / "focus_anchor_registry_scale_metrics.png"

ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5

ANCHORS = [
    # --- gold-standard in-focus set (user-curated 2026-06-30, high confidence) ---
    {
        "experiment_id": "20251125",
        "well_id": "A04",
        "time_index": 85,
        "user_label": "gold_in_focus",
        "notes": "user gold-standard in-focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "A04",
        "time_index": 93,
        "user_label": "gold_in_focus",
        "notes": "user gold-standard in-focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "A04",
        "time_index": 137,
        "user_label": "gold_in_focus",
        "notes": "user gold-standard in-focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "B01",
        "time_index": 169,
        "user_label": "gold_in_focus",
        "notes": "user gold-standard in-focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "E06",
        "time_index": 95,
        "user_label": "gold_in_focus",
        "notes": "user gold-standard in-focus",
    },
    # --- gold dorsal/bright BUT in-focus (challenging passes: must NOT be flagged) ---
    {
        "experiment_id": "20251125",
        "well_id": "E06",
        "time_index": 225,
        "user_label": "gold_dorsal_in_focus",
        "notes": "older/brighter/dorsal but in focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "E06",
        "time_index": 229,
        "user_label": "gold_dorsal_in_focus",
        "notes": "older/brighter/dorsal but in focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "H03",
        "time_index": 183,
        "user_label": "gold_dorsal_in_focus",
        "notes": "older/brighter/dorsal but in focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "H11",
        "time_index": 212,
        "user_label": "gold_dorsal_in_focus",
        "notes": "older/brighter/dorsal but in focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "H11",
        "time_index": 229,
        "user_label": "gold_dorsal_in_focus",
        "notes": "older/brighter/dorsal but in focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "D01",
        "time_index": 211,
        "user_label": "gold_dorsal_in_focus",
        "notes": "older/brighter/dorsal but in focus",
    },
    {
        "experiment_id": "20251125",
        "well_id": "A10",
        "time_index": 216,
        "user_label": "motion_and_partial_blur",
        "notes": (
            "motion artifact AND partial defocus: the HEAD is out of focus while the "
            "tail reads sharp, so whole-embryo metrics look in-focus from the tail. "
            "Test case for per-region (head->tail band) detection; motion also inflates "
            "edge lines, so treat as a known motion+partial-blur confound, not pure blur."
        ),
    },
    {
        "experiment_id": "20260206",
        "well_id": "A10",
        "time_index": 76,
        "user_label": "fail_blur_candidate",
        "notes": "user says pretty out of focus",
    },
    {
        "experiment_id": "20260206",
        "well_id": "G08",
        "time_index": 108,
        "user_label": "fail_blur_caught",
        "notes": "user says caught and actually out of focus",
    },
    {
        "experiment_id": "20260206",
        "well_id": "G01",
        "time_index": 47,
        "user_label": "pass_in_focus",
        "notes": "good in-focus control",
    },
    {
        "experiment_id": "20251125",
        "well_id": "H04",
        "time_index": 82,
        "user_label": "pass_dorsal_bright_review",
        "notes": "dorsal bright; likely brightness warning not blur",
    },
    {
        "experiment_id": "20251125",
        "well_id": "H09",
        "time_index": 66,
        "user_label": "dorsal_bright_review",
        "notes": "dorsal view, flagged bright/out of focus but maybe not blur",
    },
    {
        "experiment_id": "20250305",
        "well_id": "D03",
        "time_index": 262,
        "user_label": "dead_negative_control",
        "notes": "user calls dead/dead-like negative control",
    },
    {
        "experiment_id": "20251125",
        "well_id": "C09",
        "time_index": 143,
        "user_label": "dorsal_bright_review",
        "notes": "dorsal view, likely brightness warning not blur",
    },
    {
        "experiment_id": "20251125",
        "well_id": "E01",
        "time_index": 153,
        "user_label": "dorsal_bright_review",
        "notes": "dorsal view, likely brightness warning not blur",
    },
    {
        "experiment_id": "20251125",
        "well_id": "A10",
        "time_index": 224,
        "user_label": "fail_blur_anchor",
        "notes": "previous true out-of-focus anchor",
    },
    {
        "experiment_id": "20251125",
        "well_id": "A10",
        "time_index": 226,
        "user_label": "pass_after_blur_anchor",
        "notes": "nearby pass control after A10 t224",
    },
    {
        "experiment_id": "20251125",
        "well_id": "A06",
        "time_index": 76,
        "user_label": "saturation_fail_anchor",
        "notes": "previous very bright/saturation failure",
    },
    {
        "experiment_id": "20251125",
        "well_id": "A06",
        "time_index": 40,
        "user_label": "pass_anchor",
        "notes": "previous pass control",
    },
    {
        "experiment_id": "20251125",
        "well_id": "F01",
        "time_index": 166,
        "user_label": "pass_or_gray_anchor",
        "notes": "previous sharp-ish entropy failure review case",
    },
]


def entropy_u8(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    vals = np.clip(values, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(vals, bins=256, range=(0, 255))
    p = hist[hist > 0].astype(np.float64)
    if p.size == 0:
        return float("nan")
    p /= p.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def bbox(mask: np.ndarray, pad: int = 96) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    return (
        max(int(ys.min()) - pad, 0),
        min(int(ys.max()) + pad + 1, mask.shape[0]),
        max(int(xs.min()) - pad, 0),
        min(int(xs.max()) + pad + 1, mask.shape[1]),
    )


def robust_u8(img: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    vals = img[valid] if valid is not None else img.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return img.astype(np.uint8)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)


def current_focus_status(row: pd.Series) -> str:
    ent = bool(row["ff_rel_entropy"] < ENTROPY_CUT)
    lap = bool(row["ff_lap_abs_ratio"] < LAP_RATIO_CUT)
    if ent and lap:
        return "rejected_by_both"
    if ent:
        return "entropy_only_reject"
    if lap:
        return "sharpness_only_reject"
    return "accepted"


def load_anchor_rows() -> pd.DataFrame:
    focus = pd.read_csv(FOCUS_CSV, low_memory=False)
    focus["experiment_id"] = focus["experiment_id"].astype(str)
    focus["well_id"] = focus["well_id"].astype(str)
    focus["time_index_int"] = pd.to_numeric(focus["time_index"], errors="coerce").astype("Int64")
    rows = []
    for anchor in ANCHORS:
        hit = focus[
            (focus["experiment_id"] == anchor["experiment_id"])
            & (focus["well_id"] == anchor["well_id"])
            & (focus["time_index_int"] == anchor["time_index"])
        ]
        if hit.empty:
            rows.append({**anchor, "anchor_error": "not found"})
            continue
        rec = hit.iloc[0].to_dict()
        rec.update(anchor)
        rec["anchor_error"] = ""
        rec["current_focus_status_recomputed"] = current_focus_status(pd.Series(rec))
        rows.append(rec)
    return pd.DataFrame(rows)


def crop_and_metrics(row: pd.Series) -> tuple[dict, dict]:
    if row.get("anchor_error"):
        return {"metric_error": row["anchor_error"]}, {}

    img = np.array(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(row["mask_path"])).astype(bool)
    if mask.shape != img.shape or not mask.any():
        return {"metric_error": "bad mask"}, {}

    y0, y1, x0, x1 = bbox(mask)
    img_c = img[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]

    dil_inner = ndi.binary_dilation(mask_c, iterations=5)
    dil_outer = ndi.binary_dilation(mask_c, iterations=28)
    annulus = dil_outer & ~dil_inner
    if annulus.sum() < 256:
        dil_outer = ndi.binary_dilation(mask_c, iterations=60)
        annulus = dil_outer & ~dil_inner

    valid = mask_c | annulus
    img_norm = robust_u8(img_c, valid)
    norm01 = img_norm.astype(np.float32) / 255.0

    emb = mask_c
    eroded = ndi.binary_erosion(mask_c, iterations=5)
    if eroded.sum() < 256:
        eroded = mask_c
    bg = annulus

    emb_px = img_c[emb]
    bg_px = img_c[bg]
    emb_norm_px = img_norm[emb]
    bg_norm_px = img_norm[bg]

    sobel_norm = filters.sobel(norm01)
    log_norm = np.abs(ndi.gaussian_laplace(norm01, sigma=1.0))

    scale_sigmas = [1, 2, 4, 8]
    scale_metrics = {}
    for sigma in scale_sigmas:
        # Scale-normalized LoG approximates comparable blob/edge energy across feature sizes.
        response = np.abs((sigma**2) * ndi.gaussian_laplace(norm01, sigma=sigma))
        scale_metrics[f"log_scale_energy_sigma{sigma}_emb"] = float(np.nanmean(response[eroded]))
    fine = scale_metrics["log_scale_energy_sigma1_emb"] + scale_metrics["log_scale_energy_sigma2_emb"]
    coarse = scale_metrics["log_scale_energy_sigma4_emb"] + scale_metrics["log_scale_energy_sigma8_emb"]
    scale_metrics["log_fine_to_coarse_ratio"] = float(fine / (coarse + 1e-9))

    # Sobel feature-size sweep: Gaussian-smooth at increasing sigma before the Sobel
    # operator, then measure embryo edge energy at each scale. Defocus is a low-pass
    # process that erases fine detail while preserving coarse gradients, so the
    # fine/coarse ratio captures spectral tilt (focus) while dividing out absolute
    # content magnitude (stage/orientation). This is the Sobel analogue of the LoG
    # ratio, but less sensitive to saturation clipping than the LoG response.
    sobel_sigmas = [0, 1, 2, 4]
    for sigma in sobel_sigmas:
        work = ndi.gaussian_filter(norm01, sigma=sigma) if sigma > 0 else norm01
        sobel_s = filters.sobel(work)
        scale_metrics[f"sobel_scale_energy_sigma{sigma}_emb"] = float(np.nanmean(sobel_s[eroded]))
    sobel_fine = (
        scale_metrics["sobel_scale_energy_sigma0_emb"]
        + scale_metrics["sobel_scale_energy_sigma1_emb"]
    )
    sobel_coarse = (
        scale_metrics["sobel_scale_energy_sigma2_emb"]
        + scale_metrics["sobel_scale_energy_sigma4_emb"]
    )
    scale_metrics["sobel_fine_to_coarse_ratio"] = float(sobel_fine / (sobel_coarse + 1e-9))

    metrics = {
        "metric_error": "",
        "raw_entropy_emb_localcrop": entropy_u8(emb_px),
        "raw_entropy_local_annulus": entropy_u8(bg_px),
        "raw_rel_entropy_local_annulus": entropy_u8(emb_px) - entropy_u8(bg_px),
        "norm_entropy_emb": entropy_u8(emb_norm_px),
        "norm_entropy_local_annulus": entropy_u8(bg_norm_px),
        "norm_rel_entropy_local_annulus": entropy_u8(emb_norm_px) - entropy_u8(bg_norm_px),
        "emb_iqr_local": float(np.percentile(emb_px, 75) - np.percentile(emb_px, 25)),
        "emb_p50": float(np.percentile(emb_px, 50)),
        "emb_p95": float(np.percentile(emb_px, 95)),
        "emb_p99": float(np.percentile(emb_px, 99)),
        "emb_frac_gt245": float(np.mean(emb_px >= 245)),
        "emb_frac_gt250": float(np.mean(emb_px >= 250)),
        "emb_frac_eq255": float(np.mean(emb_px == 255)),
        "annulus_iqr": float(np.percentile(bg_px, 75) - np.percentile(bg_px, 25)) if bg_px.size else float("nan"),
        "sobel_mean_emb_norm": float(np.nanmean(sobel_norm[eroded])),
        "sobel_mean_annulus_norm": float(np.nanmean(sobel_norm[bg])) if bg.any() else float("nan"),
        "log_mean_emb_norm": float(np.nanmean(log_norm[eroded])),
        "log_mean_annulus_norm": float(np.nanmean(log_norm[bg])) if bg.any() else float("nan"),
        **scale_metrics,
    }
    metrics["sobel_ratio_norm"] = metrics["sobel_mean_emb_norm"] / (metrics["sobel_mean_annulus_norm"] + 1e-9)
    metrics["log_ratio_norm"] = metrics["log_mean_emb_norm"] / (metrics["log_mean_annulus_norm"] + 1e-9)

    images = {
        "raw": img_c,
        "norm": img_norm,
        "mask": mask_c,
        "sobel": sobel_norm,
        "log": log_norm,
    }
    return metrics, images


def compute_anchor_metrics() -> tuple[pd.DataFrame, list[dict]]:
    anchors = load_anchor_rows()
    rows = []
    image_rows = []
    for _, row in anchors.iterrows():
        metrics, images = crop_and_metrics(row)
        rec = row.to_dict()
        rec.update(metrics)
        rows.append(rec)
        if images:
            image_rows.append({"row": rec, "images": images})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    return out, image_rows


def plot_anchor_panel(image_rows: list[dict]) -> None:
    n = len(image_rows)
    fig, axes = plt.subplots(n, 4, figsize=(13, max(3, n * 2.4)), squeeze=False)
    for r, item in enumerate(image_rows):
        row = item["row"]
        images = item["images"]
        panels = [
            ("raw", images["raw"], "gray", 0, 255),
            ("local norm", images["norm"], "gray", 0, 255),
            ("Sobel norm", images["sobel"], "magma", None, None),
            ("LoG norm", images["log"], "magma", None, None),
        ]
        for c, (title, arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(images["mask"].astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.55)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title)
        label = (
            f"{row['experiment_id']} {row['well_id']} t{int(row['time_index'])}\n"
            f"{row['user_label']} | current={row.get('current_focus_status_recomputed', '')}\n"
            f"ff_ent={float(row['ff_rel_entropy']):.2f} IQR={row['emb_iqr_local']:.0f} "
            f"f245={row['emb_frac_gt245']:.2f} f250={row['emb_frac_gt250']:.2f}\n"
            f"LoGemb={row['log_mean_emb_norm']:.4f} fine/coarse={row['log_fine_to_coarse_ratio']:.2f}"
        )
        axes[r, 0].set_ylabel(label, fontsize=7)
    fig.suptitle("Focus anchor registry: raw/norm/embryo-only structure diagnostics", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)


def plot_scale_metrics(df: pd.DataFrame) -> None:
    plot_df = df[df["metric_error"].fillna("") == ""].copy()
    labels = [f"{r.experiment_id} {r.well_id} t{int(r.time_index)}\n{r.user_label}" for _, r in plot_df.iterrows()]
    y = np.arange(len(plot_df))
    metrics = [
        ("emb_frac_gt245", "sat frac >=245"),
        ("emb_iqr_local", "embryo IQR"),
        ("norm_rel_entropy_local_annulus", "local entropy delta"),
        ("log_mean_emb_norm", "embryo LoG"),
        ("log_fine_to_coarse_ratio", "LoG fine/coarse"),
        ("sobel_fine_to_coarse_ratio", "Sobel fine/coarse"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), max(4, len(plot_df) * 0.38)), sharey=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        ax.scatter(plot_df[metric], y, s=36)
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("Anchor metrics: saturation, information content, embryo-only sharpness, feature scale")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_SCALE_FIG, dpi=180)
    plt.close(fig)


def main() -> None:
    df, image_rows = compute_anchor_metrics()
    plot_anchor_panel(image_rows)
    plot_scale_metrics(df)
    print(f"Saved anchor metrics -> {OUT_CSV}")
    print(f"Saved anchor panel -> {OUT_FIG}")
    print(f"Saved scale metric figure -> {OUT_SCALE_FIG}")


if __name__ == "__main__":
    main()
