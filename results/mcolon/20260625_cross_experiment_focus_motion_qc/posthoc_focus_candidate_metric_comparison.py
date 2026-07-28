#!/usr/bin/env python
"""Compare candidate focus metrics that are less brightness-confounded."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, exposure


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "focus_candidate_metrics"
FIGURES.mkdir(parents=True, exist_ok=True)

FOCUS_CSV = TABLES / "focus_metrics_by_experiment.csv"
OUT_CSV = TABLES / "focus_candidate_metrics_20251125.csv"

ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5
RANDOM_SEED = 1729


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
        lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    out = exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 255))
    return np.clip(out, 0, 255).astype(np.uint8)


def metric_row(row: pd.Series) -> dict:
    img = np.array(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(row["mask_path"])).astype(bool)
    if mask.shape != img.shape or not mask.any():
        raise ValueError("bad mask")

    y0, y1, x0, x1 = bbox(mask)
    img_c = img[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]

    dil_outer = ndi.binary_dilation(mask_c, iterations=28)
    dil_inner = ndi.binary_dilation(mask_c, iterations=5)
    annulus = dil_outer & ~dil_inner
    if annulus.sum() < 256:
        dil_outer = ndi.binary_dilation(mask_c, iterations=60)
        annulus = dil_outer & ~dil_inner

    valid_crop = mask_c | annulus
    img_norm = robust_u8(img_c, valid_crop)

    sobel_raw = filters.sobel(img_c / 255.0)
    sobel_norm = filters.sobel(img_norm.astype(np.float32) / 255.0)
    log_raw = np.abs(ndi.gaussian_laplace(img_c.astype(np.float32) / 255.0, sigma=1.0))
    log_norm = np.abs(ndi.gaussian_laplace(img_norm.astype(np.float32) / 255.0, sigma=1.0))

    emb = mask_c
    bg = annulus
    emb_px = img_c[emb]
    bg_px = img_c[bg]
    emb_norm_px = img_norm[emb]
    bg_norm_px = img_norm[bg]

    def ratio(arr: np.ndarray) -> float:
        bg_mean = float(np.nanmean(arr[bg])) if bg.any() else float("nan")
        emb_mean = float(np.nanmean(arr[emb])) if emb.any() else float("nan")
        if not np.isfinite(bg_mean) or bg_mean <= 0:
            return float("nan")
        return emb_mean / bg_mean

    return {
        "local_bg_n_px": int(bg.sum()),
        "raw_entropy_emb_localcrop": entropy_u8(emb_px),
        "raw_entropy_local_annulus": entropy_u8(bg_px),
        "raw_rel_entropy_local_annulus": entropy_u8(emb_px) - entropy_u8(bg_px),
        "norm_entropy_emb": entropy_u8(emb_norm_px),
        "norm_entropy_local_annulus": entropy_u8(bg_norm_px),
        "norm_rel_entropy_local_annulus": entropy_u8(emb_norm_px) - entropy_u8(bg_norm_px),
        "sobel_mean_emb_raw": float(np.nanmean(sobel_raw[emb])),
        "sobel_mean_annulus_raw": float(np.nanmean(sobel_raw[bg])),
        "sobel_ratio_raw": ratio(sobel_raw),
        "sobel_mean_emb_norm": float(np.nanmean(sobel_norm[emb])),
        "sobel_mean_annulus_norm": float(np.nanmean(sobel_norm[bg])),
        "sobel_ratio_norm": ratio(sobel_norm),
        "log_mean_emb_raw": float(np.nanmean(log_raw[emb])),
        "log_mean_annulus_raw": float(np.nanmean(log_raw[bg])),
        "log_ratio_raw": ratio(log_raw),
        "log_mean_emb_norm": float(np.nanmean(log_norm[emb])),
        "log_mean_annulus_norm": float(np.nanmean(log_norm[bg])),
        "log_ratio_norm": ratio(log_norm),
        "emb_frac_gt245": float(np.mean(emb_px >= 245)),
        "emb_frac_gt250": float(np.mean(emb_px >= 250)),
        "emb_p01": float(np.percentile(emb_px, 1)),
        "emb_p05": float(np.percentile(emb_px, 5)),
        "emb_p95": float(np.percentile(emb_px, 95)),
        "emb_p99": float(np.percentile(emb_px, 99)),
        "emb_iqr_local": float(np.percentile(emb_px, 75) - np.percentile(emb_px, 25)),
        "annulus_p05": float(np.percentile(bg_px, 5)) if bg_px.size else float("nan"),
        "annulus_p95": float(np.percentile(bg_px, 95)) if bg_px.size else float("nan"),
        "annulus_iqr": float(np.percentile(bg_px, 75) - np.percentile(bg_px, 25)) if bg_px.size else float("nan"),
    }


def focus_status(df: pd.DataFrame) -> pd.Series:
    ent = df["ff_rel_entropy"] < ENTROPY_CUT
    lap = df["ff_lap_abs_ratio"] < LAP_RATIO_CUT
    return np.select(
        [ent & lap, ent & ~lap, ~ent & lap],
        ["rejected_by_both", "entropy_only_reject", "sharpness_only_reject"],
        default="accepted",
    )


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(FOCUS_CSV)
    df = df[(df["experiment_id"].astype(str) == "20251125") & df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["ff_rel_entropy", "ff_lap_abs_ratio"]).copy()
    df["status"] = focus_status(df)
    df["well_s"] = df["well_id"].fillna(df.get("well", "")).astype(str).str.upper()
    df["time_index"] = pd.to_numeric(df["time_index"], errors="coerce").astype(int)

    anchors = df[
        ((df["well_s"].str.contains("A06")) & df["time_index"].isin([40, 76]))
        | ((df["well_s"].str.contains("F01")) & (df["time_index"] == 166))
    ].copy()
    anchors["cohort"] = "anchor"

    parts = [anchors]
    for status, n in [
        ("entropy_only_reject", 250),
        ("accepted", 250),
        ("rejected_by_both", 50),
    ]:
        sub = df[df["status"] == status].copy()
        if len(sub) > n:
            sub = sub.sample(n=n, random_state=RANDOM_SEED)
        sub["cohort"] = status
        parts.append(sub)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["well_id", "time_index"])


def compute() -> pd.DataFrame:
    sample = load_sample()
    rows = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        base = row.to_dict()
        try:
            rows.append({**base, **metric_row(row)})
        except Exception as exc:
            rows.append({**base, "candidate_metric_error": str(exc)})
        if i % 100 == 0 or i == len(sample):
            print(f"{i}/{len(sample)} candidate focus examples processed", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    return out


def plot_anchor_table(df: pd.DataFrame) -> None:
    anchors = df[df["cohort"] == "anchor"].copy()
    cols = [
        "ff_rel_entropy",
        "raw_rel_entropy_local_annulus",
        "norm_rel_entropy_local_annulus",
        "sobel_ratio_raw",
        "sobel_ratio_norm",
        "log_ratio_raw",
        "log_ratio_norm",
        "emb_frac_gt245",
        "emb_iqr_local",
    ]
    labels = [f"{r.well_id} t{r.time_index}" for _, r in anchors.iterrows()]
    data = anchors[cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, max(2.2, len(anchors) * 0.6)))
    ax.axis("off")
    table_data = []
    for label, vals in zip(labels, data):
        table_data.append([label] + [f"{v:.3g}" for v in vals])
    table = ax.table(cellText=table_data, colLabels=["example"] + cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    ax.set_title("Anchor examples: raw/local/normalized entropy and structure metrics")
    fig.tight_layout()
    fig.savefig(FIGURES / "anchor_candidate_metric_table.png", dpi=180)
    plt.close(fig)


def scatter(df: pd.DataFrame, x: str, y: str, out_name: str) -> None:
    colors = {
        "accepted": "#1a8820",
        "entropy_only_reject": "#b5651d",
        "rejected_by_both": "#7b1fa2",
        "anchor": "black",
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    for cohort, grp in df.groupby("cohort", observed=True):
        ax.scatter(grp[x], grp[y], s=16 if cohort != "anchor" else 70, alpha=0.55, label=cohort, color=colors.get(cohort))
    anchors = df[df["cohort"] == "anchor"]
    for _, r in anchors.iterrows():
        ax.text(r[x], r[y], f"{r.well_id} t{r.time_index}", fontsize=7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / out_name, dpi=180)
    plt.close(fig)


def plot(df: pd.DataFrame) -> None:
    plot_anchor_table(df)
    scatter(df, "ff_rel_entropy", "raw_rel_entropy_local_annulus", "current_vs_local_annulus_entropy.png")
    scatter(df, "ff_rel_entropy", "norm_rel_entropy_local_annulus", "current_vs_norm_local_entropy.png")
    scatter(df, "emb_frac_gt245", "sobel_ratio_norm", "saturation_vs_norm_sobel_ratio.png")
    scatter(df, "emb_frac_gt245", "log_ratio_norm", "saturation_vs_norm_log_ratio.png")
    scatter(df, "emb_iqr_local", "norm_rel_entropy_local_annulus", "iqr_vs_norm_local_entropy.png")


def main() -> None:
    df = compute()
    plot(df)
    print(f"Saved candidate metrics -> {OUT_CSV}")
    print(f"Saved figures -> {FIGURES}")


if __name__ == "__main__":
    main()
