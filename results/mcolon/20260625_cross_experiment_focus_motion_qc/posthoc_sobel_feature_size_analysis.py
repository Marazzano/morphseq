#!/usr/bin/env python
"""Sobel-ranked focus review and feature-size profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "sobel_feature_size"
FIGURES.mkdir(parents=True, exist_ok=True)

BENCHMARK = TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv"
ANCHORS = TABLES / "focus_anchor_registry_metrics.csv"
OUT_METRICS = TABLES / "focus_20251125_sobel_feature_size_metrics_benchmark.csv"

ENTROPY_CUT = -0.55
IQR_FAIL_CUT = 50.0
IQR_PASS_CUT = 60.0
SAT245_FAIL_CUT = 0.20
SAT250_FAIL_CUT = 0.10
N_DECILES = 10
N_PER_DECILE = 12

PROBLEM_COLORS = {
    "saturation_fail": "#d62728",
    "low_iqr_fail": "#9467bd",
    "current_entropy_fail": "#b5651d",
    "accepted_or_other": "#1a8820",
}

ANCHOR_COLORS = {
    "fail_blur_candidate": "#d62728",
    "fail_blur_caught": "#d62728",
    "fail_blur_anchor": "#d62728",
    "saturation_fail_anchor": "#9467bd",
    "dead_negative_control": "#000000",
    "pass_in_focus": "#1a8820",
    "pass_anchor": "#1a8820",
    "pass_after_blur_anchor": "#1a8820",
    "pass_dorsal_bright_review": "#ff7f0e",
    "dorsal_bright_review": "#ff7f0e",
    "pass_or_gray_anchor": "#7f7f7f",
}


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


def robust_u8(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = img[valid & np.isfinite(img)]
    if vals.size == 0:
        vals = img[np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)


def read_crop(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.array(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(row["mask_path"])).astype(bool)
    if mask.shape != img.shape or not mask.any():
        raise ValueError("bad mask")
    y0, y1, x0, x1 = bbox(mask)
    img_c = img[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]
    dil_inner = ndi.binary_dilation(mask_c, iterations=5)
    dil_outer = ndi.binary_dilation(mask_c, iterations=28)
    annulus = dil_outer & ~dil_inner
    if annulus.sum() < 256:
        dil_outer = ndi.binary_dilation(mask_c, iterations=60)
        annulus = dil_outer & ~dil_inner
    return img_c, mask_c, annulus


def metric_row(row: pd.Series) -> dict:
    img_c, mask_c, annulus = read_crop(row)
    valid = mask_c | annulus
    img_norm = robust_u8(img_c, valid).astype(np.float32) / 255.0
    emb = ndi.binary_erosion(mask_c, iterations=5)
    if emb.sum() < 256:
        emb = mask_c

    out: dict[str, float] = {"feature_metric_error": ""}
    sobel_sigmas = [0, 1, 2, 4]
    for sigma in sobel_sigmas:
        work = ndi.gaussian_filter(img_norm, sigma=sigma) if sigma > 0 else img_norm
        sobel = filters.sobel(work)
        out[f"sobel_sigma{sigma}_emb"] = float(np.nanmean(sobel[emb]))

    log_sigmas = [1, 2, 4, 8]
    for sigma in log_sigmas:
        log = np.abs((sigma**2) * ndi.gaussian_laplace(img_norm, sigma=sigma))
        out[f"log_scale_sigma{sigma}_emb"] = float(np.nanmean(log[emb]))

    out["sobel_fine_to_coarse_ratio"] = float(
        (out["sobel_sigma0_emb"] + out["sobel_sigma1_emb"])
        / (out["sobel_sigma2_emb"] + out["sobel_sigma4_emb"] + 1e-9)
    )
    out["log_fine_to_coarse_ratio"] = float(
        (out["log_scale_sigma1_emb"] + out["log_scale_sigma2_emb"])
        / (out["log_scale_sigma4_emb"] + out["log_scale_sigma8_emb"] + 1e-9)
    )
    return out


def problem_class(row: pd.Series) -> str:
    if row["emb_frac_gt245"] >= SAT245_FAIL_CUT or row["emb_frac_gt250"] >= SAT250_FAIL_CUT:
        return "saturation_fail"
    if row["emb_iqr_local"] < IQR_FAIL_CUT:
        return "low_iqr_fail"
    if row["ff_rel_entropy"] < ENTROPY_CUT:
        return "current_entropy_fail"
    return "accepted_or_other"


def iqr_class(row: pd.Series) -> str:
    if row["emb_iqr_local"] < IQR_FAIL_CUT:
        return "iqr_fail_<50"
    if row["emb_iqr_local"] < IQR_PASS_CUT:
        return "iqr_warn_50_60"
    return "iqr_pass_>=60"


def load_or_compute() -> pd.DataFrame:
    if OUT_METRICS.exists():
        print(f"Loading existing feature metrics -> {OUT_METRICS}", flush=True)
        return pd.read_csv(OUT_METRICS)

    df = pd.read_csv(BENCHMARK)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(
        subset=[
            "image_path",
            "mask_path",
            "emb_iqr_local",
            "emb_frac_gt245",
            "emb_frac_gt250",
            "ff_rel_entropy",
        ]
    ).copy()

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rec = row.to_dict()
        try:
            rec.update(metric_row(row))
        except Exception as exc:  # noqa: BLE001
            rec.update({"feature_metric_error": str(exc)})
        rows.append(rec)
        if i % 250 == 0 or i == len(df):
            print(f"[sobel-feature] processed {i}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)
    out["problem_class"] = out.apply(problem_class, axis=1)
    out["iqr_class"] = out.apply(iqr_class, axis=1)
    out.to_csv(OUT_METRICS, index=False)
    print(f"Saved feature metrics -> {OUT_METRICS}", flush=True)
    return out


def crop_for_gallery(image_path: object, mask_path: object, pad: int = 60) -> tuple[np.ndarray | None, np.ndarray | None]:
    ip = Path(str(image_path))
    mp = Path(str(mask_path))
    if not ip.exists() or not mp.exists():
        return None, None
    img = np.array(Image.open(ip).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(mp)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        return img, None
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def sample_sobel_deciles(df: pd.DataFrame, metric: str = "sobel_sigma1_emb") -> pd.DataFrame:
    use = df[df["feature_metric_error"].fillna("") == ""].dropna(subset=[metric]).copy()
    use = use.sort_values(metric, ascending=True)
    use["sobel_decile"] = (
        pd.qcut(use[metric].rank(method="first"), N_DECILES, labels=False, duplicates="drop").astype(int) + 1
    )
    rows = []
    for decile, grp in use.groupby("sobel_decile", observed=True):
        grp = grp.sort_values(metric)
        if len(grp) <= N_PER_DECILE:
            picks = grp
        else:
            picks = grp.iloc[np.linspace(0, len(grp) - 1, N_PER_DECILE).astype(int)]
        picks = picks.copy()
        picks["sample_rank_within_decile"] = np.arange(1, len(picks) + 1)
        rows.append(picks)
    manifest = pd.concat(rows, ignore_index=True)
    out = TABLES / "focus_20251125_sobel_sigma1_decile_samples_benchmark.csv"
    manifest.to_csv(out, index=False)
    print(f"Saved Sobel decile samples -> {out}", flush=True)
    return manifest


def plot_sobel_gallery(manifest: pd.DataFrame, metric: str = "sobel_sigma1_emb") -> None:
    fig, axes = plt.subplots(N_DECILES, N_PER_DECILE, figsize=(N_PER_DECILE * 1.8, N_DECILES * 1.85), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for r, decile in enumerate(sorted(manifest["sobel_decile"].astype(int).unique())):
        grp = manifest[manifest["sobel_decile"].astype(int) == decile].sort_values("sample_rank_within_decile")
        for c, (_, row) in enumerate(grp.head(N_PER_DECILE).iterrows()):
            ax = axes[r, c]
            img, mask = crop_for_gallery(row["image_path"], row["mask_path"])
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                if mask is not None and mask.any():
                    ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.45)
            ax.axis("on")
            cls = str(row["problem_class"])
            for spine in ax.spines.values():
                spine.set_color(PROBLEM_COLORS.get(cls, "#444444"))
                spine.set_linewidth(2.0)
            ax.set_title(
                f"{row['well_id']} t{int(row['time_index'])}\n"
                f"Sobel={row[metric]:.3f} IQR={row['emb_iqr_local']:.0f}\n"
                f"f245={row['emb_frac_gt245']:.2f} ent={row['ff_rel_entropy']:.2f}",
                fontsize=5,
            )
        axes[r, 0].set_ylabel(f"D{decile}", fontsize=8)

    handles = [Patch(facecolor=v, edgecolor="none", label=k) for k, v in PROBLEM_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=4, frameon=False)
    fig.suptitle("20251125 not-dead benchmark: Sobel sigma=1 embryo-only deciles (D1 low -> D10 high)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / "20251125_sobel_sigma1_emb_deciles_problem_colored.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved Sobel decile gallery -> {out}", flush=True)


def plot_feature_profiles(df: pd.DataFrame) -> None:
    use = df[df["feature_metric_error"].fillna("") == ""].copy()
    sobel_cols = ["sobel_sigma0_emb", "sobel_sigma1_emb", "sobel_sigma2_emb", "sobel_sigma4_emb"]
    log_cols = ["log_scale_sigma1_emb", "log_scale_sigma2_emb", "log_scale_sigma4_emb", "log_scale_sigma8_emb"]
    sobel_x = [0, 1, 2, 4]
    log_x = [1, 2, 4, 8]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    class_order = ["iqr_fail_<50", "iqr_warn_50_60", "iqr_pass_>=60"]
    for cls in class_order:
        grp = use[use["iqr_class"] == cls]
        if grp.empty:
            continue
        axes[0, 0].plot(sobel_x, grp[sobel_cols].median(), marker="o", label=f"{cls} n={len(grp)}")
        axes[0, 1].plot(log_x, grp[log_cols].median(), marker="o", label=f"{cls} n={len(grp)}")

    pass_df = use[use["emb_iqr_local"] >= IQR_PASS_CUT].copy()
    pass_df["sobel_decile"] = (
        pd.qcut(pass_df["sobel_sigma1_emb"].rank(method="first"), N_DECILES, labels=False, duplicates="drop").astype(int) + 1
    )
    groups = [
        ("IQR-pass Sobel D1-D2", pass_df[pass_df["sobel_decile"].isin([1, 2])]),
        ("IQR-pass Sobel D5-D6", pass_df[pass_df["sobel_decile"].isin([5, 6])]),
        ("IQR-pass Sobel D9-D10", pass_df[pass_df["sobel_decile"].isin([9, 10])]),
    ]
    for label, grp in groups:
        if grp.empty:
            continue
        axes[1, 0].plot(sobel_x, grp[sobel_cols].median(), marker="o", label=f"{label} n={len(grp)}")
        axes[1, 1].plot(log_x, grp[log_cols].median(), marker="o", label=f"{label} n={len(grp)}")

    for ax in axes[:, 0]:
        ax.set_xlabel("Gaussian smoothing sigma before Sobel (pixels)")
        ax.set_ylabel("median embryo Sobel energy")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[:, 1]:
        ax.set_xlabel("scale-normalized LoG sigma (pixels)")
        ax.set_ylabel("median embryo LoG energy")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[0, 0].set_title("Sobel feature-scale profile by IQR class")
    axes[0, 1].set_title("LoG feature-scale profile by IQR class")
    axes[1, 0].set_title("Within IQR-pass embryos: Sobel profile by Sobel rank")
    axes[1, 1].set_title("Within IQR-pass embryos: LoG profile by Sobel rank")
    fig.suptitle("20251125 feature-size sensitivity: information-rich vs low-IQR embryos")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / "20251125_feature_size_profiles_by_iqr_and_sobel_rank.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved feature-size profiles -> {out}", flush=True)


def plot_anchor_sorted() -> None:
    if not ANCHORS.exists():
        return
    df = pd.read_csv(ANCHORS)
    df = df[df["metric_error"].fillna("") == ""].copy()
    order = {
        "fail_blur_candidate": 1,
        "fail_blur_caught": 1,
        "fail_blur_anchor": 1,
        "saturation_fail_anchor": 2,
        "dead_negative_control": 2,
        "dorsal_bright_review": 3,
        "pass_dorsal_bright_review": 3,
        "pass_or_gray_anchor": 4,
        "pass_in_focus": 5,
        "pass_anchor": 5,
        "pass_after_blur_anchor": 5,
    }
    df["sort_key"] = df["user_label"].map(order).fillna(99)
    df = df.sort_values(["sort_key", "experiment_id", "well_id", "time_index"]).reset_index(drop=True)
    labels = [f"{r.experiment_id} {r.well_id} t{int(r.time_index)}\n{r.user_label}" for _, r in df.iterrows()]
    y = np.arange(len(df))
    metrics = [
        ("emb_frac_gt245", "sat >=245"),
        ("emb_iqr_local", "IQR"),
        ("norm_rel_entropy_local_annulus", "local entropy"),
        ("sobel_mean_emb_norm", "Sobel emb"),
        ("log_mean_emb_norm", "LoG emb"),
        ("log_fine_to_coarse_ratio", "LoG fine/coarse"),
        ("sobel_fine_to_coarse_ratio", "Sobel fine/coarse"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), max(5, len(df) * 0.45)), sharey=True)
    colors = [ANCHOR_COLORS.get(str(v), "#7f7f7f") for v in df["user_label"]]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        ax.scatter(df[metric], y, c=colors, s=42)
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    handles = [
        Patch(facecolor="#d62728", label="blur fail anchors"),
        Patch(facecolor="#9467bd", label="saturation/dead-like"),
        Patch(facecolor="#ff7f0e", label="dorsal bright review"),
        Patch(facecolor="#1a8820", label="pass anchors"),
        Patch(facecolor="#7f7f7f", label="gray"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False)
    fig.suptitle("Anchor metrics sorted by problem class")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / "focus_anchor_metrics_sorted_by_problem_colored.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved sorted anchor metric plot -> {out}", flush=True)


def write_summary(df: pd.DataFrame) -> None:
    use = df[df["feature_metric_error"].fillna("") == ""].copy()
    summary = (
        use.groupby(["problem_class", "iqr_class"], observed=True)
        .agg(
            n=("image_id", "size"),
            sobel_sigma1_median=("sobel_sigma1_emb", "median"),
            sobel_sigma1_q10=("sobel_sigma1_emb", lambda x: float(np.quantile(x, 0.10))),
            sobel_sigma1_q90=("sobel_sigma1_emb", lambda x: float(np.quantile(x, 0.90))),
            log_fine_to_coarse_median=("log_fine_to_coarse_ratio", "median"),
            emb_iqr_median=("emb_iqr_local", "median"),
            emb_frac_gt245_median=("emb_frac_gt245", "median"),
            ff_rel_entropy_median=("ff_rel_entropy", "median"),
        )
        .reset_index()
    )
    out = TABLES / "focus_20251125_sobel_feature_size_group_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Saved Sobel/feature group summary -> {out}", flush=True)


def main() -> None:
    df = load_or_compute()
    manifest = sample_sobel_deciles(df)
    plot_sobel_gallery(manifest)
    plot_feature_profiles(df)
    plot_anchor_sorted()
    write_summary(df)


if __name__ == "__main__":
    main()
