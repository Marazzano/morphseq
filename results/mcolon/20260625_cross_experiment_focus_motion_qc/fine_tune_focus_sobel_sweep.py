#!/usr/bin/env python
"""Fine-tune focus QC Sobel threshold and normalization choices.

This script intentionally writes only under results/mcolon/20260625_cross_experiment_focus_motion_qc/
fine_tuning/. It reuses the population and anchor manifests from the latest focus handoff and
recomputes the interior strong-edge fraction from image/mask pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
OUT = HERE / "fine_tuning"
OUT_TABLES = OUT / "tables"
OUT_FIGURES = OUT / "figures"

POP_CSV = TABLES / "interior_structure_cross_experiment_metrics.csv"
ANCHOR_CSV = TABLES / "anchor_interior_structure_metrics.csv"

SOBEL_THRESHOLDS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
FOCUS_GATES = (0.20, 0.25)
NORMALIZATION_MODES = ("local_context", "mask_only", "raw_255", "crop_global")


FOCUS_COLORS = {
    "fail_blur_ghost": "#8b0000",
    "motion_and_partial_blur": "#e377c2",
    "fail_blur_anchor": "#d62728",
    "fail_blur_candidate": "#d62728",
    "fail_blur_caught": "#d62728",
    "dorsal_bright_review": "#ff7f0e",
    "pass_dorsal_bright_review": "#ff7f0e",
    "gold_dorsal_in_focus": "#1f77b4",
    "gold_in_focus": "#1a8820",
    "pass_in_focus": "#1a8820",
    "pass_anchor": "#1a8820",
    "pass_after_blur_anchor": "#1a8820",
    "pass_or_gray_anchor": "#7f7f7f",
    "dead_negative_control": "#000000",
    "saturation_fail_anchor": "#9467bd",
}


def successful_rows(df: pd.DataFrame) -> pd.Series:
    if "metric_error" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["metric_error"].fillna("") == ""


@dataclass(frozen=True)
class MetricInputs:
    image_crop: np.ndarray
    mask_crop: np.ndarray
    interior: np.ndarray
    valid_local_context: np.ndarray


def bbox(mask: np.ndarray, pad: int = 32) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    return (
        max(int(ys.min()) - pad, 0),
        min(int(ys.max()) + pad + 1, mask.shape[0]),
        max(int(xs.min()) - pad, 0),
        min(int(xs.max()) + pad + 1, mask.shape[1]),
    )


def robust01(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = img[valid & np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32)


def load_metric_inputs(image_path: str, mask_path: str) -> MetricInputs:
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(mask_path)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        raise ValueError("bad mask")

    y0, y1, x0, x1 = bbox(mask, pad=32)
    img_c = img[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]

    dil = ndi.binary_dilation(mask_c, iterations=20)
    valid_local_context = mask_c | dil

    interior = ndi.binary_erosion(mask_c, iterations=12)
    if interior.sum() < 200:
        interior = ndi.binary_erosion(mask_c, iterations=6)
    if interior.sum() < 200:
        interior = mask_c

    return MetricInputs(
        image_crop=img_c,
        mask_crop=mask_c,
        interior=interior,
        valid_local_context=valid_local_context,
    )


def normalized_image(inputs: MetricInputs, mode: str) -> np.ndarray:
    if mode == "local_context":
        return robust01(inputs.image_crop, inputs.valid_local_context)
    if mode == "mask_only":
        return robust01(inputs.image_crop, inputs.mask_crop)
    if mode == "raw_255":
        return np.clip(inputs.image_crop / 255.0, 0, 1).astype(np.float32)
    if mode == "crop_global":
        return robust01(inputs.image_crop, np.isfinite(inputs.image_crop))
    raise ValueError(f"unknown normalization mode: {mode}")


def metric_rows_for_record(record: pd.Series, source: str) -> list[dict]:
    inputs = load_metric_inputs(str(record["image_path"]), str(record["mask_path"]))
    rows: list[dict] = []

    base = {
        "source": source,
        "experiment_id": str(record.get("experiment_id", "")),
        "well_id": str(record.get("well_id", record.get("well", ""))),
        "time_index": int(record["time_index"]) if pd.notna(record.get("time_index")) else -1,
        "snip_id": str(record.get("snip_id", "")),
        "user_label": str(record.get("user_label", "")),
        "top_spread_p99_p90": float(record.get("top_spread_p99_p90", np.nan)),
        "interior_mean": float(record.get("interior_mean", np.nan)),
        "interior_n_px": int(inputs.interior.sum()),
    }

    for mode in NORMALIZATION_MODES:
        img01 = normalized_image(inputs, mode)
        grad = filters.sobel(ndi.gaussian_filter(img01, sigma=1.0))
        g = grad[inputs.interior]
        for threshold in SOBEL_THRESHOLDS:
            rows.append(
                {
                    **base,
                    "normalization_mode": mode,
                    "sobel_threshold": threshold,
                    "interior_strong_edge_frac": float(np.mean(g > threshold)),
                    "interior_grad_p50": float(np.percentile(g, 50)),
                    "interior_grad_p90": float(np.percentile(g, 90)),
                    "interior_grad_p95": float(np.percentile(g, 95)),
                    "interior_grad_mean": float(np.mean(g)),
                }
            )
    return rows


def compute_metrics() -> pd.DataFrame:
    pop = pd.read_csv(POP_CSV, low_memory=False)
    pop = pop[pop["interior_error"].fillna("") == ""].copy()
    anchors = pd.read_csv(ANCHOR_CSV, low_memory=False)
    anchors = anchors[anchors["struct_error"].fillna("") == ""].copy()

    rows: list[dict] = []
    for i, (_, record) in enumerate(pop.iterrows(), start=1):
        try:
            rows.extend(metric_rows_for_record(record, source="population"))
        except Exception as exc:
            rows.append(
                {
                    "source": "population",
                    "experiment_id": str(record.get("experiment_id", "")),
                    "well_id": str(record.get("well_id", record.get("well", ""))),
                    "time_index": record.get("time_index", np.nan),
                    "snip_id": str(record.get("snip_id", "")),
                    "metric_error": str(exc),
                }
            )
        if i % 250 == 0 or i == len(pop):
            print(f"population {i}/{len(pop)}", flush=True)

    for i, (_, record) in enumerate(anchors.iterrows(), start=1):
        try:
            rows.extend(metric_rows_for_record(record, source="anchor"))
        except Exception as exc:
            rows.append(
                {
                    "source": "anchor",
                    "experiment_id": str(record.get("experiment_id", "")),
                    "well_id": str(record.get("well_id", "")),
                    "time_index": record.get("time_index", np.nan),
                    "user_label": str(record.get("user_label", "")),
                    "metric_error": str(exc),
                }
            )
        if i % 10 == 0 or i == len(anchors):
            print(f"anchors {i}/{len(anchors)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "focus_sobel_threshold_sweep_metrics.csv", index=False)
    return out


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = metrics[successful_rows(metrics)].copy()
    pop = ok[ok["source"] == "population"].copy()
    anchor = ok[ok["source"] == "anchor"].copy()

    summary_rows = []
    for gate in FOCUS_GATES:
        tmp = pop.copy()
        tmp["focus_flag"] = tmp["interior_strong_edge_frac"] < gate
        grp = tmp.groupby(["normalization_mode", "sobel_threshold", "experiment_id"], dropna=False)
        summary_rows.extend(
            {
                "normalization_mode": mode,
                "sobel_threshold": threshold,
                "focus_gate": gate,
                "experiment_id": exp,
                "n": len(g),
                "flag_n": int(g["focus_flag"].sum()),
                "flag_frac": float(g["focus_flag"].mean()),
            }
            for (mode, threshold, exp), g in grp
        )
        grp_all = tmp.groupby(["normalization_mode", "sobel_threshold"], dropna=False)
        summary_rows.extend(
            {
                "normalization_mode": mode,
                "sobel_threshold": threshold,
                "focus_gate": gate,
                "experiment_id": "ALL",
                "n": len(g),
                "flag_n": int(g["focus_flag"].sum()),
                "flag_frac": float(g["focus_flag"].mean()),
            }
            for (mode, threshold), g in grp_all
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_TABLES / "focus_sobel_threshold_sweep_summary.csv", index=False)

    anchor_rows = []
    for gate in FOCUS_GATES:
        tmp = anchor.copy()
        tmp["focus_flag"] = tmp["interior_strong_edge_frac"] < gate
        grp = tmp.groupby(["normalization_mode", "sobel_threshold", "user_label"], dropna=False)
        anchor_rows.extend(
            {
                "normalization_mode": mode,
                "sobel_threshold": threshold,
                "focus_gate": gate,
                "user_label": label,
                "n": len(g),
                "flag_n": int(g["focus_flag"].sum()),
                "flag_frac": float(g["focus_flag"].mean()),
                "median_edge_frac": float(g["interior_strong_edge_frac"].median()),
            }
            for (mode, threshold, label), g in grp
        )
    anchor_summary = pd.DataFrame(anchor_rows)
    anchor_summary.to_csv(OUT_TABLES / "focus_sobel_threshold_anchor_summary.csv", index=False)

    return summary, anchor_summary


def plot_histograms(metrics: pd.DataFrame) -> None:
    ok = metrics[successful_rows(metrics)].copy()
    pop = ok[
        (ok["source"] == "population")
        & (ok["sobel_threshold"] == 0.04)
    ].copy()
    anchors = ok[
        (ok["source"] == "anchor")
        & (ok["sobel_threshold"] == 0.04)
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    bins = np.linspace(0, 1, 81)
    for ax, mode in zip(axes, NORMALIZATION_MODES, strict=True):
        sub = pop[pop["normalization_mode"] == mode]
        counts, _, _ = ax.hist(sub["interior_strong_edge_frac"], bins=bins, color="#777", alpha=0.85)
        y_top = float(np.nanmax(counts)) if len(counts) else 1.0
        ax.axvline(0.20, color="#d62728", ls=":", lw=1.2, label="gate 0.20")
        ax.axvline(0.25, color="#d62728", ls="--", lw=1.2, label="gate 0.25")

        anchor_sub = anchors[anchors["normalization_mode"] == mode].copy()
        if not anchor_sub.empty:
            label_offsets: dict[str, int] = {}
            for _, row in anchor_sub.sort_values(["user_label", "interior_strong_edge_frac"]).iterrows():
                label = str(row["user_label"])
                x = float(row["interior_strong_edge_frac"])
                color = FOCUS_COLORS.get(label, "#444")
                count_for_label = label_offsets.get(label, 0)
                label_offsets[label] = count_for_label + 1
                y0 = y_top * (0.03 + 0.025 * (count_for_label % 4))
                y1 = y_top * (0.18 + 0.025 * (count_for_label % 4))
                ax.vlines(x, y0, y1, color=color, lw=1.4, alpha=0.95)
                ax.text(
                    x,
                    y1 + y_top * 0.015,
                    label.replace("_", " "),
                    rotation=90,
                    fontsize=5,
                    color=color,
                    ha="center",
                    va="bottom",
                    clip_on=True,
                )

        ax.set_title(f"{mode}; Sobel > 0.04")
        ax.set_xlabel("interior_strong_edge_frac")
        ax.set_ylabel("snip count")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Population distribution of interior strong-edge fraction by normalization mode")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "interior_strong_edge_frac_histograms_by_mode.png", dpi=170)
    plt.close(fig)


def plot_focus_vs_top_spread(metrics: pd.DataFrame) -> None:
    ok = metrics[successful_rows(metrics)].copy()
    thresholds = (0.02, 0.04, 0.06, 0.08)
    fig, axes = plt.subplots(
        len(NORMALIZATION_MODES),
        len(thresholds),
        figsize=(4.1 * len(thresholds), 3.5 * len(NORMALIZATION_MODES)),
        sharex=True,
        sharey=True,
    )
    for r, mode in enumerate(NORMALIZATION_MODES):
        for c, threshold in enumerate(thresholds):
            ax = axes[r, c]
            sub = ok[(ok["normalization_mode"] == mode) & (ok["sobel_threshold"] == threshold)]
            pop = sub[sub["source"] == "population"].dropna(subset=["top_spread_p99_p90"])
            anchor = sub[sub["source"] == "anchor"].dropna(subset=["top_spread_p99_p90"])
            ax.scatter(
                pop["interior_strong_edge_frac"],
                pop["top_spread_p99_p90"],
                s=5,
                c="#cccccc",
                alpha=0.35,
                linewidths=0,
            )
            for label, grp in anchor.groupby("user_label"):
                ax.scatter(
                    grp["interior_strong_edge_frac"],
                    grp["top_spread_p99_p90"],
                    s=45,
                    c=FOCUS_COLORS.get(label, "#444"),
                    edgecolor="black",
                    linewidth=0.4,
                )
            ax.axvline(0.25, color="#d62728", ls="--", lw=1.0, alpha=0.75)
            ax.axhline(4, color="#333333", ls=":", lw=1.0, alpha=0.65)
            ax.set_title(f"{mode}; grad>{threshold:g}", fontsize=9)
            ax.grid(alpha=0.2)
            if r == len(NORMALIZATION_MODES) - 1:
                ax.set_xlabel("interior_strong_edge_frac")
            if c == 0:
                ax.set_ylabel("top_spread_p99_p90")
    fig.suptitle("Focus edge fraction vs bright-tail spread across normalization/Sobel thresholds")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "focus_vs_top_spread_threshold_grid.png", dpi=170)
    plt.close(fig)


def plot_flag_heatmap(summary: pd.DataFrame) -> None:
    all_summary = summary[(summary["experiment_id"] == "ALL") & (summary["focus_gate"] == 0.25)].copy()
    pivot = all_summary.pivot(index="normalization_mode", columns="sobel_threshold", values="flag_frac")
    pivot = pivot.reindex(index=list(NORMALIZATION_MODES), columns=list(SOBEL_THRESHOLDS))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Sobel strong-edge threshold")
    ax.set_title("Population focus-flag fraction at edge-fraction gate 0.25")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="flag fraction")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "focus_flag_fraction_heatmap.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    metrics_path = OUT_TABLES / "focus_sobel_threshold_sweep_metrics.csv"
    if metrics_path.exists():
        print(f"Loading existing metrics: {metrics_path}", flush=True)
        metrics = pd.read_csv(metrics_path, low_memory=False)
    else:
        metrics = compute_metrics()

    summary, _ = summarize(metrics)
    plot_histograms(metrics)
    plot_focus_vs_top_spread(metrics)
    plot_flag_heatmap(summary)
    print(f"Saved fine-tuning outputs under {OUT}", flush=True)


if __name__ == "__main__":
    main()
