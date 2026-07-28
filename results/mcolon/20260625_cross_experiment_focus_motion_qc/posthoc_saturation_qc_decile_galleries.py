#!/usr/bin/env python
"""Alive-only saturation QC decile galleries across experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "saturation_qc"
FIGURES.mkdir(parents=True, exist_ok=True)

FOCUS_CSV = TABLES / "focus_metrics_by_experiment.csv"
SAT_CSV = TABLES / "focus_saturation_metrics_not_dead_snips.csv"

ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5
N_DECILES = 10
N_PER_EXPERIMENT_DECILE = 6

STATUS_COLORS = {
    "accepted": "#1a8820",
    "entropy_only_reject": "#b5651d",
    "sharpness_only_reject": "#1f77b4",
    "rejected_by_both": "#7b1fa2",
}

METRICS = [
    ("emb_frac_gt245", "fraction embryo pixels >=245"),
    ("emb_frac_gt250", "fraction embryo pixels >=250"),
    ("emb_p99", "embryo p99 intensity"),
]


def focus_status(df: pd.DataFrame) -> pd.Series:
    ent = df["ff_rel_entropy"] < ENTROPY_CUT
    lap = df["ff_lap_abs_ratio"] < LAP_RATIO_CUT
    out = pd.Series("accepted", index=df.index, dtype=object)
    out.loc[ent & ~lap] = "entropy_only_reject"
    out.loc[~ent & lap] = "sharpness_only_reject"
    out.loc[ent & lap] = "rejected_by_both"
    return out


def load_focus_not_dead(sample_per_experiment: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(FOCUS_CSV, low_memory=False)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["image_path", "mask_path", "ff_rel_entropy", "ff_lap_abs_ratio"]).copy()
    df["current_focus_status"] = focus_status(df)
    if sample_per_experiment is not None:
        pieces = []
        for _, grp in df.groupby("experiment_id", observed=True):
            n = min(int(sample_per_experiment), len(grp))
            pieces.append(grp.sample(n=n, random_state=1729))
        df = pd.concat(pieces, ignore_index=True)
    return df


def read_masked_pixels(image_path: object, mask_path: object) -> tuple[np.ndarray, np.ndarray]:
    image = np.array(Image.open(str(image_path)).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(str(mask_path))).astype(bool)
    if mask.shape != image.shape or not mask.any():
        raise ValueError("bad or empty mask")
    return image[mask], image


def compute_saturation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rec = row.to_dict()
        try:
            emb, _ = read_masked_pixels(row["image_path"], row["mask_path"])
            rec.update(
                {
                    "sat_metric_error": "",
                    "emb_n_px": int(emb.size),
                    "emb_p50": float(np.percentile(emb, 50)),
                    "emb_p90": float(np.percentile(emb, 90)),
                    "emb_p95": float(np.percentile(emb, 95)),
                    "emb_p99": float(np.percentile(emb, 99)),
                    "emb_max": float(np.max(emb)),
                    "emb_frac_gt240": float(np.mean(emb >= 240)),
                    "emb_frac_gt245": float(np.mean(emb >= 245)),
                    "emb_frac_gt250": float(np.mean(emb >= 250)),
                    "emb_frac_eq255": float(np.mean(emb == 255)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rec.update(
                {
                    "sat_metric_error": str(exc),
                    "emb_n_px": 0,
                    "emb_p50": np.nan,
                    "emb_p90": np.nan,
                    "emb_p95": np.nan,
                    "emb_p99": np.nan,
                    "emb_max": np.nan,
                    "emb_frac_gt240": np.nan,
                    "emb_frac_gt245": np.nan,
                    "emb_frac_gt250": np.nan,
                    "emb_frac_eq255": np.nan,
                }
            )
        rows.append(rec)
        if i % 5000 == 0 or i == total:
            print(f"[saturation] processed {i}/{total}", flush=True)
    return pd.DataFrame(rows)


def saturation_csv_for_sample(sample_per_experiment: int | None) -> Path:
    if sample_per_experiment is None:
        return SAT_CSV
    return TABLES / f"focus_saturation_metrics_not_dead_snips_sample{sample_per_experiment}_per_experiment.csv"


def output_suffix(sample_per_experiment: int | None) -> str:
    if sample_per_experiment is None:
        return "not_dead"
    return f"not_dead_sample{sample_per_experiment}_per_experiment"


def load_or_compute_saturation(recompute: bool = False, sample_per_experiment: int | None = None) -> pd.DataFrame:
    sat_csv = saturation_csv_for_sample(sample_per_experiment)
    if sat_csv.exists() and not recompute:
        print(f"Loading existing saturation metrics -> {sat_csv}", flush=True)
        return pd.read_csv(sat_csv, low_memory=False)
    focus = load_focus_not_dead(sample_per_experiment=sample_per_experiment)
    print(f"Computing saturation metrics for {len(focus)} not-dead snips", flush=True)
    sat = compute_saturation_metrics(focus)
    sat.to_csv(sat_csv, index=False)
    print(f"Saved saturation metrics -> {sat_csv}", flush=True)
    return sat


def assign_deciles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for exp, grp in df.dropna(subset=[metric]).groupby("experiment_id", observed=True):
        ordered = grp.sort_values(metric, ascending=True).copy()
        ordered["saturation_decile"] = (
            pd.qcut(
                ordered[metric].rank(method="first"),
                N_DECILES,
                labels=False,
                duplicates="drop",
            ).astype(int)
            + 1
        )
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True)


def sample_deciles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    with_deciles = assign_deciles(df, metric)
    rows = []
    for exp in sorted(with_deciles["experiment_id"].astype(str).unique()):
        exp_df = with_deciles[with_deciles["experiment_id"].astype(str) == exp]
        for decile in sorted(exp_df["saturation_decile"].dropna().astype(int).unique()):
            grp = exp_df[exp_df["saturation_decile"].astype(int) == decile].sort_values(metric)
            if len(grp) <= N_PER_EXPERIMENT_DECILE:
                picks = grp
            else:
                picks = grp.iloc[np.linspace(0, len(grp) - 1, N_PER_EXPERIMENT_DECILE).astype(int)]
            picks = picks.copy()
            picks["sample_rank_within_decile"] = np.arange(1, len(picks) + 1)
            rows.append(picks)
    manifest = pd.concat(rows, ignore_index=True)
    keep = [
        "experiment_id",
        "saturation_decile",
        "sample_rank_within_decile",
        "well_id",
        "well",
        "time_index",
        "frame_index",
        "snip_id",
        "image_id",
        "image_path",
        "mask_path",
        "predicted_stage_hpf",
        metric,
        "emb_frac_gt245",
        "emb_frac_gt250",
        "emb_p99",
        "ff_mean_emb",
        "ff_iqr_emb",
        "ff_rel_entropy",
        "ff_lap_abs_ratio",
        "current_focus_status",
    ]
    keep = list(dict.fromkeys(c for c in keep if c in manifest.columns))
    return manifest[keep]


def crop_image(image_path: object, mask_path: object, pad: int = 60) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def plot_gallery(manifest: pd.DataFrame, metric: str, metric_label: str, out_path: Path) -> None:
    experiments = sorted(manifest["experiment_id"].astype(str).unique())
    deciles = sorted(manifest["saturation_decile"].dropna().astype(int).unique())
    block_rows = 2
    block_cols = 3
    nrows = len(deciles) * block_rows
    ncols = len(experiments) * block_cols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.65, nrows * 1.65), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for d_i, decile in enumerate(deciles):
        for e_i, exp in enumerate(experiments):
            grp = manifest[
                (manifest["experiment_id"].astype(str) == exp)
                & (manifest["saturation_decile"].astype(int) == decile)
            ].sort_values("sample_rank_within_decile")
            for k, (_, row) in enumerate(grp.head(block_rows * block_cols).iterrows()):
                rr = d_i * block_rows + k // block_cols
                cc = e_i * block_cols + k % block_cols
                ax = axes[rr, cc]
                crop, mask = crop_image(row["image_path"], row["mask_path"])
                if crop is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
                else:
                    ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
                    if mask is not None and mask.any():
                        ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.45)
                ax.axis("on")
                status = str(row.get("current_focus_status", ""))
                for spine in ax.spines.values():
                    spine.set_color(STATUS_COLORS.get(status, "#444444"))
                    spine.set_linewidth(2.0)
                ax.set_title(
                    f"{exp} {row.get('well_id', row.get('well', ''))} t{int(row['time_index'])}\n"
                    f"{metric}={row[metric]:.3g} p99={row['emb_p99']:.0f}\n"
                    f"f245={row['emb_frac_gt245']:.2f} f250={row['emb_frac_gt250']:.2f}",
                    fontsize=5,
                )
            if d_i == 0:
                axes[0, e_i * block_cols + 1].set_title(exp, fontsize=9)
        axes[d_i * block_rows, 0].set_ylabel(f"D{decile}", fontsize=8)

    handles = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="none", label=k)
        for k in ["accepted", "entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=4, frameon=False)
    fig.suptitle(
        f"Not-dead saturation QC deciles by experiment: {metric_label} (D1 low -> D10 high)",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_decile_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    with_deciles = assign_deciles(df, metric)
    rows = []
    for (exp, decile), grp in with_deciles.groupby(["experiment_id", "saturation_decile"], observed=True):
        counts = grp["current_focus_status"].value_counts()
        rows.append(
            {
                "experiment_id": exp,
                "metric": metric,
                "saturation_decile": int(decile),
                "n": int(len(grp)),
                "metric_min": float(grp[metric].min()),
                "metric_median": float(grp[metric].median()),
                "metric_max": float(grp[metric].max()),
                "emb_frac_gt245_median": float(grp["emb_frac_gt245"].median()),
                "emb_frac_gt250_median": float(grp["emb_frac_gt250"].median()),
                "emb_p99_median": float(grp["emb_p99"].median()),
                "ff_mean_emb_median": float(grp["ff_mean_emb"].median()),
                "ff_iqr_emb_median": float(grp["ff_iqr_emb"].median()),
                "ff_rel_entropy_median": float(grp["ff_rel_entropy"].median()),
                "accepted_frac": float(counts.get("accepted", 0) / len(grp)),
                "entropy_only_reject_frac": float(counts.get("entropy_only_reject", 0) / len(grp)),
                "sharpness_only_reject_frac": float(counts.get("sharpness_only_reject", 0) / len(grp)),
                "rejected_by_both_frac": float(counts.get("rejected_by_both", 0) / len(grp)),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(METRICS), 3, figsize=(15, 10), squeeze=False)
    experiments = sorted(df["experiment_id"].astype(str).unique())
    for r, (metric, label) in enumerate(METRICS):
        for c, exp in enumerate(experiments):
            ax = axes[r, c]
            sub = df[df["experiment_id"].astype(str) == exp]
            vals = sub[metric].dropna()
            if vals.empty:
                continue
            if metric.startswith("emb_frac"):
                bins = np.linspace(0, min(1.0, max(0.25, float(vals.quantile(0.995)))), 60)
            else:
                bins = np.linspace(0, 255, 60)
            for status, grp in sub.groupby("current_focus_status", observed=True):
                gvals = grp[metric].dropna()
                if gvals.empty:
                    continue
                ax.hist(
                    gvals,
                    bins=bins,
                    histtype="step",
                    density=True,
                    linewidth=1.7,
                    color=STATUS_COLORS.get(status, "#555555"),
                    label=status,
                )
            ax.set_title(f"{exp}: {label}")
            ax.set_xlabel(metric)
            ax.set_ylabel("density")
            ax.grid(alpha=0.25)
    handles = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="none", label=k)
        for k in ["accepted", "entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=4, frameon=False)
    fig.suptitle("Not-dead saturation metric histograms by current focus status", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = FIGURES / "saturation_metric_histograms_by_experiment_status.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved saturation histograms -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--sample-per-experiment", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sat = load_or_compute_saturation(
        recompute=args.recompute,
        sample_per_experiment=args.sample_per_experiment,
    )
    sat = sat[sat["sat_metric_error"].fillna("") == ""].copy()
    suffix = output_suffix(args.sample_per_experiment)
    summary_rows = []
    for metric, label in METRICS:
        manifest = sample_deciles(sat, metric)
        manifest_path = TABLES / f"focus_saturation_decile_samples_{metric}_{suffix}.csv"
        figure_path = FIGURES / f"focus_saturation_deciles_{metric}_{suffix}.png"
        manifest.to_csv(manifest_path, index=False)
        plot_gallery(manifest, metric, label, figure_path)
        summary_rows.append(write_decile_summary(sat, metric))
        print(f"Saved {metric} manifest -> {manifest_path}")
        print(f"Saved {metric} gallery -> {figure_path}")

    summary = pd.concat(summary_rows, ignore_index=True)
    summary_path = TABLES / f"focus_saturation_decile_summary_{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved saturation decile summary -> {summary_path}")
    plot_metric_histograms(sat)


if __name__ == "__main__":
    main()
