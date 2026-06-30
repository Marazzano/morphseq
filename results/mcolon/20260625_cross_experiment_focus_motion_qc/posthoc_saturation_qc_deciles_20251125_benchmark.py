#!/usr/bin/env python
"""Fast saturation decile galleries from the existing 20251125 candidate table."""

from __future__ import annotations

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

INPUT = TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv"
ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5
N_DECILES = 10
N_PER_DECILE = 12

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


def status(df: pd.DataFrame) -> pd.Series:
    ent = df["ff_rel_entropy"] < ENTROPY_CUT
    lap = df["ff_lap_abs_ratio"] < LAP_RATIO_CUT
    out = pd.Series("accepted", index=df.index, dtype=object)
    out.loc[ent & ~lap] = "entropy_only_reject"
    out.loc[~ent & lap] = "sharpness_only_reject"
    out.loc[ent & lap] = "rejected_by_both"
    return out


def load_df() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(
        subset=[
            "image_path",
            "mask_path",
            "ff_rel_entropy",
            "ff_lap_abs_ratio",
            "emb_frac_gt245",
            "emb_frac_gt250",
            "emb_p99",
        ]
    ).copy()
    df["current_focus_status"] = status(df)
    return df


def crop(image_path: object, mask_path: object, pad: int = 60) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def sample_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.sort_values(metric, ascending=True).copy()
    out["saturation_decile"] = (
        pd.qcut(out[metric].rank(method="first"), N_DECILES, labels=False, duplicates="drop").astype(int) + 1
    )
    rows = []
    for decile, grp in out.groupby("saturation_decile", observed=True):
        grp = grp.sort_values(metric)
        if len(grp) <= N_PER_DECILE:
            picks = grp
        else:
            picks = grp.iloc[np.linspace(0, len(grp) - 1, N_PER_DECILE).astype(int)]
        picks = picks.copy()
        picks["sample_rank_within_decile"] = np.arange(1, len(picks) + 1)
        rows.append(picks)
    return pd.concat(rows, ignore_index=True)


def plot_gallery(manifest: pd.DataFrame, metric: str, label: str) -> None:
    ncols = N_PER_DECILE
    fig, axes = plt.subplots(N_DECILES, ncols, figsize=(ncols * 1.8, N_DECILES * 1.85), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for r, decile in enumerate(sorted(manifest["saturation_decile"].unique())):
        grp = manifest[manifest["saturation_decile"] == decile].sort_values("sample_rank_within_decile")
        for c, (_, row) in enumerate(grp.head(ncols).iterrows()):
            ax = axes[r, c]
            img, mask = crop(row["image_path"], row["mask_path"])
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                if mask is not None and mask.any():
                    ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.45)
            ax.axis("on")
            color = STATUS_COLORS.get(str(row["current_focus_status"]), "#444444")
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.0)
            ax.set_title(
                f"{row['well_id']} t{int(row['time_index'])}\n"
                f"{metric}={row[metric]:.3g}\n"
                f"f245={row['emb_frac_gt245']:.2f} p99={row['emb_p99']:.0f}",
                fontsize=5,
            )
        axes[r, 0].set_ylabel(f"D{decile}", fontsize=8)

    handles = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="none", label=k)
        for k in ["accepted", "entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=4, frameon=False)
    fig.suptitle(f"20251125 not-dead benchmark saturation deciles: {label} (D1 low -> D10 high)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / f"20251125_benchmark_saturation_deciles_{metric}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {metric} gallery -> {out}")


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.sort_values(metric).copy()
    out["saturation_decile"] = (
        pd.qcut(out[metric].rank(method="first"), N_DECILES, labels=False, duplicates="drop").astype(int) + 1
    )
    rows = []
    for decile, grp in out.groupby("saturation_decile", observed=True):
        counts = grp["current_focus_status"].value_counts()
        rows.append(
            {
                "metric": metric,
                "saturation_decile": int(decile),
                "n": int(len(grp)),
                "metric_min": float(grp[metric].min()),
                "metric_median": float(grp[metric].median()),
                "metric_max": float(grp[metric].max()),
                "emb_frac_gt245_median": float(grp["emb_frac_gt245"].median()),
                "emb_frac_gt250_median": float(grp["emb_frac_gt250"].median()),
                "emb_p99_median": float(grp["emb_p99"].median()),
                "ff_rel_entropy_median": float(grp["ff_rel_entropy"].median()),
                "accepted_frac": float(counts.get("accepted", 0) / len(grp)),
                "entropy_only_reject_frac": float(counts.get("entropy_only_reject", 0) / len(grp)),
                "rejected_by_both_frac": float(counts.get("rejected_by_both", 0) / len(grp)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_df()
    summaries = []
    for metric, label in METRICS:
        manifest = sample_metric(df, metric)
        manifest_path = TABLES / f"20251125_benchmark_saturation_decile_samples_{metric}.csv"
        manifest.to_csv(manifest_path, index=False)
        plot_gallery(manifest, metric, label)
        summaries.append(summarize(df, metric))
        print(f"Saved {metric} manifest -> {manifest_path}")
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = TABLES / "20251125_benchmark_saturation_decile_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved saturation summary -> {summary_path}")


if __name__ == "__main__":
    main()
