#!/usr/bin/env python
"""Review current entropy failures stratified by normalized LoG ratio."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "focus_candidate_metrics"
FIGURES.mkdir(parents=True, exist_ok=True)

BENCHMARK_CSV = TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv"
ANCHOR_CSV = TABLES / "focus_candidate_metrics_20251125.csv"
SUMMARY_OUT = TABLES / "focus_current_fail_log_ratio_rescue_review.csv"
SCATTER_OUT = FIGURES / "20251125_current_fail_log_ratio_rescue_scatter.png"
GALLERY_OUT = FIGURES / "20251125_current_fail_log_ratio_rescue_gallery.png"
HIGHRES_DIR = FIGURES / "log_rescue_highres_crops"

ENTROPY_CUT = -0.55
LOG_LOW_Q = 0.20
LOG_HIGH_Q = 0.80
N_PER_GROUP = 12


def read_inputs() -> pd.DataFrame:
    parts = [pd.read_csv(BENCHMARK_CSV)]
    if ANCHOR_CSV.exists():
        anchors = pd.read_csv(ANCHOR_CSV)
        anchors = anchors[anchors["cohort"].eq("anchor")].copy()
        parts.append(anchors)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["well_id", "time_index"], keep="last")
    df = df.dropna(subset=["ff_rel_entropy", "log_ratio_norm", "norm_rel_entropy_local_annulus"]).copy()
    df["current_global_entropy_fail"] = df["ff_rel_entropy"] < ENTROPY_CUT
    df["anchor_label"] = ""
    for well, t in [("A06", 40), ("A06", 76), ("F01", 166)]:
        m = df["well_id"].astype(str).str.upper().str.contains(well) & (pd.to_numeric(df["time_index"], errors="coerce") == t)
        df.loc[m, "anchor_label"] = f"{well} t{t}"
    return df


def assign_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    fail = df[df["current_global_entropy_fail"]].copy()
    low_cut = float(fail["log_ratio_norm"].quantile(LOG_LOW_Q))
    high_cut = float(fail["log_ratio_norm"].quantile(LOG_HIGH_Q))

    df["log_rescue_group"] = "not_current_fail"
    df.loc[df["current_global_entropy_fail"] & (df["log_ratio_norm"] <= low_cut), "log_rescue_group"] = "current_fail_low_log"
    df.loc[df["current_global_entropy_fail"] & (df["log_ratio_norm"] >= high_cut), "log_rescue_group"] = "current_fail_high_log"
    mid = df["current_global_entropy_fail"] & df["log_rescue_group"].eq("not_current_fail")
    df.loc[mid, "log_rescue_group"] = "current_fail_mid_log"
    return df, low_cut, high_cut


def write_summary(df: pd.DataFrame, low_cut: float, high_cut: float) -> None:
    rows = []
    for group, grp in df.groupby("log_rescue_group", observed=True):
        rows.append(
            {
                "log_rescue_group": group,
                "n": int(len(grp)),
                "log_low_cut_p20": low_cut,
                "log_high_cut_p80": high_cut,
                "ff_rel_entropy_median": float(grp["ff_rel_entropy"].median()),
                "norm_local_entropy_median": float(grp["norm_rel_entropy_local_annulus"].median()),
                "log_ratio_norm_median": float(grp["log_ratio_norm"].median()),
                "sobel_ratio_norm_median": float(grp["sobel_ratio_norm"].median()),
                "emb_frac_gt245_median": float(grp["emb_frac_gt245"].median()),
                "emb_iqr_local_median": float(grp["emb_iqr_local"].median()),
            }
        )
    pd.DataFrame(rows).sort_values("log_rescue_group").to_csv(SUMMARY_OUT, index=False)


def plot_scatter(df: pd.DataFrame, low_cut: float, high_cut: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "not_current_fail": "#1a8820",
        "current_fail_low_log": "#7b1fa2",
        "current_fail_mid_log": "#b5651d",
        "current_fail_high_log": "#2ca25f",
    }
    labels = {
        "not_current_fail": "current pass",
        "current_fail_low_log": "current fail, low LoG",
        "current_fail_mid_log": "current fail, mid LoG",
        "current_fail_high_log": "current fail, high LoG",
    }
    for group, grp in df.groupby("log_rescue_group", observed=True):
        ax.scatter(
            grp["ff_rel_entropy"],
            grp["log_ratio_norm"],
            s=15,
            alpha=0.45,
            color=colors.get(group, "#555555"),
            label=labels.get(group, group),
            linewidths=0,
        )
    anchors = df[df["anchor_label"].ne("")]
    ax.scatter(anchors["ff_rel_entropy"], anchors["log_ratio_norm"], s=90, color="black", alpha=0.75)
    for _, row in anchors.iterrows():
        ax.text(row["ff_rel_entropy"], row["log_ratio_norm"], row["anchor_label"], fontsize=8)
    ax.axvline(ENTROPY_CUT, color="black", linestyle="--", linewidth=1)
    ax.axhline(low_cut, color="#7b1fa2", linestyle=":", linewidth=1)
    ax.axhline(high_cut, color="#2ca25f", linestyle=":", linewidth=1)
    ax.set_xlabel("current global relative entropy")
    ax.set_ylabel("normalized LoG ratio")
    ax.set_title("20251125: current global entropy failures split by normalized LoG ratio")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SCATTER_OUT, dpi=180)
    plt.close(fig)


def sample_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    grp = df[df["log_rescue_group"] == group].copy()
    anchors = grp[grp["anchor_label"].ne("")].copy()
    if group == "current_fail_low_log":
        ordered = grp.sort_values("log_ratio_norm", ascending=True)
    elif group == "current_fail_high_log":
        ordered = grp.sort_values("log_ratio_norm", ascending=False)
    elif group == "current_fail_mid_log":
        ordered = grp.sort_values("ff_rel_entropy", ascending=False)
    elif group == "not_current_fail":
        ordered = grp.sort_values("ff_rel_entropy", ascending=False)
    else:
        ordered = grp
    out = pd.concat([anchors, ordered], ignore_index=True).drop_duplicates(["well_id", "time_index"])
    return out.head(N_PER_GROUP)


def read_image(path: object) -> np.ndarray | None:
    p = Path(str(path))
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("L"))


def read_mask(path: object, shape: tuple[int, int]) -> np.ndarray | None:
    p = Path(str(path))
    if not p.exists():
        return None
    mask = np.array(Image.open(p)).astype(bool)
    if mask.shape != shape:
        return None
    return mask


def crop_bounds(mask: np.ndarray, pad: int = 70) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def plot_gallery(df: pd.DataFrame) -> None:
    groups = [
        ("current_fail_high_log", "current fail / high LoG"),
        ("current_fail_mid_log", "current fail / mid LoG"),
        ("current_fail_low_log", "current fail / low LoG"),
    ]
    samples = [(label, sample_group(df, group)) for group, label in groups]
    nrows = len(samples)
    ncols = N_PER_GROUP
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.7, nrows * 2.0), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    group_colors = {
        "current fail / high LoG": "#2ca25f",
        "current fail / mid LoG": "#b5651d",
        "current fail / low LoG": "#7b1fa2",
    }
    for r, (label, grp) in enumerate(samples):
        for c, (_, row) in enumerate(grp.iterrows()):
            ax = axes[r, c]
            img = read_image(row["image_path"])
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            else:
                ax.imshow(img, cmap="gray")
            ax.axis("on")
            color = group_colors[label]
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.2)
            anchor = row.get("anchor_label", "")
            title0 = anchor if isinstance(anchor, str) and anchor else f"{row.get('well_id', '')} t{row.get('time_index', '')}"
            ax.set_title(
                f"{title0}\n"
                f"LoG={row['log_ratio_norm']:.2f} glob={row['ff_rel_entropy']:.2f}\n"
                f"loc={row['norm_rel_entropy_local_annulus']:.2f} sat={row['emb_frac_gt245']:.2f}",
                fontsize=5.4,
            )
        axes[r, 0].set_ylabel(label, fontsize=8)

    handles = [Patch(facecolor=color, label=label) for label, color in group_colors.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.suptitle("Current global entropy failures: does high normalized LoG identify rescue candidates?", y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(GALLERY_OUT, dpi=170)
    plt.close(fig)


def plot_highres_crop_group(df: pd.DataFrame, group: str, label: str, out_path: Path) -> None:
    grp = sample_group(df, group)
    if grp.empty:
        return
    ncols = 4
    nrows = int(np.ceil(len(grp) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 4.4), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for i, (_, row) in enumerate(grp.iterrows()):
        ax = axes[i // ncols, i % ncols]
        img = read_image(row["image_path"])
        if img is None:
            ax.text(0.5, 0.5, "missing image", ha="center", va="center", fontsize=8)
            continue
        mask = read_mask(row["mask_path"], img.shape)
        if mask is not None and mask.any():
            y0, y1, x0, x1 = crop_bounds(mask)
            crop = img[y0:y1, x0:x1]
            mask_crop = mask[y0:y1, x0:x1]
            ax.imshow(crop, cmap="gray")
            ax.contour(mask_crop, levels=[0.5], colors=["lime"], linewidths=1.2)
        else:
            ax.imshow(img, cmap="gray")
        ax.axis("on")
        for spine in ax.spines.values():
            spine.set_color("#2ca25f" if group == "current_fail_high_log" else "#7b1fa2" if group == "current_fail_low_log" else "#b5651d")
            spine.set_linewidth(2.0)
        anchor = row.get("anchor_label", "")
        title0 = anchor if isinstance(anchor, str) and anchor else f"{row.get('well_id', '')} t{row.get('time_index', '')}"
        ax.set_title(
            f"{title0}\n"
            f"LoG={row['log_ratio_norm']:.2f} Sobel={row.get('sobel_ratio_norm', np.nan):.2f}\n"
            f"global={row['ff_rel_entropy']:.2f} local={row['norm_rel_entropy_local_annulus']:.2f} sat={row['emb_frac_gt245']:.2f}",
            fontsize=8,
        )

    fig.suptitle(label, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_highres_crops(df: pd.DataFrame) -> list[Path]:
    HIGHRES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    specs = [
        ("current_fail_high_log", "Current global entropy fail, HIGH normalized LoG ratio: possible rescue/pass-like cases"),
        ("current_fail_mid_log", "Current global entropy fail, MID normalized LoG ratio"),
        ("current_fail_low_log", "Current global entropy fail, LOW normalized LoG ratio: stronger fail-like cases"),
    ]
    for group, label in specs:
        out = HIGHRES_DIR / f"20251125_{group}_embryo_crops.png"
        plot_highres_crop_group(df, group, label, out)
        outputs.append(out)
    return outputs


def main() -> None:
    df = read_inputs()
    df, low_cut, high_cut = assign_groups(df)
    write_summary(df, low_cut, high_cut)
    plot_scatter(df, low_cut, high_cut)
    plot_gallery(df)
    highres_outputs = plot_highres_crops(df)
    print(f"LoG fail split cuts among current failures: p20={low_cut:.3f}, p80={high_cut:.3f}")
    print(f"Saved summary -> {SUMMARY_OUT}")
    print(f"Saved scatter -> {SCATTER_OUT}")
    print(f"Saved gallery -> {GALLERY_OUT}")
    for out in highres_outputs:
        print(f"Saved high-res crop gallery -> {out}")


if __name__ == "__main__":
    main()
