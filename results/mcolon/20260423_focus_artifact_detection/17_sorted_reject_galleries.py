"""
17_sorted_reject_galleries.py
=============================
Continuous ramp galleries: embryos laid out in a grid SORTED by a focus metric,
every tile bordered by which threshold(s) would reject it. This exposes where
the two cuts agree and disagree along the sorted axis.

Cuts:
  entropy:  ff_rel_entropy  < -0.55
  sharp:    ff_lap_abs_ratio < 1.5

Per-tile border color (ALL tiles bordered -- the per-tile fix):
  GREEN        = accepted by both
  ORANGE-BROWN = rejected by entropy only
  BLUE         = rejected by sharpness only
  PURPLE bold  = rejected by both

Two galleries: sorted by ff_rel_entropy, and sorted by ff_lap_abs_ratio.

Run after the full projected scan + merge.
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/17_sorted_reject_galleries.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
HERE = Path(__file__).resolve().parent
FULL_CSV = HERE / "outputs/projected/scan/ff_metrics_full.csv"
IMAGES_DIR = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images"
OUT_DIR = HERE / "outputs/comparison/sorted_reject"
DATE = "20250912"

ENT_CUT = -0.55
RATIO_CUT = 1.5

COLS = 12
N_TILES = 180  # sampled evenly across the full sorted range

GREEN = "#1a8820"
ORANGE = "#b5651d"   # orange-brown: entropy-only
BLUE = "#1f77b4"     # sharpness-only
PURPLE = "#7b1fa2"   # both


def img_path(well: str, t: int) -> Path:
    return IMAGES_DIR / f"{DATE}_{well}" / f"{DATE}_{well}_ch00_t{t:04d}.jpg"


def reject_style(row) -> tuple[str, float]:
    rej_ent = row["ff_rel_entropy"] < ENT_CUT
    rej_sharp = row["ff_lap_abs_ratio"] < RATIO_CUT
    if rej_ent and rej_sharp:
        return PURPLE, 4.0
    if rej_ent:
        return ORANGE, 2.5
    if rej_sharp:
        return BLUE, 2.5
    return GREEN, 2.0


def build(df: pd.DataFrame, sort_col: str, label: str, suffix: str = "") -> None:
    d = df[df[sort_col].notna()].sort_values(sort_col).reset_index(drop=True)
    if len(d) > N_TILES:
        idx = np.linspace(0, len(d) - 1, N_TILES).astype(int)
        d = d.iloc[idx].reset_index(drop=True)
    nrows = int(np.ceil(len(d) / COLS))

    fig, axes = plt.subplots(nrows, COLS, figsize=(COLS * 1.7, nrows * 1.9))
    axes = np.atleast_2d(axes)
    for i in range(nrows * COLS):
        r, c = divmod(i, COLS)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        if i >= len(d):
            ax.axis("off")
            continue
        row = d.iloc[i]
        well, t = str(row["well"]), int(row["t"])
        p = img_path(well, t)
        if p.exists():
            ax.imshow(np.array(Image.open(p).convert("L")), cmap="gray")
        else:
            ax.text(0.5, 0.5, "no img", ha="center", va="center", fontsize=7)
        color, lw = reject_style(row)
        for sp in ax.spines.values():
            sp.set_color(color); sp.set_linewidth(lw)
        ax.set_title(
            f"ff={row['ff_rel_entropy']:.2f} rat={row['ff_lap_abs_ratio']:.2f}",
            fontsize=5.5,
        )

    handles = [
        plt.Line2D([0], [0], color=GREEN, lw=3, label="accept (both pass)"),
        plt.Line2D([0], [0], color=ORANGE, lw=3, label=f"reject: entropy only (ff<{ENT_CUT})"),
        plt.Line2D([0], [0], color=BLUE, lw=3, label=f"reject: sharpness only (rat<{RATIO_CUT})"),
        plt.Line2D([0], [0], color=PURPLE, lw=4, label="reject: BOTH"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Embryos sorted by {label}{suffix}  (low->high, left-to-right, top-to-bottom)",
                 fontsize=12, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    safe = sort_col.replace("/", "_")
    tag = suffix.replace(" ", "").replace("(", "_").replace(")", "").replace("<", "lt").replace(",", "")
    out = OUT_DIR / f"sorted_by_{safe}{tag}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    # counts for the record
    n_both = int(((d.ff_rel_entropy < ENT_CUT) & (d.ff_lap_abs_ratio < RATIO_CUT)).sum())
    n_ent = int(((d.ff_rel_entropy < ENT_CUT) & (d.ff_lap_abs_ratio >= RATIO_CUT)).sum())
    n_sharp = int(((d.ff_rel_entropy >= ENT_CUT) & (d.ff_lap_abs_ratio < RATIO_CUT)).sum())
    print(f"Saved -> {out}  (shown n={len(d)}: both={n_both} ent_only={n_ent} sharp_only={n_sharp})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-max", type=int, default=None,
                    help="Only embryos with t < this (earliest-timepoint diagnostic).")
    ap.add_argument("--entropy-only", action="store_true",
                    help="Only render the ff_rel_entropy-sorted gallery.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FULL_CSV)
    df = df[df["has_mask"].astype(bool)] if "has_mask" in df else df

    suffix = ""
    if args.t_max is not None:
        df = df[df["t"] < args.t_max]
        suffix = f"  (t < {args.t_max})"

    # full-dataset disagreement tally (not just the sampled tiles)
    re = df.ff_rel_entropy < ENT_CUT
    rs = df.ff_lap_abs_ratio < RATIO_CUT
    print(f"SUBSET n={len(df)}{suffix}: reject_both={int((re&rs).sum())} "
          f"entropy_only={int((re&~rs).sum())} sharp_only={int((~re&rs).sum())} "
          f"accept={int((~re&~rs).sum())}")
    build(df, "ff_rel_entropy", "projected ff_rel_entropy", suffix)
    if not args.entropy_only:
        build(df, "ff_lap_abs_ratio", "sharpness ratio (emb/bg)", suffix)


if __name__ == "__main__":
    main()
