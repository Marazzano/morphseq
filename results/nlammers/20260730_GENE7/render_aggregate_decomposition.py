"""Standalone pooled-decomposition figures, at two FDR thresholds.

Writes to ``figures/slides/``. Nothing is refit -- these read the cached coefficient tables and
re-gate them, so the only difference between the q<0.10 and q<0.05 versions is the threshold.

Two families, decomposed exactly as they are in gene7_synthesis.ipynb:

    s vs binary      binary only / shared / morph only        (§2 head-to-head)
    slope model      baseline / indirect / direct             (§5 gain accounting)

Each figure carries two bars: all contrasts, and the subset whose morphology axis passed the §1
validity gate. Within a family the x-limits are shared across thresholds, so the q<0.10 and q<0.05
versions can be overlaid directly -- which is the whole point of making them separate files rather
than one two-panel figure.

Usage:
    conda activate morphseq-env
    python render_aggregate_decomposition.py [--outdir figures/slides]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edger_plots as ep  # noqa: E402
import gene7_config as cfg  # noqa: E402

EDGER = HERE / "data" / "edger"
THRESHOLDS = (0.10, 0.05)

BASELINE, SHARED, INDIRECT, MORPH = "0.62", "#8FA9C4", "#F3A18E", "#C0392B"


def head_to_head(coefficients, quality, *, arm, threshold):
    """Pooled binary-only / shared / morph-only, for all contrasts and validated ones."""
    overlap = ep.overlap_counts(coefficients, coefficients, arm=arm, reference="binary_z",
                                q_threshold=threshold, exclude=cfg.EXCLUDED_CONTRASTS)
    overlap = overlap.merge(quality[["contrast", "usable"]], on="contrast", how="left")
    overlap["usable"] = overlap["usable"].fillna(False).astype(bool)
    out = []
    for label, block in (("all", overlap), ("validated axis", overlap[overlap["usable"]])):
        out.append({
            "label": f"{label}  ({len(block)})",
            "segments": [("binary only", int(block["reference_only"].sum()), BASELINE),
                         ("shared", int(block["shared"].sum()), SHARED),
                         (f"{arm} only", int(block["arm_only"].sum()), MORPH)],
            "morph": int(block["arm_only"].sum()),
        })
    return out


def slope_gain(check, quality, *, threshold):
    """Pooled baseline / indirect / direct, for all contrasts and validated ones."""
    out = []
    for label, usable_only in (("all", False), ("validated axis", True)):
        figure, table = ep.slope_gain_bars(check, quality, q_threshold=threshold,
                                           usable_only=usable_only)
        plt.close(figure)
        indirect, direct = int(table["indirect"].sum()), int(table["direct"].sum())
        out.append({
            "label": f"{label}  ({len(table)})",
            "segments": [("binary baseline", int(table["baseline"].sum()), BASELINE),
                         ("indirect", indirect, INDIRECT),
                         ("direct", direct, MORPH)],
            "morph": indirect + direct,
        })
    return out


def draw(bars, *, title, threshold, xlimit, path, dpi):
    figure, axis = plt.subplots(figsize=(9.0, 2.9))
    positions = np.arange(len(bars))
    for row, bar in enumerate(bars):
        left = 0.0
        for name, width, colour in bar["segments"]:
            axis.barh(row, width, left=left, height=0.52, color=colour, edgecolor="white",
                      linewidth=1.1, zorder=3, label=name if row == 0 else None)
            if width >= 0.035 * xlimit:
                axis.text(left + width / 2, row, f"{width}", ha="center", va="center",
                          fontsize=9.5, zorder=4,
                          color="white" if colour == MORPH else "0.15")
            left += width
        share = bar["morph"] / left if left else np.nan
        axis.text(left + 0.012 * xlimit, row, f"morph {share:.1%}", va="center", fontsize=9,
                  color="0.25")

    axis.set_yticks(positions)
    axis.set_yticklabels([bar["label"] for bar in bars], fontsize=9.5)
    axis.invert_yaxis()
    axis.set_xlim(0, xlimit)
    axis.set_xlabel("cell types resolved (pooled across contrasts)", fontsize=9.5)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(fontsize=9, loc="lower right", ncol=3)
    axis.set_title(f"{title}      q < {threshold:g}", fontsize=11.5, pad=9)
    figure.tight_layout()
    target = Path(path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi)
    plt.close(figure)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(HERE / "figures" / "slides"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--arm", default="s_z")
    arguments = parser.parse_args()
    out = Path(arguments.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ep.use_house_style()
    coefficients = pd.read_csv(EDGER / "coefficients.csv")
    quality = pd.read_csv(EDGER / "contrast_axis_quality.csv")
    check = pd.read_csv(EDGER / "binary_coefficient_check.csv")
    check = check[~check["contrast"].isin(cfg.EXCLUDED_CONTRASTS)]
    quality_kept = quality[~quality["contrast"].isin(cfg.EXCLUDED_CONTRASTS)]

    families = {
        f"{arguments.arm}_vs_binary": (
            f"Aggregate detections: binary indicator versus {arguments.arm}",
            {t: head_to_head(coefficients, quality, arm=arguments.arm, threshold=t)
             for t in THRESHOLDS}),
        "slope_gain": (
            "Aggregate detections: binary baseline plus morphology slopes",
            {t: slope_gain(check, quality_kept, threshold=t) for t in THRESHOLDS}),
    }

    for name, (title, by_threshold) in families.items():
        # One x-limit per family, taken from the LOOSER gate, so the two versions overlay.
        widest = max(sum(width for _, width, _ in bar["segments"])
                     for bars in by_threshold.values() for bar in bars)
        xlimit = widest * 1.22
        for threshold, bars in by_threshold.items():
            tag = f"q{int(round(threshold * 100)):02d}"
            target = draw(bars, title=title, threshold=threshold, xlimit=xlimit,
                          path=out / f"pooled_{name}_{tag}", dpi=arguments.dpi)
            summary = "  ".join(f"{n} {w}" for n, w, _ in bars[0]["segments"])
            print(f"  {target.name:38s} all: {summary}  | morph share "
                  f"{bars[0]['morph'] / sum(w for _, w, _ in bars[0]['segments']):.1%}")

    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
