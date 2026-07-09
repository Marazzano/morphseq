"""
13_threshold_band_eval.py
=========================
Targeted threshold evaluation in the decision band.

The global z-stack vs projected correlation is r=0.989, but that hides an
asymmetry at the decision boundary: for a given embryo the z-stack
rel_entropy_mean sits MORE NEGATIVE than the projected ff_rel_entropy, i.e. the
z-stack metric is the more STRINGENT of the two. The candidate threshold lives
around rel_entropy_mean in [-0.55, -0.35].

This script zooms into that band and lays both metrics side by side, GROUPED BY
the stringent axis (z-stack rel_entropy_mean) so we can read how the projected
metric behaves within each z-stack slice — does crossing a z-stack cut also move
you across a sensible ff cut?

Outputs -> outputs/comparison/threshold_band/
  - band_grouped_box.png        ff_rel_entropy distribution per z-stack bin
  - band_paired_scatter.png     ff vs z in-band, y=x + offset reference
  - band_grouped_summary.csv    per-bin n / medians / delta
  - band_rows.csv               raw in-band joined rows

Run AFTER 12_compare_distributions.py.
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/13_threshold_band_eval.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOINED_CSV = HERE / "outputs/comparison/joined_zstack_projected.csv"
OUT_DIR = HERE / "outputs/comparison/threshold_band"

# Decision band on the stringent (z-stack) axis.
BAND_LO = -0.55
BAND_HI = -0.35
BIN_WIDTH = 0.05  # z-stack slices: -0.55..-0.50, ... -0.40..-0.35


def load_band() -> pd.DataFrame:
    df = pd.read_csv(JOINED_CSV)
    band = df[df["rel_entropy_mean"].between(BAND_LO, BAND_HI, inclusive="both")].copy()
    edges = np.round(np.arange(BAND_LO, BAND_HI + 1e-9, BIN_WIDTH), 3)
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges) - 1)]
    band["z_bin"] = pd.cut(band["rel_entropy_mean"], bins=edges, labels=labels,
                           right=False, include_lowest=True)
    band["delta_ff_minus_z"] = band["ff_rel_entropy"] - band["rel_entropy_mean"]
    return band, labels


def grouped_box(band: pd.DataFrame, labels: list[str]) -> None:
    """ff_rel_entropy distribution within each z-stack (stringent) bin."""
    fig, ax = plt.subplots(figsize=(9, 6))
    data, used = [], []
    for lab in labels:
        vals = band.loc[band["z_bin"] == lab, "ff_rel_entropy"].dropna().values
        if len(vals):
            data.append(vals)
            used.append(f"{lab}\nn={len(vals)}")
    ax.boxplot(data, vert=True, showmeans=True, widths=0.6)
    ax.set_xticklabels(used, rotation=0, fontsize=9)
    ax.set_xlabel("z-stack rel_entropy_mean bin (stringent axis)")
    ax.set_ylabel("projected  ff_rel_entropy")
    ax.axhline(BAND_LO, color="#999", ls=":", lw=1)
    ax.axhline(BAND_HI, color="#999", ls=":", lw=1)
    ax.set_title("Projected metric grouped by stringent (z-stack) bin\n"
                 f"decision band z in [{BAND_LO}, {BAND_HI}]")
    fig.tight_layout()
    out = OUT_DIR / "band_grouped_box.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def paired_scatter(band: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sc = ax.scatter(band["rel_entropy_mean"], band["ff_rel_entropy"],
                    c=band["ff_lap_abs_ratio"], s=14, alpha=0.6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="ff_lap_abs_ratio (sharpness)")
    lo = min(band["rel_entropy_mean"].min(), band["ff_rel_entropy"].min())
    hi = max(band["rel_entropy_mean"].max(), band["ff_rel_entropy"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="#d62728", lw=1, label="y = x")
    med_delta = band["delta_ff_minus_z"].median()
    ax.plot([lo, hi], [lo + med_delta, hi + med_delta], ":", color="#1f77b4", lw=1,
            label=f"y = x + {med_delta:.2f} (median offset)")
    ax.set_xlabel("z-stack rel_entropy_mean (stringent)")
    ax.set_ylabel("projected ff_rel_entropy")
    ax.set_title(f"In-band paired comparison (n={len(band)})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "band_paired_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def summarize(band: pd.DataFrame, labels: list[str]) -> None:
    rows = []
    for lab in labels:
        sub = band[band["z_bin"] == lab]
        if not len(sub):
            continue
        rows.append({
            "z_bin": lab,
            "n": len(sub),
            "z_median": round(sub["rel_entropy_mean"].median(), 4),
            "ff_median": round(sub["ff_rel_entropy"].median(), 4),
            "delta_median": round(sub["delta_ff_minus_z"].median(), 4),
            "ff_q25": round(sub["ff_rel_entropy"].quantile(0.25), 4),
            "ff_q75": round(sub["ff_rel_entropy"].quantile(0.75), 4),
            "frac_ff_gt_z": round((sub["ff_rel_entropy"] > sub["rel_entropy_mean"]).mean(), 4),
        })
    summ = pd.DataFrame(rows)
    out = OUT_DIR / "band_grouped_summary.csv"
    summ.to_csv(out, index=False)
    print(f"Saved -> {out}")
    print(summ.to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    band, labels = load_band()
    band.to_csv(OUT_DIR / "band_rows.csv", index=False)
    print(f"In-band rows (z in [{BAND_LO}, {BAND_HI}]): {len(band)}")
    grouped_box(band, labels)
    paired_scatter(band)
    summarize(band, labels)


if __name__ == "__main__":
    main()
