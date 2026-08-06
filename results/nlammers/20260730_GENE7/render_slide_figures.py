"""Slide-ready PNGs, rendered standalone so they do not depend on running a notebook.

Writes to ``figures/slides/``. Nothing here recomputes a model — it reads the cached coefficient
tables and re-renders, so it is fast and always consistent with whatever the fitting scripts last
produced.

The bar figures are the three-colour decomposition of each arm's hit set against the binary
indicator: **binary-only / shared / arm-only**, per contrast plus pooled. All three arms render on
identical axes and colours so they can sit side by side on a slide:

    s                supervised LDA distance         data/edger/coefficients.csv
    pooled PC1       unsupervised, no labels         data/unsupervised/coefficients.csv
    within PC1       unsupervised, crispants only    data/unsupervised/coefficients.csv

EXCLUDED_CONTRASTS is applied to every panel. See the note in gene7_synthesis.ipynb §0: atf6 |
34C | 30hpf is driven by one crispant at s_z = 4.37 (next highest 0.61) and one control at 0.99
(next -0.06), and is dropped on the shape of the morphology distribution alone.

Usage:
    conda activate morphseq-env
    python render_slide_figures.py [--outdir figures/slides]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import edger_plots as ep  # noqa: E402

EDGER = HERE / "data" / "edger"
UNSUPERVISED = HERE / "data" / "unsupervised"

EXCLUDED_CONTRASTS = ["atf6 vs ctrl | 34C | 30hpf"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(HERE / "figures" / "slides"))
    parser.add_argument("--dpi", type=int, default=220)
    arguments = parser.parse_args()
    out = Path(arguments.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ep.use_house_style()
    supervised = pd.read_csv(EDGER / "coefficients.csv")
    quality = pd.read_csv(EDGER / "contrast_axis_quality.csv")
    unsupervised_path = UNSUPERVISED / "coefficients.csv"
    unsupervised = pd.read_csv(unsupervised_path) if unsupervised_path.exists() else None

    jobs = [("s_z", supervised, "$s$  (supervised LDA distance)", "bars_binary_vs_s")]
    if unsupervised is not None:
        jobs += [
            ("pooled_pc1", unsupervised, "pooled PC1  (unsupervised, no labels)",
             "bars_binary_vs_pooled_pc1"),
            ("within_pc1", unsupervised, "within PC1  (unsupervised, crispants only)",
             "bars_binary_vs_within_pc1"),
        ]
    else:
        print(f"  [skip] {unsupervised_path} not found — unsupervised arms not rendered")

    for arm, frame, label, name in jobs:
        figure, table = ep.hit_overlap_bars(
            frame, quality, arm=arm, reference_coefficients=supervised, arm_label=label,
            exclude=EXCLUDED_CONTRASTS,
        )
        target = ep.save(figure, out / name)
        usable = table[table["usable"]]
        print(f"  {name:30s} n={len(table):2d} (usable {len(usable):2d}) | "
              f"binary-only {int(table['reference_only'].sum()):4d}  "
              f"shared {int(table['shared'].sum()):4d}  "
              f"{arm}-only {int(table['arm_only'].sum()):4d}  | "
              f"totals {int(table['reference_total'].sum())} vs "
              f"{int(table['arm_total'].sum())}")
        table.to_csv(target.with_suffix(".csv"), index=False)

    # section 5's partition: baseline / indirect / direct, as a two-step build
    check = pd.read_csv(EDGER / "binary_coefficient_check.csv")
    check = check[~check["contrast"].isin(EXCLUDED_CONTRASTS)]
    quality_kept = quality[~quality["contrast"].isin(EXCLUDED_CONTRASTS)]
    for usable_only, tag in ((False, "all"), (True, "gated")):
        for layers, suffix in (("binary", "1_binary_only"), ("both", "2_binary_plus_morph")):
            figure, table = ep.slope_gain_bars(check, quality_kept, usable_only=usable_only,
                                               layers=layers)
            ep.save(figure, out / f"gain_{tag}_{suffix}")
        print(f"  gain_{tag:5s} n={len(table):2d} | baseline {int(table['baseline'].sum()):3d}  "
              f"indirect +{int(table['indirect'].sum()):2d}  direct +{int(table['direct'].sum()):2d}  "
              f"| lost -{int(table['lost'].sum())}  neg. control {int(table['control_slope'].sum())}")

    # the three-panel supervised version, for anyone who wants the scatter alongside
    figure, _ = ep.hit_overlap_plot(
        supervised[~supervised["contrast"].isin(EXCLUDED_CONTRASTS)],
        quality[~quality["contrast"].isin(EXCLUDED_CONTRASTS)], arm="s_z")
    ep.save(figure, out / "hit_overlap_s_three_panel")

    print(f"\nexcluded: {', '.join(EXCLUDED_CONTRASTS)}")
    print(f"wrote -> {out}")
    print("\nNOTE: this host cannot reach a local Google Drive mount. Copy or rsync from the path")
    print("      above to wherever the slide deck lives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
