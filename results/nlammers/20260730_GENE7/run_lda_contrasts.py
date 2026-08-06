"""Fit the 36 supervised crispant-vs-control morphology contrasts and render the figures.

Writes to ``data/lda/``::

    contrast_summary.csv      one row per contrast: AUC, permutation p/q, stability, confound angles
    contrast_scores.csv       one row per (contrast, embryo): s, s_z, s_loo, class label
    contrast_directions.csv   long-form unit loadings over the shared dimensions
    direction_similarity.csv  pairwise |cos| between the 36 directions
    dispersion.csv            per-contrast within-group SD of s (the mosaic-F0 check)
    image_strips.csv          resolved snip paths, ordered along s

and to ``figures/lda/``: the overview heatmap, per-target score panels, the dispersion scatter,
the stability plot, the confound checks, the direction heatmaps, and one image strip per contrast.

The 10D GENE7-native basis is read from ``data/gene7_global_scores.csv`` (written by
``run_cohort_axes.py``) rather than refit, so the contrasts live in exactly the coordinate system the
cohort-PC covariates use. ``--refit-basis`` rebuilds it from the master table instead.

Usage::

    python run_lda_contrasts.py                    # full run
    python run_lda_contrasts.py --quick            # 200 permutations, no strips -- for iterating
    python run_lda_contrasts.py --no-strips
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")

import lda_contrasts as lc  # noqa: E402
import lda_plots as lp  # noqa: E402

DATA_DIR = HERE / "data" / "lda"
FIGURE_DIR = HERE / "figures" / "lda"
STRIP_DIR = FIGURE_DIR / "strips"
CACHED_BASIS = HERE / "data" / "gene7_global_scores.csv"


def load_scores(*, refit: bool = False) -> pd.DataFrame:
    """The GENE7 wells in the shared 10D basis."""
    if not refit and CACHED_BASIS.is_file():
        frame = pd.read_csv(CACHED_BASIS)
        print(f"basis: {len(frame)} wells from {CACHED_BASIS.name}")
        return frame

    import cohort_axes as ca
    import morph_pca_spline as mps
    from morphseq_integration import build_master_table

    print("basis: refitting from the master table ...")
    gene7 = mps.gene7_latents_from_master(build_master_table().query("has_morph").copy())
    basis = ca.fit_global_basis(gene7, whiten=False)
    return basis.scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=lc.N_PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=lc.N_BOOTSTRAPS)
    parser.add_argument("--dims", type=int, default=lc.N_SHARED_DIMS)
    parser.add_argument("--refit-basis", action="store_true")
    parser.add_argument("--no-strips", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="200 permutations / 200 bootstraps and no image strips")
    arguments = parser.parse_args()

    if arguments.quick:
        arguments.permutations = 200
        arguments.bootstraps = 200
        arguments.no_strips = True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    lp.use_house_style()

    scores = load_scores(refit=arguments.refit_basis)
    columns = lc.shared_columns(scores, arguments.dims)
    print(f"       shared subspace: {columns}")

    print(f"\nfitting contrasts ({arguments.permutations} permutations, "
          f"{arguments.bootstraps} bootstraps) ...")
    started = time.time()
    axes = lc.fit_all_contrasts(
        scores,
        n_dims=arguments.dims,
        n_permutations=arguments.permutations,
        n_bootstraps=arguments.bootstraps,
    )
    print(f"       {len(axes)} contrasts in {time.time() - started:.0f}s")

    summary = lc.contrast_summary(axes, n_dims=arguments.dims)
    per_embryo = lc.contrast_scores(axes)
    directions = lc.contrast_directions(axes, n_dims=arguments.dims)
    similarity = lc.direction_similarity(axes)

    summary.to_csv(DATA_DIR / "contrast_summary.csv", index=False)
    per_embryo.to_csv(DATA_DIR / "contrast_scores.csv", index=False)
    directions.to_csv(DATA_DIR / "contrast_directions.csv", index=False)
    similarity.to_csv(DATA_DIR / "direction_similarity.csv")

    print("\n=== how many contrasts have a usable axis? ===")
    print(f"  permutation q < 0.10 (LOO-AUC) : {int((summary['q_auc'] < 0.10).sum())} / {len(summary)}")
    print(f"  Hotelling p < 0.05             : {int((summary['hotelling_p'] < 0.05).sum())} / {len(summary)}")
    print(f"  better determined than random  : "
          f"{int((summary['boot_angle_p95'] < summary['random_angle_median']).sum())} / {len(summary)}")
    print(f"  usable (both gates)            : {int(summary['usable'].sum())} / {len(summary)}")
    print(f"  median LOO-AUC                 : {summary['loo_auc'].median():.3f}")
    print(f"  median shrinkage               : {summary['shrinkage'].median():.3f}")
    print(f"  stage-confounded (angle < 35)  : {int(summary['stage_confounded'].sum())} / {len(summary)}")

    print("\n=== ordinal stability: does the embryo RANKING survive resampling? ===")
    print(f"  median bootstrap rank rho          : {summary['rank_rho_median'].median():.3f}")
    print(f"  median random-direction floor      : {summary['random_rank_rho_median'].median():.3f}")
    print(f"  rank-stable (BH q<0.1 vs floor)    : "
          f"{int(summary['rank_stable'].sum())} / {len(summary)}")
    print(f"  usable by ORDINAL gate             : "
          f"{int(summary['usable_ordinal'].sum())} / {len(summary)}")
    print(f"  usable by ANGULAR gate             : {int(summary['usable'].sum())} / {len(summary)}")
    print(f"  agree                              : "
          f"{int((summary['usable'] == summary['usable_ordinal']).sum())} / {len(summary)}")
    print(f"  median top-tercile retention       : {summary['top_retention_median'].median():.3f} "
          f"(floor {summary['random_top_retention_median'].median():.3f})")
    print(f"  Spearman(angle instability, rank stability) : "
          f"{summary[['boot_angle_p95','rank_rho_median']].corr(method='spearman').iloc[0,1]:+.3f}")
    print("\n=== mosaic-F0 premise: are crispants more dispersed along s? ===")
    print(f"  contrasts with SD_crispant > SD_control : "
          f"{int((summary['log_sd_ratio'] > 0).sum())} / {len(summary)}")
    print(f"  median log SD ratio                     : {summary['log_sd_ratio'].median():+.3f} "
          f"(= {np.exp(summary['log_sd_ratio'].median()):.2f}x)")
    print(f"  individually q < 0.10                   : "
          f"{int((summary['q_dispersion'] < 0.10).sum())} / {len(summary)}")

    agreement = lc.direction_similarity_summary(
        axes, n_dims=arguments.dims, restrict_usable=summary.set_index("contrast")["usable"]
    )
    agreement.to_csv(DATA_DIR / "direction_agreement.csv", index=False)
    print("\n=== do targets keep a signature across conditions? (usable contrasts only) ===")
    print(agreement.round(4).to_string(index=False))

    print("\nrendering figures ...")
    lp.save(lp.auc_heatmap(summary), FIGURE_DIR / "auc_heatmap")

    for target in sorted(summary["target"].unique()):
        safe = target.replace(",", "-")
        lp.save(lp.score_panels(per_embryo, summary, target=target),
                FIGURE_DIR / f"scores_{safe}")

    figure, dispersion = lp.dispersion_scatter(per_embryo, summary)
    lp.save(figure, FIGURE_DIR / "dispersion")
    dispersion.to_csv(DATA_DIR / "dispersion.csv", index=False)

    lp.save(lp.stability_plot(axes, summary), FIGURE_DIR / "stability")
    lp.save(lp.rank_stability_plot(axes, summary), FIGURE_DIR / "rank_stability")
    lp.save(lp.angle_versus_rank(summary), FIGURE_DIR / "angle_versus_rank")
    for label in summary.sort_values("loo_auc", ascending=False)["contrast"].head(2):
        contrast_axis = next(a for a in axes if a.label == label)
        safe = label.replace(" ", "").replace("|", "_").replace(",", "-").replace("vsctrl", "")
        lp.save(lp.rank_churn_figure(contrast_axis), FIGURE_DIR / f"rank_churn_{safe}")
    lp.save(lp.confound_scatter(summary), FIGURE_DIR / "confounds")
    lp.save(lp.direction_heatmap(directions, summary), FIGURE_DIR / "directions")
    lp.save(lp.similarity_heatmap(similarity), FIGURE_DIR / "direction_similarity")

    if not arguments.no_strips:
        STRIP_DIR.mkdir(parents=True, exist_ok=True)
        strips = []
        for axis in axes:
            strip = lc.contrast_image_strip(axis)
            strips.append(strip)
            if not strip["image_exists"].any():
                print(f"  [skip] {axis.label}: no images on disk")
                continue
            row = summary.loc[summary["contrast"] == axis.label].iloc[0]
            figure = lp.contrast_strip_figure(
                strip,
                title=f"{axis.label}   LOO-AUC {row['loo_auc']:.2f}  (q={row['q_auc']:.3f})",
            )
            safe = axis.label.replace(" ", "").replace("|", "_").replace(",", "-").replace("vsctrl", "")
            lp.save(figure, STRIP_DIR / f"strip_{safe}")
        all_strips = pd.concat(strips, ignore_index=True)
        all_strips.to_csv(DATA_DIR / "image_strips.csv", index=False)
        print(f"  {len(list(STRIP_DIR.glob('*.png')))} strips -> {STRIP_DIR}")

    print(f"\ndata    -> {DATA_DIR}")
    print(f"figures -> {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
