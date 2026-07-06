# Feature / QC Visualization — Migration Notes (scratch, NOT a contract)

**Status:** field notes, moved out of `feature_world.md` (they had been pasted mid-sentence into the
`Not This World` section). These are candidate **legacy results scripts** to mine when building the
*debug/review plots* for the migrated feature and QC products. They are not doctrine, not a target
spec, and define no contract — purely a pointer list for whoever builds the review tooling.

Paths point into the sibling `proj/morphseq/` results tree (not `-docs`).

---

  The strongest candidates I found are these.

  Best QC migration candidates

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20260326_pbx_crispant_analysis/scripts/14_qc_reason_death_review.py:1
    This is probably the “nice per-embryo QC viz” you were remembering. It writes:
      - dead_flag_persistence_review.png
      - SAM2/frame/granular QC reason breakdown plots
      - CSV summaries/manifests
        It is explicitly wired around dead_flag, dead_flag2, fraction_alive, and
        persistence review.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20260326_pbx_crispant_analysis/scripts/attrition_qc/review.py:765
    This contains the actual per-embryo death-review plotting function
    plot_dead_flag_review(...). It also has select_dead_flag_review_embryos(...) and
    summarize_dead_flag_agreement(...) nearby. If you want to move one reusable debug
    plot into quality_control/death_detection, this is the cleanest source.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20260213_subtle_phenotype_methods/plot_persistence_diagnostics.py:1
    More cohort-level than embryo-level, but still directly relevant to persistence/
    death-style QC. It generates scan/heatmap diagnostics for embryo-first persistence
    runs.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20251010_sa_outlier_analysis/README.md:1
    This whole folder maps well onto future surface_area_qc.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20251010_sa_outlier_analysis/build_sa_reference.py:1
    Builds the stage-binned SA reference and writes reference_plot.png.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20251010_sa_outlier_analysis/tune_thresholds.py:1
    Produces test_embryos_vs_reference.png and flagging_rate_heatmap.png.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20251010_sa_outlier_analysis/apply_two_sided_qc.py:1
    Produces two_sided_qc_validation.png and embryo-level flagged summaries.

  Curvature / feature visualization code

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/
    embryo_curvature_analysis.py:1
    Single-embryo curvature analysis with multiple centerline methods and curvature
    plots.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/
    compare_curvature_methods.py:1
    Direct method-comparison visualization for curvature extraction.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251028/
    curvature_validation/visualize_curvature_metrics.py:1
    Histograms plus top/bottom embryo examples with mask, spline, baseline, and metric
    annotations. This is a strong candidate for future curvature_metrics debugging.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027/
    visualize_metric_examples.py:1
    Gallery of real masks for morphology metric ranges like perimeter/area and solidity.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027/
    visualize_extent_examples.py:1
    Real-mask examples for low-extent shapes, useful for artifact/shape sanity checks.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20251029_curvature_temporal_analysis/README.md:1
    This folder is the main temporal curvature-viz bundle:
      - 01_individual_trajectories.py
      - 02_horizon_plots.py
      - 06d_visualize_troublesome_masks.py
      - trajectory_visualization.py

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20251113_curvature_pair_analysis/analyze_pairs.py:1
    Pair/genotype trajectory plots for curvature phenotype comparisons.

  Other QC/artifact galleries worth keeping in mind
    mcolon/20260421_motion_artifact_detection/04_pair_metrics_plot.py:1
    Z-pair metric traces for NCC / phase shift.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20260423_focus_artifact_detection/08_rel_entropy_visual_gallery.py:1
    Ranked focus gallery for rel_entropy_mean.

  - /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20260607_sci_cilia_gene14_imaging_qc/3f_embryo_portfolio.py:1 and /net/
    trapnell/vol1/home/mdcolon/proj/morphseq/results/
    mcolon/20260605_sci_cilia_qc_first_pass/make_embryo_portfolio.py:1
    Not feature-specific, but these are solid per-embryo review canvases for QC spot-
    checking.

  If you want the short version: the first things I’d consider moving are the death-
  review bundle from 20260326_pbx_crispant_analysis/scripts/attrition_qc/ and the
  surface-area QC bundle in 20251010_sa_outlier_analysis/. After that, the best curvature
  debug visual is 20251028/curvature_validation/visualize_curvature_metrics.py.

  Next useful step would be to turn this into a small migration map: future module ->
  candidate results scripts -> likely reusable plotting functions.
