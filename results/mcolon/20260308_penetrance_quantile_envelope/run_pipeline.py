"""
Results-local orchestration entrypoint for the WT quantile-envelope penetrance pipeline.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from calls import mark_penetrant
from config import (
    BROAD_CATEGORIES,
    CATEGORY_COL,
    CATEGORY_COLORS,
    DPI,
    EMBRYO_CALL_MODE,
    EMBRYO_BIN_AGG,
    EMBRYO_COL,
    FIGSIZE_CURVES,
    FIGSIZE_DIAGNOSTIC,
    FIGSIZE_HEATMAP,
    FIGURE_DIR,
    FRAME_DIAGNOSTIC_CALL_MODE,
    GENOTYPE_COL,
    GENOTYPE_COLORS,
    HET_GENOTYPE,
    LOESS_CANDIDATE_FRACS,
    LOESS_FRAC_OVERRIDE,
    METRIC_NAME,
    MIN_WT_EMBRYOS_PER_BIN,
    MIN_WT_FRAMES_PER_BIN,
    PRESENTATION_CURVE_FRAC,
    PRESENTATION_CURVE_MODE,
    PRESENTATION_CURVE_SHOW_POINTS,
    PRESENTATION_CURVE_SMOOTH_SE,
    QUANTILE_HIGH,
    QUANTILE_LOW,
    TABLE_DIR,
    TIME_COL,
    UPPER_BOUND_ONLY,
    WT_GENOTYPE,
)
from data_loading import load_data, split_by_genotype
from envelope import aggregate_embryo_bins, compute_wt_envelope
from penetrance_plots import (
    plot_embryo_consistency,
    plot_outside_rate_by_group,
    plot_penetrance_curves,
    plot_penetrance_heatmap,
    plot_quantile_smoother_selection,
    plot_scatter_and_penetrance,
    plot_wt_envelope_diagnostic,
)
from smoothing import loess_smooth
from summaries import (
    compute_calibration,
    compute_penetrance_by_group_and_time,
    compute_penetrance_consistency,
)


def save_loess_selection_figure(
    df_env,
    lower_result,
    upper_result,
    output_dir: Path,
    output_name: str,
    title: str,
):
    """
    Save a candidate smoother figure using the envelope table already computed.
    """
    if lower_result is not None and upper_result is not None:
        fig, _ = plot_quantile_smoother_selection(
            df_env["time_bin"].to_numpy(dtype=float),
            df_env["raw_low"].to_numpy(),
            df_env["raw_high"].to_numpy(),
            lower_result,
            upper_result,
            supported_mask=df_env["supported"].to_numpy(),
            figsize=(16, 6),
        )
    else:
        times = df_env["time_bin"].to_numpy(dtype=float)
        supported = df_env["supported"].to_numpy(dtype=bool)
        tx = times[supported]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        titles = ["Lower curve (2.5%)", "Upper curve (97.5%)"]
        raw_cols = ["raw_low", "raw_high"]
        cmap = plt.cm.cool
        for ax_i, panel_title, raw_col in zip(axes, titles, raw_cols):
            raw_vals = df_env.loc[supported, raw_col].to_numpy()
            ax_i.plot(tx, raw_vals, "ko", ms=4, label="Raw quantile")
            for fi, frac in enumerate(sorted(LOESS_CANDIDATE_FRACS)):
                sm = loess_smooth(tx, raw_vals, frac)
                color = cmap(fi / max(len(LOESS_CANDIDATE_FRACS) - 1, 1))
                lw = 2.5 if frac == LOESS_FRAC_OVERRIDE else 1.0
                label = f"frac={frac}" + (" ← selected" if frac == LOESS_FRAC_OVERRIDE else "")
                ax_i.plot(tx, sm, color=color, lw=lw, label=label)
            ax_i.set_title(panel_title, fontsize=14, fontweight="bold")
            ax_i.set_xlabel("Time bin (hpf)", fontsize=12)
            ax_i.legend(fontsize=7, ncol=2)
            ax_i.spines["top"].set_visible(False)
            ax_i.spines["right"].set_visible(False)
            ax_i.grid(True, alpha=0.3)
        fig.suptitle(title, y=1.01, fontsize=14)
        fig.tight_layout()

    fig.savefig(output_dir / output_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / output_name}")


def _render_penetrance_curve_variants(presentation_dir: Path, pen_cat):
    variants = [
        {
            "suffix": "band_only",
            "title": "Embryo-level penetrance by broad category (SE band + smooth)",
            "curve_mode": PRESENTATION_CURVE_MODE,
            "curve_frac": PRESENTATION_CURVE_FRAC,
            "band_mode": "se",
            "show_band": True,
            "show_line": True,
            "show_points": False,
        },
        {
            "suffix": "dots_only",
            "title": "Embryo-level penetrance by broad category (dots only)",
            "curve_mode": "raw",
            "curve_frac": None,
            "band_mode": "se",
            "show_band": False,
            "show_line": False,
            "show_points": True,
        },
        {
            "suffix": "line_only",
            "title": "Embryo-level penetrance by broad category (smooth line only)",
            "curve_mode": PRESENTATION_CURVE_MODE,
            "curve_frac": PRESENTATION_CURVE_FRAC,
            "band_mode": "se",
            "show_band": False,
            "show_line": True,
            "show_points": False,
        },
        {
            "suffix": "band_line_dots",
            "title": "Embryo-level penetrance by broad category (SE band + smooth + dots)",
            "curve_mode": PRESENTATION_CURVE_MODE,
            "curve_frac": PRESENTATION_CURVE_FRAC,
            "band_mode": "se",
            "show_band": True,
            "show_line": True,
            "show_points": True,
        },
    ]

    for variant in variants:
        fig, _ = plot_penetrance_curves(
            pen_cat,
            x_col="time_bin",
            y_col="penetrance",
            se_col="se",
            band_lower_col="q25",
            band_upper_col="q75",
            group_col="group",
            colors=CATEGORY_COLORS,
            group_order=BROAD_CATEGORIES,
            y_label="Embryo-level penetrance",
            curve_mode=variant["curve_mode"],
            curve_frac=variant["curve_frac"],
            band_mode=variant["band_mode"],
            show_band=variant["show_band"],
            show_line=variant["show_line"],
            show_points=variant["show_points"],
            smooth_se=PRESENTATION_CURVE_SMOOTH_SE,
            title=variant["title"],
            figsize=FIGSIZE_CURVES,
        )
        out_path = presentation_dir / f"penetrance_curves_by_category_embryo__{variant['suffix']}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def _render_scatter_variants(presentation_dir: Path, embryo_bin_df, embryo_env, geno_order):
    variants = [
        {
            "suffix": "band_only",
            "title": "Embryo-bin summaries + embryo-level penetrance by genotype (SE band + smooth)",
            "curve_mode": PRESENTATION_CURVE_MODE,
            "curve_frac": PRESENTATION_CURVE_FRAC,
            "band_mode": "se",
            "show_band": True,
            "show_line": True,
            "show_points": False,
        },
        {
            "suffix": "dots_only",
            "title": "Embryo-bin summaries + embryo-level penetrance by genotype (dots only)",
            "curve_mode": "raw",
            "curve_frac": None,
            "band_mode": "se",
            "show_band": False,
            "show_line": False,
            "show_points": True,
        },
        {
            "suffix": "line_only",
            "title": "Embryo-bin summaries + embryo-level penetrance by genotype (smooth line only)",
            "curve_mode": PRESENTATION_CURVE_MODE,
            "curve_frac": PRESENTATION_CURVE_FRAC,
            "band_mode": "se",
            "show_band": False,
            "show_line": True,
            "show_points": False,
        },
        {
            "suffix": "band_line_dots",
            "title": "Embryo-bin summaries + embryo-level penetrance by genotype (SE band + smooth + dots)",
            "curve_mode": PRESENTATION_CURVE_MODE,
            "curve_frac": PRESENTATION_CURVE_FRAC,
            "band_mode": "se",
            "show_band": True,
            "show_line": True,
            "show_points": True,
        },
    ]

    for variant in variants:
        fig, _ = plot_scatter_and_penetrance(
            embryo_bin_df[embryo_bin_df[GENOTYPE_COL].isin(geno_order)],
            embryo_env,
            time_col=TIME_COL,
            metric_col=METRIC_NAME,
            embryo_col=EMBRYO_COL,
            group_col=GENOTYPE_COL,
            group_order=geno_order,
            colors=GENOTYPE_COLORS,
            upper_only=UPPER_BOUND_ONLY,
            title=variant["title"],
            figsize_per_col=(7, 9),
            envelope_lower_col="smoothed_low",
            envelope_upper_col="smoothed_high",
            top_ylabel=f"{METRIC_NAME} ({EMBRYO_BIN_AGG} per embryo/bin)",
            overall_label="Overall embryo-bin penetrant",
            bottom_label="Embryo-level penetrance (%)",
            penetrance_curve_mode=variant["curve_mode"],
            penetrance_curve_frac=variant["curve_frac"],
            penetrance_band_mode=variant["band_mode"],
            show_penetrance_band=variant["show_band"],
            show_penetrance_line=variant["show_line"],
            show_penetrance_points=variant["show_points"],
        )
        out_path = presentation_dir / f"scatter_penetrance_by_genotype_embryo__{variant['suffix']}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    presentation_dir = FIGURE_DIR / "presentation"
    diagnostics_dir = FIGURE_DIR / "diagnostics"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print("=== Load data ===")
    frame_df, _ = load_data()
    wt_frame_df, _ = split_by_genotype(frame_df)

    print(f"\n=== Aggregate embryo-bin summaries ({EMBRYO_BIN_AGG}) ===")
    embryo_bin_df = aggregate_embryo_bins(frame_df, agg=EMBRYO_BIN_AGG)
    wt_embryo_bin_df = embryo_bin_df[embryo_bin_df[GENOTYPE_COL] == WT_GENOTYPE].copy()
    print(
        f"  {len(embryo_bin_df):,} embryo-bin summaries "
        f"from {embryo_bin_df[EMBRYO_COL].nunique():,} embryos"
    )

    print("\n=== Main embryo-level envelope ===")
    embryo_env, embryo_lower_result, embryo_upper_result = compute_wt_envelope(
        wt_embryo_bin_df,
        min_units=MIN_WT_EMBRYOS_PER_BIN,
        unit_label="embryo-bin summary",
    )
    embryo_env.to_csv(TABLE_DIR / "wt_threshold_summary_embryo.csv", index=False)
    print(f"Saved: {TABLE_DIR / 'wt_threshold_summary_embryo.csv'}")

    print(f"\n=== Main embryo-level classification ({EMBRYO_CALL_MODE} thresholds) ===")
    embryo_bin_df = mark_penetrant(embryo_bin_df, embryo_env, call_mode=EMBRYO_CALL_MODE)
    embryo_class_cols = [
        EMBRYO_COL,
        "time_bin",
        TIME_COL,
        METRIC_NAME,
        "n_frames_in_bin",
        GENOTYPE_COL,
        CATEGORY_COL,
        "cluster_subcategories",
        "threshold_low",
        "threshold_high",
        "threshold_source_low",
        "threshold_source_high",
        "threshold_call_mode",
        "raw_low",
        "raw_high",
        "smoothed_low",
        "smoothed_high",
        "supported",
        "penetrant",
    ]
    keep_cols = [c for c in embryo_class_cols if c in embryo_bin_df.columns]
    embryo_bin_df[keep_cols].to_csv(TABLE_DIR / "embryo_bin_classification.csv", index=False)
    print(f"Saved: {TABLE_DIR / 'embryo_bin_classification.csv'}")

    print("\n=== Embryo-level penetrance by category ===")
    pen_cat = compute_penetrance_by_group_and_time(
        embryo_bin_df,
        CATEGORY_COL,
        unit_col=EMBRYO_COL,
        count_col_name="n_embryos",
    )
    pen_cat.to_csv(TABLE_DIR / "category_penetrance_by_time_embryo.csv", index=False)
    print(f"Saved: {TABLE_DIR / 'category_penetrance_by_time_embryo.csv'}")

    print("\n=== Embryo-level consistency across bins ===")
    emb_consistency = compute_penetrance_consistency(embryo_bin_df, CATEGORY_COL)
    emb_consistency.to_csv(TABLE_DIR / "embryo_penetrance_consistency_embryo.csv", index=False)
    print(f"Saved: {TABLE_DIR / 'embryo_penetrance_consistency_embryo.csv'}")

    print("\n=== Embryo-level category summary (all bins) ===")
    for grp_name, grp in pen_cat.groupby("group"):
        overall_p = grp["n_penetrant"].sum() / max(grp["n_embryos"].sum(), 1)
        print(f"  {grp_name}: {overall_p:.3f}")

    print("\n=== Frame-level diagnostics ===")
    frame_env, frame_lower_result, frame_upper_result = compute_wt_envelope(
        wt_frame_df,
        min_units=MIN_WT_FRAMES_PER_BIN,
        unit_label="frame",
    )
    frame_env.to_csv(TABLE_DIR / "wt_threshold_summary_frame_diagnostic.csv", index=False)
    print(f"Saved: {TABLE_DIR / 'wt_threshold_summary_frame_diagnostic.csv'}")

    frame_diag_df = mark_penetrant(frame_df, frame_env, call_mode=FRAME_DIAGNOSTIC_CALL_MODE)
    cal_overall, cal_by_time = compute_calibration(frame_diag_df, count_col_name="n_frames")
    print("\n=== WT / Het frame calibration (diagnostic) ===")
    for _, row in cal_overall.iterrows():
        print(f"  {row['genotype']}: outside-rate = {row['outside_rate']:.3f} (n={row['n_frames']:,} frames)")

    print("\n=== Saving figures ===")
    fig, _ = plot_wt_envelope_diagnostic(
        wt_embryo_bin_df,
        embryo_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        scatter_label="WT embryo-bin summaries",
        title="WT Embryo-Bin Envelope (raw thresholds)",
        show_smoothed=False,
        show_raw=True,
        envelope_lower_col="raw_low",
        envelope_upper_col="raw_high",
        figsize=FIGSIZE_DIAGNOSTIC,
    )
    fig.savefig(presentation_dir / "wt_envelope_diagnostic_embryo_raw.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'wt_envelope_diagnostic_embryo_raw.png'}")

    fig, _ = plot_wt_envelope_diagnostic(
        wt_embryo_bin_df,
        embryo_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        scatter_label="WT embryo-bin summaries",
        title="WT Embryo-Bin Envelope (robust smoothed display)",
        show_smoothed=True,
        show_raw=True,
        envelope_lower_col="smoothed_low",
        envelope_upper_col="smoothed_high",
        exclude_lower_col="smooth_excluded_low",
        exclude_upper_col="smooth_excluded_high",
        figsize=FIGSIZE_DIAGNOSTIC,
    )
    fig.savefig(presentation_dir / "wt_envelope_diagnostic_embryo_smoothed.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'wt_envelope_diagnostic_embryo_smoothed.png'}")

    save_loess_selection_figure(
        embryo_env,
        embryo_lower_result,
        embryo_upper_result,
        diagnostics_dir,
        "loess_frac_selection_embryo_diagnostic.png",
        "LOESS Frac Selection (embryo-bin diagnostic)",
    )

    fig, _ = plot_wt_envelope_diagnostic(
        wt_frame_df,
        frame_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        scatter_label="WT frames",
        title="WT Frame Envelope Diagnostic",
        show_smoothed=True,
        show_raw=True,
        envelope_lower_col="smoothed_low",
        envelope_upper_col="smoothed_high",
        figsize=FIGSIZE_DIAGNOSTIC,
    )
    fig.savefig(diagnostics_dir / "wt_envelope_diagnostic_frame.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {diagnostics_dir / 'wt_envelope_diagnostic_frame.png'}")

    save_loess_selection_figure(
        frame_env,
        frame_lower_result,
        frame_upper_result,
        diagnostics_dir,
        "loess_frac_selection_frame.png",
        "LOESS Frac Selection (frame diagnostic)",
    )

    cal_plot_df = cal_by_time.rename(columns={"genotype": "group"})
    fig, ax = plot_outside_rate_by_group(
        cal_plot_df,
        expected_rate=1 - (QUANTILE_HIGH - QUANTILE_LOW),
        x_col="time_bin",
        y_col="outside_rate",
        group_col="group",
        colors=GENOTYPE_COLORS,
        y_label="Outside-envelope rate (frame diagnostic)",
        figsize=(10, 5),
    )
    ax.set_title("WT & Het Calibration Check (frame diagnostic)", fontsize=14, fontweight="bold")
    fig.savefig(diagnostics_dir / "wt_het_calibration_frame.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {diagnostics_dir / 'wt_het_calibration_frame.png'}")

    fig, _ = plot_penetrance_curves(
        pen_cat,
        x_col="time_bin",
        y_col="penetrance",
        se_col="se",
        group_col="group",
        colors=CATEGORY_COLORS,
        group_order=BROAD_CATEGORIES,
        y_label="Embryo-level penetrance",
        curve_mode="raw",
        show_points=True,
        smooth_se=False,
        title="Embryo-level penetrance by broad category (raw)",
        figsize=FIGSIZE_CURVES,
    )
    fig.savefig(presentation_dir / "penetrance_curves_by_category_embryo_raw.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'penetrance_curves_by_category_embryo_raw.png'}")

    fig, _ = plot_penetrance_curves(
        pen_cat,
        x_col="time_bin",
        y_col="penetrance",
        se_col="se",
        group_col="group",
        colors=CATEGORY_COLORS,
        group_order=BROAD_CATEGORIES,
        y_label="Embryo-level penetrance",
        curve_mode=PRESENTATION_CURVE_MODE,
        curve_frac=PRESENTATION_CURVE_FRAC,
        show_points=PRESENTATION_CURVE_SHOW_POINTS,
        smooth_se=PRESENTATION_CURVE_SMOOTH_SE,
        title=f"Embryo-level penetrance by broad category ({PRESENTATION_CURVE_MODE})",
        figsize=FIGSIZE_CURVES,
    )
    fig.savefig(presentation_dir / "penetrance_curves_by_category_embryo_smoothed.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'penetrance_curves_by_category_embryo_smoothed.png'}")

    _render_penetrance_curve_variants(presentation_dir, pen_cat)

    fig, _ = plot_penetrance_heatmap(
        pen_cat,
        x_col="time_bin",
        y_col="group",
        value_col="penetrance",
        group_order=BROAD_CATEGORIES,
        colorbar_label="Embryo-level penetrance",
        title="Embryo-level penetrance heatmap (broad categories × time)",
        figsize=FIGSIZE_HEATMAP,
    )
    fig.savefig(presentation_dir / "penetrance_heatmap_category_embryo.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'penetrance_heatmap_category_embryo.png'}")

    fig, _ = plot_embryo_consistency(
        emb_consistency,
        group_col="group",
        value_col="frac_penetrant",
        colors=CATEGORY_COLORS,
        group_order=BROAD_CATEGORIES,
        xlabel="Fraction of bins penetrant",
        title="Per-embryo penetrance consistency across bins",
    )
    fig.savefig(presentation_dir / "embryo_consistency_histograms_embryo.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'embryo_consistency_histograms_embryo.png'}")

    geno_order = [WT_GENOTYPE, HET_GENOTYPE, "cep290_homozygous"]
    fig, _ = plot_scatter_and_penetrance(
        embryo_bin_df[embryo_bin_df[GENOTYPE_COL].isin(geno_order)],
        embryo_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        group_col=GENOTYPE_COL,
        group_order=geno_order,
        colors=GENOTYPE_COLORS,
        upper_only=UPPER_BOUND_ONLY,
        title="Embryo-bin summaries + embryo-level penetrance by genotype",
        figsize_per_col=(7, 9),
        envelope_lower_col="raw_low",
        envelope_upper_col="raw_high",
        top_ylabel=f"{METRIC_NAME} ({EMBRYO_BIN_AGG} per embryo/bin)",
        overall_label="Overall embryo-bin penetrant",
        bottom_label="Embryo-level penetrance (%)",
    )
    fig.savefig(presentation_dir / "scatter_penetrance_by_genotype_embryo_raw.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'scatter_penetrance_by_genotype_embryo_raw.png'}")

    fig, _ = plot_scatter_and_penetrance(
        embryo_bin_df[embryo_bin_df[GENOTYPE_COL].isin(geno_order)],
        embryo_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        group_col=GENOTYPE_COL,
        group_order=geno_order,
        colors=GENOTYPE_COLORS,
        upper_only=UPPER_BOUND_ONLY,
        title="Embryo-bin summaries + robust smoothed envelope by genotype",
        figsize_per_col=(7, 9),
        envelope_lower_col="smoothed_low",
        envelope_upper_col="smoothed_high",
        top_ylabel=f"{METRIC_NAME} ({EMBRYO_BIN_AGG} per embryo/bin)",
        overall_label="Overall embryo-bin penetrant",
        bottom_label="Embryo-level penetrance (%)",
        penetrance_curve_mode=PRESENTATION_CURVE_MODE,
        penetrance_curve_frac=PRESENTATION_CURVE_FRAC,
        show_penetrance_points=PRESENTATION_CURVE_SHOW_POINTS,
    )
    fig.savefig(presentation_dir / "scatter_penetrance_by_genotype_embryo_smoothed.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {presentation_dir / 'scatter_penetrance_by_genotype_embryo_smoothed.png'}")

    _render_scatter_variants(presentation_dir, embryo_bin_df, embryo_env, geno_order)

    fig, _ = plot_scatter_and_penetrance(
        frame_diag_df[frame_diag_df[GENOTYPE_COL].isin(geno_order)],
        frame_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        group_col=GENOTYPE_COL,
        group_order=geno_order,
        colors=GENOTYPE_COLORS,
        upper_only=UPPER_BOUND_ONLY,
        title="Frame scatter + diagnostic penetrance by genotype",
        figsize_per_col=(7, 9),
        envelope_lower_col="smoothed_low",
        envelope_upper_col="smoothed_high",
        top_ylabel=f"{METRIC_NAME} (frame diagnostic)",
        overall_label="Overall frame diagnostic outside-rate",
        bottom_label="Any-frame embryo penetrance (%)",
    )
    fig.savefig(diagnostics_dir / "scatter_penetrance_by_genotype_frame_diagnostic.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {diagnostics_dir / 'scatter_penetrance_by_genotype_frame_diagnostic.png'}")

    print("\n=== Done ===")
    print(f"  Presentation figures: {presentation_dir}")
    print(f"  Diagnostic figures:   {diagnostics_dir}")
    print(f"  Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()
