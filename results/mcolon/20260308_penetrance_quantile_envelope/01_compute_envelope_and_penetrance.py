"""
WT Quantile Envelope Penetrance Pipeline
=========================================

Design:
- Penetrance is a FRAME-LEVEL property; embryo summaries are secondary/descriptive.
- LOESS frac selected by envelope shape stability (no oscillations, non-negative),
  NOT by targeting WT outside-rate.
- WT calibration is a diagnostic; inference deferred to phase 2.
- Het serves as a WT-like calibration reference group.

Note on ``compute_penetrance_by_time`` / ``mark_threshold_violations``:
  These exist in ``src/analyze/difference_detection/penetrance_threshold.py`` with
  embryo-level semantics.  Functions here are FRAME-level and have different semantics;
  no collision.

Outputs (outputs/figures/ and outputs/tables/):
  Figures:
    wt_envelope_diagnostic.png
    loess_frac_selection.png
    wt_het_calibration.png
    penetrance_curves_by_category.png
    penetrance_heatmap_category.png
    embryo_consistency_histograms.png
  Tables:
    wt_threshold_summary.csv
    category_penetrance_by_time.csv
    embryo_penetrance_consistency.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Imports from this folder
# ---------------------------------------------------------------------------
from data_loading import load_data, split_by_genotype
from config import (
    METRIC_NAME, TIME_COL, EMBRYO_COL, GENOTYPE_COL,
    CATEGORY_COL, SUBCATEGORY_COL,
    WT_GENOTYPE, HET_GENOTYPE,
    TIME_BIN_WIDTH,
    QUANTILE_LOW, QUANTILE_HIGH,
    LOESS_CANDIDATE_FRACS, LOESS_FRAC_OVERRIDE, LOESS_FALLBACK_FRAC,
    MIN_WT_FRAMES_PER_BIN, METRIC_NONNEG, UPPER_BOUND_ONLY,
    BROAD_CATEGORIES, SUBCATEGORIES,
    CATEGORY_COLORS, SUBCATEGORY_COLORS, GENOTYPE_COLORS,
    OUTPUT_DIR, FIGURE_DIR, TABLE_DIR,
    FIGSIZE_DIAGNOSTIC, FIGSIZE_CURVES, FIGSIZE_HEATMAP, FIGSIZE_BARS,
    KEY_STAGES_HPF, DPI,
)
from smoothing import loess_smooth, select_quantile_curve_smoother, SmoothedCurveSelection
from penetrance_plots import (
    plot_wt_envelope_diagnostic,
    plot_quantile_smoother_selection,
    plot_outside_rate_by_group,
    plot_penetrance_curves,
    plot_penetrance_heatmap,
    plot_embryo_consistency,
)


# ---------------------------------------------------------------------------
# Step 3.1: Raw quantiles per time bin
# ---------------------------------------------------------------------------

def compute_raw_wt_quantiles(wt_df, min_frames=MIN_WT_FRAMES_PER_BIN):
    """
    Compute raw 2.5th / 97.5th percentile of METRIC_NAME per time bin.

    Returns
    -------
    pd.DataFrame with columns:
        time_bin, raw_low, raw_high, n_wt_frames, n_wt_embryos, supported
    """
    rows = []
    for tb, grp in wt_df.groupby("time_bin"):
        vals = grp[METRIC_NAME].dropna().values
        n_frames = len(vals)
        n_embryos = grp[EMBRYO_COL].nunique()
        supported = n_frames >= min_frames
        if n_frames > 0:
            raw_low = float(np.quantile(vals, QUANTILE_LOW))
            raw_high = float(np.quantile(vals, QUANTILE_HIGH))
        else:
            raw_low = raw_high = np.nan
        rows.append(dict(
            time_bin=tb,
            raw_low=raw_low,
            raw_high=raw_high,
            n_wt_frames=n_frames,
            n_wt_embryos=n_embryos,
            supported=supported,
        ))
    df_q = pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)
    print(f"\nRaw quantiles: {df_q['supported'].sum()}/{len(df_q)} bins supported "
          f"(>= {min_frames} WT frames)")
    return df_q


# ---------------------------------------------------------------------------
# Step 3.3: Envelope validation
# ---------------------------------------------------------------------------

def validate_envelope(lower_sm, upper_sm, supported_mask, metric_nonneg=True):
    """
    Ensure lower < upper and lower >= 0 (if nonneg). Clips and warns.

    Returns validated (lower_sm, upper_sm).
    """
    lower_sm = lower_sm.copy()
    upper_sm = upper_sm.copy()

    valid = np.where(supported_mask)[0]

    # Clip lower to 0
    if metric_nonneg:
        neg = valid[lower_sm[valid] < 0]
        if len(neg) > 0:
            print(f"  WARNING: Clipping {len(neg)} bins where smoothed lower < 0")
            lower_sm[neg] = 0.0

    # Ensure lower < upper
    crossed = valid[lower_sm[valid] >= upper_sm[valid]]
    if len(crossed) > 0:
        print(f"  WARNING: {len(crossed)} bins where lower >= upper; nudging upper up")
        for i in crossed:
            upper_sm[i] = lower_sm[i] + 1e-6

    return lower_sm, upper_sm


# ---------------------------------------------------------------------------
# Compute full WT envelope
# ---------------------------------------------------------------------------

def compute_wt_envelope(wt_df):
    """
    Compute raw quantiles and smoothed envelope for WT data.

    Returns
    -------
    df_env : pd.DataFrame
        Columns: time_bin, raw_low, raw_high, smoothed_low, smoothed_high,
                 lower_frac, upper_frac, n_wt_frames, n_wt_embryos, supported
    lower_result, upper_result : SmoothedCurveSelection or None
        None when LOESS_FRAC_OVERRIDE is set.
    """
    print("\n=== Step 1: Raw WT quantiles ===")
    df_q = compute_raw_wt_quantiles(wt_df)

    times = df_q["time_bin"].values.astype(float)
    supported = df_q["supported"].values

    lower_result = upper_result = None

    if LOESS_FRAC_OVERRIDE is not None:
        print(f"\n=== Step 2: Smoothing (OVERRIDE frac={LOESS_FRAC_OVERRIDE}) ===")
        frac_low = frac_high = LOESS_FRAC_OVERRIDE
        valid = ~np.isnan(df_q["raw_low"].values)
        sm_low = np.full(len(times), np.nan)
        sm_high = np.full(len(times), np.nan)
        sm_low[valid] = loess_smooth(
            times[valid], df_q["raw_low"].values[valid], frac_low)
        sm_high[valid] = loess_smooth(
            times[valid], df_q["raw_high"].values[valid], frac_high)
    else:
        print("\n=== Step 2: Smoothing frac selection by shape stability ===")
        lower_result = select_quantile_curve_smoother(
            times, df_q["raw_low"].values,
            candidate_fracs=LOESS_CANDIDATE_FRACS,
            nonnegative=METRIC_NONNEG,
            fallback_frac=LOESS_FALLBACK_FRAC,
        )
        upper_result = select_quantile_curve_smoother(
            times, df_q["raw_high"].values,
            candidate_fracs=LOESS_CANDIDATE_FRACS,
            nonnegative=METRIC_NONNEG,
            fallback_frac=LOESS_FALLBACK_FRAC,
        )

        # Caller-level logging (not inside utilities)
        for name, result in [("lower", lower_result), ("upper", upper_result)]:
            fb = " [FALLBACK]" if result.used_fallback else ""
            print(f"  [{name}] selected frac={result.selected_frac}{fb}")
            for frac, diag in result.diagnostics.items():
                status = "ok" if diag["passed"] else f"fail:{diag['failed_checks']}"
                print(f"    frac={frac}: {status}")

        frac_low = lower_result.selected_frac
        frac_high = upper_result.selected_frac
        sm_low = lower_result.smoothed_y
        sm_high = upper_result.smoothed_y

    print("\n=== Step 3: Envelope validation ===")
    sm_low, sm_high = validate_envelope(sm_low, sm_high, supported, metric_nonneg=METRIC_NONNEG)

    df_env = df_q.copy()
    df_env["smoothed_low"] = sm_low
    df_env["smoothed_high"] = sm_high
    df_env["lower_frac"] = frac_low if LOESS_FRAC_OVERRIDE is None else LOESS_FRAC_OVERRIDE
    df_env["upper_frac"] = frac_high if LOESS_FRAC_OVERRIDE is None else LOESS_FRAC_OVERRIDE

    return df_env, lower_result, upper_result


# ---------------------------------------------------------------------------
# Step 4: Penetrance definitions
# ---------------------------------------------------------------------------

def mark_penetrant(df, df_env, upper_bound_only=UPPER_BOUND_ONLY):
    """
    Mark each frame as penetrant (1), inside (0), or unknown (NaN).

    Parameters
    ----------
    upper_bound_only : bool
        If True, only frames exceeding the upper bound are penetrant.
        Use for deviation metrics where "too low" is not a phenotype.

    Vectorized merge on time_bin.
    """
    env_lookup = df_env.set_index("time_bin")[["smoothed_low", "smoothed_high", "supported"]]

    df = df.copy()
    df = df.join(env_lookup, on="time_bin")

    if upper_bound_only:
        outside = df[METRIC_NAME] > df["smoothed_high"]
    else:
        outside = (df[METRIC_NAME] < df["smoothed_low"]) | (df[METRIC_NAME] > df["smoothed_high"])

    penetrant = np.where(~df["supported"], np.nan, np.where(outside, 1.0, 0.0))
    df["penetrant"] = penetrant
    df.drop(columns=["smoothed_low", "smoothed_high", "supported"], inplace=True)
    return df


def compute_penetrance_by_group_and_time(df, group_col):
    """
    Frame-level penetrance per group × time_bin.

    Returns
    -------
    pd.DataFrame with columns:
        group, time_bin, penetrance, n_frames, n_penetrant, n_embryos, se
    """
    rows = []
    for (group, tb), grp in df.groupby([group_col, "time_bin"]):
        valid = grp["penetrant"].dropna()
        n_frames = len(valid)
        if n_frames == 0:
            continue
        n_penetrant = int(valid.sum())
        p = n_penetrant / n_frames
        se = np.sqrt(p * (1 - p) / n_frames) if n_frames > 0 else np.nan
        n_embryos = grp[EMBRYO_COL].nunique()
        rows.append(dict(
            group=group, time_bin=tb,
            penetrance=p, n_frames=n_frames,
            n_penetrant=n_penetrant, n_embryos=n_embryos, se=se,
        ))
    return pd.DataFrame(rows).sort_values(["group", "time_bin"]).reset_index(drop=True)


def compute_embryo_penetrance_consistency(df, group_col):
    """
    Per-embryo fraction of frames that are penetrant.

    Returns
    -------
    pd.DataFrame: embryo_id, group, n_total_frames, n_penetrant_frames, frac_penetrant
    """
    rows = []
    for embryo, grp in df.groupby(EMBRYO_COL):
        group = grp[group_col].iloc[0]
        valid = grp["penetrant"].dropna()
        n_total = len(valid)
        n_pen = int(valid.sum()) if n_total > 0 else 0
        frac = n_pen / n_total if n_total > 0 else np.nan
        rows.append(dict(
            embryo_id=embryo, group=group,
            n_total_frames=n_total, n_penetrant_frames=n_pen,
            frac_penetrant=frac,
        ))
    return pd.DataFrame(rows).sort_values(["group", "embryo_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5: WT + Het calibration
# ---------------------------------------------------------------------------

def compute_calibration(df, df_env):
    """
    Compute outside-envelope rate for WT and Het, overall and by time bin.

    Returns
    -------
    overall : pd.DataFrame — genotype, outside_rate, n_frames
    by_time : pd.DataFrame — genotype, time_bin, outside_rate, n_frames
    """
    cal_df = df[df[GENOTYPE_COL].isin([WT_GENOTYPE, HET_GENOTYPE])].copy()

    overall_rows = []
    time_rows = []

    for geno, grp in cal_df.groupby(GENOTYPE_COL):
        valid = grp["penetrant"].dropna()
        n = len(valid)
        rate = valid.mean() if n > 0 else np.nan
        overall_rows.append(dict(genotype=geno, outside_rate=rate, n_frames=n))

        for tb, tgrp in grp.groupby("time_bin"):
            tv = tgrp["penetrant"].dropna()
            tn = len(tv)
            tr = tv.mean() if tn > 0 else np.nan
            time_rows.append(dict(genotype=geno, time_bin=tb, outside_rate=tr, n_frames=tn))

    overall = pd.DataFrame(overall_rows)
    by_time = pd.DataFrame(time_rows).sort_values(["genotype", "time_bin"]).reset_index(drop=True)

    print("\n=== WT / Het Calibration (diagnostic) ===")
    for _, row in overall.iterrows():
        print(f"  {row['genotype']}: outside-rate = {row['outside_rate']:.3f} (n={row['n_frames']:,} frames)")
    print(f"  (Expected ~{QUANTILE_LOW*2:.3f} = {(QUANTILE_HIGH-QUANTILE_LOW):.0%} envelope covers WT by construction)")

    return overall, by_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df, time_bins = load_data()
    wt_df, het_df = split_by_genotype(df)

    # -----------------------------------------------------------------------
    # WT envelope
    # -----------------------------------------------------------------------
    df_env, lower_result, upper_result = compute_wt_envelope(wt_df)

    # Save threshold table
    threshold_cols = [
        "time_bin", "n_wt_frames", "n_wt_embryos",
        "raw_low", "raw_high", "smoothed_low", "smoothed_high",
        "lower_frac", "upper_frac", "supported",
    ]
    df_env[threshold_cols].to_csv(TABLE_DIR / "wt_threshold_summary.csv", index=False)
    print(f"\nSaved: {TABLE_DIR / 'wt_threshold_summary.csv'}")

    # -----------------------------------------------------------------------
    # Mark penetrant frames (all genotypes)
    # -----------------------------------------------------------------------
    print("\n=== Step 4: Marking penetrant frames ===")
    df = mark_penetrant(df, df_env)
    n_valid = df["penetrant"].notna().sum()
    n_pen = df["penetrant"].sum()
    print(f"  {int(n_pen):,} / {n_valid:,} valid frames marked penetrant ({n_pen/n_valid:.3f})")

    # -----------------------------------------------------------------------
    # WT + Het calibration
    # -----------------------------------------------------------------------
    cal_overall, cal_by_time = compute_calibration(df, df_env)

    # -----------------------------------------------------------------------
    # Penetrance by category
    # -----------------------------------------------------------------------
    print("\n=== Step 5: Frame-level penetrance by category ===")
    pen_cat = compute_penetrance_by_group_and_time(df, CATEGORY_COL)
    pen_cat.to_csv(TABLE_DIR / "category_penetrance_by_time.csv", index=False)
    print(f"  Saved: {TABLE_DIR / 'category_penetrance_by_time.csv'}")

    # -----------------------------------------------------------------------
    # Embryo consistency
    # -----------------------------------------------------------------------
    print("\n=== Step 6: Embryo-level consistency (secondary/descriptive) ===")
    emb_pen = compute_embryo_penetrance_consistency(df, CATEGORY_COL)
    emb_pen.to_csv(TABLE_DIR / "embryo_penetrance_consistency.csv", index=False)
    print(f"  Saved: {TABLE_DIR / 'embryo_penetrance_consistency.csv'}")

    # -----------------------------------------------------------------------
    # Print group summaries
    # -----------------------------------------------------------------------
    print("\n=== Category penetrance summary (all time) ===")
    for grp_name, grp in pen_cat.groupby("group"):
        overall_p = grp["n_penetrant"].sum() / max(grp["n_frames"].sum(), 1)
        print(f"  {grp_name}: {overall_p:.3f}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\n=== Saving figures ===")

    # 1. WT envelope diagnostic
    fig, ax = plot_wt_envelope_diagnostic(
        wt_df, df_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        figsize=FIGSIZE_DIAGNOSTIC,
    )
    fig.savefig(FIGURE_DIR / "wt_envelope_diagnostic.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: wt_envelope_diagnostic.png")

    # 2. LOESS frac selection
    if lower_result is not None and upper_result is not None:
        times = df_env["time_bin"].values.astype(float)
        fig, axes = plot_quantile_smoother_selection(
            times,
            df_env["raw_low"].values,
            df_env["raw_high"].values,
            lower_result,
            upper_result,
            supported_mask=df_env["supported"].values,
            figsize=(16, 6),
        )
    else:
        # Override mode: recreate a simple candidate plot using loess_smooth directly
        times = df_env["time_bin"].values.astype(float)
        supported = df_env["supported"].values
        tx = times[supported]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        titles = ["Lower curve (2.5%)", "Upper curve (97.5%)"]
        raw_cols = ["raw_low", "raw_high"]
        sel_frac = LOESS_FRAC_OVERRIDE
        cmap = plt.cm.cool
        fracs = sorted(LOESS_CANDIDATE_FRACS)
        for ax_i, (title, raw_col) in zip(axes, zip(titles, raw_cols)):
            raw_vals = df_env.loc[supported, raw_col].values
            ax_i.plot(tx, raw_vals, "ko", ms=4, label="Raw quantile")
            for fi, frac in enumerate(fracs):
                sm = loess_smooth(tx, raw_vals, frac)
                color = cmap(fi / max(len(fracs) - 1, 1))
                lw = 2.5 if frac == sel_frac else 1.0
                label = f"frac={frac}" + (" ← selected" if frac == sel_frac else "")
                ax_i.plot(tx, sm, color=color, lw=lw, label=label)
            ax_i.set_title(title, fontsize=14, fontweight="bold")
            ax_i.set_xlabel("Time bin (hpf)", fontsize=12)
            ax_i.legend(fontsize=7, ncol=2)
            ax_i.spines["top"].set_visible(False)
            ax_i.spines["right"].set_visible(False)
            ax_i.grid(True, alpha=0.3)
        fig.suptitle("LOESS Frac Selection (OVERRIDE)", y=1.01, fontsize=14)
        fig.tight_layout()

    fig.savefig(FIGURE_DIR / "loess_frac_selection.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: loess_frac_selection.png")

    # 3. WT/Het calibration — rename genotype → group for generic API
    cal_plot_df = cal_by_time.rename(columns={"genotype": "group"})
    expected_rate = 1 - (QUANTILE_HIGH - QUANTILE_LOW)
    fig, ax = plot_outside_rate_by_group(
        cal_plot_df,
        expected_rate=expected_rate,
        x_col="time_bin",
        y_col="outside_rate",
        group_col="group",
        colors=GENOTYPE_COLORS,
        figsize=(10, 5),
    )
    ax.set_title("WT & Het Calibration Check (diagnostic)", fontsize=14, fontweight="bold")
    fig.savefig(FIGURE_DIR / "wt_het_calibration.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: wt_het_calibration.png")

    # 4. Penetrance curves by category
    fig, ax = plot_penetrance_curves(
        pen_cat,
        x_col="time_bin",
        y_col="penetrance",
        se_col="se",
        group_col="group",
        colors=CATEGORY_COLORS,
        group_order=BROAD_CATEGORIES,
        title="Frame-level penetrance by broad category",
        figsize=FIGSIZE_CURVES,
    )
    fig.savefig(FIGURE_DIR / "penetrance_curves_by_category.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: penetrance_curves_by_category.png")

    # 5. Penetrance heatmap
    fig, ax = plot_penetrance_heatmap(
        pen_cat,
        x_col="time_bin",
        y_col="group",
        value_col="penetrance",
        group_order=BROAD_CATEGORIES,
        title="Penetrance heatmap (broad categories × time)",
        figsize=FIGSIZE_HEATMAP,
    )
    fig.savefig(FIGURE_DIR / "penetrance_heatmap_category.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: penetrance_heatmap_category.png")

    # 6. Embryo consistency histograms
    fig, axes = plot_embryo_consistency(
        emb_pen,
        group_col="group",
        value_col="frac_penetrant",
        colors=CATEGORY_COLORS,
        group_order=BROAD_CATEGORIES,
    )
    fig.savefig(FIGURE_DIR / "embryo_consistency_histograms.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: embryo_consistency_histograms.png")

    print("\n=== Done ===")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()
