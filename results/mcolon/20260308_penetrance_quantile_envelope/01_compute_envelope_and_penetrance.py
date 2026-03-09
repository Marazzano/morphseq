"""
WT Quantile Envelope Penetrance Pipeline
========================================

Design:
- Primary presentation outputs are embryo-level: one embryo-bin summary per time bin.
- Raw WT embryo-bin quantiles are used for classification.
- LOESS smoothing is kept for diagnostics and plotting only.
- Frame-level calibration and scatter remain available as diagnostics.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from data_loading import load_data, split_by_genotype
from config import (
    BROAD_CATEGORIES,
    CATEGORY_COL,
    CATEGORY_COLORS,
    DPI,
    EMBRYO_CALL_MODE,
    EMBRYO_BIN_AGG,
    EMBRYO_COL,
    FRAME_DIAGNOSTIC_CALL_MODE,
    FIGSIZE_CURVES,
    FIGSIZE_DIAGNOSTIC,
    FIGSIZE_HEATMAP,
    FIGURE_DIR,
    GENOTYPE_COL,
    GENOTYPE_COLORS,
    HET_GENOTYPE,
    LOESS_CANDIDATE_FRACS,
    LOESS_FALLBACK_FRAC,
    LOESS_FRAC_OVERRIDE,
    METRIC_NAME,
    METRIC_NONNEG,
    MIN_WT_EMBRYOS_PER_BIN,
    MIN_WT_FRAMES_PER_BIN,
    QUANTILE_HIGH,
    QUANTILE_LOW,
    PRESENTATION_CURVE_FRAC,
    PRESENTATION_CURVE_MODE,
    PRESENTATION_CURVE_SHOW_POINTS,
    PRESENTATION_CURVE_SMOOTH_SE,
    ROBUST_SMOOTHING_MIN_POINTS,
    ROBUST_SMOOTHING_MIN_RESID_FRACTION,
    ROBUST_SMOOTHING_SIGMA_THRESHOLD,
    ROBUST_SMOOTHING_WINDOW,
    TABLE_DIR,
    TIME_COL,
    UPPER_BOUND_ONLY,
    WT_GENOTYPE,
)
from penetrance_plots import (
    plot_embryo_consistency,
    plot_outside_rate_by_group,
    plot_penetrance_curves,
    plot_penetrance_heatmap,
    plot_quantile_smoother_selection,
    plot_scatter_and_penetrance,
    plot_wt_envelope_diagnostic,
    summarize_binary_penetrance,
)
from smoothing import detect_curve_inliers, loess_smooth, select_quantile_curve_smoother


def aggregate_embryo_bins(df: pd.DataFrame, agg: str = EMBRYO_BIN_AGG) -> pd.DataFrame:
    """
    Collapse frame rows into one embryo_id × time_bin summary row.

    The main presentation path uses the median metric within each embryo/time bin.
    """
    agg_fn = {"median": "median", "mean": "mean"}.get(agg)
    if agg_fn is None:
        raise ValueError(f"agg must be 'median' or 'mean', got {agg!r}")

    meta_cols = [
        c for c in [GENOTYPE_COL, CATEGORY_COL, "cluster_subcategories", "experiment_id", "experiment_date"]
        if c in df.columns
    ]
    group_cols = [EMBRYO_COL, "time_bin"]

    agg_spec: dict[str, str] = {
        TIME_COL: "median",
        METRIC_NAME: agg_fn,
    }
    for col in meta_cols:
        agg_spec[col] = "first"

    embryo_bins = (
        df.groupby(group_cols, as_index=False)
        .agg(agg_spec)
        .sort_values([EMBRYO_COL, "time_bin"])
        .reset_index(drop=True)
    )

    frame_counts = (
        df.groupby(group_cols)
        .size()
        .rename("n_frames_in_bin")
        .reset_index()
    )
    embryo_bins = embryo_bins.merge(frame_counts, on=group_cols, how="left")
    return embryo_bins


def compute_raw_wt_quantiles(
    wt_df: pd.DataFrame,
    *,
    min_units: int,
    unit_label: str,
) -> pd.DataFrame:
    """
    Compute raw WT quantiles for the rows provided.

    The caller decides whether rows represent frames or embryo-bin summaries.
    """
    rows = []
    for tb, grp in wt_df.groupby("time_bin"):
        vals = grp[METRIC_NAME].dropna().to_numpy()
        n_units = len(vals)
        n_embryos = grp[EMBRYO_COL].nunique()
        supported = n_units >= min_units

        raw_low = float(np.quantile(vals, QUANTILE_LOW)) if n_units > 0 else np.nan
        raw_high = float(np.quantile(vals, QUANTILE_HIGH)) if n_units > 0 else np.nan
        rows.append(
            {
                "time_bin": tb,
                "raw_low": raw_low,
                "raw_high": raw_high,
                "n_wt_units": n_units,
                "n_wt_embryos": n_embryos,
                "supported": supported,
            }
        )

    df_q = pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)
    print(
        f"\nRaw quantiles ({unit_label}): {df_q['supported'].sum()}/{len(df_q)} bins "
        f"supported (>= {min_units} {unit_label}s)"
    )
    return df_q


def validate_envelope(lower_sm, upper_sm, supported_mask, metric_nonneg=True):
    """Ensure lower < upper and lower >= 0 (if nonnegative)."""
    lower_sm = lower_sm.copy()
    upper_sm = upper_sm.copy()

    valid = np.where(supported_mask)[0]

    if metric_nonneg:
        neg = valid[lower_sm[valid] < 0]
        if len(neg) > 0:
            print(f"  WARNING: Clipping {len(neg)} bins where smoothed lower < 0")
            lower_sm[neg] = 0.0

    crossed = valid[lower_sm[valid] >= upper_sm[valid]]
    if len(crossed) > 0:
        print(f"  WARNING: {len(crossed)} bins where lower >= upper; nudging upper up")
        for i in crossed:
            upper_sm[i] = lower_sm[i] + 1e-6

    return lower_sm, upper_sm


def compute_wt_envelope(
    wt_df: pd.DataFrame,
    *,
    min_units: int,
    unit_label: str,
):
    """
    Compute raw WT quantiles and a smoothed diagnostic envelope for the given rows.
    """
    print("\n=== Step 1: Raw WT quantiles ===")
    df_q = compute_raw_wt_quantiles(wt_df, min_units=min_units, unit_label=unit_label)

    times = df_q["time_bin"].to_numpy(dtype=float)
    supported = df_q["supported"].to_numpy(dtype=bool)

    lower_result = upper_result = None
    lower_fit_mask = supported.copy()
    upper_fit_mask = supported.copy()
    lower_fit_diag: dict = {}
    upper_fit_diag: dict = {}

    if LOESS_FRAC_OVERRIDE is not None:
        print(f"\n=== Step 2: Smoothing (OVERRIDE frac={LOESS_FRAC_OVERRIDE}) ===")
        frac_low = frac_high = LOESS_FRAC_OVERRIDE
        valid = ~np.isnan(df_q["raw_low"].to_numpy())
        sm_low = np.full(len(times), np.nan)
        sm_high = np.full(len(times), np.nan)
        sm_low[valid] = loess_smooth(times[valid], df_q.loc[valid, "raw_low"].to_numpy(), frac_low)
        sm_high[valid] = loess_smooth(times[valid], df_q.loc[valid, "raw_high"].to_numpy(), frac_high)
    else:
        print("\n=== Step 2: Smoothing frac selection by shape stability ===")
        if supported.sum() >= max(ROBUST_SMOOTHING_MIN_POINTS, 3):
            supported_idx = np.where(supported)[0]
            low_inliers, lower_fit_diag = detect_curve_inliers(
                times[supported],
                df_q.loc[supported, "raw_low"].to_numpy(),
                window=ROBUST_SMOOTHING_WINDOW,
                min_points=ROBUST_SMOOTHING_MIN_POINTS,
                sigma_threshold=ROBUST_SMOOTHING_SIGMA_THRESHOLD,
                min_resid_fraction=ROBUST_SMOOTHING_MIN_RESID_FRACTION,
            )
            high_inliers, upper_fit_diag = detect_curve_inliers(
                times[supported],
                df_q.loc[supported, "raw_high"].to_numpy(),
                window=ROBUST_SMOOTHING_WINDOW,
                min_points=ROBUST_SMOOTHING_MIN_POINTS,
                sigma_threshold=ROBUST_SMOOTHING_SIGMA_THRESHOLD,
                min_resid_fraction=ROBUST_SMOOTHING_MIN_RESID_FRACTION,
            )
            lower_fit_mask = np.zeros(len(df_q), dtype=bool)
            upper_fit_mask = np.zeros(len(df_q), dtype=bool)
            lower_fit_mask[supported_idx] = low_inliers
            upper_fit_mask[supported_idx] = high_inliers

        lower_result = select_quantile_curve_smoother(
            times,
            df_q["raw_low"].to_numpy(),
            candidate_fracs=LOESS_CANDIDATE_FRACS,
            nonnegative=METRIC_NONNEG,
            fallback_frac=LOESS_FALLBACK_FRAC,
            fit_mask=lower_fit_mask,
        )
        upper_result = select_quantile_curve_smoother(
            times,
            df_q["raw_high"].to_numpy(),
            candidate_fracs=LOESS_CANDIDATE_FRACS,
            nonnegative=METRIC_NONNEG,
            fallback_frac=LOESS_FALLBACK_FRAC,
            fit_mask=upper_fit_mask,
        )
        lower_result.fit_diagnostics = lower_fit_diag
        upper_result.fit_diagnostics = upper_fit_diag

        for name, result in [("lower", lower_result), ("upper", upper_result)]:
            fallback = " [FALLBACK]" if result.used_fallback else ""
            print(f"  [{name}] selected frac={result.selected_frac}{fallback}")
            excluded_bins = [int(v) for v in result.fit_diagnostics.get("excluded_x", [])]
            if excluded_bins:
                print(f"    excluded outlier bins from smoothing: {excluded_bins}")
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
    df_env["lower_frac"] = frac_low
    df_env["upper_frac"] = frac_high
    df_env["smooth_fit_low"] = lower_fit_mask
    df_env["smooth_fit_high"] = upper_fit_mask
    df_env["smooth_excluded_low"] = supported & ~lower_fit_mask
    df_env["smooth_excluded_high"] = supported & ~upper_fit_mask
    return df_env, lower_result, upper_result


def mark_penetrant(
    df: pd.DataFrame,
    df_env: pd.DataFrame,
    *,
    call_mode: str = "raw",
    raw_lower_col: str = "raw_low",
    raw_upper_col: str = "raw_high",
    smoothed_lower_col: str = "smoothed_low",
    smoothed_upper_col: str = "smoothed_high",
    lower_excluded_col: str = "smooth_excluded_low",
    upper_excluded_col: str = "smooth_excluded_high",
    upper_bound_only: bool = UPPER_BOUND_ONLY,
) -> pd.DataFrame:
    """
    Mark rows as penetrant using raw, smoothed, or hybrid thresholds.
    """
    valid_modes = {"raw", "smoothed", "hybrid"}
    if call_mode not in valid_modes:
        raise ValueError(f"call_mode must be one of {sorted(valid_modes)}, got {call_mode!r}")

    lookup_cols = [
        raw_lower_col,
        raw_upper_col,
        smoothed_lower_col,
        smoothed_upper_col,
        "supported",
    ]
    for optional_col in [lower_excluded_col, upper_excluded_col]:
        if optional_col in df_env.columns:
            lookup_cols.append(optional_col)

    env_lookup = df_env.set_index("time_bin")[lookup_cols]

    out = df.copy()
    out = out.join(env_lookup, on="time_bin")

    low_excluded = out.get(lower_excluded_col, pd.Series(False, index=out.index)).fillna(False).astype(bool)
    high_excluded = out.get(upper_excluded_col, pd.Series(False, index=out.index)).fillna(False).astype(bool)
    supported = out["supported"].fillna(False).astype(bool)

    if call_mode == "raw":
        active_low = out[raw_lower_col]
        active_high = out[raw_upper_col]
        low_source = np.where(supported, "raw", "unsupported")
        high_source = np.where(supported, "raw", "unsupported")
    elif call_mode == "smoothed":
        active_low = out[smoothed_lower_col]
        active_high = out[smoothed_upper_col]
        low_source = np.full(len(out), "smoothed", dtype=object)
        high_source = np.full(len(out), "smoothed", dtype=object)
    else:
        use_smoothed_low = (~supported) | low_excluded
        use_smoothed_high = (~supported) | high_excluded
        active_low = np.where(use_smoothed_low, out[smoothed_lower_col], out[raw_lower_col])
        active_high = np.where(use_smoothed_high, out[smoothed_upper_col], out[raw_upper_col])
        low_source = np.where(use_smoothed_low, "smoothed", "raw")
        high_source = np.where(use_smoothed_high, "smoothed", "raw")

    out["threshold_low"] = active_low
    out["threshold_high"] = active_high
    out["threshold_source_low"] = low_source
    out["threshold_source_high"] = high_source
    out["threshold_call_mode"] = call_mode

    if upper_bound_only:
        outside = out[METRIC_NAME] > out["threshold_high"]
        valid_call = ~pd.isna(out["threshold_high"])
    else:
        outside = (out[METRIC_NAME] < out["threshold_low"]) | (out[METRIC_NAME] > out["threshold_high"])
        valid_call = ~(pd.isna(out["threshold_low"]) | pd.isna(out["threshold_high"]))

    out["penetrant"] = np.where(valid_call, np.where(outside, 1.0, 0.0), np.nan)
    return out


def compute_penetrance_by_group_and_time(
    df: pd.DataFrame,
    group_col: str,
    *,
    unit_col: str | None,
    count_col_name: str,
) -> pd.DataFrame:
    """
    Compute penetrance from binary calls, optionally collapsing repeated rows per unit.
    """
    if unit_col is None:
        rows = []
        for (group, tb), grp in df.groupby([group_col, "time_bin"]):
            valid = grp.dropna(subset=["penetrant"])
            if valid.empty:
                continue
            unit_flags = valid["penetrant"].astype(float)
            n_units = len(unit_flags)
            if n_units == 0:
                continue
            penetrance = float(unit_flags.mean())
            se = np.sqrt(penetrance * (1 - penetrance) / n_units)
            rows.append(
                {
                    "group": group,
                    "time_bin": tb,
                    "penetrance": penetrance,
                    count_col_name: n_units,
                    "n_penetrant": int(unit_flags.sum()),
                    "se": se,
                    "q25": float(np.quantile(unit_flags, 0.25)),
                    "q75": float(np.quantile(unit_flags, 0.75)),
                }
            )
        return pd.DataFrame(rows).sort_values(["group", "time_bin"]).reset_index(drop=True)

    summary = summarize_binary_penetrance(
        df,
        group_col=group_col,
        bin_col="time_bin",
        penetrant_col="penetrant",
        unit_col=unit_col,
        value_scale=1.0,
    ).rename(columns={"n_units": count_col_name})
    return summary.sort_values(["group", "time_bin"]).reset_index(drop=True)


def compute_penetrance_consistency(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Per-embryo fraction of supported bins that are penetrant.
    """
    rows = []
    for embryo, grp in df.groupby(EMBRYO_COL):
        group = grp[group_col].iloc[0]
        valid = grp["penetrant"].dropna()
        n_total_bins = len(valid)
        n_penetrant_bins = int(valid.sum()) if n_total_bins > 0 else 0
        frac_penetrant = n_penetrant_bins / n_total_bins if n_total_bins > 0 else np.nan
        rows.append(
            {
                "embryo_id": embryo,
                "group": group,
                "n_total_bins": n_total_bins,
                "n_penetrant_bins": n_penetrant_bins,
                "frac_penetrant": frac_penetrant,
            }
        )

    return pd.DataFrame(rows).sort_values(["group", "embryo_id"]).reset_index(drop=True)


def compute_calibration(df: pd.DataFrame, *, count_col_name: str):
    """
    Compute outside-envelope rate for WT and Het, overall and by time bin.
    """
    cal_df = df[df[GENOTYPE_COL].isin([WT_GENOTYPE, HET_GENOTYPE])].copy()

    overall_rows = []
    time_rows = []
    for geno, grp in cal_df.groupby(GENOTYPE_COL):
        valid = grp["penetrant"].dropna()
        n = len(valid)
        overall_rows.append({"genotype": geno, "outside_rate": valid.mean(), count_col_name: n})

        for tb, tgrp in grp.groupby("time_bin"):
            tv = tgrp["penetrant"].dropna()
            time_rows.append({"genotype": geno, "time_bin": tb, "outside_rate": tv.mean(), count_col_name: len(tv)})

    overall = pd.DataFrame(overall_rows)
    by_time = pd.DataFrame(time_rows).sort_values(["genotype", "time_bin"]).reset_index(drop=True)
    return overall, by_time


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
    embryo_bin_df = mark_penetrant(
        embryo_bin_df,
        embryo_env,
        call_mode=EMBRYO_CALL_MODE,
    )
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

    frame_diag_df = mark_penetrant(
        frame_df,
        frame_env,
        call_mode=FRAME_DIAGNOSTIC_CALL_MODE,
    )
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

    penetrance_variants = [
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

    for variant in penetrance_variants:
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

    geno_order = [
        WT_GENOTYPE,
        HET_GENOTYPE,
        "cep290_homozygous",
    ]
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

    scatter_variants = [
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

    for variant in scatter_variants:
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
