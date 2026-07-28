"""Portable staging/plotting helpers for the 20260805 hotfish temperature-vs-stage figures.

Refactored from results/nlammers/20260528/figure_utils.py and
make_tempo_noise_bootstrap_se.py, with three changes:

  1. Paths are driven by a single configurable ROOT (env var MORPHSEQ_DATA_ROOT,
     else auto-detected) instead of hard-coded /Users/nick/... Mac paths.
  2. A single INCLUDE_19C toggle controls both which rows are kept and which
     figure sub-directory is written, so the two versions never overwrite.
  3. The staging provenance is documented inline (see STAGING NOTES below).

STAGING NOTES
-------------
morph stage  = column `mdl_stage_hpf`.
    Produced upstream by morph_stage_model.joblib (sklearn Pipeline:
    PolynomialFeatures -> LinearRegression) applied to morph-VAE PCA coords.
    Already precomputed in the cached hotfish tables; we do NOT re-fit it here.
    (A separate `spline_stage_hpf` = nearest WT-spline knot exists but is NOT
    what the 20260528 figures use.)  `morph_dist_spline` = Euclidean distance to
    the WT reference spline, used as the "morphological noise" axis.

transcriptional stage = column `pseudostage` (aka seq_stage_hpf).
    Produced upstream by a Hooke regression (bead_expt_linear) ->
    time_predictions.csv, keyed by sequencing `sample`, then merged onto morph
    via morphseq_metadata.csv.  Not re-fit here.

Arrhenius expected-stage reference
    The true reference (per hf2_seq_morph_arrhenius.ipynb, results/nlammers/20250319)
    is an Arrhenius developmental-rate model:
        stage(t, T) = A * t * exp(-E / (R * T_kelvin))
    with R = 8.314, E = 65.2 kJ/mol (Toulany et al. 2023), T_kelvin = temperature_C
    + 273.15, and the single scale A fit by least-squares to the 19/25/28.5C cohorts.
    ``fit_arrhenius`` returns A; ``arrhenius_expected_stage`` evaluates the model.
    (A separate *linear approximation* 6 + (t-6)*(0.055*T - 0.57) exists in the
    20260504 seq notebooks; it is NOT the Arrhenius reference and is not used here.)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Set to True to keep the 19C cohort in every analysis and write to a separate
#: figure directory.  Default False (matches the 20260528 pass).  The notebook
#: overrides this module-level default by calling ``set_include_19c``.
INCLUDE_19C = False

#: Temperatures excluded when INCLUDE_19C is False.
COLD_TEMPERATURES = (19.0,)


def _default_data_root() -> Path:
    """Locate the morphseq data cache root across machines.

    Priority:
      1. $MORPHSEQ_DATA_ROOT
      2. Nick's Mac cache (the 20260528 build target)
      3. cluster data tree
      4. a local ./data folder next to this file (fallback)
    """
    env = os.environ.get("MORPHSEQ_DATA_ROOT")
    if env:
        return Path(env)
    candidates = [
        Path("/Users/nick/Projects/data/morphseq/results/20260528"),
        Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/results/20260528"),
        Path(__file__).resolve().parent / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the first (Mac) path so error messages are informative.
    return candidates[0]


CACHE_DIR = _default_data_root()

#: Base directory for figure output.  Figures are written to the shared data
#: results tree (env MORPHSEQ_FIG_ROOT overrides), NOT next to the notebook.
FIG_BASE = Path(
    os.environ.get("MORPHSEQ_FIG_ROOT",
                   "/Users/nick/Projects/data/morphseq/results/20260805")
)


def fig_dir() -> Path:
    """Figure output directory, keyed on the 19C toggle so runs don't clobber."""
    sub = "with_19C" if INCLUDE_19C else "no_19C"
    return FIG_BASE / sub


def set_include_19c(value: bool) -> None:
    """Notebook entry point: flip the 19C toggle for the whole module."""
    global INCLUDE_19C
    INCLUDE_19C = bool(value)


def excluded_temperatures() -> tuple[float, ...]:
    return () if INCLUDE_19C else COLD_TEMPERATURES


def drop_cold(df: pd.DataFrame, temperature_col: str = "temperature") -> pd.DataFrame:
    """Drop 19C rows unless INCLUDE_19C is set."""
    excl = excluded_temperatures()
    if not excl or temperature_col not in df.columns:
        return df.copy()
    temp = pd.to_numeric(df[temperature_col], errors="coerce")
    return df.loc[~temp.isin(excl)].copy()


# --------------------------------------------------------------------------- #
# Column conventions (single source of truth for the two stage estimates)
# --------------------------------------------------------------------------- #

MORPH_STAGE_COL = "mdl_stage_hpf"       # morphology-inferred stage (hpf)
SEQ_STAGE_COL = "pseudostage"           # transcription-inferred stage (hpf)
MORPH_DIST_COL = "morph_dist_spline"    # distance from WT spline (morph "noise")

MORPH_STAGE_LABEL = "morphology-inferred stage (hpf)"
SEQ_STAGE_LABEL = "transcription-inferred stage (hpf)"

#: Control temperature used to standardize the transcriptional vs morphological
#: stage offset (the 28.5C cohort is the physiological control).
OFFSET_REF_TEMP = 28.5


def seq_morph_offset(df, *, seq_col=SEQ_STAGE_COL, morph_col=MORPH_STAGE_COL,
                     temperature_col="temperature", ref_temp=OFFSET_REF_TEMP):
    """Mean (seq - morph) stage offset at the control temperature.

    Subtracting this from the transcriptional stage puts the control cohort on
    the y=x line, standardizing the two staging axes.  Returns 0.0 if the
    control cohort or either column is missing (no-op standardization).
    """
    if seq_col not in df.columns or morph_col not in df.columns:
        return 0.0
    ref = df.loc[pd.to_numeric(df[temperature_col], errors="coerce") == ref_temp]
    diff = pd.to_numeric(ref[seq_col], errors="coerce") - pd.to_numeric(ref[morph_col], errors="coerce")
    diff = diff.dropna()
    return float(diff.mean()) if len(diff) else 0.0


# Arrhenius developmental-rate model (hf2_seq_morph_arrhenius.ipynb).
#   stage = A * t * exp(-E / (R * T_kelvin))
# BOTH A and E are fit by scipy.least_squares (source used x0=[1, 200], with E in
# J/mol).  E is NOT pinned to a literature value -- it is a free parameter; the
# effective activation energy that best describes the data is what we plot.
ARRHENIUS_R = 8.314          # gas constant, J/(mol*K)
ARRHENIUS_E0 = 200.0         # E initial guess, J/mol (source x0)
ARRHENIUS_A0 = 1.0           # A initial guess (source x0)
ARRHENIUS_FIT_TEMPS = (25.0, 28.5, 32.0)  # cohorts used to fit A and E (no 19C)
KELVIN = 273.15


def _arrhenius_rate(temperature_c, E, R=ARRHENIUS_R):
    """exp(-E / (R * T_kelvin)); E in J/mol (matches source parameterization)."""
    t_k = pd.to_numeric(temperature_c, errors="coerce") + KELVIN
    return np.exp(-E / (R * t_k))


def fit_arrhenius(df, stage_col, *, temperature_col="temperature",
                  timepoint_col="timepoint", fit_temps=ARRHENIUS_FIT_TEMPS):
    """Fit BOTH A and E in stage = A * t * exp(-E/(R*T)) by least-squares,
    using only the control cohorts (default 25/28.5/32C).

    Replicates ``least_squares`` on ``A*t*exp(-E/RT) - stage`` from the source
    notebook (x0=[1, 200]).  Returns (A, E) as floats, or (nan, nan) if there is
    nothing to fit.
    """
    from scipy.optimize import least_squares

    d = df.loc[pd.to_numeric(df[temperature_col], errors="coerce").isin(fit_temps)]
    t = pd.to_numeric(d[timepoint_col], errors="coerce")
    stage = pd.to_numeric(d[stage_col], errors="coerce")
    t_k = pd.to_numeric(d[temperature_col], errors="coerce") + KELVIN
    mask = np.isfinite(t) & np.isfinite(stage) & np.isfinite(t_k)
    t, stage, t_k = t[mask].to_numpy(), stage[mask].to_numpy(), t_k[mask].to_numpy()
    if t.size == 0:
        return float("nan"), float("nan")

    def residual(params):
        A, E = params
        return A * t * np.exp(-E / (ARRHENIUS_R * t_k)) - stage

    res = least_squares(residual, x0=[ARRHENIUS_A0, ARRHENIUS_E0])
    return float(res.x[0]), float(res.x[1])


def arrhenius_expected_stage(timepoint, temperature, params):
    """Evaluate the fitted Arrhenius model stage = A * t * exp(-E/(R*T)).

    ``params`` is the (A, E) tuple returned by ``fit_arrhenius``.
    """
    A, E = params
    timepoint = pd.to_numeric(timepoint, errors="coerce")
    return A * timepoint * _arrhenius_rate(temperature, E)


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

TEMP_CMAP = "RdBu_r"
TEMP_VMIN = 24
TEMP_CENTER = 28.5
TEMP_VMAX = 35

TIMEPOINT_MARKERS = {24.0: "o", 30.0: "s", 36.0: "^"}

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260528  # keep the original seed for reproducibility


def temperature_norm() -> mpl.colors.Normalize:
    """Diverging normalize centred on the 28.5C control, spanning the cold/hot
    extremes so 19C (when included) still maps to a distinct blue."""
    vmin = 18 if INCLUDE_19C else TEMP_VMIN
    return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=TEMP_CENTER, vmax=TEMP_VMAX)


def set_light_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    """Write {name}.png and {name}.pdf into the toggle-keyed figure dir."""
    out = fig_dir()
    out.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out / f"{name}{suffix}", dpi=300)
    return out / f"{name}.png"


# --------------------------------------------------------------------------- #
# Scatter helpers (temperature color + collection-timepoint marker)
# --------------------------------------------------------------------------- #

def _marker_for_timepoint(timepoint) -> str:
    try:
        return TIMEPOINT_MARKERS.get(float(timepoint), "o")
    except (TypeError, ValueError):
        return "o"


def _timepoint_label(timepoint) -> str:
    try:
        return f"{float(timepoint):g} hpf"
    except (TypeError, ValueError):
        return str(timepoint)


def add_timepoint_legend(ax, timepoints, title: str = "collection") -> None:
    unique = sorted(pd.Series(timepoints).dropna().unique(), key=lambda v: float(v))
    handles = [
        Line2D([0], [0], marker=_marker_for_timepoint(tp), linestyle="none",
               markerfacecolor="white", markeredgecolor="#333333",
               markeredgewidth=0.8, markersize=6, label=_timepoint_label(tp))
        for tp in unique
    ]
    if handles:
        ax.legend(handles=handles, title=title, frameon=False, loc="best",
                  fontsize=8, title_fontsize=8)


def temperature_timepoint_scatter(
    ax, x, y, temp, timepoint, *,
    add_colorbar=False, add_legend=True, colorbar_label="temperature (C)", **kwargs,
):
    """Scatter colored by temperature, marker-coded by collection timepoint."""
    plot_df = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
        "temp": pd.to_numeric(temp, errors="coerce"),
        "timepoint": timepoint,
    }).dropna(subset=["x", "y", "temp", "timepoint"])

    norm = temperature_norm()
    cmap = plt.get_cmap(TEMP_CMAP)
    scatter_kwargs = {"s": 38, "alpha": 0.9, "edgecolor": "black",
                      "linewidth": 0.25, "rasterized": True, **kwargs}
    for tp, group in plot_df.groupby("timepoint", sort=True):
        ax.scatter(group["x"], group["y"], c=group["temp"], cmap=cmap, norm=norm,
                   marker=_marker_for_timepoint(tp), **scatter_kwargs)

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    if add_colorbar:
        cb = plt.colorbar(mappable, ax=ax)
        cb.set_label(colorbar_label)
    if add_legend:
        add_timepoint_legend(ax, plot_df["timepoint"])
    return mappable


def temperature_scatter_fixed_marker(
    ax, x, y, temp, marker, *, s=90, alpha=0.95, linewidth=0.6,
    facecolor_alpha=1.0, edgecolor="black", **kwargs,
):
    """Scatter colored by temperature with a single fixed marker symbol.

    Used to overlay a second stage estimate (e.g. transcriptional) that must
    stay temperature-colored like the morph markers, distinguished ONLY by its
    symbol.  Set ``facecolor_alpha`` < 1 for a lighter "open-ish" look while
    still carrying the temperature color (unlike a pure open marker, which loses
    it).  Returns the ScalarMappable for an optional shared colorbar.
    """
    plot_df = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
        "temp": pd.to_numeric(temp, errors="coerce"),
    }).dropna()
    norm = temperature_norm()
    cmap = plt.get_cmap(TEMP_CMAP)
    colors = cmap(norm(plot_df["temp"].to_numpy()))
    colors[:, 3] = facecolor_alpha
    ax.scatter(plot_df["x"], plot_df["y"], facecolors=colors, marker=marker,
               s=s, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha,
               **kwargs)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    return mappable


def add_stage_type_legend(ax, entries, *, title="stage estimate", loc="best"):
    """Legend keying symbol -> stage type (e.g. triangle=morph, diamond=seq),
    drawn in neutral grey so it reads as a symbol key, not a temperature."""
    handles = [
        Line2D([0], [0], marker=m, linestyle="none", markerfacecolor="#cccccc",
               markeredgecolor="#333333", markeredgewidth=0.8, markersize=8, label=lab)
        for m, lab in entries
    ]
    ax.legend(handles=handles, title=title, frameon=False, loc=loc,
              fontsize=8, title_fontsize=8)


def add_identity(ax, x=None, y=None, **kwargs):
    """Dashed y=x reference line spanning the data (or current axis) range."""
    if x is None or y is None:
        lo, hi = ax.get_xlim()
        yo, yh = ax.get_ylim()
        mn, mx = min(lo, yo), max(hi, yh)
    else:
        vals = np.asarray(pd.concat([pd.Series(x), pd.Series(y)], ignore_index=True), dtype=float)
        vals = vals[np.isfinite(vals)]
        mn, mx = float(vals.min()), float(vals.max())
    pad = 0.03 * (mx - mn)
    style = {"color": "#555555", "linestyle": "--", "linewidth": 1, **kwargs}
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], **style)


# --------------------------------------------------------------------------- #
# Bootstrap standard errors
# --------------------------------------------------------------------------- #

def bootstrap_mean_se(values: pd.Series, rng: np.random.Generator, n_bootstrap: int = BOOTSTRAP_N) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    return float(np.std(np.mean(samples, axis=1), ddof=1))


def bootstrap_std_se(values: pd.Series, rng: np.random.Generator, n_bootstrap: int = BOOTSTRAP_N) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    return float(np.std(np.std(samples, axis=1, ddof=1), ddof=1))


def cohort_stage_summary(
    df: pd.DataFrame, *, stage_cols=(MORPH_STAGE_COL, SEQ_STAGE_COL),
    group_cols=("temperature", "timepoint"), id_col="snip_id",
    n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-(temperature, timepoint) mean, sd, and bootstrap SE-of-sd for each
    stage column present in ``df``.  Cohorts of size < 2 get NaN SEs.

    Only stage columns actually present are summarized, so a morph-only table
    (no ``pseudostage``) still works.
    """
    rng = np.random.default_rng(seed)
    present = [c for c in stage_cols if c in df.columns]
    rows = []
    for keys, group in df.groupby(list(group_cols), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(group[id_col].count()) if id_col in group else len(group)
        for col in present:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1))
            row[f"{col}_std_boot_se"] = bootstrap_std_se(vals, rng, n_bootstrap)
            row[f"{col}_mean_boot_se"] = bootstrap_mean_se(vals, rng, n_bootstrap)
        rows.append(row)
    return pd.DataFrame(rows)


def timepoint_average_variability_bootstrap(
    df: pd.DataFrame, value_col: str, *,
    temperature_col: str = "temperature", timepoint_col: str = "timepoint",
    n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-temperature variability = mean over collection timepoints of the
    within-cohort sd, with a bootstrap SE.  Matches the 20260528 method exactly.
    """
    rng = np.random.default_rng(seed)
    rows = []
    data = drop_cold(df, temperature_col)
    for temperature, temp_df in data.groupby(temperature_col, sort=True):
        groups = [
            pd.to_numeric(g[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            for _, g in temp_df.groupby(timepoint_col, sort=True)
        ]
        groups = [a for a in groups if a.size >= 2]
        if not groups:
            continue
        observed = np.array([np.std(a, ddof=1) for a in groups], dtype=float)
        boot = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            boot[i] = np.mean([np.std(rng.choice(a, size=a.size, replace=True), ddof=1) for a in groups])
        rows.append({
            "temperature": temperature,
            "variability_mean": float(np.mean(observed)),
            "variability_boot_se": float(np.std(boot, ddof=1)),
            "n_timepoints": len(groups),
            "n": int(sum(a.size for a in groups)),
        })
    return pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)
