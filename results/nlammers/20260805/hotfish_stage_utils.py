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

Arrhenius expected-stage reference:  6 + (timepoint-6)*(0.055*temperature - 0.57)
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

#: Base directory for figure output.  Figures land in a sub-directory of the
#: notebook folder so they live next to the analysis, not in the data tree.
FIG_BASE = Path(__file__).resolve().parent / "figures"


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


def arrhenius_expected_stage(timepoint, temperature):
    """Linear-Arrhenius expected stage used as the reference diagonal/line."""
    timepoint = pd.to_numeric(timepoint, errors="coerce")
    temperature = pd.to_numeric(temperature, errors="coerce")
    return 6 + (timepoint - 6) * (0.055 * temperature - 0.57)


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
