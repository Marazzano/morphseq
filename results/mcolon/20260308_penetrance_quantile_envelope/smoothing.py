"""
smoothing.py — Generic LOESS and domain-specific quantile envelope frac selection.

Note on naming:
- ``loess_smooth`` is a generic math utility.
- ``select_quantile_curve_smoother`` is domain-specific (quantile envelope fitting);
  the name makes the abstraction honest and prevents misuse as a generic smoother selector.

Note on scope:
- ``compute_penetrance_by_time`` / ``mark_threshold_violations`` already exist in
  ``src/analyze/difference_detection/penetrance_threshold.py`` (embryo-level semantics).
  The functions here operate at the FRAME level and have different semantics; there is
  no collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Generic LOESS
# ---------------------------------------------------------------------------

def loess_smooth(
    x,
    y,
    frac: float,
    *,
    dropna: bool = True,
    require_sorted: bool = False,
) -> np.ndarray:
    """
    Locally weighted linear regression (LOESS) for 1-D data.

    Parameters
    ----------
    x, y : array-like
        Input coordinates. Must have the same length.
    frac : float
        Bandwidth fraction in (0, 1].
    dropna : bool
        If True (default), fit on non-NaN rows only; return NaN at NaN positions.
        If False, raise ValueError when any NaN is present.
    require_sorted : bool
        If True, raise ValueError if x is not monotone non-decreasing.

    Returns
    -------
    np.ndarray
        Same length as input; NaN where y was NaN (when dropna=True).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length; got {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError(f"Need at least 2 points; got {len(x)}")
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1]; got {frac}")
    if require_sorted and np.any(np.diff(x) < 0):
        raise ValueError("x is not monotone non-decreasing (require_sorted=True)")

    nan_mask = np.isnan(y) | np.isnan(x)
    if nan_mask.any():
        if not dropna:
            raise ValueError("NaN values present and dropna=False")
        xv, yv = x[~nan_mask], y[~nan_mask]
        smoothed_valid = _loess_core(xv, yv, frac)
        out = np.full(len(x), np.nan)
        out[~nan_mask] = smoothed_valid
        return out

    return _loess_core(x, y, frac)


def _loess_core(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
    """Tricube-weighted local linear regression; no NaN handling."""
    n = len(x)
    h = max(int(np.ceil(frac * n)), 2)
    smoothed = np.empty(n)
    for i in range(n):
        dists = np.abs(x - x[i])
        idx = np.argsort(dists)[:h]
        d_max = dists[idx].max()
        if d_max == 0:
            smoothed[i] = y[i]
            continue
        u = dists[idx] / d_max
        w = (1 - u**3)**3
        xi, yi, wi = x[idx], y[idx], w
        wsum = wi.sum()
        wmean_x = (wi * xi).sum() / wsum
        wmean_y = (wi * yi).sum() / wsum
        beta = (
            (wi * (xi - wmean_x) * (yi - wmean_y)).sum()
            / max((wi * (xi - wmean_x)**2).sum(), 1e-12)
        )
        smoothed[i] = wmean_y + beta * (x[i] - wmean_x)
    return smoothed


# ---------------------------------------------------------------------------
# Roughness helper
# ---------------------------------------------------------------------------

def curve_roughness(y) -> float:
    """
    Discrete 2nd-difference roughness: std(diff(y, n=2)).

    Parameters
    ----------
    y : array-like, length >= 3

    Returns
    -------
    float
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        raise ValueError(f"Need at least 3 points for roughness; got {len(y)}")
    return float(np.diff(y, n=2).std())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SmoothedCurveSelection:
    """
    Result bundle from ``select_quantile_curve_smoother``.

    Attributes
    ----------
    selected_frac : float
    smoothed_y : np.ndarray
        Full-length output (NaN for unsupported / NaN-input positions).
    candidate_curves : dict[float, np.ndarray]
        frac → smoothed array evaluated at valid (non-NaN) positions only.
    diagnostics : dict[float, dict]
        frac → structured result dict with keys:
          passed, failed_checks, roughness, roughness_threshold,
          sign_change_rate, residual_range, value_range, any_negative.
    used_fallback : bool
        True if no candidate frac passed and fallback_frac was used.
    """
    selected_frac: float
    smoothed_y: np.ndarray
    candidate_curves: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# Domain-specific smoother selection
# ---------------------------------------------------------------------------

def select_quantile_curve_smoother(
    x,
    y,
    *,
    candidate_fracs,
    nonnegative: bool = False,
    fallback_frac: float = 0.10,
) -> SmoothedCurveSelection:
    """
    Domain-specific LOESS frac selection for quantile envelope fitting.

    Sweeps ``candidate_fracs`` (smallest first) and selects the smallest frac
    whose smoothed curve passes all shape-stability checks.  Not a generic
    utility — the name makes the abstraction honest.

    Parameters
    ----------
    x, y : array-like
        Time-bin centres and raw quantile values. y may contain NaN for
        unsupported bins.
    candidate_fracs : sequence of float
        Fracs to try, each in (0, 1].
    nonnegative : bool
        If True, reject fracs that produce any negative smoothed values.
    fallback_frac : float
        Frac to use if no candidate passes; must be in (0, 1].

    Returns
    -------
    SmoothedCurveSelection
    """
    candidate_fracs = list(candidate_fracs)
    if not candidate_fracs:
        raise ValueError("candidate_fracs must not be empty")
    if any(not (0 < f <= 1) for f in candidate_fracs):
        raise ValueError("All candidate_fracs must be in (0, 1]")
    if not (0 < fallback_frac <= 1):
        raise ValueError(f"fallback_frac must be in (0, 1]; got {fallback_frac}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError(f"Need at least 3 points; got {len(x)}")

    valid_mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid_mask], y[valid_mask]
    value_range = float(yv.max() - yv.min()) if len(yv) > 1 else 1.0
    roughness_threshold = 0.05 * value_range

    diagnostics: dict[float, dict] = {}
    candidate_curves: dict[float, np.ndarray] = {}
    selected_frac = None
    selected_sm = None

    for frac in sorted(candidate_fracs):
        sm = _loess_core(xv, yv, frac)
        candidate_curves[frac] = sm

        failed_checks: list[str] = []

        # Check 1: non-negativity
        any_negative = bool(np.any(sm < 0))
        if nonnegative and any_negative:
            failed_checks.append("negativity")

        # Check 2: roughness
        roughness = curve_roughness(sm) if len(sm) >= 3 else 0.0
        if roughness > roughness_threshold:
            failed_checks.append("roughness")

        # Check 3: oscillation in residual
        residual = yv - sm
        residual_range = float(residual.max() - residual.min()) if len(residual) > 1 else 0.0
        sign_changes = int(np.sum(np.diff(np.sign(residual - residual.mean())) != 0))
        sign_change_rate = sign_changes / max(len(sm), 1)
        large_oscillation = (residual_range > 0.1 * value_range) and (sign_change_rate > 0.40)
        if large_oscillation:
            failed_checks.append("oscillation")

        passed = len(failed_checks) == 0
        diagnostics[frac] = {
            "passed": passed,
            "failed_checks": failed_checks,
            "roughness": roughness,
            "roughness_threshold": roughness_threshold,
            "sign_change_rate": sign_change_rate,
            "residual_range": residual_range,
            "value_range": value_range,
            "any_negative": any_negative,
        }

        if passed and selected_frac is None:
            selected_frac = frac
            selected_sm = sm

    used_fallback = selected_frac is None
    if used_fallback:
        selected_frac = fallback_frac
        if fallback_frac not in candidate_curves:
            selected_sm = _loess_core(xv, yv, fallback_frac)
            candidate_curves[fallback_frac] = selected_sm
        else:
            selected_sm = candidate_curves[fallback_frac]

    # Build full-length output (NaN for unsupported bins)
    smoothed_y = np.full(len(x), np.nan)
    smoothed_y[valid_mask] = selected_sm

    return SmoothedCurveSelection(
        selected_frac=selected_frac,
        smoothed_y=smoothed_y,
        candidate_curves=candidate_curves,
        diagnostics=diagnostics,
        used_fallback=used_fallback,
    )
