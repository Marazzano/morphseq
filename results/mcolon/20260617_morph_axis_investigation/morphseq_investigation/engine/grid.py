"""``Grid`` construction (ontology §1b, Invariant #4/#9).

TASK_A builds the primitive the handoff left open:

- :func:`build_grid` — turn POOLED feature values into a :class:`Grid`
  (feature-unit axes, deterministic ``grid_id`` via ``engine.identifiers.make_grid_id``).

No coordinate frame, no inverse, no "canonical" basis anywhere (axes stay in
feature units per decision (b)); whitening (``pooled_mad_scaled``) only picks
*bounds and spacing*, never a change of basis.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .identifiers import make_grid_id
from .objects import Grid

_VALID_METHODS = (
    "pooled_min_max",
    "pooled_quantile",
    "pooled_mad_scaled",
    "fixed_bounds",
)


# --------------------------------------------------------------------------- #
# build_grid
# --------------------------------------------------------------------------- #
def _normalize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic param normalization (sorted keys, rounded floats).

    ``make_grid_id`` already canonicalizes params for hashing; this local
    normalization just guarantees ``resolution`` is a plain int and any
    bounds/quantile floats are plain Python floats before we do arithmetic
    with them (avoids e.g. numpy scalar surprises in axis construction).
    """
    normalized: dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (np.floating,)):
            normalized[key] = float(value)
        elif isinstance(value, (np.integer,)):
            normalized[key] = int(value)
        else:
            normalized[key] = value
    return normalized


def _resolution(params: Mapping[str, Any]) -> int:
    resolution = params.get("resolution")
    if resolution is None:
        raise ValueError("construction params must carry 'resolution' (cells per axis)")
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2 cells per axis, got {resolution}")
    return resolution


def _bounds_min_max(pooled_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pooled_values.min(axis=0), pooled_values.max(axis=0)


def _bounds_quantile(
    pooled_values: np.ndarray, params: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    q_low = float(params.get("q_low", 0.0))
    q_high = float(params.get("q_high", 1.0))
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"require 0 <= q_low < q_high <= 1, got q_low={q_low}, q_high={q_high}")
    lo = np.quantile(pooled_values, q_low, axis=0)
    hi = np.quantile(pooled_values, q_high, axis=0)
    return lo, hi


def _bounds_mad_scaled(
    pooled_values: np.ndarray, params: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """MAD picks bounds/spacing; the RETURNED bounds stay in feature units.

    We center on the pooled median and extend ``mad_multiplier`` scaled-MADs
    on each side (converted to feature units via the standard 1.4826 MAD->std
    scale factor) — this only affects how wide/narrow the feature-unit window
    is, never a change of basis (decision (b): no whitened coordinates are
    ever stored).
    """
    mad_multiplier = float(params.get("mad_multiplier", 5.0))
    median = np.median(pooled_values, axis=0)
    mad = np.median(np.abs(pooled_values - median), axis=0)
    # Guard degenerate (zero-MAD) features with the min/max range as a floor.
    scaled = mad * 1.4826
    value_range = pooled_values.max(axis=0) - pooled_values.min(axis=0)
    floor = np.where(value_range > 0, value_range / 2.0, 1.0)
    half_width = np.where(scaled > 0, scaled * mad_multiplier, floor)
    return median - half_width, median + half_width


def _bounds_fixed(
    n_features: int, params: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    bounds = params.get("bounds")
    if bounds is None:
        raise ValueError(
            "fixed_bounds requires params['bounds'] = a sequence of (lo, hi) per feature"
        )
    bounds = list(bounds)
    if len(bounds) != n_features:
        raise ValueError(
            f"fixed_bounds params['bounds'] must have one (lo, hi) pair per feature: "
            f"{len(bounds)} pairs vs {n_features} features"
        )
    lo = np.array([float(b[0]) for b in bounds], dtype=float)
    hi = np.array([float(b[1]) for b in bounds], dtype=float)
    return lo, hi


def build_grid(
    feature_names: Sequence[str],
    pooled_values: np.ndarray,
    fit_sample_ids: Sequence[str],
    method: str,
    params: Mapping[str, Any],
) -> Grid:
    """Build a :class:`Grid` from POOLED target+reference feature values.

    ``pooled_values`` is ``(n_samples, n_features)`` with columns ordered as
    ``feature_names`` (mirrors ``Distribution.feature_values``). Axes are
    ALWAYS in feature units (ontology §1b decision (b)) — ``method`` only
    changes how bounds/spacing are picked:

    - ``pooled_min_max``    — bounds = pooled per-feature min/max.
    - ``pooled_quantile``   — bounds = pooled per-feature quantiles
                               (``params['q_low']``, ``params['q_high']``).
    - ``pooled_mad_scaled`` — bounds sized from the pooled median absolute
                               deviation (``params['mad_multiplier']``), but
                               the stored ``axis_values`` remain feature-unit
                               (never whitened/z-scored coordinates).
    - ``fixed_bounds``      — explicit ``params['bounds']`` per feature.

    ``params['resolution']`` sets cells-per-axis for every method. Params are
    normalized deterministically before being handed to
    ``engine.identifiers.make_grid_id`` (which does its own canonicalization
    for hashing) together with the produced ``axis_values`` and the explicit
    ``fit_sample_ids`` (order-independent per TASK_0).
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"unknown construction_method {method!r}; must be one of {_VALID_METHODS}")

    feature_names = tuple(feature_names)
    fit_sample_ids = tuple(fit_sample_ids)
    pooled_values = np.asarray(pooled_values, dtype=float)
    if pooled_values.ndim != 2:
        raise ValueError(f"pooled_values must be 2-D (n_samples, n_features); got shape {pooled_values.shape}")
    if pooled_values.shape[1] != len(feature_names):
        raise ValueError(
            "pooled_values second axis must match feature_names length: "
            f"{pooled_values.shape[1]} columns vs {len(feature_names)} feature_names"
        )
    if pooled_values.shape[0] != len(fit_sample_ids):
        raise ValueError(
            "pooled_values first axis must match fit_sample_ids length: "
            f"{pooled_values.shape[0]} rows vs {len(fit_sample_ids)} fit_sample_ids"
        )

    normalized_params = _normalize_params(params)
    resolution = _resolution(normalized_params)

    if method == "pooled_min_max":
        lo, hi = _bounds_min_max(pooled_values)
    elif method == "pooled_quantile":
        lo, hi = _bounds_quantile(pooled_values, normalized_params)
    elif method == "pooled_mad_scaled":
        lo, hi = _bounds_mad_scaled(pooled_values, normalized_params)
    else:  # fixed_bounds
        lo, hi = _bounds_fixed(len(feature_names), normalized_params)

    axis_values = tuple(
        np.linspace(float(lo[j]), float(hi[j]), resolution) for j in range(len(feature_names))
    )

    grid_id = make_grid_id(
        feature_names=feature_names,
        construction_method=method,
        construction_params=normalized_params,
        axis_values=axis_values,
        fit_sample_ids=fit_sample_ids,
    )

    return Grid(
        grid_id=grid_id,
        feature_names=feature_names,
        axis_values=axis_values,
        construction_method=method,
        construction_params=normalized_params,
        fit_sample_ids=fit_sample_ids,
    )
