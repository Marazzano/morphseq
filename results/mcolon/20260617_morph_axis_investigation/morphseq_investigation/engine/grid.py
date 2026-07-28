"""``Grid``/``DensityGrid`` construction + evaluation (ontology §1b, Invariant #4/#9).

TASK_A builds the two primitives the handoff left open:

- :func:`build_grid` — turn POOLED feature values into a :class:`Grid`
  (feature-unit axes, deterministic ``grid_id`` via ``engine.identifiers.make_grid_id``).
- :func:`evaluate_density` — KDE arbitrary samples ON a Grid's ``axis_values``,
  producing a :class:`DensityGrid` tagged with the SAME ``grid_id``.

No coordinate frame, no inverse, no "canonical" basis anywhere (axes stay in
feature units per decision (b)); whitening (``pooled_mad_scaled``) only picks
*bounds and spacing*, never a change of basis. The 1-D case is not special
cased: a single-feature Grid + ``evaluate_density`` IS a KDE strip.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .identifiers import make_grid_id
from .objects import DensityGrid, Grid

# Reuse the existing dimension-agnostic KDE kernel (core/bandwidth_tuning.py) —
# it evaluates a dense isotropic Gaussian KDE from precomputed squared
# distances and does not assume 2-D, only ``precompute_squared_distances``
# (in the same module) hardcodes shape (n, 2). We roll our own N-D squared
# distance so build_grid/evaluate_density work for any feature count,
# including the 1-D strip case TASK_D needs.
from ..core.bandwidth_tuning import evaluate_isotropic_gaussian_kde_from_dist2

_VALID_METHODS = (
    "pooled_min_max",
    "pooled_quantile",
    "pooled_mad_scaled",
    "fixed_bounds",
)

# The robust-bound policy used by the Valley analysis before its final
# visualization-only square-box expansion. Comparison grids apply this policy
# independently per native feature axis; they deliberately do not square or
# normalize the numeric spans.
VALLEY_SCAN_PERCENTILE = 2.0
VALLEY_RANGE_PADDING_FRACTION = 0.35
VALLEY_MARGIN_FRACTION = 0.06


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


def derive_robust_pooled_axes(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    resolution: int = 40,
    scan_percentile: float = VALLEY_SCAN_PERCENTILE,
    range_padding_fraction: float = VALLEY_RANGE_PADDING_FRACTION,
    margin_fraction: float = VALLEY_MARGIN_FRACTION,
) -> tuple[np.ndarray, ...]:
    """Feature-blind numeric shared axes under the Valley robust-bound policy.

    This primitive knows only two aligned numeric matrices. For each column it
    pools both populations, takes the ``scan_percentile`` and complementary
    percentiles, pads that robust range by ``range_padding_fraction``, then
    adds ``margin_fraction`` of the padded span. This is the pre-squaring
    numeric policy in ``plotting.modal_distribution_plotting.density_box``.

    The same coordinate count is used independently on every axis. Numeric
    spans remain in native units and are never equalized or normalized.
    Feature names, sample IDs, distributions, and catalogs belong to the
    semantic composition layer and are intentionally absent from this API.
    """
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("shared-grid inputs must both be 2-D")
    if left.shape[1] != right.shape[1]:
        raise ValueError("shared-grid inputs must have the same feature count")
    if left.shape[1] == 0 or left.shape[0] + right.shape[0] == 0:
        raise ValueError("shared-grid inputs must contain at least one feature and one row")
    pooled = np.concatenate((left, right), axis=0)
    if not np.all(np.isfinite(pooled)):
        raise ValueError("shared-grid values must all be finite")

    resolution = int(resolution)
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    scan = float(scan_percentile)
    padding = float(range_padding_fraction)
    margin = float(margin_fraction)
    if not 0.0 <= scan < 50.0:
        raise ValueError("scan_percentile must satisfy 0 <= value < 50")
    if padding < 0.0 or margin < 0.0:
        raise ValueError("padding and margin fractions must be nonnegative")

    robust_lo, robust_hi = np.percentile(pooled, [scan, 100.0 - scan], axis=0)
    robust_span = robust_hi - robust_lo
    primary_pad = robust_span * padding
    primary_pad = np.where(primary_pad > 0.0, primary_pad, 1.0)
    padded_lo = robust_lo - primary_pad
    padded_hi = robust_hi + primary_pad
    outer_margin = (padded_hi - padded_lo) * margin
    lo = padded_lo - outer_margin
    hi = padded_hi + outer_margin
    return tuple(
        np.linspace(float(lo[j]), float(hi[j]), resolution)
        for j in range(pooled.shape[1])
    )


def build_shared_grid(
    feature_names: Sequence[str],
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_sample_ids: Sequence[str],
    right_sample_ids: Sequence[str],
    *,
    resolution: int = 40,
    scan_percentile: float = VALLEY_SCAN_PERCENTILE,
    range_padding_fraction: float = VALLEY_RANGE_PADDING_FRACTION,
    margin_fraction: float = VALLEY_MARGIN_FRACTION,
) -> Grid:
    """Attach ordered-feature/sample identity to feature-blind shared axes.

    Numeric derivation is wholly delegated to ``derive_robust_pooled_axes``.
    This higher-level constructor owns semantic validation, provenance, and
    deterministic identity only.
    """
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise ValueError("shared-grid inputs must both be 2-D")
    if left_values.shape[1] != right_values.shape[1]:
        raise ValueError("shared-grid inputs must have the same feature count")
    feature_names = tuple(feature_names)
    if len(feature_names) != left_values.shape[1]:
        raise ValueError("ordered feature names must match the numeric column count")
    fit_sample_ids = (*tuple(left_sample_ids), *tuple(right_sample_ids))
    if len(fit_sample_ids) != left_values.shape[0] + right_values.shape[0]:
        raise ValueError("sample IDs must align one-to-one with the pooled numeric rows")

    axis_values = derive_robust_pooled_axes(
        left_values,
        right_values,
        resolution=resolution,
        scan_percentile=scan_percentile,
        range_padding_fraction=range_padding_fraction,
        margin_fraction=margin_fraction,
    )
    params = {
        "resolution": int(resolution),
        "scan_percentile": float(scan_percentile),
        "range_padding_fraction": float(range_padding_fraction),
        "margin_fraction": float(margin_fraction),
    }
    grid_id = make_grid_id(
        feature_names=feature_names,
        construction_method="valley_robust_pooled_bounds",
        construction_params=params,
        axis_values=axis_values,
        fit_sample_ids=fit_sample_ids,
    )
    return Grid(
        grid_id=grid_id,
        feature_names=feature_names,
        axis_values=axis_values,
        construction_method="valley_robust_pooled_bounds",
        construction_params=params,
        fit_sample_ids=fit_sample_ids,
    )


# --------------------------------------------------------------------------- #
# evaluate_density
# --------------------------------------------------------------------------- #
def _grid_mesh_points(axis_values: Sequence[np.ndarray]) -> np.ndarray:
    """Flatten a Grid's per-axis coordinates into ``(n_cells, n_axes)`` mesh points.

    ``indexing="ij"`` so the flattened row-major order matches
    ``density.reshape(tuple(len(a) for a in axis_values))`` — i.e. axis i
    varies slowest in the flat ordering, matching numpy's default C order for
    an array of that shape.
    """
    if len(axis_values) == 1:
        return axis_values[0].reshape(-1, 1)
    mesh = np.meshgrid(*axis_values, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def _cell_volume(axis_values: Sequence[np.ndarray]) -> float:
    """Product of per-axis cell spacing — the N-D generalization of ``cell_area``."""
    volume = 1.0
    for axis in axis_values:
        if len(axis) > 1:
            volume *= float(axis[1] - axis[0])
    return volume


def _resolve_bandwidth(bandwidth_spec: Mapping[str, Any] | float) -> float:
    """Accept either a bare scalar bandwidth or a ``{"bandwidth": h}``-style mapping.

    Only isotropic scalar bandwidths are supported (matches the reused kernel,
    ``evaluate_isotropic_gaussian_kde_from_dist2``); no separate bandwidth path
    is added here.
    """
    if isinstance(bandwidth_spec, Mapping):
        if "bandwidth" not in bandwidth_spec:
            raise ValueError("bandwidth_spec mapping must carry a 'bandwidth' key")
        return float(bandwidth_spec["bandwidth"])
    return float(bandwidth_spec)


def evaluate_density(
    grid: Grid,
    sample_values: np.ndarray,
    bandwidth_spec: Mapping[str, Any] | float,
) -> DensityGrid:
    """KDE ``sample_values`` ON ``grid.axis_values`` -> a :class:`DensityGrid`.

    Tagged with the SAME ``grid_id`` as ``grid`` (never re-derived) — this is
    the primitive both peak runs and 1-D KDE strips share (ontology §1b): a
    1-feature ``Grid`` + ``evaluate_density`` IS a KDE strip, no special case.

    Reuses ``core.bandwidth_tuning.evaluate_isotropic_gaussian_kde_from_dist2``
    (dimension-agnostic dense isotropic Gaussian KDE from squared distances)
    rather than adding a second KDE code path; only the squared-distance
    computation here is written fresh since the module's own
    ``precompute_squared_distances`` hardcodes 2-D points.

    Never interpolates a density across grids — to compare on a shared grid,
    re-evaluate the samples on that grid.
    """
    sample_values = np.asarray(sample_values, dtype=float)
    if sample_values.ndim != 2:
        raise ValueError(f"sample_values must be 2-D (n_samples, n_features); got shape {sample_values.shape}")
    if sample_values.shape[1] != len(grid.feature_names):
        raise ValueError(
            "sample_values second axis must match grid.feature_names length: "
            f"{sample_values.shape[1]} columns vs {len(grid.feature_names)} feature_names"
        )

    mesh_points = _grid_mesh_points(grid.axis_values)  # (n_cells, n_axes)
    bandwidth = _resolve_bandwidth(bandwidth_spec)

    if sample_values.shape[0] == 0:
        density_flat = np.zeros(mesh_points.shape[0], dtype=float)
    else:
        # (n_cells, n_samples) squared Euclidean distances, N-D generalization
        # of core.bandwidth_tuning.precompute_squared_distances (2-D only).
        diff = mesh_points[:, None, :] - sample_values[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        cell_volume = _cell_volume(grid.axis_values)
        density_flat = evaluate_isotropic_gaussian_kde_from_dist2(
            dist2,
            bandwidth,
            cell_area=cell_volume,
            normalize_grid=True,
        )

    shape = tuple(len(a) for a in grid.axis_values)
    density = density_flat.reshape(shape)

    return DensityGrid(
        grid_id=grid.grid_id,
        feature_names=grid.feature_names,
        density=density,
    )
