"""Structured id constructors — compose from typed parts, NEVER parse back.

Mirrors ``src/data_pipeline/shared/identifiers`` doctrine (ontology #2): ids are
*rendered* from stored typed parts; nothing ever string-archaeologies the result.

``make_grid_id`` LOCKS the hash the handoff left open (ontology line 510, handoff
line 162). The property that MUST hold: ``same grid_id <=> same evaluation
coordinates`` (raster comparability). See :func:`make_grid_id`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

# Fixed float precision for deterministic param canonicalization. Params like
# bandwidth / bounds are rounded here so cosmetically-different floats that mean
# the same grid do not fork the id.
_PARAM_FLOAT_DECIMALS = 9
# axis_values are hashed from their raw bytes at this dtype so the id tracks the
# ACTUAL evaluation coordinates (not a rounded shadow of them).
_AXIS_DTYPE = np.float64


def make_distribution_id(scope_id: str, time_bin: Any, role: str) -> str:
    """``make_distribution_id("b9d2", 30, "reference") -> "b9d2_30hpf_reference"``.

    ``time_bin`` is rendered with an ``hpf`` unit suffix when numeric (the repo's
    binning grain); non-numeric bins render as their string form. Parts are the
    stored truth on :class:`~engine.objects.Distribution`; this string is only a
    rendering.
    """
    if isinstance(time_bin, (int, float)) and not isinstance(time_bin, bool):
        bin_token = f"{int(time_bin) if float(time_bin).is_integer() else time_bin}hpf"
    else:
        bin_token = str(time_bin)
    return f"{scope_id}_{bin_token}_{role}"


def make_sample_set_id(distribution_id: str, name: str) -> str:
    """``make_sample_set_id("b9d2_30hpf_reference", "peak_0")``
    ``-> "b9d2_30hpf_reference__peak_0"``.

    Double-underscore separator (ontology §2) so target/reference ``peak_0``s
    never collide in exports/joins.
    """
    return f"{distribution_id}__{name}"


def _canonicalize_params(params: Mapping[str, Any]) -> str:
    """Deterministic string form of construction params.

    Sorted keys, floats rounded to a fixed precision, stringified with
    ``sort_keys`` JSON so two callers passing equivalent params get one id.
    """

    def _norm(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, _PARAM_FLOAT_DECIMALS)
        if isinstance(value, (np.floating,)):
            return round(float(value), _PARAM_FLOAT_DECIMALS)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, Mapping):
            return {str(k): _norm(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_norm(v) for v in value]
        return value

    normalized = {str(k): _norm(v) for k, v in dict(params).items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _hash_axis_values(axis_values: Sequence[np.ndarray]) -> str:
    """Hash the ACTUAL produced cell coordinates.

    This is what makes ``grid_id <=> evaluation coordinates``: the id depends on
    the real axis coordinates, not on the fit ids alone. Per-axis lengths are
    folded in so two axes cannot alias by concatenation.
    """
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(f"n_axes={len(axis_values)}".encode())
    for i, axis in enumerate(axis_values):
        arr = np.ascontiguousarray(np.asarray(axis, dtype=_AXIS_DTYPE))
        hasher.update(f"|axis{i}:len={arr.shape[0]}|".encode())
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def make_grid_id(
    feature_names: Sequence[str],
    construction_method: str,
    construction_params: Mapping[str, Any],
    axis_values: Sequence[np.ndarray],
    fit_sample_ids: Sequence[str] = (),
) -> str:
    """Deterministic, stable ``grid_id`` — LOCKED (ontology §1b, TASK_0).

    Hash inputs (per §1b):
      - ordered ``feature_names`` (order is identity — PC1,PC2 != PC2,PC1),
      - ``construction_method``,
      - normalized ``construction_params`` (sorted keys, rounded floats),
      - a hash of the PRODUCED ``axis_values`` (the actual cell coordinates),
      - order-independent hash of ``fit_sample_ids``.

    Guarantees (both tested in ``test_identifiers.py``):
      - ``same grid_id <=> same evaluation coordinates``. Same pool in a
        different row order -> same id (``fit_sample_ids`` sorted). Changed
        feature *values* under the same fit ids -> different id (axis_values
        differ). No stateful registry.
      - Uses ``blake2b`` over a canonical byte serialization — NOT Python's
        salted ``hash()`` — so the id string is stable across processes.
    """
    sorted_fit = sorted(str(s) for s in fit_sample_ids)
    payload = "\x1f".join(
        [
            "feature_names=" + "\x1e".join(str(f) for f in feature_names),
            "construction_method=" + str(construction_method),
            "construction_params=" + _canonicalize_params(construction_params),
            "axis_values=" + _hash_axis_values(axis_values),
            "fit_sample_ids=" + "\x1e".join(sorted_fit),
        ]
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"grid_{digest}"
