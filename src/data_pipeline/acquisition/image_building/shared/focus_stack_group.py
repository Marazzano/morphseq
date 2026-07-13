"""Shared focus-stack **group** processing for all microscope adapters.

This module owns the focus-projection *image math* that both Keyence and YX1 adapters
share.  Adapters own raw-data access, channel/Z selection, well/time grouping, and (for
Keyence) tile composition — they must NOT implement their own contrast normalization or
LoG-to-uint8 conversion.  See ``README.md`` in this directory for the ownership rationale
and the 2026-07 audit that motivated the boundary.

The public entry point is :func:`focus_stack_group`.  For one logical output frame the
caller provides a **stack group**:

* YX1: one raw ``(Z, Y, X)`` uint16 stack (a one-element group);
* Keyence: all raw tile stacks for one ``(well, channel, time)`` frame, each
  ``(Z, Y, X)`` uint16 ordered consistently with its ``z_indices``.

The implementation performs, exactly once for the whole group:

1. Compute deterministic intensity bounds across the *complete* group with an exact
   65,536-bin uint16 histogram and the 0.1 / 99.9 percentiles (no sampling).
2. Build a float32 LoG scoring tensor from raw values + shared bounds.  ``shared_clipped``
   (the conservative migration default) clips scores to ``[0, 1]``; ``shared_unclipped``
   keeps the affine map for evaluation only.
3. Run the shared LoG stacker once per tile and derive an integer stack-axis focus-index
   map via ``argmax`` over Z.
4. Gather focused pixel values from the **raw uint16 stack**, not the scoring tensor.
5. Apply one shared display transform (same bounds) to the focused raw pixels, clip to
   ``[0, 1]``, convert to uint8.
6. Return uint8 focused tiles, focus-index maps, and intensity/algorithm provenance.

Invariant: changing the display encoding must NOT change the focus-index map (focus
selection is decoupled from rendering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from data_pipeline.acquisition.image_building.shared.log_focus import LoG_focus_stacker
from data_pipeline.utils.cuda_diagnostics import resolve_device

# Method tag recorded in provenance for how the shared intensity bounds were computed.
BOUND_METHOD = "exact_uint16_histogram"
# uint16 has 65,536 distinct values; one histogram bin per value gives exact percentiles.
_UINT16_BINS = 1 << 16


@dataclass(frozen=True)
class FocusStackConfig:
    """Configuration for one focus-stack group operation.

    Attributes:
        low_percentile / high_percentile: intensity percentiles (over the whole group) that
            define the shared display / scoring bounds.  Defaults match legacy Build01.
        filter_size: LoG filter size passed to the shared stacker.
        scoring_mode: ``"shared_clipped"`` (conservative migration default — clips the
            affine-mapped scoring tensor to ``[0, 1]``, best matching the visually validated
            legacy result) or ``"shared_unclipped"`` (affine map with no clip, evaluation
            only — must never silently become the default).
        algorithm_version: opaque provenance tag for the construction algorithm.
    """

    low_percentile: float = 0.1
    high_percentile: float = 99.9
    filter_size: int = 3
    scoring_mode: Literal["shared_clipped", "shared_unclipped"] = "shared_clipped"
    algorithm_version: str = "shared_raw_gather_v1"


@dataclass(frozen=True)
class FocusStackResult:
    """One tile's focus-stack output.

    Attributes:
        projection_u8: focus-stacked 2D frame, ``uint8 (Y, X)``.
        focus_index_map: per-pixel STACK-AXIS OFFSET ``int32 (Y, X)`` — value ``k`` is the
            Z plane (0-based axis offset into the input stack) with the sharpest LoG
            response.  NOT an acquisition ``z_index`` label; the caller pairs it with an
            ordered ``z_indices`` array to recover labels.
    """

    projection_u8: np.ndarray
    focus_index_map: np.ndarray


@dataclass(frozen=True)
class FocusStackGroupResult:
    """Result for a whole stack group plus construction provenance.

    Attributes:
        tiles: one :class:`FocusStackResult` per input stack, in input order.
        intensity_lo / intensity_hi: the single shared uint16 bound pair used for every tile.
        config: the resolved :class:`FocusStackConfig` used.
    """

    tiles: tuple[FocusStackResult, ...]
    intensity_lo: int
    intensity_hi: int
    config: FocusStackConfig

    @property
    def bound_method(self) -> str:
        return BOUND_METHOD


def exact_uint16_histogram_bounds(
    stacks_zyx: Sequence[np.ndarray],
    *,
    low_percentile: float,
    high_percentile: float,
) -> tuple[int, int]:
    """Return the ``(lo, hi)`` uint16 intensity bounds across the *whole* group.

    Uses an exact 65,536-bin histogram (one bin per uint16 value) accumulated across every
    stack — every tile AND every Z plane share this ONE bound pair, so no tile/plane renders
    on a different tone curve. Deterministic and independent of tile ordering or batching (no
    random/strided sampling).

    Percentile convention is **nearest-rank**, byte-identical to the visually validated
    reference in ``results/mcolon/20260712_legacy_focus_stack_comparison/
    generate_improved_comparison.py`` (which produced the accepted ``shared_clean`` A02
    output at ``lo=5654, hi=56629``). For a population of ``n`` values the rank is
    ``floor((p/100)*(n-1)) + 1`` and the bound is the smallest value whose cumulative count
    reaches that rank (``searchsorted(..., side="left")``). Do NOT swap this for
    ``numpy.percentile`` — its interpolation lands on different uint16 levels and would break
    reproduction of the validated pixels.

    Args:
        stacks_zyx: the group's raw stacks; each must be uint16 (raw acquisition dtype).
        low_percentile / high_percentile: percentiles in ``[0, 100]``.

    Returns:
        ``(lo, hi)`` as Python ints.

    Raises:
        ValueError: on empty input, empty stacks, or a degenerate ``hi <= lo`` group (no
            contrast to stretch — a flat frame is a real problem, not something to paper over).
    """
    if not stacks_zyx:
        raise ValueError("focus_stack_group requires a non-empty stack group.")

    hist = np.zeros(_UINT16_BINS, dtype=np.int64)
    for stack in stacks_zyx:
        values = np.asarray(stack)
        if values.dtype != np.uint16:
            raise TypeError(
                f"focus_stack_group bounds require uint16 stacks (raw acquisition dtype); "
                f"got dtype {values.dtype}."
            )
        hist += np.bincount(values.reshape(-1), minlength=_UINT16_BINS).astype(np.int64)

    cumulative = np.cumsum(hist, dtype=np.int64)
    n = int(cumulative[-1])
    if n == 0:
        raise ValueError("focus_stack_group received only empty stacks.")

    low_rank = int(np.floor((low_percentile / 100.0) * (n - 1))) + 1
    high_rank = int(np.floor((high_percentile / 100.0) * (n - 1))) + 1
    lo = int(np.searchsorted(cumulative, low_rank, side="left"))
    hi = int(np.searchsorted(cumulative, high_rank, side="left"))
    if hi <= lo:
        raise ValueError(
            f"focus_stack_group derived degenerate shared bounds lo={lo}, hi={hi} — the group "
            "has no intensity contrast to stretch. Check the raw Z-stacks for this frame."
        )
    return lo, hi


def _affine_to_unit(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Map raw values through the shared bounds: ``(v - lo) / (hi - lo)``, float32.

    ``hi > lo`` is guaranteed by :func:`exact_uint16_histogram_bounds`. The result is NOT
    clipped here — clipping is the caller's explicit choice (scoring mode / display transform).
    """
    return (np.asarray(values, dtype=np.float32) - float(lo)) / float(hi - lo)


def _scoring_tensor(stacks_nzyx: np.ndarray, lo: int, hi: int, mode: str) -> np.ndarray:
    """Build the float32 LoG scoring input for the whole ``(N, Z, Y, X)`` group."""
    scored = _affine_to_unit(stacks_nzyx, lo, hi)
    if mode == "shared_clipped":
        return np.clip(scored, 0.0, 1.0)
    if mode == "shared_unclipped":
        return scored
    raise ValueError(f"Unknown scoring_mode {mode!r}.")


def _display_u8(focused_raw: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """One shared display transform: affine map through bounds → clip [0,1] → uint8.

    Uses ``np.rint`` (round-half-to-even) exactly as the validated reference does, so the
    emitted bytes match ``generate_improved_comparison.py``.
    """
    unit = np.clip(_affine_to_unit(focused_raw, lo, hi), 0.0, 1.0)
    return np.rint(unit * 255.0).astype(np.uint8)


def focus_stack_group(
    stacks_zyx: Sequence[np.ndarray],
    *,
    config: FocusStackConfig,
    device: str,
) -> FocusStackGroupResult:
    """Focus-stack a whole group of raw Z-stacks with ONE shared intensity mapping.

    See the module docstring for the 6-step contract.  All tiles in ``stacks_zyx`` share a
    single ``(lo, hi)`` bound pair derived once from the complete group; per-tile independent
    normalization is exactly the regression this API prevents.

    Args:
        stacks_zyx: the group's raw stacks. Each ``(Z, Y, X)`` uint16, and ALL stacks in the
            group must share one ``(Z, Y, X)`` — the group is composed into a single
            ``(N, Z, Y, X)`` batch so no per-tile code path (and thus no per-tile tone curve)
            exists. Callers group by ``(well, channel, time)`` where tiles are same-geometry.
        config: focus-stack configuration (bounds percentiles, filter, scoring mode).
        device: torch device preference for the LoG convolutions (``"cpu"`` / ``"cuda"``);
            resolved through :func:`resolve_device` so ``"cuda"`` falls back to CPU cleanly.

    Returns:
        :class:`FocusStackGroupResult` with one :class:`FocusStackResult` per input stack
        (input order preserved) plus the shared bounds and resolved config.

    The input stacks are never mutated. Bounds come from the whole-group histogram and every
    tile's projection is a pure function of the raw values + the single shared bounds, so any
    future GPU-sized sub-batching of the LoG pass cannot change bounds or output bytes.
    """
    if not stacks_zyx:
        raise ValueError("focus_stack_group requires a non-empty stack group.")
    shapes = []
    for i, stack in enumerate(stacks_zyx):
        arr = np.asarray(stack)
        if arr.ndim != 3:
            raise ValueError(
                f"focus_stack_group stack {i} must be (Z, Y, X); got shape {arr.shape}."
            )
        shapes.append(arr.shape)
    # The group is processed as ONE composed (N, Z, Y, X) batch so every tile/Z plane shares
    # the same LoG scoring pass and the same tone curve — nothing tile-specific can slip in.
    # That composition requires a common (Z, Y, X) across the group; callers group by
    # (well, channel, time), where tiles are same-geometry acquisitions.
    if len(set(shapes)) != 1:
        raise ValueError(
            "focus_stack_group composes the group into one (N, Z, Y, X) batch and requires a "
            f"uniform (Z, Y, X) across all stacks; got shapes {shapes}."
        )

    resolved_device = resolve_device(device)

    lo, hi = exact_uint16_histogram_bounds(
        stacks_zyx,
        low_percentile=config.low_percentile,
        high_percentile=config.high_percentile,
    )

    # (1)+(2): compose the whole group and build ONE shared scoring tensor (N, Z, Y, X).
    raw_nzyx = np.stack([np.asarray(s) for s in stacks_zyx], axis=0)
    scoring = _scoring_tensor(raw_nzyx, lo, hi, config.scoring_mode)

    # (3): one LoG pass over the whole batch; focus-index map = argmax over the Z axis (axis 1).
    _, abs_log = LoG_focus_stacker(
        scoring.astype(np.float32), config.filter_size, resolved_device
    )
    abs_log_np = abs_log.cpu().numpy() if torch.is_tensor(abs_log) else np.asarray(abs_log)
    focus_index_maps = np.argmax(abs_log_np, axis=1).astype(np.int32)  # (N, Y, X)

    # (4): gather focused pixels from the RAW stacks (not the scoring tensor).
    focused_raw = np.take_along_axis(
        raw_nzyx, focus_index_maps[:, np.newaxis, :, :], axis=1
    )[:, 0]  # (N, Y, X)

    # (5): one shared display transform for the whole group → uint8.
    projections_u8 = _display_u8(focused_raw, lo, hi)  # (N, Y, X)

    tiles = tuple(
        FocusStackResult(
            projection_u8=projections_u8[i],
            focus_index_map=focus_index_maps[i],
        )
        for i in range(raw_nzyx.shape[0])
    )

    return FocusStackGroupResult(
        tiles=tiles,
        intensity_lo=int(lo),
        intensity_hi=int(hi),
        config=config,
    )
