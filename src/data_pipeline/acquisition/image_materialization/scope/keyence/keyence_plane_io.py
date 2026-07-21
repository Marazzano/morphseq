"""Single read boundary for Keyence raw plane TIFFs.

Every Keyence plane read goes through :func:`read_keyence_plane` so the two raw-data realities
below are handled in exactly one place instead of being re-discovered at each ``skio.imread``
call site (there are three: the FF/z_stack reads in ``materialize_well_keyence`` and the
sample-frame read in ``build_keyence_stitch_map``).

1. **8-bit acquisitions.** The 2023-era BZ-X exports are ``uint8`` 480x640; everything from
   2023-08 onward is ``uint16``. ``focus_stack_group`` hard-requires ``uint16`` (its bounds pass
   builds an exact 65536-bin histogram), so an 8-bit plane raises ``TypeError``. We promote with a
   plain ``astype(np.uint16)`` — a *widening cast*, NOT a rescale.

   Why not rescale (``* 257`` to map 255 -> 65535)? Because it changes nothing downstream and costs
   fidelity. ``focus_stack_group`` derives percentile bounds ``(lo, hi)`` from the data and then
   affine-stretches ``[lo, hi] -> [0, 255]``; a constant factor cancels exactly in that
   normalization. Verified on a real 8-bit stack: plain cast and ``*257`` produce **byte-identical**
   uint8 projections (max abs diff 0). The cast is preferable because rescaling would leave 256
   populated levels spread across 65536 bins, making the histogram sparse for no benefit.

   Note this does not recover information: 8-bit data has 256 intensity levels regardless. The
   resulting projections are self-normalized like every other experiment, so they are comparable in
   *display* terms, but they are genuinely coarser in quantization depth.

2. **Zero-byte / unreadable planes.** A full scan of the 2023 cohort (~3.4M TIFFs) found exactly one
   zero-byte file. Left unhandled it aborts the entire experiment: the reader hits it, the metadata
   scraper reports a misleading "No <Data> XML block" (the block is fine in every intact file — it
   sits ~8.5KB from EOF, well inside the 32KB window), and ~405k good planes are lost with it. A
   single corrupt plane is a data defect, not a reason to discard an experiment, so callers may opt
   into ``missing_ok`` and receive ``None`` to skip that plane.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import skimage.io as skio

log = logging.getLogger(__name__)


def read_keyence_plane(path: str | Path, *, missing_ok: bool = False) -> np.ndarray | None:
    """Read one Keyence plane TIFF, promoting 8-bit acquisitions to ``uint16``.

    Args:
        path: Path to the plane TIFF.
        missing_ok: When True, return ``None`` for a zero-byte/unreadable file instead of raising,
            so one corrupt plane cannot take down an otherwise-good experiment.

    Returns:
        ``(Y, X)`` array, ``uint16`` for any integer input; ``None`` only when ``missing_ok`` and the
        file is unreadable.
    """
    p = Path(path)

    try:
        if p.stat().st_size == 0:
            raise ValueError(f"zero-byte TIFF: {p}")
        image = np.asarray(skio.imread(str(p)))
    except Exception as exc:
        if missing_ok:
            log.warning("read_keyence_plane: skipping unreadable plane %s: %s", p, exc)
            return None
        raise

    # Widening cast only. uint16 passes through untouched, so 16-bit experiments are bit-for-bit
    # unaffected by this boundary.
    if image.dtype == np.uint16:
        return image
    if image.dtype == np.uint8:
        return image.astype(np.uint16)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        if info.min >= 0 and info.max <= np.iinfo(np.uint16).max:
            return image.astype(np.uint16)

    raise TypeError(
        f"Unsupported Keyence plane dtype {image.dtype} for {p}. Expected uint8/uint16 raw "
        "acquisition data."
    )
