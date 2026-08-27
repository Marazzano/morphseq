"""The background null: a well's annuli, pooled over time, for one channel.

WHY POOLED AND NOT PER-EMBRYO. A single embryo's annulus is a few hundred to a few thousand pixels --
the least biased estimate available, and the noisiest. The background we are estimating is a property
of the WELL (medium autofluorescence, plastic, lamp, detector offset), not of the embryo, so pooling
across every embryo-time in the well trades a little bias for a great deal of stability.

Pooling is by PIXEL, not by embryo. An embryo whose annulus was mostly eaten by neighbor exclusion
contributes proportionally little, automatically -- which is the correct weighting and the concrete
argument against median-of-medians, where a crowded well's tiny noisy annuli would count as much as
clean ones.

WHY A MODE. The pooled annulus distribution is not symmetric noise. Two contaminants push it right:
the well rim autofluoresces (~1019 DN vs ~575 mid-well), and residual embryo halo bleeds into the
inner edge of the ring. Neighbor exclusion removes other fish but CANNOT remove the rim, which is not
a mask. A mean is dragged by both contaminants; a median partially. The mode is the value the camera
reports where nothing is -- exactly what should be subtracted -- and it ignores a smaller right-hand
lobe entirely. That robustness is the third independent argument for it.

WHY THIS IS A FEATURE, NOT AN EXTRACTION. object_extraction extracts objects; deriving a statistical
null over a population of them is a feature. Keeping the estimator here is what lets it change
without re-reading a single pixel -- the extraction rows are raw evidence and never move.

RAW STAYS RAW. Nothing here rewrites an extracted column. Corrected values are explicitly named
additions, and they are NOT clipped at zero: clipping hides over-subtraction, and a negative
corrected value is diagnostically useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from data_pipeline.object_extraction.channel_intensity import (
    HIST_BIN_WIDTH_DN,
    HIST_MIN_DN,
    HIST_N_BINS,
)

# The estimator's identity, carried on every null row. A later change ADDS a value rather than
# silently redefining what an old column meant -- the same discipline as ORIENTATION_POLICY in the
# snip path. Without this, a table written under one estimator and read under another looks
# perfectly well-formed.
NULL_ESTIMATOR = "pooled_annulus_mode_v1"

# A pooled null built from very few pixels is not a null, it is noise with a name. Below this the
# row is emitted with null_valid=False rather than dropped, so the reason is visible downstream.
MINIMUM_POOLED_PIXELS = 1000


class ChannelIntensityNullError(ValueError):
    """A null could not be estimated, or would have been silently meaningless."""


@dataclass(frozen=True)
class PooledNull:
    """The background estimate for one (well, source product), plus the evidence for judging it."""

    counts: np.ndarray
    mode_dn: float
    median_dn: float
    p16_dn: float
    p84_dn: float
    robust_sigma_dn: float
    pooled_px: int
    n_rows_pooled: int
    valid: bool


def _bin_centers() -> np.ndarray:
    """Value at the middle of each bin. Quantiles are reported at bin resolution, not interpolated --
    pretending to sub-bin precision would overstate what the stored evidence supports."""
    return HIST_MIN_DN + (np.arange(HIST_N_BINS) + 0.5) * HIST_BIN_WIDTH_DN


def pool_histograms(hist_counts: Iterable[Sequence[int]]) -> np.ndarray:
    """Sum per-embryo histograms into one.

    THE OPERATION THE WHOLE DESIGN RESTS ON. Summation is exact, commutative, and order-independent,
    so the pooled histogram of a well is indistinguishable from having histogrammed every annulus
    pixel at once. No summary statistic has that property, which is why the extraction stage emits
    counts rather than means.
    """
    pooled = np.zeros(HIST_N_BINS, dtype=np.int64)
    for counts in hist_counts:
        arr = np.asarray(counts, dtype=np.int64)
        if arr.shape != (HIST_N_BINS,):
            raise ChannelIntensityNullError(
                f"pool_histograms: expected {HIST_N_BINS} bins, got {arr.shape}. Rows written under "
                "a different bin spec cannot be pooled -- check intensity_recipe_version."
            )
        pooled += arr
    return pooled


def _quantile_dn(counts: np.ndarray, q: float) -> float:
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    target = q * total
    idx = int(np.searchsorted(np.cumsum(counts), target, side="left"))
    return float(_bin_centers()[min(idx, HIST_N_BINS - 1)])


def estimate_null(
    hist_counts: Iterable[Sequence[int]],
    *,
    minimum_pooled_pixels: int = MINIMUM_POOLED_PIXELS,
) -> PooledNull:
    """Pool annulus histograms and estimate the background mode.

    The mode is the argmax bin's center. Reported at bin resolution deliberately: with 32-DN bins on
    a background whose body spans a few hundred DN, that is honest precision, and interpolating a
    peak position would invent detail the stored evidence does not carry.

    ``robust_sigma_dn`` is (p84 - p16) / 2 -- a spread that survives the right tail the mean does
    not. It is NOT multiplied by any Gaussian-consistency factor: that would be a distributional
    assumption, and it belongs downstream where someone is deliberately making it.
    """
    # Materialize once: the argument is an iterable, and counting it after pooling would exhaust a
    # generator and silently report zero rows.
    as_list = list(hist_counts)
    counts = pool_histograms(as_list)
    pooled_px = int(counts.sum())
    n_rows = len(as_list)

    if pooled_px == 0:
        return PooledNull(
            counts=counts, mode_dn=float("nan"), median_dn=float("nan"),
            p16_dn=float("nan"), p84_dn=float("nan"), robust_sigma_dn=float("nan"),
            pooled_px=0, n_rows_pooled=0, valid=False,
        )

    centers = _bin_centers()
    p16 = _quantile_dn(counts, 0.16)
    p84 = _quantile_dn(counts, 0.84)
    return PooledNull(
        counts=counts,
        mode_dn=float(centers[int(np.argmax(counts))]),
        median_dn=_quantile_dn(counts, 0.5),
        p16_dn=p16,
        p84_dn=p84,
        robust_sigma_dn=float((p84 - p16) / 2.0),
        pooled_px=pooled_px,
        n_rows_pooled=n_rows,
        valid=pooled_px >= minimum_pooled_pixels,
    )


def estimate_well_null(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The null row for one (well, source product), with drift measured rather than assumed.

    POOLING OVER TIME ASSUMES THE BACKGROUND IS STATIONARY, AND IT MAY NOT BE. Photobleaching, lamp
    drift, and media evaporation all move it, and that drift is confounded with developmental stage
    -- an embryo imaged later is both older and sitting in a different background.

    So the pooled-over-time null is emitted as the primary estimate (it maximizes pixel count), and
    ``null_mode_dn_by_time`` plus ``null_drift_dn`` are emitted ALONGSIDE it. Both come free from the
    same histograms. If drift is small relative to the 1-vs-2-copy separation, the pooled null is
    vindicated by evidence rather than by assertion; if it is not, the per-timepoint null is already
    computed and the estimator can switch without touching a pixel. That option is the concrete
    payoff for storing histograms instead of means.
    """
    if not rows:
        raise ChannelIntensityNullError(
            "estimate_well_null: no rows. A well with no measured annuli has no background estimate; "
            "emit no null row rather than a fabricated one."
        )

    pooled = estimate_null(r["annulus_hist_counts"] for r in rows)

    by_time: dict[int, float] = {}
    for time_index in sorted({int(r["time_index"]) for r in rows}):
        at_t = [r["annulus_hist_counts"] for r in rows if int(r["time_index"]) == time_index]
        null_t = estimate_null(at_t)
        if null_t.valid:
            by_time[time_index] = null_t.mode_dn

    modes = list(by_time.values())
    return {
        "null_estimator": NULL_ESTIMATOR,
        "null_mode_dn": pooled.mode_dn,
        "null_median_dn": pooled.median_dn,
        "null_p16_dn": pooled.p16_dn,
        "null_p84_dn": pooled.p84_dn,
        "null_robust_sigma_dn": pooled.robust_sigma_dn,
        "null_pooled_px": pooled.pooled_px,
        "null_n_rows_pooled": len(rows),
        "null_valid": bool(pooled.valid),
        # The stationarity evidence. Empty when no single timepoint cleared the pixel floor, which
        # is itself worth seeing rather than papering over with a zero.
        "null_mode_dn_by_time": {int(k): float(v) for k, v in by_time.items()},
        "null_drift_dn": float(max(modes) - min(modes)) if len(modes) > 1 else 0.0,
        "null_n_timepoints": len(by_time),
        "hist_bin_width_dn": HIST_BIN_WIDTH_DN,
        "hist_n_bins": HIST_N_BINS,
    }
