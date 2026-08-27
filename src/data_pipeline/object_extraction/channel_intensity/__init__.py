"""Native-grid fluorescence evidence, per embryo-time.

Channel-generic on purpose: GFP will arrive and nothing here is RFP-specific.

This package extracts RAW, POOLABLE evidence. It does not estimate a background, subtract one, or
normalize -- those are well-grain operations over a population of embryos and live in
``feature_extraction/channel_intensity``. The seam exists so the background null can be re-estimated
later without re-reading a single pixel.
"""

from .extraction import (
    HIST_BIN_WIDTH_DN,
    HIST_MIN_DN,
    HIST_N_BINS,
    INTENSITY_RECIPE_VERSION,
    ChannelIntensityError,
    RegionEvidence,
    build_annulus,
    expected_annulus_area_px,
    extract_embryo_intensity_evidence,
    histogram_region,
    neighbor_union_mask,
)

__all__ = [
    "ChannelIntensityError",
    "RegionEvidence",
    # The composed per-embryo-time extraction: everything below, in the right order.
    "extract_embryo_intensity_evidence",
    # Primitives, exposed so the pooled estimator downstream can be tested against the same
    # geometry the extractor used rather than a reimplementation of it.
    "build_annulus",
    "expected_annulus_area_px",
    "histogram_region",
    "neighbor_union_mask",
    # THE BIN SPEC. Fixed and global -- adaptive per-embryo edges would be unpoolable, and would
    # reintroduce per-frame rescaling one layer up.
    "HIST_BIN_WIDTH_DN",
    "HIST_N_BINS",
    "HIST_MIN_DN",
    # Bumped when the MEANING of the evidence changes, so an old table cannot be silently
    # reinterpreted under new semantics.
    "INTENSITY_RECIPE_VERSION",
]
