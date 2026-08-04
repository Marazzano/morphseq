"""The channel_intensity row contract — what every shard must carry.

WHY THIS EXISTS AS A SEPARATE MODULE. The experiment-level merge is a ``pd.concat`` over per-well
shards, and ``concat`` will happily UNION two drifted schemas into a table full of NaN without
complaining. Passing these columns to the merge turns that silent union into a loud failure, which
is the whole point: a shard written by an older version of the extractor must not quietly become a
column of nulls in the merged table.

RAW EVIDENCE ONLY. No corrected or background-subtracted column appears here, deliberately. Pooling
the background null is a well-grain estimator that lives in ``feature_extraction/channel_intensity``,
and a corrected column at this grain would couple "did we measure correctly" to "is our background
model current" -- changing the estimator would then force re-extraction. It does not.
"""

from __future__ import annotations

# Identity: which embryo, when, measured off WHICH pixels. source_image_product_key is part of the
# key rather than metadata because intensity off a CLAHE'd raster is a different quantity from
# intensity off a quantitative one -- the same reason snip products are keyed by their source.
CHANNEL_INTENSITY_IDENTITY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "time_index",
    "mask_id",
    "source_image_product_key",
    "source_image_id",
    "intensity_recipe_version",
)

# The measurement. Histograms pool EXACTLY by elementwise summation; sum/sumsq pool too and give
# exact moments the binned counts only approximate. Both are carried -- they cost 16 bytes.
CHANNEL_INTENSITY_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "embryo_hist_counts",
    "embryo_sum_dn",
    "embryo_sumsq_dn",
    "embryo_px",
    # SATURATION IS NOT OPTIONAL. The 2-copy embryo is the most likely to clip, and clipping
    # compresses 2-copy toward 1-copy -- destroying the dosage comparison while looking like clean
    # data. A merge that dropped this column would hide exactly that.
    "embryo_clipped_px",
    "annulus_hist_counts",
    "annulus_sum_dn",
    "annulus_sumsq_dn",
    "annulus_px",
    "annulus_clipped_px",
)

# How the annulus was built, so a row is self-describing and a parameter change is visible rather
# than inferred from the surrounding code at the time it ran.
CHANNEL_INTENSITY_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "annulus_inner_radius_um",
    "annulus_outer_radius_um",
    "annulus_excluded_px",
    "annulus_neighbor_count",
    "hist_bin_width_dn",
    "hist_n_bins",
    "hist_min_dn",
    "image_micrometers_per_pixel",
)

CHANNEL_INTENSITY_COLUMNS: tuple[str, ...] = (
    *CHANNEL_INTENSITY_IDENTITY_COLUMNS,
    *CHANNEL_INTENSITY_MEASUREMENT_COLUMNS,
    *CHANNEL_INTENSITY_GEOMETRY_COLUMNS,
)
