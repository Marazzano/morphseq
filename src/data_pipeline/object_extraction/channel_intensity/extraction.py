"""Per-embryo fluorescence evidence, measured on the NATIVE raster.

WHAT THIS IS FOR. Each embryo carries 0, 1, or 2 copies of a transgene, and the point of the whole
path is that *relative* intensity survives to analysis. That requirement is what makes several
choices here non-negotiable where a display pipeline could be sloppy.

WHY NOT ON THE SNIP. A snip is rendered through ``INTER_AREA``, a local average that mixes photons
across pixel boundaries — biasing both saturation counts and the tails of the distribution — and it
fills ``border_mode`` constant outside the source, which would contaminate any annulus near a frame
edge. Photometry belongs on the native uint16 raster; the snip is a display/model product.

The consequence is worth stating plainly: **this module needs no snip transform and reads no
rendered snip.** Its identity is

    physical_embryo_id x time_index x source_image_product_key x intensity_recipe_version

carrying ``snip_transform_id`` only as a join key. That independence is why it lives under
``object_extraction`` beside segmentation rather than inside the snip package.

WHY HISTOGRAMS AND NOT SUMMARY STATISTICS. The background null is POOLED over a well's annuli across
time. Mean/median/MAD are not sufficient statistics — you cannot recover a pooled quantile or a mode
from them — and median-of-medians would weight every embryo equally regardless of how much annulus
survived neighbor exclusion, which is exactly the quantity that varies. A fixed-bin histogram is the
only compact per-embryo emission that pools EXACTLY, by elementwise summation.

The estimator downstream is a MODE, not a mean. The annulus background is not symmetric noise: the
well rim autofluoresces (measured ~1019 DN vs ~575 mid-well) and embryo halo contaminates from the
inside — both right-tailed. A mean is dragged by both; the mode is what the camera reports where
nothing is, which is the quantity to subtract.

RAW ONLY. Nothing here subtracts a background, normalizes, or corrects. Pooling is a well-grain
estimator over a population of embryos and lives in ``feature_extraction/channel_intensity``. The
seam: object_extraction emits raw poolable evidence; feature_extraction chooses an estimator. That
is what lets the null be re-estimated later without re-reading a single pixel.

Import direction: numpy/scipy and the shared mask helpers only. MUST NOT import the entrypoint,
orchestration, tasks, or Snakemake rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from image_geometry import BoxYX

# Bumped when the MEANING of the emitted evidence changes -- a different annulus construction, a
# different membership rule, a different bin spec. Carried on every row so an old table cannot be
# reinterpreted under new semantics by a parser that looks perfectly innocent.
INTENSITY_RECIPE_VERSION = "native_annulus_hist_v1"

# THE BIN SPEC, FIXED AND GLOBAL. Per-snip adaptive edges would be unpoolable -- and worse, they
# would reintroduce the per-frame-rescale bug one layer up, since each embryo's counts would then be
# expressed against its own scale. 2048 bins x 32 DN spans the full uint16 domain at ~8 KB/embryo.
HIST_BIN_WIDTH_DN = 32
HIST_N_BINS = 2048
HIST_MIN_DN = 0
_HIST_SHIFT = 5  # value >> 5 == value // 32; exact for the power-of-two bin width above.


class ChannelIntensityError(ValueError):
    """Evidence could not be extracted, or would have been silently wrong."""


@dataclass(frozen=True)
class RegionEvidence:
    """Raw, uncorrected evidence for one pixel region. Poolable by summation."""

    hist_counts: np.ndarray  # uint32[HIST_N_BINS]
    sum_dn: float
    sumsq_dn: float
    n_px: int
    clipped_px: int

    def as_columns(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_hist_counts": self.hist_counts.tolist(),
            f"{prefix}_sum_dn": float(self.sum_dn),
            f"{prefix}_sumsq_dn": float(self.sumsq_dn),
            f"{prefix}_px": int(self.n_px),
            f"{prefix}_clipped_px": int(self.clipped_px),
        }


def histogram_region(values: np.ndarray, *, dtype_max: int) -> RegionEvidence:
    """Bin one region's pixel values, and count saturation.

    ``sum``/``sumsq`` are carried alongside the histogram because they pool by summation too and give
    EXACT pooled moments, which the binned counts only approximate. They cost 16 bytes.

    SATURATION IS NOT OPTIONAL. The 2-copy embryo is the most likely to saturate, and saturation
    compresses 2-copy toward 1-copy -- destroying the dosage comparison while looking like clean
    data. Max-over-Z projection makes it worse by taking the brightest plane per pixel.
    """
    flat = np.asarray(values).ravel()
    if flat.size == 0:
        return RegionEvidence(
            hist_counts=np.zeros(HIST_N_BINS, dtype=np.uint32),
            sum_dn=0.0,
            sumsq_dn=0.0,
            n_px=0,
            clipped_px=0,
        )

    as_int = flat.astype(np.int64, copy=False)
    if as_int.min() < 0:
        raise ChannelIntensityError(
            "histogram_region: negative pixel values, which no unsigned detector produces. The "
            "source raster has been through a signed transform, so its intensities no longer mean "
            "what this module assumes."
        )

    # Clamp into the top bin rather than dropping: a value above the bin domain is real signal, and
    # silently discarding it would understate exactly the brightest embryos.
    bins = np.minimum(as_int >> _HIST_SHIFT, HIST_N_BINS - 1)
    counts = np.bincount(bins, minlength=HIST_N_BINS).astype(np.uint32)

    as_float = flat.astype(np.float64, copy=False)
    return RegionEvidence(
        hist_counts=counts,
        sum_dn=float(as_float.sum()),
        sumsq_dn=float(np.square(as_float).sum()),
        n_px=int(flat.size),
        clipped_px=int(np.count_nonzero(as_int >= dtype_max)),
    )


def _distance_um(shape_yx: tuple[int, int], mask: np.ndarray, um_per_px_yx: tuple[float, float]):
    """Physical distance from the mask, in micrometers, for every pixel.

    ONE EDT rather than two dilations: at the radii this module uses, a distance transform is both
    cheaper and exact, and it yields both radii from a single pass.
    """
    um_y, um_x = um_per_px_yx
    # sampling= makes the transform physical directly, so no scalar "radius in pixels" ever exists to
    # be ambiguous on an anisotropic grid.
    return ndimage.distance_transform_edt(~mask, sampling=(um_y, um_x))


def build_annulus(
    mask: np.ndarray,
    *,
    inner_radius_um: float,
    outer_radius_um: float,
    um_per_px_yx: tuple[float, float],
) -> np.ndarray:
    """The background ring around one embryo, tested in MICROMETERS.

    Membership is ``inner <= distance < outer`` -- inclusive inner, exclusive outer, so adjacent
    rings tile without double-counting.

    THE TEST IS PHYSICAL, NOT PIXEL-DENOMINATED. A scalar ``radius_px = radius_um / um_per_px``
    assumes isotropy, which no contract guarantees and which an axis-dependent downsample or a
    write-policy ``orientation`` could break. Testing distance in micrometers removes the ambiguity
    instead of asserting it away, and on an anisotropic grid yields a physical circle -- which is a
    pixel-space ellipse, the correct shape. A circle in pixels would be an ellipse in the specimen.

    The inner radius is a GAP, not decoration: it keeps PSF and optical spillover from the embryo out
    of the embryo's own background.
    """
    if not (outer_radius_um > inner_radius_um >= 0):
        raise ChannelIntensityError(
            f"build_annulus: need outer > inner >= 0; got inner={inner_radius_um!r}, "
            f"outer={outer_radius_um!r}. A non-positive ring samples nothing."
        )
    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        raise ChannelIntensityError(
            "build_annulus: the target mask is empty, so there is no embryo to ring. The caller must "
            "treat this as an invalid row rather than measuring an arbitrary region."
        )
    distance_um = _distance_um(binary.shape, binary, um_per_px_yx)
    return (distance_um >= inner_radius_um) & (distance_um < outer_radius_um)


def expected_annulus_area_px(
    *,
    inner_radius_um: float,
    outer_radius_um: float,
    um_per_px_yx: tuple[float, float],
) -> int:
    """Ring area on an UNBOUNDED reference grid at the native calibration.

    The denominator of ``area_fraction``, and it must be counted on the SAME grid as the numerator --
    an earlier draft defined it at the snip's target calibration while sampling actual pixels
    natively, which would compare a native numerator against a snip-grid denominator.

    RASTERIZED, not analytic ``pi(r_o^2 - r_i^2)``: the analytic area disagrees with any
    rasterization at small radii, so a fraction built from it would drift with radius in a way that
    looks like data. Rasterizing makes this a pure function of the radii and calibration --
    cacheable, testable, and identical for every embryo sharing a grid.

    Note this is the ring the geometry ASKS for: the target mask is the hole rather than
    contamination, and neighbors are deliberately NOT excluded here. What exclusion removes is
    measured against this.
    """
    um_y, um_x = um_per_px_yx
    half_y = int(np.ceil(outer_radius_um / um_y))
    half_x = int(np.ceil(outer_radius_um / um_x))
    dy = (np.arange(-half_y, half_y + 1) * um_y)[:, None]
    dx = (np.arange(-half_x, half_x + 1) * um_x)[None, :]
    distance_um = np.hypot(dy, dx)
    return int(np.count_nonzero((distance_um >= inner_radius_um) & (distance_um < outer_radius_um)))


def neighbor_union_mask(
    *,
    target_mask_id: str,
    target_box: BoxYX,
    neighbors: Iterable[Mapping[str, object]],
    decode: "callable",
    shape_yx: tuple[int, int],
    outer_radius_um: float,
    exclude_radius_um: float,
    um_per_px_yx: tuple[float, float],
) -> tuple[np.ndarray, int]:
    """Union of the OTHER embryos in this frame, dilated -- the "close fish" exclusion.

    Returns ``(dilated_union, n_candidates_decoded)``.

    THE BBOX PREFILTER MUST EXPAND BY THE OUTER RADIUS. A neighbor sitting outside the target's bbox
    but inside its annulus is exactly the contamination case this exists for, so testing the
    unexpanded box misses precisely the rows that matter.

    EXCLUDE BY ``mask_id``, never by row position -- ``mask_id`` is the unique key of frame_masks, and
    excluding by position would subtract the target from its own annulus while producing a row that
    looks entirely well-formed.

    INVALID MASKS ARE STILL INCLUDED. ``is_valid_mask=False`` is a tracking/QC judgement; an invalid
    mask is still a fish emitting photons into its neighbour's background. Filtering on validity here
    would silently admit contamination from exactly the embryos the QC flagged as problematic.

    UNION FIRST, DILATE ONCE. Dilating each neighbor separately is N distance transforms and N
    chances to disagree; one EDT on the union is cheaper and cannot drift. The exclusion radius is
    typically LARGER than the annulus outer radius, because a neighbor's halo reaches past its mask
    edge.
    """
    padded = target_box.pad(
        int(np.ceil(outer_radius_um / um_per_px_yx[0])),
        int(np.ceil(outer_radius_um / um_per_px_yx[1])),
    )

    union = np.zeros(shape_yx, dtype=bool)
    n_decoded = 0
    for row in neighbors:
        if str(row["mask_id"]) == str(target_mask_id):
            continue
        neighbor_box = BoxYX(
            y0=int(row["bbox_y_min_px"]),
            y1=int(row["bbox_y_max_px"]),
            x0=int(row["bbox_x_min_px"]),
            x1=int(row["bbox_x_max_px"]),
        )
        if not padded.intersects(neighbor_box):
            continue
        union |= np.asarray(decode(row)).astype(bool)
        n_decoded += 1

    if n_decoded == 0 or exclude_radius_um <= 0:
        return union, n_decoded

    distance_um = _distance_um(union.shape, union, um_per_px_yx)
    return (distance_um < exclude_radius_um), n_decoded


def extract_embryo_intensity_evidence(
    *,
    image: np.ndarray,
    target_mask: np.ndarray,
    target_mask_id: str,
    neighbors: Sequence[Mapping[str, object]],
    decode: "callable",
    um_per_px_yx: tuple[float, float],
    inner_radius_um: float,
    outer_radius_um: float,
    exclude_radius_um: float,
    dtype_max: int,
) -> dict[str, object]:
    """All raw evidence for ONE embryo at ONE timepoint, on the native grid.

    RAW ONLY -- no background is subtracted here, because the null is pooled over the whole well and
    is not knowable while measuring the first embryo. Emitting corrected values here would also
    couple "did we measure correctly" to "is our background model current," so changing the estimator
    would force re-extraction. It does not.

    ``image`` must be the NATIVE materialized frame for the fluorescence product, un-rescaled. If it
    has been through a per-frame ``rescale_intensity``, every number below is meaningless for
    cross-embryo comparison and no downstream correction can recover it.
    """
    if image.shape[:2] != target_mask.shape[:2]:
        raise ChannelIntensityError(
            f"extract_embryo_intensity_evidence: image {image.shape[:2]} and mask "
            f"{target_mask.shape[:2]} are on different grids. The mask does not describe these "
            "pixels; check that the fluorescence product and the BF segmentation share a field of "
            "view and calibration."
        )

    embryo = np.asarray(target_mask).astype(bool)
    annulus = build_annulus(
        embryo,
        inner_radius_um=inner_radius_um,
        outer_radius_um=outer_radius_um,
        um_per_px_yx=um_per_px_yx,
    )

    box = BoxYX.from_mask(embryo)
    if box is None:  # build_annulus already rejects an empty mask; this is belt-and-braces.
        raise ChannelIntensityError("extract_embryo_intensity_evidence: empty target mask.")

    exclusion, n_neighbors = neighbor_union_mask(
        target_mask_id=target_mask_id,
        target_box=box,
        neighbors=neighbors,
        decode=decode,
        shape_yx=embryo.shape,
        outer_radius_um=outer_radius_um,
        exclude_radius_um=exclude_radius_um,
        um_per_px_yx=um_per_px_yx,
    )

    annulus_valid = annulus & ~exclusion
    expected_px = expected_annulus_area_px(
        inner_radius_um=inner_radius_um,
        outer_radius_um=outer_radius_um,
        um_per_px_yx=um_per_px_yx,
    )
    annulus_px = int(np.count_nonzero(annulus))
    valid_px = int(np.count_nonzero(annulus_valid))

    columns: dict[str, object] = {
        "intensity_recipe_version": INTENSITY_RECIPE_VERSION,
        "hist_bin_width_dn": HIST_BIN_WIDTH_DN,
        "hist_n_bins": HIST_N_BINS,
        "hist_min_dn": HIST_MIN_DN,
        # The membership test is physical; these are reported per axis so a row is self-describing
        # and a calibration change is visible rather than inferred. DIAGNOSTIC, not the test.
        "annulus_inner_radius_um": float(inner_radius_um),
        "annulus_outer_radius_um": float(outer_radius_um),
        "annulus_exclude_radius_um": float(exclude_radius_um),
        "annulus_inner_radius_px_y": float(inner_radius_um / um_per_px_yx[0]),
        "annulus_inner_radius_px_x": float(inner_radius_um / um_per_px_yx[1]),
        "annulus_outer_radius_px_y": float(outer_radius_um / um_per_px_yx[0]),
        "annulus_outer_radius_px_x": float(outer_radius_um / um_per_px_yx[1]),
        # Both counted on the NATIVE grid, so the fraction compares like with like.
        "annulus_expected_area_px": expected_px,
        "annulus_total_px": annulus_px,
        "annulus_excluded_px": annulus_px - valid_px,
        # Reported rather than used as a validity gate: a partly clipped annulus is still usable if
        # enough clean pixels remain, and invalidating on any clipping would create
        # location-dependent missingness -- embryos near a frame edge would drop out systematically.
        "annulus_out_of_frame_px": max(0, expected_px - annulus_px),
        "annulus_area_fraction": (float(valid_px) / expected_px) if expected_px else 0.0,
        "annulus_neighbor_count": int(n_neighbors),
        "image_micrometers_per_pixel_y": float(um_per_px_yx[0]),
        "image_micrometers_per_pixel_x": float(um_per_px_yx[1]),
        "dtype_max": int(dtype_max),
    }
    columns.update(
        histogram_region(image[annulus_valid], dtype_max=dtype_max).as_columns("annulus")
    )
    columns.update(histogram_region(image[embryo], dtype_max=dtype_max).as_columns("embryo"))
    return columns
