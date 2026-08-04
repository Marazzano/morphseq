"""Orientation-candidate MECHANICS: enumerate rot x flip, place each on a grid.

This module is deliberately biology-free. A PCA major axis fixes an elongated object's
orientation only up to a 180-degree ambiguity, and mirroring it is a second free choice.
Together those give four discrete candidate placements. Choosing among them requires
knowing which end of the object is which, and that is domain knowledge -- so selection
does NOT live here. See ``embryo_geometry.orientation`` for the policy that consumes
these candidates.

What this module owns:

- the closed candidate set ``{0, 180} x {False, True}`` and its ORDER
- building the OpenCV affine that rotates+scales about a source centroid and lands the
  result centered on an output grid
- warping a mask (and optional companion mask) through that affine, then applying the
  horizontal mirror

The order matters and is part of the contract: ``(0, False), (0, True), (180, False),
(180, True)``. Selection rules downstream are argmin/argmax over scalar scores, so ties
are resolved by first-wins iteration order. Changing this order would silently change
which candidate wins on a tie.

Coordinate conventions follow the rest of the package: coordinate VALUES are yx-ordered,
OpenCV affine matrices are xy-ordered internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - exercised implicitly wherever cv2 is installed
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


#: The closed candidate set, in contractual order. First-wins tie-breaking depends on it.
CANDIDATE_KEYS: tuple[tuple[int, bool], ...] = (
    (0, False),
    (0, True),
    (180, False),
    (180, True),
)

#: Candidate keys when mirroring is forbidden.
CANDIDATE_KEYS_NO_FLIP: tuple[tuple[int, bool], ...] = ((0, False), (180, False))


def candidate_keys(*, allow_flip: bool = True) -> tuple[tuple[int, bool], ...]:
    """Return the ordered candidate keys ``(rot_add_deg, flip_x)``."""
    return CANDIDATE_KEYS if allow_flip else CANDIDATE_KEYS_NO_FLIP


@dataclass(frozen=True)
class PlacedCandidate:
    """One of the four discrete placements of an object on the output grid.

    Attributes
    ----------
    rot_add_deg:
        The 180-degree ambiguity resolution: 0 or 180, ADDED to the base rotation.
    flip_x:
        Whether a horizontal mirror was applied AFTER the warp.
    mask:
        The warped (and possibly mirrored) primary mask, float32, on the output grid.
    companion:
        The same operations applied to an optional second mask -- a sub-region whose
        position within the object is what a downstream rule reads. None when no
        companion was supplied.
    affine_2x3:
        The OpenCV affine used for the warp, BEFORE the mirror. The mirror is a separate
        indexing operation, matching how consumers rebuild their transform chains.
    """

    rot_add_deg: int
    flip_x: bool
    mask: np.ndarray
    companion: Optional[np.ndarray]
    affine_2x3: np.ndarray

    @property
    def key(self) -> tuple[int, bool]:
        return (int(self.rot_add_deg), bool(self.flip_x))


def pixel_center_affine(affine_2x3: np.ndarray) -> np.ndarray:
    """Re-express a naive-convention 2x3 affine so it maps PIXEL CENTERS.

    THE DEFECT THIS REPAIRS. A pixel with integer index ``x`` is a SAMPLE OF AREA centered at
    ``x + 0.5``, not a point at ``x``. ``cv2.resize`` honors that -- it implements
    ``x_out = sf * (x_src + 0.5) - 0.5``, pinned from first principles (and cross-checked
    against skimage) in ``tests/image_geometry/test_resize_coordinate_convention.py``.
    ``cv2.warpAffine`` applies whatever matrix it is handed and supplies NO correction of its
    own, so a matrix built the naive way, ``x_out = sf * x_src``, samples ``(sf - 1) / 2``
    away from where the identical scale expressed as a resize samples.

    WHY THAT IS WORSE THAN NOISE, AND WHY IT SURVIVED SO LONG. The offset is constant and
    direction-consistent: every object shifts by the SAME amount in the SAME direction. That
    makes it invisible to every aggregate statistic -- area, IoU, mean error and pixel diffs
    all stay clean -- while the whole population sits off-grid relative to the resize-based
    coordinates it is compared against. It also survived a dedicated equivalence suite,
    because that suite pinned this module against the canonical aligner, which carried the
    identical defect. Two clocks five minutes slow agree perfectly.

    THE DERIVATION. For ``out = A @ src + t`` under the naive rule, requiring instead that
    pixel CENTERS map to pixel CENTERS means ``(out + 0.5) = A @ (src + 0.5) + t``, so the
    translation column gains ``A @ [0.5, 0.5] - 0.5``.

    APPLIED UNCONDITIONALLY, ON PURPOSE. For a pure translation ``A = I`` the correction is
    identically zero, so gating it on a scale or rotation test would add a branch that can
    only ever be wrong -- and a gate is exactly how this class of bias creeps back. The
    invariant suite pins both the nonzero case and the translation no-op.
    """
    out = np.asarray(affine_2x3, dtype=np.float64).copy()
    half = np.array([0.5, 0.5], dtype=np.float64)
    out[:, 2] += out[:, :2] @ half - half
    return out


def centered_placement_affine(
    *,
    rotation_deg: float,
    scale: float,
    src_center_xy: tuple[float, float],
    out_shape_yx: tuple[int, int],
) -> np.ndarray:
    """Rotate+scale about ``src_center_xy`` and land that center at the output center.

    Returned in the PIXEL-CENTER convention, so this seam agrees with every resize seam. See
    ``pixel_center_affine`` for the defect that correction repairs and why an agreement test
    could not detect it.

    The ``W / 2`` centering is retained: it is off by half a pixel from the true pixel-grid
    center ``(W - 1) / 2``, but that is an independent question about WHERE the object is
    parked on the canvas, not about what a coordinate MEANS. Changing it would move every
    object relative to the historical embedding space, so it is deliberately left alone here.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("cv2 is required for centered_placement_affine.")
    h_out, w_out = out_shape_yx
    cx, cy = float(src_center_xy[0]), float(src_center_xy[1])
    affine = cv2.getRotationMatrix2D((cx, cy), float(rotation_deg), float(scale))
    affine[0, 2] += (w_out / 2) - cx
    affine[1, 2] += (h_out / 2) - cy
    return pixel_center_affine(affine)


def enumerate_orientation_candidates(
    mask: np.ndarray,
    *,
    base_rotation_deg: float,
    scale: float,
    src_center_xy: tuple[float, float],
    out_shape_yx: tuple[int, int],
    companion: Optional[np.ndarray] = None,
    allow_flip: bool = True,
) -> list[PlacedCandidate]:
    """Place ``mask`` on the output grid in each of the (up to four) candidate poses.

    Parameters
    ----------
    mask:
        Primary binary/boolean mask in source coordinates.
    base_rotation_deg:
        The rotation that brings the PCA major axis onto the target axis. Each candidate
        adds 0 or 180 to it.
    scale:
        Isotropic scale factor applied within the affine. NOTE: a large downscale folded
        into an affine cannot anti-alias (see the package README); this function is
        nearest-neighbour on masks, where that is acceptable, and callers must not route
        grayscale images through it.
    src_center_xy:
        Rotation pivot in source pixels, xy-ordered (typically the mask centroid).
    out_shape_yx:
        Output grid shape.
    companion:
        Optional second mask transformed identically, so a caller's selection rule can
        read its position in each candidate frame without re-warping.
    allow_flip:
        When False only the two non-mirrored candidates are produced.

    Returns
    -------
    list[PlacedCandidate]
        In ``candidate_keys(allow_flip=...)`` order.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("cv2 is required for enumerate_orientation_candidates.")
    h_out, w_out = int(out_shape_yx[0]), int(out_shape_yx[1])
    src = np.asarray(mask).astype(np.float32)
    comp_src = None if companion is None else np.asarray(companion).astype(np.float32)

    out: list[PlacedCandidate] = []
    for rot_add, do_flip in candidate_keys(allow_flip=allow_flip):
        affine = centered_placement_affine(
            rotation_deg=float(base_rotation_deg) + float(rot_add),
            scale=scale,
            src_center_xy=src_center_xy,
            out_shape_yx=(h_out, w_out),
        )
        affine_f32 = affine.astype(np.float32)
        warped = cv2.warpAffine(src, affine_f32, (w_out, h_out), flags=cv2.INTER_NEAREST)
        warped_comp = (
            None
            if comp_src is None
            else cv2.warpAffine(comp_src, affine_f32, (w_out, h_out), flags=cv2.INTER_NEAREST)
        )
        if do_flip:
            warped = cv2.flip(warped, 1)
            if warped_comp is not None:
                warped_comp = cv2.flip(warped_comp, 1)
        out.append(
            PlacedCandidate(
                rot_add_deg=int(rot_add),
                flip_x=bool(do_flip),
                mask=warped,
                companion=warped_comp,
                affine_2x3=affine,
            )
        )
    return out


def vertical_flip_partner(key: tuple[int, bool]) -> tuple[int, bool]:
    """The candidate that differs from ``key`` by a VERTICAL mirror.

    There is no standalone vertical-flip candidate in the enumeration: flipping about the
    horizontal axis equals a 180-degree rotation composed with a horizontal mirror. So the
    partner of ``(rot, flip)`` is ``((rot + 180) % 360, not flip)``. Consumers that want to
    invert an up/down decision look this up in the already-computed table rather than
    warping again.
    """
    rot_add, flip_x = key
    return ((int(rot_add) + 180) % 360, not bool(flip_x))


def pca_major_axis_angle_deg(mask: np.ndarray) -> tuple[float, tuple[float, float], bool]:
    """Major-axis angle (degrees) and centroid (xy) of the nonzero pixels of ``mask``.

    Returns ``(angle_deg, centroid_xy, well_defined)``. ``well_defined`` is False for an
    empty mask or a degenerate (zero-variance) point cloud, in which case the angle is 0.
    The angle is only defined modulo 180 degrees -- resolving the remaining ambiguity is
    exactly what the candidate enumeration above exists for.
    """
    arr = np.asarray(mask)
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return 0.0, (0.0, 0.0), False
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if eigvals[0] <= 0:
        return 0.0, (float(mean[0]), float(mean[1])), False
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return angle, (float(mean[0]), float(mean[1])), True
