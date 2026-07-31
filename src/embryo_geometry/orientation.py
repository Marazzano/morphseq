"""The shared embryo-orientation engine, with the evidence behind each decision.

WHY THIS EXISTS
---------------
Embryo orientation is decided in two divergent places, and a third would be invented for
the next consumer:

- the snip pipeline (``data_pipeline.object_extraction.snip_processing.rotation``) uses a
  skimage ``regionprops`` axis plus a bare ``vert_ratio >= 0.5`` scalar latch;
- the analysis stack (``analyze.utils.coord.grids.canonical.CanonicalAligner``) uses a PCA
  axis plus a four-candidate enumeration with a yolk-left AP rule and a back-above-yolk DV
  rule, falling back to a diagonal "mass toward upper-left" score when the yolk is missing.

Two engines answering the same biological question will drift. This module is the one
engine. What it does NOT do is quietly declare one of the existing answers canonical:

THERE ARE CURRENTLY *THREE* DISTINCT RULES IN PRODUCTION, NOT ONE.

    1. yolk-left AP + back-above-yolk DV        (CanonicalAligner, yolk present)
    2. -(com_y + com_x) "mass toward upper-left" (CanonicalAligner, yolk missing)
    3. vert_ratio >= 0.5                         (snip pipeline, yolk missing)

and there is a fourth in the snip pipeline for the yolk-present case (yolk above the
embryo centroid), which is a different anatomical claim from rule 1. Rules 2 and 3 are
BOTH "the no-yolk policy" -- of different callers, and they do not agree. So the fallback
is a parameter here, from a closed set, and every decision reports which one ran and why.
Converging on a single fallback then becomes a later, deliberate, attributable change
rather than something that happens by accident inside an extraction.

THIS IS AN EXTRACTION, NOT A POLICY CHANGE. Each caller can request the policy it uses
today and get the decision it makes today, byte for byte. What is new is that the evidence
is returned instead of discarded.

THE DECISION IS CANVAS-INDEPENDENT
----------------------------------
``EmbryoOrientationDecision`` describes ORIENTATION, not rendering. It carries no grid
shape, no um/px, no anchor, no clipping policy. It reports ``major_axis_angle_deg`` and
``rotation_adjustment_deg`` (the 0/180 disambiguation) separately, so each consumer
composes its own final rotation against its own canvas. In particular
``target_angle = 0 if landscape else 90`` in ``canonical.py`` is a property of the CANVAS
ASPECT RATIO, not of the embryo -- that logic stays with the renderer.

NO CALIBRATED CONFIDENCE NUMBER IS EMITTED
------------------------------------------
The margins are RAW and not commensurable. ``ap_margin_px`` is pixels of yolk-x
separation, ``dv_margin_px`` is ``|back_y - yolk_y|`` in pixels, ``vert_ratio_margin`` is
``|vert_ratio - 0.5|`` and dimensionless. A 10px AP margin is not "the same amount of
evidence" as a 0.03 vert-ratio margin, and combining them would need calibration against
embryo size, yolk radius, and empirical error rates that nobody has done. So no
probability and no score is returned -- only the measurements, for a caller to threshold
deliberately.

AMBIGUITIES FOUND WHILE STATING THE RULES PRECISELY
---------------------------------------------------
1. Every decision point is an UNREPORTED THRESHOLD LATCH. AP is an ``argmin`` with no
   margin requirement; DV is a strict sign test with no deadband; ``vert_ratio >= 0.5`` is
   a hard latch on a continuous statistic. Distance from the cliff was computed and thrown
   away. Embryo ``20250416_D09`` flips 180 degrees THREE times within one timelapse, with
   ``vert_ratio`` of 0.4967 / 0.5068 / 0.4843 -- three coin flips reported as three
   confident decisions. That is why the margins are returned.

2. STEP 2 CAN UNDO STEP 1. The vertical-flip partner mirrors horizontally too, so
   enforcing dorsal-up can move the yolk to the RIGHT, violating the rule Step 1 just
   applied. The two rules are not jointly satisfiable in general and DV silently wins.
   ``dv_override_ap_choice`` reports it. Yolk-left is therefore NOT an invariant of the
   output, despite being stated as the rule.

3. The DV test is ``> 0``, so ``back_y == yolk_y`` exactly -- the back level with the
   yolk, carrying no dorsal information -- silently keeps Step 1's answer instead of
   flagging the axis as undetermined. ``dv_margin_px == 0.0`` makes it visible.

4. THE TWO YOLK-PRESENT RULES DISAGREE IN KIND. CanonicalAligner asks where the BACK is
   relative to the YOLK; the snip pipeline asks where the YOLK is relative to the EMBRYO
   CENTROID. These are different anatomical claims and can select opposite poses on the
   same animal. Both are preserved; neither is declared correct here.

5. "Yolk present" is decided AFTER warping, per candidate, in CanonicalAligner. A yolk
   nonempty in source coordinates but warped off the output grid degrades to the fallback
   with no error, and the yolk landmark then silently becomes the EMBRYO centroid -- so a
   coordinate alone cannot distinguish "yolk here" from "no yolk". Read ``yolk_used``.

6. ``vert_ratio`` counts OCCUPIED ROWS, not pixels: ``sum(y_indices > com_y) / len(...)``
   over rows where the mask has any pixel. It is a measure of row extent, not of mass, so
   a long thin tail counts as much as a bulky head. Preserved verbatim.

7. The two axis measurements are NOT the same quantity. skimage's ``regionprops
   .orientation`` measures from the vertical axis; the PCA helper here measures from the
   horizontal via ``arctan2``, giving ``skimage ~= -(pca + 90)`` mod 180. The
   ``legacy_vert_ratio`` policy therefore keeps skimage's own measurement rather than
   deriving it, or equivalence would break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from image_geometry.candidates import (
    PlacedCandidate,
    candidate_keys,
    enumerate_orientation_candidates,
    pca_major_axis_angle_deg,
    vertical_flip_partner,
)

# ---------------------------------------------------------------------------
# Closed vocabularies. Consumers may branch on these; new members are additions,
# never silent reinterpretations of an existing one.
# ---------------------------------------------------------------------------

#: The anatomical rule: yolk-left for AP, then back-above-yolk for DV.
YOLK_BACK = "yolk_back"
#: What CanonicalAligner does today when the yolk is missing: -(com_y + com_x).
LEGACY_UPPER_LEFT_COM = "legacy_upper_left_com"
#: What the snip pipeline does today: the vert_ratio >= 0.5 latch.
LEGACY_VERT_RATIO = "legacy_vert_ratio"

NO_YOLK_POLICIES = (LEGACY_UPPER_LEFT_COM, LEGACY_VERT_RATIO)

#: ``fallback_reason`` values.
MISSING_YOLK = "missing_yolk"
YOLK_DISABLED = "yolk_disabled"
YOLK_WARPED_OFF_GRID = "yolk_warped_off_grid"

#: Degenerate outcomes.
EMPTY_MASK = "empty_mask"
DEGENERATE_MASK = "degenerate_mask"


@dataclass(frozen=True)
class EmbryoOrientationDecision:
    """An orientation decision plus the RAW evidence that produced it.

    Canvas-independent by construction: no grid shape, no um/px, no anchor. Compose the
    final rotation yourself::

        target_angle = 0.0 if canvas_is_landscape else 90.0
        final_rotation = (d.major_axis_angle_deg - target_angle) + d.rotation_adjustment_deg

    Angles are degrees; coordinates are yx-ordered.

    Attributes
    ----------
    major_axis_angle_deg:
        The measured axis of the mask, in the convention of whichever measurement the
        active policy uses (see AMBIGUITIES note 7 -- these are NOT interchangeable).
        Defined only modulo 180; resolving that is what the adjustment is for.
    axis_convention:
        Which measurement produced it: ``"pca_arctan2_from_horizontal"`` or
        ``"skimage_regionprops_from_vertical"``.
    rotation_adjustment_deg:
        The chosen disambiguation, 0 or 180. ADD this to your own base rotation.
    flip_x:
        Whether a horizontal mirror is applied after the rotation. Always False for
        ``legacy_vert_ratio``, which has no mirror candidate at all.
    orientation_source:
        The rule that actually decided -- one of the policy constants, or a degenerate
        outcome. Provenance, never inferred from other fields.
    fallback_policy / fallback_reason:
        Populated only when the anatomical rule could not run. ``fallback_policy`` is the
        no-yolk policy that ran instead; ``fallback_reason`` says why it had to.
    yolk_used / back_used:
        Whether each cue was actually available. ``back_used`` is False whenever the back
        point degenerated to a centroid fallback.
    ap_margin_px:
        Winner-vs-runner-up gap on yolk-x. Pixels, in the frame the candidates were
        evaluated on. Small means the AP call was a coin flip.
    dv_margin_px:
        ``abs(back_y - yolk_y)`` in the selected candidate. Pixels. Zero means the DV axis
        carried no information and Step 1's answer was kept by default.
    vert_ratio / vert_ratio_margin:
        The snip pipeline's statistic and ``abs(vert_ratio - 0.5)``. DIMENSIONLESS -- not
        comparable to the pixel margins above.
    back_estimation_status:
        Which branch the back-direction routine took. Only
        ``"yolk_surrounding_centroid"`` is the good path.
    back_support_pixels:
        Embryo pixels inside the yolk-centered sampling disk. Few means a noisy DV call.
    selected_candidate_index:
        Index into the ordered candidate list, or 0 where no enumeration happened.
    dv_override_ap_choice:
        True when Step 2 replaced Step 1's winner -- dorsal-up and yolk-left disagreed and
        dorsal-up won. See AMBIGUITIES note 2.
    """

    major_axis_angle_deg: float
    axis_convention: str
    rotation_adjustment_deg: float
    flip_x: bool

    orientation_source: str
    fallback_policy: Optional[str] = None
    fallback_reason: Optional[str] = None

    yolk_used: bool = False
    back_used: bool = False

    ap_margin_px: Optional[float] = None
    dv_margin_px: Optional[float] = None
    vert_ratio: Optional[float] = None
    vert_ratio_margin: Optional[float] = None

    back_estimation_status: Optional[str] = None
    back_support_pixels: Optional[int] = None

    selected_candidate_index: int = 0
    dv_override_ap_choice: bool = False

    # Landmarks in the evaluated candidate frame, yx-ordered. QC overlays and the metadata
    # the existing consumers already record.
    yolk_yx: Optional[tuple[float, float]] = None
    back_yx: Optional[tuple[float, float]] = None

    def final_rotation_deg(self, *, target_angle_deg: float = 0.0) -> float:
        """Compose the rotation against a caller-chosen target axis.

        ``target_angle_deg`` is a property of the CONSUMER'S CANVAS (0 for landscape, 90
        for portrait in the analysis renderer), not of the embryo, which is why it is an
        argument rather than baked into the decision.
        """
        return float(self.major_axis_angle_deg - float(target_angle_deg) + self.rotation_adjustment_deg)


@dataclass(frozen=True)
class EmbryoOrientationPolicy:
    """Knobs of the orientation policy, separate from the raster it is applied to.

    ``no_yolk_policy`` is REQUIRED to be explicit at the call site rather than defaulted,
    because the two callers disagree and a default would silently pick a winner. It
    defaults here only to CanonicalAligner's behavior so the analysis path is unchanged;
    the pipeline must pass ``legacy_vert_ratio``.

    ``allow_flip=False`` removes the mirror candidates, which also disables the Step-2 DV
    correction -- the vertical-flip partner of a non-mirrored candidate IS a mirrored one,
    so without mirrors the correction cannot be expressed.
    """

    no_yolk_policy: str = LEGACY_UPPER_LEFT_COM
    allow_flip: bool = True
    back_sample_radius_k: float = 1.75

    def __post_init__(self) -> None:
        if self.no_yolk_policy not in NO_YOLK_POLICIES:
            raise ValueError(
                f"no_yolk_policy must be one of {NO_YOLK_POLICIES!r}, got {self.no_yolk_policy!r}"
            )


# ---------------------------------------------------------------------------
# Landmark primitives
# ---------------------------------------------------------------------------


def _center_of_mass(mask: np.ndarray, *, empty_default_yx: tuple[float, float]) -> tuple[float, float]:
    """Centroid of a mask's weight, yx-ordered.

    ``empty_default_yx`` is returned for an empty mask. CanonicalAligner returns the OUTPUT
    GRID CENTER there -- a shape-dependent sentinel indistinguishable from a real
    centroid. Preserved for equivalence, but passed in explicitly so the dependence is at
    least visible rather than reaching into ``self.H/self.W``.
    """
    arr = np.asarray(mask, dtype=np.float64)
    total = arr.sum()
    if arr.size == 0 or total == 0:
        return (float(empty_default_yx[0]), float(empty_default_yx[1]))
    ys, xs = np.indices(arr.shape)
    return (float((ys * arr).sum() / total), float((xs * arr).sum() / total))


def _project_point_into_mask_within_disk(
    mask: np.ndarray,
    yx: tuple[float, float],
    *,
    disk_center_yx: tuple[float, float],
    disk_radius: float,
) -> tuple[float, float]:
    """Snap ``yx`` onto the nearest ON pixel of ``mask``, preferring pixels in the disk.

    The raw back centroid can land in a concavity outside the mask, but the back point must
    be ON the embryo. Restricting the search to the disk keeps the projection local rather
    than letting it jump to a distant limb.
    """
    y, x = float(yx[0]), float(yx[1])
    arr = np.asarray(mask)
    iy = int(np.clip(round(y), 0, arr.shape[0] - 1))
    ix = int(np.clip(round(x), 0, arr.shape[1] - 1))
    if arr[iy, ix] > 0.5:
        return float(iy), float(ix)

    ys, xs = np.where(arr > 0.5)
    if ys.size == 0:
        return y, x

    dy_disk = ys.astype(np.float64) - float(disk_center_yx[0])
    dx_disk = xs.astype(np.float64) - float(disk_center_yx[1])
    in_disk = (dy_disk**2 + dx_disk**2) <= float(disk_radius) ** 2
    if not in_disk.any():
        d2 = (ys.astype(np.float64) - y) ** 2 + (xs.astype(np.float64) - x) ** 2
        idx = int(np.argmin(d2))
        return float(ys[idx]), float(xs[idx])

    ys_d, xs_d = ys[in_disk], xs[in_disk]
    d2 = (ys_d.astype(np.float64) - y) ** 2 + (xs_d.astype(np.float64) - x) ** 2
    idx = int(np.argmin(d2))
    return float(ys_d[idx]), float(xs_d[idx])


def compute_back_point(
    mask: np.ndarray,
    yolk_mask: Optional[np.ndarray],
    *,
    back_sample_radius_k: float,
    empty_default_yx: tuple[float, float],
) -> tuple[tuple[float, float], dict]:
    """Locate the embryo's back relative to the yolk.

    The back is the centroid of embryo pixels within a disk centered on the yolk, then
    projected onto the mask. The disk radius is ``k`` times the yolk's equivalent-circle
    radius, so it scales with the animal rather than being a fixed pixel count.

    Returns ``((back_y, back_x), debug)``. Every ``debug["selected"]`` branch other than
    ``"yolk_surrounding_centroid"`` is degenerate: the returned point carries no dorsal
    information, and a DV decision resting on it is not a measurement.
    """
    debug: dict = {}
    arr = np.asarray(mask)

    if yolk_mask is None or np.sum(yolk_mask) == 0:
        fallback = _center_of_mass(arr, empty_default_yx=empty_default_yx)
        debug["selected"] = "no_yolk_fallback"
        debug["back_yx"] = (float(fallback[0]), float(fallback[1]))
        debug["n_pixels_in_disk"] = None
        return fallback, debug

    yolk_com_y, yolk_com_x = _center_of_mass(yolk_mask, empty_default_yx=empty_default_yx)
    debug["yolk_com_yx"] = (float(yolk_com_y), float(yolk_com_x))

    yolk_area = float(np.asarray(yolk_mask).sum())
    r_yolk = float(np.sqrt(yolk_area / np.pi)) if yolk_area > 0 else 0.0
    r_sample = float(back_sample_radius_k) * r_yolk
    debug["r_yolk_px"] = r_yolk
    debug["r_sample_px"] = r_sample

    ys, xs = np.where(arr > 0.5)
    if ys.size == 0:
        debug["selected"] = "empty_mask"
        debug["back_yx"] = (float(yolk_com_y), float(yolk_com_x))
        debug["n_pixels_in_disk"] = 0
        return (yolk_com_y, yolk_com_x), debug

    dy = ys.astype(np.float64) - yolk_com_y
    dx = xs.astype(np.float64) - yolk_com_x
    in_disk = (dy**2 + dx**2) <= (r_sample**2)
    n_in_disk = int(in_disk.sum())
    debug["n_pixels_in_disk"] = n_in_disk

    if n_in_disk == 0:
        debug["selected"] = "empty_disk"
        debug["back_yx"] = (float(yolk_com_y), float(yolk_com_x))
        return (yolk_com_y, yolk_com_x), debug

    back_centroid_y = float(ys[in_disk].mean())
    back_centroid_x = float(xs[in_disk].mean())
    debug["raw_back_centroid_yx"] = (back_centroid_y, back_centroid_x)

    back_y, back_x = _project_point_into_mask_within_disk(
        arr,
        (back_centroid_y, back_centroid_x),
        disk_center_yx=(yolk_com_y, yolk_com_x),
        disk_radius=r_sample,
    )
    debug["selected"] = "yolk_surrounding_centroid"
    debug["back_yx"] = (float(back_y), float(back_x))
    return (back_y, back_x), debug


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

PCA_AXIS = "pca_arctan2_from_horizontal"
REGIONPROPS_AXIS = "skimage_regionprops_from_vertical"


def _degenerate(source: str, *, angle_deg: float = 0.0, axis_convention: str = PCA_AXIS):
    return EmbryoOrientationDecision(
        major_axis_angle_deg=float(angle_deg),
        axis_convention=axis_convention,
        rotation_adjustment_deg=0.0,
        flip_x=False,
        orientation_source=source,
    )


def orientation_policy(
    mask: np.ndarray,
    yolk: Optional[np.ndarray] = None,
    *,
    no_yolk_policy: str = LEGACY_UPPER_LEFT_COM,
    candidate_shape_yx: Optional[tuple[int, int]] = None,
    scale: float = 1.0,
    policy: Optional[EmbryoOrientationPolicy] = None,
    use_pca: bool = True,
    use_yolk: bool = True,
) -> EmbryoOrientationDecision:
    """Decide how to orient an embryo, and report the raw evidence.

    Parameters
    ----------
    mask:
        Embryo mask in SOURCE pixel coordinates.
    yolk:
        Optional yolk mask in the same coordinates. Absent, empty, or ``use_yolk=False``
        routes to ``no_yolk_policy``.
    no_yolk_policy:
        Which existing fallback to reproduce when the anatomical rule cannot run --
        ``legacy_upper_left_com`` (CanonicalAligner's) or ``legacy_vert_ratio`` (the snip
        pipeline's). These do not agree; the caller must choose deliberately. Ignored when
        ``policy`` is given.
    candidate_shape_yx:
        The frame the rot/flip candidates are evaluated on. This is an EVALUATION detail,
        not a rendering target -- the decision itself is canvas-independent -- but it does
        affect the margins and, on the ``legacy_upper_left_com`` path, the decision, since
        that rule scores positions on a canvas. Required for the candidate-based policies;
        unused by ``legacy_vert_ratio``, which measures on its own expanded-bounds frame.
    scale:
        Source-to-evaluation-frame pixel scale. Affects margins, reported in that frame.
    policy:
        Full policy object. Overrides ``no_yolk_policy``.
    use_pca:
        When False the axis is taken as 0 -- the mask is assumed already axis-aligned.
    use_yolk:
        When False the yolk is ignored even if supplied, forcing the fallback with
        ``fallback_reason="yolk_disabled"``.

    Returns
    -------
    EmbryoOrientationDecision
        Rotation adjustment, mirror flag, and the evidence. Nothing is rendered; apply the
        decision in whatever frame you need.
    """
    pol = policy or EmbryoOrientationPolicy(no_yolk_policy=no_yolk_policy)

    arr = np.asarray(mask)
    if arr.size == 0 or arr.sum() == 0:
        return _degenerate(EMPTY_MASK)

    yolk_available = use_yolk and yolk is not None and np.asarray(yolk).sum() > 0

    # The vert_ratio policy owns its own axis measurement and its own frame; it never
    # enters the candidate enumeration. Route to it before doing any of that work.
    if not yolk_available and pol.no_yolk_policy == LEGACY_VERT_RATIO:
        reason = YOLK_DISABLED if (not use_yolk and yolk is not None) else MISSING_YOLK
        return _decide_vert_ratio(arr, fallback_reason=reason)

    if candidate_shape_yx is None:
        raise ValueError(
            "candidate_shape_yx is required for the yolk_back and legacy_upper_left_com "
            "policies: both score candidate POSITIONS, so the evaluation frame changes "
            "the answer. Pass the frame you evaluate on."
        )

    angle_deg, centroid_xy, axis_well_defined = (
        pca_major_axis_angle_deg(arr) if use_pca else (0.0, _xy_centroid(arr), False)
    )
    if use_pca and not axis_well_defined:
        # A single pixel or a perfectly isotropic blob has no major axis. Reporting 0 as
        # though it were a measurement would be a lie.
        return _degenerate(DEGENERATE_MASK, angle_deg=angle_deg)

    h_out, w_out = int(candidate_shape_yx[0]), int(candidate_shape_yx[1])
    grid_center_yx = (h_out / 2.0, w_out / 2.0)

    candidates = enumerate_orientation_candidates(
        arr,
        base_rotation_deg=float(angle_deg),
        scale=float(scale),
        src_center_xy=centroid_xy,
        out_shape_yx=(h_out, w_out),
        companion=yolk if use_yolk else None,
        allow_flip=pol.allow_flip,
    )

    # Per-candidate landmarks. Preserved quirk: when the warped yolk is unusable the
    # "yolk" landmark silently becomes the EMBRYO centroid (AMBIGUITIES note 5).
    landmarks: dict[tuple[int, bool], tuple[tuple[float, float], tuple[float, float], dict]] = {}
    for cand in candidates:
        yolk_w = cand.companion
        usable = use_yolk and yolk_w is not None and yolk_w.sum() > 0
        feature = yolk_w if usable else cand.mask
        yolk_yx = _center_of_mass(feature, empty_default_yx=grid_center_yx)
        back_yx, back_dbg = compute_back_point(
            cand.mask,
            yolk_w if use_yolk else None,
            back_sample_radius_k=pol.back_sample_radius_k,
            empty_default_yx=grid_center_yx,
        )
        landmarks[cand.key] = (yolk_yx, back_yx, back_dbg)

    # "Yolk present" is judged AFTER warping, per candidate (AMBIGUITIES note 5).
    has_yolk_on_grid = use_yolk and yolk is not None and any(
        c.companion is not None and c.companion.sum() > 0 for c in candidates
    )

    keys = list(candidate_keys(allow_flip=pol.allow_flip))
    if has_yolk_on_grid:
        return _decide_yolk_back(
            keys=keys, landmarks=landmarks, angle_deg=float(angle_deg), policy=pol
        )

    if not use_yolk and yolk is not None:
        reason = YOLK_DISABLED
    elif yolk_available:
        # Nonempty in source, gone after warping.
        reason = YOLK_WARPED_OFF_GRID
    else:
        reason = MISSING_YOLK
    return _decide_upper_left_com(
        keys=keys, landmarks=landmarks, angle_deg=float(angle_deg), fallback_reason=reason
    )


def _xy_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(np.asarray(mask))
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _decide_yolk_back(
    *,
    keys: list[tuple[int, bool]],
    landmarks: dict,
    angle_deg: float,
    policy: EmbryoOrientationPolicy,
) -> EmbryoOrientationDecision:
    """Step 1 (yolk farthest LEFT), then Step 2 (back ABOVE yolk)."""
    # --- Step 1: AP axis. argmin over yolk-x; ties go to the first key in contract order.
    yolk_x = {k: landmarks[k][0][1] for k in keys}
    ap_key = min(keys, key=lambda k: yolk_x[k])

    # The margin the legacy code computed and discarded: how decisively did it win?
    rivals = sorted(yolk_x[k] for k in keys if k != ap_key)
    ap_margin = float(rivals[0] - yolk_x[ap_key]) if rivals else None

    # --- Step 2: DV axis. Back above yolk means back_y - yolk_y < 0.
    sel = ap_key
    yolk_yx, back_yx, back_dbg = landmarks[sel]
    dv_override = False
    if back_yx[0] - yolk_yx[0] > 0:
        partner = vertical_flip_partner(sel)
        # With allow_flip=False the partner was never enumerated, so the correction simply
        # cannot be applied -- the mirror it needs does not exist.
        if partner in landmarks:
            sel = partner
            yolk_yx, back_yx, back_dbg = landmarks[sel]
            dv_override = True

    status = back_dbg.get("selected")
    support = back_dbg.get("n_pixels_in_disk")
    return EmbryoOrientationDecision(
        major_axis_angle_deg=angle_deg,
        axis_convention=PCA_AXIS,
        rotation_adjustment_deg=float(sel[0]),
        flip_x=bool(sel[1]),
        orientation_source=YOLK_BACK,
        fallback_policy=None,
        fallback_reason=None,
        yolk_used=True,
        back_used=status == "yolk_surrounding_centroid",
        ap_margin_px=ap_margin,
        dv_margin_px=float(abs(back_yx[0] - yolk_yx[0])),
        back_estimation_status=status,
        back_support_pixels=None if support is None else int(support),
        selected_candidate_index=keys.index(sel),
        dv_override_ap_choice=dv_override,
        yolk_yx=(float(yolk_yx[0]), float(yolk_yx[1])),
        back_yx=(float(back_yx[0]), float(back_yx[1])),
    )


def _decide_upper_left_com(
    *,
    keys: list[tuple[int, bool]],
    landmarks: dict,
    angle_deg: float,
    fallback_reason: str,
) -> EmbryoOrientationDecision:
    """CanonicalAligner's no-yolk rule: the diagonal "mass toward upper-left" score.

    With no yolk both landmark points collapse to statistics of the embryo mask itself, so
    the score reduces to a pull toward the upper-left of the evaluation canvas. That has no
    anatomical meaning -- it is an arbitrary but deterministic tiebreak, and note that it
    depends on the canvas, which the embryo does not. It is preserved verbatim (equal
    weights, as reached from ``embryo_canonical_alignment`` where the weight knobs are
    unreachable) because changing it would change production output.
    """
    best_key = None
    best_score = None
    for k in keys:
        yolk_yx, back_yx, _ = landmarks[k]
        score = (back_yx[1] + back_yx[0]) - (yolk_yx[1] + yolk_yx[0])
        if best_score is None or score > best_score:
            best_score, best_key = score, k

    assert best_key is not None
    yolk_yx, back_yx, back_dbg = landmarks[best_key]
    return EmbryoOrientationDecision(
        major_axis_angle_deg=angle_deg,
        axis_convention=PCA_AXIS,
        rotation_adjustment_deg=float(best_key[0]),
        flip_x=bool(best_key[1]),
        orientation_source=LEGACY_UPPER_LEFT_COM,
        fallback_policy=LEGACY_UPPER_LEFT_COM,
        fallback_reason=fallback_reason,
        yolk_used=False,
        back_used=False,
        back_estimation_status=back_dbg.get("selected"),
        selected_candidate_index=keys.index(best_key),
        yolk_yx=(float(yolk_yx[0]), float(yolk_yx[1])),
        back_yx=(float(back_yx[0]), float(back_yx[1])),
    )


def _decide_vert_ratio(mask: np.ndarray, *, fallback_reason: str) -> EmbryoOrientationDecision:
    """The snip pipeline's no-yolk rule: the ``vert_ratio >= 0.5`` latch.

    Rotate the mask flat by its skimage axis, then ask what FRACTION OF OCCUPIED ROWS lies
    below the centroid. At or above half, keep the angle; otherwise add 180.

    Two things about this are worth stating plainly, because both are load-bearing and
    neither is obvious from the original code:

    - The statistic counts ROWS, not pixels (AMBIGUITIES note 6), so it measures vertical
      extent rather than mass distribution.
    - It is a hard latch on a continuous quantity with no deadband. Embryo 20250416_D09
      produced 0.4967 / 0.5068 / 0.4843 within one timelapse and flipped 180 degrees three
      times. ``vert_ratio_margin`` is returned so a caller can see how close to 0.5 it was.

    This path is canvas-free: the rotation expands its own bounds, so nothing here depends
    on a target grid. It also never mirrors -- ``flip_x`` is always False, because the
    original has no mirror candidate at all.
    """
    # Imported lazily: this keeps the anatomical path free of a skimage dependency, and the
    # convention mismatch (note 7) means only this policy may use regionprops.
    from skimage.measure import regionprops

    arr = np.asarray(mask)
    rp = regionprops(arr.astype(int))
    if not rp:
        return _degenerate(EMPTY_MASK, axis_convention=REGIONPROPS_AXIS)

    # skimage measures from the vertical axis, in radians. The pipeline rotates by -angle.
    angle_rad = float(rp[0].orientation)
    angle_deg = float(np.rad2deg(angle_rad))

    rotated = _rotate_expand(arr, -angle_deg)
    com = _center_of_mass(rotated, empty_default_yx=(0.0, 0.0))

    y_indices = np.where(np.max(rotated, axis=1))[0]
    if len(y_indices) == 0:
        vert_ratio = None
        adjustment = 0.0
    else:
        vert_ratio = float(np.sum(y_indices > com[0]) / len(y_indices))
        adjustment = 0.0 if vert_ratio >= 0.5 else 180.0

    return EmbryoOrientationDecision(
        # Reported in the pipeline's own sign convention: it rotates by -orientation, so
        # the value that composes correctly is the negated angle.
        major_axis_angle_deg=-angle_deg,
        axis_convention=REGIONPROPS_AXIS,
        rotation_adjustment_deg=adjustment,
        flip_x=False,
        orientation_source=LEGACY_VERT_RATIO,
        fallback_policy=LEGACY_VERT_RATIO,
        fallback_reason=fallback_reason,
        yolk_used=False,
        back_used=False,
        vert_ratio=vert_ratio,
        vert_ratio_margin=None if vert_ratio is None else float(abs(vert_ratio - 0.5)),
    )


def _rotate_expand(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate about the image center, expanding bounds so nothing is cropped.

    Byte-compatible with ``data_pipeline...rotation.rotate_image``, including its integer
    truncation of the expanded bounds -- the vert_ratio statistic is measured on this exact
    raster, so any difference here would change the decision.
    """
    import cv2

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    abs_cos, abs_sin = abs(m[0, 0]), abs(m[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    m[0, 2] += bound_w / 2 - center[0]
    m[1, 2] += bound_h / 2 - center[1]
    return cv2.warpAffine(image, m, (bound_w, bound_h))


# Backwards-compatible alias for the descriptive verb.
decide_embryo_orientation = orientation_policy
