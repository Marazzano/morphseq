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

THERE ARE CURRENTLY *FOUR* DISTINCT RULES IN PRODUCTION, NOT ONE.

    yolk present:
      1. yolk-left AP (canvas argmin) + back-above-yolk DV  (CanonicalAligner)
      4. embryo COM vs yolk COM along the body axis          (snip pipeline)
    yolk missing:
      2. -(com_y + com_x) "mass toward upper-left"           (CanonicalAligner)
      3. vert_ratio >= 0.5                                   (snip pipeline)

Rules 2 and 3 are BOTH "the no-yolk policy", of different callers, and they do not agree.
Rules 1 and 4 are BOTH "the yolk-present policy", and they do not agree either. So BOTH
policy slots are parameters here, each from a closed set, and every decision reports which
one ran and why. Converging on a single rule then becomes a later, deliberate,
attributable change rather than something that happens by accident inside an extraction.

RULE 1's AP HALF READS THE WRONG FRAME
--------------------------------------
It is tempting to rank rule 1 above rule 4 because it is the more elaborate and is
yolk-aware. It is yolk-aware, but it reads the yolk in CANVAS COORDINATES: "pick the
candidate whose yolk centroid has the smallest x on the output grid". That is a statement
about the rendering, not about the animal. Rule 4 compares two landmarks that are both ON
THE EMBRYO -- a signed vector between embryo COM and yolk COM -- and is therefore
frame-independent, and structurally the better rule.

This is not merely aesthetic. Measured over 132 synthetic cases (see
``test_orientation_policy.py``):

    rule 1: after the DV step, its own AP predicate still holds in  66/132 cases
    rule 4: after the DV step, its own AP predicate still holds in 132/132 cases

Under rule 1, DV silently breaks AP in exactly half the cases -- the two are not jointly
satisfiable and the later one wins. Under rule 4 they ARE jointly satisfiable, because a
vertical flip negates the y-components of both landmarks together and so PRESERVES the
sign of the AP vector (verified: 528/528 partner pairs). AP and DV become independent
axes instead of two rules fighting over one answer.

BUT THE TWO RULES SELECT THE SAME POSE TODAY -- a negative result worth stating plainly.
``enumerate_orientation_candidates`` rotates about the mask centroid and lands it at the
canvas centre, so all four candidates share an embryo COM and their yolk-x values form two
mirror pairs symmetric about it (measured asymmetry: 0.0000 px). Under that symmetry the
canvas centre IS the embryo COM, so "leftmost yolk on the canvas" and "yolk anterior to
embryo COM" partition the candidates identically. Rule 1 is accidentally right.

So rule 1's frame-dependence is a LATENT defect, not an active one, and switching a caller
to rule 4 should be a no-op on this call path. The defect surfaces only for a caller that
evaluates candidates without centring the embryo, or that compares poses across
differently-placed embryos -- and rule 1's correctness silently depends on a centring
invariant that nothing states or enforces. That is the case for keeping rule 4, and it is
weaker than "rule 1 is producing wrong answers today", which would not be true.

Rule 4 is nonetheless still a latch (``>= 0``, no deadband). Its margin is reported as
``ap_com_margin_px``, a DIFFERENT quantity from ``ap_margin_px`` -- see below.

THE AP AXIS IS NOT THE SAME AXIS IN BOTH STACKS
-----------------------------------------------
This is the likely mechanism by which the two stacks drifted apart, and it is invisible
unless you check which way the body ends up pointing.

``rotation.py`` compares index ``[0]`` -- the ROW, y -- which looks like a dorsal/ventral
test. It is not. skimage's ``regionprops.orientation`` is measured FROM THE VERTICAL, so
rotating by ``-orientation`` leaves the body's LONG axis along y (measured: post-rotation
extent is ~218 rows tall by ~61 columns wide across every test angle). So its ``[0]``
comparison is along the long axis, and rule 4 constrains AP.

``CanonicalAligner`` rotates the long axis onto x for its landscape canvas, and its AP
test reads ``yolk_yx[1]`` -- the COLUMN, x. Also the long axis, also AP.

So both yolk-present rules DO constrain the same anatomical axis, AP -- but they express
it through OPPOSITE array indices, because the two stacks leave the body pointing in
perpendicular directions. Comparing ``[0]`` in one stack against ``[0]`` in the other
compares AP against DV. Any future unification must convert through the anatomical axis,
never by matching index positions.

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
The margins are RAW and not commensurable -- not even the two AP ones. ``ap_margin_px`` is
the winner-vs-runner-up gap in CANVAS yolk-x under rule 1; ``ap_com_margin_px`` is
``|embryo_com - yolk_com|`` along the body axis under rule 4. Both are "pixels" and both
are "the AP margin", but they measure different geometry -- a gap between CANDIDATES
versus a distance between LANDMARKS on the animal -- so they are separate fields and must
never be substituted for one another. ``dv_margin_px`` is ``|back_y - yolk_y|``, and
``vert_ratio_margin`` is ``|vert_ratio - 0.5|``, dimensionless. A 10px margin of one kind
is not "the same amount of evidence" as 10px of another, let alone as 0.03 of a
vert-ratio, and combining them would need calibration against embryo size, yolk radius,
and empirical error rates that nobody has done. So no probability and no score is
returned -- only the measurements, for a caller to threshold deliberately.

AMBIGUITIES FOUND WHILE STATING THE RULES PRECISELY
---------------------------------------------------
1. Every decision point is an UNREPORTED THRESHOLD LATCH. AP is an ``argmin`` with no
   margin requirement; DV is a strict sign test with no deadband; ``vert_ratio >= 0.5`` is
   a hard latch on a continuous statistic. Distance from the cliff was computed and thrown
   away. Embryo ``20250416_D09`` flips 180 degrees THREE times within one timelapse, with
   ``vert_ratio`` of 0.4967 / 0.5068 / 0.4843 -- three coin flips reported as three
   confident decisions. That is why the margins are returned.

2. UNDER RULE 1 ONLY, STEP 2 CAN UNDO STEP 1. The vertical-flip partner mirrors
   horizontally too, so enforcing dorsal-up can move the yolk to the RIGHT, violating the
   rule Step 1 just applied. The two are not jointly satisfiable and DV silently wins, in
   exactly half the measured cases. Yolk-left is therefore NOT an invariant of rule 1's
   output, despite being stated as the rule. ``dv_override_ap_choice`` reports it.

   This is a defect of the CANVAS FRAME, not of two-step selection as such: rule 4's AP
   predicate survives its own DV step in every measured case. Choosing the right frame
   dissolves the conflict rather than trading it for another.

3. The DV test is ``> 0``, so ``back_y == yolk_y`` exactly -- the back level with the
   yolk, carrying no dorsal information -- silently keeps Step 1's answer instead of
   flagging the axis as undetermined. ``dv_margin_px == 0.0`` makes it visible.

4. THE TWO YOLK-PRESENT RULES DISAGREE IN KIND, AND ONE IS BETTER. CanonicalAligner picks
   the candidate whose yolk sits leftmost ON THE CANVAS; the snip pipeline compares the
   yolk to the EMBRYO CENTROID, a signed relation between two points on the animal. Both
   are preserved and selectable, but they are not equally principled: only the second is
   frame-independent. See "RULE 1's AP HALF READS THE WRONG FRAME" above.

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

#: CanonicalAligner's yolk-present rule: AP by argmin over yolk-x IN CANVAS COORDINATES,
#: then back-above-yolk for DV. The AP half reads the yolk's position on the output grid,
#: so it is a property of the RENDERING, not of the animal.
YOLK_BACK = "yolk_back"
#: The snip pipeline's yolk-present rule, and the anatomically correct one: AP by the
#: SIGNED comparison of two landmarks ON THE EMBRYO (embryo COM vs yolk COM along the
#: body axis). Frame-independent. See the AP AXIS section in the module docstring.
YOLK_VS_EMBRYO_COM = "yolk_vs_embryo_com"

YOLK_PRESENT_POLICIES = (YOLK_BACK, YOLK_VS_EMBRYO_COM)

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
        RULE 1 ONLY. Winner-vs-runner-up gap on yolk-x, in CANVAS pixels of the evaluation
        frame. Small means the AP call was a coin flip. Not comparable to
        ``ap_com_margin_px`` despite both being pixels and both being "the AP margin".
    ap_com_margin_px:
        RULE 4 ONLY. ``abs(embryo_com - yolk_com)`` along the body axis, in pixels. A
        distance between two landmarks ON THE ANIMAL, so it is frame-independent -- unlike
        ``ap_margin_px``. Still a latch with no deadband; this is how far from it you were.
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
    ap_com_margin_px: Optional[float] = None
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
    #: Embryo centroid -- rule 4's second AP landmark, and the one that makes its test
    #: frame-independent. Populated only where it was actually used.
    embryo_com_yx: Optional[tuple[float, float]] = None

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

    BOTH policy slots are explicit, because in both cases the two existing callers disagree
    and a default would silently crown a winner. They default to CanonicalAligner's
    behavior so the analysis path is unchanged; the pipeline must pass
    ``yolk_present_policy="yolk_vs_embryo_com"`` and ``no_yolk_policy="legacy_vert_ratio"``.

    Note that ``yolk_present_policy`` defaulting to ``yolk_back`` is a COMPATIBILITY
    choice, not an endorsement: ``yolk_vs_embryo_com`` is the better rule (frame-
    independent, and jointly satisfiable with the DV step). See the module docstring.

    ``allow_flip=False`` removes the mirror candidates, which also disables the Step-2 DV
    correction -- the vertical-flip partner of a non-mirrored candidate IS a mirrored one,
    so without mirrors the correction cannot be expressed.
    """

    no_yolk_policy: str = LEGACY_UPPER_LEFT_COM
    yolk_present_policy: str = YOLK_BACK
    allow_flip: bool = True
    back_sample_radius_k: float = 1.75

    def __post_init__(self) -> None:
        if self.no_yolk_policy not in NO_YOLK_POLICIES:
            raise ValueError(
                f"no_yolk_policy must be one of {NO_YOLK_POLICIES!r}, got {self.no_yolk_policy!r}"
            )
        if self.yolk_present_policy not in YOLK_PRESENT_POLICIES:
            raise ValueError(
                f"yolk_present_policy must be one of {YOLK_PRESENT_POLICIES!r}, "
                f"got {self.yolk_present_policy!r}"
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
    yolk_present_policy: str = YOLK_BACK,
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
    yolk_present_policy:
        Which yolk-present rule to apply -- ``yolk_back`` (CanonicalAligner's canvas-frame
        AP argmin plus back-above-yolk DV) or ``yolk_vs_embryo_com`` (the snip pipeline's
        signed embryo-COM-vs-yolk-COM AP test, which is frame-independent and the better
        rule). These do not agree either. Ignored when ``policy`` is given.
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
    pol = policy or EmbryoOrientationPolicy(
        no_yolk_policy=no_yolk_policy, yolk_present_policy=yolk_present_policy
    )

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

    # Per-candidate landmarks: (yolk_yx, back_yx, back_debug, embryo_com_yx).
    # Preserved quirk: when the warped yolk is unusable the "yolk" landmark silently
    # becomes the EMBRYO centroid (AMBIGUITIES note 5). The embryo COM is tracked
    # separately regardless, because rule 4 needs it as a landmark in its own right.
    landmarks: dict[
        tuple[int, bool],
        tuple[tuple[float, float], tuple[float, float], dict, tuple[float, float]],
    ] = {}
    for cand in candidates:
        yolk_w = cand.companion
        usable = use_yolk and yolk_w is not None and yolk_w.sum() > 0
        feature = yolk_w if usable else cand.mask
        yolk_yx = _center_of_mass(feature, empty_default_yx=grid_center_yx)
        embryo_com_yx = _center_of_mass(cand.mask, empty_default_yx=grid_center_yx)
        back_yx, back_dbg = compute_back_point(
            cand.mask,
            yolk_w if use_yolk else None,
            back_sample_radius_k=pol.back_sample_radius_k,
            empty_default_yx=grid_center_yx,
        )
        landmarks[cand.key] = (yolk_yx, back_yx, back_dbg, embryo_com_yx)

    # "Yolk present" is judged AFTER warping, per candidate (AMBIGUITIES note 5).
    has_yolk_on_grid = use_yolk and yolk is not None and any(
        c.companion is not None and c.companion.sum() > 0 for c in candidates
    )

    keys = list(candidate_keys(allow_flip=pol.allow_flip))
    if has_yolk_on_grid:
        if pol.yolk_present_policy == YOLK_VS_EMBRYO_COM:
            return _decide_yolk_vs_embryo_com(
                keys=keys, landmarks=landmarks, angle_deg=float(angle_deg)
            )
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
    yolk_yx, back_yx, back_dbg, _ = landmarks[sel]
    dv_override = False
    if back_yx[0] - yolk_yx[0] > 0:
        partner = vertical_flip_partner(sel)
        # With allow_flip=False the partner was never enumerated, so the correction simply
        # cannot be applied -- the mirror it needs does not exist.
        if partner in landmarks:
            sel = partner
            yolk_yx, back_yx, back_dbg, _ = landmarks[sel]
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


def _decide_yolk_vs_embryo_com(
    *,
    keys: list[tuple[int, bool]],
    landmarks: dict,
    angle_deg: float,
) -> EmbryoOrientationDecision:
    """Rule 4: AP from the SIGNED embryo-COM-vs-yolk-COM relation, then back-above-yolk DV.

    This is ``rotation.py:79`` generalized to the candidate enumeration. The original asks
    ``(embryo_com[0] - yolk_com[0]) >= 0`` on a frame where the body axis is VERTICAL, so
    the comparison is along the long axis. Here the candidates are evaluated on a landscape
    frame where the long axis is x, so the same anatomical test reads index ``[1]``. Same
    axis, different index -- see "THE AP AXIS IS NOT THE SAME AXIS IN BOTH STACKS".

    Both landmarks lie ON THE ANIMAL, so unlike rule 1 the predicate does not depend on
    where the canvas happens to put the embryo. That also makes it compatible with the DV
    step: a vertical flip negates both y-components together, leaving the sign of the AP
    vector untouched, so AP and DV constrain independent axes and cannot fight. Measured:
    the AP predicate survives the DV step in 132/132 cases, versus 66/132 under rule 1.

    Selection among candidates satisfying AP is by first-wins contract order, matching the
    original's binary keep-or-add-180 structure -- it never ranked candidates.
    """
    ap_ok = [k for k in keys if (landmarks[k][3][1] - landmarks[k][0][1]) >= 0]
    # The predicate is a strict dichotomy over a closed candidate set, so it cannot be
    # unsatisfiable; falling back to all keys would silently mask a real bug if it were.
    assert ap_ok, "no candidate satisfies the AP predicate; the candidate set is malformed"
    sel = ap_ok[0]

    yolk_yx, back_yx, back_dbg, embryo_com_yx = landmarks[sel]
    dv_override = False
    if back_yx[0] - yolk_yx[0] > 0:
        partner = vertical_flip_partner(sel)
        if partner in landmarks:
            sel = partner
            yolk_yx, back_yx, back_dbg, embryo_com_yx = landmarks[sel]
            dv_override = True

    status = back_dbg.get("selected")
    support = back_dbg.get("n_pixels_in_disk")
    return EmbryoOrientationDecision(
        major_axis_angle_deg=angle_deg,
        axis_convention=PCA_AXIS,
        rotation_adjustment_deg=float(sel[0]),
        flip_x=bool(sel[1]),
        orientation_source=YOLK_VS_EMBRYO_COM,
        yolk_used=True,
        back_used=status == "yolk_surrounding_centroid",
        # Rule 4's own margin. Deliberately NOT ap_margin_px: that field means a
        # candidate-separation gap in canvas coordinates and is not this quantity.
        ap_com_margin_px=float(abs(embryo_com_yx[1] - yolk_yx[1])),
        dv_margin_px=float(abs(back_yx[0] - yolk_yx[0])),
        back_estimation_status=status,
        back_support_pixels=None if support is None else int(support),
        selected_candidate_index=keys.index(sel),
        dv_override_ap_choice=dv_override,
        yolk_yx=(float(yolk_yx[0]), float(yolk_yx[1])),
        back_yx=(float(back_yx[0]), float(back_yx[1])),
        embryo_com_yx=(float(embryo_com_yx[0]), float(embryo_com_yx[1])),
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
        yolk_yx, back_yx, _dbg, _com = landmarks[k]
        score = (back_yx[1] + back_yx[0]) - (yolk_yx[1] + yolk_yx[0])
        if best_score is None or score > best_score:
            best_score, best_key = score, k

    assert best_key is not None
    yolk_yx, back_yx, back_dbg, _ = landmarks[best_key]
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
