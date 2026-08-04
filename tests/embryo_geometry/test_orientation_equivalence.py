"""THE EXTRACTION CHECK, and it is TWO-SIDED.

This is an extraction, so the only acceptable behavioral difference is none -- for BOTH
existing callers, which do not agree with each other:

  side A: analysis  -- CanonicalAligner._coarse_candidate_select
                       (yolk_back, and legacy_upper_left_com when the yolk is missing)
  side B: pipeline  -- get_embryo_rotation_angle in snip_processing/rotation.py
                       (legacy_vert_ratio when the yolk is missing)

Both sides must match on ``(rotation, flip_x)``. If one fails, the extraction changed
policy -- find what diverged; do not adjust the expected value.

WHAT THIS FILE CANNOT DETECT -- read before trusting a green run here.
======================================================================
Every assertion below compares one of OUR implementations against another of OUR
implementations. That makes this file blind, BY CONSTRUCTION, to any defect the two sides
SHARE. That is not hypothetical. ``image_geometry.candidates`` and
``analyze...canonical.CanonicalAligner`` both built placement affines under the naive rule
``x_out = scale * x_src``, when a pixel index is a sample CENTERED at ``x + 0.5`` and the
correct rule is ``x_out = scale * (x_src + 0.5) - 0.5``. The two differ by ``(scale-1)/2``
-- a systematic, direction-consistent shift applied to every embryo. This suite passed at
abs=1e-6 the entire time, because both clocks were five minutes slow and therefore in
perfect agreement with each other.

The defect was found from outside and fixed on both sides (image_geometry in 8afce454,
canonical.py in bfaedd02). Fixing ONE side first made 26 tests here fail; that is the
SIGNATURE of a mirrored defect, and the correct response was to fix the second side, NOT to
edit the expected values. Adjusting them would have restored agreement and re-hidden the
bug permanently -- and the resulting diff would have looked nearly identical to the real fix.

So: a failure here means the two sides DISAGREE. It does not tell you which one is RIGHT.
Rightness is established only by ``tests/image_geometry/test_pixel_center_invariant.py``,
which derives every expected value from the convention itself and never compares two of our
implementations to each other. The two files are complementary and neither substitutes for
the other: the invariant suite anchors each side to the intended geometry, and only then
does the agreement asserted here carry information. If you are about to change a number in
THIS file to make a test pass, stop -- the invariant suite is where the question you are
actually asking gets answered.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from analyze.utils.coord.grids.canonical import CanonicalAligner
from data_pipeline.object_extraction.snip_processing.rotation import (
    get_embryo_rotation_angle,
)
from embryo_geometry import (
    LEGACY_UPPER_LEFT_COM,
    LEGACY_VERT_RATIO,
    YOLK_BACK,
    EmbryoOrientationPolicy,
    orientation_policy,
)
from image_geometry.candidates import pca_major_axis_angle_deg

from synthetic import make_embryo, orientation_cases

GRID_HW = (256, 576)
_IDS = lambda v: v if isinstance(v, str) else ""


# ===========================================================================
# SIDE A -- analysis: CanonicalAligner._coarse_candidate_select
# ===========================================================================


def _legacy_canonical(mask, yolk, *, um_per_px=10.0, allow_flip=True, use_yolk=True):
    """Drive the legacy selector exactly as ``embryo_canonical_alignment`` does."""
    aligner = CanonicalAligner(
        target_shape_hw=GRID_HW, target_um_per_pixel=10.0, allow_flip=allow_flip
    )
    scale = float(um_per_px) / aligner.target_res
    angle_deg, (cx, cy), _ = aligner._pca_angle_deg(mask)
    target_angle = 0.0 if aligner.is_landscape else 90.0
    rotation_needed = float(angle_deg) - target_angle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rot, flip, yolk_yx, back_yx = aligner._coarse_candidate_select(
            mask, yolk if use_yolk else None, rotation_needed, scale, cx, cy, use_yolk=use_yolk
        )
    return rot, flip, yolk_yx, back_yx


def _new_canonical(mask, yolk, *, um_per_px=10.0, allow_flip=True, use_yolk=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return orientation_policy(
            mask,
            yolk,
            candidate_shape_yx=GRID_HW,
            scale=float(um_per_px) / 10.0,
            policy=EmbryoOrientationPolicy(
                no_yolk_policy=LEGACY_UPPER_LEFT_COM, allow_flip=allow_flip
            ),
            use_yolk=use_yolk,
        )


@pytest.mark.parametrize("name,mask,yolk", orientation_cases(), ids=_IDS)
def test_sideA_matches_canonical_aligner(name, mask, yolk):
    """The composed rotation and the mirror flag must be identical."""
    legacy_rot, legacy_flip, _, _ = _legacy_canonical(mask, yolk)
    new = _new_canonical(mask, yolk)
    assert new.flip_x == legacy_flip, f"{name}: flip diverged"
    # The decision is canvas-independent, so the caller composes against its own target
    # angle -- here 0, the landscape value CanonicalAligner uses for a 256x576 canvas.
    assert new.final_rotation_deg(target_angle_deg=0.0) == pytest.approx(
        legacy_rot, abs=1e-9
    ), f"{name}: rotation diverged"


@pytest.mark.parametrize("name,mask,yolk", orientation_cases(), ids=_IDS)
def test_sideA_landmarks_match(name, mask, yolk):
    """Equal rotations could in principle come from different landmarks; close that gap."""
    _, _, legacy_yolk_yx, legacy_back_yx = _legacy_canonical(mask, yolk)
    new = _new_canonical(mask, yolk)
    assert new.yolk_yx == pytest.approx(legacy_yolk_yx, abs=1e-6), f"{name}: yolk landmark"
    assert new.back_yx == pytest.approx(legacy_back_yx, abs=1e-6), f"{name}: back landmark"


@pytest.mark.parametrize("um_per_px", [5.0, 10.0, 20.0])
def test_sideA_across_scales(um_per_px):
    """Scale enters the affine and can change which candidate wins."""
    mask, yolk = make_embryo(angle_deg=18.0, yolk_at="right", dorsal_side="down")
    legacy_rot, legacy_flip, _, _ = _legacy_canonical(mask, yolk, um_per_px=um_per_px)
    new = _new_canonical(mask, yolk, um_per_px=um_per_px)
    assert new.flip_x == legacy_flip
    assert new.final_rotation_deg() == pytest.approx(legacy_rot)


def test_sideA_with_flip_disabled():
    """allow_flip=False drops the mirrors AND thereby disables the DV correction."""
    mask, yolk = make_embryo(angle_deg=7.0, yolk_at="right", dorsal_side="down")
    legacy_rot, legacy_flip, _, _ = _legacy_canonical(mask, yolk, allow_flip=False)
    new = _new_canonical(mask, yolk, allow_flip=False)
    assert new.flip_x is False and legacy_flip is False
    assert new.final_rotation_deg() == pytest.approx(legacy_rot)


def test_sideA_no_yolk_uses_upper_left_com():
    """CanonicalAligner's own fallback, reproduced and labelled."""
    mask, yolk = make_embryo(angle_deg=33.0, yolk_at=None)
    legacy_rot, legacy_flip, _, _ = _legacy_canonical(mask, yolk)
    new = _new_canonical(mask, yolk)
    assert new.orientation_source == LEGACY_UPPER_LEFT_COM
    assert new.fallback_policy == LEGACY_UPPER_LEFT_COM
    assert new.fallback_reason == "missing_yolk"
    assert new.flip_x == legacy_flip
    assert new.final_rotation_deg() == pytest.approx(legacy_rot)


def test_sideA_yolk_explicitly_ignored():
    mask, yolk = make_embryo(angle_deg=-14.0, yolk_at="left", dorsal_side="up")
    legacy_rot, legacy_flip, _, _ = _legacy_canonical(mask, yolk, use_yolk=False)
    new = _new_canonical(mask, yolk, use_yolk=False)
    assert new.flip_x == legacy_flip
    assert new.final_rotation_deg() == pytest.approx(legacy_rot)
    assert new.fallback_reason == "yolk_disabled"


def test_shared_pca_helper_matches_legacy_pca():
    aligner = CanonicalAligner(target_shape_hw=GRID_HW)
    for _n, mask, _y in orientation_cases():
        l_angle, l_centroid, l_ok = aligner._pca_angle_deg(mask)
        angle, centroid, ok = pca_major_axis_angle_deg(mask)
        assert angle == pytest.approx(l_angle, abs=1e-9)
        assert centroid == pytest.approx(l_centroid, abs=1e-6)
        assert ok == l_ok


# ===========================================================================
# SIDE B -- pipeline: get_embryo_rotation_angle (legacy_vert_ratio)
# ===========================================================================

_NO_YOLK_ANGLES = [0.0, 12.0, -25.0, 40.0, 88.0, 155.0, -60.0, 30.0, 7.5, 123.0]


@pytest.mark.parametrize("angle", _NO_YOLK_ANGLES)
def test_sideB_matches_pipeline_vert_ratio(angle):
    """The no-yolk pipeline rule, reproduced to the degree.

    ``get_embryo_rotation_angle`` returns RADIANS of total rotation; the engine reports the
    axis and the 0/180 adjustment separately, so compose before comparing.
    """
    mask, yolk = make_embryo(angle_deg=angle, yolk_at=None)
    legacy_rad = get_embryo_rotation_angle(mask.astype(np.uint8), yolk.astype(np.uint8))

    new = orientation_policy(mask, yolk, no_yolk_policy=LEGACY_VERT_RATIO)
    assert new.orientation_source == LEGACY_VERT_RATIO
    assert new.flip_x is False  # the pipeline rule has no mirror candidate at all
    assert np.rad2deg(legacy_rad) == pytest.approx(new.final_rotation_deg(), abs=1e-9)


@pytest.mark.parametrize("angle", _NO_YOLK_ANGLES)
def test_sideB_vert_ratio_evidence_is_populated(angle):
    """The statistic that decides must be reported, not just its verdict."""
    mask, yolk = make_embryo(angle_deg=angle, yolk_at=None)
    new = orientation_policy(mask, yolk, no_yolk_policy=LEGACY_VERT_RATIO)
    assert new.vert_ratio is not None
    assert 0.0 <= new.vert_ratio <= 1.0
    assert new.vert_ratio_margin == pytest.approx(abs(new.vert_ratio - 0.5))
    # The latch is exactly `vert_ratio >= 0.5`; the reported adjustment must agree with it.
    expected = 0.0 if new.vert_ratio >= 0.5 else 180.0
    assert new.rotation_adjustment_deg == expected


def test_sideB_is_canvas_independent():
    """The vert_ratio path expands its own bounds, so no grid shape may be required."""
    mask, yolk = make_embryo(angle_deg=21.0, yolk_at=None)
    a = orientation_policy(mask, yolk, no_yolk_policy=LEGACY_VERT_RATIO)
    # No candidate_shape_yx passed at all -- this must not raise.
    assert a.orientation_source == LEGACY_VERT_RATIO
    b = orientation_policy(
        mask, yolk, no_yolk_policy=LEGACY_VERT_RATIO, candidate_shape_yx=(1024, 1024)
    )
    assert (a.rotation_adjustment_deg, a.vert_ratio) == (
        b.rotation_adjustment_deg,
        b.vert_ratio,
    )


def test_sideB_near_the_latch_still_matches_legacy():
    """Near vert_ratio == 0.5 the rule is a coin flip; equivalence must hold there too.

    This is the regime where 20250416_D09 flipped three times in one timelapse
    (0.4967 / 0.5068 / 0.4843). If the extraction diverges anywhere, it diverges here.
    """
    checked = 0
    for angle in np.linspace(-90, 90, 61):
        for radii in [(22.0, 110.0), (30.0, 60.0), (18.0, 140.0)]:
            mask, yolk = make_embryo(
                angle_deg=float(angle), body_radii_yx=radii, yolk_at=None
            )
            new = orientation_policy(mask, yolk, no_yolk_policy=LEGACY_VERT_RATIO)
            if new.vert_ratio_margin is None or new.vert_ratio_margin > 0.05:
                continue
            legacy = get_embryo_rotation_angle(mask.astype(np.uint8), yolk.astype(np.uint8))
            assert np.rad2deg(legacy) == pytest.approx(new.final_rotation_deg(), abs=1e-9), (
                f"diverged near the latch at angle={angle}, ratio={new.vert_ratio}"
            )
            checked += 1
    assert checked > 0, "no near-latch cases generated; the test proved nothing"


# ===========================================================================
# The two sides are genuinely different policies
# ===========================================================================


def test_the_two_no_yolk_policies_actually_disagree():
    """Guards the whole reason ``no_yolk_policy`` is a parameter.

    If these ever agreed everywhere, a single shared fallback would be free. They do not,
    so picking one is a behavior change that must be made deliberately -- not absorbed
    into an extraction.
    """
    disagreements = 0
    for angle in np.linspace(-90, 90, 37):
        mask, yolk = make_embryo(angle_deg=float(angle), yolk_at=None)
        com = orientation_policy(
            mask, yolk, no_yolk_policy=LEGACY_UPPER_LEFT_COM, candidate_shape_yx=GRID_HW
        )
        vr = orientation_policy(mask, yolk, no_yolk_policy=LEGACY_VERT_RATIO)
        # Compare the pose choice modulo the differing axis conventions: what differs is
        # whether each rule adds the 180, and whether a mirror is applied.
        if (com.rotation_adjustment_deg, com.flip_x) != (vr.rotation_adjustment_deg, vr.flip_x):
            disagreements += 1
    assert disagreements > 0, (
        "The two production no-yolk rules agreed on every synthetic case. That would be "
        "surprising; re-check the fixtures before concluding the fallbacks are unifiable."
    )


# ===========================================================================
# The agreement above is only meaningful if both sides are independently anchored
# ===========================================================================


def test_both_sides_are_anchored_to_the_pixel_center_invariant():
    """The guard that would have caught the mirrored defect FROM THIS FILE.

    Everything else here asks "do the two implementations agree?". That question has a
    passing answer even when both are wrong in the same way, which is exactly what happened
    with the naive ``x_out = scale * x_src`` placement rule. This test asks the different
    question: "is each side anchored to the intended convention?" -- by checking, for BOTH
    sides, against a value derived from the convention rather than from either implementation.

    Kept deliberately small. The thorough treatment is in
    tests/image_geometry/test_pixel_center_invariant.py; this exists so that a reader who
    only ever runs the equivalence suite cannot walk away believing agreement alone is proof.
    """
    from analyze.utils.coord.grids.canonical import CanonicalAligner
    from image_geometry.candidates import centered_placement_affine

    scale = 0.3231  # the production 3.2308 -> 10.0 um/px ratio
    x_src = 7.0
    # Derived from the convention itself, not read off either implementation.
    expected = scale * (x_src + 0.5) - 0.5
    naive = scale * x_src
    assert abs(expected - naive) > 1.0 / 32.0, "test scale cannot resolve the defect"

    aligner = CanonicalAligner(target_shape_hw=GRID_HW)
    a_side = aligner._placement_affine(0.0, 0.0, 0.0, scale)
    assert a_side[0, 0] * x_src + a_side[0, 2] - aligner.W / 2 == pytest.approx(
        expected, abs=1e-9
    ), "analysis side is not on the pixel-center convention"

    b_side = centered_placement_affine(
        rotation_deg=0.0, scale=scale, src_center_xy=(0.0, 0.0), out_shape_yx=GRID_HW
    )
    assert b_side[0, 0] * x_src + b_side[0, 2] - GRID_HW[1] / 2 == pytest.approx(
        expected, abs=1e-9
    ), "image_geometry side is not on the pixel-center convention"
