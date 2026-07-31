"""Behavioral coverage the orientation rules never had.

``CanonicalAligner`` had ZERO tests for orientation, so nothing pinned the anatomical rule
itself -- only that some pose came out. These tests assert what the rule IS, on synthetic
masks where the right answer is known by construction, and that the evidence fields are
actually populated rather than left None.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from embryo_geometry import (
    DEGENERATE_MASK,
    EMPTY_MASK,
    LEGACY_UPPER_LEFT_COM,
    LEGACY_VERT_RATIO,
    NO_YOLK_POLICIES,
    YOLK_BACK,
    EmbryoOrientationPolicy,
    compute_back_point,
    orientation_policy,
)
from image_geometry.candidates import (
    CANDIDATE_KEYS,
    candidate_keys,
    enumerate_orientation_candidates,
    pca_major_axis_angle_deg,
    vertical_flip_partner,
)

from synthetic import make_embryo

GRID_HW = (256, 576)


def _decide(mask, yolk, **kw):
    kw.setdefault("candidate_shape_yx", GRID_HW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return orientation_policy(mask, yolk, **kw)


# ---------------------------------------------------------------------------
# Candidate enumeration (image_geometry mechanics)
# ---------------------------------------------------------------------------


def test_enumeration_yields_four_distinct_poses_in_contract_order():
    """The order is contractual: argmin ties are resolved first-wins."""
    mask, yolk = make_embryo(angle_deg=15.0, yolk_at="left")
    cands = enumerate_orientation_candidates(
        mask,
        base_rotation_deg=15.0,
        scale=1.0,
        src_center_xy=(200.0, 100.0),
        out_shape_yx=GRID_HW,
        companion=yolk,
    )
    assert [c.key for c in cands] == list(CANDIDATE_KEYS)
    assert [c.key for c in cands] == [(0, False), (0, True), (180, False), (180, True)]
    # Distinct poses, not four copies of the same raster.
    flat = [c.mask.tobytes() for c in cands]
    assert len(set(flat)) == 4


def test_enumeration_respects_allow_flip():
    mask, _ = make_embryo()
    cands = enumerate_orientation_candidates(
        mask,
        base_rotation_deg=0.0,
        scale=1.0,
        src_center_xy=(200.0, 100.0),
        out_shape_yx=GRID_HW,
        allow_flip=False,
    )
    assert [c.key for c in cands] == [(0, False), (180, False)]
    assert candidate_keys(allow_flip=False) == ((0, False), (180, False))


def test_companion_is_transformed_identically_to_the_mask():
    """The yolk must ride along, or the AP rule reads a stale position."""
    mask, yolk = make_embryo(angle_deg=0.0, yolk_at="left")
    for c in enumerate_orientation_candidates(
        mask,
        base_rotation_deg=0.0,
        scale=1.0,
        src_center_xy=(200.0, 100.0),
        out_shape_yx=GRID_HW,
        companion=yolk,
    ):
        assert c.companion is not None
        # The yolk is a subset of the embryo, and must remain one after the transform.
        assert np.all((c.companion > 0.5) <= (c.mask > 0.5))


def test_vertical_flip_partner_is_an_involution_and_leaves_the_set():
    for k in CANDIDATE_KEYS:
        p = vertical_flip_partner(k)
        assert p in CANDIDATE_KEYS
        assert p != k
        assert vertical_flip_partner(p) == k


def test_pca_angle_is_degenerate_for_isotropic_and_empty_masks():
    empty = np.zeros((50, 50), dtype=np.uint8)
    assert pca_major_axis_angle_deg(empty) == (0.0, (0.0, 0.0), False)

    single = np.zeros((50, 50), dtype=np.uint8)
    single[25, 25] = 1
    _angle, _centroid, ok = pca_major_axis_angle_deg(single)
    assert ok is False


# ---------------------------------------------------------------------------
# Step 1: the AP rule -- yolk goes LEFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yolk_at", ["left", "right"])
@pytest.mark.parametrize("angle", [0.0, 20.0, -35.0])
def test_ap_rule_puts_the_yolk_on_the_left(yolk_at, angle):
    """Whichever end the yolk starts on, it should end up left of the embryo center.

    Skipped when Step 2 overrode Step 1 -- in that case the DV rule deliberately won and
    yolk-left is NOT an invariant (see AMBIGUITIES note 2). That skip is itself the finding.
    """
    mask, yolk = make_embryo(angle_deg=angle, yolk_at=yolk_at, dorsal_side="up")
    d = _decide(mask, yolk)
    assert d.orientation_source == YOLK_BACK
    if d.dv_override_ap_choice:
        pytest.skip("DV overrode AP; yolk-left is not guaranteed on this case by design")
    assert d.yolk_yx is not None
    assert d.yolk_yx[1] < GRID_HW[1] / 2.0, "yolk did not land left of center"


def test_ap_selection_is_the_argmin_over_yolk_x():
    """State the rule directly against the candidate table, not just its outcome."""
    mask, yolk = make_embryo(angle_deg=10.0, yolk_at="right")
    d = _decide(mask, yolk)
    cands = enumerate_orientation_candidates(
        mask,
        base_rotation_deg=d.major_axis_angle_deg,
        scale=1.0,
        src_center_xy=pca_major_axis_angle_deg(mask)[1],
        out_shape_yx=GRID_HW,
        companion=yolk,
    )
    xs = {}
    for c in cands:
        comp = c.companion
        tot = comp.sum()
        ys_i, xs_i = np.indices(comp.shape)
        xs[c.key] = float((xs_i * comp).sum() / tot)
    ap_winner = min(xs, key=lambda k: xs[k])
    if not d.dv_override_ap_choice:
        assert (d.rotation_adjustment_deg, d.flip_x) == (float(ap_winner[0]), ap_winner[1])


# ---------------------------------------------------------------------------
# Step 2: the DV rule -- back goes ABOVE the yolk, and may override Step 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dorsal", ["up", "down"])
@pytest.mark.parametrize("yolk_at", ["left", "right"])
def test_dv_rule_puts_the_back_above_the_yolk(dorsal, yolk_at):
    """The invariant the DV step exists to enforce: back_y <= yolk_y in the chosen pose."""
    mask, yolk = make_embryo(angle_deg=8.0, yolk_at=yolk_at, dorsal_side=dorsal)
    d = _decide(mask, yolk)
    assert d.back_yx is not None and d.yolk_yx is not None
    assert d.back_yx[0] - d.yolk_yx[0] <= 0.0, "back ended up BELOW the yolk"


def test_dv_override_is_reported_when_it_happens():
    """Step 2 undoing Step 1 must be visible, since it breaks the yolk-left rule."""
    seen = False
    for angle in np.linspace(-80, 80, 33):
        for yolk_at in ("left", "right"):
            for dorsal in ("up", "down"):
                m, y = make_embryo(angle_deg=float(angle), yolk_at=yolk_at, dorsal_side=dorsal)
                d = _decide(m, y)
                if d.dv_override_ap_choice:
                    seen = True
                    # An override means the partner pose was taken.
                    assert d.orientation_source == YOLK_BACK
    assert seen, "no DV override occurred anywhere; the override path is untested"


def test_dv_correction_cannot_run_without_mirror_candidates():
    """allow_flip=False removes the very candidate the DV fix needs."""
    mask, yolk = make_embryo(angle_deg=5.0, yolk_at="right", dorsal_side="down")
    d = _decide(mask, yolk, policy=EmbryoOrientationPolicy(allow_flip=False))
    assert d.flip_x is False
    assert d.dv_override_ap_choice is False


# ---------------------------------------------------------------------------
# Fallbacks: two of them, explicitly chosen
# ---------------------------------------------------------------------------


def test_no_yolk_policy_must_be_from_the_closed_set():
    with pytest.raises(ValueError, match="no_yolk_policy"):
        EmbryoOrientationPolicy(no_yolk_policy="whatever_seems_right")
    assert set(NO_YOLK_POLICIES) == {LEGACY_UPPER_LEFT_COM, LEGACY_VERT_RATIO}


@pytest.mark.parametrize("pol", list(NO_YOLK_POLICIES))
def test_missing_yolk_routes_to_the_requested_fallback_with_provenance(pol):
    mask, yolk = make_embryo(angle_deg=17.0, yolk_at=None)
    d = _decide(mask, yolk, no_yolk_policy=pol)
    assert d.orientation_source == pol
    assert d.fallback_policy == pol
    assert d.fallback_reason == "missing_yolk"
    assert d.yolk_used is False and d.back_used is False


def test_empty_yolk_array_is_the_same_as_no_yolk():
    """20250416 ships all-zero yolk masks rather than None; both must take the fallback."""
    mask, _ = make_embryo(angle_deg=11.0, yolk_at="left")
    zeros = np.zeros_like(mask)
    a = _decide(mask, zeros, no_yolk_policy=LEGACY_UPPER_LEFT_COM)
    b = _decide(mask, None, no_yolk_policy=LEGACY_UPPER_LEFT_COM)
    assert a.orientation_source == b.orientation_source == LEGACY_UPPER_LEFT_COM
    assert (a.rotation_adjustment_deg, a.flip_x) == (b.rotation_adjustment_deg, b.flip_x)


def test_use_yolk_false_reports_a_different_reason_than_a_missing_yolk():
    """Provenance must distinguish "there was none" from "we chose to ignore it"."""
    mask, yolk = make_embryo(angle_deg=9.0, yolk_at="left")
    d = _decide(mask, yolk, use_yolk=False)
    assert d.fallback_reason == "yolk_disabled"


def test_candidate_policies_refuse_to_guess_an_evaluation_frame():
    """legacy_upper_left_com scores POSITIONS, so the frame changes the answer."""
    mask, yolk = make_embryo(angle_deg=3.0, yolk_at=None)
    with pytest.raises(ValueError, match="candidate_shape_yx"):
        orientation_policy(mask, yolk, no_yolk_policy=LEGACY_UPPER_LEFT_COM)


# ---------------------------------------------------------------------------
# Evidence is populated, not merely declared
# ---------------------------------------------------------------------------


def test_anatomical_path_populates_its_margins():
    mask, yolk = make_embryo(angle_deg=14.0, yolk_at="left", dorsal_side="up")
    d = _decide(mask, yolk)
    assert d.orientation_source == YOLK_BACK
    assert d.ap_margin_px is not None and d.ap_margin_px >= 0.0
    assert d.dv_margin_px is not None and d.dv_margin_px >= 0.0
    assert d.back_estimation_status == "yolk_surrounding_centroid"
    assert d.back_support_pixels is not None and d.back_support_pixels > 0
    assert d.back_used is True and d.yolk_used is True
    assert d.selected_candidate_index in range(4)


def test_ap_margin_is_the_gap_to_the_runner_up():
    """Not just non-None -- the right number."""
    mask, yolk = make_embryo(angle_deg=0.0, yolk_at="left", dorsal_side="up")
    d = _decide(mask, yolk)
    if d.dv_override_ap_choice:
        pytest.skip("margin refers to the AP winner, which DV replaced")
    cands = enumerate_orientation_candidates(
        mask,
        base_rotation_deg=d.major_axis_angle_deg,
        scale=1.0,
        src_center_xy=pca_major_axis_angle_deg(mask)[1],
        out_shape_yx=GRID_HW,
        companion=yolk,
    )
    xs = []
    for c in cands:
        comp = c.companion
        _ys_i, xs_i = np.indices(comp.shape)
        xs.append(float((xs_i * comp).sum() / comp.sum()))
    xs.sort()
    assert d.ap_margin_px == pytest.approx(xs[1] - xs[0], abs=1e-6)


def test_no_calibrated_confidence_number_is_emitted():
    """Margins are on different scales; a single score would imply a calibration
    nobody has done. Guards against one being reintroduced casually."""
    mask, yolk = make_embryo(angle_deg=5.0, yolk_at="left")
    d = _decide(mask, yolk)
    for banned in ("confidence", "orientation_confident", "score", "quality"):
        assert not hasattr(d, banned), f"{banned} implies a calibration that does not exist"


def test_vert_ratio_margin_is_dimensionless_and_pixel_margins_are_absent():
    """The two fallbacks report incommensurable evidence; neither fakes the other's."""
    mask, yolk = make_embryo(angle_deg=22.0, yolk_at=None)
    d = _decide(mask, yolk, no_yolk_policy=LEGACY_VERT_RATIO)
    assert d.vert_ratio_margin is not None and 0.0 <= d.vert_ratio_margin <= 0.5
    assert d.ap_margin_px is None and d.dv_margin_px is None


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_mask_is_reported_as_such_and_does_not_raise():
    empty = np.zeros((120, 240), dtype=np.uint8)
    d = _decide(empty, None)
    assert d.orientation_source == EMPTY_MASK
    assert d.rotation_adjustment_deg == 0.0 and d.flip_x is False
    assert d.ap_margin_px is None and d.vert_ratio is None


def test_single_pixel_mask_has_no_axis_and_says_so():
    """A point has no major axis. Returning 0 degrees as a measurement would be a lie."""
    m = np.zeros((120, 240), dtype=np.uint8)
    m[60, 120] = 1
    d = _decide(m, None)
    assert d.orientation_source == DEGENERATE_MASK


def test_perfectly_circular_mask_is_not_treated_as_a_confident_axis():
    """A disc's major axis is arbitrary; whichever way it resolves, no anatomy was used."""
    h, w = 120, 120
    ys, xs = np.mgrid[0:h, 0:w]
    disc = (((ys - 60) ** 2 + (xs - 60) ** 2) <= 30**2).astype(np.uint8)
    d = _decide(disc, None, no_yolk_policy=LEGACY_UPPER_LEFT_COM)
    assert d.yolk_used is False and d.back_used is False


def test_back_point_degrades_gracefully_with_no_yolk():
    mask, _ = make_embryo(angle_deg=0.0, yolk_at="left")
    back, dbg = compute_back_point(
        mask, None, back_sample_radius_k=1.75, empty_default_yx=(0.0, 0.0)
    )
    assert dbg["selected"] == "no_yolk_fallback"
    assert dbg["n_pixels_in_disk"] is None
    assert np.isfinite(back).all()


def test_back_point_reports_an_empty_sampling_disk():
    """A yolk far from the embryo leaves no pixels in the disk -- a real failure mode."""
    mask, _ = make_embryo(angle_deg=0.0, yolk_at="left")
    far = np.zeros_like(mask)
    far[2:5, 2:5] = 1  # tiny yolk in the corner -> tiny radius, no embryo pixels near it
    back, dbg = compute_back_point(
        mask, far, back_sample_radius_k=1.75, empty_default_yx=(0.0, 0.0)
    )
    assert dbg["selected"] == "empty_disk"
    assert dbg["n_pixels_in_disk"] == 0
    assert back == pytest.approx((3.0, 3.0), abs=1.0)


def test_back_point_with_empty_embryo_but_present_yolk():
    empty = np.zeros((100, 100), dtype=np.uint8)
    yolk = np.zeros((100, 100), dtype=np.uint8)
    yolk[40:60, 40:60] = 1
    back, dbg = compute_back_point(
        empty, yolk, back_sample_radius_k=1.75, empty_default_yx=(0.0, 0.0)
    )
    assert dbg["selected"] == "empty_mask"
    assert back == pytest.approx((49.5, 49.5))


def test_decision_carries_no_canvas_state():
    """The decision describes ORIENTATION; rendering parameters belong to the caller."""
    mask, yolk = make_embryo(angle_deg=6.0, yolk_at="left")
    d = _decide(mask, yolk)
    fields = set(vars(d).keys())
    for leaked in ("grid_shape_yx", "um_per_px", "anchor_yx", "shape_hw", "canonical_grid"):
        assert leaked not in fields
    # The same decision composes against either canvas aspect.
    assert d.final_rotation_deg(target_angle_deg=90.0) == pytest.approx(
        d.final_rotation_deg(target_angle_deg=0.0) - 90.0
    )
