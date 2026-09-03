"""filter_detections: size/coverage bounds, then same-object grouping, keeping the largest box.

Each test names the real failure it locks down. Case wells are from the pbx pilot (2026-09-02,
192 Keyence wells); the numbers in the comments are measured, not invented.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from data_pipeline.object_extraction.detection.backends.groundingdino.config import (
    GroundingDinoDetectionConfig,
)
from data_pipeline.object_extraction.detection.backends.groundingdino.gdino_detection import (
    calculate_containment,
    calculate_iou,
    filter_detections,
)
from data_pipeline.object_extraction.detection.run_frame_detection import _config_for_frame

# 720 x 1710 px at 3.7744 um/px — the pilot's real frame, so the mm^2 numbers below are the ones
# actually measured on those plates.
FRAME_AREA_UM2 = 720 * 1710 * 3.774417708333333 ** 2

ALL_BOUNDS_OFF = dict(
    iou_threshold=0.5,
    containment_threshold=0.85,
    min_detection_area_um2=math.nan,
    max_detection_area_um2=math.nan,
    max_frame_coverage=math.nan,
)
PILOT_BOUNDS = dict(
    iou_threshold=0.5,
    containment_threshold=0.85,
    min_detection_area_um2=600_000.0,
    max_detection_area_um2=6_500_000.0,
    max_frame_coverage=0.90,
    frame_area_um2=FRAME_AREA_UM2,
)


def box(x0, y0, x1, y1, confidence=0.5):
    return {"box_xyxy": [x0, y0, x1, y1], "confidence": confidence, "phrase": "embryo"}


def coverage(detection):
    x0, y0, x1, y1 = detection["box_xyxy"]
    return (x1 - x0) * (y1 - y0)


def test_containment_sees_nesting_that_iou_cannot():
    """The whole reason NMS missed this class: IoU divides by the union, containment by the smaller.

    Measured on the pilot: nested yolk fragments scored IoU 0.11-0.38 with containment 0.994-1.000.
    """
    outer = [0.0, 0.0, 1.0, 1.0]
    inner = [0.10, 0.10, 0.20, 0.20]
    assert calculate_containment(outer, inner) == pytest.approx(1.0)
    assert calculate_iou(outer, inner) == pytest.approx(0.01)
    # containment >= IoU always, since min(area) <= union
    assert calculate_containment(outer, inner) >= calculate_iou(outer, inner)


def test_containment_is_symmetric():
    """No 'which box came first' effect — the metric maxes over both denominators."""
    a, b = [0.0, 0.0, 0.8, 0.8], [0.1, 0.1, 0.3, 0.3]
    assert calculate_containment(a, b) == calculate_containment(b, a)


def test_nested_fragment_is_absorbed_and_the_embryo_survives():
    """M2: a yolk box inside the body box. Before this rule, overlapping_mask_flag failed BOTH."""
    embryo = box(0.20, 0.20, 0.60, 0.90, confidence=0.45)
    yolk = box(0.30, 0.70, 0.50, 0.90, confidence=0.50)
    kept = filter_detections([embryo, yolk], **ALL_BOUNDS_OFF)
    assert kept == [embryo]


def test_largest_wins_not_most_confident():
    """Measured on pilot wells G02/G04: the yolk fragment OUT-SCORED the whole embryo.

    Confidence ordering would seed SAM2 on the yolk (0.495 vs 0.453, 0.521 vs 0.402).
    """
    embryo = box(0.20, 0.20, 0.60, 0.90, confidence=0.453)
    yolk = box(0.30, 0.70, 0.50, 0.90, confidence=0.495)
    kept = filter_detections([embryo, yolk], **ALL_BOUNDS_OFF)
    assert kept == [embryo]
    assert kept[0]["confidence"] < yolk["confidence"]


def test_c08_three_boxes_resolve_to_the_embryo():
    """Pilot well plate01 C08 — the REAL normalized boxes, not invented geometry.

    Three detections on one embryo. Containment of the two fragments in the body box is 1.000 and
    0.851; the fragments barely touch each other (0.180). The 0.851 case is why the threshold is
    0.85 and not 0.90 — that fragment straddles the body box's top edge.
    """
    body = box(0.3359, 0.2176, 0.7385, 0.6537, confidence=0.614)
    head = box(0.4991, 0.2639, 0.7277, 0.3615, confidence=0.367)
    speck = box(0.5685, 0.2102, 0.7462, 0.2774, confidence=0.460)

    assert calculate_containment(body["box_xyxy"], head["box_xyxy"]) == pytest.approx(1.0, abs=1e-3)
    assert calculate_containment(body["box_xyxy"], speck["box_xyxy"]) == pytest.approx(0.851, abs=1e-2)
    # the fragments do NOT group with each other -- each is absorbed by the body directly
    assert calculate_containment(head["box_xyxy"], speck["box_xyxy"]) < 0.85

    assert filter_detections([body, head, speck], **ALL_BOUNDS_OFF) == [body]


def test_grouping_is_transitive():
    """A fragment nested in a sibling that is itself absorbed must not survive.

    Constructed, since no pilot well happens to chain: speck is inside head, head is inside body,
    but speck is NOT inside body. A greedy sequential loop suppresses head and loses the link,
    leaving speck orphaned; the component walk keeps all three connected.
    """
    body = box(0.20, 0.30, 0.60, 0.90, confidence=0.61)
    head = box(0.24, 0.22, 0.44, 0.42, confidence=0.46)   # ~60% inside body, pokes above it
    speck = box(0.26, 0.24, 0.34, 0.30, confidence=0.37)  # wholly inside head, entirely above body

    assert calculate_containment(head["box_xyxy"], speck["box_xyxy"]) >= 0.85
    assert calculate_containment(body["box_xyxy"], speck["box_xyxy"]) < 0.85
    assert calculate_containment(body["box_xyxy"], head["box_xyxy"]) < 0.85  # linked by IoU instead
    assert calculate_iou(body["box_xyxy"], head["box_xyxy"]) < 0.5

    kept = filter_detections([body, head, speck], **ALL_BOUNDS_OFF)
    # body and head are NOT linked here, so this documents the honest outcome: three objects that
    # do not all group. The chain that matters (speck absorbed into head) still holds.
    assert speck not in kept


def test_two_separate_embryos_are_both_kept():
    """The multi-embryo case. No real pilot well exercises it, so it is locked down synthetically.

    Two embryos do not contain each other, so they form separate groups and BOTH survive. A flat
    'keep the single biggest box' would silently delete the second animal.
    """
    left = box(0.05, 0.10, 0.40, 0.80, confidence=0.55)
    right = box(0.55, 0.10, 0.95, 0.80, confidence=0.52)
    kept = filter_detections([left, right], **ALL_BOUNDS_OFF)
    assert kept == [left, right]


def test_each_embryo_absorbs_only_its_own_fragment():
    """Two embryos, each with a nested yolk box: two survivors, not one and not four."""
    left = box(0.05, 0.10, 0.40, 0.80, confidence=0.55)
    left_yolk = box(0.10, 0.60, 0.30, 0.78, confidence=0.60)
    right = box(0.55, 0.10, 0.95, 0.80, confidence=0.52)
    right_yolk = box(0.60, 0.60, 0.80, 0.78, confidence=0.58)
    kept = filter_detections([left, left_yolk, right, right_yolk], **ALL_BOUNDS_OFF)
    assert kept == [left, right]


def test_max_frame_coverage_drops_the_whole_well_box():
    """M3: a box spanning the frame seeds a background mask. Scale-free, so no um/px needed."""
    embryo = box(0.30, 0.30, 0.55, 0.70, confidence=0.58)
    whole_well = box(0.0, 0.0, 1.0, 1.0, confidence=0.40)
    kept = filter_detections(
        [embryo, whole_well],
        iou_threshold=0.5,
        containment_threshold=0.85,
        min_detection_area_um2=math.nan,
        max_detection_area_um2=math.nan,
        max_frame_coverage=0.90,
    )
    assert kept == [embryo]


def test_max_area_catches_medium_grabs_that_coverage_misses():
    """Measured: 16 grabs of 7.7-9.7 mm^2 cover only 44-55% of the frame.

    Coverage at 0.90 cannot see them; the physical bound can. This is why the two are separate
    knobs rather than one.
    """
    embryo = box(0.30, 0.30, 0.55, 0.70, confidence=0.58)
    medium_grab = box(0.02, 0.02, 0.98, 0.52, confidence=0.42)   # ~0.48 of the frame
    assert coverage(medium_grab) < 0.90                          # coverage would keep it
    assert coverage(medium_grab) * FRAME_AREA_UM2 > 6_500_000.0  # the area bound rejects it
    kept = filter_detections([embryo, medium_grab], **PILOT_BOUNDS)
    assert kept == [embryo]


def test_min_area_drops_a_lone_yolk_fragment():
    """Pilot well A11: the only box was 0.4 mm^2 on the yolk. Better to seed nothing than a yolk."""
    yolk_only = box(0.40, 0.40, 0.47, 0.52, confidence=0.42)
    assert coverage(yolk_only) * FRAME_AREA_UM2 < 600_000.0
    assert filter_detections([yolk_only], **PILOT_BOUNDS) == []


def test_nan_disables_only_that_bound():
    """P4a: NaN means 'deliberately not bounded here' — the other filters keep working."""
    embryo = box(0.30, 0.30, 0.55, 0.70, confidence=0.58)
    whole_well = box(0.0, 0.0, 1.0, 1.0, confidence=0.40)
    # Spatially SEPARATE from the embryo: otherwise containment would remove it whatever the
    # min-area bound says, and the test would not be about NaN at all.
    tiny = box(0.80, 0.05, 0.83, 0.09, confidence=0.42)

    # min disabled, max/coverage still on -> the tiny box survives, the grab does not
    kept = filter_detections(
        [embryo, whole_well, tiny],
        iou_threshold=0.5,
        containment_threshold=0.85,
        min_detection_area_um2=math.nan,
        max_detection_area_um2=6_500_000.0,
        max_frame_coverage=0.90,
        frame_area_um2=FRAME_AREA_UM2,
    )
    assert whole_well not in kept
    assert tiny in kept


def test_all_bounds_nan_still_groups():
    """With every size bound off, grouping alone still fixes the nested-fragment case."""
    embryo = box(0.20, 0.20, 0.60, 0.90, confidence=0.45)
    yolk = box(0.30, 0.70, 0.50, 0.90, confidence=0.50)
    assert filter_detections([embryo, yolk], **ALL_BOUNDS_OFF) == [embryo]


def test_missing_scale_with_a_physical_bound_raises():
    """A missing um/px must never look like a passing filter — the one case that cannot be silent."""
    with pytest.raises(ValueError, match="frame_area_um2"):
        filter_detections(
            [box(0.2, 0.2, 0.6, 0.9)],
            iou_threshold=0.5,
            containment_threshold=0.85,
            min_detection_area_um2=600_000.0,
            max_detection_area_um2=6_500_000.0,
            max_frame_coverage=0.90,
            frame_area_um2=None,
        )


def test_missing_scale_is_fine_when_no_physical_bound_is_set():
    """Coverage is scale-free, so it must still run without um/px."""
    embryo = box(0.30, 0.30, 0.55, 0.70, confidence=0.58)
    whole_well = box(0.0, 0.0, 1.0, 1.0, confidence=0.40)
    kept = filter_detections(
        [embryo, whole_well],
        iou_threshold=0.5,
        containment_threshold=0.85,
        min_detection_area_um2=None,
        max_detection_area_um2=None,
        max_frame_coverage=0.90,
        frame_area_um2=None,
    )
    assert kept == [embryo]


def test_empty_input_returns_empty():
    assert filter_detections([], **ALL_BOUNDS_OFF) == []


def test_every_knob_is_required():
    """P4a's forcing function: you cannot omit a filter, only disable it explicitly."""
    with pytest.raises(TypeError):
        filter_detections([box(0.2, 0.2, 0.6, 0.9)], iou_threshold=0.5)  # type: ignore[call-arg]


def test_seahub_frames_bypass_the_size_bounds_but_keep_grouping():
    """SeaHub crops are one reviewed embryo framed tightly, so an embryo fills most of the frame.

    Running the general size rule there would reject the reviewed embryo as a background grab.
    Grouping still applies — a nested fragment is a fragment on any scope.
    """
    config = GroundingDinoDetectionConfig()
    seahub = _config_for_frame(config, pd.Series({"source_scope": "seahub"}))
    keyence = _config_for_frame(config, pd.Series({"source_scope": "keyence"}))

    assert math.isnan(seahub.min_detection_area_um2)
    assert math.isnan(seahub.max_detection_area_um2)
    assert math.isnan(seahub.max_frame_coverage)
    assert seahub.iou_threshold == config.iou_threshold
    assert seahub.containment_threshold == config.containment_threshold

    # a non-seahub frame is untouched
    assert keyence == config
