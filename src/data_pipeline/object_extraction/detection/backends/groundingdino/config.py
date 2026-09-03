"""GroundingDINO detection config — the seam where backend knobs live.

These are the GroundingDINO-specific filtering knobs (prompt + thresholds). The shared runner and
shared validator know nothing about them; they only see the resulting ``is_kept`` outcome.

Confidence gating happens exactly once, at ``box_threshold``. There used to be a second,
stricter ``confidence_threshold`` (0.45) re-applied after detection — a duplicate gate on the
same score that silently dropped real detections between 0.35 and 0.45 (13.4% of all detections
above ``box_threshold`` dataset-wide; confirmed via the F05 recovery investigation in the SAM3
exemplar review). Removed: ``box_threshold`` is the one confidence knob.

## Box resolution: size/coverage bounds, then same-object grouping

Added 2026-09-02 from the pbx pilot (192 Keyence wells, both plates). Detection boxes SEED SAM2, so
a bad box is not cosmetic — it produces a bad mask that then fails the whole well at snip_qc. Two
failure modes were measured, and neither is recoverable downstream:

* **the background is segmented** — a box around the well interior or the agarose mold. 46 of 192
  wells had one. The resulting mask spans 5000-8000 um and trips sa_outlier/edge/focus, taking the
  real embryo in that well with it.
* **the embryo is detected several times** — a yolk/head box nested INSIDE the body box. The old
  IoU/NMS dedup cannot see this: a small box inside a large one has low IoU *by construction*
  (measured 0.11-0.38) while its containment is 0.994-1.000. ``overlapping_mask_flag`` then failed
  the good embryo alongside its own fragment — 6 of 7 such embryos carried that flag and nothing
  else.

The three size knobs answer DIFFERENT questions and are deliberately separate. Measured counts of
raw boxes each uniquely rejects, out of 237:

* ``max_frame_coverage`` — "this box is essentially the whole image". Needs NO physical scale, so it
  still works when um/px is unavailable or the objective changes. Catches 34 full-frame grabs.
  Coverage has a wide empty band on real data (nothing between 0.70 and 0.90), so it is uncritical.
* ``max_detection_area_um2`` — "bigger than an embryo can be". Needs um/px. Catches 16 MEDIUM grabs
  (7.7-9.7 mm^2) covering only 44-55% of the frame, which coverage alone misses entirely.
* ``min_detection_area_um2`` — "smaller than an embryo can be". Rejects yolk/head fragments.

Measured area separation, both plates, 48 hpf, with empty gaps either side of the embryo band —
which is why these are absolute PHYSICAL bounds and never frame fractions:

    yolk/head fragments        0.135 - 0.482 mm^2
    ------- empty -------
    real embryo boxes          0.7   - 5.2   mm^2
    ------- empty -------
    background / mold grabs    7.7   - 17.3  mm^2

Note the embryo BOX band (up to 5.2 mm^2) is looser than the embryo MASK band (up to 1.5 mm^2): a
box around a diagonal embryo contains a lot of background. Do not reuse mask bounds here.

## Grouping: containment beside IoU, and largest-wins

``iou_threshold`` is retained: it catches two similar-sized boxes on one object, which containment
at 0.85 can miss (equal-size boxes overlapping 70-80% score IoU 0.54-0.67 but containment 0.70-0.80).
``containment_threshold`` catches the nested case IoU cannot see. On the pilot, IoU uniquely caught
0 of 49 overlapping pairs while containment uniquely caught 44 — the IoU edge is kept anyway,
because "did not fire on two plates" is not evidence it never will.

Resolution keeps the LARGEST box, not the most confident. Confidence is unreliable here: in 2 of 7
nested-fragment wells the yolk fragment out-scored the whole embryo (0.495 vs 0.453, 0.521 vs
0.402), which would seed SAM2 on the yolk. Grouping is transitive (connected components), because a
fragment may be nested in a sibling that is itself suppressed — a greedy sequential loop loses that
link and leaves the fragment orphaned.

Every bound may be NaN, meaning "deliberately not bounded for this scope" — see PIPELINE_PHILOSOPHY
P4a. The values below are the CONFIG's job to supply; ``filter_detections`` itself has no defaults
and requires every one of them to be named.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingDinoDetectionConfig:
    """Filtering / inference knobs for the GroundingDINO detector backend.

    Size/grouping values are carried HERE — config owns provenance (P4a); ``filter_detections``
    takes them as required arguments and invents nothing.
    """

    text_prompt: str = "individual embryo"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cpu"

    # --- same-object grouping -------------------------------------------------------------------
    # Two similar-sized boxes on one object (the historical NMS criterion).
    iou_threshold: float = 0.5
    # One box nested inside another. 0.85 rather than 0.90 because the tightest real fragment
    # measured 0.851, straddling the embryo box's edge by a few pixels.
    containment_threshold: float = 0.85

    # --- size / coverage bounds (NaN = deliberately not bounded) ---------------------------------
    # Floor sits in the measured 0.482 -> 0.7 mm^2 gap between fragments and real embryo boxes.
    min_detection_area_um2: float = 600_000.0      # 0.6 mm^2
    # Ceiling sits in the measured 5.2 -> 7.7 mm^2 gap. G07's real embryo box is 5.2 mm^2, so a
    # 5 mm^2 ceiling would reject a good embryo; 6.5 clears it with margin.
    max_detection_area_um2: float = 6_500_000.0    # 6.5 mm^2
    # Scale-free backstop. Real embryo boxes reach 0.55 of the frame; grabs sit at 0.90-1.00, with
    # nothing measured between 0.70 and 0.90.
    max_frame_coverage: float = 0.90
