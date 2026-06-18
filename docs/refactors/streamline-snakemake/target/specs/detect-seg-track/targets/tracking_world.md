# Tracking World

**Status:** target-planning draft.

---

## Role

Tracking answers:

> Which masks across frames are the same embryo?

Tracking may initially accept SAM2 object IDs as the track IDs. That should be an explicit backend
choice, not an accidental side effect of the segmentation runner.

---

## Inputs

Primary input:

```text
mask_instances[well]
```

Optional supporting inputs:

- `frame_detections`, if detector evidence is used to repair track identity;
- frame timing from `model_frame_view`;
- QC masks or focus/motion flags in later phases.

---

## Output Contract: `track_instances`

One row per frame/object membership in a temporal track.

Candidate columns:

| Column | Meaning |
|---|---|
| `experiment_id` | experiment atom |
| `well_id` | global well id |
| `image_id` | frame identity |
| `frame_index` / `time_index` | temporal coordinate |
| `track_id` | stable per-well temporal identity |
| `segmentation_object_id` | source segmentation object |
| `embryo_id` | canonical embryo identity, derived from `well_id` + track |
| `track_backend` | tracking backend or policy |
| `track_confidence` | optional score |
| `track_event` | optional: normal, gap, split, merge, reidentified |

The target contract should distinguish backend-local object IDs from pipeline embryo IDs.

---

## Validator

`validate_track_instances_for_well` should prove:

- every row references a valid mask instance;
- `track_id` is stable and unique enough for downstream `embryo_id` construction;
- an embryo does not have two masks in the same frame unless explicitly represented as a split/merge;
- frame ordering is coherent;
- gaps are allowed only if policy says so;
- `embryo_id` is derived through identifier helpers, not inline string formatting.

---

## Presentation Contract

`segmentation_tracking.csv` is built after tracking. It combines:

- frame snapshot fields needed by snip processing;
- mask geometry and RLE;
- track/embryo identity;
- snip IDs;
- provenance from detection/segmentation/tracking backends.

This contract is downstream-facing. It should be stable even when detector or segmenter backends
change.

---

## Open Design Choice

Do we need an explicit tracker immediately, or can SAM2 object IDs be accepted as `track_id` under
a `sam2_object_ids` tracking backend?

Recommendation: accept SAM2 object IDs first, but route them through `link_tracks_for_well` so the
seam exists before a real tracker or identity-repair step is added.

