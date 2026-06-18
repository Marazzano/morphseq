# Current Audit Notes — Detect / Segment / Track

**Status:** scratch notes from the 2026-06-18 read-through. Update as code changes.

---

## Live Source Shape Observed

The current live runner is:

```text
src/data_pipeline/segmentation/grounded_sam2/run_per_well.py
```

It currently does all of the following in one function:

- load frame manifest;
- filter one `(experiment_id, well_id, channel_id)`;
- compute frame snapshot hash;
- load GroundingDINO;
- detect embryos per frame;
- filter detections;
- choose seed frame;
- convert detector boxes to SAM2 format;
- load SAM2;
- prepare temporary frame directories;
- propagate masks bidirectionally;
- mint embryo IDs and snip IDs;
- encode masks to RLE;
- write `segmentation_tracking.csv`;
- write `.segmentation_tracking.validated`.

Helpers already exist for pieces of the work:

- `gdino_detection.py` — model load, detection, filtering, seed frame choice, box conversion.
- `propagation.py` — SAM2 model load, forward/bidirectional propagation, mask decoding/RLE.
- `frame_organization_for_sam2.py` — temporary sequential frame views for SAM2.
- `merge_segmentation_tracking.py` — experiment-level concat.
- `csv_formatter.py` — older/alternate flattening and snapshot helpers.

---

## Current Couplings To Remove

- Segmentation reads experiment-level `frame_contract.csv`; target is per-well `frame_inventory`.
- Detection output is in-memory only.
- Seed selection is in-memory only.
- SAM2 mask output is in-memory only.
- Tracking identity is implicit in SAM2 object IDs and final row construction.
- Final `segmentation_tracking.csv` is the only persisted product before snip processing.
- `video_id` still appears in target-adjacent schema/code; target vocabulary should use `well_id`.
- Snakefile paths are hardcoded in the live tree read here; target should use `paths.py`.

---

## Immediate Lesson

The code is already partially modular by helper file, but not modular by **contract**. The first
refactor should create explicit products and validators before splitting scheduler rules.

