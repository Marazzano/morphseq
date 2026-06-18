# Open Questions — Detect / Segment / Track

Use this file for back-and-forth decisions before promoting text into `targets/`.

---

## Product Boundaries

- Should `segmentation_seed_selection` be its own artifact, or part of `frame_detections`?
- Should `model_frame_view` be persisted or only constructed at runtime from `frame_inventory`?
- Is `mask_instances` enough, or do we need separate `raw_mask_instances` and `validated_mask_instances`?
- Does `track_instances` need to persist before `segmentation_tracking`, or can the first pass build
  it in memory while writing a sidecar?

## Vocabulary

- Final choice: `frame_index` or `time_index` in downstream segmentation contracts?
- Final choice: `mask_instances` vs `segmentation_masks`?
- Final choice: `track_id` format and whether `embryo_id` is derived from it.

## Backend Routing

- What is the first Facebook detector backend name/config?
- Does the detector produce boxes, masks, points, or all three?
- Does the detector run on every frame or only candidate seed frames?
- Should detector model provenance be a sidecar JSON or table columns?

## Validation Strictness

- Which validators are dataframe-only in the first pass?
- Which validators decode masks or open images?
- What overlay/visual evidence is required before replacing GroundingDINO?

## Scheduling

- Should the first split still be one GPU job that writes sidecars?
- When do we promote sidecars into separate Snakemake rules?
- How do we avoid reloading large models multiple times if detection and segmentation split into
  distinct rules?

