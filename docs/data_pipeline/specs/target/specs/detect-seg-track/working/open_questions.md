# Open Questions — Detect / Segment / Track

Use this file for back-and-forth decisions before promoting text into `targets/`.

---

## Product Boundaries

- Tentative decision: `segmentation_seed_selection` should be its own artifact/view. It is the seam
  where kept detections become segmenter prompts.
- Should `model_frame_view` be persisted or only constructed at runtime from `frame_inventory`?
- Is `mask_instances` enough, or do we need separate `raw_mask_instances` and `validated_mask_instances`?
- Tentative decision: first SAM2 strangler pass can keep one GPU job but should write
  `mask_instances` and `track_instances` sidecars beside `segmentation_tracking`.

## Vocabulary

- Tentative decision: use `time_index` in target contracts; carry `frame_index` only as a temporary
  compatibility alias while current SAM2 code still indexes frames that way.
- Tentative decision: use `segment_masks` for the stage/action and `mask_instances` for the artifact.
- **RESOLVED 2026-06-21:** `embryo_id` is minted in `snip_processing` (MVP system of record).
  `frame_masks` carries `track_id` (pipeline track identity). snip_processing normalizes
  `track_id → raw_track_index` → one-based `local_embryo_index` → `physical_embryo_id` →
  `embryo_id`. See `targets/snip_world.md` for the full three-level identity grammar
  (physical_embryo_id / embryo_id / snip_id) and the identity constructors in
  `src/data_pipeline/shared/identifiers/`.
  A discrete `tracks.csv` stage can be promoted later when track-level QC is needed, at which
  point `track_id → physical_embryo_id` resolution moves there.

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
