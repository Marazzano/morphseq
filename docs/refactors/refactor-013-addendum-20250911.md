Refactor-013 Addendum (2025-09-11): Per‑Experiment Mask Manifest and Video Rendering

- Stage 5 (06_export_masks.py) now writes a per‑experiment mask export manifest when the SAM2 annotations JSON contains exactly one experiment.

- Manifest paths:
  - Per‑experiment: sam2_pipeline_files/exported_masks/<exp_id>/mask_export_manifest_<exp_id>.json
  - Monolithic JSON (multiple experiments): sam2_pipeline_files/exported_masks/mask_export_manifest.json (unchanged)

- Implementation:
  - scripts/utils/simple_mask_exporter.py detects per‑experiment JSONs and uses the suffixed manifest path.
  - Manifest is updated on every run (even when no new masks are exported) to ensure presence and freshness.

- Validation:
  - Verified on 20250529_36hpf_ctrl_atf6 and an additional experiment; CSV export valid; mask file paths validated.

- Video generation utilities:
  - render_eval_video.py: import path made robust; supports per‑experiment JSONs.
  - make_eval_video.sh: auto‑detects per‑experiment JSONs; derives EXP_ID from video IDs; sensible default output dir.

