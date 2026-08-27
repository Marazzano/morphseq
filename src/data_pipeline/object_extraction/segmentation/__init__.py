"""
Segmentation Module

Embryo detection and tracking using SAM2 + GroundingDINO (primary)
and UNet auxiliary masks (for QC).

The former `_archive/` (pre-disentanglement grounded_sam2, video_generation, and
segmentation_and_tracking clusters) was removed on 2026-08-04, reaching the destination its
own docstring had declared. It was never wired into PIPELINE_STEPS or Snakemake; git history
preserves it, which is the correct place for code kept "for reference".
"""
