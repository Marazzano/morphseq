"""
Segmentation Module

Embryo detection and tracking using SAM2 + GroundingDINO (primary)
and UNet auxiliary masks (for QC).

`_archive/` holds the pre-disentanglement grounded_sam2, video_generation, and
segmentation_and_tracking clusters — not wired into PIPELINE_STEPS/Snakemake, kept
for reference pending removal.
"""
