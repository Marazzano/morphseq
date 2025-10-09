Directory Structure
{data_pipeline_root}/
│
├── inputs/                                  # ALL USER-PROVIDED INPUTS
│   ├── raw_image_data/                          # Raw microscope data (moved from root)
│   │   ├── Keyence/{experiment_id}/
│   │   └── YX1/{experiment_id}/
│   │
│   ├── reference_data/
│   │   ├── surface_area_ref.csv         # HPF stage reference (was: stage_ref_df.csv)
│   │   └── perturbation_catalog.csv         # Perturbation metadata (was: perturbation_name_key.csv)
│   │
│   └── plate_metadata/                       # Per-experiment well annotations (was: well_metadata/)
│       └── {experiment_id}_plate_layout.xlsx
│
├── models/                                  # PRE-TRAINED MODEL CHECKPOINTS (symlinked folders)
│   ├── segmentation/
│   │   ├── unet/                            # Legacy UNet models
│   │   │   ├── embryo_mask_v0_0100/
│   │   │   ├── viability_v1_0100/
│   │   │   ├── yolk_v1_0050/
│   │   │   ├── focus_v0_0100/
│   │   │   └── bubble_v0_0100/
│   │   │
│   │   ├── sam2/                            # [SYMLINK] SAM2 entire folder
│   │   │ 
│   │   │   └── ...
│   │   │
│   │   └── grounding_dino/                  # [SYMLINK] Grounding DINO entire folder
│   │      
│   │       └── ...
│   │
│   └── embeddings/
│       ├── morphology_vae_2024/             # Current production VAE
│       └── legacy/                          # Older models
│
├── processed_images/                           # Processed image outputs
│   └── stitched_FF/                           # Standardized FF images
│       ├── {experiment_id}/
│       │   └── {well_id}/
│       │       └── {image_id}.jpg
│       │
│       └── preprocessing_logs/
│           └── {experiment_id}_preprocessing.csv  # Build metadata (microscope, paths, timestamps)
│
├── segmentation/                            # Segmentation outputs
│   ├── embryo_tracking/                     # Primary SAM2 segmentation + tracking
│   │   └── {experiment_id}/
│   │       ├── initial_detections.json      # GroundingDINO seed bboxes
│   │       ├── propagated_masks.json        # SAM2 tracked masks (all frames)
│   │       ├── tracking_table.csv           # Flattened: image_id, embryo_id, bbox, area, etc.
│   │       └── masks/                       # Exported PNG masks
│   │           └── {video_id}/
│   │               └── {image_id}_embryo_{N}.png
│   │
│   └── auxiliary_masks/                     # UNet QC support masks
│       └── {experiment_id}/
│           ├── embryo/                      # Embryo mask (validation only, NOT primary)
│           ├── viability/                   # Dead/alive regions
│           ├── yolk/                        # Yolk sac
│           ├── focus/                       # Out-of-focus regions
│           └── bubbles/                     # Air bubbles
│
├── extracted_snips/                       # Cropped embryo images + manifest
│   └── {experiment_id}/
│       ├── snip_manifest.csv               # snip_id, frame_path, crop_path, rotation metadata
│       └── {snip_id}.jpg
│
├── computed_features/                       # Extracted measurements (per snip)
│   └── {experiment_id}/
│       ├── morphology.csv                   # Area, perimeter, aspect_ratio, contours
│       ├── position.csv                     # Centroid, bbox, orientation_angle
│       ├── developmental_stage.csv          # Predicted HPF, confidence
│       └── movement.csv                     # Speed, trajectory, displacement
│
├── quality_control_flags/                           # QC assessments (pr flag family)
│   └── {experiment_id}/
│       ├── auxiliary_unet_imaging_quality.csv    # Boundary, yolk, focus, bubble flags (UNet-derived)
│       ├── viability_tracking.csv                # Death detection, inflection points
│       ├── tracking_metrics.csv                  # Speed outliers, trajectory discontinuities
│       ├── segmentation_quality.csv              # SAM2 mask quality issues
│       └── size_validation.csv                   # Area outliers, abnormal growth
│
├── latent_embeddings/                       # VAE latents (QC-passed `use_embryo` snips only)
│   └── {model_name}/
│       └── {experiment_id}_latents.csv      # snip_id, z0, z1, ..., z{dim-1}
│
└── (optional downstream tables handled per-analysis)
