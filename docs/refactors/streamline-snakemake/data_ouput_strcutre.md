Directory Structure
{data_pipeline_root}/
│
├── inputs/                                           # ALL USER-PROVIDED INPUTS
│   ├── raw_image_data/                               # Raw microscope exports
│   │   ├── Keyence/{experiment_id}/
│   │   └── YX1/{experiment_id}/
│   │
│   ├── reference_data/
│   │   ├── surface_area_ref.csv                      # HPF stage reference (former stage_ref_df.csv)
│   │   └── perturbation_catalog.csv                  # Perturbation metadata (former perturbation_name_key.csv)
│   │
│   └── plate_metadata/                               # Per-experiment well annotations
│       └── {experiment_id}_plate_layout.xlsx
│
├── models/                                           # PRE-TRAINED MODEL CHECKPOINTS (symlink targets)
│   ├── segmentation/
│   │   ├── unet/
│   │   │   ├── embryo_mask_v0_0100/
│   │   │   ├── viability_v1_0100/
│   │   │   ├── yolk_v1_0050/
│   │   │   ├── focus_v0_0100/
│   │   │   └── bubble_v0_0100/
│   │   ├── sam2/
│   │   └── grounding_dino/
│   └── embeddings/
│       ├── morphology_vae_2024/                      # Current production VAE
│       └── legacy/
│
├── processed_images/                                 # Standardized stitched FF images
│   └── stitched_FF/
│       ├── {experiment_id}/
│       │   └── {well_id}/
│       │       └── {image_id}.jpg
│       └── preprocessing_logs/
│           └── {experiment_id}_preprocessing.csv     # Microscope + stitching metadata
│
├── segmentation/                                     # Segmentation outputs (per experiment)
│   ├── embryo_tracking/                              # SAM2 primary segmentation/tracking
│   │   └── {experiment_id}/
│   │       ├── initial_detections.json               # GroundingDINO seed bounding boxes
│   │       ├── propagated_masks.json                 # SAM2 tracked masks (all frames)
│   │       ├── tracking_table.csv                    # Flattened table incl. snip_id, bbox, area_px
│   │       └── masks/
│   │           └── {video_id}/
│   │               └── {image_id}_embryo_{N}.png
│   └── auxiliary_masks/                              # UNet auxiliary masks for QC
│       └── {experiment_id}/
│           ├── embryo/                               # Validation only (SAM2 is authoritative)
│           ├── viability/                            # Dead/alive regions
│           ├── yolk/                                 # Yolk sac
│           ├── focus/                                # Out-of-focus regions
│           └── bubbles/                              # Air bubbles
│
├── extracted_snips/                                  # Cropped embryo images + manifest
│   └── {experiment_id}/
│       ├── snip_manifest.csv                         # snip_id, frame_path, crop_path, rotation metadata
│       └── {snip_id}.jpg
│
├── computed_features/                                # Feature extraction outputs (per snip_id)
│   └── {experiment_id}/
│       ├── mask_geometry_metrics.csv                 # SAM2 mask geometry w/ px→μm² conversions (feature_extraction/mask_geometry_metrics.py)
│       ├── pose_kinematics_metrics.csv               # Embryo pose + motion (feature_extraction/pose_kinematics_metrics.py)
│       ├── developmental_stage.csv                   # Predicted HPF + confidence (stage_inference.py)
│       └── consolidated_snip_features.csv            # Merge of tracking_table + mask_geometry + pose_kinematics + stage (one row per snip_id)
│
├── quality_control/                                  # QC assessments organised by dependency
│   └── {experiment_id}/
│       ├── auxiliary_mask_qc/                        # Requires UNet auxiliary masks
│       │   ├── imaging_quality.csv                   # Frame boundary, focus, yolk, bubble flags
│       │   └── embryo_viability.csv                  # fraction_alive + dead_flag (embryo_viability_qc.py)
│       ├── segmentation_qc/                          # SAM2-only validations
│       │   ├── segmentation_quality.csv              # Mask integrity flags
│       │   └── tracking_metrics.csv                  # Tracking speed / discontinuity checks
│       ├── morphology_qc/                            # Feature-derived validations
│       │   └── size_validation.csv                   # Surface area outlier flags (SA outlier = critical signal)
│       └── consolidated/
│           ├── consolidated_qc_flags.csv             # Row-wise merge of all QC outputs
│           └── use_embryo_flags.csv                  # Final boolean gating for embeddings/analysis
│
├── latent_embeddings/                                # VAE latents (QC-passed `use_embryo == True`)
│   └── {model_name}/
│       └── {experiment_id}_latents.csv               # snip_id, z0, z1, ..., z{dim-1}
│
├── analysis_ready/                                   # Final analysis tables (per experiment)
│   └── {experiment_id}features_qc_embeddings.csv
│                                                     # consolidated_snip_features + consolidated_qc_flags + embeddings
│                                                     # Includes `embedding_calculated` column for filtering and, when multi-model,
│                                                     # optional `embedding_model` metadata
│
└── (optional downstream tables handled per analysis)
