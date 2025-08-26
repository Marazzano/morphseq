# Refactor PRD 003: Finalized Segmentation Integration via Metadata Bridge

## 1. Objective & Guiding Principle

- **Objective:** To execute the most efficient and robust integration of the SAM2 segmentation pipeline by transforming the build scripts from **data-processors** into **data-consumers**.
- **Guiding Principle:** The `GroundedSam2Annotations.json` file produced by the SAM2 pipeline is the single source of truth for all embryo-specific metadata (ID, tracking, area, bbox). The build scripts should consume this information, not recalculate it.

## 2. History & Evolution of the Plan

This document outlines the final, most efficient strategy, building on insights from previous iterations:

- **Insight from `001` (MVP Approach):** The initial idea of a simple "mask-format-detector" was a good first step but ultimately insufficient. It applied a band-aid to the problem without solving the core complexity of the legacy `region_label` tracking system.

- **Insight from `002` (Surgical Replacement):** The plan evolved to a direct replacement of the `region_label` system. This was a major improvement, as it correctly identified the need to remove the old tracking logic. However, it still left the build scripts doing redundant work (e.g., running `regionprops` to re-calculate area and centroids that SAM2 had already computed).

- **The Final `003` Insight (The Metadata Bridge):** The most efficient architecture is to **not** have the build scripts parse image masks to discover embryos at all. Instead, a simple "bridge" script will flatten the rich data from `GroundedSam2Annotations.json` into a simple CSV. The build scripts will then read this CSV and use it as a direct set of instructions, completely eliminating the need for them to perform any discovery or primary metadata calculation.

## 3. The Final Two-Phase Implementation Plan

This plan is simpler, more robust, and more efficient than its predecessors.

- **Phase 1: Create the Metadata Bridge Script.**
    - **Deliverable:** A new utility script: `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`.
    - **Functionality:**
        1. Takes `GroundedSam2Annotations.json` as input.
        2. Parses the nested JSON structure.
        3. Outputs a simple, flat CSV file (`sam2_metadata.csv`) with one row for every unique embryo in every frame.
    - **Key CSV Columns:** `image_id`, `embryo_id`, `snip_id` (converted to `_t` format for compatibility), `area_px`, `bbox_x_min`, `bbox_y_min`, `bbox_x_max`, `bbox_y_max`, `mask_confidence`, `exported_mask_path`.

- **Phase 2: Gut and Refactor the Build Scripts.**
    - **Target File:** `src/build/build03A_process_images.py`.
    - **Refactoring Steps:**
        1. **Change Data Source:** The script will no longer glob for image files to start its work. Its first step will be `pd.read_csv("sam2_metadata.csv")`. This DataFrame becomes the definitive list of work to be done.
        2. **Delete Redundant Functions:** The `count_embryo_regions` and `do_embryo_tracking` functions will be **deleted entirely**.
        3. **Simplify `get_embryo_stats`:** This function will be heavily refactored. It will receive a row from the new DataFrame. Its only remaining responsibilities are to:
            - Load the correct integer mask using the `exported_mask_path` from the row.
            - Isolate the single embryo's pixels using the `embryo_id` from the row.
            - Perform the existing **QC checks** against the other U-Net masks (yolk, bubble, focus).
            - It will **not** calculate area, centroids, or any other primary stats, as they are already provided in the row.

## 4. New Data Flow Diagram

```
GroundedSam2Annotations.json
           ↓
[export_sam2_metadata_to_csv.py]
           ↓
      sam2_metadata.csv
           ↓
[build03A_process_images.py (simplified)]
           ↓
     Final QC'd Data
```

## 5. Success Criteria

- [ ] The `count_embryo_regions` and `do_embryo_tracking` functions are completely removed from the codebase.
- [ ] The new bridge script (`export_sam2_metadata_to_csv.py`) is created and functions correctly.
- [ ] `build03A_process_images.py` is significantly smaller, simpler, and starts its execution by loading the bridge CSV.
- [ ] The end-to-end build process runs successfully, producing the same final outputs but via a more efficient and robust workflow.
