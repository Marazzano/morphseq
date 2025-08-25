# Segmentation Sandbox Pipeline Integration

This document summarizes how the **segmentation_sandbox** pipeline fits with the existing build scripts and where genotype information is introduced through well metadata.

## Segmentation pipeline

The segmentation pipeline is orchestrated by [`run_pipeline.sh`](../segmentation_sandbox/scripts/pipelines/run_pipeline.sh). It performs:

1. **Video preparation** – copies stitched bright‑field videos into a working directory.
2. **GroundingDINO detection** – runs object detection to locate embryos.
3. **SAM2 video processing** – segments embryos frame by frame.
4. **SAM2 quality control** – flags segmentation issues.
5. **Mask export** – saves final per‑embryo masks for downstream use.

## Mask output compatibility

`SimpleMaskExporter` writes labeled mask images where each embryo ID becomes the pixel value and files are saved as `<image_id>_masks_emnum_<N>.[png|jpg]` under `exported_masks/<experiment>/masks`【F:segmentation_sandbox/scripts/utils/simple_mask_exporter.py†L118-L149】.
Downstream build scripts search for binary masks in the repository's `segmentation` folder and interpret pixels with value 255 as the embryo region【F:src/build/build03A_process_images.py†L748-L750】【F:src/functions/image_utils.py†L41-L44】.
To make the sandbox outputs usable, masks must be relocated into the `segmentation` directory and rescaled so embryo pixels equal 255 (background 0); otherwise the existing pipeline will not detect them correctly.

## Tasks for integrating sandbox masks into build scripts

The simplest path is to convert the sandbox's integer‑labeled masks to binary (0/255) on load within the build scripts. Edits concentrate in `build03A_process_images.py` and `build03B_export_z_snips.py`:

### `build03A_process_images.py`

1. **`estimate_image_background` (lines 84–88)** – read masks and threshold by embryo label rather than normalize:
   - `im_mask_raw = io.imread(im_emb_path)`
   - `im_mask = (im_mask_raw == row["region_label"]).astype(int)`
   - `im_via = (io.imread(im_via_path) > 0).astype(int)`

2. **`export_embryo_snips` (lines 168–176)** – binarize masks before calling `process_masks`:
   - `im_mask = ((io.imread(im_emb_path) == row["region_label"]) * 255).astype("uint8")`
   - `im_yolk = ((io.imread(im_yolk_path) > 0) * 255).astype("uint8")`

3. **`process_mask_images` (lines 341–352)** – replace division‑by‑255 logic with a simple threshold:
   - `im_mask = (io.imread(image_path) > 0).astype(np.uint8)`
   - Load the viability mask with `io.imread(via_path)` and binarize via `> 0`.

### `build03B_export_z_snips.py`

4. **Mask loading (lines 87–94)** – apply the same binarization as in `export_embryo_snips` before `process_masks`.

These localized changes allow downstream steps to treat sandbox masks like legacy binary masks without modifying `SimpleMaskExporter`.

## Integration points with build scripts

1. Build scripts such as `build01A_compile_keyence_torch.py` and `build01B_compile_yx1_images_torch.py` generate stitched FF images in `built_image_data/stitched_FF_images`. These paths are provided to the segmentation pipeline as the `STITCHED_DIR_OF_EXPERIMENTS` input, allowing `01_prepare_videos.py` to package the images for segmentation.
2. After masks are exported, downstream build steps (e.g. QC or snip generation) can consume the segmentation outputs instead of legacy U‑Net masks.
3. The optional pipeline step `07_embryo_metadata_update.py` can be used after segmentation to create structured embryo metadata that includes fields for genotype and treatments.

## Well metadata and genotype upload

Genotype information is attached at the well level through the `build_experiment_metadata` utility in [`export_utils.py`](../src/build/export_utils.py). This function loads per‑well metadata from an Excel file with sheets for **medium**, **genotype**, chemical perturbations, and other fields, then merges it into the experiment metadata DataFrame. It is invoked in the build scripts that prepare raw images:

- `build01A_compile_keyence_torch.py`
- `build01B_compile_yx1_images_torch.py`

When these build scripts run, the genotype column from the plate metadata spreadsheet is merged into the per‑well metadata, ensuring genotype information is available for each well.

## Summary

* Run the build scripts to create stitched FF images and per‑well metadata.
* Execute the segmentation pipeline (`run_pipeline.sh`) to detect and segment embryos and export masks.
* Use `build_experiment_metadata` during the build steps to upload well metadata— including genotype—so later steps and analysis can reference genotype by well.

