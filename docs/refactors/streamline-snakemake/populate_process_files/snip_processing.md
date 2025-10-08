# Snip Processing Module Population Plan

Goal: break the Build03 monolith into focused functions for cropping, rotating, augmenting, writing snip images, and computing features, while keeping behaviour identical to the current pipeline.

---

## `snip_processing/extraction.py`
**Responsibilities**
- Crop embryo regions from stitched FF images using SAM2 mask metadata.
- Manage padding, bounding boxes, and crop alignment.

**Functions to implement**
- `load_frame(image_path: Path, device: torch.device | str = "cpu") -> np.ndarray`
- `extract_crop(image: np.ndarray, mask: np.ndarray, padding: int, max_size: tuple[int, int]) -> np.ndarray`
- `extract_snips_for_frame(image_path: Path, masks: list[np.ndarray], config: dict) -> list[dict]`
- `extract_snips_for_experiment(sam2_csv: Path, image_root: Path, output_dir: Path, config: dict) -> list[dict]`

**Source material**
- `build03A_process_images.py` (cropping helpers)
- `segmentation_sandbox/scripts/pipelines/06_export_masks.py` (mask/file naming)

**Cleanup notes**
- Avoid in-function globbing; rely on explicit paths passed from Snakemake.
- Return plain dicts describing each snip (`snip_id`, `frame_path`, `crop_path`, metadata).
- Handle GPU acceleration only where necessary; majority can stay in NumPy.

---

## `snip_processing/rotation.py`
**Responsibilities**
- Standardize snip orientation (e.g., PCA principal axis alignment).
- Calculate angle metadata stored alongside snips.

**Functions to implement**
- `compute_principal_axis(mask: np.ndarray) -> float`
- `rotate_snip(image: np.ndarray, angle: float, fill_value: int = 0) -> np.ndarray`
- `align_snips(snips: list[dict], config: dict) -> list[dict]`

**Source material**
- `build03A_process_images.py` (PCA rotation logic)
- `build04_perform_embryo_qc.py` (orientation metadata usage)

**Cleanup notes**
- Keep math in NumPy/scikit-image; no need for torch here.
- Update snip dicts with `rotation_angle`, `rotation_quality` fields.

---

## `snip_processing/augmentation.py`
**Responsibilities**
- Generate synthetic snips for training (noise, flips, brightness, etc.).

**Functions to implement**
- `augment_snip(image: np.ndarray, mask: np.ndarray, config: dict) -> np.ndarray`
- `augment_snip_batch(snips: list[dict], config: dict) -> list[dict]`

**Source material**
- `build03A_process_images.py` augmentation blocks

**Cleanup notes**
- Use deterministic random seeds when provided for reproducibility.
- Keep augmentation optional; config toggles all behaviours.

---

## `snip_processing/io.py`
**Responsibilities**
- Save cropped/rotated snips and write manifest CSVs.

**Functions to implement**
- `save_snip(image: np.ndarray, destination: Path, image_format: str = "png") -> None`
- `write_snip_manifest(snips: list[dict], output_csv: Path) -> None`
- `load_snip_manifest(manifest_csv: Path) -> list[dict]`

**Source material**
- `build03A_process_images.py`
- Existing analysis scripts consuming snip manifests

**Cleanup notes**
- Centralize file naming (snip IDs) via `identifiers` helpers.
- Ensure manifests include references to masks, rotations, and feature rows.

---

## `snip_processing/embryo_features/shape.py`
**Responsibilities**
- Compute geometry-based features (area, perimeter, contour-based metrics).

**Functions to implement**
- `compute_shape_features(mask: np.ndarray) -> dict`
- `compute_contour_stats(contour: np.ndarray) -> dict`

**Source material**
- Feature calculations in `build03A_process_images.py`

**Cleanup notes**
- Stick to NumPy/scikit-image; no external dependencies.
- Document feature schema (keys, units).

---

## `snip_processing/embryo_features/spatial.py`
**Responsibilities**
- Compute spatial features (centroid, bounding boxes, positional metadata).

**Functions to implement**
- `compute_spatial_features(mask: np.ndarray, pixel_size_um: float) -> dict`
- `compute_movement_metrics(current_snip: dict, previous_snip: dict) -> dict`

**Source material**
- `build03A_process_images.py`
- Portions of Build04 QC for positional deltas

**Cleanup notes**
- Keep everything in real units (microns) when possible; use config to define pixel size.
- Avoid global state; functions work on the snip dicts.

---

## `snip_processing/embryo_features/stage_inference.py`
**Responsibilities**
- Infer developmental stage (HPF) from morphology features.

**Functions to implement**
- `load_stage_model(model_path: Path) -> object`
- `predict_stage(features: dict, model: object, config: dict) -> dict`
- `annotate_snips_with_stage(snips: list[dict], model: object, config: dict) -> list[dict]`

**Source material**
- Stage inference logic in `build04_perform_embryo_qc.py`
- Existing analysis notebooks referencing HPF predictions

**Cleanup notes**
- Keep model interactions minimal (likely scikit-learn joblib model).
- Return per-snip stage predictions with confidence/metadata.

---

## Cross-cutting refactor tasks
- Align naming conventions with `identifiers` module (`snip_id`, `well_id`, etc.).
- Ensure all functions return plain dicts/lists suitable for CSV serialization.
- Factor shared config (padding, rotation defaults) into `data_pipeline.config.snips`.
- Provide smoke tests that run extraction → rotation → feature computation on a tiny sample.
