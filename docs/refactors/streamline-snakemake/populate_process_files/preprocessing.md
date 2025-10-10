# Preprocessing Module Population Plan

Goal: extract the minimal, well-documented functions needed to turn raw microscope dumps into stitched FF images plus metadata, while ditching legacy class wrappers and environment probing. Everything here should be importable from Snakemake rules without side effects.

---

## `preprocessing/keyence/stitching.py`
**Responsibilities**
- Assemble tile grids into per-frame FF images.
- Apply flat-field / illumination corrections if required.
- Emit logs describing tiles consumed, focus scores, stitch diagnostics.

**Functions to implement**
- `load_keyence_tiles(experiment_dir: Path, well_id: str, device: torch.device | str) -> list[torch.Tensor]`
- `stitch_keyence_tiles(tiles: list[torch.Tensor], layout: KeyenceLayout, device: torch.device, output_path: Path) -> None`
- `apply_flatfield(image: torch.Tensor, config: FlatfieldConfig, device: torch.device) -> torch.Tensor`
- `write_stitch_log(experiment_id: str, well_id: str, metrics: dict, output_csv: Path) -> None`
- Optional GPU helper: `resolve_torch_device(prefer_gpu: bool) -> torch.device`

**Source material**
- `src/build/build01A_compile_keyence_torch.py`
- `src/build/build01AB_stitch_keyence_z_slices.py` (focus measures)

- Retain GPU acceleration via torch (CPU fallback only for debugging).
- Pull layout constants from `data_pipeline.config.microscopes`.
- No filesystem discovery inside functions; all paths flow in via Snakemake inputs.
- Manage device selection via shared helper (`data_pipeline.config.runtime.resolve_device`).

---

## `preprocessing/keyence/z_stacking.py`
**Responsibilities**
- Reduce multi-z stacks into a single focused image per tile/FF frame.
- Provide focus metrics for QA.

**Functions to implement**
- `compute_focus_measure(tile_stack: np.ndarray, method: str = "variance") -> np.ndarray`
- `select_focus_plane(tile_stack: np.ndarray, method: str = "max_var") -> np.ndarray`
- `collapse_z_stack(stack_dir: Path, output_dir: Path, focus_log: Path) -> None`

**Source material**
- `src/build/build01AB_stitch_keyence_z_slices.py`

**Cleanup notes**
- Remove ad-hoc filesystem globbing; accept `Path` arguments.
- Ensure deterministic ordering of z-slices (sort filenames once).
- Expose focus measure thresholds via `config.microscopes`.
- Reuse `resolve_device` helper where torch tensors are created.

---

## `preprocessing/keyence/metadata.py`
**Responsibilities**
- Parse Keyence experiment metadata (CSV/JSON exports).
- Produce a consolidated per-experiment CSV used downstream.

**Functions to implement**
- `read_keyence_metadata(raw_dir: Path) -> pd.DataFrame`
- `normalize_keyence_metadata(df: pd.DataFrame) -> pd.DataFrame`
- `write_metadata_table(df: pd.DataFrame, output_csv: Path) -> None`

**Source material**
- `build01A_compile_keyence_torch.py::compile_keyence_metadata`
- `segmentation_sandbox/scripts/utils/parsing_utils.py` for ID helpers

**Cleanup notes**
- Strip obsolete columns (e.g., sandbox debug fields).
- Keep column naming consistent with `data_pipeline.identifiers`.
- Add type annotations and docstrings per function.

---

## `preprocessing/yx1/processing.py`
**Responsibilities**
- Convert YX1 microscope dumps into FF images.
- Handle channel ordering, bit depth, and flatfield steps unique to YX1.

**Functions to implement**
- `load_yx1_frames(raw_dir: Path, well_id: str, device: torch.device | str) -> list[torch.Tensor]`
- `process_yx1_frame(frame: torch.Tensor, config: YX1ProcessingConfig, device: torch.device) -> torch.Tensor`
- `write_yx1_image(frame: torch.Tensor, destination: Path) -> None`
- `build_yx1_experiment(raw_dir: Path, output_dir: Path, log_csv: Path, device: torch.device) -> None`
- Optional helper: `resolve_torch_device(prefer_gpu: bool) -> torch.device` (reuse from keyence)

**Source material**
- `src/build/build01B_compile_yx1_images_torch.py`

**Cleanup notes**
- Keep the torch-based GPU pipeline as the default; add a controlled CPU fallback for environments without CUDA.
- Drop environment variable lookups; accept explicit configs instead.
- Reuse shared IO helpers (`data_pipeline.io.savers`) once available.
- Document device expectations and memory requirements in module docstrings.

---

## `preprocessing/yx1/metadata.py`
**Responsibilities**
- Extract plate layouts and acquisition settings from YX1 exports.
- Produce a metadata CSV aligned with Keyence column naming.

**Functions to implement**
- `read_yx1_metadata(raw_dir: Path) -> pd.DataFrame`
- `normalize_yx1_metadata(df: pd.DataFrame) -> pd.DataFrame`
- `write_metadata_table(df: pd.DataFrame, output_csv: Path) -> None`

**Source material**
- `build01B_compile_yx1_images_torch.py::write_metadata`
- Existing plate metadata utilities under `metadata/`

**Cleanup notes**
- Unify timestamp formats (UTC ISO-8601).
- Ensure `experiment_id`, `well_id`, `time_int` columns match identifiers module.
- Document any microscope-specific quirks (exposure units, channel codes).

---

## Cross-cutting refactor tasks
- Replace repeated glob patterns with helpers in `data_pipeline.config.paths`.
- Centralize logging (use `structlog` or basic logging configured once).
- Provide a shared device-resolver (`data_pipeline.config.runtime.resolve_device`) so GPU preference is handled consistently.
- Cover each module with focused unit tests (I/O mocked using temp directories).
- Provide notebook-style documentation after migration showing before/after sample output.
