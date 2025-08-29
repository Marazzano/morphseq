# Refactor 007: MVP SAM2 Embryo Mask Integration

- Objective: Minimal, surgical swap to load embryo masks from the segmentation_sandbox pipeline while leaving yolk/other masks as-is (legacy). No changes to core QC or processing.

## What Changed
- Added sandbox embryo mask resolver (no legacy fallback; fail fast to validate pipeline):
  - `src/build/build03A_process_images.py`: `resolve_sandbox_embryo_mask(...)`
  - `src/build/build03B_export_z_snips.py`: direct sandbox load via glob
- Embryo mask load points now:
  - Load integer-labeled mask from `segmentation_sandbox/data/exported_masks/<date>/masks/` (or `MORPHSEQ_SANDBOX_MASKS_DIR` override)
  - Convert to single-embryo binary using `row['region_label']`: `(im == region_label) * 255`
  - Set `row['region_label']=1` when calling `process_masks()` to avoid changing downstream selection logic
- Non-embryo masks (yolk, etc.): unchanged; continue with legacy locations under `built_image_data/segmentation/` (Z-snips require yolk; 2D snips warn and proceed empty if missing).

## Files Touched
- `src/build/build03A_process_images.py`
  - Added `resolve_sandbox_embryo_mask`
  - Updated `export_embryo_snips()` and `get_embryo_stats()` to use sandbox embryo masks
- `src/build/build03B_export_z_snips.py`
  - Updated embryo mask load to sandbox, convert to binary, preserve existing yolk behavior

## Environment Variable
- `MORPHSEQ_SANDBOX_MASKS_DIR` (optional): overrides base path for sandbox masks.
  - Default: `<repo_root>/segmentation_sandbox/data/exported_masks`

## How To Test
This is a lightweight runbook another agent can execute.

1) Prerequisites
- Have a SAM2 CSV in the repo root: `sam2_metadata_<EXP>.csv` (e.g., `sam2_metadata_20240418.csv`).
- Ensure sandbox masks exist under:
  - `${MORPHSEQ_SANDBOX_MASKS_DIR:-<repo>/segmentation_sandbox/data/exported_masks}/<EXP>/masks/`
- For Z-snips only: legacy yolk masks present under `built_image_data/segmentation/*yolk*/<EXP>/` and z-stacks available (Keyence stitched or YX1 ND2).

2) Export env var (optional if default path is used)
```
export MORPHSEQ_SANDBOX_MASKS_DIR=/abs/path/to/segmentation_sandbox/data/exported_masks
```

3) 2D smoke test (subset of 10 rows)
```
python - << 'PY'
from pathlib import Path
import os
from src.build.build03A_process_images import segment_wells_sam2_csv, compile_embryo_stats, extract_embryo_snips, REPO_ROOT

root = REPO_ROOT

# Auto-detect one CSV in repo root
csvs = sorted(p for p in root.glob('sam2_metadata_*.csv'))
assert csvs, 'Missing sam2_metadata_*.csv in repo root'
sam2_csv = csvs[0]
exp = sam2_csv.stem.replace('sam2_metadata_', '')

# Verify sandbox masks
base = os.environ.get('MORPHSEQ_SANDBOX_MASKS_DIR', None)
mask_dir = (Path(base) if base else (root / 'segmentation_sandbox' / 'data' / 'exported_masks')) / exp / 'masks'
assert mask_dir.exists(), f'Masks not found: {mask_dir}'

tracked = segment_wells_sam2_csv(root, exp_name=exp, sam2_csv_path=sam2_csv)
tracked = tracked.head(10)
stats = compile_embryo_stats(root, tracked)
extract_embryo_snips(root, stats_df=stats, outscale=6.5, dl_rad_um=50, overwrite_flag=False)
print('OK: 2D snip export complete (subset).')
PY
```

4) Validate 2D outputs
- Images: `training_data/bf_embryo_snips/<EXP>/`
- Uncropped: `training_data/bf_embryo_snips_uncropped/<EXP>/` (if present)
- Masks: `training_data/bf_embryo_masks/emb_*.jpg`, `training_data/bf_embryo_masks/yolk_*.jpg`

5) Optional: Z-snips smoke test (requires legacy yolk + z-stacks)
```
python src/build/build03B_export_z_snips.py
```

6) Validate Z outputs
- Images: `training_data/bf_embryo_snips_z05/<EXP>/` (or z01/z03 depending on config)
- Uncropped: `training_data/bf_embryo_snips_z05_uncropped/<EXP>/`
- Temp metadata: `metadata/metadata_files_temp_z05/*.csv`

7) Troubleshooting
- FileNotFoundError for sandbox mask: confirm `MORPHSEQ_SANDBOX_MASKS_DIR` or default mask path exists for `<EXP>/masks/`.
- FileNotFoundError for yolk (Z): ensure legacy yolk directory exists under `built_image_data/segmentation/*yolk*/<EXP>/`.
- No sam2_metadata CSV: place `sam2_metadata_<EXP>.csv` in repo root or adjust the test block to point to the correct path.

## Notes / Next (PRD-007)
- SAM2 already performs embryo tracking and snip_id generation (via `embryo_id`, `snip_id`, JSON). For MVP we continue with morphseq’s formats; future work should consume SAM2 snips/IDs directly to simplify the build scripts and remove `region_label` handling.
