### SAM2 Integration (Segmentation) – Stub

This runner does not orchestrate the SAM2 segmentation pipeline. Provide a precomputed SAM2 bridge CSV via `--sam2-csv` to run Build03 with SAM2 masks. A future version can add a `sam2` subcommand that shells out to `segmentation_sandbox/scripts/pipelines/run_pipeline.sh` or Python equivalents to generate:
- `segmentation_sandbox/data/exported_masks/{exp}/masks/*.png`
- `sam2_metadata_{exp}.csv`

For now, the `build03` command will accept `--sam2-csv` and skip legacy Build02.

---

# MorphSeq Centralized Runner

Centralized CLI to invoke MorphSeq pipeline steps (Build01→Build05) with a consistent, parameterized interface. Keeps “results/” runners optional by exposing a single module entrypoint.

Key features:
- Subcommands for `build01`, `combine-metadata`, `build02`, `build03`, `build04`, `build05`, `e2e`, `validate`.
- Supports SAM2-driven Build03 with `--sam2-csv` (no SAM2 orchestration here).
- Writes df01 to `metadata/combined_metadata_files/embryo_metadata_df01.csv` for Build04 compatibility.
- Subset sampling for Build03: `--by-embryo`, `--frames-per-embryo`, `--max-samples`.

## Install/Run

- Run via module:
  - `python -m src.run_morphseq_pipeline.cli <subcommand> [args]`

## Path Conventions (Defaults)

- Stitched FF images (Build01 output): `built_image_data/stitched_FF_images/{exp}`
- Per‑experiment built metadata CSV: `metadata/built_metadata_files/{exp}_metadata.csv`
- Experiment metadata (global): `metadata/experiment_metadata.csv`
- Per‑experiment well metadata Excel: `metadata/well_metadata/{exp}_well_metadata.xlsx`
- Combined master well metadata: `metadata/combined_metadata_files/master_well_metadata.csv`
- Build03 df01 (input to Build04): `metadata/combined_metadata_files/embryo_metadata_df01.csv`

## Subcommands

- `build01` (Keyence or YX1)
  - Example: `python -m src.run_morphseq_pipeline.cli build01 --root /data/morphseq --exp 20250612_30hpf_ctrl_atf6 --microscope keyence`
  - Writes stitched FF images and `{exp}_metadata.csv`.

- `combine-metadata`
  - Builds `master_well_metadata.csv` from experiment metadata, well xlsx, and built metadata.
  - Example: `python -m src.run_morphseq_pipeline.cli combine-metadata --root /data/morphseq`

- `build02`
  - Legacy segmentation (optional if using SAM2). Example: `--mode legacy`.
  - Example: `python -m src.run_morphseq_pipeline.cli build02 --root /data/morphseq --mode legacy --model-name mask_v1_0050`

- `build03`
  - With SAM2 CSV: `python -m src.run_morphseq_pipeline.cli build03 --root /data/morphseq --exp 20250612_30hpf_ctrl_atf6 --sam2-csv /data/morphseq/sam2_metadata_20250612_30hpf_ctrl_atf6.csv --by-embryo 5 --frames-per-embryo 3`
  - Legacy (no `--sam2-csv`): uses `segment_wells` (requires legacy masks from Build02).
  - Writes df01 to `metadata/combined_metadata_files/embryo_metadata_df01.csv`.

- `build04`
  - QC + stage inference; reads df01; writes df02 and curation CSVs.
  - Example: `python -m src.run_morphseq_pipeline.cli build04 --root /data/morphseq`

- `build05`
  - Training snips/folders from df02 + snips. Example:
  - `python -m src.run_morphseq_pipeline.cli build05 --root /data/morphseq --train-name train_ff_20250612`

- `e2e`
  - Orchestrate 03→04→05. Example:
  - `python -m src.run_morphseq_pipeline.cli e2e --root /data/morphseq --exp 20250612_30hpf_ctrl_atf6 --sam2-csv /data/morphseq/sam2_metadata_20250612_30hpf_ctrl_atf6.csv --by-embryo 5 --frames-per-embryo 3 --train-name train_ff_20250612`

- `validate`
  - Schema/units/path checks for df01.
  - `python -m src.run_morphseq_pipeline.cli validate --root /data/morphseq --checks schema,units,paths`

## Notes

- SAM2 segmentation is not orchestrated here; provide `--sam2-csv` to Build03.
- Build03 subset sampling helps validate integration quickly on small subsets.
- df01 write location is now consistent with Build04 expectation.

