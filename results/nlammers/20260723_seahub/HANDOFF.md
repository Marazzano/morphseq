# Handoff — monitor jobs + kick off SeaHub

## (i) Monitor existing jobs
- **YX1 back_half — SGE job `22656979`** (the only live pipeline job). Now `-tc 4` (4 GPUs). Tasks 40–44 running, 45–56 queued.
  - Status: `qstat -u nlammers | grep back_half`
  - Per-task exits: `qacct -j 22656979 | awk '/^taskid/{t=$2}/^exit_status/{print t,$2}'` (nonzero = failed; tasks 1,17–22,30 already failed — a suspicious 17–22 run, flag if more cluster there).
  - Completed datasets land as `.../pipeline/output/analysis_ready/{exp}/analysis_ready/{exp}_analysis_ready.parquet`.
- Front-end Keyence z-regen is **done** (6 near-complete failures: 20260324_cep290_18hpf_24hpf_plate02, _18hpf_plate01, 20260331_b9d2_18hpf_plate01, 20260414_b9d2_14hpf_plate02, 20230525, 20250623_chem_35C_T02_1204) — re-runnable but not blocking.
- **Do not** kill running tasks or raise `-tc` above 4 without owner say-so (trapnell GPUs are saturated; owner freed 2 for a colleague).

## (ii) Kick off SeaHub — separate job, cluster only (never a login node)
Env: `morphseq-env` for reconcile/plan (pandas); GroundingDINO detection is GPU.
Code lives in `src/data_pipeline/acquisition/seahub/`; contract in `src/data_pipeline/docs/seahub_refactor/SEAHUB_IMPLEMENTATION_CONTRACT.md`. **Uncommitted fixes are in the working tree (reconciliation.py, +42/−3) — commit before running.**

1. **Reconcile** (already validated; regenerates `reconciled_fovs.csv`):
   ```
   PYTHONPATH=src python -m data_pipeline.acquisition.seahub reconcile \
     --image-reconciliation-csv results/nlammers/20260723_seahub/outputs/image_metadata_reconciliation.csv \
     --collection-metadata-xlsx /net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/metadata/collection_metadata.xlsx \
     --output-dir <integration_dir>
   ```
2. **Full-corpus GroundingDINO detection** → produce the complete `embryo_manifest.csv` (all ~1,019 included FOVs × 8 boxes). GPU, SGE array. (Reuse `results/nlammers/20260723_seahub/seahub_workflow.py` detection.)
3. **build-bundle** (ONE invocation over the full manifest — canvas is corpus-wide per call; splitting breaks uniform frame size):
   ```
   PYTHONPATH=src python -m data_pipeline.acquisition.seahub build-bundle \
     --reconciled-fovs-csv <integration_dir>/reconciled_fovs.csv \
     --detection-manifest-csv <full_embryo_manifest.csv> \
     --output-root <bundle_root>          # add --plan-only first to dry-run
   ```
   Produces one shard per source experiment: `experiments/{id}/` with images, `dropin_frame_inventory.csv[.validated]`, `plate_metadata.csv[.validated]`, `runtime_config.yaml`.
4. **materialize-shard** per experiment (cluster array, one task each; writes images + strict source validation):
   ```
   PYTHONPATH=src python -m data_pipeline.acquisition.seahub materialize-shard \
     --bundle-root <bundle_root> --experiment-id <experiment_id>
   ```
5. **Back half** — run the standard pipeline on each shard's `runtime_config.yaml` (drop-in flows through unchanged; QC annotates, never gates single_z). Model this on `submit_back_half_archive.sge`; **use a NEW job**, separate from 22656979.

### Watch-outs
- Audit is signed off (GO). Known non-blockers: canvas single-invocation invariant (step 3); µm/px `7.8` is a placeholder (no physical-size analysis on `source_scope=='seahub'`); focus/motion QC will false-flag single_z (annotate-only — ignore for seahub rows).
- QC tests can't run in one env (morphseq-env lacks skimage; mdcolon env shadows data_pipeline).
