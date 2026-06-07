# 👋 HANDOFF — Run the front end on 2 wells each (YX1 + Keyence)

**For:** a fresh model picking this up cold. Read this whole file top-to-bottom first.
**Author:** mdcolon-directed session, 2026-06-07.
**Goal (today):** get the **refactored front end** to produce real artifacts for **2 wells of YX1**
and **2 wells of Keyence**, one well at a time. Prove the wired front end works on real data.

> **You are NOT designing anything new.** The front end is already built and wired. Your job is to
> RUN it on real data, find where it breaks on real inputs, and fix the smallest thing that makes it
> work. Lean HEAVILY on the old/legacy pipeline — it already interfaces this exact data and encodes
> the lessons. Keyence is fragile; go slow and expect failure modes.

---

## 🧭 ORIENTATION — read these first (in order)
1. `target/pipeline_file_philosophy.md` — **the conventions any code you touch must follow.** The two
   hard constraints: paths via `orchestration/paths.py`, ids via `shared/identifiers/`; the two
   kingdoms stay separate. Conformance checklist at the bottom.
2. `target/front_end_naming_and_flow.md` — what the front-end stages ARE (the ingest lineages, the
   fan, the convergence line). The flow you're running:
   `ingest_scope_metadata → map_series_to_wells → join_series_mapping_to_scope_metadata → discover_wells`.
3. `target/frame_inventory_well_runner_audit.md` — known gaps (e.g. frame_inventory rules are a dormant
   DAG branch; don't be surprised they don't run).
4. This file's "two-tree" note below before you touch git.

---

## 🌳 ENVIRONMENT & GIT (don't get confused — a prior session did)
- **Canonical working tree:** `/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs` on branch
  `mdcolon/20260222_docs_snakemake_remake`. The "-docs" name is a misnomer; the pipeline CODE lives
  and runs here. (`proj/morphseq` is the SAME repo parked on `main` — ignore it.)
- **Conda env:** `segmentation_grounded_sam`. Snakemake was just installed into it (`7.32.4`).
  Run python ONLY via `conda run -n segmentation_grounded_sam --no-capture-output python ...`. Never
  bare `python`/`python3`/`conda activate`.
- **⚠️ PATH trap:** a bare `snakemake` may resolve to `/bin/snakemake` (system python, NO pandas →
  "No module named pandas" at Snakefile line 17). ALWAYS launch snakemake through the env:
  `conda run -n segmentation_grounded_sam --no-capture-output snakemake ...`. Verify with
  `conda run -n segmentation_grounded_sam which snakemake` → must be the env's bin, not `/bin/`.
- **`env.yaml`** (gitignored, per-machine) already exists at
  `src/data_pipeline/pipeline_orchestrator/env.yaml` with input/output/models roots all under
  `<repo>/data_pipeline_output`. Don't commit it.
- **Tests:** `PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output
  python -m pytest tests/data_pipeline/... -q`. 63 front-end tests are green; keep them green.

---

## 🗂️ THE DATA (verified on disk 2026-06-07)
| Microscope | Experiment | Raw input | Wells available |
|---|---|---|---|
| **YX1** | `20250912` | `data_pipeline_output/inputs/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2` (single ND2, all series) | A01..H12 (plate); pick **2** |
| **Keyence** | start with `test_data/real_subset_keyence/raw_image_data/Keyence/test_keyence_001` (curated, small, SAFE first run) | per-well dirs | small subset; then a real one under `inputs/raw_image_data/Keyence/2023xxxx` if needed |

Plate metadata (optional for the front end): `inputs/plate_metadata/{exp}_well_metadata.xlsx`
(`20250912_well_metadata.xlsx` exists).

> **Keyence raw layout quirk (real):** experiment dirs hold BOTH `A01.lnk`/`W001/` well dirs AND
> `XnnnYnnn.lnk` tile links. The well is parsed from the path by `_extract_well_from_path`
> (handles `XY##`, `W0##`, and an `[A-H]\d\d` filename fallback). These are `.lnk`/symlink-ish — watch
> for resolution issues on a real run.

---

## 🛠️ HOW THE OLD/LEGACY PIPELINE DOES IT (the lessons — STUDY before running)
The refactored front end routes through `tasks.py` verbs, but the actual logic lives in the
per-microscope modules — read these to understand the real-data behavior:

**YX1:**
- Ingest: `src/data_pipeline/metadata_ingest/scope/yx1/extract_scope_metadata.py`
  (`--raw-yx1-experiment-dir --experiment-id --output-csv`) — reads the ND2, extracts µm/px, timing,
  dims, channels.
- Mapping: `src/data_pipeline/metadata_ingest/scope/yx1/map_series_to_wells.py` — the SUBTLE one.
  Matches ND2 stage XY positions to a reference plate grid. Knobs that matter on real data:
  `--use-xy-reference`, `--max-distance-um 4500`, `--row-y-tol-um 1200`, `--col-x-tol-um 1200`,
  `--dx-cv-tol/--dy-cv-tol 0.15`, `--allow-unmapped-wells`. Reference grid built by
  `yx1/generate_xy_reference.py`, validated by `yx1/validate_xy_reference_grid.py`. **This is where
  YX1 real-data failures concentrate** (mis-mapped wells, unmapped series).

**Keyence (fragile — the careful one):**
- Ingest: `src/data_pipeline/metadata_ingest/scope/keyence/extract_scope_metadata.py` — scrapes XML
  from BZ-X TIFFs (`<Data>` tags), normalizes channel names, **derives the well from the folder path
  inline** (`_extract_well_from_path`), parses time from `T####` dir/filename tokens.
- Mapping: `keyence/map_series_to_wells.py` — near-passthrough (well already resolved at ingest).
- **Legacy FF/stitch builder (real-data-tested):** `src/build/build01A_compile_keyence_torch.py` and
  `src/build/build01AB_stitch_keyence_z_slices.py`. The stitched-handoff contract
  (`target/stitched_handoff_contract.md`, "Upstream Capability to Transfer") documents two REAL
  failure modes you must respect: **heterogeneous tile counts** (3 vs 6 tiles/well) and
  **varying Z-depth** (14 vs 15 planes) — legacy forces `batch_size=1` on heterogeneity. The new
  per-well stitch avoids the cross-well case but must still handle within-well varying Z without
  aborting.

---

## ✅ THE THREE TASKS (in order; one well at a time)

### Task A — Study old pipeline real-data handling (DO FIRST, ~read-only)
Read the modules above for BOTH microscopes. Write down (append to this file or a scratch note): the
exact CLI each stage expects, the real-data knobs, and the Keyence failure modes. **Do not run yet.**
Deliverable: you can state, for each stage, what it reads and what breaks on real data.

### Task B — YX1 front end, 2 wells of `20250912`
1. Set `config.yaml` `experiment_wells: {20250912: [<well1>, <well2>]}` and `microscope: "YX1"`
   (already YX1). Remember: config only FILTERS discovered wells; the checkpoint decides existence.
2. Dry-run the front-end target to confirm the DAG resolves:
   ```
   cd src/data_pipeline/pipeline_orchestrator
   conda run -n segmentation_grounded_sam --no-capture-output snakemake -n -p \
     <output_root>/experiment_metadata/20250912/discovered_wells.txt
   ```
3. Real run (remove `-n`). Expected chain: `ingest_scope_metadata` (writes
   `scope_metadata__yx1.csv`) → `map_series_to_wells` (`series_well_mapping.csv` + `.provenance.json`)
   → `join_series_mapping_to_scope_metadata` (`scope_metadata_mapped.csv` + `.validated`) →
   `discover_wells` (`discovered_wells.txt`, global `well_id`s).
4. **Verify the outputs by reading them:** `scope_metadata_mapped.csv` has a global `well_id` column
   (`20250912_A01`, not `A01`); `discovered_wells.txt` is one global `well_id` per line.
5. Fix the smallest thing per the philosophy doc if a stage breaks. Keep the 63 tests green.

### Task C — Keyence front end, 2 wells
1. **Start with `test_data/real_subset_keyence/.../test_keyence_001`** (smallest, safest). Point the
   raw dir / experiment at it; set `microscope: "Keyence"`.
2. **Wiring gap to expect (front_end doc, "Keyence wiring gap"):** the Keyence `map_series_to_wells`
   and stitch rules may not be wired into the Snakefile yet even though the CODE exists. Wire the
   missing rule(s) following the philosophy (config-dispatch by `config["microscope"]`, one stage =
   one rule, not `_yx1`/`_keyence` suffixes). This is the one place you may add code.
3. Run the same front-end chain to `discovered_wells.txt`. Verify global `well_id`s.
4. Watch the Keyence failure modes: `.lnk` path resolution, well-from-folder parsing, the µm/px =
   width_um/width_px calc, single-timepoint (`time_int=0`) snapshots.

---

## 🚫 HARD CONSTRAINTS (from the philosophy doc — do not violate)
- Resolve EVERY artifact path via `orchestration/paths.py` helpers. No raw path strings in rules/tasks.
- Mint/split EVERY id via `shared/identifiers/`. No inline f-strings / `.split("_")`.
- `discover_wells` reads `scope_metadata_mapped.csv` (metadata-only discovery), emits global `well_id`s.
- `paths.py` never inspects disk. Snakemake input functions declare expected paths only.
- `tasks.py` verbs stay thin (parse + delegate). New stage = registry row + compute fn + thin verb +
  templated rule.
- Commit in small, clear commits. Don't commit `env.yaml`. End commit messages with the Co-Authored-By
  line.

## 📌 WHAT "DONE FOR TODAY" LOOKS LIKE
- `scope_metadata_mapped.csv` + `discovered_wells.txt` exist and are CORRECT for 2 YX1 wells AND 2
  Keyence wells, produced by the refactored front end via snakemake (not hand-run scripts).
- Any code you changed conforms to the philosophy doc and the front-end tests are still green.
- A short note of what broke on real data and how you fixed it (the real deliverable — these are the
  lessons).
