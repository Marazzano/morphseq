# Planned Pipeline Revisions

Living tracker for deferred pipeline changes surfaced while running real data
(2026-07, the 20260702 hotchem Keyence run). Priority is **throughput** — the pipeline is
richly structured but under-optimized for actually pushing data through efficiently.

Status legend: 🔴 not started · 🟡 partial/workaround in place · 🟢 done

---

## 1. Batching (`run_batch`) for model-heavy steps — 🔴 HIGH PRIORITY

**Problem.** Every model-heavy step runs as one Snakemake job *per well*: `conda run` →
cold-load model → process one well → exit. Across ~576 wells that is thousands of full model
reloads (GroundingDINO, SAM2, 4× UNet), which dominates wall-clock. Observed: a serial
(`--cores 1`) full run did ~⅓ of the wells in ~16 h.

**Current state.** `paths.py` defines `EXECUTION_RUN_BATCH` and even tags `frame_masks` with it
("SAM2 loads its model once and processes all run wells"), **but nothing consumes the field** —
no dispatch logic reads `execution`, and the rule is `frame_masks_per_well` regardless. So
`run_batch` is *declared but unimplemented*.

**What proper batching needs.**
1. **Batch-capable entrypoints** — `cmd_frame_detections` / `cmd_frame_masks` /
   `cmd_snip_auxiliary_masks` accept a *well list* (or "all discovered"), load the model **once**,
   loop wells.
2. **A batch Snakefile rule** — one job producing *all* per-well shards for the run
   (checkpoint-aware; expands over discovered wells like the merge helpers).
3. **Incremental write+validate inside the loop** — stamp each well's shard `.validated` as it
   finishes, so a crashed batch leaves completed wells cached and a restart resumes (preserves the
   presence-based merge).
4. *(optional)* generic dispatcher that chooses per_well vs batch from the `execution` field
   instead of hand-wiring per step.

**Why low-downside (per-well philosophy preserved).** Batching is a compute optimization,
*invisible to the output contract*: it still writes the same per-well shards, merged identically.
Only per-well **job** granularity is traded away — and a single GPU serializes inference anyway,
so no real parallelism is lost. Net: load once, keep every shard, ~order-of-magnitude less
overhead. Steps that benefit: `frame_detections` (GDINO), `frame_masks` (SAM2),
`snip_auxiliary_masks` (4× UNet), `latent_embeddings` (VAE).

**Interim workaround (2026-07).** Run `--cores 8` with GPU steps capped to 1-at-a-time via
`--resources gpu=1 --set-resources <rule>:gpu=1`; parallelizes the many CPU steps across wells.
Helps, but does NOT remove the per-well model reloads — batching is the real fix.

---

## 2. Keyence `z_stack` materialization — 🟡 workaround in place

**Problem.** `motion_blur_qc` and `focus_qc` require the `BF__z_stack` product, but Keyence
`z_stack` materialization is unimplemented — `materialize_well_keyence.py` raises
`NotImplementedError` ("Keyence z_stack fanout deferred — requires per-tile plane handling").
So the full QC path (and anything needing z-planes) cannot complete for Keyence projection-only
data.

**What it needs.** Implement Keyence z_stack materialization: per-tile plane fanout + stitching so
each `(well, channel, z, time)` plane is materialized like the YX1 path, satisfying the
`BF__z_stack` product the z-dependent QC reads.

**Interim workaround (2026-07).** `snip_qc.exclusion_flags` config override drops `focus_flag` and
`motion_blur_flag`, so `snip_qc` aggregates only the z_stack-free QC steps (death_detection,
surface_area, mask_quality). Lets Keyence runs reach a QC verdict without z-planes.

---

## 3. Merge/validation race — 🟢 fixed for snip_inventory · 🔴 frame_detections gap

**Problem.** A merge rule collects only `.validated` per-well shards, but some merge rules declared
their Snakemake `input:` as the per-well `.csv` *without* the `.validated` sentinels. Under
`--cores > 1` the merge can start before validation finishes → `collect_well_shard_paths` sees zero
shards → `ValueError: no shards to concatenate`.

**State.** Audited all merge rules: only two lacked `per_well_validated`.
- `merge_snip_inventory` — 🟢 **fixed** (branch `pipeline-merge-validated-fix`): added
  `_snip_inventory_validated_for_run` + `per_well_validated=` input, mirroring `merge_frame_masks`.
- `merge_frame_detections` — 🔴 **cannot** be fixed the same way: `frame_detections` has **no
  validate rule** and writes no `.validated` sentinels, so depending on them would break the DAG.
  Its merge is `all`-only (detection is consumed per-well by SAM2). Needs a
  `validate_frame_detections_for_well` rule first, then the `per_well_validated` edge.

---

## 4. GPU resource declaration on rules — 🔴

Model-heavy rules declare no `resources: gpu=1`, so nothing stops Snakemake from scheduling many
GPU jobs at once and OOM-ing a single card. Currently worked around on the CLI
(`--set-resources`). Bake `resources: gpu=1` into the GPU rules so any `--cores > 1` invocation is
OOM-safe by default (and pairs naturally with the batching work in §1).
