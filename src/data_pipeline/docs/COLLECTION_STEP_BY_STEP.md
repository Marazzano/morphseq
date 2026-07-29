# Collection implementation — step by step (the working checklist)

Companion to `EXPERIMENT_GROUP_PLATE_MODEL.md` (the locked model). This is the per-step worklist:
for EACH acquisition step, what it does today, what the collection nuance is, and the surgery.
We go one at a time; mark status as we land each.

**Model recap (keystone):** the plate is ONE `experiment_id`. A collection provenance artifact is
generated ONCE at discovery: `{is_collection, sources:[{file, raw_path, declared_hpf, time_index}],
start_age_by_time_index}`. Disk-touching steps READ it (never re-glob `_coll`). After
frame_inventory, the only provenance consumer is stage_predictions. Composition keys on
`well_id + time_index + source` → scope-free → ONE shared helper.

Test case: `chem28c_coll_plate01` (Keyence), 2 sources (t28hpf, t52hpf), 2 wells (A01, A02).

---

## Step 0 — DISCOVERY: generate the collection provenance artifact  [status: DONE 2026-07-28]
- Today: `resolve_experiment_ids` expands `_coll` → `{coll}_{plate}` ids; `classify_experiment`
  writes the classify artifact (`is_collection`, `sources`, `start_age_by_time_index`).
- Nuance: the artifact must GROW to carry per-source `{file, raw_path, declared_hpf, time_index}`
  (the full on-disk provenance) — the single source of truth every disk step reads.
- Surgery: extend `collection_classification` payload + contract with the per-source records
  (file, raw_path, time_index). `find_collection_plate_sources` already finds the files;
  `SourceChild.sort_key` already gives the ordering → time_index. Generate at discovery.
- DONE 2026-07-28: the artifact now carries per-source provenance records
  `{file, raw_path, declared_hpf, time_index}` (contract + validator updated; verified on real
  chem28c `_coll`). This is the keystone artifact Step 2 reads for the files + their time_index.

## Step 1 — ingest_scope_metadata (Snakefile:432)  [status: WRONG GRAIN — rework]
- Today (single): reads one raw dir → scope_metadata (`channel_id`* + x/y/timing) +
  acquisition_inventory. *NOTE channel-name convergence (channel→channel_id) still pending.
- Nuance: read the N source files (from provenance) → union into the plate's scope_metadata +
  acquisition_inventory, time_index per source, sources tracked.
- Surgery: `collection_acquisition_ingest` already unions per-source inventories — REPOINT its
  input to be provenance-driven (read the artifact's `sources`, not re-glob). Confirm it produces
  BOTH the acquisition_inventory AND a real scope_metadata-shaped output (see the crash: apply
  needs scope shape). Open sub-question: scope_metadata vs acquisition_inventory column parity.

## Step 2 — map_positions_to_wells (Snakefile:484)  [status: DESIGN LOCKED 2026-07-28]

**Why per-file, not map-once:** for Keyence the well is a folder marker (`XY01/_A01`) — fixed on
disk, same in every source. But for **YX1 the well is derived from stage `x_um/y_um` matched to a
reference, and each source is its OWN ND2 with its OWN stage frame** (plate re-seated across the gap).
So source-t28's `position_index=1` and source-t52's `position_index=1` are DIFFERENT physical
acquisitions — you CANNOT map once and reuse. **Each file must be mapped independently.**

**How we map each file:** run the EXISTING per-scope map function, unchanged, once per source file,
on that file's own inputs — a source file is single-experiment-shaped (one raw dir / one scope
metadata), which is exactly what the map function already handles:
- Keyence: `map_positions_to_wells_keyence(file_raw_dir, file_scope_metadata, experiment_id=PLATE, …)`
- YX1: `map_positions_to_wells_yx1(file_scope_metadata, experiment_id=PLATE, ref_xy_csv, …)`

**experiment_id passed = the PLATE id** (option b) — the map function uses it to compose
`well_id = build_well_id(experiment_id, well_index)`, so each file's mapping comes out ALREADY
plate-keyed (`chem28c_coll_plate01_A01`). No re-key step. The well_index (A01) is what the file's
own raw/scope resolves; the plate id just prefixes it.

**Output shape — ONE file, per-file blocks concatenated (the better DAG option):**
Separate per-file mappings = variable-arity DAG outputs (checkpoints/dynamic — the fan-in
complexity we avoid). Instead ONE `position_well_mapping.csv` per experiment, N sources as ROWS
(matches how the acquisition inventory already holds N sources). DAG artifact count unchanged;
downstream reads one file.

```
position_well_mapping.csv  (ONE artifact per experiment)
  experiment_id         position_index  well_index  well_id                   time_index  source_file
  chem28c_coll_plate01  1               A01         chem28c_coll_plate01_A01  0           20250622_plate01_t28hpf
  …  (t28 block, time_index 0)
  chem28c_coll_plate01  1               A01         chem28c_coll_plate01_A01  1           20250623_plate01_t52hpf
  …  (t52 block, time_index 1)
```

**Contract gains two columns (both scopes, always present):**
- `time_index` — REQUIRED, the KEY. Tags which source-block each mapping row belongs to; makes the
  concat unambiguous. It EQUALS the source's `time_index` from the collection artifact (NOT
  re-derived) so the mapping and the acquisition inventory speak the SAME source key and join later.
- `source_file` — PROVENANCE. Which raw file (human-readable, audit).
- **Single experiment:** one block, `time_index=0`, `source_file=`the experiment's raw. Existing
  behavior + these two columns.

**Where values come from:** `time_index` + `source_file` per block ← the collection artifact
(`sources[].time_index`, `sources[].file`). `position_index/well_index/well_id` per row ← the
per-scope map function (run per file, plate id).

**The collection step, in one line:** read the collection artifact for the files + their
time_index/source_file → run the existing per-scope map on each (plate id) → concat, stamping
time_index + source_file per block → one `position_well_mapping.csv`. The map functions are
UNCHANGED; the collection logic is the per-file loop + concat (driven by the artifact).

## Step 3 — apply_position_to_well_mapping (Snakefile:525)  [status: BLOCKER SEEN]
- Today: joins scope_metadata + mapping → scope_metadata_mapped (well_id, image_id).
- Nuance: crashed on `channel` (collection scope_csv had `channel_id`, not `channel`). Also: its
  output `scope_metadata_mapped` feeds ONLY the LEGACY `materialize_stitched_images`, NOT native
  materialize. Confirm whether the collection path needs apply at all, or only the native path.
- Surgery: TBD — depends on channel convergence + whether legacy stitch is live.

## Step 4 — materialize (materialize_well_native.smk:44)  [status: LIKELY UNCHANGED]
- Today: BY WELL — `select_well_acquisition_rows` slices one well's rows → stitch/project/write →
  per-well frame_inventory. Consumes acquisition_inventory + position_well_mapping.
- Nuance: a collection well's rows span N sources (time_index 0,1) with per-row source_*_path;
  materialize pulls each source's pixels by that path — this is the NORMAL by-well, source-tracked
  behavior. Expected: NO surgery (the unified inventory already carries it).
- Verify: `select_well_acquisition_rows` returns all a well's rows across sources; materialize
  writes one frame_inventory shard with distinct time_index.

## Step 5 — frame_inventory → discover_wells (checkpoint, Snakefile:587)  [status: TODO verify]
- The seam. After here provenance is done (except stage). discover_wells writes DISCOVERED_WELLS_TXT
  from the acquisition inventory. Verify a collection's wells discover correctly (one well per
  well_id, not per source).

## Step 6 — detection → SAM2 → physical_embryo_registry  [status: BUILT (policy) / GPU-untested]
- Unchanged spine (well_id, time_index). Registry applies EmbryoMergePolicy (n_sources → normal/
  bridge/fracture). BUILT. The open SAM2 track_id-collision question is settled only by the GPU run.

## Step 7 — stage_predictions  [status: BUILT]
- The ONLY post-frame_inventory provenance consumer. Reads classify artifact →
  start_age_by_time_index[time_index]. BUILT + tested (byte-identical for singles).

---

## Cross-cutting: channel → channel_id convergence  [status: TODO, its own commit]
Scope metadata calls the clean token `channel`; acquisition_inventory calls it `channel_id`
(SAME value "BF"). `raw_channel_name` is the distinct raw string. Converge scope's `channel` →
`channel_id` (one canonical token, already the declared converged name; `_to_channel_id`/`channel_map`
is already the ONE generator). 7 live files. Fixes the apply crash's proximate cause.

## Known drift already fixed (not collection work): calibration col + stage_x/y/z_nm fixtures.
## Known pre-existing, out of scope: 2 z_stack keyence tests (StopIteration, stitch-count change).
