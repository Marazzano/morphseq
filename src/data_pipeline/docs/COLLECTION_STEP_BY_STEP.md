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

## Step 0 — DISCOVERY: generate the collection provenance artifact  [status: PARTIAL]
- Today: `resolve_experiment_ids` expands `_coll` → `{coll}_{plate}` ids; `classify_experiment`
  writes the classify artifact (`is_collection`, `sources`, `start_age_by_time_index`).
- Nuance: the artifact must GROW to carry per-source `{file, raw_path, declared_hpf, time_index}`
  (the full on-disk provenance) — the single source of truth every disk step reads.
- Surgery: extend `collection_classification` payload + contract with the per-source records
  (file, raw_path, time_index). `find_collection_plate_sources` already finds the files;
  `SourceChild.sort_key` already gives the ordering → time_index. Generate at discovery.
- DONE so far: classify artifact + is_collection + start_age_by_time_index. TODO: per-source
  file/raw_path/time_index records.

## Step 1 — ingest_scope_metadata (Snakefile:432)  [status: WRONG GRAIN — rework]
- Today (single): reads one raw dir → scope_metadata (`channel_id`* + x/y/timing) +
  acquisition_inventory. *NOTE channel-name convergence (channel→channel_id) still pending.
- Nuance: read the N source files (from provenance) → union into the plate's scope_metadata +
  acquisition_inventory, time_index per source, sources tracked.
- Surgery: `collection_acquisition_ingest` already unions per-source inventories — REPOINT its
  input to be provenance-driven (read the artifact's `sources`, not re-glob). Confirm it produces
  BOTH the acquisition_inventory AND a real scope_metadata-shaped output (see the crash: apply
  needs scope shape). Open sub-question: scope_metadata vs acquisition_inventory column parity.

## Step 2 — map_positions_to_wells (Snakefile:484)  [status: TODO]
- Today: single experiment → position→well mapping (YX1: x/y solve vs ref; Keyence: near-passthrough,
  well at ingest).
- Nuance: each source's positions resolve to well_id the NORMAL way; the plate's mapping composes
  from the sources (all share the plate layout). Merge/track by well_id.
- Surgery: TBD — walk it. Keyence: well_id already at ingest. YX1: each source solves vs the plate
  reference. The shared merge keys on well_id.

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
