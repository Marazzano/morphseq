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

## Step 1 — ingest_scope_metadata (Snakefile:432)  [status: DESIGN LOCKED 2026-07-28]

**Two outputs, two consumer worlds (preserve what each ingests):**
- `scope_metadata` (scope_csv) → `map_positions` + `apply` — needs per-position GEOMETRY
  (`raw_position_label`, `x_um`, `y_um` — the YX1 mapping key) + `channel_id` (the clean token;
  currently named `channel` in scope_csv — a laggard, see the convergence note) + `raw_channel_name`.
- `acquisition_inventory` → `build_keyence_stitch_map` + `materialize` — needs image-level
  `source_path` (pixels) + acquisition axes.
These are NOT interchangeable (the earlier "dump acquisition inventory into scope_csv" was wrong,
and caused the `channel` crash). Both originate from the same per-source scope read, but each keeps
the columns ITS consumer ingests.

**CRUCIAL: acquisition facts DIFFER per source — do NOT assume shared.** Each source is its own
acquisition with its OWN x_um/y_um (plate re-seated → different stage frame), OWN calibration
(micrometers_per_pixel, image dims), OWN timing (absolute_start_time, frame_interval_s). There is NO
"the plate's scope metadata" — there is each source's, and they legitimately differ. So scope
metadata is RECORDED PER source/time_index: the concat is "more rows keyed by time_index," but each
block is a DISTINCT acquisition's real facts, NOT redundant tags on shared values. (This is exactly
why YX1 must map per-source: each source's x/y is its own — you can't map once and reuse.)

**The design (same philosophy as Step 2):** read each source ONCE; produce its normal
scope_metadata + acquisition_inventory (its own acquisition-specific values); stamp BOTH with the
same source identity (source_id, source_path, time_index); concat into one experiment-level
scope_metadata and one experiment-level acquisition_inventory (experiment = PLATE), keyed by
time_index. Each source's block preserves that source's own geometry/calibration/timing.

**The union HOLDS the per-source metadata; Step 2 re-splits by `time_index` to recover it.** The
union is NOT a lossy collapse — it is the labeled container of every source's distinct metadata,
concatenated and tagged by `time_index`:
```
unioned scope_metadata:
  time_index 0 → t28's rows (t28's OWN x/y, calibration, timing)
  time_index 1 → t52's rows (t52's OWN x/y, calibration, timing)   ← distinct values, preserved
```
So Step 2's map does `groupby("time_index")` → gets each source's block back → maps it against that
source's own geometry. Nothing is reconstructed — the per-source data was never collapsed, just
concatenated with its tag. This is the fork RESOLVED: ONE unioned artifact (not N per-source files —
keeps the DAG one-artifact-per-experiment), and consumers recover per-source via the `time_index`
key. The guarantee that makes this safe: the union is **per-source-lossless** (each block keeps its
source's real acquisition facts; do NOT dedup/collapse x/y assuming they're shared across sources).

**Canonical source identity — the KEYSTONE (same key in every source-aware artifact):**
```
source_id    ← the source child name (e.g. 20250622_plate01_t28hpf). STABLE internal key.
source_path  ← raw path (provenance + pixel access; = today's per-row source_file/tiff path).
time_index   ← the temporal coordinate (the ONE thing Step 1 ASSIGNS — read from the artifact).
```
Separate roles on purpose: `source_id` is identity (stable), `source_path` is provenance (moves if
data moves), `time_index` is temporal. For snapshots one source == one time_index; if a source is
itself a timelapse, one `source_id` spans a RANGE of time_index — so source_id is NOT derived from
time_index.

**Step 1 INGESTS the collection artifact** to know who-is-what: `sources[].file → source_id`,
`sources[].raw_path → source_path`, `sources[].time_index → time_index`. Step 1 doesn't invent or
re-derive time_index — it READS it from the artifact and stamps it. Nothing re-globbed.

**Scope difference to tolerate:** Keyence resolves well at ingest (scope_csv HAS well_index/well_id);
YX1 does not (well_id minted later at map/apply). The union must tolerate well_id present-or-absent.

**THE KEY is `time_index`** (reads best; consistent with Step 2's mapping). `time_index` is the
canonical join/composition key in EVERY source-aware artifact; `source_id` (stable identity) and
`source_path` (provenance/pixels) ride alongside every row but are not the join key.

**THE KEYSTONE INVARIANT** (the real point — not just "both have source tags"):
```
time_index (+ source_id / source_path riding along) means the SAME thing in:
  collection artifact  →  scope_metadata  →  position_mapping  →  acquisition_inventory
```
Because all four take these from the ONE artifact, `time_index` is part of the explicit JOIN KEY
(map/apply/materialize join on `(time_index, raw_position_label)`) — NOT an ad-hoc "loop per file"
hint scattered in each consumer. (`source_id` disambiguates provenance/audit; the join keys on
`time_index`.)

**Refactor note:** today's per-row `source_file` (the tiff PATH) is renamed to `source_path`; add
`source_id` + assigned `time_index`. `collection_acquisition_ingest` already unions the acquisition
inventory — REPOINT it to be artifact-driven (read `sources`, not re-glob) and ALSO produce the real
scope_metadata union (not a fake). channel→channel_id convergence is a SEPARATE cleanup, no longer a
blocker (a real scope_csv union carries the clean-token column — named `channel` today, converging
to `channel_id` — so apply won't crash on it).

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

## Cross-cutting: bare `channel` is a MISNAMED `channel_id` — converge it  [status: TODO, own commit]
There is NO legitimate bare `channel` column. There are exactly TWO channel columns:
- **`channel_id`** — the clean token ("BF"). Scope metadata currently calls this `channel` (a
  laggard name); it holds the SAME value the acquisition_inventory calls `channel_id`.
- **`raw_channel`** / `raw_channel_name` — the raw microscope value ("BrightField").

So the fix is a RENAME, not reconciling two concepts: scope metadata's `channel` → `channel_id`
(the already-declared canonical token; `_to_channel_id`/`channel_map` is already the ONE generator).
`raw_channel_name` is untouched. Do NOT treat `channel` as a distinct thing — it's `channel_id`
under an old name. ~7 live files. Separate commit; not a collection blocker (a real scope_csv union
carries whichever name is current, and this rename converges it).

## Known drift already fixed (not collection work): calibration col + stage_x/y/z_nm fixtures.
## Known pre-existing, out of scope: 2 z_stack keyence tests (StopIteration, stitch-count change).
