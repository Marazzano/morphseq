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

## Step 0 — DISCOVERY: generate the collection provenance artifact  [status: DONE 2026-07-29]
- Today: `resolve_experiment_ids` expands `_coll` → `{coll}_{plate}` ids; `build_collection_provenance`
  writes the provenance artifact (`is_collection`, `sources`, `start_age_by_time_index`).
- Nuance: the artifact must GROW to carry per-source `{file, raw_path, declared_hpf, time_index}`
  (the full on-disk provenance) — the single source of truth every disk step reads.
- Surgery: extend `collection_provenance` payload + contract with the per-source records
  (file, raw_path, time_index). `find_collection_plate_sources` already finds the files;
  `SourceChild.sort_key` already gives the ordering → time_index. Generate at discovery.
- DONE 2026-07-28: the artifact now carries per-source provenance records
  `{file, raw_path, declared_hpf, time_index}` (contract + validator updated; verified on real
  chem28c `_coll`). This is the keystone artifact Step 2 reads for the files + their time_index.

## Step 1 — ingest_scope_metadata (Snakefile:432)  [status: DONE 2026-07-29]

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

## Step 2 — map_positions_to_wells (Snakefile:484)  [status: DONE 2026-07-29]

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
source_ordinal/source_file → run the existing per-scope map on each (plate id) → concat, stamping
source_ordinal + source_file per block → one `position_well_mapping.csv`. The map functions are
UNCHANGED; the collection logic is the per-file loop + concat (driven by the artifact).

**CORRECTION to the design above (2026-07-29):** the mapping is keyed on `source_ordinal`, NOT
`time_index`. The mapping is a per-SOURCE fact (which well sits at which stage position for that
acquisition), and one source can span MANY merged `time_index` values — keying on time_index would
force one mapping row per frame and reintroduce exactly the source-vs-frame conflation this work
removes. Every frame of a source inherits its source's mapping. `source_file` stays as readable
provenance.

## Step 3 — apply_position_to_well_mapping (Snakefile:525)  [status: DONE 2026-07-29]
- Today: joins scope_metadata + mapping → scope_metadata_mapped (well_id, image_id).
- The `channel` crash is GONE: it was caused by writing the acquisition inventory into the scope_csv
  slot (that table has no `channel`/geometry). Step 1 now emits a REAL scope_metadata union, so the
  column is present and no channel convergence was needed to unblock this.
- SECOND blocker, found by the YX1 run: the merge branch used
  `validate="many_to_one"` on `(experiment_id, position_index)`. A collection mapping repeats
  position_index once per source, so the right side is not unique → `MergeError` for ANY multi-source
  YX1 plate. Keyence sidesteps that branch (`scope_has_identity`), which is why it stayed hidden.
- Surgery: the merge is now SOURCE-KEYED on `(experiment_id, position_index, source_ordinal)`, gated
  on the column being present so single experiments are byte-identical. dtypes coerced with
  `errors="raise"` (a CSV round-trip can leave one side object/float64 → silent zero matches), and
  the uncovered-position error previews the full join key.
- Verified on BOTH real collections (Keyence chem28c, YX1 pbx 3-source).

## Step 4 — materialize (materialize_well_native.smk:44)  [status: DONE 2026-07-29 — verified, real GPU run]
- Today: BY WELL — `select_well_acquisition_rows` slices one well's rows → stitch/project/write →
  per-well frame_inventory. Consumes acquisition_inventory + position_well_mapping.
- Nuance: a collection well's rows span N sources (time_index 0,1) with per-row source_*_path;
  materialize pulls each source's pixels by that path — this is the NORMAL by-well, source-tracked
  behavior. Expected: NO surgery (the unified inventory already carries it).
- Verify: `select_well_acquisition_rows` returns all a well's rows across sources; materialize
  writes one frame_inventory shard with distinct time_index.

## Step 5 — frame_inventory → discover_wells (checkpoint, Snakefile:587)  [status: DONE 2026-07-29]
- The seam. After here provenance is done (except stage). discover_wells writes DISCOVERED_WELLS_TXT
  from the acquisition inventory. Verify a collection's wells discover correctly (one well per
  well_id, not per source).

## Step 6 — detection → SAM2 → physical_embryo_registry  [status: DONE 2026-07-29 — GPU-verified, BRIDGED]
- Unchanged spine (well_id, time_index). Registry applies EmbryoMergePolicy (n_sources → normal/
  bridge/fracture).
- **THE SAM2 QUESTION IS ANSWERED: BRIDGED.** Real GPU run to the registry (rc=0):
  `merge_policy: {'bridged': 2}`, `n_sources: [2]`, ONE physical_embryo_id per well
  (A01_e01, A02_e01). SAM2 did NOT collide track_id across the merged 27h snapshot gap — it
  tracked each embryo through, so the registry bridged the sources rather than fracturing.
- TWO pre-existing bugs had to be fixed first, both invisible until a collection ran the back half:
  (a) `build_physical_embryo_registry_for_well` never passed `--frame-inventory-csv`, which the verb
  REQUIRES — every run reaching Step 6 died on argparse; (b) neither materializer emitted
  `n_sources`, so `backfill_n_sources` defaulted it to 1 and the merged well looked UNMERGED. With
  the silent 1 the policy was 'normal' and the question could not be settled.

## Step 7 — stage_predictions  [status: BUILT / one KNOWN LIMITATION]
- The ONLY post-frame_inventory provenance consumer. Reads the provenance artifact's age map.
- Now reads `start_age_by_source_ordinal` (canonical), falling back to the legacy
  `start_age_by_time_index`. The legacy name was misleading: its keys were ALWAYS source ordinals,
  never merged frame indices.
- KNOWN LIMITATION (all-snapshot collections only): the snip carries the MERGED `time_index`, while
  the map is keyed by `source_ordinal`. Those coincide only when every source contributes one frame.
  It now fails loud naming the reason instead of silently reading a neighbouring source's age.
  TODO(collection-source-ordinal-through-snips): thread `source_ordinal` from the union through
  frame_inventory into snips, then key the lookup on it.

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

## Cross-cutting: source_ordinal vs time_index — DO NOT conflate  [status: DONE 2026-07-29]
Two different things, previously one column:
- **`source_ordinal`** — WHICH SOURCE (0, 1, 2 …). The machine JOIN KEY; keys the age map. Assigned
  by `declared_hpf` alone (no date, no filename tiebreaker). Two sources declaring the SAME age are
  AMBIGUOUS → `assert_source_order_unambiguous` raises rather than letting glob order decide.
- **`time_index`** — the merged FRAME coordinate. One `source_ordinal` can span MANY `time_index`
  values (a timelapse source), so neither is derived from the other. They coincide only when every
  source is a single snapshot, which is why the conflation went unnoticed.
- **`raw_time_index`** — the source-native frame index, preserved for audit. `remap_source_time_indices`
  (collection_merge_primitives) is the ONE helper both unions use, so the scope-metadata union and the
  acquisition-inventory union cannot drift. Remapping is by RANK, so sparse/1-based source numbering
  becomes a dense merged axis.
- Guarded by tests/…/test_collection_cross_artifact_consistency.py: `source_ordinal` identical across
  all four artifacts, and `(source_ordinal, raw_time_index) → time_index` identical in both unions.

## Cross-cutting: ND2 axes are read BY NAME  [status: DONE 2026-07-29]
Pre-existing bug, exposed by the real YX1 collection and NOT collection-specific (any 5-D ND2 with
no T axis was affected, single experiments included). `nd.shape` is positional and was unpacked as
`(T, W, Z)`; the pbx pilot is `(P, Z, C, Y, X)`, so 96 POSITIONS were read as 96 timepoints, 9 focal
planes as 9 positions, and 2 channels as 2 planes — 1728 plausible-looking rows and "2 wells".

Worse on the PIXEL path: `_get_stack` gated channel-selection on `ndim == 6`, so for this 5-D file it
was SKIPPED and the channel axis stood in for the Z stack — focus-stacking would have run over
`[BF, tdTomato]` as focal planes (verified: old slice returned (2, 2304, 2304), correct is
(9, 2304, 2304)).

`scope/yx1/nd2_axes.py` is now the ONE reader and documents the model:
- **SEQUENCE axes** (`nd.experiment`: TimeLoop/XYPosLoop/ZStackLoop) are what `frame_metadata(i)`
  addresses; `frameCount == product of loop counts`. **C is NOT a sequence axis** — one frame carries
  all its channels.
- **WITHIN-FRAME axes** (C, Y, X) describe a frame's contents.
- So: row grain = `sizes` (P×Z×C); frame address = loops (P×Z). `axes_of()` cross-checks its derived
  frame count against the ND2's declared `frameCount` as a tripwire.
All four ND2 readers go through it (extractor, materializer, `_get_stack`, generate_xy_reference).
Keyence is untouched — its dimensions come from TIFF path tokens, already name-anchored.

## Cross-cutting: channel_index is a RECORDED FACT, not a name match  [status: DONE 2026-07-29]
`channel_index` (the position on the raw array's channel axis) is a fact of the FILE. The scope
adapter mints the 1:1:1 triple `channel_index ↔ raw_channel_name ↔ channel_id` once into the
acquisition inventory (guarded by `assert_channel_mapping_consistent`); consumers LOOK IT UP via
`scope/shared/acquisition_channels.resolve_channel_index`.

The violation found: `_determine_bf_channel` was a SECOND channel vocabulary competing with the
scope's `channel_map.py` — it matched "BF"/"EYES - Dia"/"Empty" by name, knew nothing of
"BF-no bin", and therefore could not resolve the real fluorescence plate at all; it raised and told
callers to set `YX1_BF_CHANNEL_INDEX` (a haunted global that silently reassigns channel identity).
Deleted, along with the env override. The materializer now resolves the requested product's
`channel_id`, so it serves BF/RFP/any future product rather than being hardcoded to brightfield.

LAYERING: the resolver is NOT in `shared/channel_vocabulary.py`. That module owns the LANGUAGE
(which `channel_id` tokens exist). "BF" means the same thing everywhere; that "BF" is index 0 in one
ND2 and 1 in another is acquisition metadata — so the DataFrame query lives with the acquisition
facts. The precise rule: supported consumers use the recorded mapping; a LEGACY reader with no
inventory may translate raw names through the ONE canonical `channel_map.py` and must fail loud when
ambiguous. Private alias tables and env overrides are never acceptable.

## Known drift already fixed (not collection work): calibration col + stage_x/y/z_nm fixtures.
## Known pre-existing, out of scope: 12 image_materialization tests (Keyence stitch-map + shard
## merge) fail identically with and without this work — verified by stashing the changes.

## Cross-cutting: vocabulary + layering settled 2026-07-29  [status: DONE]

Four commits, in this order (correctness before cosmetics):

**A. Provenance is AUTHORITATIVE.** `ingest_collection_acquisition_inventory` used to re-glob the
`_coll` dir and re-probe disk for a `.nd2` suffix, so the source manifest was advisory rather than
authoritative — the two ingest paths could disagree if files were added/removed/renamed between
steps. Both unions now consume the artifact's `sources` (file / raw_path / source_ordinal).

**B. `SourceChild` → `PlateSource`** (`child_name` → `source_id`). "Child" named a filesystem
relation; the object is one raw acquisition contributing to a plate.

**C. ONE filesystem interpretation.** `experiment_collection.py` had TWO independent
`iterdir → parse → group` loops (one filtering by composed id, one grouping). Consolidated onto a
private `_discover_collection_plates`; both public callers are thin views:
`resolve_experiment_ids` → its KEYS, `discover_plate_sources` → its VALUES. Renamed
`collection_discovery.py` to pair with the provenance producer as *discover → freeze*. Exactly one
`iterdir` remains in the collection layer.

  identifiers interpret NAMES (verified: `shared/identifiers` touches no disk)
  discovery interprets the FILESYSTEM (the only `iterdir`)
  provenance FREEZES discovery into an artifact
  downstream steps CONSUME provenance

**D. `collection_classification` → `collection_provenance`** — module, contract, functions,
constants, PIPELINE_STEPS key, artifact key + filename, rule name, CLI verb, and flags. The old
name framed the product as a verdict, which is what let the misleading `start_age_by_time_index`
live in it. Old artifacts regenerate.

**The `_coll` marker is single-source.** Defined once (`_COLLECTION_SUFFIX`, with
`_COLLECTION_ID_MARKER` derived); no production module detects it inline. A guard test scans `src/`
for `endswith("_coll")` / `"_coll" in ...` / `.split("_coll")` and names the offending file:line —
inline detection is subtle because a consumer testing the suffix disagrees with the grammar the
moment the marker or its placement moves (suffix-on-a-name and the `_coll_` marker INSIDE a plate id
are different questions).
