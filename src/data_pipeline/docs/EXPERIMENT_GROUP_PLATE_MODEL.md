# Experiment Collection / Plate / Event model (DRAFT SPEC)

Status: **draft — pinning the model, not yet implemented. MVP, no config changes.**
Owner: mdcolon. Started 2026-07-23.

## Why

Today `experiment_id` is atomic and doubles as the plate: one raw folder = one
`experiment_id` = one plate = one `well_metadata.xlsx`. That silently assumed
**one imaging event per plate**. It breaks for snapshot + timelapse experiments
where the *same physical plate* is imaged several times — and even **across multiple
dates** (`t45hpf` on one day, `t72hpf` the next). Downstream code (e.g.
`results/.../1_attach_phenotype_labels.py::_acq_priority`) currently re-derives the
grouping and acquisition role by **string-sniffing** embryo ids (`_sci_`, `_t02_`,
`_t01_`) and hardcodes a `30to48 -> 48` rule. This model removes that.

## Core principle (LOCKED 2026-07-24 — MERGE model)

**A collection's PLATE is an experiment. Its t-events are TIMEPOINTS inside that
experiment.** The plate condenses into one `experiment_id = {collection}_{plate_token}`;
the different `t<NN>hpf` acquisitions of that plate **merge** into it as distinct time
coordinates on the same wells — a snapshot collection becomes a *coarse timelapse*. This
is the "same well_id, different time" unification: snapshot and timelapse become the same
object.

```
raw/cilia_snapshots_coll/                    (ONE scope per collection — never mixed)
  20260607_plate01_t45hpf/  ┐
  20260608_plate01_t72hpf/  ┴─ GROUP BY PLATE + MERGE ACROSS TIME
                                   │
                                   ▼
   experiment_id = cilia_snapshots_coll_plate01
     well A01:  frame @ t45hpf (time_index 0),  frame @ t72hpf (time_index 1)
     well_id   = cilia_snapshots_coll_plate01_A01     (shared across timepoints)
     well_metadata: cilia_snapshots_coll_plate01_well_metadata.xlsx
```

Key consequences:
- **experiment_id = `{collection}_{plate_token}`** — the plate token IS the id; date and
  `t<NN>hpf` are NOT kept in identity. (There is no separate "plate_id parsed up from
  experiment_id" — the earlier draft's parse-up framing is REPLACED by this merge.)
- **`t<NN>hpf` becomes the TIME axis**, not QC metadata. `declared_hpf` finally earns its
  keep here: it is the frame's time coordinate. (`sci`/no-`t` → a plain single timepoint.)
- **`well_id` and everything below the frame inventory are UNCHANGED** — it's just an
  experiment with multiple timepoints, which the pipeline already handles for timelapse.

## Two kinds of "merge" — we do the CHEAP one (LOCKED 2026-07-24)

Separate **identity/processing merge** from **pixel/acquisition merge**. We want the first,
NOT the second:

- **Merge-A (identity — WHAT WE DO):** the t-events share `experiment_id = {coll}_{plate}`
  so they run as one unit, share one well_metadata, one output tree, get compared together.
  But each t-event's raw source is **still read independently** (one-read-per-source) and
  contributes its own frames, stamped into the same id with a distinct `time_index`.
- **Merge-B (pixels — AVOIDED):** fusing N raw sources into a single reconciled read.
  Unnecessary, because the **age escape hatch** (below) makes stage correct without a
  shared elapsed clock.

**The age escape hatch.** `predicted_stage_hpf = start_age_hpf + elapsed/3600 × rate(temp)`
(`stage_predictions/compute.py`). Stage does NOT need a fused time axis — it needs
`start_age_hpf`. So `t<NN>hpf` sets `start_age_hpf` **per timepoint**, and each source
keeps its own `elapsed_time_s`. No pixel fusion required for correctness.

```
experiment_id = cilia_snapshots_coll_plate01                (ONE id — processed together)

  frames from raw 20260607_plate01_t45hpf/ → time_index 0, start_age_hpf 45   (read independently)
  frames from raw 20260608_plate01_t72hpf/ → time_index 1, start_age_hpf 72   (read independently)
  well_id = cilia_snapshots_coll_plate01_A01   (shared across timepoints)
  well_metadata: cilia_snapshots_coll_plate01_well_metadata.xlsx   (ONE sheet, shared biology)

  stage @ t0 = 45 + elapsed₀ ·rate ;  stage @ t1 = 72 + elapsed₁ ·rate     ✓ no merge needed
```

**The invariant relaxes mildly**, not radically: from "one experiment = one raw read" to
"one experiment = one inventory, ASSEMBLED from one-read-per-source." Each source is still
read exactly once; the acquisition inventory for `{coll}_{plate}` is their **union under
one id**, each source keeping its own read and contributing a distinct `time_index`.

```
COLLECTION EXPANDER   group children by plate_token → experiment_id = {coll}_{plate}
   + ACQ UNION         read EACH source independently; stamp t<NN>hpf → (time_index, start_age_hpf);
                       union the per-source inventories under {coll}_{plate}
   + PLATE_META BUILD   shared biology sheet + per-timepoint start_age_hpf from the token
──────────────────────  frame inventory onward: UNCHANGED (one exp, multiple timepoints) ──
```

Key change to plate_metadata: `start_age_hpf` becomes **per-(well, timepoint)** rather than
one-per-experiment — which the stage formula already tolerates (it reads plate_metadata by
`well_id`; extend to `(well_id, time_index)`).

## Where does the source stop being visible? (the seam)

Merge-A becomes **seamless at the frame inventory handoff** — the designed scope→well seam
(overview A1). Provenance columns (`image_path`, `scope_name`) still *record* each frame's
source for audit, but NO downstream stage *keys* on them — object_extraction onward reads
the frame spine (`well_id`, `time_index`, `image_id`). So from object extraction on, nothing
can tell t45 and t72 came from separate files: they're just time_index 0 and 1 of one well.

```
acquisition inventory   source explicit (per-source read)
frame inventory         union under {coll}_{plate}; source knowable via provenance cols only
──────────────────────  THE SEAM ──────────────────────
object_extraction on    keys on (well_id, time_index); source invisible
```

### `physical_embryo_id` merge policy (LOCKED 2026-07-25 — supersedes source_group draft)

Frames from separate snapshot acquisitions arrive at the tracker under ONE `well_id` as
consecutive `time_index` values (Merge-A), so SAM2 tracks over time exactly like a
timelapse — **the primary path needs no source awareness.** The only fact that cannot be
re-derived downstream is *how many raw acquisitions were merged into a well*: a snapshot's
`(time_index 0, 1)` is indistinguishable from a timelapse's first two frames. So the
**union records `n_sources` per well in the frame_inventory** (a count, not a source label —
weaker than identity, keeps the "source invisible after the seam" spirit).

The registry consumes `frame_masks` **and** `frame_inventory` (to read `n_sources`) and
applies one visible policy, `EmbryoMergePolicy`:

```
per well:  n_sources = frame_inventory lookup ;  n_tracks = distinct track_id in well
  n_sources == 1                    → NORMAL    one physical_embryo_id per track (today's behavior)
  n_sources > 1  AND  n_tracks == 1 → BRIDGE    ONE physical_embryo_id across timepoints
                                                  (only one correspondence possible — like a timelapse)
  n_sources > 1  AND  n_tracks > 1  → FRACTURE  which-maps-to-which is AMBIGUOUS → do NOT guess:
                                                  disjoint _e blocks per source
                                                  (source0: e01,e02 ; source1: e03,e04 ; …)
```

**Grammar UNCHANGED:** `physical_embryo_id = {well_id}_e{NN}`. Fracture just offsets the `_e`
counter per source; no id-format variant, no hpf in the name (time already lives on
`time_index` / `start_age_hpf` — don't duplicate it into the id).

**Visible at every layer:** an `EmbryoMergePolicy` enum in code; `merge_policy`
(`normal`/`bridged`/`fractured`) + `n_sources` **columns on the registry output** (audit a
non-obvious identity decision in-place); this policy table in the spec; a one-liner in
PIPELINE_OVERVIEW C2.

**Known layout consequence (flagged for validation, NOT a bug):** a FRACTURED embryo exists
at a single `time_index`, so its `embryo_id`/`snip_id` chain is single-timepoint. This is
structurally fine — mid-course timelapse embryos (appear/die partway) already do this — but
any downstream code that ASSUMES every physical_embryo spans the full timecourse must
tolerate it. Validate this holds; do not "fix" it by forcing a full chain.

**OPEN QUESTION — does SAM2 collide `track_id` across the merged gap? (GPU-run gated)**
The FRACTURE branch groups a well's tracks by `time_index` and keys `local_embryo_index` on
`track_id`. This is only correct if `track_id` is UNIQUE across the merged well. Two cases:
- If SAM2, fed the well as ONE time-ordered video (`sam2_frame_view` sorts all `time_index`
  into one sequence), assigns GLOBALLY-unique object ids across the whole series → track_ids
  don't collide → current FRACTURE keying is correct, and a well with a real embryo at each
  timepoint likely TRACKS THROUGH (→ n_tracks==1 → BRIDGE, the common expected case).
- If SAM2 instead RESTARTS numbering per source (t0 and t1 both emit `track0000` for different
  animals) → the same `track_id` string appears twice as different animals → keying on
  `track_id` alone COLLAPSES them (wrong). Then FRACTURE must key on `(time_index, track_id)`.

Which happens is EMPIRICAL — only the GPU detection→tracking→registry run on a real merged
collection settles it. Until then the FRACTURE keying is provisional. (A synthetic frame_masks
with hand-forced `track0000` at both timepoints reproduces the collapse, confirming the risk is
real IF SAM2 restarts numbering.) Do not finalize FRACTURE keying before the GPU run.

This is a local rule at the registry mint site (C2 step 3). Upstream: only the union +
frame_inventory gain `n_sources`. Downstream reads `physical_embryo_id` as given — no
snip_processing/report join changes (there is no cross-source track_id collision, because
tracking runs over one time-ordered series per well).

## Scope rule (MVP)

A collection lives inside **ONE scope dir — scopes never mix**. So the expander doesn't
sniff scope; the (already-known) scope says what a child is:
- **Keyence:** each child is a **folder** (its `.tif` planes underneath) — extends the
  legacy folder-name-is-experiment semantics.
- **YX1:** keep the collection **flat — children are `.nd2` files, no sub-folders**.

The marker that a child is an experiment (either scope): a **plate token in the child's
own name** (file stem or dir name).

## Legacy layout (no `_coll`) — unchanged

A directory WITHOUT the `_coll` marker is a legacy single experiment: folder-name (Keyence)
/ `.nd2`-stem (YX1) → experiment_id, exactly as today. Dual-mode by the marker.

## DAG wiring — status & the remaining seam (2026-07-27)

BUILT and real-data-proven (Keyence chem28c_coll):
- `resolve_experiment_ids` + `resolve-experiment-ids` CLI (collection → flat id list, SGE EXP_FILE).
- `find_collection_plate_sources` (inverse: `{coll}_{plate}` → its raw source children).
- `collection_acquisition_ingest` module + `ingest-collection-acquisition` CLI verb: finds
  sources → per-scope acquisition-inventory builder per source → UNION → one valid inventory
  (6336 rows, n_sources=2, passes the real Keyence validator).

REMAINING SEAM (the deep part, blocks the GPU `through_line` run):
The collection path **collapses scope-read + acquisition-inventory into ONE union step**, but
the single-experiment DAG has them as SEPARATE artifacts (`scope_metadata_csv` THEN
`acquisition_inventory_csv`, consumed by `map_positions_to_wells` etc.). To run a collection
through the existing Snakefile, the collection ingest must EITHER (a) also emit the
`scope_metadata_csv` shape the downstream rules expect, OR (b) the front-end rules must accept
the unioned acquisition inventory directly for collection experiments. This is a real
reconciliation of two artifact shapes — do it carefully so the single-experiment path is
untouched. Recommended: the `ingest_scope_metadata` rule stays ONE rule; its task detects a
collection experiment (`is_collection`-style on the id) and delegates to the collection ingest,
emitting BOTH artifacts. Until this lands, the GPU run (Stage 3, which settles the SAM2
track_id-collision question) cannot execute end-to-end through the DAG.

## Older draft levels (SUPERSEDED by the merge model above — kept for history)

The earlier framing below treated each event as its own experiment_id sharing a parsed
`plate_id`. The MERGE model replaces it: events don't stay separate, they fold into
`{coll}_{plate}`. Retained only so the reasoning trail is visible.

```
collection        cilia_snapshots_coll        marked by the _coll suffix on the dir
   └─ plate_id       cilia_snapshots_coll_plate01   the PHYSICAL plate  ← well_metadata binds HERE
        └─ event        20260607 / plate01 / t45hpf  one acquisition → experiment_id → wells → snips
             declared_hpf   45                        event's DECLARED (planned) age  (sci → None)
             date           20260607                  event acquisition fact — NOT identity
```

- **collection** — a directory whose name ends in **`_coll`**. The marker is the
  *detection signal*: `_coll` present ⟹ "collection mode, look inside for plates+events".
  Folders WITHOUT `_coll` are legacy single experiments (unchanged behavior). The
  collection has **no date** of its own.
- **plate_id** — the physical plate = `{collection}_{plate_token}`. **Date-independent**
  by construction (date is never in it), so `20260607_plate01_*` and
  `20260608_plate01_*` under the same collection are ONE plate. Biology (genotype,
  geometry) is a property of this level. **One `well_metadata` per plate_id.**
- **event** — one acquisition of one plate. Maps to `experiment_id`;
  `well_id = build_well_id(experiment_id, well_index)` is stamped per event. Snips stay
  per-event (no cross-date re-identification claim).
- **declared_hpf** — the event's *declared/planned* age, from the event label:
  `t<NN>hpf` present → NN (snapshot at that age); **no `t` suffix → None** (a plain
  event, no declared age). There is NO reserved timelapse token — `sci` was just an
  experiment name, not special. (Stitching separate timelapses together is out of scope
  for the MVP.) `declared_hpf` is NEVER per-embryo truth — `predicted_stage_hpf` stays
  the measured truth.

## Raw layout — collection dir, dated child folders

```
raw/
  cilia_snapshots_coll/                 ← collection (marker _coll), NO date on it
    20260607_plate01_t45hpf/            ← date=0607, plate01, event t45hpf
    20260608_plate01_t72hpf/            ← date=0608, plate01, event t72hpf  ← SAME plate, later date
    20260607_plate02_t45hpf/            ← date=0607, plate02
    20260608_plate02_t72hpf/
```

Collection membership is **structural** (you're under the `_coll` dir). Plate membership
across dates is by the **plate_token**, which is identical wherever that plate appears.
This kills the fragile suffix-stripping AND the `_acq_priority` sniffing.

## Child folder name grammar (positional, inside a `_coll` dir)

```
{date}_{plate_token}_{event_label}

date        20260607   → event acquisition fact (→ frame inventory acquisition_time_s)
plate_token plate01    → plate identity (namespaced by the collection)
event_label t45hpf     → declared_hpf = 45   (no `t` suffix → None)
```

## Derived keys

```
experiment_id = {collection}_{date}_{plate_token}_{event_label}
              = cilia_snapshots_coll_20260607_plate01_t45hpf     (globally unique, self-describing)

  parse ▶ experiment_collection = cilia_snapshots_coll
  parse ▶ plate_id              = cilia_snapshots_coll_plate01   (DATE DROPPED)
  parse ▶ event_label           = t45hpf
            └─ declared_hpf      = 45   (None for sci/timelapse)
  parse ▶ acquisition_date      = 20260607   (event fact, not identity)
```

`well_id` and everything below it (`image_id`, `physical_embryo_id`, `embryo_id`,
`snip_id`) are **UNCHANGED** — they still hang off a single `experiment_id`. This model
only adds derivable keys *above* `well_id`.

New parsers live in `shared/identifiers/` (never string-split in consumer code):
- `parse_experiment_collection(experiment_id) -> str | None`  (None ⟹ legacy single)
- `parse_plate_id(experiment_id) -> str`
- `parse_event_label(experiment_id) -> str | None`
- `parse_declared_hpf(experiment_id) -> int | None`
- `parse_acquisition_date(experiment_id) -> str | None`

## Binding: well_metadata ⇄ plate

Excel files live in the **flat central store** `metadata/plate_metadata/` (NOT next to
raw images). One file per plate_id, reused by every event of that plate on any date:

```
metadata/plate_metadata/
  cilia_snapshots_coll_plate01__well_metadata.xlsx   ← ALL plate01 events (both dates) load this
  cilia_snapshots_coll_plate02__well_metadata.xlsx
```

Ingest, given an event `experiment_id`, parses `plate_id`, loads
`{plate_id}__well_metadata.xlsx`. `process_plate_layout` still stamps `well_id` per
**event** `experiment_id`; only the *biology source* is shared. One authored sheet,
many events (and dates) inherit it. This is the redundancy killer:
`_t01`/`_t02`-per-plate Excels collapse to one.

## Collection as a RUN TARGET — one resolver, two callers (LOCKED 2026-07-24)

The pipeline runs on `experiment_id`s. A collection is just a coarser *handle* that must
be **expanded to experiment_ids before anything runs** — and BOTH the SGE array submitter
AND the pipeline need this same expansion (a plate ≈ an experiment_id to the pipeline,
since `well_id`s are unique to it). So expansion is ONE shared primitive, not an SGE hack.

```
                  ┌──────────────────────────────────┐
 mixed input ────▶│  resolve_experiment_ids(list)    │────▶  flat experiment_id list
(ids + colls)     │  expand collections, passthru ids│           │
                  └──────────────────────────────────┘           ├─▶ SGE: N = len(list), qsub -t 1-N
                                                                  ├─▶ Snakemake --config experiments=[...]
                                                                  └─▶ each experiment runs as today
```

**Detection:** an input entry ending in **`_coll`** is a collection → expand; anything
else is a literal `experiment_id` → pass through. Pure suffix test (collection names ARE
experiment-id-shaped strings today), no filesystem probe to classify.

**Expansion source:** glob the raw child folders `raw/<name>_coll/*/`, mint one
`experiment_id` per child. Source of truth = disk, always current, no manifest to
maintain.

**DRY — the wrapper CONSUMES the existing singular primitive, never re-mints:**
```
resolve_experiment_ids(mixed)          ← PLURAL wrapper (NEW): expand _coll + passthrough
   └─ resolve_experiment_id(folder)    ← singular (EXISTS, experiment_identity.py): folder → 1 id
        └─ sanitize_experiment_id       ← string grammar (EXISTS, shared/identifiers)
```
The wrapper only decides *which folders to feed* the singular resolver; it contains NO
id-construction logic, so nothing can drift. (Analogue of the config rule: a collection
may NAME a set of experiments but never INVENTS identity — it expands to ids that run
exactly as today.)

**SGE impact:** the array template is UNCHANGED (it already eats a flat `EXP_FILE`). A
thin submit wrapper calls `resolve_experiment_ids`, writes the list, sets `N = len`,
and `qsub -t 1-N -v EXP_FILE=...`. The human no longer hand-counts N.

## What this deliberately does NOT do (MVP boundary)

- No config changes. Detection is by folder `_coll` marker + name grammar, not config.
- No new spine identifier. `plate_id`/`collection` are DERIVED keys parsed from
  `experiment_id`, not columns threaded through every contract/validator.
- Snapshot siblings keep separate `physical_embryo_id`s.
- The unit the pipeline runs on stays the event (`experiment_id`).

## OPEN QUESTIONS (before implementation)

1. **Legacy coexistence.** The ~40 existing FLAT folders/Excels
   (`20260416_..._plate01_t02`, `..._plate01_t02_well_metadata.xlsx`) have no `_coll`
   marker. Parsers must run **dual-mode**: `_coll` present ⟹ new grammar; absent ⟹
   legacy single experiment. Do old flat snapshot experiments stay as-is (legacy), or
   get migrated into `_coll` collections? MVP leans: leave legacy alone, new data uses
   `_coll`.
2. **event_label vocabulary.** Closed set: `sci` = timelapse (declared_hpf None);
   `t<NN>hpf` = snapshot at NN. Anything else? Is event always the LAST token, plate the
   one before it, date the first?
3. **plate_token format.** `plate01` vs `p01` — pick one, enforce it in the parser.
4. **Who consumes `declared_hpf`.** Which QC rules actually need expected age today —
   trace `quality_control/` before wiring the column.
```
