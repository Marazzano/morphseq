# Experiment Collection / Plate / Event model (SPEC)

Status: **design LOCKED 2026-07-27. Acquisition/identity + registry work BUILT and real-data-proven
on branch `feat/experiment-collections` (PR #21, draft). DAG wiring + GPU run remain.**
Owner: mdcolon. Started 2026-07-23.

> This doc was rewritten 2026-07-27 to remove a trail of superseded framings. The reasoning trail is
> preserved in git history + PR #21 commits; this file states only the CURRENT truth.

## Why

Today `experiment_id` is atomic and doubles as the plate: one raw folder = one `experiment_id` = one
plate = one `well_metadata.xlsx`. That silently assumed **one imaging event per plate**. It breaks
for snapshot experiments where the *same physical plate* is imaged several times, even across dates
(`t28hpf` one day, `t52hpf` the next). Downstream code (e.g.
`results/.../1_attach_phenotype_labels.py::_acq_priority`) re-derives grouping + acquisition role by
**string-sniffing** embryo ids (`_sci_`, `_t02_`, `_t01_`) and hardcodes a `30to48 → 48` rule. This
model removes that.

---

## Core model

**A collection's PLATE is ONE experiment. Its `t<NN>hpf` acquisitions are TIMEPOINTS inside it.**
The plate condenses into `experiment_id = {collection}_{plate_token}`; the sources merge into it as
distinct `time_index` values on shared `well_id`s — a snapshot collection becomes a *coarse
timelapse*. Snapshot and timelapse become the same object downstream.

```
raw/chem28c_coll/                         (_coll marker; ONE scope per collection — never mixed)
  20250622_plate01_t28hpf/  ┐  COLLAPSE at acquisition ingest:
  20250623_plate01_t52hpf/  ┘  read+map EACH source, pool into ONE acquisition inventory
                               │
                               ▼
   experiment_id = chem28c_coll_plate01
     well A01:  frame @ time_index 0 (from t28 source),  frame @ time_index 1 (from t52 source)
     well_id   = chem28c_coll_plate01_A01                (shared across timepoints)
```

- **experiment_id = `{collection}_{plate_token}`** — the plate token IS the id; date + `t<NN>hpf`
  are NOT kept in identity (they become the time axis + the age, respectively).
- **`well_id` and everything below the frame inventory are UNCHANGED** — just an experiment with
  multiple timepoints, which the pipeline already handles for timelapse.

### Collapse happens at ACQUISITION INGEST (not later)
The `_coll` dir becomes one experiment_id AT acquisition ingest. Each source is read + position-
mapped per-source INSIDE the ingest, then pooled into one acquisition inventory. This is what
`collection_acquisition_ingest` already does — real-data-proven on Keyence chem28c: 6336 rows,
`n_sources=2`, shared well_id, `time_index` 0/1, passes the real Keyence acquisition validator.
(The `time_index_claimed` raw atom rides the same per-source block offset — see the fix below.)

### Merge identity, NOT pixels
Each source is read independently (one-read-per-source); the acquisition inventory for the
collection is their **union under one id**, each source contributing a distinct `time_index` block.
No pixel fusion. Stage correctness does NOT need a fused elapsed clock — it needs `start_age_hpf`
per timepoint (see "Age" below).

---

## The governing principle: CLASSIFY ONCE, CONSUME EVERYWHERE

Determine at the **start of the DAG** whether an experiment is a collection; pass that DECLARED fact
to every downstream step that branches. Steps must NOT independently re-infer collection status from
`experiment_id`, `n_sources`, nulls, or filesystem structure.

- **Early DAG step** → a small experiment-level metadata artifact carrying `is_collection`
  (+ basic collection facts). The single source of truth.
- `is_collection_plate_id()` (in `shared/identifiers/`) may CREATE the initial fact; downstream steps
  CONSUME the declared artifact, never re-derive.
- `n_sources` (per-well, in frame_inventory) describes frame-level multiplicity — but is NOT the
  canonical definition of collection-ness. The artifact is.

---

## Age = a separate per-timepoint product (additive; NO plate_metadata schema surgery)

`start_age_hpf` is really a per-TIMEPOINT fact (t28 → age 28 applies to every well at t0),
historically fused into per-well plate_metadata. For collections, ingest ALSO emits a separate
**`time_index → start_age_hpf` mapping product** alongside plate_metadata. **plate_metadata itself is
UNCHANGED** — no null trap, no contract change.

The ONE consumer is `stage_predictions/compute.py` (`predicted_stage_hpf = start_age_hpf +
elapsed/3600 × rate(temp)`), which today reads `start_age_hpf` from `plate_by_well[well_id]`. It
branches on the DECLARED `is_collection`:
- **collection** → read `start_age_hpf` from the mapping by `time_index`;
- **non-collection** → existing per-well plate_metadata path, **byte-identical**.

Prefer-then-fallback; single experiments untouched. (Do not refactor age out of plate_metadata for
singles — add the mapping additively.)

---

## `physical_embryo_id` merge policy (registry — BUILT)

Frames from separate snapshot acquisitions arrive at the tracker under ONE `well_id` as consecutive
`time_index` values, so SAM2 tracks over time like a timelapse. The only fact that can't be
re-derived downstream is *how many acquisitions merged into a well* — recorded as `n_sources`
(per-well) in the frame_inventory (a count, not a source label).

The registry consumes `frame_masks` + `frame_inventory` (`n_sources`) and applies `EmbryoMergePolicy`:

```
per well:  n_sources = frame_inventory lookup ;  n_tracks = distinct track_id in well
  n_sources == 1                    → NORMAL    one physical_embryo_id per track (today's behavior)
  n_sources > 1  AND  n_tracks == 1 → BRIDGE    ONE physical_embryo_id across timepoints
  n_sources > 1  AND  n_tracks > 1  → FRACTURE  ambiguous → do NOT guess: disjoint _e blocks per source
```

Grammar UNCHANGED (`physical_embryo_id = {well_id}_e{NN}`; fracture just offsets the `_e` counter per
source). Visible: `EmbryoMergePolicy` enum + `merge_policy`/`n_sources` columns on the registry
output + PIPELINE_OVERVIEW C2 one-liner.

**Known layout consequence (validate, NOT a bug):** a FRACTURED embryo is single-`time_index`, so its
`embryo_id`/`snip_id` chain is single-timepoint — fine (mid-course timelapse embryos already do this).
Don't "fix" by forcing a full chain.

### OPEN QUESTION — does SAM2 collide `track_id` across the merged gap? (GPU-gated)
FRACTURE keys `local_embryo_index` on `track_id`, correct ONLY if `track_id` is unique across the
merged well. If SAM2 (fed the well as ONE sorted video via `sam2_frame_view`) assigns globally-unique
ids → correct, and a real embryo-per-timepoint likely TRACKS THROUGH (→ n_tracks==1 → BRIDGE). If
SAM2 restarts numbering per source (both emit `track0000`) → keying on `track_id` alone COLLAPSES two
animals → FRACTURE must key on `(time_index, track_id)`. **Empirical — only the GPU run settles it.**
FRACTURE keying is provisional until then. (Synthetic masks with forced `track0000` at both
timepoints reproduce the collapse, confirming the risk is real IF SAM2 restarts numbering.)

---

## Naming grammar

### `_coll` dir + dated child folders
```
raw/chem28c_coll/                     ← collection (marker _coll), NO date of its own
  20250622_plate01_t28hpf/            ← {date}_{plate_token}_{event_label}
  20250623_plate01_t52hpf/            ← same plate, later date → SAME plate_token
```
Collection membership is **structural** (under the `_coll` dir). Plate membership across dates is by
the `plate_token`. This kills the fragile suffix-stripping and the `_acq_priority` sniffing.

### Child name → keys (positional; parsed in `shared/identifiers/`, never string-split in consumers)
```
{date}_{plate_token}_{event_label}
  date        20250622  → acquisition fact (→ frame_inventory acquisition_time_s)
  plate_token plate01   → plate identity (namespaced by the collection)
  event_label t28hpf    → declared_hpf = 28   (no `t` suffix → None)
```
`experiment_id = {collection}_{plate_token}` (date + event DROPPED — all a plate's t-events share it).
Built parsers: `is_collection`, `is_collection_plate_id`, `parse_plate_token`, `parse_event_label`,
`parse_declared_hpf`, `parse_collection_name_from_plate_id`, `compose_collection_experiment_id`.

### Scope rule (MVP)
One scope per collection — scopes never mix. Keyence child = a folder (its `.tif` planes underneath);
YX1 child = a flat `.nd2` file. The marker that a child is a source: a plate token in its own name.

### Legacy (no `_coll`) — unchanged
A dir WITHOUT the `_coll` marker is a legacy single experiment: folder-name (Keyence) / `.nd2`-stem
(YX1) → experiment_id, exactly as today. Dual-mode by the marker.

---

## Collection as a RUN TARGET — one resolver, two callers (BUILT)

A collection is a coarse *handle* expanded to `experiment_id`s before anything runs — needed by BOTH
the SGE array submitter AND the pipeline. `resolve_experiment_ids(mixed, raw_root)`: a `_coll` entry
expands (glob children, group by plate_token → `{coll}_{plate}` ids); a bare id passes through the
singular `resolve_experiment_id`. DRY (mints nothing itself). `resolve-experiment-ids` CLI writes the
flat EXP_FILE + prints N for `qsub -t 1-N`; the array template is unchanged.

---

## What's BUILT (PR #21, real-data-proven) vs REMAINING

**BUILT + green (183+ tests; Stage 0–2 proven on real Keyence chem28c):**
- Grammar parsers + `resolve_experiment_ids` + `resolve-experiment-ids` CLI.
- `collection_acquisition_ingest` (collapse at ingest → union → one valid acquisition inventory)
  + `derive_position_well_mapping` + `ingest-collection-acquisition` CLI.
- Registry `EmbryoMergePolicy` (NORMAL/BRIDGE/FRACTURE) + `n_sources`/`merge_policy` columns.
- Real-data fix: union offsets `time_index_claimed` in lockstep with `time_index` (cell-key collision).
- Detection primitives `is_collection_plate_id` / `parse_collection_name_from_plate_id`.

**REMAINING — build order (design locked, not yet built):**
1. **Early classify step** → experiment-level `is_collection` artifact; thread it to branching rules.
2. **Collection ingest emits** the `time_index → start_age_hpf` mapping product.
3. **Stage/validation consume** the declared fact + the mapping (collection) / plate_metadata (single).
4. **Revert the detour:** remove `pool_well_acquisition_rows_across_sources` from
   `select_well_acquisition_rows.py` (that "pool at materialization" relocation was abandoned —
   collapse stays at acquisition ingest, where `collection_acquisition_ingest` already does it).
5. **Wire DAG + GPU run** on 2 wells of chem28c → detection→tracking→registry. SETTLES the SAM2
   `track_id` question. Then finalize FRACTURE keying, un-draft #21, merge.

**DAG note (from a dry-run):** targeting a collection id hit `MissingInputException` on
`ingest_scope_metadata` because the collection id has no raw dir of its own. That's expected — the
early classify step + collection-ingest rule must produce the collection's acquisition inventory
(from its `_coll` dir), and the collection must NOT be routed as a plain per-experiment scope read.
The clean Snakemake pattern is type-exclusive rule wiring (like the existing native-vs-dropin
FRONT_END_MODE): a collection experiment's acquisition inventory + position mapping come from the
collection-ingest rule; downstream materialize→object_extraction is the UNCHANGED native path.

---

## MVP boundary (what this deliberately does NOT do)
- No new spine identifier — `plate_token`/`collection` are derived from `experiment_id`, not threaded
  through every contract.
- plate_metadata schema unchanged (age is a separate product).
- Single (non-collection) experiments are byte-identical to today throughout.
