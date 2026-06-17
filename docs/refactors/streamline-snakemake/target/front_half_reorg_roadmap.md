# Front-Half Reorg Roadmap — wire the graph, then go per-well (🟢 TARGET)

**Status:** sequencing + package-layout roadmap, mdcolon 2026-06-16. The *physical reorg* companion
to the data-flow specs. Where the other `target/` docs say what each artifact MEANS, this one says
where the code LIVES and in WHAT ORDER to move it.
**Companion to:** `current_state_and_next_steps.md` (verified on-disk state),
`recompose_yx1_front_end.md` (YX1 build), `acquisition_inventory_flow.md` (scope-specific upstream
record + eligibility), `frame_inventory_handoff_contract.md` (the shared downstream seam),
`well_id_throughline_refactor_plan.md` (the Scopes 1–5 spine this roadmap rides), and
`schema_layout.md` (the domain-owned contract layout this roadmap now assumes).

> **🟡 IMPLEMENTATION STATUS (2026-06-17): plan LOCKED, building NEXT — in stages.** Design is settled
> and internally reviewed; we implement the YX1 path next, **one step at a time, each verified before
> the next** (Steps 1→5 below). Do NOT do it as one big change — the byte-compare gate (Steps 3→4)
> and the per-step verifies are the whole point. This commit is the plan; code follows incrementally.

---

## 🎯 OVERALL GOAL — get the pipeline running PER-WELL

The end state is a pipeline where **one well flows end-to-end on its own**
(`stitch_well[well_id] → segment → features → QC`), so a single well can be rerun, debugged, and
reasoned about in isolation. That is the destination.

> **What this phase DOES (and the precise line it stops at):** the two beats are split by **where**
> per-well execution is built, not whether:
>
> ```
> BEAT 1 (this roadmap):  wire the GRAPH from raw read through a PER-WELL stitch → per-well
>                         frame_inventory handoff. The microscope-aware stages (incl. stitch)
>                         become per-well here, because stitch IS the last microscope-aware stage
>                         and frame_inventory is born per-well. STOPS at a validated per-well
>                         frame_inventory shard.
> BEAT 2 (next, separate): make the DOWNSTREAM (already-agnostic) stages per-well — segmentation,
>                         features, QC — onto the well spine via well_runner.py. Includes repointing
>                         segmentation to read the frame_inventory shard.
> ```
>
> **🏁 THE FINISH LINE for this roadmap = a validated PER-WELL `frame_inventory` shard.** That is the
> END of the microscope-aware part: raw data flows `ingest → map → join → discover →
> stitch_well[well_id] → build/validate_frame_inventory_for_well` and produces a valid per-well shard.
> **The handoff is the boundary; Beat 1 builds the producer side (up to and including the shard +
> its gate). Beat 1 does NOT repoint the consumer side** — segmentation/features/QC keep reading what
> they read today until Beat 2 migrates them onto the shard. **The stitcher is the dam; Beat 1 gets
> the river over the dam and into a validated per-well shard. Beat 2 carries it downstream.**
>
> **Why stitch is per-well in Beat 1 but segmentation isn't:** the per-well *fan* is one early shared
> point (`discover_wells`); the microscope *boundary* is at stitch. Stitch sits at the intersection —
> it is both the last microscope-aware stage AND must be per-well so the already-per-well
> frame_inventory has a per-well producer. The downstream stages are agnostic and already work at
> their current grain; re-graining them is a separate, lower-risk pass (Beat 2).

**Two kinds of "per-well" — keep them distinct** (the thing the other docs blur):
- **per-well IDENTITY** = `well_id` means one global thing. Covered (Scope 1/2).
- **per-well EXECUTION** = actually run ONE well through a stage. **Beat 1 builds it for the
  microscope-aware spine** (`stitch_well` + `build_frame_inventory_for_well`, which already exist
  per-well). **Beat 2 extends it across the agnostic back half** via `well_runner.py` —
  `selected_well_ids_for_experiment` and the generic per-well stage template are still the hole there.

---

## End Goal (artifact river)

Build a front-half architecture where each artifact has one meaning and one home:

```
raw scope data
  → canonical metadata + acquisition inventory
  → physical well discovery
  → scope/acquisition validation and eligibility
  → run-well selection
  → microscope-specific image materialization
  → shared frame inventory contract
```

**Key decision — `discovered_wells` is NOT post-QC.** `discovered_wells.txt` means "these wells
physically exist in canonical metadata." It does NOT mean "passed validation" or "should run."

```
run_wells = discovered_wells ∩ target_wells [ ∩ eligible_wells ]     ← ∩ eligible is TARGET, deferred
```

`eligible_wells` comes from explicit validation/eligibility artifacts, never from `discover_wells`.

> **⚠️ Eligibility is the TARGET shape, NOT on the YX1 path.** On the YX1 path today and through Beat
> 1, `run_wells = discovered ∩ target` (the `∩ eligible` term is **absent in code** and stays so).
> `well_acquisition_summary__{scope}.csv` + `resolve_acquisitions` are **Keyence-weighted** (they
> exist to quarantine re-acquisition collisions, which YX1 can't have). For YX1, "is this well good?"
> is answered by the **per-well frame_inventory validator** (Step 5), not a separate eligibility
> artifact. So every `well_acquisition_summary` / `resolve_acquisitions` / `∩ eligible` row below is
> **target architecture, built when Keyence forces it** — see "NOT in this plan."

---

## Core Artifact Semantics

| Artifact | Meaning | Owns | Grain |
|---|---|---|---|
| `scope_metadata__{scope}.csv` | raw scope metadata normalized for this microscope | `ingest_scope_metadata` | experiment |
| `acquisition_inventory__{scope}.csv` | raw acquisition units/axes before well fanout | `ingest_scope_metadata` | experiment |
| `position_well_mapping.csv` | raw position/series → well mapping | `map_positions_to_wells` | experiment |
| `scope_metadata_mapped.csv` | canonical metadata after `well_id` minting | `join_series_mapping_to_scope_metadata` | experiment |
| `discovered_wells.txt` | physical well identities from canonical metadata | `discover_wells` | experiment fan point |
| `well_acquisition_summary__{scope}.csv` | per-well acquisition/stitch eligibility | `resolve_acquisitions` / eligibility | experiment |
| `run_wells` | computed set (not necessarily persisted at first) | well_runner | planning/runtime selection |
| stitched images | materialized image artifacts | `materialize_stitched_images` | per well / image tree |
| frame inventory | shared downstream image contract | `frame_inventory` | per-well then merge |

---

## The handoff strictness model — ONE manifest, THREE producers, ONE validator

The native/external asymmetry is **not** two validators. It is one contract filled three ways.

```
THE CONTRACT (one):     frame_inventory.csv  — downstream consumes the MANIFEST, never "folder vibes."

THREE PRODUCERS of it:
  native      → layout.py CONSTRUCTS source_image_path from the frame key (canonical, enforced).
  external A  → build_frame_inventory_from_layout.py PARSES a canonical tree → manifest.  ← SUGAR, not a 2nd contract
  external B  → user AUTHORS the manifest; paths arbitrary but explicit, readable, unique, complete.

ONE VALIDATOR checks all three (THREE tiers — Beat 1 ships the first two):
  Level 1   — identity/uniqueness  (cheap, dataframe-only; exists today, just fix the key)   ← Beat 1
  Level 1.5 — path EXISTENCE       (every source_image_path resolves; cheap, no image open)  ← Beat 1
  Level 2   — file CONTENT         (images OPEN, dims match, µm/px>0, BF contiguous, rectangular)
                                   ← the strict gate; NEXT PHASE, identical for native + external.
```

**Strictness lives in the manifest (identity + existence + uniqueness — never bends).
Flexibility lives only in physical LAYOUT (bends only at the external door, only via a manifest).**

| dimension | native | external A (canonical tree) | external B (manifest) |
|---|---|---|---|
| frame identity (atoms) | strict | strict | strict |
| path exists / readable / unique | strict | strict | strict |
| physical layout | **canonical, enforced (mismatch = FAIL)** | canonical (the helper parses it) | arbitrary |
| how the manifest is obtained | `layout.py` constructs | `build_frame_inventory_from_layout` parses *(next phase)* | user authors |

> **Atoms vs derived ids.** The user/producer authors ATOMS (`experiment_id, well_index, channel_id,
> time_index`) + path + dims + µm/px. `well_id`/`image_id` are DERIVED — the validator recomputes
> them from atoms and fails loud on disagreement, never trusts a supplied composed id.
> (Reconciles `frame_inventory_handoff_contract.md` Decisions 3, 10, 11–12: native mismatch is a
> hard FAIL, not a warning; the warning survives ONLY at the external drop-in door.)

> **The frame-inventory contract PURL is domain-owned, not global-schema-owned.**
> The source of truth for this handoff is
> `image_materialization/stitched/contracts/frame_inventory_contract.py`. The existing
> `schemas/frame_contract.py` is a legacy compatibility module only: do not add new semantics there.
> It is currently stale three ways (named `frame_contract`; key uses `time_int` not `time_index`;
> keyed on `well_id` not the atoms). Fix the domain contract first, then migrate importers to it.

---

## Domain-Level Contracts — own meaning beside the data product

This roadmap uses **domain-level schemas**: a data product's contract lives with the package that
owns that product. Shared code provides validation mechanics only.

```
shared/table_contracts.py
  TableContract
  assert_columns_present
  assert_unique_on_key
  assert_positive_numeric
  assert_allowed_values

metadata_ingest/well_discovery/contracts.py
  discovered_wells.txt contract

metadata_ingest/contracts/well_acquisition_summary.py
  well_acquisition_summary__{scope}.csv contract

metadata_ingest/scope/yx1/acquisition_inventory.py
  YX1 acquisition inventory contract  (already owns schema + key)

metadata_ingest/scope/keyence/acquisition_inventory.py
  Keyence acquisition inventory contract

image_materialization/stitched/contracts/frame_inventory_contract.py
  frame_inventory.csv contract: atoms, derived ids, required columns, unique key
```

**Rule:** contract modules own meaning; `shared/table_contracts.py` owns mechanics. The old
`data_pipeline/schemas/` package remains as a legacy compatibility layer until importers are
migrated. Do not widen it with new target semantics.

> **⚠️ Don't build contracts you don't need yet (anti-whale guard).** This section shows the full
> domain-contract END STATE — it is NOT a prerequisite checklist. Build each contract module only when
> a step actually consumes it (follow `schema_layout.md`'s migration order). For the YX1 path that
> means **exactly two** contracts, built when their step needs them:
> - `well_discovery/contracts.py` + `shared/table_contracts.py` — **Step 1** (discovery consumes them).
> - `image_materialization/stitched/contracts/frame_inventory_contract.py` — **Step 2** (the smallest,
>   most load-bearing contract: the stitch-consume cutover and the native producer both key on its
>   atoms, so it earns its place at Step 2, before stitch is touched).
>
> **DEFERRED (do NOT pre-build):** `metadata_ingest/contracts/well_acquisition_summary.py` (eligibility
> — Keyence-weighted, see "NOT in this plan") and `scope/keyence/acquisition_inventory.py`. Building
> all five up front re-creates the swallow-the-whale problem the acquisition-inventory doc warns
> against.

See `schema_layout.md` for the target layout, import rules, and migration order.

---

## Target Package Layout

> **Legend:** ✅ built+wired · 🔨 NEW (this plan) · 🔧 modified · ⏸ deferred (Keyence/Beat 2) · ⏳ move LAST

```
data_pipeline/

  shared/                                    # KINGDOM: identity + generic mechanics (no domain logic)
    identifiers/                             # ✅ BUILT + wired (Scope 1)
      constructors.py                        #   build_well_id / build_image_id (_t{time_index:04d})
      parsers.py                             #   split_well_id / parse_image_id
      validators.py                          #   validate_well_id (fail loud on bare A01) / recompute_and_check
    table_contracts.py                       # 🔨 NEW (Step 1/2) — generic MECHANICS only: TableContract,
                                             #   assert_columns_present/unique_on_key/positive_numeric/allowed_values.
                                             #   Imports NOTHING domain.
    path_contracts.py                        # (existing) shared path helpers

  metadata_ingest/                           # UPSTREAM of stitch — scope-specific
    plate/
      plate_processing.py

    scope/
      yx1/
        acquisition_inventory.py             # ✅ BUILT — owns YX1 schema + tensor cell key
        extract_yx1_scope_metadata.py        # ✅ BUILT — the ONE raw ND2 read
        map_yx1_positions_to_wells.py        # ✅ BUILT (renamed from map_series_to_wells)
        generate_xy_reference.py             # ✅ exists (offline tool; not a DAG node)
        validate_xy_reference_grid.py        # ✅ exists (the exemplar validator)
      keyence/                               # ⏸ SEPARATE TRACK (not YX1 Beat 1)
        acquisition_inventory.py             # ⏸ DEFERRED (Keyence collision model)
        extract_keyence_scope_metadata.py
        map_keyence_positions_to_wells.py
        resolve_keyence_acquisitions.py      # ⏸ DEFERRED (collision resolve — Keyence-only)
      shared/
        acquisition_checks.py                # ✅ BUILT — scope-agnostic check primitives
        apply_position_mapping.py            # ✅ exists (the join / convergence line)
        validate_physical_well_mapping.py    # ✅ exists

    well_discovery/                          # 🔨 NEW (Step 1) — SHARED, source-contract-dispatched
      __init__.py
      contracts.py                           #   discovered_wells.txt contract (uses table_contracts)
      from_scope_metadata.py                 #   discover_wells_from_scope_metadata(mapped_csv → wells)
      from_frame_inventory.py                #   ⏸ DEFERRED (drop-in twin; external path)
      discover_wells.py                      #   dispatcher by SOURCE, not microscope

    contracts/                               # ⏸ DEFERRED home (eligibility — Keyence-weighted)
      well_acquisition_summary.py            #   ⏸ NOT built on YX1 path (∩ eligible deferred)

  image_materialization/                     # DOWNSTREAM of stitch — frame_inventory lives HERE
    stitched/
      materialize_stitched_images.py         # 🔧 Step 4: thin DISPATCHER (microscope → backend)
                                             #   today: 1 mixed 43-branch file in metadata_ingest/stitched_index/
      layout.py                              # 🔨 NEW (Step 4) — the TREE: native pixel paths; ONE FILE
                                             #   (→ subpackage later only if it earns it); imports identifiers
      frame_inventory.py                     # 🔧 Step 5: the TABLE — build + validate; IMPORTS the contract
                                             #   below; native per-well producer; keys on time_index
      contracts/
        frame_inventory_contract.py          # 🔨 NEW (Step 2) — the MEANING (PURL): identity ATOMS,
                                             #   DERIVED ids, REQUIRED columns, UNIQUE_KEY (atoms; time_index)
      scope/
        yx1/
          materialize_yx1_stitched_images.py # 🔨 Step 4 — ND2 tensor slice + LoG focus
        keyence/
          materialize_keyence_stitched_images.py  # ⏸ stub now (route to legacy); full = Keyence track

    shared/                                  # ⏳ MOVE LAST (only after stitched pkg is stable)
      log_focus.py                           #   ⏳ keep in image_building/shared/ until then
      frame_tiler.py                         #   ⏳ LIVE engine — do NOT move yet
      image_io.py

  schemas/                                   # ⚠️ LEGACY COMPAT ONLY during migration
    frame_contract.py                        #   do NOT add target semantics; retire gradually
    stitched_image_index.py                  #   ⏸ RETIRE (Step 5c) — intermediate absorbed into frame_inventory

  pipeline_orchestrator/
    tasks.py                                 # thin: parse + delegate only
    orchestration/
      paths.py                               # ✅ PIPELINE_STEPS registry — TABULAR artifact paths
      well_runner.py                         # ✅ partial — concat_well_shards_to_file exists;
                                             #   selected_well_ids_for_experiment = Beat 2 hole
```

**Timing caveats:**
- Do NOT move `log_focus.py` / `frame_tiler.py` into `image_materialization/shared/` until the
  stitched package is stable.

### `PIPELINE_STEPS` registry deltas (`orchestration/paths.py`) — the path-of-record changes

> **The plan moves files; this names the ARTIFACT PATHS.** Hard Constraint 1: every artifact path
> comes from `paths.py`, no raw strings in rules. **Today stitch VIOLATES this** —
> `stitched_image_index.csv`, the `.materialize_stitched_images.done` sentinel, and `frame_contract.csv`
> are **inline strings in the Snakefile** (`Snakefile:305-337`), NOT registry rows. The plan must
> register stitch and re-grain `frame_inventory`. (`.validated`/`.provenance.json`/`.done` are DERIVED
> by helpers, never listed as artifacts.)

**Already correct (no change):** `ingest_scope_metadata` (incl. `acquisition_inventory__{scope}.csv`),
`map_positions_to_wells`, `join_series_mapping_to_scope_metadata`, `discover_wells`.

**🔨 NEW row — `stitch_well` (register the off-registry stitch; Step 4b):**
```python
"stitch_well": {
    "stage": "built_image_data",            # the pixel store root (off the experiment_metadata stage)
    "fanout": PER_WELL,                      # ← the re-grain: was inline {experiment}, becomes per-well
    "artifacts": {
        # the stitched IMAGE TREE is OFF-REGISTRY (layout.py owns pixel paths). The registry holds
        # only the per-well DONE sentinel + (transitional) the per-well stitched index, if kept.
        "done": ".well_{well_id}.done",      # per-well sentinel (replaces the experiment .done)
    },
},
```
> Note: the image files themselves stay **off-registry** — `layout.py::stitched_frame_path(...)` owns
> them (image trees are not tabular artifacts). Only the sentinel (+ any retained index) is a row.

**🔧 CHANGED row — `frame_inventory` (de-stale; Step 5):**
```python
"frame_inventory": {
    "stage": "experiment_metadata",          # OPEN: dedicated frame_inventory/ stage (findings #5) — decide at build
    "fanout": PER_WELL_THEN_MERGE,           # already correct
    "artifacts": {
        "inventory": {
            PATH_MODE_PER_WELL: "{well_id}_frame_inventory.csv",        # already correct
            PATH_MODE_MERGED:   "{experiment_id}_frame_inventory.csv",  # already correct
        },
    },
    # CHANGE: producer repoints from frame_contract.csv → per-well stitch output (Step 5a);
    #         product keys on time_index, not time_int (Step 5c, gradual).
},
```

**⏸ RETIRE from the path-of-record (Step 5c, gradual):**
- `stitched_image_index.csv` + its `.validated` — **delete** (intermediate absorbed; never was a registry row, so this is removing the inline Snakefile strings).
- `frame_contract.csv` + `.frame_contract.validated` — **collapse on the YX1 producer**; the inline Snakefile strings retire as downstream readers migrate to the shard. `schemas/frame_contract.py` stays legacy-compat.

**⏸ DEFERRED rows (Keyence-weighted; do NOT add on the YX1 path):** `resolve_acquisitions` (→
`acquisition_conflicts__{scope}.csv`, `acquisition_resolution__{scope}.csv`,
`resolved_acquisition_inventory__{scope}.csv`, `well_acquisition_summary__{scope}.csv`). These are the
eligibility artifacts — registered only when Keyence forces them.

### The three colliding names — drawn straight (acquisition inventory vs frame inventory vs layout)

These three got blurred; the lines are clean once stated:

```
acquisition_inventory   metadata_ingest/scope/{scope}/   the RAW record — UPSTREAM of stitch, scope-specific
frame_inventory         image_materialization/stitched/  the FRAME table — DOWNSTREAM of stitch, agnostic
layout.py               image_materialization/stitched/  the TREE — WHERE the .tif pixels live (a path)
```

- **Acquisition inventory is NOT frame inventory.** Acquisition inventory lives in
  `metadata_ingest` (the scope-specific raw record, before stitch). Frame inventory lives in
  `image_materialization` (the agnostic per-frame table, after stitch). The stitcher is the boundary.
  *(Today's `metadata_ingest/frame_inventory/` is a legacy behavior-preserving adapter; the target
  HOME for the frame inventory is `image_materialization/stitched/`.)*
- **`layout.py` ≠ `frame_inventory.py`.** `layout.py` answers *"where does this frame's pixel file
  go?"* → a **path**. `frame_inventory.py` answers *"what do we know about this frame, and is it
  valid?"* → a **table + the gate**. `layout` builds paths → frame_inventory records them → the
  validator verifies they resolve. `layout` never reads the table; `frame_inventory` never invents a
  path (native: gets it from `layout`; drop-in: takes it as authored).
- **`frame_contract` is just the OLD NAME for `frame_inventory`** (rename target, handoff doc
  Decision 18) — it is NOT a separate artifact and does NOT appear as a box in the target tree. Only
  surviving as `schemas/frame_contract.py` for legacy compatibility until importers move to the
  domain contract.

> **`layout.py` is ONE FILE now — it MAY graduate to a subpackage later, if it earns it.** Today it
> only needs the path constructors. When Mode A (`build_frame_inventory_from_layout`, the parse-tree
> sugar) and the native enforcement ("mismatch = FAIL") actually get written — Phase 4/5 — that is
> when `layout.py` → `layout/` (`constructors.py` · `parse.py` · `contract.py`) earns the split. Not
> before. Off-registry either way: `layout` owns IMAGE-TREE paths (off the tabular registry);
> `lib/paths.py` owns TABULAR artifact paths. Different families, same ID grammar (both import
> `shared/identifiers`, neither inline-mints — two-kingdoms holds).

---

## well_discovery/ Organization

`well_discovery` is **shared and source-contract-specific, not microscope-specific.**

```
well_discovery/
  contracts.py
    read_discovered_wells(path) -> list[well_id]
    write_discovered_wells(path, wells)
    validate_discovered_wells(wells)
  from_scope_metadata.py
    discover_wells_from_scope_metadata(mapped_csv, output_wells)
  from_frame_inventory.py
    discover_wells_from_frame_inventory(frame_inventory_csv, output_wells)   # future / drop-in
  discover_wells.py
    dispatcher by SOURCE CONTRACT, not microscope:  source="scope_metadata" | "frame_inventory"
```

Rules: `discover_wells_from_scope_metadata` reads `scope_metadata_mapped.csv`; requires global
`well_id`; emits all physical wells; does NOT check image existence; does NOT filter failed QC; does
NOT dispatch on YX1 vs Keyence (it is BELOW the convergence line — microscope-agnostic).

---

## Validation And Eligibility Split

Per-scope validation logic exists, but it must NOT live inside `discover_wells`.

```
discover_wells                      physical identity only
resolve_acquisitions / eligibility  per-scope acquisition validation, collision resolution,
                                    active_for_stitch decision
well_runner                         discovered ∩ target ∩ eligible
```

Summary contract (the ONLY thing shared orchestration reads — eligibility, not logic):

```
well_acquisition_summary__{scope}.csv
   well_id · active_for_stitch · quarantine_reason
```

- YX1: `active_for_stitch=True` for clean/passthrough wells (every well clean by construction —
  nearly free; steps 3–4 below are Keyence-weighted).
- Keyence: depends on collision/reacquisition resolution.

A broader post-image QC concept gets its OWN artifact later (e.g. `well_qc_summary.csv`). **Do not
overload `discovered_wells.txt`.**

---

## layout.py Ownership

`image_materialization/stitched/layout.py` is the only place that knows native stitched image layout.
It owns the path constructors:

```python
def stitched_well_dir(root: Path, well_id: str) -> Path: ...
def stitched_channel_dir(root: Path, well_id: str, channel_id: str) -> Path: ...
def stitched_frame_path(root, well_id, channel_id, time_index, ext="tif") -> Path: ...
```

**Important rule — `layout.py` imports `shared/identifiers`; it validates/builds identity tokens through
identifiers, never inline-mints:**

```python
well_id  = validate_well_id(well_id)
image_id = build_image_id(well_id, channel_id, time_index)
return stitched_channel_dir(root, well_id, channel_id) / f"{image_id}.{ext}"
```

So layout is centralized; ID grammar still belongs to `shared/identifiers`.

---

## Phases — wiring the graph (Beat 1)

> Each phase is a graph-wiring step. None of them shard execution per-well yet — that is Beat 2,
> mapped separately after this. Phases are sequenced by RISK and DEPENDENCY (identifiers gates
> layout/validator), not by artifact order alone.

### Phase 0: Identifiers (`shared/identifiers/`) — the foundation, do FIRST
Empty/additive/zero-risk. Constructors (`build_well_id`/`build_image_id` `_t{time_index:04d}`),
parsers, validators (`validate_well_id` fails loud on bare `A01`; `recompute_and_check`). Layout and
the validator both depend on this — it cannot come second. (Scope 1.)

### Phase 1: Extract Well Discovery From tasks.py
New: `well_discovery/{__init__,contracts,from_scope_metadata}.py`. Edit: `tasks.py`. Artifact:
`experiment_metadata/{experiment}/discovered_wells.txt`. DAG step `discover_wells`, experiment fan
point. **Behavior change: none.** Tests: unique wells preserve first-seen order; duplicates collapse;
missing `well_id` fails loud; local IDs like `A01` fail validation; `tasks.py` delegates (no
pandas/business logic).

### Phase 1.5: Establish Domain-Level Contracts
Create the contract homes before changing frame-inventory behavior or moving packages:

```
shared/table_contracts.py
metadata_ingest/contracts/well_acquisition_summary.py
image_materialization/stitched/contracts/frame_inventory_contract.py
```

The frame-inventory contract is atom-based:

```
FRAME_INVENTORY_IDENTITY_ATOMS = ("experiment_id", "well_index", "channel_id", "time_index")
DERIVED_COLUMNS_FRAME_INVENTORY = ("well_id", "image_id")
UNIQUE_KEY_FRAME_INVENTORY = FRAME_INVENTORY_IDENTITY_ATOMS
```

It defines required columns, derived-id recomputation rules, and compatibility aliases for the
current transition (`time_int` -> `time_index`, `stitched_image_path` -> `source_image_path`) only
where needed. `schemas/frame_contract.py` becomes legacy compatibility; new code imports the domain
contract. **Behavior change: none** until importers are migrated.

### Phase 2: Formalize Acquisition Inventory Contracts (YX1 already shipped — Keyence catches up)
The scope/shared validator split is already built for YX1 (`acquisition_checks.py`,
`scope/yx1/acquisition_inventory.py`). This phase is mostly **Keyence catching up** to YX1's pattern.

```
scope/shared/acquisition_checks.py   assert_columns_present · assert_positive_column ·
                                     assert_unique_on_key · assert_channel_mapping_consistent
scope/yx1/acquisition_inventory.py   declares YX1 schema + uniqueness key, calls shared primitives  ✅ DONE
scope/keyence/acquisition_inventory.py  later: Keyence schema + raw/collision key, calls primitives
```

Artifact `acquisition_inventory__{scope}.csv` on `ingest_scope_metadata` (experiment grain, before
`well_id` fanout). YX1 record-only already emits; Keyence later.

### Phase 3: Add Acquisition/Stitch Eligibility
New artifact `well_acquisition_summary__{scope}.csv`; DAG step `resolve_acquisitions` (Keyence needs
real conflict resolution; YX1 passthrough). Grain experiment. Behavior: does NOT change
`discovered_wells.txt`; emits per-well eligibility; well-runner uses it when selecting wells. Tests:
quarantined well stays discovered but is excluded from the run set; YX1 passthrough marks clean wells
active; Keyence later marks collision-failed wells inactive.

### Phase 4: Rehome Stitch Into image_materialization — split per-scope backends
> **Two different-risk things used to be fused here; do not implement them in the same commit.**
> First gate the data-source cutover — stitch CONSUMING `acquisition_inventory__yx1.csv` as its
> `(well → position/channel/z)` lookup, killing the in-stitch `_select_yx1_channel_index`/
> `yx1_series_map` re-derivation. This is intended output-preserving and requires a byte-compare.
> Then do the package move into `image_materialization/stitched/` as import churn with no behavior
> change.

```
materialize_stitched_images.py        dispatcher — selects backend by microscope
layout.py                             canonical path constructors (one file; subpackage only if it earns it)
frame_inventory.py                    shared output table: domain-contract import + build + validate
scope/yx1/materialize_yx1_stitched_images.py        ND2 tensor slice + Z projection / LoG focus
scope/keyence/materialize_keyence_stitched_images.py  TIFF tiles + mosaic + collision-aware
```

Shared layer = `layout.py` + frame inventory schema/validator (+ focus primitive where honestly
reusable) — NOT shared producer code. Canonical native layout falls out **by construction** (backends
write through `layout.py`). Do NOT move `log_focus.py` / `frame_tiler.py` yet — only after this
package passes smoke tests.

### Phase 5: Flip the validator to strict-for-native + remove legacy
- **Validator flip:** native layout mismatch warning → hard FAIL (`source_image_path` derived for
  native; warning survives only at external drop-in). This is the only thing Phase 5 ADDS over
  Phase 4 — the layout was already enforced by construction.
- **Legacy removal (after smoke tests):** no shims; direct import updates; delete old packages;
  grep/import audit for stale refs. Candidate removals: `metadata_ingest/stitched_index/`,
  `image_building/yx1/`, `image_building/keyence/`, `image_building/scope/`.
- Validation: focused unit tests pass; `snakemake -n` parses; front-end smoke path flows through
  `discover_wells`; stitch smoke passes when GPU/runtime constraints allow.

---

## Defaults And Non-Negotiables

- **The goal of this roadmap is to reach the END of the microscope-aware pipeline** — raw read
  through stitch to a valid `frame_inventory` handoff (the first agnostic artifact). Overall goal
  is per-well; this phase wires the graph to that finish line so per-well (Beat 2) is possible next.
- `discovered_wells.txt` is physical identity, not QC-passed identity.
- validated/eligible wells are separate and intersected later.
- `discover_wells` does NOT dispatch by microscope; image materialization DOES.
- `layout.py` (one file; subpackage later only if it earns it) owns stitched image paths; uses
  `shared/identifiers` for ID grammar.
- frame inventory lives in `image_materialization/stitched/` (DOWNSTREAM of stitch); acquisition
  inventory lives in `metadata_ingest/scope/` (UPSTREAM). They are NOT the same artifact.
- `paths.py` owns TABULAR artifact paths and fanout enforcement; `layout.py` owns OFF-registry image
  paths. Different families, same ID grammar.
- Domain-level contract modules own schema meaning; `shared/table_contracts.py` owns reusable
  validation mechanics. `data_pipeline/schemas/` is legacy compatibility during migration.
- `tasks.py` only parses and delegates.
- sidecars are derived via path helpers, not registered as first-class artifacts.
- ONE manifest contract; shared validator enforces manifest invariants, while the native builder
  enforces canonical layout by construction through `layout.py`.

---

## 🛠️ YX1 IMPLEMENTATION PLAN — from where we actually are (verified on disk 2026-06-16)

> The phases above are the full front-half target. This section is the **concrete next YX1 work**,
> grounded in what is actually on disk today — not the doc's aspirations.

> **🚧 BINDING COMMIT BOUNDARIES (review guardrails, 2026-06-17 — not optional).** The design is clean;
> the implementation gets muddy only if a step is swallowed whole. So these are **separate commits,
> each verified before the next** — do NOT land them together:
> - **4a ≠ 4b.** `4a` moves files/imports ONLY (no DAG grain change) → byte-compare. `4b` converts
>   `materialize_stitched_images` → `stitch_well[well_id]` ONLY → byte-compare (per-well union == old
>   experiment output). Moving files and flipping the fan point in one diff is where refactor gremlins
>   nest.
> - **Step 5 splits into 5a–5e** (one wiring change per commit): `5a` build frame_inventory from the
>   per-well stitch output · `5b` add the validated per-well shard as a live front-end target · `5c`
>   absorb the path-existence check + retire `stitched_image_index` on the YX1 path · `5d` simplify the
>   merge (call `concat_well_shards_to_file`, drop the double-validate) · `5e` keep a `time_int` compat
>   alias for not-yet-migrated downstream readers.
> - **Do NOT migrate segmentation/features/QC in Beat 1.** Repointing consumers is Beat 2. "Everything
>   to frame_inventory" means the **YX1 PRODUCER** path — downstream consumers migrate gradually.

### Reality baseline (verified)

| Thing | State |
|---|---|
| Scope 1 `shared/identifiers/` | ✅ **built + wired** (`constructors`/`parsers`/`validators`/README; 12 importers, incl. stitch) — **Phase 0 is DONE** |
| YX1 metadata (Phase 1) | ✅ shipped — `extract_yx1_scope_metadata` · `map_yx1_positions_to_wells` · join · discover all run |
| YX1 acquisition inventory (Phase 1C) | ✅ shipped — record-only, 161k-row smoke verified |
| `metadata_ingest/well_discovery/` | ❌ **does not exist** — discovery still inline in `tasks.py` |
| Stitch consumes the inventory | ❌ **no** — still inline `_select_yx1_channel_index` (:121), `yx1_series_map` (:445,:501), `drop_duplicates(keep="first")` (:431) |
| `image_materialization/` | ❌ **does not exist** — stitch lives in `metadata_ingest/stitched_index/materialize_stitched_images.py` (43 microscope branches) |

**Net:** identifiers is already off the list. The real next work is **5 steps, increasing risk**
(Step 4 has two parts): extract discovery → front-half domain contracts → stitch consume cutover →
[4a] rehome+split [4b] re-grain stitch per-well → activate the (already-built) per-well
frame_inventory branch as the live spine.

### ⚠️ THE TRUE DAG TODAY (verified on disk 2026-06-17) — read before the steps

The stitch→handoff stretch is **more wired than the roadmap implied, and at the WRONG GRAIN.** What
is actually on disk:

```
materialize_stitched_images[{exp}]   → stitched_image_index.csv        ← EXPERIMENT grain (one rule, {experiment})
        ↓
validate_stitched_image_index[{exp}] → .stitched_image_index.validated
        ↓
build_frame_contract[{exp}]          → frame_contract.csv              ← EXPERIMENT grain, the LEGACY live spine
        ↓
validate_frame_contract[{exp}]       → .frame_contract.validated
        ↓ ───────────────── per-well fan currently happens HERE (too late) ─────────────────
build_frame_inventory_for_well[well_id]  → per_well/{well_id}/{well_id}_frame_inventory.csv  ✅ exists
        ↓                                  (an ADAPTER: reads frame_contract.csv, SELECTS this well's rows)
validate_frame_inventory_for_well[well_id]   ✅ exists (WEAK — schema+nulls+key only; Level-2 deferred)
        ↓
merge_frame_inventory[{exp}]         → experiment-level frame_inventory  🏁 ← DEAD branch (nothing targets it)
```

**Four load-bearing facts this surfaces (from `frame_inventory_well_runner_audit.md`):**
1. **`frame_inventory` is an ADAPTER over `frame_contract.csv`, not over stitch directly.** The
   per-well shards are built by selecting rows from an experiment-grain `frame_contract.csv`.
2. **The per-well FAN is currently LATE** — at `build_frame_inventory_for_well`, downstream of an
   experiment-grain `frame_contract`. The handoff-contract TARGET is to fan at `discover_wells` and
   make **stitch itself** per-well (`stitch_well[well_id]`). So "re-grain stitch per-well" = **move
   the fan point earlier** (Step 4).
3. **The frame_inventory branch is DEAD** — `rule all` stops at features; `merge_frame_inventory` is
   never requested; **segmentation still reads `frame_contract.csv` directly.** Reaching the finish
   line = **making the per-well frame_inventory the live spine and retiring `frame_contract`** (Step 5).
4. **The per-well rules ALREADY EXIST** (`build_/validate_/merge_frame_inventory`). Step 5 is
   **wiring + collapsing duplicates, not building** — point the live path at the shards.

**The TARGET DAG after Steps 4b+5 (what we collapse TO):**

```
discover_wells (checkpoint = THE FAN)
        ↓  ⟱ per-well ⟱
stitch_well[well_id]                  → per-well stitched images (via layout.py)   🔴 re-grained (4b)
        ↓
build_frame_inventory_for_well[well_id]  → {well_id}_frame_inventory.csv  (NATIVE per-well producer; keys on time_index)
        ↓
validate_frame_inventory_for_well[well_id]  (absorbs the old stitched-index file-existence check)
        ↓
merge_frame_inventory[{exp}]          → 🏁 the agnostic handoff (LIVE; segmentation reads the shard)
```

> **GONE on the YX1 path after Step 5:** `stitched_image_index.csv` + `validate_stitched_image_index`
> (intermediate, absorbed). The product speaks **`time_index`** natively. `frame_contract.csv` is
> collapsed for the YX1 PRODUCER, but **downstream `frame_contract` readers (segmentation, features)
> migrate GRADUALLY** — a `time_int` compat alias survives during transition; no big-bang rename.
> **The YX1 PRODUCER path moves to frame_inventory as the canonical handoff; downstream consumers
> migrate gradually (Beat 2).**

### Step 1 — Extract `well_discovery/` from `tasks.py`  *(low risk, NO behavior change)*

Pure structure; touches no images; the one Beat-1 graph piece genuinely missing.
- Create `metadata_ingest/well_discovery/{__init__,contracts,from_scope_metadata,discover_wells}.py`.
- Move the `discover-wells` logic out of `tasks.py` → `discover_wells_from_scope_metadata(mapped_csv,
  output_wells)`; `tasks.py` just delegates (no pandas). Dispatcher keys on **source**
  (`scope_metadata`), not microscope.
- **Verify:** `snakemake -n` parses; 20250912 produces identical `discovered_wells.txt`; tests (dup
  collapse, missing `well_id` fails, local `A01` fails, `tasks.py` has no business logic).
- **Skip:** `from_frame_inventory.py` (the drop-in twin) — external path, not needed for YX1.

### Step 2 — Establish front-half domain contracts  *(low risk, NO behavior change)*

Add only the **two** contracts the YX1 path consumes (per the anti-whale guard) before touching frame
inventory or moving packages.
- Add `shared/table_contracts.py` for generic mechanics only (if Step 1 didn't already).
- Add `image_materialization/stitched/contracts/frame_inventory_contract.py` as the atom-based PURL
  for the stitched handoff (`experiment_id, well_index, channel_id, time_index` are the key;
  `well_id` and `image_id` are derived/checkable). The Step 3 stitch-consume cutover and the Step 5
  native producer both key on these atoms, so it earns its place here, before stitch is touched.
- **DO NOT add** `metadata_ingest/contracts/well_acquisition_summary.py` — eligibility is deferred
  (Keyence-weighted; see "NOT in this plan").
- Leave `schemas/frame_contract.py` as legacy compatibility; do not add new target semantics there.
- **Verify:** import-only/unit tests for required columns, unique keys, and derived-id rules. No
  Snakemake behavior changes yet.

### Step 3 — YX1 stitch CONSUMES the acquisition inventory  *(data-source cutover; intended output-preserving)*

The heart of it and the payoff of Phase 1C: stitch stops re-deriving what the inventory records.
Three inline derivations die here:

```
_select_yx1_channel_index (:121)     → read channel_index from acquisition_inventory__yx1.csv
yx1_series_map (:445, :501)          → read position_index from the inventory
drop_duplicates(keep="first") (:431) → assert unique on the tensor cell (YX1 can't collide → fail-loud)
```

- Do this **IN PLACE** in `metadata_ingest/stitched_index/materialize_stitched_images.py` — BEFORE
  any rehome. Add `acquisition_inventory__yx1.csv` as a declared stitch `input:` (YX1); replace the
  three derivations with a `(well → position_index, channel_index, z)` lookup; replace `keep="first"`
  with a fail-loud uniqueness assertion on the cell key.
- **Verify:** stitch one YX1 well on 20250912 and **byte-compare** the stitched output against the
  current pipeline's output — must be IDENTICAL (this is a refactor, not a behavior change). GPU step;
  needs the cluster GPU.

> **Why in-place, before the move:** don't combine "new data source" with "new file location." If the
> byte-compare fails you want to know it was the data wiring, not the rehome.

### Step 4 — Rehome + split per-scope **AND re-grain stitch per-well**  *(churn + the grain flip — only after Step 3 is green)*

Two things land together here: the package move (no behavior change) and the **fan-point flip**
(experiment-grain → per-well). Do them in this order, each verified:

**4a — Rehome + split (no behavior change):**
- Create `image_materialization/stitched/`; split the 43-branch file → thin dispatcher +
  `scope/yx1/materialize_yx1_stitched_images.py` (+ a Keyence stub that routes to existing logic so
  Keyence doesn't break).
- Extract `layout.py` (path constructors, imports `identifiers`) — **one file**.
- Move `frame_inventory.py` to `image_materialization/stitched/` (its DOWNSTREAM home) only after it
  imports the domain-owned `contracts/frame_inventory_contract.py`.
- Update Snakefile + `paths.py` + `tasks.py` import paths. **Do NOT move** `log_focus.py` /
  `frame_tiler.py` yet.
- **Verify:** `snakemake -n` parses; re-run Step 3's byte-compare (still identical).

**4b — Re-grain stitch to per-well (`stitch_well[well_id]`):** this is the FAN-POINT FLIP the audit
calls for — move the per-well fan from `build_frame_inventory_for_well` UP to stitch.
- Convert `rule materialize_stitched_images[{experiment}]` → `rule stitch_well[well_id]` (post-fan,
  one well per job). It reads this well's mapping rows + acquisition-inventory rows; writes this
  well's stitched images through `layout.py`; sentinel `.well_{well_id}.done`.
- **Register it** — add the `stitch_well` `PIPELINE_STEPS` row (see "registry deltas" above) and move
  the inline Snakefile path strings (`stitched_image_index.csv`, `.done`) onto `paths.py` /
  `layout.py`. This fixes the current Hard-Constraint-1 violation (stitch paths are inline today).
- The well list comes from the **discover_wells checkpoint** fan (`run_well_ids_for_experiment` =
  `discovered ∩ target`), not the experiment-grain `--selected-wells` param.
- **Why now, not Beat 2:** `build_frame_inventory_for_well` is ALREADY per-well; a per-well consumer
  reading an experiment-grain producer is the grain mismatch. Re-graining stitch is what lets the
  already-per-well frame_inventory have a per-well image producer — required for the finish line.
- **Verify:** `snakemake -n` shows `stitch_well` fanning per well; stitch the 2 wells of `20250912`;
  byte-compare against Step 3's experiment-grain output (per-well union == experiment output).

### 🏁 Step 5 — Activate the per-well `frame_inventory` branch = REACH THE FINISH LINE
The per-well rules **already exist** (`build_/validate_/merge_frame_inventory`). This step is
**WIRING + collapsing duplicates, not building new logic**: make the per-well frame_inventory the
**live spine**, retire the legacy `frame_contract` AND `stitched_image_index` intermediates, and
flip the axis name. This is the END of the microscope-aware pipeline — the top of the dam.

**5a — Native per-well producer:** point `build_frame_inventory_for_well` at the **per-well stitch
output** (Step 4b) instead of selecting rows from an experiment-grain `frame_contract.csv` — the
adapter becomes a native per-well producer.

**5b — Make the branch LIVE up to the shard (NOT the consumer):** add the per-well
`{well_id}_frame_inventory.csv.validated` sentinels to a front-end target so the branch is reachable
and runs (today it's a dead branch — audit finding #1). **Beat 1 stops at the validated shard.**
Repointing **segmentation** to read the shard (instead of `frame_contract.csv`) is the first
**Beat 2** task, not this step — it's a downstream/agnostic-consumer change, on the far side of the
handoff boundary. (Keeping it out of Beat 1 is the scope discipline: Beat 1 owns the producer +
gate; Beat 2 owns the consumers.)

**5c — The YX1 PRODUCER path moves to frame_inventory as the canonical handoff — and the `time_index`
shift is GRADUAL, not big-bang (decided mdcolon 2026-06-17). Downstream consumers migrate gradually
(Beat 2):**
- **The frame_inventory PRODUCT speaks `time_index` natively.** The domain contract (Step 2) declares
  `time_index`; the new per-well producer keys/sorts on `time_index`. **Legacy `frame_contract.csv`
  readers (segmentation, features) migrate GRADUALLY** — keep a `time_int` compat alias / view during
  transition rather than flipping every downstream rule in one commit. No big-bang rename. (The
  end-state is handoff-contract Decision 19 + the Scope-2 `frame_contract → frame_inventory` rename,
  reached incrementally.)
- **Retire `frame_contract` on the YX1 path incrementally:** the `stitched_image_index → frame_
  contract` chain collapses into the per-well `frame_inventory` for the YX1 producer; downstream
  consumers repoint to the shard one at a time. `schemas/frame_contract.py` stays as legacy compat
  until no importer needs it (do not add target semantics there).
- **Retire `stitched_image_index.csv` + `validate_stitched_image_index`** (settled 2026-06-17): it
  was an experiment-grain INTERMEDIATE between stitch and frame_contract. Its three checks (schema,
  unique key, **file-existence**) are exactly what the per-well frame_inventory validator's
  **Level 1.5** does (path-existence only — see "NOT in this plan") — so **absorb the file-existence
  check into the per-well gate and DELETE the intermediate table + validator.** Nothing of value lost;
  one redundant table + one redundant validator gone. (Per-well stitch already knows its own paths via
  `layout.py`, so the intermediate index has no remaining job.) **Image-open/dim checks (Level 2) are
  NOT added here** — existence only.

**5d — Collapse the audit's duplications (cheap, do here while rewiring; ~40 lines total):**
- **Audit #4** — `merge_frame_inventory_shards` hand-rolls concat+drift+sort+key (~30 lines). Replace
  the body with a call to the canonical `well_runner.concat_well_shards_to_file(...)`. One function.
- **Audit #3** — the merged file is validated twice (merge calls `_validate_unique_keys`, then the
  `.smk` `validate_frame_inventory` rule re-checks it). **Drop the merged-level validate rule**; the
  per-well `validate_frame_inventory_for_well` gate stays (the load-bearing one).

- **Verify (the done-for-the-microscope-aware-part check):** for the 2 YX1 wells of `20250912`, the
  chain `ingest → map_positions → join → discover → stitch_well[well_id] → build/validate_frame_
  inventory_for_well` runs end-to-end and produces per-well frame inventories that (a) key on
  `time_index` (no `time_int`), (b) have derived `well_id`/`image_id` recomputing from atoms, and
  (c) have resolving `source_image_path`s. No `stitched_image_index.csv` is produced on the YX1 path
  (absorbed); `frame_contract.csv` is collapsed for the producer but a compat view may still serve
  not-yet-migrated downstream readers. **At this point YX1 has crossed the microscope boundary and
  Beat 1 is DONE** — the validated per-well shard exists. Pointing segmentation/features/QC AT that
  shard is **Beat 2** (downstream consumers, far side of the handoff); they keep reading what they
  read today until then.

> **🏁 When Step 5 passes, the microscope-aware part of the pipeline is DONE for YX1.** That is the
> goal of this roadmap. (Keyence walks the same steps; Beat 2 deepens per-well execution across the
> whole back half via `well_runner.py`.)

### NOT in this YX1 plan (explicit)
- **Eligibility / `resolve_acquisitions` as a SEPARATE artifact** — settled: eligibility IS the
  validator ("are there actually good wells?"), shared mechanics + scope-specific checks, living in
  the **per-well frame_inventory validator** (Step 5), NOT a standalone `well_acquisition_summary`.
  The roadmap's `∩ eligible` term stays **deferred until Keyence** forces collision-based quarantine.
- **Validator — THREE tiers, Step 5 ships the first two:**
  - **Level 1 — identity/uniqueness** (schema, nulls, unique key, derived-id recompute). Exists today.
  - **Level 1.5 — path EXISTENCE** (every `source_image_path` resolves). This is the one check
    absorbed from the retired `validate_stitched_image_index` (Step 5c). Cheap (`Path.exists()`, no
    image open). **Step 5 ships Levels 1 + 1.5.**
  - **Level 2 — file CONTENT** (images OPEN, real dims == declared, BF contiguous, channels
    rectangular). Opens every image → slow. **Deferred to the next, shared phase — NOT Step 5.**
  > The line: Step 5's gate proves *the wells exist and the manifest is self-consistent and its paths
  > resolve* — it does NOT open images. That's the honest "is this well good?" gate for Beat 1.
- **Full `well_runner.py` per-well EXECUTION across the back half** — Beat 2 (Step 4b makes *stitch*
  per-well; Beat 2 makes *every* stage per-well via the runner).
- **Keyence** — separate track (collisions, `resolve_acquisitions`, heterogeneous tile/Z).

### The one-line sequence
> identifiers ✅ → **extract `well_discovery`** (safe) → **domain-level contracts** (safe, as consumed)
> → **YX1 stitch consumes inventory** (data-source cutover; output-preserving, byte-compare) →
> **rehome+split AND re-grain stitch per-well** (`stitch_well[well_id]`; fan-point flip) → **🏁 activate
> the per-well frame_inventory branch** (wiring + collapsing duplicates: retire `frame_contract` +
> `stitched_image_index`, flip `time_int → time_index`, YX1 producer path → frame_inventory) = end of
> the microscope-aware pipeline. Each step verified before the next; the **byte-compare (Steps 3→4) is
> the load-bearing gate**; **Step 4b's fan-point flip and Step 5's live frame_inventory are the finish
> line.** (Downstream consumers migrate in Beat 2 — NOT here.)
