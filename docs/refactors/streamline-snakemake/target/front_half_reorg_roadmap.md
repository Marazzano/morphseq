# Front-Half Reorg Roadmap — wire the graph, then go per-well (🟢 TARGET)

**Status:** sequencing + package-layout roadmap, mdcolon 2026-06-16. The *physical reorg* companion
to the data-flow specs. Where the other `target/` docs say what each artifact MEANS, this one says
where the code LIVES and in WHAT ORDER to move it.
**Companion to:** `current_state_and_next_steps.md` (verified on-disk state),
`specs/front_end/recompose_yx1_front_end.md` (YX1 build), `specs/front_end/acquisition_inventory_flow.md`
(scope-specific upstream record + eligibility), `specs/front_end/frame_inventory_handoff_contract.md`
(the shared downstream seam), `specs/front_end/run_well_schema.md` (discovered/eligible/runnable
well-state + Keyence reacquisition test plan), `specs/well_id_throughline_refactor_plan.md` (the
Scopes 1–5 spine this roadmap rides), and `specs/schema_layout.md` (the domain-owned contract layout).
*(Paths reflect the 2026-06-17 `target/` reorg — see `README.md`.)*

> **🟡 IMPLEMENTATION STATUS (2026-06-17): plan LOCKED, building NEXT — in stages.** Design is settled
> and internally reviewed; we implement the YX1 path next, **one step at a time, each verified before
> the next** (Steps 1→7 below, a STRANGLER migration). Do NOT do it as one big change — the new
> per-well stitch grows BESIDE legacy and is proven on one well at a three-layer comparison gate
> (ending in mdcolon's visual sign-off) before promotion. This commit is the plan; code follows.

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
> **Why stitch is per-well in Beat 1 but segmentation isn't:** stitch sits in the **Stitch Overlap**
> (per-well AND scope-aware); segmentation is **past** the overlap (per-well but agnostic). See
> "🧩 The Two Overlapping Zones" below — it's the lens for the whole plan.

**Two kinds of "per-well" — keep them distinct** (the thing the other docs blur):
- **per-well IDENTITY** = `well_id` means one global thing. Covered (Scope 1/2).
- **per-well EXECUTION** = actually run ONE well through a stage. **Beat 1 builds it for the
  microscope-aware spine** (`stitch_well` + `build_frame_inventory_for_well`, which already exist
  per-well). **Beat 2 extends it across the agnostic back half** via `well_runner.py` —
  `selected_well_ids_for_experiment` and the generic per-well stage template are still the hole there.

---

## 🧩 THE TWO OVERLAPPING ZONES — the organizing lens (read this first)

The pipeline has **two data-engineering zones defined by DIFFERENT axes**, and they **OVERLAP** — they
are not sequential. Almost every awkward placement question in this refactor dissolves once you see
the overlap.

```
 stage:  ingest → map → join │ discover_wells │ stitch_well │ frame_inventory │ segment → features → QC
 ════════════════════════════════════════════════════════════════════════════════════════════════════►

 ┌──────────── MICROSCOPE ZONE (scope-aware production) ───────────┐
 │ raw reads · scope schemas · scope mapping · scope STITCH backends│   YX1 vs Keyence do DIFFERENT work
 │ exits only at the validated per-well frame_inventory shard       │
 └──────────────────────────────────────────────────────────────────┘
                          ┌──────────────────── PER-WELL ZONE (well-SHARDED execution) ──────────────────┐
                          │ every stage runs ONE well at a time, on the well spine (well_runner)           │
                          └───────────────────────────────────────────────────────────────────────────────┘
                          ▲                  ╔═══════════════════╗                  ▲
                   discover_wells            ║  THE STITCH       ║           frame_inventory
                   = BOOTSTRAP / FAN         ║  OVERLAP          ║           = EXIT microscope land
                   (left edge of overlap;    ║  per-well AND     ║           (right edge; pure per-well
                    well_runner born here)   ║  scope-aware      ║            + agnostic from here on)
                                             ╚═══════════════════╝
```

- **MICROSCOPE ZONE** = scope-aware production (raw → … → stitch backends). YX1 and Keyence
  diverge here. **Exits at the validated per-well `frame_inventory` shard.**
- **PER-WELL ZONE** = well-SHARDED execution (discover_wells → … → end). **Starts at discover_wells.**
- **THE STITCH OVERLAP** (`discover_wells → stitch_well → frame_inventory`) = the graph region where
  well-sharded execution begins before microscope-specific production is fully gone. **This
  roadmap's whole job is to build the overlap correctly and exit it cleanly into `frame_inventory`.**

**The special machinery — it lives IN the overlap, which is why it never placed cleanly:**

| Thing | Position | Why it's special |
|---|---|---|
| **`discover_wells`** | LEFT EDGE of the overlap (bootstrap/fan) | turns experiment-grain scope metadata into the per-well world — the moment the Per-Well Zone *begins*. Microscope-agnostic itself. |
| **`well_runner`** | spans the overlap | the per-well scheduler born at the fan; decides `run_wells = discovered ∩ target [∩ eligible]`. |
| **stitch backends** | BODY of the overlap | per-well jobs (Per-Well Zone) + scope-specific code (Microscope Zone) — the ONLY place both are maximally true. |
| **pre-handoff validators** | inside the Microscope Zone / overlap | scope-aware validators over scope-shaped inputs: physical mapping, acquisition inventory, acquisition resolution. They prevent bad microscope evidence from reaching stitch. |
| **`frame_inventory` builder** | RIGHT EDGE adapter | scope-aware producer code writes the shared manifest rows from native stitched output. |
| **`frame_inventory` validator** | EXIT gate | microscope-agnostic shared contract validator. It checks manifest atoms, derived IDs, uniqueness, and path existence/content tiers; it does not learn YX1/Keyence logic. |
| **`frame_inventory`** | RIGHT EDGE of the overlap | crossing it = you LEAVE the Microscope Zone. Pure Per-Well + agnostic from here = **"post-microscope land."** |

> **Beat 1 = build the STITCH OVERLAP and exit into `frame_inventory`.** Beat 2 = the pure Per-Well
> Zone past the overlap (segmentation/features/QC). "Why is stitch per-well but segmentation isn't?"
> → stitch is IN the overlap (per-well + scope-aware); segmentation is PAST it (per-well + agnostic).

### Validator Ladder — what each validator IS

Use validator names by **stage contract**, not a generic "the validator." The confusion is exactly
that some validators are scope-aware and some are shared:

| Stage / artifact | Validator role | Kind | What it proves |
|---|---|---|---|
| `map_positions_to_wells` → `position_well_mapping.csv` | physical mapping validator | scope-aware check over a shared mapping role | raw positions map to canonical wells without illegal ambiguity. |
| `scope_metadata_mapped.csv` | canonical metadata validator | shared-ish metadata contract | rows have valid global `well_id`s; `discover_wells` can trust the file. |
| `acquisition_inventory__{scope}.csv` | acquisition inventory validator | scope backend contract | raw acquisition evidence is well-formed for that microscope. |
| `resolve_acquisitions` outputs | acquisition resolution validator | scope backend resolver | each well has either one active source or an explicit quarantine reason. |
| `stitch_well` | producer/runtime checks | scope backend | the backend can materialize frames from resolved input. |
| `frame_inventory.csv` | frame inventory contract validator | shared, microscope-agnostic | manifest atoms/derived IDs/uniqueness/path tiers satisfy the shared handoff. |
| SAM2 ingest view | consumer layout validator | shared consumer-side view | frames can be presented as ordered `NNNN.ext` per well/channel. |

Rule:

```text
Scope-aware validators exist BEFORE the handoff.
The frame_inventory validator IS the handoff and stays microscope-agnostic.
```

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
> is answered by the **per-well frame_inventory validator** (Step 6), not a separate eligibility
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
  Level 3   — SAM2 ingestion layout (frames for one well/channel can be presented as one ordered
              folder, including NNNN.ext symlink/view generation)                 ← NEXT PHASE.
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

**Naming rule:** if a file owns a data contract, its filename should say `*_contract.py`; if a file
only validates, its filename should say `validate_*` or `*_validators.py`. Avoid vague names like
`contracts.py` and avoid generic helpers that look like first-class pipeline contracts.

```
shared/table_validators.py
  assert_columns_present
  assert_unique_on_key
  assert_positive_numeric
  assert_allowed_values

metadata_ingest/well_discovery/discovered_wells_contract.py
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

**Rule:** contract modules own meaning; `shared/table_validators.py` owns mechanics. The old
`data_pipeline/schemas/` package remains as a legacy compatibility layer until importers are
migrated. Do not widen it with new target semantics.

> **⚠️ Don't build contracts you don't need yet (anti-whale guard).** This section shows the full
> domain-contract END STATE — it is NOT a prerequisite checklist. Build each contract module only when
> a step actually consumes it (follow `schema_layout.md`'s migration order). For the YX1 path that
> means **exactly two** contracts, built when their step needs them:
> - `well_discovery/discovered_wells_contract.py` + maybe `shared/table_validators.py` — **Step 1**
>   (discovery consumes them; add the shared helper only if local validation code would duplicate).
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

> **Status legend:** ✅ built+wired · 🔨 NEW (this plan) · 🔧 modified · ⏸ deferred (Keyence/Beat 2) · ⏳ move LAST
>
> **Role legend:** **SHARED MECHANICS** = reusable checks/helpers, no domain meaning ·
> **SHARED CONTRACT** = domain contract consumed across scopes · **SOURCE DISPATCHER** =
> chooses by source contract, not microscope · **SCOPE BACKEND** = YX1/Keyence implementation ·
> **ORCHESTRATION** = graph/path/run-set logic · **LEGACY COMPAT** = migration shim only.

```
data_pipeline/

  shared/                                    # KINGDOM: identity + generic mechanics (no domain logic)
    identifiers/                             # ✅ BUILT + wired (Scope 1) — SHARED CONTRACT for ID grammar
      constructors.py                        #   build_well_id / build_image_id (_t{time_index:04d})
      parsers.py                             #   split_well_id / parse_image_id
      validators.py                          #   SHARED VALIDATOR: validate_well_id / recompute_and_check
    table_validators.py                      # 🔨 MAYBE (only if duplication appears) — SHARED VALIDATORS:
                                             #   assert_columns_present/unique_on_key/positive_numeric/allowed_values.
                                             #   No TableContract class unless a caller actually needs it.
    path_value_validators.py                 # ⏸ MAYBE ONLY: future explicit path-value helpers if needed.
                                             #   Must take configured roots as parameters; no hidden defaults.
                                             #   Do not create/move for Beat 1; NOT an artifact registry.

  metadata_ingest/                           # UPSTREAM of stitch — scope-specific
    plate/
      plate_processing.py

    scope/
      yx1/
        acquisition_inventory.py             # ✅ BUILT — SCOPE BACKEND CONTRACT+VALIDATOR:
                                             #   owns YX1 schema + tensor cell key
        extract_yx1_scope_metadata.py        # ✅ BUILT — SCOPE BACKEND: the ONE raw ND2 read
        map_yx1_positions_to_wells.py        # ✅ BUILT — SCOPE BACKEND: XY→well mapping
        generate_xy_reference.py             # ✅ exists (offline tool; not a DAG node)
        validate_xy_reference_grid.py        # ✅ exists — SCOPE BACKEND VALIDATOR for YX1 reference grid
      keyence/                               # ⏸ SEPARATE TRACK (not YX1 Beat 1)
        acquisition_inventory.py             # ⏸ DEFERRED — SCOPE BACKEND CONTRACT+VALIDATOR:
                                             #   Keyence raw-unit schema + collision key
        extract_keyence_scope_metadata.py    # SCOPE BACKEND: raw TIFF/XML read
        map_keyence_positions_to_wells.py    # SCOPE BACKEND: folder/layout→well mapping
        resolve_keyence_acquisitions.py      # ⏸ DEFERRED — SCOPE BACKEND RESOLVER:
                                             #   collision classification + eligibility producer
      shared/
        acquisition_checks.py                # ✅ BUILT — SHARED MECHANICS:
                                             #   scope-agnostic check primitives; scopes declare keys
        apply_position_mapping.py            # ✅ exists — SHARED JOIN: convergence line, mints well_id
        validate_physical_well_mapping.py    # ✅ exists — SHARED MECHANICS for mapping cardinality;
                                             #   YX1 requires 1 position↔1 well; Keyence differs pre-resolve

    well_discovery/                          # 🔨 NEW (Step 1) — SOURCE DISPATCHER, not microscope-dispatched
      __init__.py
      discovered_wells_contract.py           #   SHARED CONTRACT+VALIDATOR: discovered_wells.txt
      from_scope_metadata.py                 #   discover_wells_from_scope_metadata(mapped_csv → wells)
      from_frame_inventory.py                #   ⏸ DEFERRED (drop-in twin; external path)
      discover_wells.py                      #   dispatcher by SOURCE, not microscope

    contracts/                               # ⏸ DEFERRED home (eligibility — Keyence-weighted)
      well_acquisition_summary.py            #   ⏸ SHARED CONTRACT: well_id + active_for_stitch +
                                             #   quarantine_reason; producer is scope backend

  image_materialization/                     # DOWNSTREAM of stitch — frame_inventory lives HERE
    stitched/
      materialize_stitched_images.py         # 🔧 Step 6 (promote) — MICROSCOPE DISPATCHER:
                                             #   chooses backend by microscope
                                             #   today: 1 mixed 43-branch file in metadata_ingest/stitched_index/ (legacy)
      layout.py                              # 🔨 NEW (Step 3) — SHARED CONTRACT helper for native TREE:
                                             #   native pixel paths; ONE FILE (→ subpackage if it earns it); imports identifiers
      frame_inventory.py                     # 🔧 Step 6 (promote) — SHARED IMPLEMENTATION:
                                             #   build/validate shared manifest rows; imports contract below.
                                             #   Native builders feed it; validator stays microscope-agnostic.
      contracts/
        frame_inventory_contract.py          # 🔨 NEW (Step 2) — SHARED CONTRACT (PURL): identity ATOMS,
                                             #   DERIVED ids, REQUIRED columns, UNIQUE_KEY (atoms; time_index)
      scope/
        yx1/
          materialize_yx1_stitched_images.py # 🔨 Step 3 — SCOPE BACKEND:
                                             #   ND2 tensor slice + LoG focus; emits native rows/images
        keyence/
          materialize_keyence_stitched_images.py  # ⏸ SCOPE BACKEND:
                                             #   stub now; full version consumes resolved Keyence inventory

    shared/                                  # ⏳ MOVE LAST (only after stitched pkg is stable)
      log_focus.py                           #   ⏳ keep in image_building/shared/ until then
      frame_tiler.py                         #   ⏳ LIVE engine — do NOT move yet
      image_io.py

  schemas/                                   # ⚠️ LEGACY COMPAT ONLY during migration
    frame_contract.py                        #   LEGACY COMPAT; do NOT add target semantics; retire gradually
    stitched_image_index.py                  #   LEGACY COMPAT; ⏸ RETIRE (Step 7)

  pipeline_orchestrator/
    tasks.py                                 # thin: parse + delegate only
    orchestration/
      paths.py                               # ✅ ORCHESTRATION: PIPELINE_STEPS registry — tabular artifact paths.
                                             #   This is the artifact path-of-record. The old
                                             #   shared/path_contracts.py is now a deprecation tripwire.
      well_runner.py                         # ✅ ORCHESTRATION: run set + shard fan/merge helpers;
                                             #   reads shared summaries, never scope-only columns.
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

**🔨 NEW rows — `stitch_well_candidate` (Step 3, TEMPORARY) → `stitch_well` (Step 6, promoted):**
The strangler needs the candidate to write **isolated** paths so it can NEVER collide with live
output. So register a **temporary candidate row first**, promote to the real row at Step 6, drop the
candidate row at Step 7.
```python
# Step 3 — TEMPORARY candidate row (isolated `candidate/` paths; deleted at Step 7 strangle):
"stitch_well_candidate": {
    "stage": "built_image_data",
    "fanout": PER_WELL,
    "artifacts": {
        # ISOLATED under candidate/ — cannot collide with the live materialize_stitched_images output.
        "done": "candidate/{well_id}/.well_{well_id}.candidate.done",
    },
},

# Step 6 — PROMOTED live row (added when the candidate is accepted; candidate row then retired):
"stitch_well": {
    "stage": "built_image_data",            # the pixel store root
    "fanout": PER_WELL,                      # per-well (replaces the experiment-grain legacy rule)
    "artifacts": {
        "done": ".well_{well_id}.done",      # per-well sentinel (replaces the experiment .done)
    },
},
```
> Note: the image files themselves stay **off-registry** — `layout.py::stitched_frame_path(...)` owns
> them (image trees are not tabular artifacts). Only the sentinel is a registry row. The candidate's
> `candidate/` prefix is the structural guarantee that legacy stays green (Step 4's "isolated paths").

**🔧 CHANGED row — `frame_inventory` (de-stale; Step 6):**
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
    # CHANGE: producer repoints from frame_contract.csv → per-well stitch output (Step 6);
    #         product keys on time_index, not time_int (Step 6/7, gradual).
},
```

**⏸ RETIRE from the path-of-record (Step 7, gradual):**
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
> sugar) and the native enforcement ("mismatch = FAIL") actually get written — at promote (Step 6) or
> later — that is when `layout.py` → `layout/` (`constructors.py` · `parse.py` · `contract.py`) earns
> the split. Not
> before. Off-registry either way: `layout` owns IMAGE-TREE paths (off the tabular registry);
> `pipeline_orchestrator/orchestration/paths.py` owns TABULAR artifact paths. Different families, same ID grammar (both import
> `shared/identifiers`, neither inline-mints — two-kingdoms holds).

---

## well_discovery/ Organization

> **🎤 DECISION GATE (mdcolon to be interviewed at Step 1).** The layout below is a PROPOSAL, not
> locked. Before building `well_discovery/`, walk through with mdcolon: is the source-contract
> dispatcher (`from_scope_metadata` / `from_frame_inventory`) the right shape, or overkill for YX1
> now? How thin is `discovered_wells_contract.py`? Does `discover_wells.py` dispatch belong here or in `tasks.py`?
> **Do not build to this structure without that conversation.**

`well_discovery` is **shared and source-contract-specific, not microscope-specific.**

```
well_discovery/
  discovered_wells_contract.py
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
- Domain-level contract modules own schema meaning; `shared/table_validators.py` owns reusable
  validation mechanics. `data_pipeline/schemas/` is legacy compatibility during migration.
- `tasks.py` only parses and delegates.
- sidecars are derived via path helpers, not registered as first-class artifacts.
- ONE manifest contract; shared validator enforces manifest invariants, while the native builder
  enforces canonical layout by construction through `layout.py`.

---

## 🛠️ YX1 IMPLEMENTATION PLAN — from where we actually are (verified on disk 2026-06-16)

> The phases above are the full front-half target. This section is the **concrete next YX1 work**,
> grounded in what is actually on disk today — not the doc's aspirations.

> **🌿 PREFERRED MIGRATION STRATEGY (decided mdcolon 2026-06-17): STRANGLER branch + one-well vertical
> slice.** The legacy spine runs to `stitch → stitched_image_index → frame_contract` TODAY. **Do NOT
> re-grain the live stitch rule in place** — that breaks the working path the moment you start. Instead
> grow the new per-well path BESIDE the legacy one, prove it on ONE well, then cut over and strangle
> the old path. The legacy pipeline stays runnable the entire time.
>
> ```
> legacy: materialize_stitched_images[{exp}] → stitched_image_index → frame_contract   (stays GREEN, untouched)
> new:    stitch_well_candidate[well_id] → candidate frame_inventory shard             (grows beside it)
>            → COMPARISON GATE (one well) → fan → PROMOTE → strangle legacy
> ```
>
> **🚦 THE COMPARISON GATE IS THREE-LAYERED — human eyes are the final acceptance.** Per-well stitch
> may NOT be byte-identical to batched legacy stitch (focus/LoG projection or normalization can depend
> on batch composition). So byte-identity is the *ideal*, not the *gate*:
> 1. **byte/hash compare** — pass = ideal, done.
> 2. **numeric image-diff** (on mismatch) — `max/mean/p99 abs-diff`, optional SSIM, per frame.
> 3. **human visual QC** (REQUIRED for promotion) — side-by-side `legacy | candidate | abs-diff heatmap`
>    frames + an `.mp4`; **mdcolon confirms by eye before promotion.**
>
> > **Byte-identical pass is ideal. Byte mismatch is NOT automatic failure — it triggers numeric diff
> > + visual review. Final acceptance requires human visual confirmation (mdcolon's eyes).**
>
> **New QC artifact (`stitch_candidate_qc/`) — a DEDICATED stitch-comparison helper, NOT the
> segmentation video renderers** (`render_raw_video.py`/`render_overlays.py` exist but are DOWNSTREAM
> of frame_inventory — wrong layer for raw-stitch comparison):
> ```
> stitch_candidate_qc/{experiment}/{well_id}/
>     comparison_summary.csv        # image_id, legacy_path, candidate_path, same_bytes,
>                                   # max_abs_diff, mean_abs_diff, p99_abs_diff, ssim?, human_review_status
>     frame_diff_metrics.csv
>     side_by_side_frames/{image_id}_compare.jpg     # legacy | candidate | abs-diff heatmap
>     side_by_side.mp4
> ```
>
> **🚧 BINDING COMMIT BOUNDARIES (not optional).** Separate commits, each verified before the next;
> the candidate branch writes ISOLATED paths so the two stitch paths never collide. The candidate
> COMPUTE scaffolding (`stitch_well_candidate`, candidate image paths) is **throwaway — deleted at the
> strangle step**; the `stitch_candidate_qc` comparison evidence is **archived, not deleted** (Step 7).
> Do NOT migrate segmentation/features/QC in Beat 1 (Beat 2).

### Reality baseline (verified)

| Thing | State |
|---|---|
| Scope 1 `shared/identifiers/` | ✅ **built + wired** (`constructors`/`parsers`/`validators`/README; 12 importers, incl. stitch) — **Phase 0 is DONE** |
| YX1 metadata (Phase 1) | ✅ shipped — `extract_yx1_scope_metadata` · `map_yx1_positions_to_wells` · join · discover all run |
| YX1 acquisition inventory (Phase 1C) | ✅ shipped — record-only, 161k-row smoke verified |
| `metadata_ingest/well_discovery/` | ❌ **does not exist** — discovery still inline in `tasks.py` |
| Stitch consumes the inventory | ❌ **no** — still inline `_select_yx1_channel_index` (:121), `yx1_series_map` (:445,:501), `drop_duplicates(keep="first")` (:431) |
| `image_materialization/` | ❌ **does not exist** — stitch lives in `metadata_ingest/stitched_index/materialize_stitched_images.py` (43 microscope branches) |

**Net:** identifiers is already off the list. The real next work is a **7-step STRANGLER migration**
(legacy stays green throughout): extract discovery → domain contracts → `stitch_well_candidate`
beside legacy → comparison gate on one well (human visual sign-off) → fan candidate → promote to live
spine (the finish line) → strangle legacy.

**How to read the steps below:** each step is a FLOW card, not just a task list:

```text
ZONE       where this step lives in the overlapping-zone model
FLOW       which arrow in the front-half river changes
FILES      concrete source files created/edited/deferred
VERIFY     the gate before the next commit
```

This is intentional. If a step cannot name its zone and files, it is too vague to implement safely.

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
   the fan point earlier** (Steps 3+6 via the candidate).
3. **The frame_inventory branch is DEAD** — `rule all` stops at features; `merge_frame_inventory` is
   never requested; **segmentation still reads `frame_contract.csv` directly.** Reaching the finish
   line = **making the per-well frame_inventory the live spine and retiring `frame_contract`** (Steps 6–7).
4. **The per-well rules ALREADY EXIST** (`build_/validate_/merge_frame_inventory`). Step 6 is
   **wiring + collapsing duplicates, not building** — point the live path at the shards.

**The TARGET DAG after promote+strangle (Steps 6–7) — what we collapse TO:**

```
discover_wells (checkpoint = THE FAN)
        ↓  ⟱ per-well ⟱
stitch_well[well_id]                  → per-well stitched images (via layout.py)   🔴 candidate→promoted (Steps 3,6)
        ↓
build_frame_inventory_for_well[well_id]  → {well_id}_frame_inventory.csv  (NATIVE per-well producer; keys on time_index)
        ↓
validate_frame_inventory_for_well[well_id]  (absorbs the old stitched-index file-existence check)
        ↓
merge_frame_inventory[{exp}]          → 🏁 the agnostic handoff (LIVE) ── Beat 1 STOPS here ──┐
                                                                                              │
        (downstream readers — segmentation/features/QC — migrate onto the shard in BEAT 2) ◄─┘
```

> **GONE on the YX1 path after Steps 6–7:** `stitched_image_index.csv` + `validate_stitched_image_index`
> (intermediate, absorbed). The product speaks **`time_index`** natively. `frame_contract.csv` is
> collapsed for the YX1 PRODUCER, but **downstream `frame_contract` readers (segmentation, features)
> migrate GRADUALLY** — a `time_int` compat alias survives during transition; no big-bang rename.
> **The YX1 PRODUCER path moves to frame_inventory as the canonical handoff; downstream consumers
> migrate gradually (Beat 2).**

### Step 1 — Extract `well_discovery/` from `tasks.py`  ·  🧩 OVERLAP left edge (the fan/bootstrap)  *(low risk, NO behavior change)*

**ZONE:** left edge of the STITCH OVERLAP. This is where experiment-grain canonical metadata enters
the Per-Well Zone. The implementation is microscope-agnostic.

**FLOW:** `scope_metadata_mapped.csv → discovered_wells.txt`.

Pure structure; touches no images; the one Beat-1 graph piece genuinely missing.

**FILES:**
- **Create**
  - `src/data_pipeline/metadata_ingest/well_discovery/__init__.py`
  - `src/data_pipeline/metadata_ingest/well_discovery/discovered_wells_contract.py`
  - `src/data_pipeline/metadata_ingest/well_discovery/from_scope_metadata.py`
  - `src/data_pipeline/metadata_ingest/well_discovery/discover_wells.py`
- **Edit**
  - `src/data_pipeline/pipeline_orchestrator/tasks.py`
  - `src/data_pipeline/pipeline_orchestrator/Snakefile` only if the task invocation changes
- **Do not create yet**
  - `from_frame_inventory.py` (external/drop-in twin; not needed for YX1 Beat 1)

- Move the `discover-wells` logic out of `tasks.py` → `discover_wells_from_scope_metadata(mapped_csv,
  output_wells)`; `tasks.py` just delegates (no pandas). Dispatcher keys on **source**
  (`scope_metadata`), not microscope.
- **Verify:** `snakemake -n` parses; 20250912 produces identical `discovered_wells.txt`; tests (dup
  collapse, missing `well_id` fails, local `A01` fails, `tasks.py` has no business logic).

### Step 2 — Establish front-half domain contracts  ·  🧩 the shared contract the overlap exits on  *(low risk, NO behavior change)*

**ZONE:** contract rail through the STITCH OVERLAP. This step defines the shared handoff language
before the stitch producer starts emitting it.

**FLOW:** no artifact edge changes yet; this creates the contracts that Step 3 and Step 6 import.

> **🎤 DECISION GATE (mdcolon to be interviewed before building).** The **acquisition-inventory
> contract** shape (how the already-shipped `scope/yx1/acquisition_inventory.py` schema/key relates to
> the new `frame_inventory_contract.py`, and whether they share anything) is NOT locked — walk through
> it with mdcolon first. The frame_inventory contract columns/key below are a proposal to confirm, not
> a prescription.

Add only the **two** contracts the YX1 path consumes (per the anti-whale guard) before touching frame
inventory or moving packages.

**FILES:**
- **Create**
  - `src/data_pipeline/image_materialization/__init__.py`
  - `src/data_pipeline/image_materialization/stitched/__init__.py`
  - `src/data_pipeline/image_materialization/stitched/contracts/__init__.py`
  - `src/data_pipeline/image_materialization/stitched/contracts/frame_inventory_contract.py`
- **Maybe create only if duplicated locally**
  - `src/data_pipeline/shared/table_validators.py`
- **Do not create yet**
  - `src/data_pipeline/metadata_ingest/contracts/well_acquisition_summary.py`
- **Leave as legacy compat**
  - `src/data_pipeline/schemas/frame_contract.py`

- Add `shared/table_validators.py` for generic mechanics only if Step 1 would otherwise duplicate
  local validation code.
- Add `image_materialization/stitched/contracts/frame_inventory_contract.py` as the atom-based PURL
  for the stitched handoff (`experiment_id, well_index, channel_id, time_index` are the key;
  `well_id` and `image_id` are derived/checkable). The Step 3 stitch-consume cutover and the Step 6
  native producer both key on these atoms, so it earns its place here, before stitch is touched.
- **DO NOT add** `metadata_ingest/contracts/well_acquisition_summary.py` — eligibility is deferred
  (Keyence-weighted; see "NOT in this plan").
- Leave `schemas/frame_contract.py` as legacy compatibility; do not add new target semantics there.
- **Verify:** import-only/unit tests for required columns, unique keys, and derived-id rules. No
  Snakemake behavior changes yet.

### Step 3 — Add `stitch_well_candidate[well_id]` BESIDE legacy  ·  🧩 OVERLAP body (per-well + scope-aware stitch)

**ZONE:** body of the STITCH OVERLAP. This is the first place that is maximally both: per-well
execution plus YX1-specific production code.

**FLOW:** `discovered_wells.txt + acquisition_inventory__yx1.csv → stitch_well_candidate[well_id]`
beside the legacy `materialize_stitched_images[{exp}]` branch.

The strangler core: build the new per-well stitch as a **candidate branch**, legacy untouched.

**FILES:**
- **Create**
  - `src/data_pipeline/image_materialization/stitched/layout.py`
  - `src/data_pipeline/image_materialization/stitched/scope/__init__.py`
  - `src/data_pipeline/image_materialization/stitched/scope/yx1/__init__.py`
  - `src/data_pipeline/image_materialization/stitched/scope/yx1/materialize_yx1_stitched_images.py`
- **Edit**
  - `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py`
    (`stitch_well_candidate` row / isolated candidate sentinel)
  - `src/data_pipeline/pipeline_orchestrator/Snakefile` or
    `src/data_pipeline/pipeline_orchestrator/rules/frame_contracts.smk`
    (new candidate rule; legacy rule stays untouched)
  - `src/data_pipeline/pipeline_orchestrator/tasks.py`
    (candidate task entrypoint if needed)
- **Do not edit behavior in**
  - `src/data_pipeline/metadata_ingest/stitched_index/materialize_stitched_images.py`
    (legacy remains the baseline)

- Create `image_materialization/stitched/scope/yx1/materialize_yx1_stitched_images.py` as a **per-well**
  producer from the start (no in-place re-grain of the live rule). It **consumes
  `acquisition_inventory__yx1.csv`** for its `(well → position_index, channel_index, z)` lookup —
  killing the legacy inline `_select_yx1_channel_index` (:121), `yx1_series_map` (:445,:501), and the
  `drop_duplicates(keep="first")` (:431) re-derivations in one go (they live only in the legacy file,
  which stays as-is).
- Add `rule stitch_well_candidate[well_id]` writing to **ISOLATED candidate paths** (e.g.
  `built_image_data/{exp}/candidate/...` + `.well_{well_id}.candidate.done`) so it can NEVER collide
  with the live `materialize_stitched_images` output. Register via `paths.py` (no inline strings).
- Run it for **B01 only** first (legacy B01 stitched frames already exist on disk → a baseline to
  compare against). GPU step.
- **Verify (this commit):** `stitch_well_candidate[20250912_B01]` runs and writes isolated output.
  No comparison yet — just that the candidate produces frames.

### Step 4 — The COMPARISON GATE on one well (`stitch_candidate_qc/`)  ·  🧩 OVERLAP body (prove the scope backend)

**ZONE:** body of the STITCH OVERLAP. This step proves the new scope backend against the legacy
producer before anything is promoted.

**FLOW:** `legacy stitched B01 + candidate stitched B01 → stitch_candidate_qc evidence`.

Build the dedicated stitch-comparison helper (NOT the segmentation video renderers — wrong layer).

**FILES:**
- **Create**
  - `src/data_pipeline/image_materialization/stitched/stitch_candidate_qc.py`
- **Edit**
  - `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py`
    (QC evidence path helpers or registry row, if kept under the pipeline root)
  - `src/data_pipeline/pipeline_orchestrator/Snakefile` or rules include
    (optional QC rule/target)
- **Output evidence**
  - `stitch_candidate_qc/{experiment}/{well_id}/comparison_summary.csv`
  - `stitch_candidate_qc/{experiment}/{well_id}/frame_diff_metrics.csv`
  - `stitch_candidate_qc/{experiment}/{well_id}/side_by_side_frames/{image_id}_compare.jpg`
  - `stitch_candidate_qc/{experiment}/{well_id}/side_by_side.mp4`

- New `stitch_candidate_qc/{exp}/{well_id}/` artifact: `comparison_summary.csv` (`image_id,
  legacy_path, candidate_path, same_bytes, max_abs_diff, mean_abs_diff, p99_abs_diff, ssim?,
  human_review_status`), `frame_diff_metrics.csv`, `side_by_side_frames/{image_id}_compare.jpg`
  (`legacy | candidate | abs-diff heatmap`), and `side_by_side.mp4`.
- Run the three-layer gate on **B01**: (1) byte/hash; (2) on mismatch, numeric diff; (3) generate the
  side-by-side frames + video.
- **Verify (THE human gate):** **mdcolon reviews the side-by-side by eye and sets `human_review_status`.**
  Byte-identical = ideal/auto-pass. Byte mismatch ≠ failure — it's `np.allclose`-within-tolerance +
  visual confirmation that legacy and candidate are *the same image*. **No promotion without mdcolon's
  visual sign-off.**

### Step 5 — Fan the candidate over discovered wells  ·  🧩 OVERLAP body (per-well fan of the scope backend)

**ZONE:** body of the STITCH OVERLAP. Same YX1 backend, now fanned through the real well set instead
of a single B01 slice.

**FLOW:** `discover_wells checkpoint → run_well_ids_for_experiment → stitch_well_candidate[well_id]`.

**FILES:**
- **Edit**
  - `src/data_pipeline/pipeline_orchestrator/Snakefile` or rules include
    (candidate expands over the checkpoint instead of B01)
  - `src/data_pipeline/pipeline_orchestrator/orchestration/well_runner.py` only if existing
    `run_well_ids_for_experiment` cannot express `discovered ∩ target`
  - `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` only if candidate fanout paths
    need adjustment

- Once B01 is visually accepted, fan `stitch_well_candidate[well_id]` over the `discover_wells`
  checkpoint set (`run_well_ids_for_experiment` = `discovered ∩ target`) — drop the B01 hardcode.
- Run the candidate over all `20250912` wells; spot-check a few more wells through `stitch_candidate_qc`
  (don't need all by eye — B01 proved the method; sample the rest).
- **Verify:** the candidate fans per-well; QC summaries are green/accepted for the sampled wells.

### 🏁 Step 6 — PROMOTE the candidate to the live spine = REACH THE FINISH LINE  ·  🧩 EXIT the overlap → frame_inventory (post-microscope land)

**ZONE:** right edge of the STITCH OVERLAP. This is the microscope exit adapter: YX1-specific
stitch output becomes a shared, validated per-well `frame_inventory` shard.

**FLOW:** `stitch_well[well_id] → build_frame_inventory_for_well[well_id] →
validate_frame_inventory_for_well[well_id]`.

Now (and only now) cut over: the candidate becomes the real path; `build_frame_inventory_for_well`
reads it. This is the END of the microscope-aware pipeline — the top of the dam.

**FILES:**
- **Create / move into target home**
  - `src/data_pipeline/image_materialization/stitched/frame_inventory.py`
    (native per-well builder + Level 1/1.5 validator that imports `frame_inventory_contract.py`)
  - `src/data_pipeline/image_materialization/stitched/materialize_stitched_images.py`
    (thin dispatcher after promote)
- **Edit**
  - `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py`
    (`stitch_well` live row; `frame_inventory` path remains per-well then merge)
  - `src/data_pipeline/pipeline_orchestrator/rules/frame_inventory.smk`
    (builder reads native stitch output, not `frame_contract.csv`)
  - `src/data_pipeline/pipeline_orchestrator/Snakefile`
    (front-end target includes validated per-well frame inventory shards)
  - `src/data_pipeline/pipeline_orchestrator/tasks.py`
    (frame-inventory task imports new target home)
- **Leave downstream consumers alone**
  - segmentation/features/QC still migrate in Beat 2

- **Promote:** rename `stitch_well_candidate` → `stitch_well`; move candidate paths to the real
  `built_image_data` location (or repoint `paths.py` from `candidate/` to live).
- **Native per-well frame_inventory:** point `build_frame_inventory_for_well` at the per-well stitch
  output instead of selecting rows from an experiment-grain `frame_contract.csv` — the adapter becomes
  a native per-well producer; the product speaks **`time_index`** natively.
- **Make the branch LIVE up to the shard (NOT the consumer):** add the per-well
  `{well_id}_frame_inventory.csv.validated` sentinels to a front-end target so the branch runs (today
  it's dead — audit #1). **Beat 1 stops at the validated shard.** Repointing segmentation/features/QC
  is **Beat 2** (downstream consumers, far side of the handoff).
- **Validator = Level 1 + Level 1.5** (identity + path-existence; absorbs the retired stitched-index
  file-existence check). Level 2 (image-open/dims) deferred.
- **Verify (the done-for-the-microscope-aware-part check):** for `20250912`, the chain `ingest →
  map_positions → join → discover → stitch_well[well_id] → build/validate_frame_inventory_for_well`
  runs end-to-end and produces per-well frame inventories that (a) key on `time_index`, (b) have
  derived `well_id`/`image_id` recomputing from atoms, (c) have resolving `source_image_path`s.
  **YX1 has crossed the microscope boundary; Beat 1 is DONE.**

### Step 7 — STRANGLE the legacy path  ·  🧩 cleanup (remove the old Microscope-Zone chain)

**ZONE:** cleanup after exiting the STITCH OVERLAP. The live microscope-aware path is now the
per-well stitch → frame_inventory handoff; this deletes the old experiment-grain chain.

**FLOW:** remove `materialize_stitched_images[{exp}] → stitched_image_index → frame_contract` from
the YX1 producer path.

Only after Step 6 is green and stable:

**FILES:**
- **Delete / stop importing on the YX1 path**
  - `src/data_pipeline/metadata_ingest/stitched_index/materialize_stitched_images.py`
  - `src/data_pipeline/metadata_ingest/stitched_index/validate_stitched_image_index.py`
  - YX1 uses of `src/data_pipeline/metadata_ingest/microscope_data_ingest/frame_contract/`
- **Edit**
  - `src/data_pipeline/pipeline_orchestrator/Snakefile`
  - `src/data_pipeline/pipeline_orchestrator/rules/frame_contracts.smk`
  - `src/data_pipeline/pipeline_orchestrator/rules/frame_inventory.smk`
  - `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py`
  - `src/data_pipeline/pipeline_orchestrator/orchestration/well_runner.py`
- **Keep as legacy compatibility until downstream readers move**
  - `src/data_pipeline/schemas/frame_contract.py`
- **Archive, not delete**
  - `stitch_candidate_qc/` evidence

- **Delete the legacy stitch chain:** `materialize_stitched_images[{exp}]`, `stitched_image_index.csv`
  + `validate_stitched_image_index` (its file-existence check now lives in the frame_inventory gate),
  `build_frame_contract` + `validate_frame_contract` on the YX1 path, and their inline Snakefile path
  strings. `schemas/frame_contract.py` stays legacy-compat until no importer needs it.
- **Delete the candidate COMPUTE paths** (`stitch_well_candidate` rule, the `candidate/` image paths)
  — throwaway; its job (prove the new path) is done.
- **ARCHIVE, do NOT delete, the comparison evidence.** Move `stitch_candidate_qc/` (the
  `comparison_summary.csv`, `frame_diff_metrics.csv`, side-by-side frames + `.mp4`) to a dated
  `qc_archive/{date}_stitch_cutover/` folder. It is the **provenance for WHY the cutover was
  accepted** (which wells, what diff, mdcolon's visual sign-off) — keep it as migration evidence.
- **Collapse the audit's duplications** (cheap, ~40 lines): `merge_frame_inventory_shards` →
  `well_runner.concat_well_shards_to_file(...)` (audit #4); drop the merged-level double-validate
  (audit #3); keep a `time_int` compat alias for not-yet-migrated downstream readers (gradual).

> **🏁 When Steps 6+7 pass, the microscope-aware part of the pipeline is DONE for YX1.** That is the
> goal of this roadmap. (Keyence walks the same strangler steps; Beat 2 deepens per-well execution
> across the whole back half via `well_runner.py`.)

### NOT in this YX1 plan (explicit)
- **Eligibility / `resolve_acquisitions` as a SEPARATE artifact** — settled: eligibility IS the
  validator ("are there actually good wells?"), shared mechanics + scope-specific checks, living in
  the **per-well frame_inventory validator** (Step 6), NOT a standalone `well_acquisition_summary`.
  The roadmap's `∩ eligible` term stays **deferred until Keyence** forces collision-based quarantine.
- **Validator — THREE tiers, Step 6 ships the first two:**
  - **Level 1 — identity/uniqueness** (schema, nulls, unique key, derived-id recompute). Exists today.
  - **Level 1.5 — path EXISTENCE** (every `source_image_path` resolves). This is the one check
    absorbed from the retired `validate_stitched_image_index` (Step 7). Cheap (`Path.exists()`, no
    image open). **Step 6 ships Levels 1 + 1.5.**
  - **Level 2 — file CONTENT** (images OPEN, real dims == declared, BF contiguous, channels
    rectangular). Opens every image → slow. **Deferred to the next, shared phase — NOT Step 6.**
  > The line: Step 6's gate proves *the wells exist and the manifest is self-consistent and its paths
  > resolve* — it does NOT open images. That's the honest "is this well good?" gate for Beat 1.
- **Full `well_runner.py` per-well EXECUTION across the back half** — Beat 2 (Steps 3/5 make *stitch*
  per-well; Beat 2 makes *every* stage per-well via the runner).
- **Keyence** — separate track (collisions, `resolve_acquisitions`, heterogeneous tile/Z).

### The one-line sequence (STRANGLER + one-well slice)
> identifiers ✅ → **(1) extract `well_discovery`** (safe) → **(2) domain contracts** (as consumed) →
> **(3) `stitch_well_candidate` BESIDE legacy** (per-well + inventory-fed, isolated paths; run B01) →
> **(4) COMPARISON GATE on B01** (byte → numeric diff → side-by-side video; **mdcolon's eyes accept**) →
> **(5) fan candidate over discovered wells** → **🏁 (6) PROMOTE to live spine** (native per-well
> frame_inventory; `time_index`; validated shard = finish line) → **(7) STRANGLE legacy** (delete
> `materialize_stitched_images` / `stitched_image_index` / `frame_contract` + the candidate
> scaffolding; collapse the audit dups). Legacy stays GREEN through Step 5; **the human visual gate
> (Step 4) is load-bearing**; **the promoted validated shard (Step 6) is the finish line.** Downstream
> consumers migrate in Beat 2 — NOT here.
