# Front-End Naming & Frame Inventory Flow — Ingest Lineages + Well Discovery (🟢 TARGET)

**Status:** active spec for the **early phase** of the refactor (front of the pipeline,
through the fan point and its immediate post-fan tail). Being fleshed out with mdcolon
2026-06-03 onward. This is the **detailed reference** for the early phase; the findings doc
points here for front-end specifics.
**Companion to:** `../per_well_throughline_findings.md` (the north-star findings doc — defers
to THIS doc on front-end detail and the canonical-key decision) and
`../well_id_throughline_refactor_plan.md` (the formal scopes).
**Scope of THIS doc:** the two metadata **ingest lineages**, **well discovery**, the
**canonical well key** (`well_id`), the post-fan front-end tail (stitch + per-well frame inventory), and the `paths.py` registry rows for these stages. It is the input for building
`lib/paths.py`'s front end.

> **2026-06-17 vocabulary update:** the live contract uses acquisition-position vocabulary:
> `map_positions_to_wells` writes `position_well_mapping.csv`, and
> `apply_position_to_well_mapping` writes `scope_metadata_mapped.csv`. `series_number` is stale ND2
> vocabulary and is not a fallback contract. Older prose below may mention `series_*` while
> explaining the original design history; use the position names for new code and docs.

---

## 🔑 THE CENTRAL FACT — two independent roots, not one chain

The front of the pipeline is **two independent metadata lineages** that run in parallel and
**converge late** (at feature consolidation, well into Zone B/C). They are NOT a chain.

```
  Excel file ─► ingest_plate_metadata ─► plate_metadata.csv ───────────────────────┐
                (genotype/condition                                                  │ converge LATE
                 + plate geometry)                                                   │ (consolidate_features)
                                                                                     │
  raw scope ──► ingest_scope_metadata ─► scope_metadata__{scope}.csv                │
   file         [the ONLY raw read]              │                                   │
                                                 ▼                                   │
                map_positions_to_wells ─► position_well_mapping.csv  (CSV→CSV, per-scope) │
                                                 │                                   │
                          ══ CONVERGENCE LINE (microscope gone) ══                  │
                                                 ▼                                   │
                apply_position_to_well_mapping ─► scope_metadata_mapped.csv  │
                                                 │        (well_id minted here)      │
                                                 ▼                                   │
                discover_wells ─► discovered_wells.txt  (reads mapped; #13)          │
                                                 │                                   │
                                     ⟱ FAN POINT ⟱                                   ┘
```

**Why this matters (the load-bearing insight, mdcolon 2026-06-03):**
> **You do NOT need the Excel/plate metadata to discover wells.** The wells physically exist
> in the raw microscope file regardless of whether a human filled in the plate spreadsheet.
> Discovery is purely a function of the **scope/raw lineage**; the plate lineage is an
> *annotation* you join on afterward (it enriches already-discovered wells; it does not
> discover them). The scope lineage converges at `apply_position_to_well_mapping`; the
> plate lineage still converges with scope later at `consolidate_features`.

**Verified in the current Snakefile (2026-06-03):**
- `normalize_plate_metadata` output (`plate_metadata.csv`) is **not consumed by any front
  stage** — its next reader is `consolidate_features` (Snakefile:702), deep in Zone B/C.
- `map_positions_to_wells_yx1` **today** reads the **raw images dir directly** (Snakefile:177)
  and its docstring states outright: *"Does not depend on plate metadata."* (🟢 TARGET moves
  this to CSV→CSV — it reads stage XY from `scope_metadata` instead of re-opening the ND2; see
  "Why ONE raw read" below.) Either way the mapping comes from stage positions, not the Excel.

---

## 🏷️ NAMING (decided 2026-06-03)

### Verbs — two acts, two words
| Verb | Meaning | Used for |
|---|---|---|
| `ingest_*` | pull an **external source** into the pipeline's canonical form | both metadata lineages (Excel **and** raw file) |
| `discover_*` | find **what physically exists** | reserved for `discover_wells` only |

> **Why `ingest` for both (not `discover` for both):** neither metadata file is literally
> "discovered" — the Excel is *authored* by a human, the raw file is *read*. `ingest` is
> honest for both. That frees the word **`discover`** for the one place it's literally true:
> finding which wells physically exist. Two different acts deserve two different verbs.

### Nouns — `plate` vs `scope` (kept)
| Lineage | Stage name | Why this noun |
|---|---|---|
| Excel design file | `ingest_plate_metadata` | the file carries **genotype/condition AND plate geometry** (well naming, plate format, row/col structure). "Plate" is justified — the file genuinely *is about the plate*, in both the design and geometry senses. |
| raw microscope file | `ingest_scope_metadata` | physical **acquisition** facts the instrument recorded (calibration, timing, dims, channels). |

> **"Plate" ambiguity, resolved:** mdcolon flagged that the scope data is *also* "in a plate
> format," so `plate` could read as the physical object (which both lineages describe).
> Decision: **keep `plate`** anyway — because the Excel file's content (per mdcolon) is
> genotype/condition + plate geometry, "plate metadata" accurately names *that file's* role.
> The disambiguator is the verb+noun pair: `ingest_plate_metadata` = the authored plate
> sheet; `ingest_scope_metadata` = the instrument's acquisition record. We did **not** rename
> to `design`/`acquisition` (considered, rejected — `design` undersells the geometry half).

### Stage names adopted
| Today (current Snakefile) | TARGET stage name | Lineage | Microscope dispatch |
|---|---|---|---|
| `normalize_plate_metadata` | **`ingest_plate_metadata`** | plate (Excel) | n/a |
| `extract_scope_metadata_yx1` + `_keyence` | **`ingest_scope_metadata`** | scope (raw) | **one rule, config backend** (the only raw read) |
| `map_positions_to_wells_yx1` (+ keyence TODO) | **`map_positions_to_wells`** | scope | **per-scope logic, CSV→CSV** (always-present passthrough) |
| `apply_series_mapping_yx1` | **`apply_position_to_well_mapping`** | scope | shared (no dispatch) — applies the canonical position→well mapping |
| `discover_wells` (checkpoint) | `discover_wells` *(reads mapped — #13)* → `discovered_wells.txt` | scope branch | shared |

> Microscope (`yx1`/`keyence`) is a **code-dispatch** detail (chosen by `config["microscope"]`
> inside one rule — NOT a `_yx1`/`_keyence` rule suffix), not part of the stage name or the
> path. It applies to **three** per-scope stages: `ingest_scope_metadata` (the only raw read),
> `map_positions_to_wells` (per-scope CSV logic), and `stitch_well` after the fan. It converges out
> at `apply_position_to_well_mapping` and only ever appears as a filename token
> (`scope_metadata__{scope}.csv`) handled by `format_vars`. **See "What is split per-microscope"
> above for the full split, dispatch model, and the Keyence wiring gap.**

---

## 🌊 FLOW — the scope branch (ONE raw read, then CSV all the way down)

The scope lineage reads the raw metadata **exactly once** in Phase 1 (`ingest_scope_metadata`); everything
after is CSV→CSV. The **convergence line** (where the microscope stops mattering) sits right
after `map_positions_to_wells`.

```
raw file ─► ingest_scope_metadata ─► scope_metadata__{scope}.csv      [the ONLY raw read]
              [reads raw ONCE,         emits per-scope what it knows:
               per-scope logic]         · YX1: stage XY columns (well NOT yet resolved)
                                        · Keyence: well already resolved (folder-derived, inline)
                       │
                       ▼
            map_positions_to_wells ─► position_well_mapping.csv (+ .provenance.json)
              [CSV→CSV, per-scope        · YX1: XY nearest-neighbor match vs reference plate
               logic, ALWAYS runs]       · Keyence: passthrough (read already-resolved well)
                       │
  ═══════════════════ CONVERGENCE LINE (microscope gone after here) ═══════════════════
                       │
            apply_position_to_well_mapping ─► scope_metadata_mapped.csv (+ .validated)
              [SHARED, CSV join]          well_id MINTED HERE (exp + "_" + well_index)
                       │
                       ▼
            discover_wells (checkpoint) ─► discovered_wells.txt    ⟱ FAN ⟱
              [SHARED]                     (reads well_id column; one well_id per line)
```

**Why ONE raw read (verified 2026-06-03):** today `ingest_scope_metadata` and
`map_positions_to_wells` *both* open the raw microscope file independently — duplicated, expensive,
fragile. The well-determining info is already available to ingest (YX1: stage XY is in the same
ND2; **Keyence ingest already calls `_extract_well_from_path()`** — it literally parses the well
from folder names during extraction). So `ingest_scope_metadata` becomes the **sole** raw reader;
`map_positions_to_wells` works from its CSV. ("Raw read" = a stage that opens/parses the original
ND2/TIFF files — the heavy, format-specific operation. Reading a CSV we produced is not a raw
read.)

**Why `map_positions_to_wells` stays a stage (not dissolved):** it has a **constant interface**
(`scope_metadata → position_well_mapping`) and **always runs**, so the DAG is **scope-invariant** —
every scope produces `position_well_mapping.csv` at the same node, and everything downstream
(stitching, the join) wires identically regardless of microscope. What varies is only the
*logic inside*: YX1 does real XY matching; Keyence is a near-passthrough (the well was already
resolved at ingest, so this stage just emits it in the standard shape). Adding a new scope =
implement one `scope_metadata → mapping` function (trivial or complex), DAG unchanged. **Not
skippable** — an always-present passthrough beats a conditionally-absent node (downstream never
asks "did mapping happen?").

**The convergence line is the most important boundary in the front end.** Above it: per-scope
(`ingest_scope_metadata` reads raw; `map_positions_to_wells` does per-scope CSV logic). Below it:
fully shared. The rule: **read the raw file OR interpret scope-specific structure → microscope-
specific; operate only on canonical tables → shared.**

The **plate lineage does not appear in this diagram at all** — that's the whole point. It is a
separate root, joined only far downstream at `consolidate_features`.

---

## 📋 RULE-BY-RULE OUTLINE (the front-end, TARGET) — rough spec

The front end is **7 rules**: 1 plate root + 2 scope-front + 1 join + 1 fan + 2 post-fan.
"MS" = microscope-specific (config-dispatched backend). "Shared" = microscope-agnostic.

| Rule | Reads | Writes | MS? | Notes |
|---|---|---|---|---|
| `ingest_plate_metadata` | `{exp}_well_metadata.xlsx` | `plate_metadata.csv` | — | the OTHER root; genotype/condition + plate geometry. No front-end consumer. |
| `ingest_scope_metadata` | **raw ND2/TIFF (ONLY raw read)** | `scope_metadata__{scope}.csv` | **MS** | per-scope extract; emits stage XY (YX1) / resolved well (Keyence). |
| `map_positions_to_wells` | `scope_metadata__{scope}.csv` | `position_well_mapping.csv` (+ `.provenance.json`) | **MS** *(CSV→CSV)* | constant interface, always runs. YX1: XY match; Keyence: passthrough. |
| `apply_position_to_well_mapping` | `scope_metadata__{scope}.csv` + `position_well_mapping.csv` | `scope_metadata_mapped.csv` (+ `.validated`) | Shared | applies mapping; **`well_id` minted here**. ← microscope convergence line. |
| `discover_wells` *(checkpoint)* | `scope_metadata_mapped.csv` (#13) | `discovered_wells.txt` | Shared | reads the `well_id` column; emits ALL discovered well_ids. ⟱ FAN ⟱ |
| `stitch_well` *(per well_id)* | raw + this well's mapping rows + configured image-product set | materialized images + `.well_{well_id}.done` | **MS** | post-fan; off-registry image tree; one sentinel per well for the configured product set. Channel/method/z granularity lives as rows in `frame_inventory`, not Snakemake wildcards. |
| `validate_frame_inventory_well` *(per well_id)* | this well's `{well_id}_frame_inventory.csv` (built) | `…/per_well/{well_id}/{well_id}_frame_inventory.csv.validated` + report | Shared | post-fan per-well validation gate (metadata ∩ images). TARGET name (was `validate_frame_contract_well`); see `frame_inventory_handoff_contract.md`. |

**`well_id` is born at the join, read at discovery (matches code, line 68).**
`map_positions_to_wells` produces only the *mapping* (`position_index`, `raw_position_label` → `well_index`),
not a per-row table, so it can't mint `well_id`. `apply_position_to_well_mapping` is the first
point where every row reliably has both `experiment_id` and resolved `well_index` →
`well_id = build_well_id(experiment_id, well_index)`. `discover_wells` then just reads the existing column.
(See "The canonical well key" for why `well_id` is canonical everywhere after the fan.)

---

## 🔬 WHAT IS SPLIT PER-MICROSCOPE (verified from disk 2026-06-03)

The per-microscope split is **two stages** — but only **one of them reads the raw file** — and
then it **converges and never reappears.** The microscope matters for two reasons: reading the
raw format (`ingest_scope_metadata`) and interpreting the scope-specific position→well relationship
(`map_positions_to_wells`). Everything downstream of `apply_position_to_well_mapping` is
shared/agnostic.

```
yx1 raw ─►     ingest_scope_metadata [yx1]      ─┐  (ONLY raw read)
keyence ─►     ingest_scope_metadata [keyence]  ─┘
                          │
               map_positions_to_wells [per-scope, CSV→CSV]   yx1: XY match | keyence: passthrough
                          │
        ══ CONVERGENCE LINE ══►  apply_position_to_well_mapping ─► (everything shared)
        ▲ MICROSCOPE-SPECIFIC ZONE (2 stages, 1 raw read)   ▲ CONVERGENCE POINT
                                                              scope token dropped here;
                                                              well_id minted at the join
```

**Verified:**
- The join's code already lives in `scope/shared/apply_series_mapping.py` (current filename;
  TARGET renames the *stage* to `apply_position_to_well_mapping`) — the convergence is
  real, not aspirational.
- **Keyence ingest already resolves the well** — `extract_keyence_scope_metadata` calls
  `_extract_well_from_path()` (`XY01a`→`A01`) during extraction, so Keyence's `map_positions_to_wells`
  is a passthrough and needs no raw read.
- `build_frame_contract` is a **single** microscope-agnostic rule today (Snakefile:404). *(TARGET
  splits this into per-well `build_frame_inventory_well` + shared `validate_frame_inventory_well`,
  and renames the table `frame_contract` → `frame_inventory` — see `frame_inventory_handoff_contract.md`.
  Name kept here as-is because it describes today's code.)*
- Stitched-image output paths carry **no** scope token (findings doc, Microscope section).

### What's microscope-specific vs shared
| Stage | Microscope-specific? | Why |
|---|---|---|
| `ingest_scope_metadata` | **YES** — `{yx1, keyence}` | **reads the raw file format** (ND2 vs TIFF) — the only raw read |
| `map_positions_to_wells` | **YES** — `{yx1, keyence}` | per-scope position→well **logic** (YX1: XY match; Keyence: passthrough) — but **CSV→CSV, no raw** |
| `apply_position_to_well_mapping` | **NO** (shared) | pure table join; code in `scope/shared/` |
| `discover_wells` | **NO** (shared) | reads the converged `scope_metadata_mapped.csv` |
| `stitch_well` | **YES** — `{yx1, keyence}` | reads the raw file to build images (format-specific) |
| `validate_frame_inventory_well` | **NO** (shared) | reads converged metadata + stitched images (TARGET name; was `validate_frame_contract_well`) |

> So there are **two microscope-specific points**: the **metadata front** (`ingest_scope_metadata`
> + `map_positions_to_wells`, on the scope branch before convergence) and the **image build**
> (`stitch_well`, after the fan). The join is the metadata convergence line: microscope-specific
> metadata interpretation ends there. Image building has its own native-microscope segment after
> the fan (`stitch_well`) and converges again at the canonical stitched handoff tree. The common
> thread is scope-specific interpretation: ingest and stitch read raw data; map is scope-specific
> but CSV→CSV. The moment data is in canonical tables, the microscope is gone.

### TARGET dispatch model (DECISIONS, mdcolon 2026-06-03)
1. **Keyence is first-class — wire it fully.** Today Keyence is **half-wired**: it has
   `extract_scope_metadata_keyence`, but the Snakefile has **no** `map_positions_to_wells_keyence`
   and **no** `build_stitched_images_keyence` rule — even though the *code* exists
   (`mapping/series_well_mapper_keyence.py`, `image_building/keyence/stitched_ff_builder.py`).
   TARGET treats yx1 and keyence as **symmetric**: every microscope-specific stage dispatches
   both. **Wiring the missing Keyence rules is in scope.** *(⚠️ gap to close — see below.)*
2. **One stage, backend chosen by config — NOT scope-suffixed rules.** A microscope-specific
   stage is **one** stage key / `tasks.py` verb (`ingest_scope_metadata`, `map_positions_to_wells`,
   `stitch_well`); the backend (`yx1`/`keyence`) is selected **by config inside the one rule**,
   not by duplicating the rule with a `_yx1`/`_keyence` suffix. This is the "microscope =
   code-dispatch, not a path/DAG concern" principle made concrete. The scope only ever appears
   as a **filename token** (`scope_metadata__{scope}.csv` via `format_vars`) — never as a rule
   name, never as a path branch.
   ```
   rule ingest_scope_metadata:                 # ONE rule (not _yx1 / _keyence)
       params: backend = config["microscope"]  # yx1 | keyence
       # → dispatches to the right reader; writes scope_metadata__{scope}.csv
   ```
   This replaces today's two scope-suffixed rules. *(Refactor item: collapse
   `extract_scope_metadata_yx1` + `_keyence` into one `ingest_scope_metadata` with config
   dispatch; same for `map_positions_to_wells` and `stitch_well`.)*

> **⚠️ KEYENCE WIRING GAP (verified, must close for full symmetry):**
> | Stage | yx1 rule | keyence rule | keyence code exists? |
> |---|---|---|---|
> | `ingest_scope_metadata` | ✅ | ✅ | ✅ |
> | `map_positions_to_wells` | ✅ | ❌ **missing** | ✅ `series_well_mapper_keyence.py` |
> | `stitch_well` | ✅ | ❌ **missing** | ✅ `image_building/keyence/stitched_ff_builder.py` |
>
> The code is there; the **rules** aren't. Closing this gap is part of the "one stage, config
> dispatch" refactor — when each stage becomes one config-dispatched rule, Keyence is wired by
> construction (the rule calls the keyence backend when `config["microscope"]=="keyence"`).

---

## 🔑 THE CANONICAL WELL KEY — `well_id` everywhere (DECISION, mdcolon 2026-06-03)

**Decision: `well_id` (`build_well_id(exp, well_index)`, e.g. `20250912_B01`) is the canonical well key across
the ENTIRE DAG after the fan.** There is exactly **one** key downstream of discovery — no
local/global split. This **reverses** the findings-doc "image-tree = local" stance (which is
now superseded; see note below).

### Where each key lives
| Key | Form | Where it lives | Lifetime |
|---|---|---|---|
| `raw_position_label` | raw microscope token (`P03`, `XY01a`, etc.) | a **column** in the scope-metadata tables (`scope_metadata__{scope}.csv`, `position_well_mapping.csv`) — the microscope emits raw position labels; it has no concept of a local plate well yet | **born at raw ingest; resolved by mapping.** Survives only as a raw column, never as a path/wildcard/key downstream. |
| `well_index` | local plate well label (`B01`) | a **column** in the scope-metadata tables (`position_well_mapping.csv`, `scope_metadata_mapped.csv`) — mapping resolves the raw position label into a local well label | **born at mapping; promoted to `well_id` at the join.** Survives only as a column, never as a path/wildcard/key downstream. |
| `well_id` | global (`20250912_B01`) | the join output column, then **everything after the fan**: `discovered_wells.txt`, the per-well wildcard, the image dir, all sentinels, all shard paths, `target_wells` (normalized) | **canonical from the join onward.** |

### The promotion is FREE — no extra piping (the key realization)
`raw_position_label` is the raw token that map_positions_to_wells resolves into `well_index`. `well_id` is **not more information** than `well_index` — it is `well_index` plus a value the
pipeline *already has in hand*: the experiment. So the promotion is a pure local construction,
**one helper call, zero plumbing**:

```python
# inside apply_position_to_well_mapping — exp_id is on every row, well_index is resolved
well_id = build_well_id(exp_id, well_index)      # matches current code (apply_series_mapping.py:68)
```

No lookup table, no threading a value through earlier stages. **The join is the natural
promotion point** — it is the first stage where every row reliably has both `experiment_id` and
a resolved `well_index`, so `well_id` is minted there (as the code already does) and lands as a
column on `scope_metadata_mapped.csv`. `discover_wells` then just **reads that column** — no
construction at the fan. (`map_positions_to_wells` can't mint it: it produces only the
`position_index/raw_position_label → well_index` mapping, not a per-row table.)

```
raw file ─► emits RAW POSITION labels (raw_position_label)      instrument knows no local well yet
   │
   ▼
ingest + map ─► raw_position_label is resolved into well_index  ← local is CORRECT here
   │            (scope_metadata, position_well_mapping)
   ▼
apply_position_to_well_mapping                          ← microscope convergence
   │  PROMOTE: well_id = build_well_id(exp_id, well_index)
   │  → scope_metadata_mapped.csv carries well_id column
   ▼
discover_wells ─► reads well_id column → discovered_wells.txt
   │
   ▼
⟱ FAN ⟱ ─► everything speaks well_id ONLY                      ← wildcard, image dir, sentinels,
            (no local↔global conversion ever again)               shards, target_wells
```

### Why `well_id` everywhere (pros) — from the user's seat
- **One mental model.** No rule (and no user) ever asks "local or global here?" After the
  fan there is one key. Biggest usability win.
- **Unique tag, automatic.** `target_wells: [20250912_B01]` is unambiguous even across
  experiments — no "which `B01`?".
- **Copy-paste debugging.** A `well_id` from a log, a filename, a row, or a path is the *same
  string* everywhere — grep it across the whole tree, it resolves to one well.
- **Zero conversion seams downstream.** A split would force a local↔global conversion at
  every image boundary; each seam is a bug site. One key = no seams after the fan.
- **Matches the spine philosophy** — "one well flows through as an independent unit" reads
  best when that unit has one globally-unique name.

### Costs (and how real they are)
- **Image dir stops mirroring the microscope `B01/` layout** → becomes
  `stitched_ff_images/20250912_B01/`. *Real?* Weak — the dir is already under `{exp}/`, so
  the name is at worst mildly redundant, and a self-describing dir name arguably *helps* a
  human reading the tree.
- **Redundant date token** (`20250912/.../20250912_B01/`). *Real?* Cosmetic; self-documenting
  paths are worth a repeated token.
- **Image-building code change.** *Real?* **Small + localized — verified 2026-06-03.** The
  stitching helper (`yx1/stitched_ff_builder.py::compile_yx1_data`) treats the well label as
  an **opaque string**: it's only a dict key (`well_series_mapping`) and the output dir/filename
  component (`output_dir / well_name / "BF" / ...`). It never parses it. So the switch is a
  **pure substitution** — build the mapping as `{well_id: position_index}` in the task/backend that prepares stitch inputs
  (where `well_id = build_well_id(experiment_id, well_index)` is carried through or constructed
  from a table that already has `well_id`), and the dir/filename become `well_id` automatically.
  **Zero logic change in the helper; one helper call in the rule.**

> **Supersedes findings-doc "image-tree = local."** The findings doc (Lean MVP Contract,
> "`well` vs `well_id` in paths") currently says image trees use local `well` to mirror the
> microscope/plate layout, called "deliberate, not legacy." **That is now reversed:** image
> trees use `well_id` like everything else. Reason: the `{exp}/` parent already disambiguates
> the layout, and keying the addressable unit on a globally-unique id is the whole spirit of
> the per-well spine. Update that findings-doc section to point here.

---

## 🌊 POST-FAN FLOW — `well_id` everywhere (TARGET, the three-step Zone-A split)

After the fan, the front-end's tail follows the Zone-A Narrowing split (findings doc), all
keyed on `well_id`:

```
⟱ FAN: discovered_wells.txt (well_id) ⟱
   │
   ├─► stitch_well  [per well_id]                         (Zone B0, off-registry image tree)
   │     in:  this well's raw images + its scope_metadata_mapped rows
   │     out: built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/
   │          sentinel: .well_{well_id}.done              ← well_id, not well_index
   │
   ├─► build_frame_inventory_well   [per well_id]        (Zone-A spine, fanout=per_well_then_merge)
   │     in:  this well's images (from stitch_well) + this well's metadata rows
   │     out: <frame-inventory-family>/{exp}/per_well/{well_id}/{well_id}_frame_inventory.csv
   └─► validate_frame_inventory_well [per well_id]        (the shared gate)
         in:  {well_id}_frame_inventory.csv
         out: {well_id}_frame_inventory.csv.validated + {well_id}_handoff_report.md
         (per-well validation gate: metadata ∩ images on disk; sentinel = trusted)
```

> **The canonical stitched handoff tree is the public "drop-in here" point.** `stitch_well` +
> `build_frame_inventory_well` + the shared `validate_frame_inventory_well` are the boundary at
> which the shared pipeline begins. There are two entry modes into that seam: native microscope
> mode (`raw microscope data -> microscope-specific stitch runner -> canonical stitched handoff
> tree -> shared frame_inventory`) and external handoff mode (`already-canonical stitched handoff
> tree -> shared frame_inventory`). The full input contract (layout, required columns, immutable
> frame key, one-file+sentinel model, native+drop-in flows, worked walkthrough) is
> **`frame_inventory_handoff_contract.md`** — this section's post-fan counterpart. *(Names here are
> TARGET; the `paths.py` examples further below also use the TARGET `frame_inventory` name. Only
> explicit "today's code" callouts keep the legacy `frame_contract`. The code rename is a Scope-2
> migration.)*

- **`target_wells` is config-only** (no materialized `selected_wells.txt`): the checkpoint
  emits ALL discovered `well_id`s; the well-runner computes `active_wells = discovered ∩
  target_wells` from config (findings-doc Win 5). `materialize_selected_wells` dissolves.
- **`frame_inventory` is NOT a fan input** (TARGET): discovery reads only
  `scope_metadata_mapped.csv`. The frame inventory is built per-well *after* the fan
  (`build_frame_inventory_well`) and gated by `validate_frame_inventory_well`; its merged form is
  an off-spine view nothing downstream reads.
- **`stitched_inventory.csv` drops out of the spine** → optional off-spine report.

---

## 🧰 PATHS.PY REGISTRY ROWS (front-end stages) — 🟢 TARGET

Per the Lean MVP Contract (findings doc): per-stage `stage`, `fanout` ∈
{`experiment`, `per_well_then_merge`}, optional `subfolder`; per-artifact filename/template
shape; sentinels (`.validated`, `.provenance.json`) are **derived helpers**, not rows.

All pre-fan front-end stages are **experiment-grain** (`fanout=experiment`) — they run once per
experiment, before the fan. The immediate post-fan tail begins with `frame_inventory`, which is `per_well_then_merge`. They share the `experiment_metadata` stage (unchanged from today).

```python
STAGES = {
    # ── PLATE LINEAGE (Excel — authored design + plate geometry) ──────────────
    "ingest_plate_metadata": {
        "stage": "experiment_metadata",
        "fanout": "experiment",
        "artifacts": {"csv": "plate_metadata.csv"},
    },

    # ── SCOPE LINEAGE (raw microscope file — acquisition facts) ───────────────
    "ingest_scope_metadata": {
        "stage": "experiment_metadata",
        "fanout": "experiment",
        "artifacts": {
            "raw": "scope_metadata__{scope}.csv",   # {scope} → format_vars={"scope":"yx1"|"keyence"}
        },
    },
    "map_positions_to_wells": {
        "stage": "experiment_metadata",
        "fanout": "experiment",
        "artifacts": {
            "mapping": "position_well_mapping.csv",   # .provenance.json via provenance_path()
        },
    },
    "apply_position_to_well_mapping": {      # the join; well_id minted here (CONVERGENCE)
        "stage": "experiment_metadata",
        "fanout": "experiment",
        "artifacts": {
            "mapped": "scope_metadata_mapped.csv",  # .validated via validated_path()
        },
    },

    # ── FAN POINT (well discovery — checkpoint) ───────────────────────────────
    "discover_wells": {                              # emits discovered_wells.txt in well_id form
        "stage": "experiment_metadata",
        "fanout": "experiment",
        "artifacts": {"wells": "discovered_wells.txt"},
    },

    # ── POST-FAN (per well_id) — frame inventory joins the spine ──────────────
    # build_frame_inventory_well writes the shard; validate_frame_inventory_well writes
    # only the .validated sentinel + report (one-file+sentinel model — see frame_inventory_handoff_contract.md).
    "frame_inventory": {                             # stage key for the per-well shard path
        "stage": "experiment_metadata",            # <frame-inventory-stage> — see findings #5 (leaning frame_inventory/)
        "fanout": "per_well_then_merge",
        "artifacts": {
            "inventory": {
                "per_well": "{well_id}_frame_inventory.csv",  # well_id IN the filename
                "merged": "{experiment_id}_frame_inventory.csv",
            },
        },
    },
}
```

> **Stitching is OFF-registry** (image tree, not a tabular contract artifact) — it has its
> own path helper, not a `STAGES` row. Its TARGET paths still key on `well_id`:
> `built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/` and sentinel
> `built_image_data/{exp}/.well_{well_id}.done`.

### Resolved paths (all experiment-grain → `path_mode="experiment"`)
```python
artifact_path(ROOT, "ingest_plate_metadata", "csv", "20250912")
#   → {ROOT}/experiment_metadata/20250912/plate_metadata.csv

artifact_path(ROOT, "ingest_scope_metadata", "raw", "20250912", format_vars={"scope": "yx1"})
#   → {ROOT}/experiment_metadata/20250912/scope_metadata__yx1.csv

artifact_path(ROOT, "map_positions_to_wells", "mapping", "20250912")
#   → {ROOT}/experiment_metadata/20250912/position_well_mapping.csv
provenance_path(ROOT, "map_positions_to_wells", "mapping", "20250912")
#   → {ROOT}/experiment_metadata/20250912/position_well_mapping.csv.provenance.json

artifact_path(ROOT, "apply_position_to_well_mapping", "mapped", "20250912")
#   → {ROOT}/experiment_metadata/20250912/scope_metadata_mapped.csv
validated_path(ROOT, "apply_position_to_well_mapping", "mapped", "20250912")
#   → {ROOT}/experiment_metadata/20250912/scope_metadata_mapped.csv.validated

artifact_path(ROOT, "discover_wells", "wells", "20250912")
#   → {ROOT}/experiment_metadata/20250912/discovered_wells.txt   (contents: well_id per line)

# post-fan, per well_id:
artifact_path(ROOT, "frame_inventory", "inventory", "20250912",
              path_mode="per_well", well_id="20250912_B01")
#   → {ROOT}/experiment_metadata/20250912/per_well/20250912_B01/20250912_B01_frame_inventory.csv
validated_path(ROOT, "frame_inventory", "inventory", "20250912",
               path_mode="per_well", well_id="20250912_B01")
#   → {ROOT}/.../per_well/20250912_B01/20250912_B01_frame_inventory.csv.validated   (the gate's only csv-side output)
artifact_path(ROOT, "frame_inventory", "inventory", "20250912", path_mode="merged")
#   → {ROOT}/experiment_metadata/20250912/20250912_frame_inventory.csv   (off-spine merged view)

# stitching (OFF-registry, own helper) — keyed on well_id:
#   → {ROOT}/built_image_data/20250912/stitched_ff_images/20250912_B01/BF/
#   → {ROOT}/built_image_data/20250912/.well_20250912_B01.done
```

> ⚠️ **Sentinel-suffix audit still open.** Today `apply_series_mapping` writes
> `.scope_metadata_mapped.validated` (a **leading-dot** form, Snakefile:202), while the MVP
> contract assumes `{filename}.validated` (trailing). Resolve before wiring `validated_path`
> — either normalize on-disk to `{filename}.validated` or give `validated_path` a dotfile
> option. (Tracked in findings-doc AUDIT TODO.) `position_well_mapping.provenance.json` already
> matches the trailing-suffix convention.

---

## ✅ DECISIONS LOCKED IN THIS DOC
1. **Two independent roots**, not a chain; converge late at `consolidate_features`.
2. **Well discovery needs only the scope lineage** — plate/Excel is not a discovery input.
3. **`ingest_` for both metadata lineages**; **`discover_`** reserved for `discover_wells`.
4. **Keep `plate` and `scope` as the nouns** (Excel file carries genotype/condition + plate
   geometry, so "plate metadata" is accurate).
5. **`discover_wells` reads `scope_metadata_mapped.csv`** (#13) — full metadata co-located
   with the well set at discovery.
6. All **pre-fan** stages are **`fanout=experiment`**, stage **`experiment_metadata`**.
7. **`well_id` is the canonical key everywhere after the fan** (image dir, sentinels,
   wildcard, shards, `target_wells`). `raw_position_label` survives only as the raw microscope
   token in the scope tables; `well_index` is the resolved local plate well label column.
   Promotion (`well_id = build_well_id(exp, well_index)`) happens once, at
   `apply_position_to_well_mapping`; `discover_wells` only reads it. Image-building
   change verified small (opaque-string substitution in `stitched_ff_builder.py`).
   **Supersedes findings-doc "image-tree = local."**
8. **`target_wells` is config-only** (no `selected_wells.txt`); checkpoint emits all
   discovered `well_id`s, well-runner filters. `materialize_selected_wells` dissolves.
9. **ONE raw read.** `ingest_scope_metadata` is the **sole** stage that opens the raw
    microscope files in Phase 1; `map_positions_to_wells` becomes CSV→CSV. Per-microscope stages =
    three (`ingest_scope_metadata`, `map_positions_to_wells`, `stitch_well`) but only ingest +
    stitch touch raw. The microscope-specific scope branch converges at
    `apply_position_to_well_mapping`; the plate/scope roots still converge later at
    `consolidate_features`.
10. **`map_positions_to_wells` stays a stage** with a constant interface (`scope_metadata →
    position_well_mapping`), **always runs** (scope-invariant DAG). Per-scope logic inside:
    YX1 = XY match, Keyence = passthrough (well already resolved at ingest). Not skippable.
11. **`well_id` minted at the join** (`apply_position_to_well_mapping`, matches code),
    read at discovery. Join renamed to name the operation (join mapping onto scope metadata).
12. **`discovered_wells.txt`** (not `wells.txt`) — names which well-list it is (discovered,
    pre-filter), disambiguating from `active_wells`/`validated_wells`.
13. **One stage, config-dispatched backend** (`config["microscope"]`) — NOT `_yx1`/`_keyence`
    rule suffixes. Collapses today's two scope-suffixed rules into one each.
14. **Keyence is first-class** — TARGET wires it to full symmetry with yx1.

## 🪧 OPEN (carried, not decided here)
- `.validated` sentinel suffix convention (leading vs trailing dot) — audit + normalize.
- **`target_wells` input form:** accept both local+global and normalize to `well_id` (today's
  behavior), vs. require strict `well_id`. Leaning "accept both, normalize" for ergonomics —
  confirm when wiring the well-runner.
- **`<frame-inventory-stage>`** for `frame_inventory`: keep `experiment_metadata/` vs.
  dedicated `frame_inventory/` (findings #5, leaning the dedicated stage). Placeholder in the
  registry row above until decided.
- *(resolved)* `map_positions_to_wells` and `apply_position_to_well_mapping` are **kept
  as separate stages** — map is per-scope CSV logic (always-present passthrough), the join is
  the shared convergence point. Not folded.

## 🔧 REFACTOR ITEMS (migration touches, 2026-06-03)
- **One raw read:** `ingest_scope_metadata` must emit the well-determining fact per scope
  (YX1: stage XY columns — currently NOT in the scope_metadata schema, must be added; Keyence:
  the folder-derived well — already computed via `_extract_well_from_path`). Then rewire
  `map_positions_to_wells` from raw-reading to CSV→CSV.
- **Rename `apply_series_mapping` → `apply_position_to_well_mapping`** (stage/rule key;
  the code file `scope/shared/apply_series_mapping.py` can keep its name or follow).
- **Rename `wells.txt` → `discovered_wells.txt`** (checkpoint output, Snakefile:438;
  `wells_for_experiment()` reader; any `paths.py` row).
- **Close the Keyence wiring gap:** add the missing `map_positions_to_wells` + `stitch_well`
  Keyence paths (code exists, rules don't). Falls out for free once stages become
  config-dispatched single rules.
- **Collapse scope-suffixed rules** (`extract_scope_metadata_yx1`/`_keyence`, etc.) into one
  config-dispatched rule per microscope-specific stage.
- **`MICROSCOPE` config var is currently read but only printed** (Snakefile:62/68) — wire it
  into the actual backend dispatch.

## 🔗 FINDINGS-DOC UPDATE NEEDED
- The findings doc's "`well` vs `well_id` in paths" decision (image-tree = local) is
  **superseded** by Decision 7 here. Update that section to: image trees use `well_id` like
  everything else; `well_index` is a scope-table column only.
