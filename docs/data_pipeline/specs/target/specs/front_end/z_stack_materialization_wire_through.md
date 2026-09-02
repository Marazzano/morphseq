# Z-Stack Materialization Wire-Through (🟢 SHIPPED)

> **✅ SHIPPED 2026-06-25.** The identity/contract (§§A0–A1), resolver/path/executor (§§A2–A4),
> detection filter (§A4b), and the product-shard architecture (§§B1–B5) all landed (commits
> `0f5a613d`, `960a7667`, `c8bd4f2d`, `b4d99350`, `5246407b`, `6e879632`, `8e7a02f2`). The final
> reconciliation pass (2026-06-25) flipped the materialized-image grammar to **channel-first** (this
> doc was the source of truth; the code was product-first) and added the **focus_index_map
> construction-provenance** `.npz` + its L4 validation. 238 image_materialization+metadata_ingest
> tests green.
>
> **Stale references corrected for any future reader:** the Snakemake rule is
> `materialize_image_product_for_well` (THREE inputs incl. `resolved_product_plan`), not
> `materialize_well`/two-inputs as some prose below assumes; the per-frame uniqueness helper
> `_validate_unique_keys` was DELETED in `8e7a02f2` — uniqueness now lives in the single gate
> `validate_frame_inventory_identity_contract`; the real module root is
> `src/data_pipeline/image_materialization/` (not `metadata_ingest/image_materialization/`). The
> historical planning text below is preserved as the design record.

**Status:** planning spec, mdcolon 2026-06-24. Captures how to make the front-end actually
**materialize and save z-stacks** (one PNG per Z plane), building on the already-shipped
acquisition inventory (which already explodes one row per `(position, z, channel, time)`).
**No code in this doc** — it is the wire-through map, with the one load-bearing decision named.

**Companion to:** `acquisition_inventory_flow.md` (the per-scope record that is the z-stack
LOOKUP — "the inventory IS the z-stack/tile lookup"), `recompose_yx1_front_end.md` (the
materialize backend this extends), `frame_inventory_handoff_contract.md` (the stitched-side
contract that must learn `z_index` for this to be expressible).

---

## 🪨 The one-sentence answer

**The z-stack capability is already scaffolded everywhere but executed nowhere.** The vocabulary,
the path layout, and the schema policy already know `z_stack`. The raw Z planes are already
enumerated in the acquisition inventory. So wiring it in is **four edits downstream of the
inventory — plus one load-bearing contract decision** (the z-aware `image_id` grammar) that the
others all depend on.

---

## 🧭 What already exists (the scaffolding) vs what is missing

| Layer | File | z_stack status today |
|---|---|---|
| **Raw enumeration** | `metadata_ingest/scope/yx1/acquisition_inventory.py` | ✅ **DONE** — `build_yx1_acquisition_inventory_rows` already loops `z_index` (one row per `(position, z, channel, time)`); Z is exploded "for a future per-Z pass." Nothing upstream changes. |
| **Vocabulary** | `image_materialization/materialization_plan.py` | ✅ `z_stack` is a `SUPPORTED_IMAGE_PRODUCT_TYPES` member; shape grammar already requires `projection_method=None` for it (`_validate_product_shape`). |
| **Path layout** | `image_materialization/materialized_image_paths.py` | ✅ **DONE (channel-first, 2026-06-25)** — `materialized_image_path` emits `{well_id}/{channel_id}/z_stack/{z_image_id}.png` and `{well_id}/{channel_id}/projection/{projection_method}/{image_id}.png`; `projection_method` is REQUIRED in the generic constructor. `focus_index_map_path` lands under `{well_id}/{channel_id}/projection/focus_stack/focus_index_map/{image_id}.npz` and is gated by a SEPARATE `ALLOWED_PROVENANCE_SUFFIXES` (`.npz`), never `ALLOWED_IMAGE_SUFFIXES`. |
| **Resolver** | `image_materialization/scope/scope_resolver_for_materialization_plan.py` | ❌ `_resolve_yx1_product` **hard-rejects** `image_product_type != "projection"`. |
| **Executor** | `image_materialization/scope/yx1/materialize_well_yx1.py` | ❌ ignores the plan's product types entirely; hardcodes the BF focus-stack-projection loop. |
| **Contract** | `image_materialization/frame_inventory_contract.py` | ❌ **the blocker.** `image_id` is derived from FOUR atoms with **no `z_index`**; the unique key has no `z_index`; the validator recomputes `image_id` from atoms and fails loud. A z-stack cannot be expressed. |

---

## ⚠️ THE LOAD-BEARING DECISION — the z-aware `image_id` grammar (do this FIRST)

Everything else is mechanical; this is the real design move. Stated precisely so it is not
papered over:

- The frame-inventory **unique key is the four atoms** `(experiment_id, well_index, channel_id,
  time_index)` — **no `z_index`** (`frame_inventory_contract.py:70`).
- `image_id` is **derived from exactly those four atoms** via `build_image_id`
  (`derive_image_id`, line 132; grammar `{well_id}_{channel_id}_t{time_index:04d}`).
- The validator (`assert_derived_ids_consistent`, `frame_inventory_image_ids`) **recomputes**
  `image_id` from atoms and **fails loud** on disagreement, AND treats a duplicate `image_id` as
  a duplicate frame.

> **Consequence:** N z-stack planes share one `(exp, well, channel, time)`. Under today's
> contract they are N **duplicate** rows with N **identical** `image_id`s → the validator rejects
> them. This is exactly why the Step-3 plan deferred z_stack: *"add when z_index enters the
> identity grammar."* **This is that moment.**

### Decision (LOCKED 2026-06-24): `image_id` includes `z_index`

A z-stack row's identity is a **z-aware `image_id`**, distinct per plane. Projection rows are
unchanged. This keeps the system one-row-one-image downstream (segmentation never has to learn a
compound key). The z-aware filename grammar already exists; the directory layout must be updated to
the locked channel-first grammar.

```
projection row : image_id = {well_id}_{channel_id}_t{time_index:04d}              (z_index = NA)
z_stack row    : image_id = {well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}
```

### What that decision forces (the actual edits — TWO distinct files)

> **`build_image_id` is a SHARED upstream API change, not a contract sub-edit.** It lives in
> `shared/identifiers/constructors.py` and has its own importers; `frame_inventory_contract.py`
> only forwards to it. Treating it as a sub-bullet hides the blast radius — it is **edit 0** in
> the build order below.

1. **`build_image_id` gains an optional `z_index` (SHARED — `shared/identifiers/constructors.py`).**
   `z_index=None` → today's grammar **byte-for-byte** (projection). Set → append `_z{z_index:04d}`.
   `derive_image_id` in `frame_inventory_contract.py` is a thin forward of the new signature.
2. **The unique key becomes z-aware.** `UNIQUE_FRAME_INVENTORY_KEY_COLUMNS` gains `z_index` as a
   **nullable** key column (NA for projection; the (exp, well, channel, time) sub-key is unique
   among projection rows, and (… , z) is unique among z_stack rows). `frame_inventory_image_ids`
   routes `z_index` into `build_image_id` so the effective key stays anchored to the grammar.
3. **`assert_derived_ids_consistent`** recomputes `image_id` WITH `z_index` so z-stack rows
   validate instead of colliding.
4. `z_index` is already in `DOWNSTREAM_FRAME_IDENTITY_BLOCK` (nullable) — **no change there**;
   the downstream identity validator already tolerates it.

> **⚠️ The NA-in-unique-key landmine (the real implementation trap).** Adding a **nullable**
> column to a duplicate-detection key is slippery in Pandas: `groupby` **drops NA keys by
> default** (`dropna=True`), and `duplicated()`/`drop_duplicates()` treat `pd.NA` differently
> across dtypes (object vs nullable `Int64`). The failure mode is silent: projection rows keyed on
> `z_index=NA` **stop colliding** and a real duplicate frame slips through. **Fix:** normalize the
> effective key before the duplicate check — `effective_z = df["z_index"].fillna("__PROJECTION__")`
> (a real sentinel value) — or route everything through `frame_inventory_image_ids` (the z-aware
> `image_id` is never NA, so the string key collides correctly by construction). The duplicate
> check must operate on the **derived `image_id`**, not the raw nullable tuple.

> **Boundary guard (carry from `acquisition_inventory_flow.md` decision 7):** P/Z uniqueness
> lives on the **acquisition** side. Here we are adding `z_index` to the **frame-inventory**
> identity ONLY because the frame *is* a single Z plane now — not dragging raw-unit provenance
> across. The frame_inventory row still names one materialized image identity; for a z_stack row,
> the primary `source_image_path` is the single Z-plane pixel file. This is consistent, not a violation: the rule was "don't collapse
> P/Z into the projection key," and a z_stack frame is genuinely a distinct image.

---

## The build order — tests ship with each layer (contract-first, blast radius honest)

```text
0. shared identifiers + tests   build_image_id(..., z_index=None)      upstream shared API
1. frame_inventory contract + tests
                                derive_image_id + nullable z key + assert_derived_ids_consistent
2. resolver + tests             accept YX1 z_stack
3. paths + tests                channel-first z_stack/projection/focus-map paths
4. executor + tests             materialize one resolved product
```

Edits 0–1 are the contract; 2–4 are mechanical and downstream; do them in order.

> **🚦 HARD IMPLEMENTATION GATE (operational, not poetic).** After edits 0–1, run ONLY the
> shared-identifier + frame_inventory-contract tests. **Do not touch resolver / path / executor
> until ALL of these are green:**
> - projection rows with `z_index=NA` and the same `(exp, well, channel, time)` → **duplicate → fail**
> - z_stack rows with `z_index` 0 vs 1 → **not duplicate → pass**
> - z_stack rows with the same `z_index` → **duplicate → fail**
> - `build_image_id(..., z_index=None)` returns the projection grammar **byte-for-byte unchanged**
>
> The contract must be able to *express and police* z-stack identity before any code *emits* it.
> "Never let pixels outrun the contract."

> **🏷️ Naming guard — keep the constructor name STABLE.** The *grammar* becomes z-aware; the
> *public concept* stays "image identity." Extend `build_image_id` with an optional `z_index`
> parameter — do **NOT** fork a `build_z_aware_image_id` / `build_zstack_image_id` sibling. One
> constructor, one identity concept; the `z_index=None` default IS the projection world.

## The mechanical edits (downstream of the inventory, after the contract)

### 2. Resolver — open the YX1 gate to `z_stack`
`_resolve_yx1_product` (`scope_resolver_for_materialization_plan.py:78`): stop raising on
`image_product_type == "z_stack"`. Accept it (grammar already guarantees `projection_method is
None` for z_stack), keep rejecting non-BF channels, keep `xy_composition → identity`.

### 3. Path layout — enforce the channel-first product path grammar
`materialized_image_paths.py` must implement the locked grammar, not just add a wrapper:

```text
{channel_id}/z_stack/{z_image_id}.png
{channel_id}/projection/{projection_method}/{image_id}.png
{channel_id}/projection/focus_stack/focus_index_map/{image_id}.npz
```

Add `z_stack_frame_path`, update `projection_frame_path` to include `projection_method`, and add
`focus_index_map_path`. The generic constructor may remain only if it can express this grammar
unambiguously.

### 4. Executor — materialize one resolved product
The real work. Today the loop is hardcoded: per `time_index`, focus-stack the whole Z-stack into
ONE projection frame. In the product-shard model, the executor must consume exactly one resolved
product plan and emit exactly one product frame-inventory shard. Do **not** iterate
`resolved_plan.products` inside the product materializer; Snakemake owns product fanout.

`scope/yx1/materialize_well_yx1.py` should expose/route to a product-grain executor:
`materialize_yx1_product_for_well(..., resolved_product_plan_json, ...)`. The resolved product
plan represents exactly one `product_key`.

- **`image_product_type == "projection"`** → unchanged: `materialize_ff_projection(stack)`, one row,
  `z_index=pd.NA`, `image_product_type="projection"`, `projection_method="focus_stack"`, and emit
  the `BF__projection__focus_stack` product frame-inventory shard.
- **`image_product_type == "z_stack"`** → **do NOT project.** The `stack_zyx` from `_get_stack(...)`
  already has every plane. Iterate the `z_index` values **from the well's inventory rows** (the
  inventory is the system of record for which Z planes exist — do not infer from array shape),
  write one PNG per `(time_index, z_index)` via `z_stack_frame_path`, and emit the `BF__z_stack`
  product frame-inventory shard with one row per plane: real `z_index`,
  `image_product_type="z_stack"`, `projection_method=pd.NA`, z-aware `image_id`.

> The Z planes come straight off the same `_get_stack(...)` call the projection path already
> makes — projection collapses the stack, z_stack writes each plane. No new ND2 read, no new
> tensor slice; the cost is N image writes instead of 1.

### 5. Tests (the duplicate-semantics tests must exist BEFORE the executor is touched)

**Shared identifier (edit 0):**
- `build_image_id(well, channel, t, z_index=None)` returns the **old projection grammar
  byte-for-byte** (regression guard — no existing caller's id may shift).
- `build_image_id(..., z_index=3)` appends `_z0003` in the locked position.

**Contract — the NA-key duplicate semantics (the landmine; required):**
- Two **projection** rows, same `(exp, well, channel, time)`, `z_index=NA` → **duplicate → fail**
  (proves the NA key still collides; this is the regression the `groupby(dropna=True)` trap breaks).
- Two **z_stack** rows, same `(exp, well, channel, time)`, `z_index` 0 vs 1 → **not duplicate → pass**.
- Two **z_stack** rows, same `(exp, well, channel, time)`, same `z_index` → **duplicate → fail**.
- `assert_derived_ids_consistent` recomputes **both** projection (NA-z) and z_stack `image_id`
  correctly; a hand-edited wrong z-aware `image_id` fails loud.

**Resolver:** a `z_stack` request resolves (not raises); a non-BF z_stack still raises.

**Path:** `z_stack_frame_path` lands under `{channel_id}/z_stack/`, filename carries
`z{:04d}_t{:04d}`; missing `z_index` raises. `projection_frame_path` includes
`projection_method` under `{channel_id}/projection/{projection_method}/`, and `focus_index_map_path`
lands under `{channel_id}/projection/focus_stack/focus_index_map/`.

**Executor (mocked ND2, no GPU):** a `z_stack` plan emits N rows per time_index with distinct
`z_index` + distinct `image_id`; a `projection` plan still emits one NA-z row; the executor
iterates `z_index` **from the inventory**, not the array shape (feed an inventory with fewer Z
planes than the mock array has and prove only the inventory's planes are written).

---

## How a caller requests z-stacks (the config seam)

`load_image_materialization_plan` already reads `config["image_materialization"]["products"]`.
Once the gate is open, a z-stack is just a product entry — no new config grammar:

```yaml
image_materialization:
  products:
    - {channel_id: BF, image_product_type: projection, projection_method: focus_stack}
    - {channel_id: BF, image_product_type: z_stack}     # ← saves every Z plane
```

Both products can be requested in one invocation, but they fan out into separate product jobs. Each
product job consumes one resolved product plan and writes one product shard; the executor never uses
current config products to decide the persistent product universe.

---

## ✅ Decisions locked
1. **z-aware `image_id`** for z_stack rows (`…_z{z:04d}_t{t:04d}`); projection unchanged.
   Doctrine: *projection collapses Z; z_stack materializes Z; the `image_id` says which world the
   frame lives in.*
2. **The contract is edited FIRST**, and the order is FIVE edits — `build_image_id` (SHARED) →
   contract (key/validator, nullable `z_index`) → resolver → path → executor → tests. The shared
   constructor is its own edit (edit 0), not a contract sub-bullet — the blast radius is honest.
2a. **Normalize the nullable `z_index` before any duplicate check** (sentinel fill or route
   through `frame_inventory_image_ids`). Pandas `groupby(dropna=True)` / `duplicated()` silently
   stop colliding NA keys; the duplicate-semantics tests (projection-NA collides; z 0≠1; same-z
   collides) gate the executor work.
3. **The inventory is the source of truth for which Z planes exist** — the executor iterates the
   inventory's `z_index` values, never the array shape.
4. **One row per materialized pixel file holds** — a z_stack plane is a distinct image, so
   `z_index` enters frame-inventory identity legitimately (NOT raw-unit provenance leaking across
   the stitched boundary; the acquisition side keeps its own full-cell P/Z key).
5. **No new config grammar** — z_stack is an additional `products[]` entry; the default stays the
   one BF projection product.
6. **HARD implementation gate** — resolver/path/executor stay untouched until edits 0–1 ship with
   the four green tests (projection-NA collides, z 0≠1 passes, same-z collides, projection grammar
   byte-for-byte). The contract must *express and police* z-stack identity before any code emits it.
7. **Stable constructor name** — extend `build_image_id(..., z_index=None)`; do NOT fork a
   `build_z_aware_image_id` sibling. The grammar becomes z-aware; the public identity concept stays
   one constructor with `z_index=None` as the projection default.

---

> **Tiny doctrine:** *First teach identity what Z means. Then let the executor write Z. Never let
> pixels outrun the contract.*

LOCKED TARGET DECISIONS — Product-Shard Frame Inventory for Z-Stack Materialization

This target has no open design questions. The purpose is to define the desired architecture clearly enough for implementation.

================================================================================
1. Config products are requested work, not persistent dataset state
================================================================================

Decision:
  image_materialization.products means "products requested for this invocation."

It does NOT mean:
  "the complete set of products that should remain in frame_inventory."

Consequence:
  If B01 already has a validated projection product shard and this run requests only z_stack, the assembled B01 frame_inventory still includes projection plus z_stack.

Doctrine:
  Config is an instruction, not memory.
  Product shards are memory.

================================================================================
2. Product shards are the persistent product memory layer
================================================================================

Decision:
  Each materialized product writes its own per-well product frame_inventory shard.

Product shard grain:
  well_id
  channel_id
  image_product_type
  projection_method, when applicable

Examples:
  BF__projection__focus_stack_frame_inventory.csv
  BF__z_stack_frame_inventory.csv

A product shard contains only rows for that product.

Consequence:
  Rerunning z_stack replaces the z_stack product shard.
  Rerunning projection replaces the projection product shard.
  Neither operation rewrites unrelated product shards.

Doctrine:
  Products are leaves.
  Well inventory is a union of leaves.

================================================================================
3. Canonical per-well frame_inventory is assembled from discovered validated product shards
================================================================================

Decision:
  The canonical per-well frame_inventory is not written directly by the product materializer.

Instead:
  product materializer -> product shard
  product discovery -> discovered product shard manifest
  well assembler -> canonical per-well frame_inventory

The well assembler reads all discovered validated product shards for the well and concatenates them.

Consequence:
  Adding z_stack to a well that already has projection creates a z_stack product shard.
  Discovery sees projection + z_stack.
  Assembly writes a canonical well frame_inventory containing both.

Doctrine:
  Discovery finds memory.
  Well inventory is assembled memory.

================================================================================
4. Product discovery uses a manifest, not current config
================================================================================

Decision:
  Add a per-well discovered-product-shards manifest.

Suggested output:
  discovered_product_shards/per_well/{well_id}/{well_id}_discovered_product_shards.csv

Manifest columns:
  well_id
  product_key
  frame_inventory_csv
  validated_flag

Discovery behavior:
  - scans the product shard directory for validated product shards
  - includes only product shards with valid sentinels
  - ignores unvalidated product CSVs
  - ignores orphan PNGs
  - ignores current config products
  - fails loud on duplicate product_key for one well
  - fails loud on malformed product shard names

Decision:
  MVP implementation may let the assembly script read product shard paths from the manifest.
  A Snakemake checkpoint is not required for the first implementation unless needed to express concrete dynamic inputs.

Implementation decision:
  For a given run, discovery must depend on the validated sentinels for the products requested in
  the current invocation as explicit Snakemake inputs. The requested product `.validated` sentinels
  are DAG triggers, not discovery limits. Discovery still scans the full product-shard directory and
  emits all currently validated product shards for that well, including older products not requested
  in this invocation. This gives concrete DAG triggering for newly materialized products while
  preserving product-shard memory semantics. Do not rely on directory mtimes for discovery reruns.

Doctrine:
  Current config tells us what to run.
  Product discovery tells us what exists.

================================================================================
5. Product retirement is explicit
================================================================================

Decision:
  A product is not removed from the assembled well frame_inventory merely because it is absent from the current config.

To remove a product, use an explicit retirement action.

MVP retirement mechanism:
  - remove the product shard's .validated sentinel

Not in MVP:
  - retired/ moves
  - retire-product CLI verb
  - candidate/promote

Target semantics:
  Discovery includes active validated product shards.
  Discovery excludes retired or unvalidated product shards.
  MVP retirement is a manual state change. After removing a product shard's .validated sentinel,
  rerun the relevant well/experiment frame_inventory target so discovery and assembly refresh.

Consequence:
  Running z_stack-only does not remove projection.
  Removing projection requires an explicit retire projection action.

Doctrine:
  Absence from today's request is not deletion from the dataset.

================================================================================
6. Candidate promotion is deferred
================================================================================

Decision:
  Do not require candidate -> validate -> promote for the MVP product-shard implementation.

MVP:
  - product materializer writes product outputs and product frame_inventory shard
  - product shard is validated
  - well inventory is assembled from validated product shards
  - downstream trusts validated frame_inventory, not directory walks

Future hardening:
  - write product outputs to candidate/
  - validate candidate product shard
  - promote candidate product dir + product shard

Doctrine:
  Product shards make stale rows impossible.
  Candidate promotion later makes stale files disappear cleanly.

================================================================================
7. Stale files are inert in MVP
================================================================================

Decision:
  Stale PNG files on disk are allowed in MVP if they are not represented in the current validated frame_inventory.

Reason:
  Downstream stages must read image paths from validated frame_inventory.
  Downstream stages must not glob materialized image directories.

Optional cleanup:
  A product materializer may clear only its own product output directory before writing.
  It must not broadly delete a well directory unless that directory is exclusively owned by materialize_well.

Consequence:
  Stale files are a disk-hygiene problem, not a pipeline-truth problem.

Doctrine:
  The inventory is truth.
  The filesystem is storage.
  Do not let storage become discovery.

Future projection-method files may exist on disk during development, for example under
`BF/projection/max/`, but they are inert unless represented in validated product shards and accepted by
frame_inventory validation.

================================================================================
8. Unsupported requested products fail loud
================================================================================

Decision:
  If config requests a product that a scope cannot materialize, the resolver fails loud.

Example:
  YX1 z_stack BF is allowed.
  non-BF z_stack is rejected until explicitly supported.
  Unsupported image_product_type is rejected.

Do not silently omit unsupported requested products.

Reason:
  Config products are user instructions.
  If the instruction cannot be honored, the run must say so.

Doctrine:
  Resolver commits or refuses.
  It does not quietly forget.

================================================================================
9. z_stack rows enter frame_inventory
================================================================================

Decision:
  z_stack rows are real frame_inventory rows.

Reason:
  A z_stack plane is the primary materialized pixel file for its image identity.
  The frame_inventory contract is one row per materialized image identity; source_image_path points
  to the primary image file, with nullable construction-provenance paths allowed only when they
  explain that same image.

Consequence:
  z_stack requires z-aware image_id grammar.

Projection:
  {well_id}_{channel_id}_t{time_index:04d}

z_stack:
  {well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}

Doctrine:
  Projection collapses Z.
  Z-stack preserves Z.
  image_id names the true pixel identity.

================================================================================
10. Detection consumes BF focus_stack projection rows only
================================================================================

Decision:
  Detection/SAM/tracking consume the canonical BF focus_stack projection frames, not all BF rows
  and not every future BF projection method.

Add a named helper:
  _projection_bf_rows(frame_inventory)

Behavior:
  - if image_product_type is absent, treat BF rows as legacy projection/focus_stack
  - if projection_method is absent on legacy projection rows, treat them as focus_stack
  - otherwise filter:
      channel_id == "BF"
      and image_product_type == "projection"
      and projection_method == "focus_stack"

Consequence:
  z_stack BF rows and future non-focus_stack BF projections do not accidentally flow into detection.

Doctrine:
  Detection wants the canonical BF focus_stack movie, not any available 2D projection.

================================================================================
11. Final target flow
================================================================================

Requested product run:

  config products
      -> resolved product plans for this invocation
      -> materialize/update requested product shards
      -> validate product shards

Persistent assembly:

  discover validated product shards for well
      -> assemble canonical per-well frame_inventory
      -> validate per-well frame_inventory
      -> merge experiment frame_inventory
      -> validate merged frame_inventory

B01 example:

  First run projection:
    creates BF__projection__focus_stack product shard
    B01 frame_inventory = projection

  Later run z_stack only:
    creates BF__z_stack product shard
    discovery sees projection + z_stack
    B01 frame_inventory = projection + z_stack

  Later rerun z_stack:
    replaces BF__z_stack product shard
    projection shard remains
    B01 frame_inventory = projection + updated z_stack

  Later retire projection:
    explicit retirement removes/disables the projection shard's .validated sentinel
    discovery sees z_stack only
    B01 frame_inventory = z_stack only

================================================================================
12. Final doctrine
================================================================================

Config controls requested work.
Resolved product plans trigger requested product jobs.
Product shards store completed products.
Discovery finds active validated product shards.
The well frame_inventory is assembled from discovered shards.
The experiment frame_inventory is assembled from validated well inventories.

One frame_inventory row names one materialized image identity. `source_image_path` points to the primary pixel file; validated construction-provenance paths may ride with the image they explain.

Never patch the river.
Rebuild it from trusted springs.


## MVP implementation boundary (LOCKED before coding)

This spec intentionally keeps image identity narrower than the filesystem layout can represent.
The path tree may be future-proof, but the canonical frame_inventory defines what can coexist now.

### One materialized image path grammar

There is exactly one materialized image path grammar:

```text
materialized_images/{well_id}/{channel_id}/z_stack/{z_image_id}.png
materialized_images/{well_id}/{channel_id}/projection/{projection_method}/{image_id}.png
materialized_images/{well_id}/{channel_id}/projection/focus_stack/focus_index_map/{image_id}.npz
```

All path helpers and tests must assert this grammar. Do not keep product-type-first examples such as
`projection/{channel_id}/...` or `z_stack/{channel_id}/...` in implementation-facing text.

### MVP projection identity boundary

Projection method is encoded in the materialized image path and in `product_key`, but not in
`image_id`.

```text
path/product_key: BF/projection/focus_stack/..., BF__projection__focus_stack
image_id:        {well_id}_BF_t0000
```

Therefore multiple projection methods for the same `(well_id, channel_id, time_index)` cannot coexist
in the canonical frame_inventory in the MVP. If `focus_stack` and `max` projection rows for the same
well/channel/time are accidentally assembled together, validation must fail loud on duplicate
projection image identity. Product-aware projection image-id grammar is deferred until projection-method
coexistence is actually required.

Doctrine:

```text
Paths may prepare for the future.
IDs define what can coexist now.
Validators guard the boundary.
```

### Product-shard schema contract

All product frame_inventory shards must conform to the full canonical frame_inventory schema. Product-
specific fields are allowed only as nullable, contract-declared columns. In this MVP,
`focus_index_map_path` is present on every product shard: populated for projection/focus_stack rows and
NA for z_stack rows or non-focus_stack projection rows.

The assembler must fail on undeclared columns and missing canonical columns. It must not silently union
schema variants across product shards.

### Product-shard discovery and validation

Discovery output must be deterministic: sort by `product_key`.

Each discovered product shard must:

- exist and have a matching `.validated` sentinel
- contain exactly one well
- contain exactly one product-key-equivalent tuple: `channel_id`, `image_product_type`, `projection_method`
- have filename `product_key` matching the table contents
- fail loud on malformed filenames or duplicate product keys

Product-shard validation should catch a shard named `BF__z_stack` that contains projection rows before
assembly reaches the final frame_inventory validator.

### Deterministic resolved product plans

Resolved product plan JSON is a real Snakemake input and the materializer source of truth. It must be
canonical and deterministic:

- stable key order
- no timestamps
- no nondeterministic formatting
- no environment-dependent absolute noise unless required for execution

The materializer trusts the resolved product plan; it does not reread config to decide what product to
write.

### MVP retirement limitation

MVP product retirement is manual `.validated` sentinel removal. This is a developer/operator action,
not a first-class pipeline operation. Snakemake will not automatically infer that manual state change
unless the relevant discovery/frame_inventory target is rerun or forced. A proper `retire-product` CLI
is deferred.

### Frame-inventory row doctrine

A frame_inventory row names one materialized image identity. `source_image_path` points to the primary
materialized pixel file. Rows may carry validated construction-provenance paths only when those paths
explain how that same image was produced, are 1:1 with `image_id`, are co-produced by the same
materialization job, and are L4 validated.

`focus_index_map_path` is the current allowed construction-provenance path. It is not a product, not a
product key, and not a second image identity.

### Detection input boundary

Detection consumes the canonical BF focus-stack projection movie, not arbitrary BF projections.

`_projection_bf_rows(frame_inventory)` should select:

```text
channel_id == "BF"
image_product_type == "projection"
projection_method == "focus_stack"
```

Backward compatibility: if `image_product_type` is absent, treat BF rows as legacy projection/focus_stack;
if `projection_method` is absent on legacy projection rows, treat them as focus_stack.

### Z-index normalization

For z_stack rows, `z_index` must be integer-valued and non-negative. The executor sorts z planes
numerically before writing.

The executor must not confuse acquisition `z_index` labels with numpy stack-axis offsets. If
`z_index` labels are not guaranteed to equal stack-axis offsets, build an ordered `z_index -> stack_axis`
mapping from the inventory rows used to load the stack. The same distinction applies to focus maps.

### Focus-index-map value semantics

`focus_index_map` values stored in the `.npz` are stack-axis offsets, not acquisition `z_index` labels.
The companion `z_indices` array stores the ordered acquisition `z_index` labels corresponding to stack
axis offsets:

```text
focus_index_map[y, x] = k              # 0 <= k < len(z_indices)
z_indices[k]          = acquisition z_index label selected for pixel (y, x)
```

Validation checks `focus_index_map.min >= 0`, `focus_index_map.max < len(z_indices)`, and that
`z_indices` matches the ordered inventory `z_index` labels for that timepoint/channel when that context
is available.

### Build order and tests

Tests are not a final garnish; they ship with each layer:

```text
0. identifiers + tests
1. frame_inventory contract + tests
2. product-shard architecture + projection-only tests
3. resolver/path z_stack + tests
4. executor z_stack + tests
5. detection filter + tests
6. focus_index_map provenance + L4 validation tests
```


## Code layout and ownership target

This section defines where each layer lives and which artifacts each layer is allowed to write.
The implementation should extend the existing image-materialization and frame-inventory pipeline.
Do not create a parallel pipeline, second path grammar, or side workflow.

Use the real module root and existing workflow helpers:

```text
src/data_pipeline/image_materialization/
workflow/rules/frame_inventory.smk
workflow/rules/materialize_well_native.smk
workflow/orchestration/paths.py
```

Layout doctrine:

- Identifiers define names.
- Materialization plans define requested products; resolved product plans define per-well/product execution.
- Paths define locations.
- Executors write product outputs.
- Validators write sentinels.
- Discovery lists active product shards.
- Assembly writes canonical inventories.
- Consumers read canonical inventories.

Negative boundaries:

- The executor never writes canonical well `frame_inventory`.
- The assembler never writes product shards.
- Discovery never reads current config.
- Detection never glob-walks materialized images.
- Snakemake owns product fanout; product materializers do not iterate over config products.
- Snakemake rules should call CLI/task verbs, not import library functions directly.

### Artifact writer ownership

| Artifact | Written by | Read by |
|---|---|---|
| `{product_key}_resolved_product_plan.json` | `resolve_product_plan_for_well_product` | `materialize_product_for_well` |
| product `frame_inventory.csv` | `materialize_product_for_well` | `validate_product_frame_inventory`, `assemble_well_frame_inventory` |
| product `frame_inventory.validated` | `validate_product_frame_inventory` | `discover_product_shards_for_well` |
| `discovered_product_shards.csv` | `discover_product_shards_for_well` | `assemble_well_frame_inventory` |
| canonical well `frame_inventory.csv` | `assemble_well_frame_inventory` | `validate_frame_inventory_for_well` |
| merged experiment `frame_inventory.csv` | `merge_frame_inventory` | merged validation and downstream consumers |

### 1. Identity layer

Shared image identity lives in:

```text
src/data_pipeline/shared/identifiers/constructors.py
```

Responsibilities:

- Extend `build_image_id(..., z_index=None)`.
- Keep projection grammar byte-for-byte unchanged.
- Add z-aware grammar for z_stack rows.
- Add `parse_image_id_with_z_index`.
- Keep `parse_image_id` backward-compatible and prevent silent z dropping.

Frame-inventory identity and validation live in:

```text
src/data_pipeline/image_materialization/frame_inventory_contract.py
src/data_pipeline/metadata_ingest/frame_inventory/frame_inventory_validation.py
src/data_pipeline/metadata_ingest/frame_inventory/frame_inventory_validation_rules.py
```

Responsibilities:

- Route `z_index` into `derive_image_id` / `frame_inventory_image_ids`.
- Assert derived `image_id`s using `z_index`.
- Keep duplicate checks based on derived image identity, not raw nullable `z_index` tuples. For MVP projection rows, projection_method is not part of `image_id`, so accidental multiple projection methods at the same well/channel/time must fail loud instead of coexisting silently.
- Confirm temporal grain validation still works with z_stack rows.

Product identity is centralized in a new helper:

```text
src/data_pipeline/image_materialization/image_product_keys.py
```

Responsibilities:

- Build `product_key = channel_id + image_product_type + [projection_method]`.
- Keep `product_key` well-agnostic.
- Validate product-key format if needed.
- Examples: `BF__projection__focus_stack`, `BF__z_stack`.

Do not put `well_id` in `product_key`.

### 2. Plan and resolution layer

Keep `materialization_plan.py` as user/config intent. Create a separate resolved execution module:

```text
src/data_pipeline/image_materialization/materialization_plan.py
src/data_pipeline/image_materialization/resolved_product_plan.py
src/data_pipeline/image_materialization/scope/scope_resolver_for_materialization_plan.py
```

Ownership split:

- `materialization_plan.py` owns user/config product requests.
- `resolved_product_plan.py` owns one per-well/product execution commitment.
- `scope_resolver_for_materialization_plan.py` converts requested products into resolved product jobs.

Resolved product plan responsibilities:

- Represent exactly one resolved product job.
- Include `experiment_id` and `well_id` as execution context.
- Include `product_key`, `channel_id`, `image_product_type`, `projection_method`, and `xy_composition`.
- Serialize/deserialize resolved product plan JSON.
- Assert path/wildcard `well_id` matches `plan.well_id` before execution.

Resolver responsibilities:

- Accept YX1 BF z_stack.
- Keep rejecting unsupported/non-BF products loudly.
- Resolve one requested product into one resolved product plan.
- Do not silently omit unsupported requested products.

Resolved product plan path:

```text
resolved_product_plans/per_well/{well_id}/{product_key}_resolved_product_plan.json
```

### 3. Path layer

Extend existing path/orchestration helpers rather than inventing a new path grammar:

```text
workflow/orchestration/paths.py
workflow/rules/frame_inventory.smk
src/data_pipeline/image_materialization/materialized_image_paths.py
```

Path helper responsibilities:

- Add resolved product plan paths.
- Add product frame-inventory shard paths and `.validated` paths.
- Add discovered product-shard manifest paths.
- Use existing `PATH_MODE_PER_WELL` / `rule_artifact` / `rule_validated` machinery where possible.

Product-shard paths:

```text
resolved_product_plans/per_well/{well_id}/{product_key}_resolved_product_plan.json
frame_inventory_products/per_well/{well_id}/{well_id}_{product_key}_frame_inventory.csv
frame_inventory_products/per_well/{well_id}/{well_id}_{product_key}_frame_inventory.validated
discovered_product_shards/per_well/{well_id}/{well_id}_discovered_product_shards.csv
```

Path rules:

- `product_key` is a wildcard.
- `well_id` is path context.
- `product_key` does not include `well_id`.

Materialized image path responsibilities:

- Add `z_stack_frame_path` using the channel-first layout.
- Require `z_index` for z_stack paths.
- Update `projection_frame_path` to include `projection_method` under
  `{channel_id}/projection/{projection_method}/`.
- Add `focus_index_map_path` under
  `{channel_id}/projection/focus_stack/focus_index_map/`.
- Preserve the generic `materialized_image_path` API only if it can express the locked grammar
  unambiguously.

### 4. Execution layer

Keep product materialization in the existing YX1 executor for now:

```text
src/data_pipeline/image_materialization/scope/yx1/materialize_well_yx1.py
```

Target callable:

```text
materialize_yx1_product_for_well(...)
```

Responsibilities:

- Consume exactly one resolved product plan.
- Emit exactly one product frame-inventory shard.
- Projection product writes the `BF__projection__focus_stack` shard.
- z_stack product writes the `BF__z_stack` shard.
- Iterate `z_index` from acquisition inventory, not array shape.
- Optionally clear only the owned product output directory before writing.
- Never write the canonical per-well `frame_inventory` directly.

Do not create a new backend package unless this file becomes unwieldy.

### 5. Product-shard validation layer

Add a thin product-shard validation helper:

```text
src/data_pipeline/image_materialization/product_frame_inventory_validation.py
```

Responsibilities:

- Validate that a product shard contains one well/product only.
- Validate `product_key` consistency.
- Validate `image_id` derivation.
- Validate projection rows have `z_index` NA.
- Validate z_stack rows have `z_index` non-null.
- For projection/focus_stack rows, require `focus_index_map_path` in MVP. The path must exist, the
  `.npz` must load, it must contain `focus_index_map` and `z_indices`, `focus_index_map` values must
  be stack-axis offsets, and `z_indices` must map offsets to acquisition `z_index` labels.
- For z_stack rows and non-focus_stack projection rows, require `focus_index_map_path` to be NA.
- Reuse strict frame-inventory validation where appropriate.

The product-shard validator writes the product `.validated` sentinel. The assembled well validator
remains the strict final gate.

### 6. Discovery and assembly layer

Create separate modules. Discovery lists product shards; assembly concatenates product shards.

```text
src/data_pipeline/image_materialization/discover_product_shards.py
src/data_pipeline/image_materialization/assemble_well_frame_inventory.py
```

Discovery responsibilities:

- Scan one well's product-shard directory.
- Include only product shards with `.validated` sentinels.
- Ignore unvalidated CSVs and orphan PNGs.
- Ignore current config products.
- Fail loud on duplicate `product_key`.
- Fail loud on malformed shard names.
- Write `discovered_product_shards.csv`.

Discovery output:

```text
discovered_product_shards/per_well/{well_id}/{well_id}_discovered_product_shards.csv
```

Requested product `.validated` sentinels are DAG triggers, not discovery limits.

Assembly responsibilities:

- Read `discovered_product_shards.csv`.
- Read listed product frame-inventory shards.
- Concatenate rows.
- Sort deterministically.
- Write canonical per-well `frame_inventory`.
- Never read current config products.
- Never directory-walk image files.

Assembly output:

```text
frame_inventory/per_well/{well_id}/{well_id}_frame_inventory.csv
```

Existing strict per-well validation runs on the assembled file.

### 7. Workflow orchestration layer

Extend existing rule files rather than creating a separate workflow:

```text
workflow/rules/materialize_well_native.smk
workflow/rules/frame_inventory.smk
```

Target DAG:

```text
resolve_product_plan_for_well_product
    -> materialize_product_for_well
    -> validate_product_frame_inventory
    -> discover_product_shards_for_well
    -> assemble_frame_inventory_for_well
    -> validate_frame_inventory_for_well
    -> merge_frame_inventory
    -> validate merged frame_inventory
```

Implementation requirements:

- Materialization wildcard grain is `(well_id, product_key)`.
- Resolved product plan JSON is a real input.
- Requested product `.validated` sentinels are explicit inputs to discovery.
- Discovery manifest drives assembly.
- Assembly output replaces the old direct `materialize_well` per-well `frame_inventory` output.
- Rules call task/CLI verbs rather than importing library functions directly.

### 8. Consumer safety layer

Edit detection:

```text
src/data_pipeline/detection/run_frame_detection.py
```

Add helper:

```text
_projection_bf_rows(frame_inventory)
```

Behavior:

- If `image_product_type` is absent, treat BF rows as legacy projection/focus_stack.
- If `projection_method` is absent on legacy projection rows, treat those projection rows as focus_stack.
- Otherwise filter `channel_id == "BF"`, `image_product_type == "projection"`, and `projection_method == "focus_stack"`.
- Detection must not process BF z_stack rows or future non-focus_stack BF projections.

### 9. Tests

Add tests at the same layer as the code they exercise.

Minimum test groups:

- Shared identifiers.
- Frame-inventory contract duplicate semantics.
- Product-key helper.
- Resolved product plan serialization.
- Product-shard path helpers.
- `z_stack_frame_path`.
- YX1 resolver.
- Product-grain executor with mocked ND2.
- Product discovery manifest.
- Well assembler.
- Detection projection filter.
- Product-key/path alignment:
  `BF__projection__focus_stack` maps to `BF/projection/focus_stack/`, and `BF__z_stack` maps to
  `BF/z_stack/`.
- Projection-only product-shard chain reproduces current frame_inventory.
- Projection-then-z_stack additive assembly.

### 10. Non-goals

Do not implement in this pass:

- Candidate/promote.
- retire-product CLI.
- retired/ product-shard moves.
- Keyence z_stack.
- max/mean projection products.
- Broad cleanup of stale files.
- Downstream consumers beyond the BF projection detection filter.





## Future work, explicitly out of scope for this implementation
These do not block the locked YX1 BF projection + z_stack product-shard implementation.

- Whether `max`/`mean` projection methods land at the same time as z_stack (same `_get_stack`
  source; trivial once the executor branches) or stay deferred with GFP.
- Keyence z_stack: its acquisition inventory also explodes Z, so the same contract change unlocks
  it — but the Keyence backend (`keyence_wire_through.md`) must exist first.
- Downstream cost: a full z_stack is `n_z ×` the projection frame count on disk; confirm a
  retention/cleanup policy before enabling z_stack on a full plate (vs. the 2-well smoke).



# PLAN
 ⎿  Current Plan
     /net/trapnell/vol1/home/mdcolon/.claude/plans/net-trapnell-vol1-home-mdcolon-proj-mor-hazy-lollipop.md

     Z-Stack Materialization Wire-Through — Implementation Plan

     Context

     The spec docs/data_pipeline/specs/target/specs/front_end/z_stack_materialization_wire_through.md
     maps how to make the YX1 front-end actually materialize and save z-stacks (one PNG per Z plane),
     not just focus-stack projections. The user wants this wired through AND validated on real ND2 data.

     The load-bearing identity move: N z-stack planes share one (exp, well, channel, time) key, so under
     today's contract they collide as duplicate image_ids and the validator rejects them. The fix is a
     z-aware image_id grammar (…_z{z:04d}_t{t:04d}) that the whole system flows from.

     The spec (sections 1–12, LOCKED 2026-06-24) additionally redefines how the per-well frame_inventory is
     built and updated: a product-shard architecture. This is the bigger structural change and
     answers the user's question — "how does Snakemake know to include the new z-frames?" — which today it
     cannot (verified below). The plan below covers BOTH: the identity/contract edits AND the
     product-shard rebuild architecture.

     Verified facts (checked, not assumed)

     - Paths: the real module root is src/data_pipeline/image_materialization/ (the spec's
     metadata_ingest/image_materialization/ prefix is stale).
     - Rebuild trigger is broken today. materialize_well (rules/materialize_well_native.smk:13) has
     only two inputs (acquisition_inventory_csv, position_well_mapping_csv) — neither changes when a
     z_stack product is added. It does NOT pass --config-yaml; its only image_materialization read is
     smoke_max_time_indices. The committed smoke runs --rerun-triggers mtime (smoke config line 8),
     which narrows triggers to file mtime only — so even a params= fingerprint is ignored. Net:
     changing products would NOT rebuild a shard without --forcerun. → requested resolved product plans
     and requested product .validated sentinels must be real DAG inputs, not params or directory mtime
     assumptions.
     - Existing infra reused by the new layer: rule_artifact / rule_validated / PATH_MODE_PER_WELL
     / run_well_shard_paths (rules/frame_inventory.smk); the existing merge_frame_inventory (:67)
     re-merges the experiment table from validated per-well shards (the merged table is already a derived
     view — keep that). materialize_yx1_well already writes a FULL shard from scratch (good — product
     shards inherit that overwrite semantics).
     - Detection break is real. run_frame_detection.py:84 recomputes image_ids over the whole
     inventory then filters channel_id=='BF' (line 87) with no image_product_type filter → z_stack
     BF rows would flow into detection/SAM/tracking. Spec §10 requires a _projection_bf_rows helper.
     - GPU: this login node (t002, SGE grid) has no GPU; the pipeline reaches GPUs via
     qsub -l gpgpu=TRUE,cuda=1. The z_stack path needs no GPU (writes raw planes); only the
     focus-stack projection needs CUDA.

     ---
     Part A — Identity & contract (FIVE edits, contract-first, HARD GATE)

     Edits A0–A1 ship + go GREEN before anything emits z_stack (spec HARD GATE).

     A0 — Shared identifier: build_image_id gains optional z_index

     shared/identifiers/constructors.py:77
     - build_image_id(well_id, channel_id, time_int, z_index=None). z_index=None → today's grammar
     byte-for-byte; set → {well_id}_{channel_id}_z{z:04d}_t{time:04d} (z before t, matching the path
     layout). Do NOT fork a build_z_aware_image_id sibling (spec naming guard).
     - Parser (decided): keep parse_image_id a 3-tuple (well_id, channel_id, time_index) for
     back-compat and make it reject z-stack ids loudly (never silently drop z). Add sibling
     parse_image_id_with_z_index → (well_id, channel_id, time_index, z_index) (z=None for projection).
     Avoids the 3→4 tuple silent blast-radius trap on every w,c,t = parse_image_id(...) caller.

     A1 — Frame-inventory contract (the load-bearing edit)

     image_materialization/frame_inventory_contract.py + metadata_ingest/frame_inventory/frame_inventory_validation.py
     - derive_image_id(..., z_index=None) thin-forwards to build_image_id.
     - frame_inventory_image_ids (:185): route a row's z_index (if present & non-NA) into
     build_image_id. This is the NA-landmine fix by construction — the duplicate key is the z-aware
     string image_id, never the raw nullable tuple. The duplicate check in _validate_unique_keys
     (frame_inventory_validation.py:58) already runs on this Series — no change to the check itself.
     - assert_derived_ids_consistent (:163): recompute expected image_id WITH z_index.
     - UNIQUE_FRAME_INVENTORY_KEY_COLUMNS (:70): add z_index for honesty + a code comment that
     duplicate enforcement uses the derived z-aware image_id, NOT
     df.duplicated(UNIQUE_FRAME_INVENTORY_KEY_COLUMNS), so nobody "simplifies" back into the NA swamp.
     - z_index already nullable in DOWNSTREAM_FRAME_IDENTITY_BLOCK — no change.

     A1b — L3 grain rule (verify, likely no change)

     frame_inventory_validation_rules.py:_validate_well_temporal_grain (:85) — the rectangular check
     compares per-channel time_index SETS; z_stack adds rows but the time set is unchanged, so it stays
     rectangular. Confirm via test; only touch if z_stack breaks it.

     A2 — Resolver: open the YX1 gate to z_stack (spec §8: commit or refuse)

     scope/scope_resolver_for_materialization_plan.py:_resolve_yx1_product (:84) — accept projection AND
     z_stack; keep rejecting non-BF and unsupported types loudly (no silent omission); scope
     focus_stack-required to projection only; keep xy_composition → identity.

     A3 — Paths: enforce channel-first product path grammar

     materialized_image_paths.py must implement the locked grammar:
     {channel_id}/z_stack/{z_image_id}.png
     {channel_id}/projection/{projection_method}/{image_id}.png
     {channel_id}/projection/focus_stack/focus_index_map/{image_id}.npz
     Add z_stack_frame_path, update projection_frame_path to include projection_method, and add
     focus_index_map_path. Keep the generic constructor only if it can express this grammar
     unambiguously.

     A4 — Executor: materialize one resolved product

     scope/yx1/materialize_well_yx1.py should expose/route to a product-grain executor:
     materialize_yx1_product_for_well(..., resolved_product_plan_json, ...). The resolved product
     plan represents exactly one product_key. The executor consumes exactly one resolved product plan
     and emits exactly one product frame_inventory shard. Do NOT iterate resolved_plan.products inside
     the product materializer; Snakemake owns product fanout.
     - image_product_type == "projection" → unchanged: materialize_ff_projection, one row,
     z_index=NA, projection_method="focus_stack"; emit the BF__projection__focus_stack product
     frame_inventory shard (needs GPU).
     - image_product_type == "z_stack" → do NOT project; iterate z_index values from the well's
     inventory rows (spec decision 3: inventory is system of record, not array shape), write stack[z]
     PNGs via z_stack_frame_path, and emit the BF__z_stack product frame_inventory shard with one row
     per (t,z), z-aware image_id, projection_method=NA (CPU-only). smoke_max_time_indices still caps
     the time axis.

     A4b — Detection: select BF focus_stack projection frames (spec §10)

     detection/run_frame_detection.py:87 — extract _projection_bf_rows(frame_inventory) (named helper,
     not an inline mask). Backward compatibility: if image_product_type is absent, treat BF rows as
     legacy projection/focus_stack; if projection_method is absent on legacy projection rows, treat
     them as focus_stack. Otherwise filter channel_id=="BF", image_product_type=="projection", and
     projection_method=="focus_stack". Focused test: mixed inventory → detection rows only for BF
     focus_stack projection; BF z_stack and future non-focus_stack BF projections are excluded.

     ---
     Part B — Product-shard frame_inventory architecture (spec §§1–12)

     The bigger structural change. Replaces today's materialize_well → {well_id}_frame_inventory.csv
     direct write with a product-shard → discover → assemble chain. Config determines which product
     shards to build in this run; discovery determines which validated product shards are included when
     assembling the canonical well frame_inventory.

     Vocabulary (decided): product spec = well-AGNOSTIC reusable identity; resolved product plan/job = well-SPECIFIC
     concrete job. well_id is NOT part of product_key or product identity —
     it is execution context (carried inside the resolved-plan JSON alongside experiment_id).
     product_key = channel_id + image_product_type + [projection_method] ONLY.

     B1 — Product shard grain + paths (spec §2, §9)

     - A product shard = per-well, per-product table holding ONLY that product's rows.
     product_key = {channel_id}__{image_product_type}[__{projection_method}], e.g.
     BF__projection__focus_stack, BF__z_stack. Shard file:
     …/per_well/{well_id}/{well_id}_{product_key}_frame_inventory.csv (+ .validated sentinel). The
     {well_id}_ filename prefix is path context, not part of product_key.
     - Add product-shard path helpers alongside the existing rule_artifact/rule_validated in
     frame_inventory.smk / orchestration/paths.py. Reuse PATH_MODE_PER_WELL; no parallel path system.
     - The executor emits one shard per resolved product. Rerunning a product replaces only its shard
     (overwrite — inherited from the current full-shard write); unrelated product shards are untouched.

     B2 — Materialization unit = per (well_id, product_key), driven by a RESOLVED PRODUCT PLAN

     The DAG leaf is one resolved product plan → one product materialization job → one product shard →
     one validated sentinel. The product is the leaf; the well inventory is assembled later.
     - An upstream rule writes, per requested product, a per-(well, product) resolved product plan:
     resolved_product_plans/per_well/{well_id}/{product_key}_resolved_product_plan.json
     (e.g. …/20250912_B01/BF__z_stack_resolved_product_plan.json). The JSON carries the well-specific
     execution context: experiment_id, well_id, and the resolved product (channel_id,
     image_product_type, projection_method, xy_composition). The resolved product plan is generated
     only for products requested in this invocation. It is a rebuild trigger for the requested product
     shard only. It is NOT used by assembly to decide the persistent product universe; assembly uses
     discovery of validated product shards.
     - The product-shard rule takes that plan as a real Snakemake input: (NOT a param). Because the
     smoke uses --rerun-triggers mtime, only an input-file mtime change re-fires the rule — so changing
     requested z_stack creates/updates ONLY the BF__z_stack plan + its product shard. Existing
     BF__projection__focus_stack shards remain discoverable and are included by assembly. The
     materializer asserts plan.well_id/experiment_id match the wildcard/path before doing work.
     - This is the concrete answer to "how does Snakemake know to include the new z-frames?": the z_stack
     plan's content/mtime changes → that one product shard rebuilds even under mtime-only triggers; no
     per-well plan, so the old "well run owns all products" idea cannot creep back in. No need to keep
     projection in config forever; no projection rerun; no silent deletion. Config determines which
     product shards to build in this run; discovery determines which validated product shards are
     included when assembling the canonical well frame_inventory.

     B3 — Product discovery manifest (spec §4)

     - New per-well manifest discovered_product_shards/per_well/{well_id}/{well_id}_discovered_product_shards.csv
     with columns well_id, product_key, frame_inventory_csv, validated_flag.
     - Implementation decision: the discovery rule receives the validated sentinels for the products
     requested in the current invocation as explicit Snakemake inputs. The requested product
     .validated sentinels are DAG triggers, not discovery limits. Discovery still scans the full
     product-shard directory and emits all currently validated product shards for that well, including
     older products not requested in this invocation.
     - Discovery scans the product-shard dir, includes ONLY shards with a valid .validated sentinel,
     ignores unvalidated CSVs / orphan PNGs / current config, fails loud on duplicate product_key or
     malformed shard names. MVP: a plain discovery script feeding the assembler (no Snakemake checkpoint
     unless dynamic inputs force it — spec §4).
     - This gives both concrete DAG triggering for newly materialized products and persistent memory
     semantics through directory discovery. Without the requested product .validated sentinels as
     explicit inputs, the manifest can go stale because directory mtimes are not reliable Snakemake
     dependencies.

     B4 — Well assembler → canonical per-well frame_inventory (spec §3)

     - New assemble_well_frame_inventory step: read all discovered validated product shards for the well,
     concatenate them → canonical {well_id}_frame_inventory.csv, then validate per-well (strict),
     then the existing merge_frame_inventory + merged validate run unchanged on top. Reuse the existing
     concat/merge code path (merge-frame-inventory task) for the union if shapes allow.
     - Consequence (spec §1, §11): requesting z_stack-only on a well that already has a validated projection
     shard still yields a canonical well inventory = projection + z_stack (config is instruction, shards
     are memory).

     B5 — Retirement is explicit (spec §5) + stale files inert (spec §7)

     - Absence from today's config does NOT remove a product. MVP retirement = remove the product
     shard's .validated sentinel. No retired/ move, no retire-product CLI verb, no candidate/promote
     flow in MVP. After removing the sentinel, rerun the relevant well/experiment frame_inventory target
     so discovery and assembly refresh.
     - Stale PNGs are allowed/inert in MVP: downstream trusts validated frame_inventory, never globs the
     image tree — confirm no downstream stage directory-walks (detection/SAM read source_image_path
     from the inventory). A materializer may clear ONLY its own product output dir before writing; never
     broad-delete a well dir. Candidate→validate→promote is deferred (spec §6).

     ---
     Verification (real data)

     1. Unit/contract tests (no GPU; PYTHONPATH=src, --import-mode=importlib,
     conda run -n segmentation_grounded_sam): identifiers (both parsers, byte-for-byte regression),
     frame_inventory_contract (the 4 HARD-GATE duplicate cases), resolver, paths, executor (mocked ND2),
     detection focus_stack projection-filter, and focus_index_map_path L4/provenance checks. The four HARD-GATE contract tests GREEN before A2–A4 are touched.
     2. Real-ND2 z_stack smoke (CPU, here). z_stack needs no GPU. Committed overlay
     config_smoke_zstack_20250912.yaml (B01/C01, smoke_max_time_indices: 3, products incl. z_stack;
     carries products/wells/cap ONLY — the ND2 path flows via source_nd2_path in the acquisition
     inventory, not baked in). Assert: a PNG per (t,z) under
     …/BF/z_stack/{well_id}_BF_z{:04d}_t{:04d}.png; the product shard / assembled well inventory has
     n_z rows per time_index with distinct z-aware image_ids; validation passes.
     3. Rebuild + assembly check. Run B01 projection-only. Assert the projection product shard exists
     and the assembled well inventory contains projection rows. Then run B01 z_stack-only. Assert the
     z_stack product shard is created, the discovery manifest lists projection + z_stack, and the
     assembled well inventory contains projection + z_stack. Then run B01 z_stack-only again after
     deleting/replacing z_stack output. Assert only the z_stack product shard is regenerated, the
     projection shard remains untouched, and the assembled well inventory still contains projection +
     updated z_stack.
     4. Combined projection+z_stack smoke (GPU via qsub). Submit front_half with both products
     (qsub -l gpgpu=TRUE,cuda=1); resolve_device("auto") announces -> CUDA; detection (if run)
     emits rows only for BF focus_stack projection frames.

     Build order & staged commits (DECIDED: Part B before Part A's pixels)

     The HARD GATE still rules: the contract (A0–A1) must land + go green BEFORE any z_stack is emitted —
     and Part B's product shards need the z-aware image_id to express z_stack rows anyway. So the contract
     comes first, THEN the product-shard architecture (validated on projection-only first), THEN the z_stack
     capability emits into that architecture from day one.

     1. Commit 1 — z-aware identity grammar + contract tests (A0, A1, A1b + both parsers + the 4
     HARD-GATE duplicate cases). No emitter touched. GREEN before moving on.
     2. Commit 2 — product-shard architecture on projection (Part B: B1 shard paths, B2 per-(well,
     product) rule + resolved-product-plan rebuild trigger, B3 discovery manifest, B4 assembler, B5
     inert-stale/sentinel-retirement). Wire it for the EXISTING projection product only and prove the
     discover→assemble→merge chain reproduces today's per-well frame_inventory. Smoke stays projection.
     3. Commit 3 — open z_stack materialization into the product-shard model (A2 resolver, A3 path,
     A4 executor emitting a z_stack product shard) + the z_stack smoke overlay + the rebuild/assembly
     real-data check (projection-only → +z_stack → z_stack-only). z_stack needs no GPU.
     4. Commit 4 — detection restricted to BF focus_stack projection frames (A4b + mixed-inventory test)
     and focus_index_map_path L4/provenance validation. Required before any mixed
     (projection+z_stack) inventory reaches segmentation.

     Doctrine

     Config controls requested work. Resolved product plans trigger requested product jobs. Product
     shards store completed products. Discovery finds active validated product shards. The well
     frame_inventory is assembled from discovered shards. The experiment frame_inventory is assembled
     from validated well inventories. One frame_inventory row names one materialized image identity; source_image_path points to the
     primary pixel file, and validated construction-provenance paths may ride with that image. Never patch the river — rebuild it from trusted springs.
