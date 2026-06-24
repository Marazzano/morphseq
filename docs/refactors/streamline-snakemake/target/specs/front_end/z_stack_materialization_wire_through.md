# Z-Stack Materialization Wire-Through (🟡 PLANNED)

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
| **Path layout** | `image_materialization/materialized_image_paths.py` | ⚠️ the generic `materialized_image_path` already builds the `z_stack/` filename (`{well_id}_{channel_id}_z{z:04d}_t{t:04d}`), but the named `z_stack_frame_path` wrapper is **deferred** (line 173). |
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
compound key), and the z-aware filename already exists in the layout.

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
> across. The frame inventory still has one row per materialized PIXEL FILE; a z_stack plane just
> happens to be one such file. This is consistent, not a violation: the rule was "don't collapse
> P/Z into the projection key," and a z_stack frame is genuinely a distinct image.

---

## The build order — FIVE edits + tests (contract-first, blast radius honest)

```text
0. shared identifiers   build_image_id(..., z_index=None)      ← upstream shared API
1. frame_inventory      derive_image_id + nullable z key + assert_derived_ids_consistent
                        contract                                  (the load-bearing edit)
2. resolver             accept YX1 z_stack
3. paths                z_stack_frame_path wrapper
4. executor             branch projection vs z_stack
5. tests                shared identifier + contract + resolver + path + executor
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

### 3. Path layout — add the named `z_stack_frame_path` wrapper
`materialized_image_paths.py` (the deferred line 173): add the symmetric wrapper that fills
`image_product_type="z_stack"` and **requires** `z_index`. The generic constructor already does
the work; this is just the named call site, mirroring `projection_frame_path`.

### 4. Executor — branch per product type in `materialize_yx1_well`
The real work. Today the loop is hardcoded: per `time_index`, focus-stack the whole Z-stack into
ONE projection frame. Make it **iterate `resolved_plan.products`** (it currently ignores them):

- **`projection` product** → unchanged: `materialize_ff_projection(stack)`, one row,
  `z_index=pd.NA`, `image_product_type="projection"`, `projection_method="focus_stack"`.
- **`z_stack` product** → **do NOT project.** The `stack_zyx` from `_get_stack(...)` already has
  every plane. Iterate the `z_index` values **from the well's inventory rows** (the inventory is
  the system of record for which Z planes exist — do not infer from array shape), write one PNG
  per `(time_index, z_index)` via `z_stack_frame_path`, and emit one frame-inventory row per
  plane: real `z_index`, `image_product_type="z_stack"`, `projection_method=pd.NA`, z-aware
  `image_id`.

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

**Path:** `z_stack_frame_path` lands under `z_stack/{channel_id}/`, filename carries
`z{:04d}_t{:04d}`; missing `z_index` raises; projection wrapper unchanged.

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

Both products can be requested in one well job; the executor writes the projection AND the full
z-stack from the same in-memory `stack_zyx`.

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

## 🪧 Open (carried)
- Whether `max`/`mean` projection methods land at the same time as z_stack (same `_get_stack`
  source; trivial once the executor branches) or stay deferred with GFP.
- Keyence z_stack: its acquisition inventory also explodes Z, so the same contract change unlocks
  it — but the Keyence backend (`keyence_wire_through.md`) must exist first.
- Downstream cost: a full z_stack is `n_z ×` the projection frame count on disk; confirm a
  retention/cleanup policy before enabling z_stack on a full plate (vs. the 2-well smoke).
