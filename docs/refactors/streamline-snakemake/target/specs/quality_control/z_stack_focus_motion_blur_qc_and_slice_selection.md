# Z-Stack Focus QC + Motion-Blur QC + Z-Slice Selection (🟡 PLANNED)

**Status:** planning spec, mdcolon 2026-06-25. Companion to the LOCKED
`front_end/z_stack_materialization_wire_through.md`. That doc ships the **pixels** (one PNG per Z
plane + z-aware `frame_inventory` rows). This doc says what we do with those planes:
**focus QC**, **motion-blur QC**, and **z-slice selection** — and, critically, **what we persist vs
what we compute transiently**.

**No code in this doc** — it is the architecture + wire-through map.

---

## 🪨 The one-sentence answer

The z_stack PNGs (already specced) are the only durable pixel artifact we need; we add **one small
z-depth provenance extension** to the acquisition inventory plus **one materialization-derived sidecar
— the focus-stacker's per-pixel chosen-Z map (`focus_index_map`), which it already computes during the
FF projection pass and currently discards** — so slices can be chosen later; and **focus QC and
motion-blur QC recompute
their grids transiently from the inventory-addressed PNGs and persist the per-snip metric summary +
flag**, plugging into the `snip_qc` hooks (`focus` reserved; `motion_blur` added).

---

## 🧭 Load-bearing doctrine (the user's correction, stated as law)

```text
Pixels are the primary artifact.
Persist as a primary artifact ONLY what is reusable beyond QC.
If a feature's only purpose is QC, generate it AT QC TIME from the primary artifacts
  and keep only the verdict.
Selection is not QC — it needs INFORMATION recorded, not a verdict.
```

Consequence: there is **no persistent grid product** (no `feature_extraction/z_stack_grid_metrics/`,
no pre-seg `zstack_grids` stage — both were drafted and **rejected**). Laplacian/entropy/NCC grids are
QC-only derivations computed inside the QC job from the inventory-addressed z_stack PNGs and thrown
away. What persists is: the **pixels** (LOCKED), the **focus_index_map** (the focus-stacker's chosen-Z,
captured during materialization — the one derived fact that is lost if not caught in the FF pass), the
**z-depth provenance**, and the **per-snip QC summary** (metrics + flag, not flag alone).

> If a future analysis needs a z-stack-derived *feature* (not a QC flag), that is the moment to
> promote a grid to a real `feature_extraction/` product — not before. Until then, QC owns its own
> transient math, and the only durable derived artifact is the focus_index_map.

---

## The three concerns, separated

| Concern | Persisted? | Grain | Where |
|---|---|---|---|
| **z_stack pixels + FF** | ✅ durable primary | one PNG per `(well,channel,z,t)` / FF per `image_id` | `image_materialization/.../z_stack/` (LOCKED elsewhere) |
| **focus_index_map** | ✅ durable derived | one map per `image_id` (pixel→Z) | materialization product shard (the focus-stacker's `idx`) |
| **z-depth provenance** | ✅ durable primary | one row per acquisition coordinate | acquisition inventory (the GAP — added here) |
| **focus QC** | summary (metrics + `focus_flag`) | one row per `snip_id` | `quality_control/focus_qc/` |
| **motion-blur QC** | summary (metrics + `motion_blur_flag`) | one row per `snip_id` | `quality_control/motion_blur_qc/` |
| **z-slice selection** | selection output (not a flag) | one row per snip/embryo | downstream consumer, NOT QC |

---

## Part A — The one durable addition: z-depth provenance

> **Order of work (the standing rule): YX1 FIRST, then Keyence.** Part A1 wires z-depth on the YX1
> side (clean — the ND2 hands it to us). Part A2 mirrors it on Keyence, where the metadata is
> proprietary and the same fact is NOT free. Do not start A2 until A1 ships and the YX1 z_stack QC
> chain is green. The two scopes share ONE inventory contract shape (`z_position_um` / `z_step_um`
> columns) and ONE downstream consumer — only the per-scope *extraction* differs.

### A1 (YX1) — Why
The z_stack `frame_inventory` already carries `(image_id, z_index, time_index, source_image_path)`
per plane (LOCKED). That is enough to *order* planes but not to anchor selection to **physical
focus depth**. The user chose "add it now": record `z_position_um` (per plane) and `z_step_um`
(per acquisition) so z-slice selection can rank by depth later without re-opening the ND2.

### A1 (YX1) — Verified: the ND2 exposes both, and the read is nearly free
- `nd.voxel_size()` is **already called** at
  `metadata_ingest/scope/yx1/extract_yx1_scope_metadata.py:159` (`micrometers_per_pixel =
  voxel_size[0]`). **`voxel_size()[2]` is the Z spacing** → `z_step_um`, free.
- The stage read at `extract_yx1_scope_metadata.py:191-206` already reads
  `ch0.position.stagePositionUm` for `.x`/`.y`. **That same object carries `.z`.** Today the loop
  reads only the **first plane** per position (`idx = w_idx * n_z * max(n_c,1)` → z=0). To capture
  **per-plane** Z it must iterate planes within a position (`idx + z*n_c`) and read `stage.z`.

### A1 (YX1) — The edits (acquisition side — system of record for z facts)
1. **`extract_yx1_scope_metadata.py`** — read `z_step_um = voxel_size()[2]`; extend the stage loop
   to read `stage.z` per `(position, z_index)` instead of only the first plane. Pass per-plane
   `z_position_um` and the scalar `z_step_um` into the inventory builder.
2. **`scope/yx1/acquisition_inventory.py`** — add `z_position_um` and `z_step_um` to
   `YX1_ACQUISITION_INVENTORY_SCOPE_COLUMNS` (sits beside `z_index`, `n_z`, `x_um`, `y_um`). Carry
   them through `build_yx1_acquisition_inventory_rows`. Validate `z_step_um` positive; allow
   `z_position_um` NaN per-row only if the ND2 truly lacks it (don't fail loud on missing optics).
3. **No frame_inventory change for selection** — selection joins the z_stack `frame_inventory`
   (which has `z_index`) to the acquisition inventory (which now has `z_position_um`) on the
   acquisition key. The frame_inventory stays one-row-per-pixel-file; z-depth is acquisition
   provenance, not frame identity (consistent with the LOCKED boundary guard: don't drag raw-unit
   provenance across the stitched seam — selection reads it from the acquisition side directly).

> Boundary honesty: `z_position_um`/`z_step_um` are **acquisition facts**, so they live in the
> acquisition inventory next to `n_z`/`x_um`/`y_um` — NOT in QC, NOT minted into `image_id`.

> **Semantics (document at the column, not just here):** `z_position_um` is the **absolute stage Z
> position** of the plane (machine coordinate) — and `z_step_um` is the **acquisition spacing**
> between planes. Neither is "biological focus depth" or embryo-relative depth: an embryo's in-focus
> plane is wherever its tissue sits in the stage range, not a fixed `z_position_um`. So these fields
> give **ordering + physical spacing + a stable coordinate** for selection; turning them into
> embryo-relative depth (if ever needed) requires normalizing against a per-embryo reference plane —
> that is a selection-time concern, not an acquisition fact.

### A2 (Keyence) — same columns, harder extraction (do AFTER A1 is green)

Keyence already explodes Z in its acquisition inventory exactly like YX1 — raw Keyence data IS
per-Z-plane TIFFs (`...Z###_CH#.tif`), one row per `(well, tile, z_index, channel_index, time)` in
`scope/keyence/acquisition_inventory.py` (`z_index` is already a column, line 72). So the **same z_stack
materialization contract change unlocks Keyence z_stack** — but only once the Keyence materialize
backend exists (gated by `front_end/keyence_wire_through.md`). The z-depth provenance is the part that
does NOT transfer cleanly:

- **`z_index` is reliable** — it is parsed from the filename `Z###` token, not the XML.
- **`z_position_um` / `z_step_um` are NOT free, and may be absent.** Unlike YX1's `nd.voxel_size()[2]`
  + structured `stagePositionUm.z`, Keyence pulls everything from the proprietary BZ-X XML scrape
  (`extract_scope_metadata.py:_scrape_keyence_metadata`), which today extracts ONLY
  `ShootingDateTime / LensName / Observation Type / Width / Height` — **no Z stage position at all**
  (the legacy extractor even hardcoded `z_position: 0`). The module's own docstring repeatedly warns
  the Keyence XML is proprietary/unreliable.

**Decision (Keyence z-depth): investigate, then degrade gracefully — do NOT block z_stack on it.**

1. **Scraper investigation (small, first):** open a real BZ-X TIFF's `<Data>` XML and look for a
   per-plane Z stage key (candidates: a `Z`/`Focus`/`ZPosition`-like tag, or a Z pitch in the
   acquisition block). If a reliable key exists, add it to `_scrape_keyence_metadata`'s keyword list
   and surface `z_position_um` through `_scrape_keyence_plane_metadata` (the same per-plane scraper
   seam the inventory already calls at `acquisition_inventory.py:294`).
2. **`scope/keyence/acquisition_inventory.py`** — add `z_position_um` / `z_step_um` to
   `KEYENCE_ACQUISITION_INVENTORY_SCOPE_COLUMNS`, mirroring the YX1 columns so the **inventory
   contract shape is identical across scopes** and the downstream selection join is scope-agnostic.
3. **If the XML has no usable Z depth:** record `z_position_um = NaN` and derive `z_step_um` per well
   from the spacing of consecutive `z_index` values only if a pitch is scrapeable; otherwise leave
   `z_step_um` NaN too. **Keyence then falls back to `z_index`-ordering selection** (rank planes by
   computed sharpness over the ordered `z_index`) — strictly weaker than YX1's physical-depth
   selection, but honest. Do NOT fabricate a depth; do NOT fail loud on missing optics (same
   tolerance as the YX1 contract: `z_step_um` positive-or-NaN, `z_position_um` NaN-allowed).

> Doctrine for the asymmetry: **the contract shape is shared; the provenance quality is not.** YX1
> gets physical-depth selection; Keyence gets at-least-ordering selection until/unless the BZ-X XML
> proves it carries Z. Focus + motion-blur QC (Part B) need only the pixels + `z_index` ordering, so
> they work on BOTH scopes regardless of whether `z_position_um` is recoverable.

---

## Part B — Blur QC + Motion QC (verdicts; grids transient)

Both mirror the existing `quality_control/mask_quality_qc/` module shape **exactly** — the proven
template: `contract.py` (schema), `compute.py` (domain math), `config.py` (thresholds/params),
`entrypoint.py` (thin filesystem adapter). Grain: **one row per `snip_id`**, spine imported from the
minting site, validated against the `physical_embryo_registry` as verifier (`check_sources=True`),
written as a per-well shard. (See `mask_quality_qc/contract.py` and `entrypoint.py` for the pattern.)

### The compute flow (the LOCKED MVP design — focus-index map during materialization; QC grids transient)

> **Correction (LOCKED, supersedes the earlier "persist the grid" draft):** there is **no persisted
> grid stage.** The only durable materialization-derived addition is a **per-pixel focus-index map**
> (per `image_id`: "this FF output pixel was taken from Z plane k"). It is created **during the FF
> projection pass** because the focus-stacker ALREADY computes it and currently throws it away. Given
> the pixels (z_stack PNGs) + the focus-index map, both selection and focus/motion-blur QC are
> computable by **reading `frame_inventory` at QC time** — so the grids QC needs are recomputed
> transiently from the persisted pixels, never stored.

**Verified — the focus-index map already exists, free:** `LoG_focus_stacker`
(`image_building/shared/log_focus.py:76-77`) does `_, idx = abs_log.max(dim=1); ff = data.gather(1, idx…)`.
`idx[y,x]` IS the per-pixel argmax-over-Z — the focus-index map. Today only `ff` is saved; `idx` is
discarded. Persisting `idx` beside the FF is the whole materialization-side change.

```text
[MATERIALIZATION — one focus-stack call, FF + z_stack generated TOGETHER (MVP constraint)]
   z_stack PNGs (per image_id, per Z)        ── primary pixels      ┐ same LoG_focus_stacker call
   FF projection PNG (per image_id)          ── primary pixels      │ (idx is already computed there)
   focus_index_map (per image_id pixel→Z)    ── DURABLE sidecar     ┘
        │
        │   …… segmentation boundary (frame_masks / snip_inventory) ……
        ▼
[QC — reads frame_inventory, loads the pixels it points at, recomputes per image_id]
   focus_qc / motion_blur_qc: load z_stack/FF via frame_inventory → compute grid IN MEMORY
                  → reduce in embryo mask → per-snip SUMMARY (metrics + flag)   (grid NOT stored)
   z-slice selection: read focus_index_map (+ z-depth) → choose plane(s)        (no grid stored)
```

Why this is right (and why the grid-stage draft was wrong): the grid QC needs is **cheap to recompute
from the persisted pixels** at QC time, and QC already has a `frame_inventory` row pointing at every
z_stack/FF PNG — so there is no cross-stage in-memory loss problem (each QC job loads its own pixels
and computes its own grid in that same job). The ONE thing that is genuinely expensive-once and lost
forever if not captured in the FF projection pass is the focus-stacker's `idx` — so that, and only
that, is persisted. Grids stay transient *because the pixels they derive from are durable and inventory-addressable.*

> **MVP constraint (LOCKED):** FF projection and z_stack must be materialized **together** (one
> focus-stack pass), so the focus-index map is emitted in the same call that produces both. Do not
> build a separate pass that re-reads the ND2 just to recover `idx`.

> **Two DISTINCT QC modules** (not one): `focus_qc` (within-image sharpness / in-focus) and
> `motion_blur_qc` (inter-slice motion blur, NCC). Each is its own per-well shard with its own flag.

### focus_qc — within-image sharpness / in-focus (the user's real want, "more like blur")
- **Inputs:** the z_stack/FF PNGs **addressed via `frame_inventory`** + the snip's embryo mask (from
  `frame_masks`/`snip_inventory`). QC loads the pixels and computes the sharpness grid in-memory.
- **Compute:** Laplacian/entropy per plane (transient grid), reduced inside the mask.
- **Verdict:** `focus_flag` — true when the *selected/representative* slice is too soft (threshold in
  `focus_qc/config.py`). Uses the already-reserved `"focus" -> focus_flag` snip_qc hook.
- **Output:** `quality_control/focus_qc/per_well/{well_id}/{well_id}_focus_qc.csv` — the per-snip QC
  **summary**, metrics AND flag (a flag without metrics is a black box). Persist:
  ```text
  snip_id (+ spine)
  focus_score         # the threshold-driving scalar
  laplacian_score
  entropy_score
  n_mask_pixels
  n_valid_z
  focus_flag
  ```
  This is a QC summary product, NOT a feature product.

### motion_blur_qc — inter-slice motion blur (NCC)
- **Inputs:** the z_stack PNGs **addressed via `frame_inventory`** + the snip mask.
- **Compute:** NCC between adjacent planes (transient grid), reduced in the mask:
  `ncc_min`, `bad_pair_frac`, etc. (the metrics validated in the
  `results/mcolon/20260421_motion_artifact_detection/` reference work — that work is **reference,
  not code to wire**; reimplement cleanly in `motion_blur_qc/compute.py`).
- **Verdict:** `motion_blur_flag` — true on excessive inter-plane drift. Uses a NEW
  `"motion_blur" -> motion_blur_flag` snip_qc hook.
- **Output:** `quality_control/motion_blur_qc/per_well/{well_id}/{well_id}_motion_blur_qc.csv` — the
  per-snip QC **summary**, metrics AND flag. Persist:
  ```text
  snip_id (+ spine)
  ncc_min
  ncc_p05
  bad_pair_frac
  ncc_bad_tile_frac
  longest_bad_run
  n_tiles
  motion_blur_flag
  ```
  A QC summary product, NOT a feature product.

### Plug into snip_qc (the hooks already exist)
`quality_control/snip_qc/contract.py:34-37` already reserves a `"focus" -> focus_flag` hook
("after focus_qc; z-stack, in dev"). `focus_qc` uses it directly. (The doc's stale `"blur"` hook line
is the same concept — fold it into `focus`, don't ship two.)
- Add `"motion_blur" -> "motion_blur_flag"` as a NEW hook for `motion_blur_qc`.
- When `focus_qc`/`motion_blur_qc` ship and emit their named flag columns, register the reasons in
  `SNIP_QC_EXCLUSION_REASONS` and feed the shards into `snip_qc/inputs.py`. `snip_qc` already ORs
  components into `use_snip` — no new verdict logic, just one more flag source each.
- Do **not** register a hook before its product emits the column — `snip_qc` fails loud on a missing
  column **by design** (that guard is already in the contract).

---

## Part C — Z-slice selection (NOT QC — a downstream selection need)

Selection answers "which Z plane(s) hold the in-focus embryo?" It is a **selection**, not a verdict,
so it does not emit a `*_flag` and does not gate `use_snip`.

- **Inputs (all primary / already-recorded):**
  - z_stack `frame_inventory` rows (`z_index`, `image_id`, `source_image_path`) — LOCKED.
  - the acquisition **focus_index_map** (per `image_id`, per-pixel → Z) — the durable bit from the
    focus-stacker `idx`. This directly answers "which Z plane is in focus" per pixel; reduce it in the
    embryo mask (e.g. modal/median Z) to pick the snip's plane(s).
  - acquisition `z_position_um` / `z_step_um` — Part A (physical depth/spacing if a finer rank needed).
- **Output:** a selection table (e.g. `top1/top2/top3_z_index`, scores, `top1_top2_margin`,
  `n_valid_z`) at snip/embryo grain, under a **selection** home — NOT `quality_control/`. Candidate
  homes (decide when selection is built): `image_selection/z_slice_selection` if it writes only a
  choice table, or `image_materialization/z_slice_selection` if it also materializes selected images.
  Out of scope for the focus/motion-blur QC pass.

> Selection is listed here only to prove the recorded INFORMATION is sufficient: z_stack pixels +
> focus_index_map + z-depth ⇒ any selection algorithm is computable later without re-opening the ND2
> and without a persisted grid.

---

## Part D — Wiring into the DAG (no grid stage; just a materialization sidecar + ordinary QC)

Because there is **no persisted grid product**, the awkward "pre-seg, frame-grain product threaded
through snip-grain QC plumbing" problem disappears. Only two DAG facts remain:

**1. Acquisition emits one more output: the focus_index_map (per `image_id`).** It rides the LOCKED
z_stack/projection materialization — same `materialize_yx1_product_for_well` call, same per-well
product shard. The focus-stacker already computes `idx`; the executor just writes it beside the FF as
**`.npz`** (`…/focus_index_map/BF/{well_id}_BF_t{:04d}.npz`, each pixel → its Z index) and records it
in the product frame_inventory shard (a `focus_index_map_path` column on the projection row, since it
pairs with the FF). **FF + z_stack + focus_index_map all come from the one focus-stack pass** — this
is why the MVP requires them generated together.

**2. focus_qc / motion_blur_qc are ORDINARY post-seg per-well QC modules** — clone `mask_quality_qc.smk`
verbatim (build→validate→merge, `PATH_MODE_PER_WELL`, registry as verifier). They consume exactly
what `mask_quality_qc` consumes (validated `snip_inventory` + `frame_masks` + `registry`) **plus the
`frame_inventory`** (to address the z_stack PNGs to load). No new wildcard grain, no grid manifest,
no checkpoint — the segmentation-boundary ordering is already carried by depending on the validated
`snip_inventory`/`frame_masks`, exactly like every existing QC module.

```text
[ACQUISITION, per well — LOCKED z_stack/projection pass + ONE new output]
materialize_yx1_product_for_well → FF + z_stack PNGs + focus_index_map   (one focus-stack call)
        ▼
        …… segmentation / snip_inventory / frame_masks (the existing boundary) ……
        ▼
[QC, per well — clone of mask_quality_qc, +frame_inventory input]
build_focus_qc_for_well / build_motion_blur_qc_for_well
   → load z_stack PNGs via frame_inventory, compute grid in-memory, reduce in mask
   → per-snip summary shard → validate (registry verifier) → merge
```

This is why "it doesn't sing" was a symptom of the wrong design: a persisted frame-grain grid forced
a new product grain into snip-grain plumbing. Removing the grid removes the mismatch. The single
genuinely new artifact (focus_index_map) is **frame-grain by nature and lives where frame-grain pixels
already live — the materialization product shard** — so it never touches the snip-grain QC template at
all. The only schema decision left: the focus_index_map is a per-`image_id` companion to the FF, so
record it on the **projection** frame_inventory row (a `focus_index_map_path` column), not as a new
row grain.

---

## Part E — Where the artifacts actually live on disk (the real paths)

Nothing here invents a path. Two existing machineries own the two artifact kinds; we add rows/files to
them. `DATA_ROOT` itself is machine-specific (`env.yaml` / `output_root`), so paths are always
RELATIVE to it — never hardcode `DATA_ROOT`.

### 1. focus_index_map — a PIXEL-adjacent artifact (materialized-images tree)

The focus_index_map is written by the materialize executor beside the FF, so it lives in the **pixel
tree**, NOT under `quality_control/`. The locked materialized-images layout
(`image_materialization/materialized_image_paths.py`, layout LOCKED 2026-06-17) is:

```text
built_image_data/{experiment_id}/materialized_images/{well_id}/
    projection/{channel_id}/{image_id}.png          ← FF (LOCKED, image_product_type)
    z_stack/{channel_id}/{well_id}_{channel_id}_z{z:04d}_t{t:04d}.png   ← z-planes (LOCKED, image_product_type)
    focus_index_map/{channel_id}/{image_id}.npz     ← NEW: SIDECAR folder (NOT an image_product_type)
```

**SIBLING, not nested (LOCKED).** `focus_index_map/` is a **top-level sibling** under
`materialized_images/{well_id}/`, NOT nested under `projection/`. The reason is an ontology rule:
**the first folder level is the `image_product_type` (product SHAPE), not product ownership.**
`projection/` and `z_stack/` are image product shapes; `focus_index_map` is neither — it is a
materialization sidecar. Nesting it as `projection/focus_index_map/...` would make `projection/` mean
two things (the FF image product AND a sidecar explaining its construction) — a semantically leaky
path grammar. Keep product-shape folders clean; express ownership in the table, not the tree.

Explicit contract for `focus_index_map/`:
```text
focus_index_map/ is a MATERIALIZATION SIDECAR folder, not an image_product_type.
  • It does NOT get a product_key.
  • It does NOT get its own frame_inventory row.
  • It does NOT get its own .validated sentinel.
  • Ownership is expressed by focus_index_map_path on the projection frame_inventory row.
  • Lifecycle/staleness is guarded by the projection product validator (Part F, Option A).
```

- Same one-per-`image_id` grain as `projection/`. Add a `focus_index_map_path(...)` wrapper in
  `materialized_image_paths.py` mirroring `projection_frame_path` — but it is a SIDECAR path builder,
  NOT a new `image_product_type` (do not add it to `ALLOWED_IMAGE_PRODUCT_TYPES`).
- It inherits the LOCKED `candidate/` isolation (candidate runs can't overwrite the live tree) for
  free, since it's the same path constructor.
- The frame_inventory's projection row gains a `focus_index_map_path` column pointing here (the path
  is recorded in the table; downstream reads the column, never globs the tree).

> **Doctrine:** *Ownership lives in the table. Shape lives in the path. The validator ties them
> together.* A sidecar rides with projection — it is not a new kingdom beside it.

#### focus_index_map and the `product_key` scheme (LOCKED: companion, NOT its own product)

The image-materialization layer keys each product on a `product_key`
(`image_materialization/image_product_keys.py`, grammar
`{channel_id}__{image_product_type}[__{projection_method}]`; today only `projection` and `z_stack`).
Product **shards** (per-well frame_inventory) are minted one-per-`product_key`.

**The focus_index_map does NOT get its own `product_key`.** It is a **companion of the projection
product** (`BF__projection__focus_stack`) — born in the *same* `LoG_focus_stacker` call that makes the
FF (`idx` and `ff` come out together). So:

- **No new `image_product_type`, no grammar change** to `build_image_product_key` — it stays
  `{projection, z_stack}`. The focus_index_map rides the existing projection product.
- It is recorded as a **`focus_index_map_path` column on the projection frame_inventory row**, NOT as
  a new row and NOT as a new shard. "One frame_inventory row = one materialized pixel file" still
  holds: the projection row's pixel file is the FF; the `.npz` is a sibling FIELD of that row (like a
  per-row provenance path), not a second pixel-file row.
- Therefore the focus_index_map shares the projection product's lifecycle for free: same shard, same
  `.validated`, same discovery/assembly, same retirement. Rerunning the projection product rewrites
  the FF *and* the focus_index_map together (correct — they're one computation).

> Why not `BF__focus_index_map` as its own product_key? That would expand the materialization grammar
> (a third `image_product_type`), add a shard + discovery entry, and falsely frame an inseparable
> by-product of focus-stacking as an independent product. The companion-column model keeps the grammar
> at two product types and ties the map to the exact computation that creates it.

> **Note — QC products have NO `product_key`.** `product_key` is image-materialization-internal
> vocabulary. `focus_qc`/`motion_blur_qc` are keyed by the `PIPELINE_STEPS` `step`/`product_dir`
> (below), the same as every other QC product — they never touch the `product_key` grammar.

#### ⚠️ focus_index_map in the DAG — staleness & validation (THE LIKELY BREAK) [Part F]

"Rides the projection product" is precise about ownership but hides two DAG hazards. Both must be
closed or the `.npz` rots silently. The projection product's real DAG (verified in
`rules/materialize_well_native.smk` + `frame_inventory.smk`) is:

```text
write_resolved_product_plan_for_well       → {product_key}_resolved_product_plan.json
materialize_image_product_for_well         → {well}_{product_key}_frame_inventory.csv   (the SHARD)
validate_frame_inventory_product_for_well  → {…}.csv.validated   (runs --check-sources=true)
discover_product_shards_for_well           → discovered_product_shards.csv
assemble_well_frame_inventory              → canonical {well}_frame_inventory.csv
validate_frame_inventory_for_well          → {well}.materialize_well.validated
```

The focus_index_map is born inside `materialize_image_product_for_well` (the `idx` from the same
focus-stack call) and recorded as a `focus_index_map_path` column on the projection shard row. It does
**NOT** get its own `.validated` — it rides the projection shard's existing two sentinels. That is the
intent, but it only works if BOTH of these change:

**HAZARD 1 — Snakemake does not track the `.npz` as an output (staleness break).**
`materialize_image_product_for_well` declares ONE output: the shard CSV. The `.npz` is a *second file*
the rule writes that Snakemake doesn't know about. If the CSV survives but the `.npz` is deleted/
corrupted, Snakemake sees the shard up-to-date and **will not rerun** → a dangling
`focus_index_map_path`.
- **Fix (LOCKED — Option A, validator-as-guard).** The `.npz` is NOT declared as a Snakemake output.
  Declaring N per-`image_id` `.npz` files as outputs would fight the product-shard model (the rule's
  grain is the shard, not the image, and the file set is dynamic). Instead, **HAZARD-2's source check
  IS the staleness guard**: a missing/corrupt `.npz` fails `validate_frame_inventory_product_for_well`,
  so its `.csv.validated` is never (re)written, and the downstream DAG re-fires. Staleness is enforced
  by the *validated sentinel*, exactly like every other source-file guarantee in this pipeline ("the
  inventory is truth; downstream trusts validated, never globs"). This is WHY Hazard 2's validator
  extension is mandatory — under Option A it is the only thing protecting the `.npz`.
  - *(Rejected — Option B: a tracked per-well `.npz` manifest as an extra rule output. It would add a
    Snakemake-tracked file but duplicate what the shard column already says, and add a second source of
    truth for the same fact. Not pursued.)*

**HAZARD 2 — the source validator only checks `source_image_path` (silent-missing break).**
`--check-sources=true` → L4 (`frame_inventory_validation_rules.py:227`) loops rows and validates ONLY
`source_image_path` (resolve + open + dims + µm/px). It knows nothing about `focus_index_map_path`. So
a missing/corrupt `.npz` **passes validation today** → the `.validated` sentinel lies.
- **Fix:** extend the L4 source check so that, on a **projection** row that carries a non-null
  `focus_index_map_path`, it ALSO resolves that path and asserts the `.npz` exists + loads (and, ideal,
  that its `idx` array shape matches the FF's `image_height_px × image_width_px`). z_stack rows have no
  such column → skipped. This is the change that makes the projection `.validated` honestly cover the
  companion file, which is the *entire* justification for not giving the `.npz` its own sentinel.

**Staleness/update semantics once both fixes land (inherited from the projection product, for free):**
- Rerun the projection product (`materialize_image_product_for_well`) → FF **and** `.npz` rewrite
  together (one computation), shard CSV rewrites, its `.csv.validated` is re-earned (now also checking
  the `.npz`), discovery + assembly + per-well validate re-fire downstream. Correct.
- Retire the projection product (remove its `.csv.validated`) → the `.npz` retires with it (no separate
  sentinel to orphan). Correct.
- A stale `.npz` with no CSV row is inert (downstream reads the column, never globs) — same MVP
  stale-file tolerance as stray PNGs.

> **Net:** the companion model is sound ONLY with the L4 validator extension (Hazard 2). Without it,
> "rides the projection `.validated`" is a false promise. Make the L4 `.npz` check a HARD GATE of this
> work, shipped in the SAME change that adds the `focus_index_map_path` column — never let the column
> exist before the validator polices it (mirrors the z_stack contract's "never let pixels outrun the
> contract").

### 2. focus_qc / motion_blur_qc — CSV products (quality_control stage, registry rows)

These are ordinary per-well-then-merge CSV products, so they go through the `PIPELINE_STEPS` registry
in `pipeline_orchestrator/orchestration/paths.py` — **two new rows, byte-identical in shape to the
existing `mask_quality_qc` row** (`stage: quality_control`, `fanout: PER_WELL_THEN_MERGE`,
`execution: EXECUTION_PER_WELL`):

```python
"focus_qc": {
    "stage": "quality_control",
    "product_dir": "focus_qc",
    "fanout": PER_WELL_THEN_MERGE,
    "execution": EXECUTION_PER_WELL,
    "artifacts": {"focus_qc": {
        PATH_MODE_PER_WELL: "{well_id}_focus_qc.csv",
        PATH_MODE_MERGED:   "{experiment_id}_focus_qc.csv",
    }},
},
"motion_blur_qc": {
    "stage": "quality_control",
    "product_dir": "motion_blur_qc",
    "fanout": PER_WELL_THEN_MERGE,
    "execution": EXECUTION_PER_WELL,
    "artifacts": {"motion_blur_qc": {
        PATH_MODE_PER_WELL: "{well_id}_motion_blur_qc.csv",
        PATH_MODE_MERGED:   "{experiment_id}_motion_blur_qc.csv",
    }},
},
```

The `product_dir` key inserts a **per-product subfolder** under the stage (built by
`_experiment_step_dir`: `{stage}/{exp}/{product_dir}`), exactly like `mask_quality_qc` today. So these
resolve (via the existing `rule_artifact`/`rule_validated` helpers) to:

```text
quality_control/{exp}/focus_qc/per_well/{well_id}/{well_id}_focus_qc.csv             (+ .validated)
quality_control/{exp}/focus_qc/{exp}_focus_qc.csv                                    (merged)
quality_control/{exp}/motion_blur_qc/per_well/{well_id}/{well_id}_motion_blur_qc.csv (+ .validated)
quality_control/{exp}/motion_blur_qc/{exp}_motion_blur_qc.csv                        (merged)
```

i.e. `{stage}/{exp}/{product_dir}/per_well/{well_id}/{file}` — the QC stage segregates each product
into its own `{product_dir}` subfolder. `.smk` rules clone `mask_quality_qc.smk` and call the existing
`rule_artifact(step, artifact, …)` helpers — no path strings in the rule files (the registry is the
single source of path truth).

### 3. z_position_um / z_step_um — columns, not files

These are new COLUMNS on the existing acquisition-inventory CSV (Part A) — no new path. The
acquisition inventory already has its own home; we widen its schema, not its layout.

### 4. z-slice selection — deferred home (not QC)

When built, selection gets its own `PIPELINE_STEPS` row under an `image_selection` (or
`image_materialization`) stage — NOT `quality_control`. Out of scope for this pass; named here only so
nobody files it under QC by reflex.

---

## Build order

```text
1. Acquisition z-depth provenance  (Part A1, YX1): voxel_size()[2] → z_step_um;
   per-plane stage.z → z_position_um; +2 inventory columns +contract.   ← durable, do first
2. [z_stack pixels + z-aware frame_inventory ship via the LOCKED wire-through doc]
3. focus_index_map (Part D + Part F): persist the focus-stacker `idx` beside the FF in the SAME
   materialize pass (FF + z_stack + idx together — MVP); record `focus_index_map_path` on the
   projection frame_inventory row. No separate stage/sentinel. **HARD GATE (Part F Hazard 2): extend
   the L4 source validator to check `focus_index_map_path` exists+loads+shape on projection rows IN
   THE SAME change — the column must not exist before the validator polices it.**
4. focus_qc module  (clone mask_quality_qc + frame_inventory input): load z_stack via inventory,
   grid in-memory, reduce in-mask → focus_flag (uses reserved `focus` hook).
5. motion_blur_qc module (same shape): NCC grid in-memory, reduce in-mask → motion_blur_flag.
6. snip_qc: add "motion_blur"->"motion_blur_flag"; register focus/motion_blur reasons; wire shards into inputs.py
7. z-slice selection (separate, non-QC consumer) — reads focus_index_map + z-depth.
8. Keyence (Part A2 + Keyence z_stack via keyence_wire_through.md), AFTER YX1 is green end-to-end.
```

## Artifact ownership

| Artifact | On-disk location (relative to DATA_ROOT) | Written by | Read by |
|---|---|---|---|
| z_stack PNGs + FF PNG | `built_image_data/{exp}/materialized_images/{well}/{projection,z_stack}/...` (LOCKED) | materialize executor | focus_qc, motion_blur_qc, selection (via frame_inventory) |
| **focus_index_map** (`.npz`) | `built_image_data/{exp}/materialized_images/{well}/focus_index_map/{channel}/{image_id}.npz` | materialize executor (the `idx` it already computes) | z-slice selection |
| `z_position_um`/`z_step_um` | COLUMNS on the acquisition inventory CSV (no new path) | YX1/Keyence acquisition extractor | selection (via inventory join) |
| laplacian/entropy/NCC **grids** | — (transient, recomputed in QC from pixels; not stored) | focus_qc/motion_blur_qc in-memory | (debug/cache mode OPTIONAL) |
| `focus_qc` summary | `quality_control/{exp}/focus_qc/per_well/{well}/{well}_focus_qc.csv` (+ merged + `.validated`) | `focus_qc` entrypoint | `snip_qc`, debugging |
| `motion_blur_qc` summary | `quality_control/{exp}/motion_blur_qc/per_well/{well}/{well}_motion_blur_qc.csv` (+ merged + `.validated`) | `motion_blur_qc` entrypoint | `snip_qc`, debugging |
| z-slice selection table | `image_selection/...` (TBD; NOT quality_control) | selection consumer (TBD) | downstream image selection |

---

## Decisions locked
1. **QC persists the per-snip SUMMARY (reduced metrics + flag), not the flag alone** — a flag without
   its metrics is a black box. This summary table is a **QC summary product**, not a feature product.
2. **No persisted grid product. Grids are transient — recomputed in QC from the durable pixels.**
   The pixels are inventory-addressable, so each QC job loads them and computes its grid in-memory in
   the SAME job (no cross-stage loss). A grid may optionally be written behind a **debug/cache mode**
   (non-canonical QC-evidence), and promoted to `feature_extraction/` only if a real analysis need
   earns it a contract — but the default and the MVP store nothing. *A grid becomes a product only
   when reuse earns it a contract; here QC reuse does not, because recompute-from-pixels is cheap.*
3. **The focus-index map is the ONE durable materialization-derived sidecar artifact.** It is
   algorithm-derived during the FF projection pass — NOT an acquisition fact (contrast `z_position_um`/
   `z_step_um`, which ARE acquisition facts). The focus-stacker already computes the per-pixel
   argmax-over-Z (`idx`); persist it beside the FF in the same materialize pass and record
   `focus_index_map_path` on the projection frame_inventory row.
   **MVP: FF + z_stack + focus_index_map are generated together** (one focus-stack call; no second ND2
   read). This is what makes z-slice selection possible without a grid.
3a. **focus_index_map integration = companion of the projection product, guarded by the VALIDATOR
   (LOCKED Option A).** It is NOT a Snakemake-tracked output and gets NO own `.validated`. The `.npz`
   is policed by extending the L4 source check (`validate_sources` in
   `frame_inventory_validation_rules.py:224`) to resolve + load + shape-check `focus_index_map_path` on
   projection rows — so the projection shard's existing `.csv.validated` honestly covers it, and a
   missing `.npz` re-fires the DAG. **HARD GATE: ship the validator extension in the SAME change that
   adds the column.** Touch points: executor (`materialize_image_product_for_well` writes `.npz` +
   sets column), `materialized_image_paths.py` (`focus_index_map_path()` wrapper),
   `frame_inventory_contract.py` (nullable column), `frame_inventory_validation_rules.py` (L4 check).
   See Part F.
4. **z-depth provenance is added** to the acquisition inventory (`z_position_um` per plane,
   `z_step_um` per acquisition). YX1 FIRST (verified free from `voxel_size()` + `stagePositionUm.z`);
   Keyence SECOND, with graceful degradation to `z_index`-ordering if the BZ-X XML lacks Z depth.
   `z_position_um` = absolute **stage** Z (machine coord), NOT biological/embryo-relative depth.
5. **Two DISTINCT QC modules ship: `focus_qc` (focus_flag, reserved hook) and `motion_blur_qc`
   (motion_blur_flag, new hook)** — ordinary post-seg per-well modules cloned from `mask_quality_qc`
   (+`frame_inventory` input); each feeds `snip_qc` via its hook. `motion_blur_qc` = the NCC
   inter-slice motion-blur concept (renamed from `motion_qc` for clarity).
6. **Z-slice selection is NOT QC** — it consumes recorded information (z_stack pixels +
   focus_index_map + z-depth), emits a selection (not a flag), and lives outside `quality_control/`
   (`image_selection/` or `image_materialization/`).
7. **YX1 before Keyence** — the standing scope order for every layer in this spec.

> **Tiny doctrine:** *Pixels are primary. The focus-stacker's chosen-Z is the one derived fact worth
> keeping — capture it in the FF pass or lose it. QC persists summaries and verdicts, not grids —
> grids recompute from durable pixels. Selection persists choices, not flags.*
>
> *Ownership lives in the table. Shape lives in the path. The validator ties them together.*
