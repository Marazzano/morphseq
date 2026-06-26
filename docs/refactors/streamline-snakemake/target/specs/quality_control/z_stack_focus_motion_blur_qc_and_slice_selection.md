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
z-depth provenance extension** to the acquisition inventory plus **the focus-stacker's per-pixel
chosen-Z map (`focus_index_map`) — projection construction provenance it already computes and
currently discards** — so slices can be chosen later; and **focus QC and motion-blur QC recompute
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
[MATERIALIZATION]
   z_stack PNGs (per image_id, per Z)        ── primary pixels  (own independent leaf/read, as today)
   PROJECTION job (one focus-stack call):
     FF projection PNG (per image_id)        ── primary pixels      ┐ ONE LoG_focus_stacker call
     focus_index_map (per image_id pixel→Z)  ── DURABLE provenance  ┘ (idx falls out of the same call)
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

> **MVP constraint (LOCKED):** the focus_index_map is emitted by the **projection** job's single
> focus-stack call (the same call that produces the FF — `idx` and `ff` come out together). Do NOT
> build a separate pass that re-reads the ND2 or re-runs the focus-stack just to recover `idx`.
> (z_stack is unrelated here — it is its own independent leaf with its own read, as today.)

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

## Part D — Wiring into the DAG (no grid stage; projection construction-provenance + ordinary QC)

Because there is **no persisted grid product**, the awkward "pre-seg, frame-grain product threaded
through snip-grain QC plumbing" problem disappears. Only two DAG facts remain:

**1. The projection job emits one more output: the focus_index_map (per projection `image_id`).** It
is **construction provenance of the focus_stack projection** (Part E §1) — the `idx` the focus-stacker
already computes. The projection materialization job (`materialize_yx1_product_for_well`, projection
branch) writes it beside the FF as **`.npz`** under `projection/focus_index_map/` and sets the nullable
`focus_index_map_path` column on the projection frame_inventory row. **No new job, no new product_key,
no new shard, no separate inventory** — it rides the projection product entirely. (z_stack stays its
own independent product/leaf as today; the focus_index_map is NOT coupled to z_stack.)

**2. focus_qc / motion_blur_qc are ORDINARY post-seg per-well QC modules** — clone `mask_quality_qc.smk`
verbatim (build→validate→merge, `PATH_MODE_PER_WELL`, registry as verifier). They consume exactly
what `mask_quality_qc` consumes (validated `snip_inventory` + `frame_masks` + `registry`) **plus the
`frame_inventory`** (to address the z_stack PNGs to load). No new wildcard grain, no grid manifest,
no checkpoint — the segmentation-boundary ordering is already carried by depending on the validated
`snip_inventory`/`frame_masks`, exactly like every existing QC module.

```text
[MATERIALIZATION, per well — projection job gains ONE new output]
materialize_yx1_product_for_well (projection) → FF PNG + focus_index_map   (one focus-stack call)
materialize_yx1_product_for_well (z_stack)    → z_stack PNGs               (own independent leaf)
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
genuinely new artifact (focus_index_map) is **projection construction provenance** — stored under
`projection/`, recorded as a nullable column on the projection row, validated with projection (Part E
§1) — so it never touches the snip-grain QC template at all.

---

## Part E — Where the artifacts actually live on disk (the real paths)

Nothing here invents a path. Two existing machineries own the two artifact kinds; we add rows/files to
them. `DATA_ROOT` itself is machine-specific (`env.yaml` / `output_root`), so paths are always
RELATIVE to it — never hardcode `DATA_ROOT`.

### 1. focus_index_map — projection construction provenance (MVP model, LOCKED)

> **This supersedes earlier drafts** (companion-column-as-sidecar-folder; separate `derived_map`
> inventory family; `well_materialization_inventory/` parent layer). Those were chased to ground and
> rejected — see "The taxonomy we walked through" below for *why*, because the reasoning is the
> guardrail. The MVP is the scalpel: **focus_index_map is projection construction provenance, stored
> under `projection/`, recorded as a nullable column, validated with projection.**

**What it is.** The focus-stacker (`LoG_focus_stacker`, `image_building/shared/log_focus.py:76-77`)
computes `idx = abs_log.max(dim=1)` — the per-pixel argmax-over-Z — *to build the FF*, then discards
it. `idx` is **construction provenance of the focus_stack projection**: it explains how that one FF
pixel was chosen. It is strictly 1:1 with the projection `image_id`, only exists from a focus_stack
projection, and is not independently requestable.

**Path — nested under `projection/` (LOCKED).** Because it is born with, named by, and meaningless
without the projection, it lives *under* the projection product, not as a top-level sibling:

```text
built_image_data/{experiment_id}/materialized_images/{well_id}/
    projection/
      {channel_id}/{image_id}.png                ← FF (LOCKED)
      focus_index_map/{channel_id}/{image_id}.npz  ← construction provenance of the FF
    z_stack/
      {channel_id}/{well_id}_{channel_id}_z{z:04d}_t{t:04d}.png   ← z-planes (LOCKED)
```

The tree now *says the true thing*: `focus_index_map/` belongs to `projection/` but is not itself the
FF PNG. Add a `focus_index_map_path(...)` wrapper in `materialized_image_paths.py` that builds this
nested path; it is NOT a new `image_product_type` (do not add it to `ALLOWED_IMAGE_PRODUCT_TYPES`) and
gets NO `product_key`. It inherits the LOCKED `candidate/` isolation for free (same path constructor).

**Frame inventory — a nullable column, with an invariant shift.** Add `focus_index_map_path` to the
frame_inventory schema as a **nullable** column:

```text
projection (focus_stack) row : focus_index_map_path = projection/focus_index_map/BF/{image_id}.npz
projection (other method) row: focus_index_map_path = NA
z_stack row                  : focus_index_map_path = NA
```

This shifts the invariant — *honestly, and on purpose*:

```text
OLD (implicit):  one frame_inventory row = exactly one file path
NEW (explicit):  one frame_inventory row = one materialized image IDENTITY (image_id)
                 a row may carry that image's construction-provenance paths alongside it
```

The row still has exactly one identity (`image_id`) and one *image* file (`source_image_path`); the
`.npz` is that image's construction provenance, not a second image. No new row, no new shard, no new
`product_key`, no new inventory family, no new sentinel. The projection product's existing shard /
`.validated` / discovery / assembly / retirement cover it for free (rerun projection → FF + `.npz`
rewrite together; retire projection → both go).

#### 🚧 The anti-junk-drawer guardrail (the load-bearing rule — make it a contract, not a vibe)

The column is safe **only** behind a brutally narrow admission rule. Write it into the
frame_inventory contract as an explicit allowlist test, not a social convention:

```text
A path may be a frame_inventory column ONLY IF it is CONSTRUCTION PROVENANCE:
  1. strictly 1:1 with image_id
  2. emitted by the SAME materialization job as the image
  3. required to interpret or reuse that image product
  4. L4-validator-checked
  5. NA on rows where it does not apply

EXPLICITLY FORBIDDEN as frame_inventory columns (these are NOT construction provenance):
  QC grids (laplacian/NCC), debug overlays, thumbnails, entropy maps,
  selection outputs, analysis features.
  → those stay transient, or earn their own product/inventory LATER.
```

`focus_index_map_path` passes all five. The forbidden list is what stops the avalanche: the next
person who wants to bolt `entropy_grid_path` onto a row hits this rule and is sent to build a real
product instead. **Architecture-enforced, not vibes.**

#### Validator — extend L4 source check (the one real code gate)

`--check-sources=true` → L4 (`validate_sources`, `frame_inventory_validation_rules.py:224`) today
checks only `source_image_path`. Extend it: on a **projection focus_stack** row with a non-null
`focus_index_map_path`, also —

```text
resolve focus_index_map_path  (reuse _resolve_source_path — same image_root + no-`..` guard)
assert the file exists
assert the .npz loads
assert it contains the focus_index_map array (+ z_indices metadata)
assert focus_index_map.shape == (image_height_px, image_width_px)
assert values are valid z-indices
```

z_stack and non-focus_stack rows have `focus_index_map_path = NA` → skipped. This is also the
**staleness guard**: the `.npz` is not a Snakemake-tracked output (the rule's output is the shard
CSV), so a deleted/corrupt `.npz` would otherwise slip through — the L4 check fails the projection
shard's `.csv.validated`, which re-fires the DAG. **HARD GATE: ship the L4 extension in the SAME
change that adds the column — never let the column exist before the validator polices it** (mirrors
the z_stack contract's "never let pixels outrun the contract").

#### The taxonomy we walked through (recorded so the reasoning is the guardrail)

This MVP is the end of a long disentanglement. The dead ends and *why they died* are the real
protection against re-litigating this — keep them:

| Model considered | Why rejected for MVP |
|---|---|
| `focus_index_map` as its own `image_product_type` | not an image to segment; would pollute the `projection`/`z_stack` shape grammar |
| its own `product_key` (`BF__focus_index_map`) | not independently requestable — can't be produced without projection; a key that can't stand alone |
| `by-product` / `sidecar` concept | "downstream-ness" is not the producthood test (everything is downstream of something); conceptual fog |
| separate `derived_map` inventory family | a table of `(image_id, path)` with no identity beyond `image_id` is an *annotation*, not an inventory; extra rail for a 1:1 path |
| `well_materialization_inventory/` parent layer | re-paths frame_inventory (most-consumed table) for a speculative future zoo of maps; not earned by one tenant |
| **projection construction-provenance column (CHOSEN)** | smallest honest change; 1:1 with image_id; co-produced; guardrail-bounded |

**The disentanglement test (the durable principle):** *Can it be produced as an honest independent
leaf?* — `z_stack` **yes** (re-read ND2, slice planes) → its own product_key, fanout, frame_inventory
rows. `focus_index_map` **no** (only falls out of the focus_stack that makes the FF) → construction
provenance of projection, stored with it, validated with it. **Independent leaves get product_keys;
construction provenance rides its producer.**

When a SECOND map-like artifact actually arrives (confidence_map, depth_map, …), THAT is the moment
to promote to a `derived_map` inventory family — earned by a real second tenant, not built for an
imagined one.

> **Note — QC products have NO `product_key`.** `product_key` is image-materialization-internal
> vocabulary. `focus_qc`/`motion_blur_qc` are keyed by the `PIPELINE_STEPS` `step`/`product_dir`
> (below), the same as every other QC product — they never touch the `product_key` grammar.

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
3. focus_index_map (Part E §1): the projection job writes the focus-stacker `idx` as `.npz` under
   `projection/focus_index_map/`; add nullable `focus_index_map_path` column (populated on projection
   focus_stack rows, NA elsewhere) + the construction-provenance guardrail to the frame_inventory
   contract. No separate stage/product_key/inventory/sentinel. **HARD GATE: extend the L4 source
   validator (`validate_sources`) to check `focus_index_map_path` exists+loads+shape on projection
   focus_stack rows IN THE SAME change — the column must not exist before the validator polices it.**
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
| **focus_index_map** (`.npz`, projection construction provenance) | `built_image_data/{exp}/materialized_images/{well}/projection/focus_index_map/{channel}/{image_id}.npz` (+ `focus_index_map_path` column on projection row) | projection job (the `idx` it already computes) | z-slice selection |
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
3. **The focus-index map is projection CONSTRUCTION PROVENANCE — stored under `projection/`, recorded
   as a nullable column, validated with projection (MVP model, LOCKED — Part E §1).** It is
   algorithm-derived during the FF projection pass — NOT an acquisition fact (contrast `z_position_um`/
   `z_step_um`, which ARE). It fails the independence test (only born from focus_stack), is strictly
   1:1 with projection `image_id`, and is not independently requestable — so it is NOT an
   `image_product_type`, gets NO `product_key`, NO separate inventory family, NO own sentinel. Path:
   `materialized_images/{well}/projection/focus_index_map/{channel}/{image_id}.npz`. Recorded as a
   nullable `focus_index_map_path` column (projection focus_stack rows only; NA elsewhere). **Invariant
   shift (on purpose): one frame_inventory row = one materialized image IDENTITY, which may carry that
   image's construction-provenance paths** — not "exactly one file."
3a. **Anti-junk-drawer guardrail is a CONTRACT, not a vibe.** A path may be a frame_inventory column
   ONLY if it is construction provenance: (1) strictly 1:1 with image_id, (2) co-produced by the same
   materialization job, (3) required to interpret/reuse the image, (4) L4-validated, (5) NA where
   inapplicable. QC grids / overlays / thumbnails / selection outputs / analysis features are
   EXPLICITLY FORBIDDEN as columns. **HARD GATE: extend the L4 source validator (`validate_sources`,
   `frame_inventory_validation_rules.py:224`) to resolve+load+shape-check `focus_index_map_path` on
   projection focus_stack rows, in the SAME change that adds the column** — this is also the staleness
   guard (the `.npz` is not a Snakemake-tracked output; a missing one fails the projection
   `.csv.validated` and re-fires the DAG). Touch points: projection executor (writes `.npz` + sets
   column), `materialized_image_paths.py` (`focus_index_map_path()` wrapper, nested under projection),
   `frame_inventory_contract.py` (nullable column + guardrail allowlist), `frame_inventory_validation_rules.py` (L4 check).
   *(Earlier drafts — own product_key / by-product / separate `derived_map` inventory /
   `well_materialization_inventory/` parent — were chased to ground and rejected; the taxonomy table in
   Part E §1 records why. Promote to a `derived_map` inventory family only when a SECOND map-like
   artifact actually arrives.)*
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
