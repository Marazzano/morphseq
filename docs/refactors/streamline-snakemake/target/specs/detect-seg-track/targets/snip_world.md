# Snip World

**Status:** target-planning draft, mdcolon 2026-06-21.

---

## Role

Snip processing answers:

> Given validated frame masks, which per-embryo crops can be extracted, and what is each
> embryo's canonical identity across channels and time?

The target doctrine is:

```
frame_masks carry track identity.
physical_embryo_registry mints physical_embryo_id (the animal) — see physical_embryo_registry_world.md.
snip_processing RECEIVES physical_embryo_id (joined on (well_id, track_id)); it mints only
  embryo_id + snip_id — the per-channel/per-time PROJECTION of an already-registered animal.
Crops record the per-embryo, per-channel, per-time view.
snip_inventory is the stable downstream join key, and carries physical_embryo_id explicitly.
```

> **physical_embryo_id names the animal. snip_id names the crop. snip_inventory carries both. The
> contract proves they agree.** snip_processing never originates *who an animal is* — the
> `track_id → physical_embryo_id` resolution happens once, in `physical_embryo_registry`. The
> per-frame constructors snip_processing calls (`build_embryo_id`, `build_snip_id`) are **crop/product
> naming**, not identity origination: they project a registered animal onto a frame, performing no
> `track_id → animal` resolution.

Naming:

```
snip_processing   = stage / action
snip_inventory    = artifact / table
<snip_id>.png     = pixel artifact
```

Stages name operations. Artifacts name tables.

---

## Identity model

Snip processing introduces a three-level embryo identity hierarchy. All constructors and parsers
live in `src/data_pipeline/shared/identifiers/` — the single sacred home for the grammar.

```
physical_embryo_id  = {well_id}_e{local_embryo_index:02d}   (one-based, ≥ 1)
embryo_id           = {physical_embryo_id}_{channel_id}
snip_id             = {embryo_id}_t{time_index:04d}
```

Examples:

```
well_id              = 20250912_B01
physical_embryo_id   = 20250912_B01_e01        animal in the well
channel_id           = BF
embryo_id            = 20250912_B01_e01_BF     animal in a channel
time_index           = 7
snip_id              = 20250912_B01_e01_BF_t0007   animal-channel at a time (the crop)
```

**Tiny doctrine:**

```
Physical embryo ID names the animal.
Embryo ID names the animal in a channel.
Snip ID names the animal-channel at a time.
```

**Canonical `channel_id` has no underscores** — this constraint allows parsers to safely
locate the `_{channel_id}` boundary in compound IDs.

**Embryo indices are one-based** — biologist-facing. Embryo 1 is the first embryo, not embryo 0.

### Constructor signatures (in `constructors.py`)

```python
build_physical_embryo_id(well_id, local_embryo_index) -> str
build_embryo_id(physical_embryo_id, image_id) -> str   # parses image_id; verifies well_id match
build_snip_id(embryo_id, image_id) -> str              # parses both; verifies well + channel
```

### Parser signatures (in `parsers.py`)

```python
parse_image_id(image_id)                -> (well_id, channel_id, time_index)
parse_physical_embryo_id(physical_embryo_id) -> (well_id, local_embryo_index)
parse_embryo_id(embryo_id)              -> (physical_embryo_id, channel_id)
parse_snip_id(snip_id)                  -> (embryo_id, time_index)
parse_embryo_local_track_id(track_id)   -> int   # zero-based raw track index
track_index_to_embryo_index(raw_index)  -> int   # zero-based → one-based (the conversion)
```

---

## Target river

```
frame_masks[well]           validated, per_well shard (input)
  → snip_processing[well]   stage
  → snip_inventory[well]    table artifact (CSV + .validated)
  → snip pixels[well]       image files
```

`track_id → physical_embryo_id` resolution now lives in `physical_embryo_registry` (see
`physical_embryo_registry_world.md`), so snip_processing is already a **consumer** of registered
identity — it joins `physical_embryo_id` from the registry rather than minting it.

No discrete `tracks` stage for MVP. A `tracks.csv` artifact (for gap/swap/coherence QC) is a
*different grain and concern* (track quality, not entity declaration) and can be promoted later; if
it is, it would feed `physical_embryo_registry`, not replace it.

---

## Identity boundary — snip_processing receives the animal, projects the crop

`frame_masks` carries `track_id` (pipeline track identity — not raw SAM2 `sam2_object_id`, which
is recoverable via `mask_id → frame_masks`).

**`physical_embryo_id` is NOT minted here.** The `track_id → physical_embryo_id` resolution happens
exactly once, upstream, in `physical_embryo_registry` (see `physical_embryo_registry_world.md`).
snip_processing **joins** it in:

```python
# JOIN, not mint — physical_embryo_id comes from the validated registry.
#   frame_masks[well] ⋈ physical_embryo_registry[well]  on (well_id, track_id)
physical_embryo_id = registry_lookup[(well_id, track_id)]
```

Then snip_processing **projects** that registered animal onto the frame it is cropping — named
functions, no `track_id` arithmetic inline:

```python
embryo_id = build_embryo_id(physical_embryo_id, image_id)   # animal → animal-in-channel
snip_id   = build_snip_id(embryo_id, image_id)              # → the crop/product id
```

`build_embryo_id` / `build_snip_id` are **crop/product naming**, not identity origination: they
compose an already-registered `physical_embryo_id` with an `image_id`; they perform no
`track_id → animal` resolution. Fail loud if a valid mask row's `(well_id, track_id)` has no registry
match (a real embryo with no registered identity is a hard error, not a silent skip). `track_id` is
carried in `snip_inventory` as provenance only — not a join key.

---

## snip_inventory.csv — required columns

Grain: one row per `snip_id` (= one row per physical_embryo_id × channel_id × time_index).

```
# Primary identity (all three levels explicit)
snip_id                 primary key / crop artifact key
embryo_id               animal-in-channel key
physical_embryo_id      stable animal identity across time/channel

# Image context (decomposed; carried explicitly for readability + validator cross-checks)
experiment_id
well_id
image_id                source frame this snip derives from
time_index              canonical time axis (non-negative integer)
channel_id              BF today; one row per channel when multi-channel lands

# Mask provenance (audit link back to segmentation)
source_image_path       full-frame source image (from frame_inventory via frame_masks)
mask_id                 the frame_masks row this snip derives from
track_id                pipeline track identity from frame_masks (provenance, NOT join key)

# Output path
processed_snip_path     path to extracted crop PNG (nullable if is_valid_snip=false)

# Spatial provenance (factual geometry — NOT QC)
crop_x_min_px
crop_y_min_px
crop_x_max_px
crop_y_max_px
crop_width_px
crop_height_px

# Validity
is_valid_snip           boolean: crop extraction succeeded (structural, not biological quality)
error_message           nullable: failure reason if is_valid_snip=false
```

**Not in snip_inventory:** background stats (`background_mean`, `background_std`), rotation angle,
processing config hash, pipeline_version → deferred to `snip_qc.csv` under `quality_control/`.

---

## Validator

`validate_snip_inventory(df, *, check_sources: bool = False)` — structural checks only:

- All required columns present; `snip_id` unique (grain check)
- Key columns non-null: `snip_id`, `embryo_id`, `physical_embryo_id`, `well_id`, `image_id`
- `channel_id` in `VALID_CHANNEL_NAMES`
- `is_valid_snip` boolean; `processed_snip_path` non-null iff `is_valid_snip=True`
- `time_index` non-negative integer
- **Identity spine** — `snip_inventory` is a **snip-grain** table and must satisfy the pipeline-wide
  **Identity-Carrying Contract** owned by `physical_embryo_registry_world.md` (§ "Identity-Carrying
  Contract — the snip/embryo-grain identity spine"). **This doc does not restate that law** (one law,
  many citations — restating it lets it drift). `snip_inventory`'s validator **calls the shared spine
  validator** at snip grain, then adds its snip-specific checks:

  ```python
  validate_snip_grain_identity_columns(
      df, grain="snip",
      physical_embryo_registry_df=registry_df,   # consume boundary → check_sources=True
      scope_label="snip_inventory",
  )
  ```

  The spine guarantees the grain-aware identity columns + agreement
  (`physical_embryo_id`↔`embryo_id`↔`snip_id`, no parsing-to-rediscover, registry membership). The
  snip-specific additions layered on top:
  - `channel_id` == `parse_image_id(image_id).channel_id`; `time_index` ==
    `parse_image_id(image_id).time_index` (frame-derived columns agree with `image_id`);
  - `mask_id` present in valid frame_masks rows (reference validation at build time);
  - `track_id` maps to the **same** `physical_embryo_id` via the registry (provenance consistency).
- `check_sources=True` (consume boundary): assert each `processed_snip_path` exists on disk (and the
  spine's registry-membership check fires at this same boundary).

---

## No-mask frames policy

| Case | snip_inventory row? |
|---|---|
| frame_masks has only placeholder rows (`is_valid_mask=false`) | NO row — no valid mask = no crop attempt |
| Valid mask, crop extraction succeeds | Row with `is_valid_snip=true`, `processed_snip_path` set |
| Valid mask, crop extraction fails | Row with `is_valid_snip=false`, `error_message` set, path null |

Downstream joins left-outer-join on `snip_inventory`. Absence of a row is the signal.

---

## Output tree

```
object_extraction/
  <experiment_id>/
    snips/
      per_well/
        <well_id>/
          <well_id>_snip_inventory.csv
          <well_id>_snip_inventory.csv.validated
          <physical_embryo_id>/          ← grouped by animal (stable across time/channel)
            <snip_id>.png
      <experiment_id>_snip_inventory.csv
      <experiment_id>_snip_inventory.csv.validated
```

Pixel files group under `physical_embryo_id` (not `embryo_id` or `snip_id`) so all crops for
one animal across time and channel sit together on disk.

Follows the per-product convention from `output_tree_doctrine.md` exactly:
product folder = `snips/`, table name = `snip_inventory`.

---

## Reused utilities

| Utility | File |
|---|---|
| All identifier constructors/parsers | `src/data_pipeline/shared/identifiers/` |
| `VALID_CHANNEL_NAMES` | `src/data_pipeline/schemas/channel_normalization.py` |
| Crop extraction logic | `src/data_pipeline/snip_processing/extraction.py` |
| Processing ops | `src/data_pipeline/snip_processing/ops.py` |

---

## Deferred: discrete tracks stage

When track-level QC (gap detection, swap detection, track coherence) is needed, promote a
`tracks.csv` artifact between `frame_masks` and `snip_processing`. Spec: `overall_goal_and_plan.md`
`track_instances` product. At that point `track_id → physical_embryo_id` resolution moves into the
tracks stage and snip_processing becomes a pure consumer.

---

## Files to create (implementation, not this session)

| File | Role |
|---|---|
| `src/data_pipeline/snip_processing/snip_inventory_contract.py` | Schema + validator (`validate_snip_inventory`) — incl. the identity-consistency contract (physical_embryo_id carried + agreement-validated) |
| `src/data_pipeline/snip_processing/build_snip_inventory.py` | Builder (consumes frame_masks **+ physical_embryo_registry**; JOINS physical_embryo_id, projects embryo_id/snip_id, extracts crops — does NOT mint physical_embryo_id) |
| `tests/data_pipeline/snip_processing/test_snip_inventory.py` | Validator + builder tests |
| Registry entry in `orchestration/paths.py` | `snip_inventory` step under `object_extraction` |
| `Snakefile` rule `snip_processing[well]` | Per-well rule consuming frame_masks shard |
