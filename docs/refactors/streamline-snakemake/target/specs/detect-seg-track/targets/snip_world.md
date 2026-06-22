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
snip_processing mints physical_embryo_id, embryo_id, and snip_id.
Crops record the per-embryo, per-channel, per-time view.
snip_inventory is the stable downstream join key.
```

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

No discrete `tracks` stage for MVP. A `tracks.csv` artifact (for gap/swap QC) can be promoted
later; at that point `track_id → physical_embryo_id` resolution moves there and snip_processing
becomes a consumer of tracks.

---

## embryo_id minting boundary

`frame_masks` carries `track_id` (pipeline track identity — not raw SAM2 `sam2_object_id`, which
is recoverable via `mask_id → frame_masks`).

snip_processing minting chain — named functions, no arithmetic inline:

```python
raw_track_index    = parse_embryo_local_track_id(track_id)   # "embryo_0" → 0
local_embryo_index = track_index_to_embryo_index(raw_track_index)   # 0 → 1
physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
embryo_id          = build_embryo_id(physical_embryo_id, image_id)
snip_id            = build_snip_id(embryo_id, image_id)
```

Fail loud if `track_id` is unparseable on a valid mask row. `track_id` is carried in
`snip_inventory` as provenance only — not a join key.

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
- **Deterministic cross-checks** (using public parsers/constructors from `shared/identifiers`):
  - `channel_id` == `parse_image_id(image_id).channel_id`
  - `time_index` == `parse_image_id(image_id).time_index`
  - `embryo_id == build_embryo_id(physical_embryo_id, image_id)`
  - `snip_id == build_snip_id(embryo_id, image_id)`
  - `mask_id` present in valid frame_masks rows (reference validation at build time)
- `check_sources=True` (consume boundary): assert each `processed_snip_path` exists on disk

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
| `src/data_pipeline/snip_processing/snip_inventory_contract.py` | Schema + validator (`validate_snip_inventory`) |
| `src/data_pipeline/snip_processing/build_snip_inventory.py` | Builder (consumes frame_masks, mints IDs, extracts crops) |
| `tests/data_pipeline/snip_processing/test_snip_inventory.py` | Validator + builder tests |
| Registry entry in `orchestration/paths.py` | `snip_inventory` step under `object_extraction` |
| `Snakefile` rule `snip_processing[well]` | Per-well rule consuming frame_masks shard |
