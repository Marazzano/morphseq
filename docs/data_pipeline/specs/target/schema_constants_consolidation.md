# Schema/Column-Constant Consolidation — census + plan (🟡 IN PROGRESS)

**Status:** census + naming grammar LANDED, legacy purge partly pending. mdcolon 2026-06-25.

**Progress:**
- ✅ `396409e2` — snip_inventory validation consolidated behind one gate
  (`validate_snip_inventory_contract`); `SNIP_FRAME_DERIVED_COLUMNS` deduped.
- ✅ `03ee4296` — World A renamed to the composition grammar (`*_SPINE` / `*_PROVENANCE` /
  `*_PAYLOAD` / `*_TABLE`). Names now reveal composition.
- ⏳ **PENDING (hand to a demo agent):** delete the two CONFIRMED-DEAD legacy schemas
  `src/data_pipeline/schemas/auxiliary_masks.py` and `src/data_pipeline/schemas/stage_predictions.py`
  (0 live importers per the census below — verify no test-only use, then delete).
- 🔜 **Deferred:** tier-2 migration (`features.py`, frame_detections half of `segmentation.py` onto
  the composed constants) and tier-3 (back-half / Scope 5 — `frame_contract.py`, `analysis_ready.py`,
  tracking/mask_rle, legacy snip_manifest). See the tier table below.

---

## 🪨 The naming grammar (LOCKED) — names must reveal composition

The smell: a composed constant like `CURVATURE_FEATURES_REQUIRED_COLUMNS = list(SPINE + _FEATURE_COLUMNS)`
reads as a flat atomic list — the **name** hides that it's `spine + payload`, and looks identical to a
legacy flat list. Fix: the suffix names the **layer / role**, so composition is legible at the name.

```
*_SPINE_COLUMNS       identity-bearing shared columns (additive: physical_embryo → embryo → snip)
*_PROVENANCE_COLUMNS  carried source/context columns (e.g. image_id/time_index/channel_id on a snip)
*_PAYLOAD_COLUMNS     THIS table's own delta (the part added on top of spine+provenance)
*_TABLE_COLUMNS       the assembled full contract = SPINE + PROVENANCE + PAYLOAD
```

Doctrine: **`PAYLOAD` names the ROLE, not the domain.** Do NOT use `FEATURE` as a layer suffix —
"feature" is a domain concept and is ambiguous ("feature-measurement columns" vs "columns of a
feature table"). `PAYLOAD` = "the part added to the shared contract." A feature table is just one use
case of payload; snip_inventory and QC tables use the same grammar.

Worked examples (target shape):
```python
CURVATURE_PAYLOAD_COLUMNS = ("mean_curvature", "max_curvature", ...)
CURVATURE_TABLE_COLUMNS   = SNIP_FEATURE_TABLE_SPINE_COLUMNS + CURVATURE_PAYLOAD_COLUMNS

SNIP_INVENTORY_PAYLOAD_COLUMNS = ("mask_id", "track_id", "source_image_path",
                                  "processed_snip_path", "is_valid_snip", "error_message")
SNIP_INVENTORY_TABLE_COLUMNS   = (SNIP_ID_SPINE_COLUMNS
                                  + SNIP_FRAME_PROVENANCE_COLUMNS
                                  + SNIP_INVENTORY_PAYLOAD_COLUMNS)
```

Rename map (modern World A constants → grammar):
| today | →  grammar |
|---|---|
| `SNIP_ID_SPINE_COLUMNS` (and the two parents) | keep — already `_SPINE_COLUMNS` ✓ |
| `SNIP_FRAME_DERIVED_COLUMNS` | `SNIP_FRAME_PROVENANCE_COLUMNS` |
| `SNIP_FEATURE_TABLE_ID_COLUMNS` | `SNIP_FEATURE_TABLE_SPINE_COLUMNS` (spine+provenance base for feature tables) |
| `_FEATURE_COLUMNS` (per product) | `<PRODUCT>_PAYLOAD_COLUMNS` |
| `_QC_FLAG_COLUMNS` / `_DEATH_FLAG_COLUMNS` / `_SNIP_QC_VERDICT_COLUMNS` | `<PRODUCT>_PAYLOAD_COLUMNS` |
| `<PRODUCT>_FEATURES_REQUIRED_COLUMNS` / `<PRODUCT>_REQUIRED_COLUMNS` | `<PRODUCT>_TABLE_COLUMNS` |
| `_SNIP_INVENTORY_REQUIRED_NON_IDENTITY_COLUMNS` | `SNIP_INVENTORY_PAYLOAD_COLUMNS` |

---

## ✅ World A — the modern compositional spine (HEALTHY, just needs the rename)

Roots, defined ONCE, composed everywhere:
- `segmentation/physical_embryo_registry/snip_identity_contract.py`:
  `PHYSICAL_EMBRYO_ID_SPINE_COLUMNS` → `EMBRYO_ID_SPINE_COLUMNS` → `SNIP_ID_SPINE_COLUMNS`,
  + `SNIP_FRAME_DERIVED_COLUMNS` (→ rename PROVENANCE).
- `image_materialization/frame_inventory_contract.py`: `REQUIRED_FRAME_INVENTORY_COLUMNS`,
  `DOWNSTREAM_FRAME_IDENTITY_BLOCK`, `UNIQUE_FRAME_INVENTORY_KEY_COLUMNS`.
- `feature_extraction/shared/feature_table_utils.py`: `SNIP_FEATURE_TABLE_ID_COLUMNS = SPINE + DERIVED`.

Per-product composed constants (all `= SPINE + delta`) — mask_geometry, curvature, pose, stage,
fraction_alive, consolidated, surface_area_qc, mask_quality_qc, snip_qc, death_detection. **These are
correct** — the per-product delta tuples are legitimately local; only the NAMES need the grammar so
they read as compositions.

---

## ⚠️ World B — legacy `schemas/` (the pollution) — per-constant fate

`src/data_pipeline/schemas/` holds parallel flat lists, several still live-imported. Fate by module:

| schemas module | live importers | modern equivalent | fate |
|---|---|---|---|
| `channel_normalization.py` (`VALID_CHANNEL_NAMES`, `BRIGHTFIELD_CHANNELS`) | many (incl. modern parsers) | — IS canonical | **KEEP** — not legacy; it's the channel vocabulary the modern parsers use |
| `features.py` (`REQUIRED_COLUMNS_FEATURES`) | 4 (analysis_ready, feature io, consolidate_features, qc io) | `CONSOLIDATED_FEATURES_*` (composed) | **MIGRATE** importers → composed, then delete |
| `segmentation.py` (`REQUIRED_COLUMNS_*`, `UNIQUE_KEY_*` ×5) | 8 (segmentation_and_tracking, gsam2, feature io) | partial: detection has `REQUIRED_FRAME_DETECTIONS_COLUMNS`; tracking/seed/mask_rle have NONE | **SPLIT**: migrate frame_detections; the rest need a modern home FIRST (back-half / Scope 5) |
| `frame_contract.py` (`REQUIRED_COLUMNS_FRAME_CONTRACT`, `UNIQUE_KEY_FRAME_CONTRACT`) | 3 + the `frame_contract/` builder | superseded by `frame_inventory` | **LEGACY pre-frame_inventory** — wired into back half (segmentation_and_tracking, consolidate_features). Retire only when the back half moves to frame_inventory (Scope 5). DO NOT touch now. |
| `quality_control.py` (`SNIP_EXCLUSION_FLAGS`, `QC_OUTPUT_COLUMNS`, …) | 3 | modern QC contracts compose from spine | **MIGRATE** where modern QC owns the flags; reconcile flag lists |
| `scope_metadata.py` (`REQUIRED_COLUMNS_SCOPE_METADATA`) | 2 | acquisition/scope contracts | **REVIEW** — may still be the scope contract |
| `stitched_image_index.py` | 1 | — | **REVIEW** (stitched index path) |
| `snip_processing.py` (`REQUIRED_COLUMNS_SNIP_MANIFEST`) | 2 (legacy snip pipelines/) | modern = `snip_inventory` contract | **LEGACY** — the snip_manifest path is the OLD snip flow; retire with the legacy `snip_processing/pipelines/` |
| `analysis_ready.py` | 4 | — | back-half; **defer** (Scope 5) |
| `auxiliary_masks.py` | 0 | — | **DEAD** → delete (verify no test-only use) |
| `stage_predictions.py` (`REQUIRED_COLUMNS_STAGE_PREDICTIONS`, `UNIQUE_KEY`) | 0 | modern `STAGE_PREDICTION_*` contract | **DEAD** → delete (verify) |

---

## 🚧 Blast-radius honesty

The purge is NOT one clean sweep. Three tiers:
1. **Safe now (dead):** `auxiliary_masks.py`, `stage_predictions.py` schemas — 0 live importers. Verify
   no test-only dependency, delete.
2. **Migrate now (modern equivalent exists):** `features.py`, the `frame_detections` half of
   `segmentation.py`, modern-owned QC flags. Repoint live importers to the composed constants.
3. **BLOCKED on the back half (Scope 5):** `frame_contract.py`, `analysis_ready.py`, the
   tracking/seed/mask_rle half of `segmentation.py`, legacy `snip_processing` manifest. These are the
   not-yet-refactored back half; they have NO modern composed home yet. Retiring them is part of
   Scope 5, not this pass. Forcing it now would either break the back half or require building the
   modern back-half contracts first.

---

## Small twice-defined tuples (independent, safe dedup)

- `_PIXEL_SIZE_COLUMNS` = `("source_micrometers_per_pixel", "micrometers_per_pixel")` — defined twice
  (`mask_geometry/compute.py`, `feature_table_utils.py`). Hoist to one shared home.
- `_TIME_COLUMNS` = `("elapsed_time_s", "experiment_time_s", "time_s")` — twice
  (`pose_kinematics/compute.py`, `stage_predictions/compute.py`).
- death-flag tuple `("viability_dead_flag", "persistence_dead_flag")` — twice
  (`death_detection/contract.py` `_DEATH_FLAG_COLUMNS`, `grain_reconciliation.py` `_SNIP_FLAG_COLUMNS`).

---

## Recommended execution order

1. **Naming grammar rename** across World A (modern contracts) — pure rename, the healthy world, no
   importer-graph risk beyond the renamed symbols. Makes "good" visually distinct from "legacy."
2. **Dead-schema deletion** (tier 1) — `auxiliary_masks`, `stage_predictions` schemas.
3. **Migrate tier 2** (`features.py`, frame_detections half) onto the composed constants.
4. **Defer tier 3** to Scope 5 (back half) — record here, do not force.

> Doctrine: a name should say which layer it is. Spine is shared and additive; payload is the
> table's delta; table is the whole. Never let a composed contract wear a flat name.
