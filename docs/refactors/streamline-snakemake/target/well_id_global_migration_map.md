# well_id Global Migration Map (Scope 2)

**Status:** Pre-implementation map. Read-only audit, grounded against the working tree
on `mdcolon/20260222_docs_snakemake_remake` (2026-06-05). No code edited yet.
**Owner:** mdcolon
**Supersedes/decides:** the naming + video_id questions left open in
`well_id_throughline_refactor_plan.md`.

## Decided model (overrides the plan where noted)

```
experiment_id = 20240418              global experiment id        (stays)
well_index    = A01                   local well label            (KEEP THIS NAME)
well_id       = 20240418_A01          global well id = {experiment_id}_{well_index}
channel_id    = BF                    local channel token
image_id      = {well_id}_{channel_id}_t{time_int:04d}
embryo_id     = {well_id}_e{local_embryo_index:02d}
snip_id       = {embryo_id}_t{time_int:04d}
```

**DECISION 1 — keep `well_index`, REJECT the plan's `well` (overrides plan lines 53–56).**
Rationale: a column named `well` sitting next to `well_id` reads as "the well," implying
`well_id` is just `well + _id`. `well_index` is unambiguous: it is the local index/label,
plainly *not* the global id. Keep all three columns (`experiment_id` + `well_index` + `well_id`).

**DECISION 2 — collapse `video_id` INTO `well_id` now, all at once.** `video_id` is already
minted as `f"{experiment_id}_{well_slug}"` (`segmentation_and_tracking.py:108`) — it *is* the
global well id. It is absorbed, not deleted. The `video_generation` `re.match` /`rsplit("_")`
re-parsing becomes `split_well_id(well_id)`.

**INVARIANT — no half-global seam.** Every place that today re-derives `well_id` defensively
(`.split("_")[-1]`, `startswith(f"{exp}_")`, two-candidate `isin(wanted)`) must be removed in
the same pass, or an old local `A01` will silently fail to match a new global `20240418_A01`.

---

## STATUS (2026-06-05)
- **Section A (grammar): DONE.** Global `well_id`-first constructors + `split_well_id` +
  `validate_well_id` implemented; 13 tests pass.
- **Section C (6 mint sites): DONE.** All metadata-ingest mint sites pass the new args; no
  old-arity calls remain; all 10 importers green. (Also removed the now-dead `experiment_id`
  param from `ingest_propagation` — well_id-first `build_embryo_id` made it unused.)
- **Sections B, D, E, F: NOT STARTED** (deferred, downstream of the foundation).
- ⚠️ **Frame-contract rename:** `front_end_naming_and_flow.md` (Decision, lines 214/367–373)
  renames the table `frame_contract` → `frame_inventory` and splits `build_frame_contract` into
  `build_frame_inventory_well` + `validate_frame_inventory_well`. Section B's `frame_contract.py`
  row must fold in that rename when Section B is executed. Not done here.

## A. shared/identifiers/ — the grammar (Scope 1 split DONE; this is the signature flip) ✅ DONE

| Fn | Current (local) | Target (global) |
|---|---|---|
| `build_well_id` | `(well_index) -> "A01"` | `(experiment_id, well_index) -> "{sanitize(exp)}_{well_index}"` |
| `build_image_id` | `(experiment_id, well_id, channel_id, time_int)` | `(well_id, channel_id, time_int)` — well_id-first |
| `build_embryo_id` | `(experiment_id, well_id, local_track_id)` | `(well_id, local_track_id)` |
| `build_snip_id` | `(embryo_id, time_int)` | unchanged |
| `split_well_id` | stub raises | IMPLEMENT: `"{exp}_{well_index}" -> (exp, well_index)` via `rsplit("_", 1)` w/ A01 validation |
| `validate_well_id` | stub raises | IMPLEMENT: assert global form (has `_`, tail matches `^[A-H][0-9]{2}$`); reject bare `A01` |

## B. Schemas (9 files) — well_id becomes global; keep experiment_id + well_index; drop video_id

| Schema file | well cols today | change |
|---|---|---|
| `scope_metadata.py` | experiment_id, well_id, well_index | semantics: well_id global. cols unchanged. |
| `frame_contract.py` | experiment_id, well_id, well_index; UNIQUE_KEY=(exp,well_id,channel,time) | UNIQUE_KEY can drop exp (well_id now unique). |
| `plate_metadata.py` | well_index | add/confirm well_id global. |
| `snip_processing.py` | well_index, well_id | semantics only. |
| `auxiliary_masks.py` | well_index, well_id | semantics only. |
| `features.py` | well_id/well_index | semantics only. |
| `stage_predictions.py` | well_index | confirm well_id. |
| `stitched_image_index.py` | well_index | confirm well_id. |
| `segmentation.py` | **video_id** ×6, well_id ×2, well_index ×2 | **DELETE video_id** from all 6 (SEGMENTATION_TRACKING, FRAME_DETECTIONS, SEED_SELECTION, TRACK_INSTANCES, MASK_RLE, V2); UNIQUE_KEYs already use well_id. |

## C. Mint sites (6) — pass new args

| Site | line | change |
|---|---|---|
| keyence/extract_scope_metadata.py | 287 | `build_well_id(experiment_id, well_index)` |
| keyence/map_series_to_wells.py | 179 | `build_well_id(experiment_id, well_index)` |
| yx1/extract_scope_metadata.py | 212 | `build_well_id(experiment_id, well_index)` |
| stitched_index/materialize_stitched_images.py | 475 | `build_image_id(well_id, channel_id, time_int)` |
| scope/shared/apply_series_mapping.py | 84,93 | `build_well_id(exp, well_index)` + well_id-first `build_image_id` |
| microscope_data_ingest/frame_contract/build_frame_contract.py | 62 | well_id-first `build_image_id` |
| sam2_ingestor.py | 38 | `build_embryo_id(well_id, local_track_id)` |
| plate/plate_processing.py | 65 | `.map(build_well_id)` → needs exp; row-wise apply |

## D. Defensive re-derivation LANDMINES — remove in same pass (INVARIANT)

| Site | line | pattern | fix |
|---|---|---|---|
| segmentation_and_tracking.py | 70–91 | `startswith(f"{exp}_")` + `storage_well_id=split("_",1)[-1]` + `wanted={local,global}` isin | filter `well_id == requested_well_id` (both global now) |
| segmentation_and_tracking.py | 107,153 | `well_slug = canonical_well_id.split("_")[-1]` | `_, well_index = split_well_id(well_id)` |
| merge_segmentation_and_tracking_contracts.py | 22,29 | `startswith(f"{experiment}_")` + `split("_")[-1]` | shards already keyed on global well_id |
| merge_snip_manifests.py | 22,28 | same | same |
| grounded_sam2/csv_formatter.py | 206 | `well_id = video_id.split("_")[-1]` | `well_id = video_id` (it IS global); local via split_well_id |
| compute_stage_predictions.py | 71,77 | `well_id == well` (local compare) | compare global well_id |
| apply_series_mapping.py | 105 | `isin(selected_wells_set)` | selected set must be global well_ids |

## E. video_id collapse — the video_generation subsystem (~10 files) + build

| Area | files | change |
|---|---|---|
| `segmentation/video_generation/` | video_generator, render_eval_video, results_adapter, service, models, overlay_manager | rename param `video_id` → `well_id`; `re.match(r"^(.+)_([A-H]\d{2})$", …)` and `rsplit("_",1)` → `split_well_id`. **Load-bearing parsing — audit each.** |
| `segmentation_and_tracking/` normalizers | normalize_{seed_selection,track_instances,frame_detections,mask_rle}, gdino_ingestor, raw_types | drop `video_id` field; thread `well_id` |
| `segmentation_and_tracking.py` | 108,218,226,310,327 | delete `video_id = f"{exp}_{slug}"`; use `well_id` directly |
| `src/build/build03A_process_images.py` | 1198,1691,1727 | `well_id = video_id.str.extract(...)` → video_id IS well_id. **NOTE: src/build is legacy — confirm in-scope.** |
| `src/run_morphseq_pipeline/combine_experiments_parts.py` | 570,583 | drop video_id from id_columns |
| `src/build/pipeline_objects.py` | 523–600 | legacy fallback parsing; confirm scope |

> ⚠️ **src/build/ and src/run_morphseq_pipeline/ are the LEGACY build (build03A etc.), not the
> new data_pipeline.** Confirm whether the migration touches them or stops at `data_pipeline/`.

## F. Snakefile + docs

- Snakefile: `well_index` appears only in a docstring (`:191`) on this branch — verify the
  `discover_wells` checkpoint + per_well `{well_id}` keying read global well_id (the tree keys
  on `{well_id}`; once global, paths become `per_well/20240418_A01/`).
- `identifier_and_wildcard_contract.md`: flip the worked examples to global well_id, drop the
  CURRENT-vs-TARGET banner's "not yet" framing once this lands.
- `well_id_throughline_refactor_plan.md`: update the stale 2026-06-02 banner (Scope 1 is DONE
  as a split; `shared/identifiers/` exists with importers) and record DECISION 1 (`well_index`
  kept, not renamed to `well`).

## Scope boundary (DECIDED 2026-06-05)

**IN scope:** `src/data_pipeline/**` (incl. `segmentation/video_generation/`).
**OUT of scope:** `src/build/` (build03A path) and `src/run_morphseq_pipeline/` — legacy build,
not part of what the refactor rebuilds. The `video_id`/`well_id` parsing in `build03A_process_images.py`,
`combine_experiments_parts.py`, `pipeline_objects.py`, and anything under `src/_Archive/` is
**left untouched**. Section E rows tagged "legacy" are dropped from this migration.
