# Keyence Wire-Through — bring Keyence onto the per-well materialize interface (🟡 PLANNED)

**Status:** planning spec, mdcolon 2026-06-18. Captures the research for wiring **Keyence**
(BZ-X) onto the per-well image-materialization interface that YX1 already rides. This is the
"what would Keyence need" map — staged, with the open decisions left explicit. **No code yet.**

**Companion to:** `acquisition_inventory_flow.md` (the per-scope upstream record + the
collision/resolve/eligibility layer this spec reuses), `frame_inventory_handoff_contract.md`
(the post-stitch agnostic seam Keyence must emit, unchanged), `run_well_schema.md` (the
discovered/eligible/runnable model + the canonical Keyence reacquisition test case), and
`../front_half_reorg_roadmap.md` (the YX1 strangler this Keyence track mirrors).

**Scope of THIS doc:** the Keyence-specific build that plugs into the *already-built* 3-role
materialize interface — the acquisition inventory, the per-well mosaic backend, the stitch-map
(`master_params`) quirk, the comparison gate, and the re-acquisition/eligibility sub-track.

---

## 🪨 The interface Keyence plugs into (already built for YX1)

The image-materialization stage has a clean **3-role scope-backend interface**, all under
`src/data_pipeline/image_materialization/`. This IS the "step interface / scope backend /
shared validation" split the philosophy asks for. Keyence does NOT invent a new shape — it
fills the reserved seats:

```
materialization_plan.py        SCOPE-AGNOSTIC vocabulary (the step interface).
                               ImageProductRequest{channel_id, image_product_type∈{projection,z_stack},
                               projection_method∈{focus_stack,max,mean}, xy_composition∈{auto,identity,mosaic}}
                               → ResolvedImageProduct. Global product grammar only.
                               ── KEYENCE: NO CHANGE. `mosaic` is already a request token.

scope_resolver_for_materialization_plan.py   SCOPE QUIRKS live here.
                               resolve_materialization_plan(scope_name, requested_plan).
                               ── KEYENCE: `_resolve_keyence` + `_resolve_keyence_xy_composition`
                                  ALREADY EXIST as a reserved-but-unrouted sketch
                                  (auto→mosaic, reject identity). The file comment literally says
                                  "Wire route + backend together when Keyence is real."

scope/{scope}/materialize_well_{scope}.py     SCOPE BACKEND EXECUTOR.
                               Receives a ResolvedMaterializationPlan (commitment, never a request)
                               + the well's acquisition inventory df → flat frame-inventory df.
                               ── KEYENCE: NEW file, asserts xy_composition=='mosaic'
                                  (mirror of YX1's identity guard).

run_materialize_well.py        SEQUENCER. Loads plan → resolver → backend.
                               ── KEYENCE: today hard-raises for scope!='yx1'. Becomes a thin
                                  2-way backend dispatch.

frame_inventory_contract.py    SHARED, microscope-AGNOSTIC handoff seam. MUST NOT learn Keyence
                               logic — and won't need to. Keyence emits the same flat schema YX1 does.

materialized_image_paths.py    Off-registry image-tree paths; takes built_image_data_dir explicitly.
                               ── KEYENCE: already generic enough (well_id/channel/time/product_type).
```

**Conformance rule:** scope-specific config + steps STAY in the backend (orientation, n_tiles,
fallback policy, legacy-canvas/invert/transpose quirks). The plan vocabulary, the sequencer, and
the `frame_inventory` seam never learn what Keyence is.

---

## 🔑 The five Keyence facts that drive the work

1. **Keyence is MULTI-TILE** (vs YX1 single-tile). XY composition = `mosaic`. The stitching engine
   is **already scope-agnostic and exists** — `image_building/utils/frame_tiler.py`
   (`stitch_frame_tiles(tile_specs, FrameTilingConfig, FallbackParams) -> FrameTileResult` with QC)
   and `image_building/shared/log_focus.py` (`LoG_focus_stacker`). **Reused, not rebuilt.**

2. **STITCH MAP / `master_params` — "generate once, reapply many."** The headline Keyence quirk.
   Legacy `image_building/scope/keyence/stitched_ff_builder.py::build_master_params(...)` samples
   ~50 well/timepoint folders, aligns each, takes the **MEDIAN tile coords**, writes
   `master_params.json` (`coords: {tile_idx:[x,y]}`). Then `frame_tiler.FallbackParams(
   master_params_path=...)` + `run_align=False` **reapplies** those coords to every other
   well/timepoint without re-aligning. It is **experiment-grain, consumed per-well** — so it does
   NOT fit `materialize_well[well_id]` cleanly. **Where it lives is the load-bearing design call.**

3. **Keyence has NO acquisition inventory module yet.** YX1 has
   `metadata_ingest/scope/yx1/acquisition_inventory.py` (build + validate, Tier-1 shared core +
   Tier-2 extras, carries `source_nd2_path` + `elapsed_time_s`). Keyence has only
   `extract_scope_metadata.py` + `map_keyence_positions_to_wells.py` + `mappings.py`. **A Keyence
   acquisition inventory must be built** (the twin of YX1's).

4. **The source-path shape DIFFERS.** YX1 carries ONE `source_nd2_path` per well (P/Z/C are array
   axes in one ND2). The YX1 backend asserts `position_index.nunique()==1` and
   `source_nd2_path.nunique()==1`. Keyence is **one TIFF per (tile × z × channel × time) plane** —
   MANY source paths per frame. Those single-source asserts live ONLY in the YX1 backend, so they
   don't block Keyence; the Keyence backend gathers many `source_tiff_path`s per frame.

5. **Keyence RE-ACQUISITION / collisions** — the deferred eligibility track. Authoritative spec:
   `run_well_schema.md`. Canonical test case `20260414_b9d2_14hpf_plate02_B03` (6 tiles where the
   modal/expected count is 3). Needs conflict detection on the full acquisition-cell key →
   `resolve_acquisitions` → `well_acquisition_summary__keyence.csv` → `well_runner ∩ eligible`.
   **A larger, separate sub-track — do not fuse it with basic stitching.**

---

## 📐 Sequencing — what must finish first, what can start now

**Verified on-disk state (2026-06-18):**
- The YX1 per-well materialize interface is **built and LIVE** — `rule materialize_well`
  (`rules/frame_inventory.smk`) calls the new `image_materialization/` backend, emits a per-well
  `frame_inventory` shard, and `rule front_half` drives it through validation. **Beat 1 Step 6
  (finish line) has effectively landed.**
- **What's still moving:** Beat 2 (repoint segmentation/features/QC off `frame_contract.csv` onto
  the per-well shard — `rule segment_and_track_per_well` still reads `frame_contract.csv`) and
  Beat 1 Step 7 (strangle the legacy producer chain).

**The de-risking insight:** the contract Keyence *emits* (`frame_inventory_contract.py`) is
**frozen** (Step 6 landed). What's mid-migration in Beat 2 is the *consumer* side (segmentation
reading the shard), which is **downstream of** the seam Keyence produces, and is scope-agnostic +
shared with YX1. **So Keyence's producer-side work does not chase Beat 2.**

**One coupling to remember:** legacy `metadata_ingest/stitched_index/materialize_stitched_images.py`
(43 microscope branches) is the *shared* file that still runs production Keyence today. YX1 Step 7
can only strip its YX1 branches, not delete it, **until Keyence also migrates off it.**

```
START NOW (no dependency on Beat 2):
  • Stage A  — Keyence acquisition inventory (pure metadata_ingest; mirror of YX1; feedstock for all)
  • Stage C interview — the open design decisions below (design only)
  • Stage E fixture capture — the B03 reacquisition case (once Stage A lands)

THEN (the frame_inventory seam is already frozen, so these can follow Stage A directly):
  • Stage B  — per-well Keyence mosaic backend + route it
  • Stage C  — stitch-map generation + consumption (with B)
  • Stage D  — comparison gate (per-well vs legacy batch), human visual sign-off

SEPARATE LATER BEAT:
  • Stage E  — re-acquisition / eligibility (resolve_acquisitions, well_acquisition_summary,
               well_runner ∩ eligible) with the B03 test plan
```

---

## Stage A — Keyence acquisition inventory

**Goal:** the twin of `scope/yx1/acquisition_inventory.py` — maximal per-coordinate raw record,
Tier-1 shared core + Tier-2 Keyence extras, recording the MANY source TIFFs per frame, wired into
`ingest_scope_metadata`. **Record everything; collapse nothing** (no `drop_duplicates`).

> **⚠️ Feedstock note (from §8.3 resolution):** do NOT build the inventory from the current
> `extract_scope_metadata.py` output — it collapses to one FF-style row per (well,time) with
> `z_position=0`. The acquisition inventory needs the **raw per-Z-plane grain**. Lift the parsing
> nucleus from the legacy materializer: `materialize_stitched_images.py::_infer_keyence_stack_lookup`
> (`rglob("*CH*.tif")` → `{(well,time): {tile_id: [z-paths]}}`), `_parse_keyence_time_and_z`
> (the `_Z###_CH#` + `T####` grammar), and `_extract_keyence_well_and_tile`. These move into the
> Stage-A builder (or a shared `scope/keyence/` parsing module both the inventory and the legacy
> materializer can call during the strangler overlap).

**Files**
- CREATE `src/data_pipeline/metadata_ingest/scope/keyence/acquisition_inventory.py`
  (Contract → Validation → Builder banners, mirroring the YX1 file). Builder discovers raw
  `*CH*.tif` planes and explodes one row per `(well, tile, z_index, channel_index, time_index)`.
- (Likely) CREATE a shared `scope/keyence/raw_plane_parsing.py` holding the lifted parsers so the
  inventory and the legacy materializer share ONE grammar during the overlap.
- EDIT `scope/keyence/extract_scope_metadata.py` — add `acquisition_inventory_csv: Path | None`;
  emit the inventory from a raw-plane scan (NOT the collapsed FF rows it builds today).
- EDIT `pipeline_orchestrator/tasks.py::cmd_extract_scope` — pass it through the Keyence branch.
- EDIT `pipeline_orchestrator/Snakefile::rule ingest_scope_metadata` — widen the `MICROSCOPE ==
  "YX1"` gate on the output dict + `--acquisition-inventory-csv` arg so Keyence emits it too.
- CREATE `tests/data_pipeline/metadata_ingest/scope/keyence/test_acquisition_inventory.py`.

**Tier-2 Keyence schema (proposal — confirm):** Tier-1 SHARED core unchanged. Tier-2 extras:
`position_index_within_well`, `tile_id` (raster order, drives `TileSpec.tile_id`),
`n_tiles_in_well` (the modal count Stage E compares against), `z_index` (explode when present —
parallel to YX1's never-collapse rule), `channel_index`, `acquisition_time_s` (the raw atom
`elapsed_time_s` is derived from), `objective_magnification`, `orientation` (vertical/horizontal,
feeds stitch config), `x_um/y_um` (NaN today, parity), and **`source_tiff_path` per ROW** (the key
difference — one source path per exploded coordinate, not one per well).

**Keyence cell key (load-bearing for Stage E):** `well_id, position_index, z_index, channel_index,
time_index_claimed` (per `run_well_schema.md`). The validator DECLARES this key and CALLS
`scope/shared/acquisition_checks.assert_unique_on_key` — the shared mechanic. **But** unlike YX1
(clean by construction), Keyence CAN collide, so Stage A's uniqueness check is **warn-only**
(recorded evidence); quarantine is Stage E's job (open decision §8.4).

**Source readability (mode flag):** add `assert_keyence_acquisition_sources_readable(df, ...)`
gated behind `validate_keyence_acquisition_inventory(df, check_sources=...)` — the same
one-validator-two-moments doctrine as YX1's `check_sources`. Stage-B calls it with
`check_sources=True` before reading tiles.

**Reused:** Tier-1 core contract, all `acquisition_checks` primitives, `time_helpers.
add_elapsed_time_columns`, `channel_normalization.validate_channel_id`, YX1 module as template.

**VERIFY:** `snakemake -n` declares `acquisition_inventory__keyence.csv`; on a clean experiment
row-count == #TIFFs, Tier-1 complete/non-null, µm/px>0, channel triple consistent, `elapsed_time_s`
finite/non-neg, every `source_tiff_path` exists; a normal multi-Z/multi-tile well is NOT flagged a
collision (z/position differ); tasks/Snakefile hold no dataframe logic.

---

## Stage B — per-well Keyence materialize backend + routing

**Goal:** executor `scope/keyence/materialize_well_keyence.py` turning ONE well's many TIFFs into
mosaic projection frames + frame-inventory rows, plus the two routing edits. Asserts
`xy_composition=='mosaic'`.

**Files**
- CREATE `src/data_pipeline/image_materialization/scope/keyence/materialize_well_keyence.py`
  (+ `__init__.py`).
- EDIT `scope/scope_resolver_for_materialization_plan.py::resolve_materialization_plan` — add
  `if scope_name == "keyence": return _resolve_keyence(...)` (the `_resolve_keyence*` functions
  already exist). Update the "deliberately NOT routed" docstring.
- EDIT `run_materialize_well.py` — replace the `scope != "yx1"` guard with a thin 2-way dispatch
  (yx1 / keyence / else raise). The docstring already anticipates this migration.
- VERIFY (likely no edit) `select_well_acquisition_rows.py` — it filters on `well_id`, returns all
  tile rows; the single-position assumption lives only in the YX1 backend.
- CREATE `tests/data_pipeline/image_materialization/scope/keyence/test_materialize_well_keyence.py`.

**Multi-TIFF source-gathering shape (the core difference).** Per `(channel_id, time_index)` frame:
1. Group the well's rows by tile (`tile_id` / `position_index_within_well`).
2. Per tile: gather its z-plane TIFFs (`source_tiff_path` per `z_index`), read, focus-stack into
   one 2D per-tile image via `LoG_focus_stacker` (same projection primitive as YX1; only the source
   gathering differs). *(Open decision §8.3: legacy reads pre-fused FF tiles — confirm raw-Z vs
   pre-fused.)*
3. Assemble `TileSpec(tile_id, image)` per tile in raster order.
4. `frame_tiler.stitch_frame_tiles(tile_specs, FrameTilingConfig(orientation, use_legacy_canvas,
   invert_intensity, ...), FallbackParams(master_params_path=<Stage C>))` → one mosaic. The
   `run_align=False` master-fallback path IS the generate-once/reapply-many quirk.
5. Write via `materialized_image_paths.projection_frame_path(...)` (unchanged) and emit ONE
   frame-inventory row with the SAME `_EMITTED_COLUMNS` shape as YX1 (+ optional stitch-QC
   provenance: `fallback_used`, `qc.passed`). `source_image_path` = the materialized **mosaic**
   file (raw TIFF paths live only in the acquisition inventory), so the shared contract is satisfied
   unchanged.

**Reused (do NOT rebuild):** `stitch_frame_tiles`/`FrameTilingConfig`/`FallbackParams`/
`FrameTileResult`, `LoG_focus_stacker`/`im_rescale`, all `materialized_image_paths` helpers,
`frame_inventory_contract` columns, the resolver's existing `_resolve_keyence*`.

**VERIFY:** resolver returns `mosaic` for keyence (`identity` raises, `auto`→`mosaic`); sequencer
routes keyence to the new backend with yx1 unregressed; one clean 3-tile well materializes N mosaic
frames passing the shared validator (Levels 1 + 1.5); a missing `source_tiff_path` raises a NAMED
error.

---

## Stage C — stitch-map (`master_params`) 🟡 NEXT

**Status:** the DAG shape and naming are locked. `PreComputeStitchParams` (renamed from
`FallbackParams` — the old name was backwards) is already in `frame_tiler.py` with a legacy alias.
`materialize_keyence_product_for_well` already accepts `master_params_path: Path | None = None`
and passes it as `PreComputeStitchParams(master_params_path=...)`. Stage C wires the pre-computed
map into that slot.

**DAG shape (locked):** experiment-grain pre-step → per-well fan.

```
ingest_scope_metadata
    ↓
rule build_keyence_stitch_map   ← runs ONCE per experiment (experiment-grain output)
    output: master_params.json (resolved via paths.py registry, never a raw string)
    ↓
rule materialize_image_product_for_well[B01]  ← lists master_params.json as INPUT
rule materialize_image_product_for_well[C01]  ← same input; Snakemake handles the fan
```

Snakemake builds the pre-step once; all per-well jobs wait on it automatically. No special
checkpoint/DAG machinery needed — this is the standard experiment-grain-then-per-well pattern.

**C1 — Registry entry (`orchestration/paths.py`):**
Add a Keyence-only step:
```python
"keyence_stitch_map": {
    "stage": "acquisition",          # lives beside the acquisition inventory
    "fanout": "experiment",
    "artifacts": {"master_params": "keyence_stitch_map__{scope}.json"},
}
```
Resolved by callers via `artifact_path(DATA_ROOT, "keyence_stitch_map", "master_params",
"{experiment}", format_vars={"scope": "keyence"})`. This path is what the Snakefile `output:`
declares and what the per-well rule lists as `input:`. No raw `master_params.json` strings anywhere.

**C2 — Builder (`image_materialization/scope/keyence/build_keyence_stitch_map.py`):**
```python
def build_keyence_stitch_map(
    acquisition_inventory_df: pd.DataFrame,
    *,
    n_samples: int = 50,
    out_path: Path,
) -> None
```
- Sample up to `n_samples` `(well_id, time_index)` pairs from the inventory (use a fixed seed so
  the output is deterministic/byte-identical for the same experiment).
- Per sample: group rows by `tile_id`, read each tile's z-plane TIFFs, focus-stack per tile via
  `materialize_ff_projection` (same shared primitive — DRY), pass `TileSpec` list to
  `stitch_frame_tiles(..., FrameTilingConfig(orientation=...), fallback=None)` with
  `run_align=True` to get real per-tile transforms.
- Collect `tile_transforms` from each `FrameTileResult`; take **median** `(dx_px, dy_px)` per
  `tile_id` across all sampled frames.
- Write `{"coords": {tile_id: [median_x, median_y], ...}}` JSON to `out_path`.

Source of orientation: read from `acquisition_inventory_df["orientation"].mode()[0]`
(the inventory owns it — no config key needed).

EDIT `tasks.py` — add thin `cmd_build_keyence_stitch_map`:
```python
def cmd_build_keyence_stitch_map(args):
    build_keyence_stitch_map(
        acquisition_inventory_df=pd.read_csv(args.acquisition_inventory_csv),
        n_samples=args.n_samples,
        out_path=args.output_json,
    )
```
Parser args: `--acquisition-inventory-csv`, `--output-json`, `--n-samples` (default 50).

EDIT `Snakefile` — Keyence-only rule (gate with `if MICROSCOPE == "Keyence":`):
```python
rule build_keyence_stitch_map:
    input:
        acquisition_inventory_csv = SCOPE_ACQUISITION_INVENTORY_CSV,
    output:
        master_params = str(artifact_path(DATA_ROOT, "keyence_stitch_map", "master_params",
                             FRONT_END_EXPERIMENT, format_vars={"scope": SCOPE_TOKEN})),
    shell: "{RUN} -m ... build-keyence-stitch-map ..."
```

**C3 — Per-well consumption:**
In `rule materialize_image_product_for_well` (Keyence gate), add the stitch-map artifact to
`input:` and pass `--master-params-path "{input.master_params}"` to the shell command.
`tasks.py::cmd_materialize_image_product_for_well` passes it through to
`materialize_keyence_product_for_well(master_params_path=args.master_params_path)`.
`materialize_keyence_product_for_well` already accepts this arg — no backend changes needed.

**C4 — Remove legacy alias:**
Once `materialize_stitched_images.py` is strangled (Stage D), delete the `FallbackParams =
PreComputeStitchParams` alias from `frame_tiler.py`.

**VERIFY:**
- `build_keyence_stitch_map` writes JSON with `coords` for all expected tile_ids; path resolves
  via `paths.py` (grep for raw `"master_params"` string in rules/tasks returns empty).
- Per-well materialize with the map: `FrameTileResult.fallback_used == "master"` for at least
  one frame; tile transforms have `source="fallback"`.
- Same inventory → byte-identical `master_params.json` (deterministic seed sampling).
- `grep -r "FallbackParams" src/` returns only the alias line in `frame_tiler.py` (no new uses).

---

## Stage D — comparison gate (per-well vs legacy batch)

**Why it matters more than YX1's:** per-well stitch reapplying a MEDIAN master map can diverge from
legacy BATCH stitch. Byte-identity is the IDEAL, not the gate — same 3-layer model as the YX1
roadmap (byte/hash → numeric diff on mismatch → human visual QC).

- Run the NEW backend with `candidate=True` (isolated `candidate/` tree, owned by
  `materialized_image_paths.py`) on the canonical experiment, alongside untouched legacy
  `materialize_stitched_images.py` Keyence output.
- REUSE the YX1 `stitch_candidate_qc/{experiment}/{well_id}/` convention
  (`comparison_summary.csv` with `same_bytes` / `max,mean,p99 abs-diff` / `human_review_status`,
  `side_by_side_frames/*_compare.jpg`, `side_by_side.mp4`), parameterized for Keyence. Migration
  evidence, NOT a Snakemake rule. No legacy edits (it must stay green for the comparison).
- **mdcolon's eyes are the REQUIRED final acceptance** before any legacy Keyence strangle.

**VERIFY:** comparison covers every frame of ≥1 multi-tile well; numeric diffs on mismatches;
side-by-side `.mp4`; mdcolon signs off. Only then may legacy Keyence stitch be strangled (later
separate commit, mirroring YX1 Step 7 — and only then can `materialize_stitched_images.py` finally
be deleted, since stripping its YX1 branches alone cannot remove the file while Keyence used it).

---

## Stage E — re-acquisition / eligibility (its own beat)

After basic Keyence works. Authoritative spec: `run_well_schema.md`. Test case
`20260414_b9d2_14hpf_plate02_B03` (6 tiles, modal 3). **No `drop_duplicates(keep=...)` anywhere.**

Flow: `acquisition_inventory__keyence.csv` (Stage A evidence) → conflict detection on the full
cell key → `resolve_acquisitions` classifies (exact dup→collapse; byte-identical→collapse;
different bytes→real conflict; partial→malformed) → `acquisition_conflicts__keyence.csv` +
`well_acquisition_summary__keyence.csv` (`well_id, active_for_stitch, quarantine_reason`) +
`resolved_acquisition_inventory__keyence.csv`. `well_runner` reads ONLY the shared summary cols.

**Files**
- CREATE `scope/keyence/resolve_keyence_acquisitions.py` (resolver; reuses `assert_unique_on_key`).
- CREATE `metadata_ingest/contracts/well_acquisition_summary.py` (the roadmap's deferred shared
  contract, now forced by Keyence).
- EDIT `orchestration/paths.py` — DEFERRED registry rows: `acquisition_conflicts__{scope}.csv`,
  `acquisition_resolution__{scope}.csv`, `resolved_acquisition_inventory__{scope}.csv`,
  `well_acquisition_summary__{scope}.csv` (experiment-grain).
- EDIT `orchestration/well_runner.py` — add the `∩ eligible` term, reading ONLY `well_id,
  active_for_stitch, quarantine_reason`. Stays absent on YX1.
- EDIT `Snakefile` — Keyence-only experiment-grain `rule resolve_acquisitions` between ingest and
  the per-well fan; wire the summary into `well_runner`.
- EDIT the Keyence per-well `materialize_well` input — consume
  `resolved_acquisition_inventory__keyence.csv` (so a forced `materialize_well[B03]` fails loud).
- CREATE tests mirroring the `run_well_schema.md` test plan (Inventory / Conflict / Resolution /
  Run-Well Selection / Stitch Scheduling / Manual Resolution), keyed on B03.

**VERIFY (full `run_well_schema.md` plan):** B03 in `discovered_wells.txt`; conflicts file records
B03's duplicated full-cell-key rows + candidate groups explaining 6 vs 3; summary marks B03
`active_for_stitch=false` with a non-empty quarantine reason; resolved inventory has no active B03;
`run_wells` excludes B03 but includes clean targets (explicit skip); forced B03 fails loud; manual
`acquisition_resolution__keyence.csv` flips B03 active; grep confirms no `drop_duplicates(
keep="first")` on any Keyence path.

---

## 8. OPEN DESIGN DECISIONS (need mdcolon's call before building B/C/E)

1. **Stitch-map grain & home (Stage C):** recommended experiment-grain pre-step + registry JSON,
   consumed per-well. Confirm (a) pre-step rule vs lazy-on-first-well; (b) registry JSON vs
   off-registry pixel-sibling.
2. **Per-well stitch vs experiment-batch stitch:** is reapplying a median master map per well
   scientifically acceptable, or must stitching stay a batch step (which would diverge the Keyence
   backend grain from `materialize_well[well_id]`)? Stage D's gate answers empirically — decide the
   acceptance bar.
3. **FF source: pre-fused tiles vs raw Z TIFFs — ✅ RESOLVED 2026-06-18 (raw Z).** Inspection of
   the legacy materializer settled this. Raw Keyence data on disk is **per-Z-plane, per-channel,
   per-tile TIFFs** named `...XY##_NNNNN_Z###_CH#.tif` (e.g. `embryo__XY16_00003_Z001_CH1.tif`).
   The legacy path `rglob("*CH*.tif")` → parses `(well, tile_id, time_int, z_index)` via
   `materialize_stitched_images.py::_extract_keyence_well_and_tile` + `_parse_keyence_time_and_z`
   → groups `{(well,time): {tile_id: [z-paths]}}` (`_infer_keyence_stack_lookup`) → **focus-stacks
   each tile's Z-stack on the fly** via LoG (`_materialize_keyence_ff_tiles_with_log`, same
   primitive as YX1) → mosaics. "Pre-fused FF tiles" was a misread — that's the legacy
   *intermediate* (`FF_images/`), not the raw input. **Cascades:** (a) the new backend focus-stacks
   raw Z per tile via `log_focus` (matches YX1); (b) the Stage-A inventory explodes one row per
   `(well, position/tile, z_index, channel_index, time_index)`, never collapsing — like YX1;
   (c) the reacquisition cell key `(well_id, position_index, z_index, channel_index,
   time_index_claimed)` is fully realizable from the raw filename grammar.
   **⚠️ Consequence for Stage A:** the current `extract_scope_metadata.py` is the WRONG feedstock —
   it hardcodes `z_position=0` and emits one collapsed row per (well,time). The Keyence acquisition
   inventory must be built from the same raw-plane parsing the legacy materializer already does;
   `_infer_keyence_stack_lookup` + `_parse_keyence_time_and_z` + `_extract_keyence_well_and_tile`
   are the reusable parsing nucleus to lift into the Stage-A builder.
4. **Stage-A collision stance:** Keyence validator WARN (not fail) on cell-key duplication so basic
   stitching ships before eligibility? (Plan assumes yes — relax in A, quarantine in E.)
5. **Does eligibility (Stage E) ship with basic Keyence or later?** Roadmap defers it; this spec
   sequences it separately. Confirm Keyence can go live for CLEAN experiments first, with warn-only
   as the bridge.
6. **Snakefile scope-gating style:** keep inline `if MICROSCOPE == "..."` rule gates, or factor a
   scope-rule-selection mechanism as Keyence-only rules accumulate (stitch-map,
   resolve_acquisitions)? (The anti-`if YX1/elif Keyence` rule targets FUNCTIONS, not rule
   emission — so inline gates may be acceptable; confirm.)

---

## Critical files map (reuse vs new)

| Concern | File | Reuse / New |
|---|---|---|
| Plan vocabulary | `image_materialization/materialization_plan.py` | reuse — no change |
| Scope resolver | `image_materialization/scope/scope_resolver_for_materialization_plan.py` | edit — route `_resolve_keyence` (already written) |
| Sequencer | `image_materialization/run_materialize_well.py` | edit — 2-way backend dispatch |
| Handoff seam | `image_materialization/frame_inventory_contract.py` | reuse — must stay agnostic |
| Image paths | `image_materialization/materialized_image_paths.py` | reuse — already generic |
| Stitch engine | `image_building/utils/frame_tiler.py` | reuse — `master_params` reapply path |
| Focus stack | `image_building/shared/log_focus.py` | reuse |
| Legacy stitch-map algo | `image_building/scope/keyence/stitched_ff_builder.py::build_master_params` | port into Stage C |
| Acq inventory (Keyence) | `metadata_ingest/scope/keyence/acquisition_inventory.py` | **CREATE** (Stage A) |
| Mosaic backend | `image_materialization/scope/keyence/materialize_well_keyence.py` | **CREATE** (Stage B) |
| Stitch-map build | `image_materialization/scope/keyence/build_keyence_stitch_map.py` | **CREATE** (Stage C) |
| Reacquisition resolve | `metadata_ingest/scope/keyence/resolve_keyence_acquisitions.py` | **CREATE** (Stage E) |
| Eligibility contract | `metadata_ingest/contracts/well_acquisition_summary.py` | **CREATE** (Stage E) |
| Legacy Keyence stitch (still live) | `metadata_ingest/stitched_index/materialize_stitched_images.py` | strangle AFTER Stage D |
