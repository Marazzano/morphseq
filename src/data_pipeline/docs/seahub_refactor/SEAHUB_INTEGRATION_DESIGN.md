# SeaHub → morphseq pipeline: integration DESIGN & CONTRACTS

**Status:** design frozen, ready to implement.
**Companion:** `SEAHUB_INTEGRATION_WORKPLAN.md` (the executable, phased plan).
**Scope of this doc:** the *decisions* and the *contracts* the implementing agent MUST NOT violate.
The workplan is procedural and may change during execution; this doc is the stable reference.

---

## 0. One-paragraph summary

SeaHub images are single-shot, single-focal-plane RGB FOVs (1280×960) each containing **8 embryos**,
with no plate, no well, no z-stack, and no time series. We integrate them **entirely in the front
half**: a standalone materialization step (a) GroundingDINO-detects the 8 embryos per FOV, (b) pads
each box-crop to a standard grayscale frame, and (c) writes each embryo as its own **single-frame
synthetic well** into the acquisition output tree in the exact layout the back half already consumes,
plus a **per-well plate-map** carrying that embryo's perturbation, stage, and sequencing link. The
back half (detection → segmentation → snips → features → embeddings → analysis_ready) then runs
**completely unmodified** ("Scheme A", below). The only cost is cosmetic false QC flags, which
*annotate but never drop*, so the data still reaches `analysis_ready`.

---

## 1. Why Scheme A (pass-through as "dummy FF") and not a pipeline rewrite

### 1.1 The two candidate schemes considered
- **Scheme A — pass-through:** materialize SeaHub embryos into the FF/projection product slot so the
  unmodified back half treats them like any single-timepoint experiment.
- **Scheme B — new scope + QC skip-list:** add a bespoke materialization scope and config guards that
  disable specific QC products for SeaHub.

### 1.2 Why A wins decisively
- **QC annotates, it does not gate.** `analysis_ready` is a LEFT JOIN onto `snip_qc`, which "owns the
  authoritative snip_id universe" — see `analysis_ready/assemble.py` docstring and
  `config.yaml` (Phase 4 comment): *"product emits one row per snip regardless of validity (QC
  excludes, not feature extraction)."* A flagged SeaHub snip still lands in `analysis_ready` carrying
  its flag. Whoever trains filters on the flag; nothing is deleted.
- **The temporal-crash risk is empirically dead.** Many existing Keyence datasets are already
  single-timepoint and passed the back half cleanly. Verified:
  `20250612_24hpf_ctrl_atf6` and `20240813_24hpf` have `time_int` nunique = **1** (vs 61 for the
  time-lapse `20230525`). So `death_detection` / `pose_kinematics` already tolerate length-1 wells.
- **Result:** Scheme B's only remaining advantage over A is cosmetic (no false focus/motion flags in
  QC reports) — not worth an extra module + config surface. **Use Scheme A.**

### 1.3 QC/feature products that will misbehave on SeaHub (and why it's harmless)
All of these **annotate only** — they never remove the snip from `analysis_ready`:

| Product | Behavior on SeaHub | FF/temporal assumption | Consequence |
|---|---|---|---|
| `focus_qc` | false-flags ~all | FF is all-in-focus; a single z-plane is not | `focus_flag=True`, ignored downstream for seahub rows |
| `motion_blur_qc` | false-flags some | FF-sharpness baseline | `motion_blur_flag=True`, ignored |
| `death_detection` | null/no-op | needs a time series | nulls; harmless (proven on 1-frame Keyence) |
| `pose_kinematics` | null/no-op | cross-time deltas | nulls; harmless |
| `surface_area` outlier QC | possibly noisy | stage reference built on Keyence/YX1 optics + µm/px | uncalibrated (see §6); annotate-only |

**Downstream contract for the training loader:** filter/ignore `focus_flag` and `motion_blur_flag`
where `source_scope == 'seahub'`. This is a *consumer-side* rule, not a pipeline change.

---

## 2. Identity scheme (the core contract)

### 2.1 The hard constraint that shapes everything
`well_index` is a **validated 8×12 plate grid** — `shared/identifiers/validators.py`:
`_WELL_INDEX_CANONICAL_RE = ^([A-Ha-h])(\d{2})$`, rows A–H, cols 01–12, **max 96 wells**.
`validate_well_id` fails loud on anything that isn't `{experiment_id}_{well_index}`.
Arbitrary synthetic well ids (e.g. `GENE13_foxa2_5p03`) are **rejected at the ingest boundary**.
Therefore SeaHub identity MUST conform to the plate grid.

### 2.2 The scheme: pack into 96-well synthetic plates, one embryo per well
- **One embryo → one well → one physical embryo → one single frame.**
- Group ingested embryos by **SeaHub experiment** (`GENE13`, `CHEM17`, …), then pack into plates of
  ≤96 embryos in a deterministic order (see §2.4). ~10,900 embryos → **~114 plates**, comparable to
  the existing ~130 pipeline experiments; model loads amortize over up to 96 embryos per plate.
- The "plate" is a pure container. **All biology lives in the per-well plate-map** (§3), exactly as a
  real plate assigns per-well genotype.

### 2.3 Identifier construction (use `shared/identifiers` constructors — never hand-format)
```
experiment_id       = {DATE}_seahub_{EXPT}_plate{NN:02d}     e.g. 20260723_seahub_GENE13_plate01
well_index          = canonical A01..H12  (via normalize_well_index(row, col))
well_id             = build_well_id(experiment_id, well_index)          → {experiment_id}_{well_index}
physical_embryo_id  = build_physical_embryo_id(well_id, 1)              → {well_id}_e01
image_id            = build_image_id(well_id, "BF", time_index=0)       → {well_id}_BF_t0000
```
- `{DATE}` = `20260723` (the SeaHub integration date; keep constant across the corpus).
- `{EXPT}` = the SeaHub experiment token (`GENE*`/`CHEM*`) from the reconciliation `experiment_id`.
- `channel_id = "BF"` (must be in `VALID_CHANNEL_NAMES`; it is). SeaHub RGB → grayscale on write.
- **Single frame only:** `time_index = 0`; **no z index in the image_id** (projection grammar).
  The z-slice nature is carried in metadata columns (§4), NOT in the id or path.

### 2.4 Packing order (deterministic, reproducible)
Within a SeaHub experiment, sort embryos by `(stage_hpf, perturbation_key, fov_label, fov_position)`,
then fill wells column-major or row-major (pick one; **row-major A01,A02,…,A12,B01,…** recommended),
spilling to `plate{NN+1}` after H12. Persist the mapping (§3, §5) so every synthetic well is traceable
back to its exact source FOV + within-FOV position.

---

## 3. Metadata integration (instruction iii) — reconciliation → pipeline standard

### 3.1 Source of truth
`outputs/image_metadata_reconciliation.csv` (1,925 rows). Filter to `image_role == 'eight_embryo_fov'`
(1,392 FOVs). Per-embryo rows come from the GroundingDINO detection (8 per FOV) in
`outputs/segmentation_validation_3_experiments/embryo_manifest.csv` (schema for the box columns) and
the reconciliation join.

### 3.2 Ingest scope — ingest on IDENTITY, not on exact-match
Exact-match-only would discard ~5,000 usable embryos (**mostly controls**). Decision: ingest any FOV
with a usable identity `(experiment_id AND stage_hpf AND perturbation_parsed)`; carry the sequencing
link and match status as columns.

| Tier | FOVs | Embryos | Ingest? |
|---|---:|---:|---|
| identity + sequencing link (`metadata_collection_name` present) | 737 | ~5,900 | **yes** |
| identity only, no seq link (mostly `ctrl-inj`/controls) | 624 | ~5,000 | **yes** (flag `has_seq_link=False`) |
| no usable identity (somite-only `12s`, "not collected", pheno-grids) | 31 | ~250 | **mostly recoverable** — see §3.4 |

### 3.3 Field mapping: reconciliation column → pipeline column
Written into the per-well **plate_metadata.csv** (one row per well_index) unless noted.
`genotype` vs `chem_perturbation` is chosen by `perturbation_domain`/experiment prefix (`GENE*`→genotype, `CHEM*`→chem).

| pipeline column (plate-map) | source (reconciliation) | notes |
|---|---|---|
| `well_index`, `well_id`, `experiment_id` | minted (§2.3) | identity spine |
| `genotype` | `perturbation_parsed` if GENE* else `"ctrl"`/null | canonical genotype token |
| `chem_perturbation` | `perturbation_parsed` if CHEM* else null | chem name |
| `start_age_hpf` | `stage_hpf` | imaging stage = start age for a snapshot |
| `stage_hpf` | `stage_hpf` | explicit broadcast copy |
| `stage_addition_hpf` | `stage_addition_hpf` | chem addition timing (where present) |
| `embryos_per_well` | `1` | one embryo per synthetic well |
| `seahub_experiment` | `experiment_id` (GENE13/CHEM17) | provenance |
| `source_fov` | `fov_label` + filename `stem` | traceability |
| `fov_position` | `embryo_position` (1–8) | within-FOV index |
| `collection_name` | `metadata_collection_name` | **sequencing join key** |
| `collection_batch` | `metadata_collection_batch` | seq linkage |
| `collection_expt` | `metadata_expt` | seq linkage |
| `collection_ancestors` | `metadata_ancestors` | seq linkage |
| `has_seq_link` | derived (`collection_name` non-empty) | filter flag |
| `metadata_match_status` | `metadata_match_status` | `exact` / `unmatched_condition` / … |
| `metadata_match_score` | `metadata_match_score` | confidence |
| `source_scope` | `"seahub"` | provenance; the training-loader filter key |
| `image_kind` | `"single_z"` | see §4 |
| `z_position` | null (unknown token) | see §4 |
| `micrometers_per_pixel` | null/placeholder | see §6 (uncalibrated) |

These non-spine plate columns reach `analysis_ready` via `analysis_ready.contract.broadcast_plate_columns`,
which broadcasts per-well plate columns onto every snip. **Confirm the exact broadcast set** when wiring
(some columns may need to be added to the broadcast allow-list).

### 3.4 The ~31 no-identity FOVs
Almost all are somite-stage labels (`12s`, etc.) that DO carry experiment + perturbation; they were
left unconverted only because no somite→hpf table was supplied. **User will provide a somite→hpf
conversion table**; apply it to recover these (set `stage_hpf` from the table, keep the raw token in
`stage_source_label`). Only genuine non-data images ("not collected", pheno-grids, closeups) are
dropped — enumerate them explicitly in a `dropped_fovs.csv` with the reason (no silent drops).

---

## 4. The z-slice metadata (the model-training requirement)

The refactored VAE will ingest, as **standard metadata**, (i) whether a frame is a z-slice vs an FF
projection, and (ii) its z position — a real number when known, a special FF token, and a **null
token when unknown/masked** (trained for tolerance). SeaHub frames are single arbitrary focal planes,
so they must be classed as **z-slice of unknown z**:

```
image_kind = "single_z"      # NOT "projection"
z_position = null            # the unknown/masked token
```

**Critical nuance already reconciled with the pipeline:** the projection-vs-z distinction is currently
a *materialization/path* concept and is NOT carried as a column past materialization (verified: no
`is_projection`/`z_position` references in feature/snip/analysis_ready code). So:
- For **Scheme A path compatibility**, the frame is written into the `projection/` slot with a
  projection-grammar `image_id` (no `_z` token) so the back half treats it as FF.
- For **model training**, the truth rides in the `image_kind` + `z_position` **metadata columns**
  (plate-map → broadcast → analysis_ready), which is exactly the standard metadata the refactored
  loader will read. Path says "projection"; metadata says "single_z / unknown z". This is intentional
  and consistent with the user's design.

If the refactored loader's column names differ from `image_kind`/`z_position`, rename to match its
contract at wiring time — the *values* (single_z, null-token) are the fixed requirement.

---

## 5. Materialization output contract (what the front half must produce)

Mirror a real experiment's acquisition tree so the back half finds everything it expects. Reference a
finished experiment (e.g. `.../acquisition/20240813_24hpf/`) for exact shapes. Per synthetic experiment:

```
acquisition/{experiment_id}/
  materialized_images/{well_id}/BF/projection/{well_id}_BF_t0000.jpg   # padded grayscale embryo frame
  ingest_metadata/
    scope_metadata_mapped.csv      (+ .validated)   # one row per image_id (see §3, 29-col schema)
    plate_metadata.csv             (+ .validated)   # one row per well_index (the per-well biology, §3.3)
  well_identities/
    position_well_mapping.csv                        # position_index → well_id map
    discovered_wells.txt                             # global well_ids, one per line
  frame_inventory/...                                # product inventories the front-half validator checks
```
- **Image:** GroundingDINO box-crop → pad to standard size (§ instruction ii, size rule below) →
  grayscale (`mode L`) → JPEG (match `config.yaml` `jpeg_quality`). Write to `projection/`, `t0000`.
- **Standard padded size:** compute the max box width and height over ALL ingested embryos, round up
  to round numbers, and **pad (never downscale)** every crop centered to that fixed `(H, W)` with a
  neutral background fill. Rationale: uniform batch dims; embryo never touches the edge (margin for
  downstream rotation/segmentation). The final training snip is resized to
  `snip_frame_shape = [576, 256]` downstream regardless, so this frame only needs to be a clean,
  consistent, embryo-with-margin canvas. Fill value is a smoke-test parameter (§ workplan Phase 4);
  start with a light/background-estimate fill so segmentation does not detect the padding.
- **Inventories:** the front half validates materialized products against `frame_inventory` product
  inventories. Use the **drop-in scaffold path** (`acquisition/metadata_ingest/frame_inventory/
  scaffold_dropin_inventory.py`) to generate these for externally-materialized images rather than
  re-implementing the inventory writer. This is the minimal-change route (see workplan Phase 3).

---

## 6. Known caveats (document, don't silently absorb)

- **µm/px is unknown for SeaHub** → set null/placeholder. All surface-area / physical-scale features
  and SA-outlier QC are therefore **uncalibrated** for SeaHub. Harmless under Scheme A (annotate-only;
  the VAE trains on pixel-space snips), but any physical-size analysis must exclude `source_scope==
  'seahub'` or supply a calibration later.
- **False focus/motion flags** on ~all SeaHub rows — expected; consumer ignores them (§1.3).
- **Packing is a fiction:** a synthetic "plate" mixes perturbations/stages. This is fine because all
  biology is per-well, but never treat a SeaHub plate as a shared physical condition.
- **RGB→grayscale** loses color; SeaHub brightfield is effectively monochrome, so acceptable, but note
  it if any SeaHub imaging used color-informative stains.

---

## 7. Files the implementing agent must read first

- `src/data_pipeline/shared/identifiers/{constructors,parsers,validators}.py` — id grammar (non-negotiable).
- `src/data_pipeline/acquisition/image_materialization/scope/yx1/materialize_well_yx1.py` — the closest
  materialization template (single-tile, no stitch).
- `src/data_pipeline/acquisition/image_materialization/scope/scope_resolver_for_materialization_plan.py`
  — how scopes register (only if a real scope is chosen over drop-in).
- `src/data_pipeline/acquisition/metadata_ingest/frame_inventory/scaffold_dropin_inventory.py` — the
  drop-in inventory scaffolder (the minimal-change route).
- `src/data_pipeline/acquisition/metadata_ingest/plate/plate_metadata_loader.py` — per-well plate-map schema.
- `src/data_pipeline/analysis_ready/contract.py` — `broadcast_plate_columns`, spine columns.
- `results/nlammers/20260723_seahub/seahub_workflow.py` — existing GroundingDINO detection + position
  assignment + cropping to REUSE (don't reimplement detection).
- `results/nlammers/20260723_seahub/run_sam2_masks.py` — reference only (masks NOT used in Scheme A;
  instruction ii is box-crop + pad, no mask).
- A finished acquisition tree, e.g. `.../pipeline/output/acquisition/20240813_24hpf/`, as the layout
  ground-truth to diff against.
