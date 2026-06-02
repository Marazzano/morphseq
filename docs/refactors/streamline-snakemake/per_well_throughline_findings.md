# Per-Well Through-Line — Findings & North Star

**Status:** active thinking doc. Keep the goal pinned at the top; append findings below.
**Owner:** mdcolon
**Companion to:** `well_id_throughline_refactor_plan.md` (the formal scopes/plan).
**Verified against disk + Snakefile:** 2026-06-02.

---

## 🎯 THE GOAL (pin this — everything serves it)

> **Cross the experiment-grain bootstrap once; after the well list exists, run each well independently end-to-end.**
> The whole point of this refactor is to make a single well flow through the entire
> pipeline as an independent unit. Once that works, we map the file structure (`src/`
> and data output) *onto that shape* — the grain drives the layout, not the reverse.

Corollary goals that fall out of it:
- After the well list is known, **everything should be embarrassingly parallel** — one
  well per process, no cross-well barrier.
- The registry / layout should make it trivial to see, for any stage, **what artifacts
  it produces and at what grain.**

---

## 🔑 THE CENTRAL INSIGHT (the thing that reframes everything)

**Zone A cannot be per-well. That is physics, not a design flaw.**

You cannot "run well B01 end-to-end" starting from raw microscope data, because *at
that moment B01 does not yet exist as a known entity.* The raw ND2 / Keyence file is a
single blob of microscope **series**. The entire job of the front of the pipeline
(scope-metadata extraction → series→well mapping) **is the act of discovering which
wells are in the file.**

> **Well identity is an OUTPUT of the front stages, not an input.**

So the pipeline has a mandatory, irreducible **experiment-grain bootstrap prefix**
whose only job is to *manufacture the well list*. Per-well parallelism is only possible
*after* that prefix. The honest shape is:

```
  EXPERIMENT-GRAIN BOOTSTRAP            ──►   FAN POINT   ──►   PER-WELL, PARALLEL
  (process the whole experiment file          (the well        (light; one well = one
   to discover what wells exist)               list exists)      independent row, E2E)
```

This resolves the conflict that kept surfacing ("how do we make everything per-well
when some things clearly aren't?"). Answer: **don't.** Name the prefix for what it is
(a bootstrap that discovers wells), make the fan point explicit, and let `per_well` be
the contract for *everything after* it.

---

## 🌾 THE GRAIN TRANSITION (verified from the active Snakefile, 2026-06-02)

Three zones. Transition points are sharp.

### Zone A — EXPERIMENT-GRAIN BOOTSTRAP  (must process the whole experiment)
Families: `experiment_metadata/`, `built_image_data/`
| Stage | Output (under `{output_root}`) | Why it can't be per-well |
|---|---|---|
| `normalize_plate_metadata` | `experiment_metadata/{exp}/plate_metadata.csv` | the plate *is* the experiment |
| `extract_scope_metadata_{scope}` | `experiment_metadata/{exp}/scope_metadata__{scope}.csv` | reads the whole raw file |
| `map_series_to_wells` | `experiment_metadata/{exp}/series_well_mapping.csv` (+`.provenance.json`) | **this is where wells are discovered** |
| `apply_series_mapping` | `experiment_metadata/{exp}/scope_metadata_mapped.csv` (+`.validated`) | joins mapping onto scope rows |
| `build_stitched_images` | `built_image_data/{exp}/stitched_ff_images/{well}/{channel}/` | per-well *images*, but driven by the discovered well list (see fan-point note) |
| `build_frame_contract` | `experiment_metadata/{exp}/frame_contract.csv` | the canonical frame-level contract |

### ⟱ FAN POINT — the well list now exists ⟱
The first artifact that knows the full well set is **`series_well_mapping.csv`**
(wells discovered) → solidified by **`frame_contract.csv`**. Everything after this can,
in principle, run one-well-at-a-time. (`build_stitched_images` already iterates the
discovered wells — likely the natural seam; confirm the "available wells" mechanism.)

### Zone B — PER-WELL, THEN MERGED  (the only place `per_well/` exists today)
Families: `segmentation_and_tracking/`, `processed_snips/`
| Stage | Per-well shard | Merged artifact |
|---|---|---|
| `segment_and_track_per_well` → `merge_segmentation_tracking` | `segmentation_and_tracking/{exp}/per_well/{well_id}/contracts/segmentation_tracking.csv` | `segmentation_and_tracking/{exp}/contracts/segmentation_tracking.csv` |
| `run_snip_processing_per_well` → `merge_snip_manifests` | `processed_snips/{exp}/per_well/{well_id}/contracts/snip_manifest.parquet` | `processed_snips/{exp}/contracts/snip_manifest.parquet` |

This is **already** the plan's Scope-5 pattern (per-well shard + merge), built and
working — but only for these two stages.

### Zone C — currently experiment-grain, but CONFIRMED per-well-able
Families: `computed_features/`, `quality_control/`, `analysis_ready/`
- Layout is 3-level: `family/{exp}/{stage_subfolder}/{file}` (+ `.validated` sentinel beside almost every CSV).
  - e.g. `computed_features/{exp}/mask_geometry/mask_geometry_metrics.csv`,
    `computed_features/{exp}/consolidated/consolidated_snip_features.csv`,
    `quality_control/{exp}/death_detection/death_detection_flags.csv`.
- **Experiment-grain in the layout today** — no `per_well/` written under Zone C yet.
- **But it is per-well-able with certainty (mdcolon, known fact, 2026-06-02): every
  Zone-C computation is per-snip / per-embryo — there is NO cross-well computation.**
  Zone C is experiment-grain *only because* it reads the **merged** Zone-B contract.
  **The merge is the sole barrier**, not any real cohort statistic. Feed Zone C a single
  well's slice and every stage produces correct per-well output. This is no longer a
  thing to verify — it is the green light to push Zone C to per-well (the target model
  below).

---

## 🧭 DESIGN PRINCIPLES WE'VE LOCKED (from this discussion)

Three orthogonal axes — keep them separate or the model gets muddy:

| Axis | Question | Lives in | Notes |
|---|---|---|---|
| **identity** | "What is this object's canonical name?" | the **data** | `well_id = {exp}_{well}`, the through-line in *rows* AND in the canonical post-fan path. |
| **layout** | "Where does this artifact live under the root?" | the **registry (code)** | `family / {exp} / [per_well/{well_id}/] [subfolder] / file`. A fixed contract. |
| **fanout** | "Does *this stage* shard per well?" | the **registry (per stage)** | `experiment` (Zone 0 bootstrap) vs `per_well_then_merge` (everything post-fan: Zones B **and** C). This field *is* the grain-transition map. |

Pinned mantra:
> **Bootstrap is experiment-grain. After well discovery, computation is per-well.
> Merged artifacts are publication products. Scoped runs are optional scratch, not
> architecture.**

Decisions adopted from the discussion:
- **Per-well-canonical is the target (DECISION, 2026-06-02).** The real output tree
  carries `{family}/{exp}/per_well/{well_id}/...` for *every* post-fan stage (Zones B
  and C alike). "Run one well" is then **not a special mode** — it is just targeting
  that one well's canonical outputs and letting Snakemake's per-file staleness do the
  rest. This **supersedes** the earlier "root-rebasing first" stance.
- **`scoped_runs/` is demoted to optional scratch/dev — NOT architecture.** Root
  rebasing (`config.yaml` `target_wells` → `{data_root}/scoped_runs/wells-.../`) was an
  *approximation* of per-well targeting, needed only while the main tree was
  experiment-grain. Once canonical `per_well/` exists, it is redundant. Keep it as a
  scratch lever; do **not** build the registry around it; revisit removal after the
  post-fan per-well conversion is complete.
- **Every post-fan stage follows the same rhythm:** `run_X_per_well → merge_X →
  validate_X`. Merged artifacts become thin **publication products** at the end of the
  DAG, not a mid-pipeline barrier. Snakemake then gives incremental single-well
  recompute for free.
- **Registry is two levels only:** `stage → artifact`. No deeper nesting. `family`/
  `subfolder` are real on-disk folders (flat, as verified — e.g. many stages share
  `experiment_metadata/`); `fanout` is per-stage; `kind` ∈ {primary, sidecar, sentinel,
  report}; `schema` per artifact. Consider `table_grain` (frame/snip/embryo) separate
  from file placement so the two don't get conflated.
- **One pipeline-wide registry** in `pipeline_orchestrator/lib/paths.py` (orchestration
  kingdom), imported by both the Snakefile and the Python entrypoints. Identity stays in
  `identifiers/` (separate kingdom).

---

## 🎯 THE TARGET MODEL (committed 2026-06-02 — Zone C confirmed per-well-able)

```
Zone 0 — experiment bootstrap          (experiment-grain; discovers wells)
  experiment_metadata/{exp}/...        plate, scope, series_well_mapping, frame_contract
  built_image_data/{exp}/...           stitched images (driven by discovered wells)

  ── FAN POINT ──  discovered_wells = [20250912_A01, 20250912_A02, ...]

Zone 1 — per-well canonical computation  (one well = one independent unit, E2E)
  segmentation_and_tracking/{exp}/per_well/{well_id}/...
  processed_snips/{exp}/per_well/{well_id}/...
  computed_features/{exp}/per_well/{well_id}/...        ← NEW (was experiment-grain)
  quality_control/{exp}/per_well/{well_id}/...          ← NEW
  analysis_ready/{exp}/per_well/{well_id}/...           ← NEW

Zone 2 — merged / publication products  (thin concat at the END of the DAG)
  segmentation_and_tracking/{exp}/contracts/segmentation_tracking.csv
  processed_snips/{exp}/contracts/snip_manifest.parquet
  computed_features/{exp}/consolidated/...
  quality_control/{exp}/consolidated/...
  analysis_ready/{exp}/analysis_ready.csv
```

- **Full run** = build all per-well outputs for all discovered wells, then build merged.
- **One-well run** = build per-well outputs for one `well_id` (merged optional/skipped).
  No special mode — just a different target.

**Path function — `well_id` lives in the canonical path, not a separate root:**
```python
artifact_path(output_root, stage="computed_features", artifact="mask_geometry",
              experiment_id=exp, well_id=well_id, scope="per_well")
#   → computed_features/{exp}/per_well/{well_id}/mask_geometry/mask_geometry_metrics.csv

artifact_path(output_root, stage="computed_features", artifact="mask_geometry",
              experiment_id=exp, scope="merged")
#   → computed_features/{exp}/mask_geometry/mask_geometry_metrics.csv
```
`scope ∈ {per_well, merged}` selects the branch; `fanout` in the registry declares which
stages legally support `per_well`.

---

## ❓ OPEN QUESTIONS (next session — don't lose these)

1. **Where exactly is the fan point?** Candidate: right after `series_well_mapping` /
   `frame_contract`. Confirm whether `build_stitched_images` already keys off an
   "available wells" list (mdcolon's hunch) — that seam likely *is* the fan point.
2. ~~**Is Zone C truly per-well-able?**~~ **RESOLVED (mdcolon, known fact 2026-06-02):
   Zone C is 100% per-well — every computation is per-snip/per-embryo, zero cross-well.
   The merge is the only barrier. Green light to push Zone C to per-well.**
3. **What does the well list look like as a first-class artifact?** Today it's implicit
   in `series_well_mapping.csv` + `selected_wells.txt`. For one-well-E2E it may want to
   be an explicit checkpoint output that the fan reads.
4. **Map `src/` onto the grain shape — but later.** `src/` is already stage-organized
   (`metadata_ingest/`, `segmentation/`, `feature_extraction/`…). Decide depth of reorg
   (light: align names + entrypoint index, vs. heavier: group by zone) *after* the grain
   model is settled. Grain drives layout, not the reverse.

---

## ⏸️ WHERE WE PAUSED (2026-06-02)

The grain model is now **settled and committed**: experiment-grain bootstrap → fan →
per-well canonical (Zones B and C) → thin merged publication. Zone C being fully
per-well is a confirmed fact (mdcolon), not an assumption — so scoped_runs is demoted to
scratch and `per_well/{well_id}` becomes the canonical post-fan path. Next: confirm the
fan point (Q1), then write the concrete `lib/paths.py` registry against this model.
