# `target/` — streamline-snakemake refactor docs (INDEX)

**The map for this folder.** 19 markdown docs live here now; this index says *which one to open for
what*, and which are live vs reference vs history. Reorganized 2026-06-17 from an 18-doc flat folder
into role-based folders.

---

## 🚦 START HERE (the 3 live drivers — top level)

| Open this | When |
|---|---|
| **`current_state_and_next_steps.md`** | **"Where are we RIGHT NOW?"** — verified on-disk state + the immediate next action. The status doc. Read this first. |
| **`front_half_reorg_roadmap.md`** | **"What's the plan to get YX1 to stitch?"** — the BINDING front-half implementation plan (the 7-step strangler migration to a per-well `frame_inventory`). The active driver. |
| **`OVERALL_PLAN.md`** | **"What's the WHOLE refactor?"** — the top-level plan/index for the entire streamline-snakemake effort (all stages, build order, per-stage status). Broader than the front half. |

> **Reading order for a cold start:** `current_state` (where we are) → `front_half_reorg_roadmap`
> (what we're doing next) → dip into `specs/` only as the roadmap points you there.

---

## 📁 `specs/front_end/` — front-end-specific TARGET specs (the "what", raw → stitch)

The architecture for the front half — discovery, run-well state, the inventories, ingest, stitch.

| Doc | The one question it answers |
|---|---|
| `front_end_naming_and_frame_inventory_flow.md` | What ARE the front-end stages? (ingest lineages, the fan, the convergence line) |
| `acquisition_inventory_flow.md` | The UPSTREAM, scope-specific record — "what was physically acquired" (per scope) |
| `frame_inventory_handoff_contract.md` | The DOWNSTREAM, agnostic seam — the per-frame table segmentation reads (the finish line) |
| `run_well_schema.md` | discovered vs eligible vs runnable well-state; Keyence reacquisition test plan |
| `recompose_yx1_front_end.md` | The YX1-only build (microscope-scoped; the L1→L2→L3 mapping) |

## 📁 `specs/` — CROSS-CUTTING + downstream TARGET specs (whole-pipeline)

Apply beyond the front end — read when a roadmap step touches identity, contracts, or the model seam.

| Doc | The one question it answers |
|---|---|
| `pipeline_file_philosophy.md` | How every rule/module/name should READ (the two hard constraints, the conformance checklist) |
| `per_well_throughline_findings.md` | The per-well NORTH STAR (grain model, registry, well-runner, DAG mechanics) |
| `well_id_throughline_refactor_plan.md` | The formal Scopes 1–5 spine (identity grammar → migration → env → well-runner → per-well) |
| `well_id_global_migration_map.md` | The Scope-2 `well_id` local→global migration map |
| `schema_layout.md` | Domain-owned contracts — schemas live with their data product; mechanics are shared |
| `model_input_handoff_contract.md` | The MODEL/embedding seam (downstream — not front-end) |

## 📁 `archive/` — point-in-time / stale (kept for provenance, do NOT treat as current)

| Doc | Why archived |
|---|---|
| `HANDOFF_front_end_two_wells.md` | 2026-06-07 handoff; predates the rename + strangler plan (stale specifics) |
| `HANDOFF_scope2_global_well_id.md` | Scope-2 migration handoff (point-in-time) |
| `HANDOFF_segmentation_tracking_paths_wiring.md` | seg/tracking paths handoff (point-in-time) |
| `frame_inventory_well_runner_audit.md` | 2026-06-07 code review; findings folded into the roadmap |

---

## 🧭 The mental model (so the specs cohere)

> **The stitcher is a DAM.** Upstream = microscope-specific (`acquisition_inventory`); downstream =
> agnostic (`frame_inventory`). This refactor's front-half goal: get YX1's raw data over the dam into
> one validated **per-well `frame_inventory`** shard, then stop. Everything past it already works.

- **Two inventories, NOT the same:** `acquisition_inventory` (upstream, scope-shaped) ≠ `frame_inventory` (downstream, agnostic).
- **Two "per-well":** identity (`well_id` is global — done) vs execution (run one well through a stage — the work).
- **Beat 1** (the roadmap) = raw → stitch → validated `frame_inventory` shard. **Beat 2** = repoint segmentation/features/QC at it.
- **Strategy = STRANGLER:** build the new per-well stitch beside legacy, prove on one well (B01) at a 3-layer comparison gate (ending in a human visual sign-off), promote, delete legacy.
