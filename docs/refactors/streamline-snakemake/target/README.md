# `target/` — streamline-snakemake refactor docs (INDEX)

**The map for this folder.** Markdown docs live here now; this index says *which one to open for
what*, and which are live vs reference vs history. Reorganized 2026-06-17 from an 18-doc flat folder
into role-based folders.

---

## 🚦 START HERE (the 4 live drivers — top level)

| Open this | When |
|---|---|
| **`AGENT_QUICKSTART.md`** | **"What do I do at turn 1?"** — cold-start protocol, end-of-session protocol, immutable conceptual anchors, pointer map. Read this before anything else. |
| **`current_state_and_next_steps.md`** | **"Where are we RIGHT NOW?"** — append-only timestamped session log; the top snapshot block is the live state. Read the quickstart first, then this. |
| **`front_half_reorg_roadmap.md`** | **"What's the plan to get YX1 to stitch?"** — the BINDING front-half implementation plan (the 7-step strangler migration to a per-well `frame_inventory`). The active driver. |
| **`OVERALL_PLAN.md`** | **"What's the WHOLE refactor?"** — the top-level plan/index for the entire streamline-snakemake effort (all stages, build order, per-stage status). Broader than the front half. |

> **Reading order for a cold start:** `AGENT_QUICKSTART.md` (the protocol) → `current_state` (where we are) → `front_half_reorg_roadmap` (what we're doing next) → dip into `specs/` only as the roadmap points you there.

---

## 📁 `specs/front_end/` — front-end-specific TARGET specs (the "what", raw → stitch)

The architecture for the front half — discovery, run-well state, the inventories, ingest, stitch.

| Doc | The one question it answers |
|---|---|
| `front_end_naming_and_frame_inventory_flow.md` | What ARE the front-end stages? (ingest lineages, the fan, the convergence line) |
| `plate_metadata_ingest_and_entity_qc.md` | How is plate metadata ingested + validated, and when is it injected? (the 3-layer loader/contract/entity-QC doctrine) |
| `acquisition_inventory_flow.md` | The UPSTREAM, scope-specific record — "what was physically acquired" (per scope) |
| `frame_inventory_handoff_contract.md` | The DOWNSTREAM, agnostic seam — the per-frame table segmentation reads (the finish line) |
| `run_well_schema.md` | discovered vs eligible vs runnable well-state; Keyence reacquisition test plan |
| `recompose_yx1_front_end.md` | The YX1-only build (microscope-scoped; the L1→L2→L3 mapping) |
| `keyence_wire_through.md` | The Keyence-only build — staged plan to bring Keyence onto the per-well materialize interface (acq inventory, mosaic backend, stitch-map quirk, reacquisition) |
| `external_dataset_handoff_target.md` | The **outside-world → frame_inventory entrance spec** — how an OUTSIDE researcher with their own data enters the pipeline (the stricter-entrance seam: biology long-ingest, the drop-in manifest, the strict per-well validator, the file plan). Decision provenance in `_WIP_external_handoff_decisions.md`. |

## 📁 `specs/detect-seg-track/` — downstream TARGET specs (validated frames → tracked masks)

The architecture for the first post-`frame_inventory` stage: detection, segmentation, tracking,
and the adapter seams that let model backends change without changing downstream contracts.

| Doc | The one question it answers |
|---|---|
| `README.md` | What is this planning space, and where should new detect/segment/track notes go? |
| `targets/overall_goal_and_plan.md` | What is the staged plan to split `segment_and_track_per_well` safely? |
| `targets/adapter_seams.md` | How should model backends plug in without leaking backend-native output downstream? |
| `targets/detection_world.md` | What is the detector product contract and validator? |
| `targets/segmentation_world.md` | What is the mask/segmentation product contract and validator? |
| `targets/tracking_world.md` | What is the temporal identity product contract and validator? |
| `working/current_state_and_next_steps.md` | Where is detect-seg-track RIGHT NOW? (its own living status log) |
| `working/open_questions.md` | Where do unresolved decisions live while mdcolon and agents iterate? |
| `working/current_audit_notes.md` | What did the current code audit find about the fused runner? |

> **Two status logs, don't confuse them:** the top-level `current_state_and_next_steps.md` tracks the
> **front-half** work; `specs/detect-seg-track/working/current_state_and_next_steps.md` tracks the
> **detect-seg-track** work. Each is appended newest-first within its own scope.

## 📁 `specs/features/` — downstream TARGET specs (validated objects → computed features)

The architecture for computed features after object extraction. QC uses the same per-well stage
pattern but should live in a sibling QC world, not inside this section.

| Doc | The one question it answers |
|---|---|
| `README.md` | What is feature world, and what is out of scope? |
| `targets/feature_world.md` | Which computed features can start, what do they depend on, and what proves each one is ready? |

## 📁 `specs/` — CROSS-CUTTING + downstream TARGET specs (whole-pipeline)

Apply beyond the front end — read when a roadmap step touches identity, contracts, or the model seam.

| Doc | The one question it answers |
|---|---|
| `pipeline_file_philosophy.md` | How every rule/module/name should READ (the two hard constraints, the conformance checklist) |
| `per_well_throughline_findings.md` | The per-well NORTH STAR (grain model, registry, well-runner, DAG mechanics) |
| `well_id_throughline_refactor_plan.md` | The formal Scopes 1–5 spine (identity grammar → migration → env → well-runner → per-well) |
| `well_id_global_migration_map.md` | The Scope-2 `well_id` local→global migration map |
| `schema_layout.md` | Domain-owned contracts — schemas live with their data product; mechanics are shared |
| `acquisition_inventory_schema_policy.md` | Two-tier inventory schema (shared/hard-checked vs scope-specific/soft) + the time-atom policy (`elapsed_time_s`) |
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

> **TWO OVERLAPPING ZONES** (the organizing lens — full diagram in `front_half_reorg_roadmap.md`
> and `OVERALL_PLAN.md` §1b):
> - **MICROSCOPE ZONE** = scope-aware code (raw → … → stitch backends). YX1 vs Keyence differ here.
> - **PER-WELL ZONE** = well-sharded execution (`discover_wells` → … → end).
> - **THE STITCH OVERLAP** = where well-sharded execution begins before microscope-specific production
>   is fully gone (`discover_wells → stitch → frame_inventory`). The special machinery lives here:
>   `discover_wells` fan, `well_runner`, scope stitch backends, scope-aware pre-handoff validators,
>   and the shared `frame_inventory` contract/validator. Crossing `frame_inventory` = **exit the
>   Microscope Zone** ("post-microscope land").

- **The front-half goal:** build the Stitch Overlap correctly and exit into a validated **per-well
  `frame_inventory`** shard, then stop. Everything past it is pure Per-Well Zone and already agnostic.
- **Two inventories, NOT the same:** `acquisition_inventory` (Microscope Zone, scope-shaped) ≠ `frame_inventory` (the exit, agnostic).
- **Two "per-well":** identity (`well_id` is global — done) vs execution (run one well through a stage — the work).
- **Beat 1** (the roadmap) = build the overlap → validated `frame_inventory` shard. **Beat 2** = the pure Per-Well Zone past it.
- **Strategy = STRANGLER:** build the new per-well stitch beside legacy, prove on one well (B01) at a 3-layer comparison gate (ending in a human visual sign-off), promote, delete legacy.
