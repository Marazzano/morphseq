# Agent Quickstart — streamline-snakemake refactor

**What this is:** the cold-start protocol for any Claude session on this project. Read this first, then follow it. It is stable across implementation steps — it describes the *protocol*, not the current state.

---

## 🚀 COLD-START PROTOCOL (do these before any work)

**Step 1 — Read the current state (always):**
```
docs/refactors/streamline-snakemake/target/current_state_and_next_steps.md
```
The most recent `## ⭐ CURRENT SNAPSHOT` block at the top tells you exactly where we are: what shipped, what's broken, what the next action is, and any open decisions. Read the snapshot, then act on it — don't re-derive from scratch.

**Step 2 — Read the active plan (always):**
```
docs/refactors/streamline-snakemake/target/front_half_reorg_roadmap.md
```
The 7-step STRANGLER migration plan for the YX1 front half. The snapshot will name which step is next; read the corresponding step section here for FILES/FLOW/VERIFY details.

**Step 3 — Read the philosophy (always):**
```
docs/refactors/streamline-snakemake/target/specs/pipeline_file_philosophy.md
```
The two hard constraints and the naming conventions every new file must conform to. Read the conformance checklist before writing any code.

**Step 4 — Read a spec doc (only when the roadmap points to it):**
See the pointer map below. Only open a spec when the step you're implementing explicitly references it.

---

## 🪨 IMMUTABLE CONCEPTUAL ANCHORS (never violate these)

These are distilled from `pipeline_file_philosophy.md`. A change that breaks one is wrong, not just unidiomatic.

**Identity grammar:**
- `well_id`, `image_id`, `embryo_id`, `snip_id` are ATOMS → DERIVED. Always built by `shared/identifiers/` constructors, decomposed by its parsers. Never inline-mint with an f-string or split with `.split("_")`.
- Orchestration imports identity. Identity never imports orchestration.

**Artifact paths:**
- No artifact path is ever a raw string in a rule, task, or module. Always resolved through `orchestration.paths` (`artifact_path` / `validated_path` / `step_dir` / etc.).
- `PIPELINE_STEPS` registry (`orchestration/paths.py`) owns tabular artifact paths.
- `layout.py` (`image_materialization/stitched/layout.py`) owns off-registry image-tree paths.
- These two families use the same ID grammar but live in separate files. Never merge them.

**`tasks.py` is a pure dispatcher:**
- Parse args + delegate to the stage module. Zero business logic. Zero pandas. Zero domain computation.
- A `tasks.py` with stage logic is the failure mode — catch it in review.

**`discover_wells` vs `well_runner` — the fan machinery:**
- `discover_wells` = the FAN/BOOTSTRAP: turns experiment-grain canonical metadata into the per-well world. Microscope-agnostic. Emits `discovered_wells.txt` (physical identity only — not QC-passed, not runnable).
- `well_runner` = the PER-WELL SCHEDULER: computes `run_wells = discovered ∩ target [∩ eligible]`. Reads the summary contract, never scope-specific columns.
- Never conflate them. `discovered_wells.txt` ≠ `run_wells`.

**Contracts live beside their data product:**
- A data contract's source of truth lives with the package that owns that product (e.g. `image_materialization/stitched/contracts/frame_inventory_contract.py`).
- `shared/table_validators.py` owns reusable validation *mechanics* (no domain meaning).
- `data_pipeline/schemas/` is legacy-compat only. Do not add new semantics there.

**Tests live in the parallel test tree:**
- For new or changed `src/data_pipeline/...` code, add/update tests under the matching
  `tests/data_pipeline/...` path. Example:
  `src/data_pipeline/metadata_ingest/well_discovery/discovered_wells_contract.py` →
  `tests/data_pipeline/metadata_ingest/well_discovery/test_discovered_wells_contract.py`.
- Do not hide target tests inside the source package unless the package already has a local legacy
  `tests/` convention that you are explicitly preserving. The front-half target convention is the
  parallel top-level `tests/data_pipeline/...` tree.

**Scope-aware validators live BEFORE the handoff. The `frame_inventory` validator is agnostic:**
- Validators for raw position mapping, acquisition inventory, and acquisition resolution live in the Microscope Zone (upstream of `frame_inventory`).
- The `frame_inventory` contract validator is shared and microscope-agnostic. It does not learn YX1/Keyence logic.

**Sidecars are derived, never first-class:**
- `.validated` and `.provenance.json` are computed from an artifact path by helpers (`validated_path`, `provenance_path`). Never their own registry rows or hardcoded strings.

---

## 📐 THE TWO OVERLAPPING ZONES (the organizing lens)

```
 ingest → map → join │ discover_wells │ materialize_well │ frame_inventory │ segment → features → QC
 ═══════════════════════════════════════════════════════════════════════════════════════════════►

 ┌──── MICROSCOPE ZONE (scope-AWARE)   ────┐
 │ raw reads · scope schemas · stitch     │   YX1 vs Keyence differ ONLY here
 └─────────────────────────────────────────┘
                  ┌──────────────────── PER-WELL ZONE (well-SHARDED) ──────────────────────────┐
                  │ every stage runs ONE well at a time on the well spine (well_runner)          │
                  └───────────────────────────────────────────────────────────────────────────── ┘
                  ▲             ╔═══════════════╗              ▲
           discover_wells       ║ STITCH OVERLAP║       frame_inventory
           = bootstrap/FAN      ║ per-well AND  ║       = EXIT microscope land
           (well_runner born)   ║ scope-aware   ║       (pure per-well + agnostic →)
                                ╚═══════════════╝
```

The active front-half work (Beat 1) builds the Stitch Overlap and exits into a validated per-well `frame_inventory` shard. Everything past `frame_inventory` is Beat 2.

---

## 🗺️ POINTER MAP — which doc to open for what

| Question | Doc |
|---|---|
| "Where are we right now?" | `current_state_and_next_steps.md` ← **always read first** |
| "What's the YX1 stitch plan?" (the 7 steps) | `front_half_reorg_roadmap.md` ← **always read** |
| "How should this code READ?" (two hard constraints + conformance checklist) | `specs/pipeline_file_philosophy.md` ← **always read** |
| "What's the whole refactor?" (all stages, build order, per-stage status) | `OVERALL_PLAN.md` |
| "Where does every doc live?" | `README.md` |
| "What are the front-end stages?" (ingest lineages, the fan, the convergence line) | `specs/front_end/front_end_naming_and_frame_inventory_flow.md` |
| "What's the shared frame handoff contract?" | `specs/front_end/frame_inventory_handoff_contract.md` |
| "What's the YX1-specific build?" (L1→L2→L3 mapping) | `specs/front_end/recompose_yx1_front_end.md` |
| "What's the Scopes 1–5 identity spine?" | `specs/well_id_throughline_refactor_plan.md` |
| "Where do domain contracts live?" (schema layout + migration order) | `specs/schema_layout.md` |
| "What's the per-well NORTH STAR?" (grain model, registry, DAG mechanics) | `specs/per_well_throughline_findings.md` |

---

## 🏁 END-OF-SESSION PROTOCOL (do this before closing)

Prepend a new snapshot block to `current_state_and_next_steps.md`. **Never replace the previous block — append at the top of the snapshot stack.** The file is a timestamped, self-documenting session log; older blocks stay as history.

**Format — prepend this block immediately after the file header (before the previous snapshot):**

```markdown
## ⭐ CURRENT SNAPSHOT — {YYYY-MM-DD HH:MM}

**What shipped:** {verified artifacts, files committed, smoke-run results — or "nothing new this session"}
**What's broken/half-done:** {anything started but not committed or not verified}
**Next concrete action:** {the single first thing to do next session — name the file, function, and verify step}
**Open decisions:** {anything that blocks the next action and needs mdcolon's call or an interview}
```

**Rules:**
- One block per session. If the session produced nothing new, still update the timestamp and confirm the state is unchanged.
- "Next concrete action" must be specific enough that a cold-starting agent can act on it without reading the roadmap (it names the step number, the file, and the verify command).
- Do not summarize the roadmap in the snapshot — just point to the step.
