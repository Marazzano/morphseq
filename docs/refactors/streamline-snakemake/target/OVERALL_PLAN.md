# Overall Plan — streamline-snakemake refactor

**Status:** Plan / index. Produced from the 2026-06-05 audit pass (`../AUDIT_PASSOFF.md`).
**Owner:** mdcolon
**What this is:** the single top-level statement of the whole refactor — goal, end-to-end
spine, build order, per-stage status, open items, and audit findings. It is a **plan/index**:
detail lives in the linked `target/` docs, not here. Tags follow the north star:
🔵 **CURRENT** = verified in the active Snakefile today; 🟢 **TARGET** = what we build toward.

**Grounded against:** `src/data_pipeline/pipeline_orchestrator/Snakefile` (read end-to-end
2026-06-05) + the working tree on branch `mdcolon/20260605_refactor_docs_orientation`.

**Read for detail:** [[per_well_throughline_findings]] (north star) ·
[[current_state_and_next_steps]] (on-disk state) · [[well_id_throughline_refactor_plan]]
(Scopes 1–5) · [[front_end_naming_and_flow]] (ingest + fan) ·
[[stitched_handoff_contract]] (input seam) · [[model_input_handoff_contract]] (embeddings seam).

> ⚠️ **Read the Audit Findings (§6) before executing this plan.** The audit was run on the
> `20260605` orientation worktree, where the feature-extraction entrypoints were absent (F1).
> They **do exist on this branch** (`20260222_docs_snakemake_remake`, the canonical refactor
> branch with the code) — so F1 was a **branch artifact, now resolved** by consolidating docs
> onto this branch. The one remaining build-time decision is F2 (sentinel convention — DECIDED
> below). See §6 for the full verdicts.

---

## 1. Goal

Cross the experiment-grain bootstrap **once** to discover which wells exist, then run each
well **independently end-to-end** through the whole spine. After the well list is
materialized (the fan), everything is embarrassingly parallel — one well per process, no
cross-well barrier. The shape is **bootstrap → fan → per-well compute (incl. embeddings) →
merge/publish**, with merged tables as thin publication views at the *end* of the DAG, not
mid-pipeline barriers. **Embeddings is a late spine stage that FEEDS analysis_ready** (it is
not a terminal tail): latents join on `snip_id` and flip the reserved `embedding_calculated`
flag. The grain drives the file layout, governed by a lean path registry; identity
(`well_id` = `{exp}_{well}`) is the through-line in both rows and paths. Detail: [[per_well_throughline_findings]] §GOAL.

---

## 1b. The two OVERLAPPING zones (the organizing lens)

The pipeline has **two data-engineering zones on DIFFERENT axes that OVERLAP** (not sequential). This
lens explains why `discover_wells`/`well_runner`/stitch are the special, hard-to-place parts — they
live in the overlap.

```
 ingest → map → join │ discover_wells │ materialize_well │ frame_inventory │ segment → features → embeddings → QC
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════►

 ┌──── MICROSCOPE ZONE (scope-AWARE)   ────┐
 │ raw reads · scope schemas · scope       │   ← a small FRONT BUMP. YX1 vs Keyence differ ONLY here.
 │ mapping · scope STITCH backends         │
 └─────────────────────────────────────────┘
                  ┌──────────────────── PER-WELL ZONE (well-SHARDED)  ───────────────────────────────────────┐
                  │ every stage runs ONE well at a time on the well spine (well_runner) — the bulk of the DAG │
                  └──────────────────────────────────────────────────────────────────────────────────────────┘
                  ▲             ╔═══════════════╗              ▲
           discover_wells       ║ STITCH OVERLAP║       frame_inventory
           = bootstrap/FAN      ║ per-well AND  ║       = EXIT microscope land
           (well_runner born)   ║ scope-aware   ║       (pure per-well + agnostic →)
                                ╚═══════════════╝
```

- **MICROSCOPE ZONE** = scope-aware production; exits at the validated per-well `frame_inventory`
  shard. The ONLY place YX1/Keyence diverge.
- **PER-WELL ZONE** = well-sharded execution; starts at `discover_wells`, runs to the end. The bulk.
- **THE STITCH OVERLAP** (`discover_wells → stitch → frame_inventory`) = the graph region where
  well-sharded execution begins before microscope-specific production is fully gone. The special
  machinery lives here: `discover_wells` (the fan/left edge), `well_runner` (the per-well scheduler),
  stitch backends (scope-specific producers), scope-aware pre-handoff validators, the shared
  `frame_inventory` contract/validator, and `frame_inventory` itself (the right edge =
  "post-microscope land"). **Crossing `frame_inventory` exits the Microscope Zone.**

> The **front-half refactor** (`front_half_reorg_roadmap.md`) is precisely *"build the Stitch Overlap
> correctly and exit into a per-well `frame_inventory`."* Everything past frame_inventory is pure
> Per-Well Zone (this section's bulk) and already agnostic.

---

## 2. The end-to-end spine (ordered stages)

Status legend: **built** (rule + code present, this branch) · **rewire** (built but reads a
merged input — the merge-wall, Scope 5) · **not-built** (no rule/code).

🔵 **CURRENT execution order** (the fan sits LATE — after stitching, because
`build_frame_contract` consumes the stitched inventory):

| # | Stage (rule) | Family | fanout | execution | Status |
|---|---|---|---|---|---|
| **Zone A — experiment bootstrap** (`fanout=experiment`; discovers wells) |
| A1 | `normalize_plate_metadata` | experiment_metadata | experiment | single | built |
| A2 | `extract_scope_metadata_{yx1,keyence}` | experiment_metadata | experiment | single | built |
| A3 | `map_series_to_wells_yx1` | experiment_metadata | experiment | single | built · *raw well-identity discovery* |
| A4 | `apply_series_mapping_yx1` → `scope_metadata_mapped.csv` | experiment_metadata | experiment | single | built |
| **Zone B0 — image materialization** (per-well tree, NO merge; **off-registry**) |
| B0 | `build_stitched_images_yx1` (+`…_all` → `stitched_inventory.csv`) | built_image_data | per_well *(tree)* | single (loops wells) | built · 🔵 keys on local `well_index`; 🟢 → `well_id` |
| **Zone A (cont.) — frame contract + FAN** |
| A5 | `build_frame_contract` → `frame_contract.csv` | experiment_metadata | experiment | single | built · 🟢 splits per-well (Zone-A narrowing) |
| **⟱ FAN** | `discover_wells` *(checkpoint)* → 🔵 `wells.txt` / 🟢 `discovered_wells.txt` | experiment_metadata | experiment | single | built · 🔵 reads `frame_contract.csv`, writes `wells.txt`; 🟢 reads `scope_metadata_mapped.csv` (earlier), writes `discovered_wells.txt` |
| **Zone B — per-well computation** (`fanout=per_well_then_merge`) |
| B1 | `segment_and_track_per_well` → `merge_segmentation_tracking` | segmentation_and_tracking | per_well_then_merge | per_well | built |
| B2 | `run_snip_processing_per_well` → `merge_snip_manifests` | processed_snips | per_well_then_merge | per_well | built |
| B3 | `generate_auxiliary_masks` | auxiliary_masks | experiment | single | built *(per-experiment, reads frame contract)* |
| B4 | `compute_mask_geometry` | computed_features | experiment *(🟢 per_well)* | single | built (F1 resolved) |
| B5 | `compute_pose_kinematics` | computed_features | experiment *(🟢 per_well)* | single | built (F1 resolved) |
| B6 | `compute_fraction_alive` | computed_features | experiment *(🟢 per_well)* | single | built (F1 resolved) |
| B7 | `compute_stage_predictions` | computed_features | experiment *(🟢 per_well)* | single | built (F1 resolved) |
| B8 | `consolidate_features` → `consolidated_snip_features.csv` | computed_features | experiment | single | built (F1 resolved) |
| Q1 | `compute_segmentation_qc` | quality_control | experiment *(🟢 per_well)* | single | built |
| Q2 | `compute_viability_qc` | quality_control | experiment | single | **rewire** — reads `consolidated_features` (F4) |
| Q3 | `compute_death_detection` | quality_control | experiment *(🟢 per_well)* | single | built |
| Q4 | `compute_surface_area_qc` | quality_control | experiment | single | **rewire** — reads `consolidated_features` (merge-wall) |
| Q5 | `compute_auxiliary_mask_qc` | quality_control | experiment *(🟢 per_well)* | single | built |
| Q6 | `compute_focus_qc` | quality_control | experiment *(🟢 per_well)* | single | built *(stub: `focus_flag=False`)* |
| Q7 | `compute_motion_qc` | quality_control | experiment *(🟢 per_well)* | single | built *(stub: `motion_flag=False`)* |
| Q8 | `consolidate_qc` → `qc_flags.csv` | quality_control | experiment | single | built |
| **Zone B (cont.) — MODEL / EMBEDDINGS** (legacy build_06; FEEDS analysis_ready) |
| E1 | `compute_embeddings` → `per_well/{well_id}/latents.parquet` | embeddings | per_well | **single** (model loaded once) | **not-built** (active frontier) |
| E2 | `merge_embeddings` → `contracts/latents.parquet` | embeddings | per_well_then_merge | single | **not-built** |
| **Zone C / publication** |
| C1 | `assemble_analysis_ready` → `analysis_ready.csv` | analysis_ready | experiment | single | built · 🟢 gains `latents.parquet` input; `embedding_calculated` hardcoded `False` today |

> **Embeddings execution call (verified reasoning):** `fanout=per_well` + `execution=single`
> — latents land per-well (spine + incremental staleness) but **one job loads the legacy
> model once** and encodes all run wells, because a per-well `conda run -n mseq_pipeline_py3.9`
> reload (process spawn + checkpoint deserialize) would dominate the encode. Same shape as
> stitching (B0). Detail: [[model_input_handoff_contract]] §6.2.

---

## 3. Build order (with dependencies)

From [[current_state_and_next_steps]] §"Recommended order" + [[per_well_throughline_findings]]
§"Next concrete steps". **Not invented here** — reproduced so the plan is self-contained.

```
    ✅ DONE (2026-06-07 audit on 20260222):
    - Scope 1 (shared/identifiers/ split), Scope 3 (env.yaml), paths.py, well_runner.py, tasks.py
    - Front-end rules wired to TARGET shape (discover_wells reads scope_metadata_mapped.csv)
    - YX1 Phase 1 recompose: CSV→CSV map_series_to_wells, no premature IDs, x_um/y_um in scope CSV

    (F1 resolved: the feature-extraction entrypoints exist on this branch — the gap was a
     branch artifact of the orientation worktree. Verify with: ls src/data_pipeline/
     feature_extraction/entrypoints/ — expect 7 compute_*.py.)
1.  Scope 1  shared/identifiers/  ← SPLIT the existing flat `shared/identifiers.py` (58 lines,
                             already imported by 10 modules) into a package: constructors.py +
                             NEW parsers.py (split_well_id; move normalize_embryo_local_track_id)
                             + NEW validators.py (validate_well_id) + README.md. KEEP current
                             signatures so the `__init__.py` re-export shim leaves all 10 importers
                             green → zero-risk. The well_id-first signature flip + `well_index→well`
                             rename is Scope 2 (it breaks 6 mint sites). Restore/keep the
                             `identifier_and_wildcard_contract.md` reference (recovered 2026-06-05).
                             ⚠️ NOTE: the real home is `shared/identifiers/`, NOT the empty
                             top-level `src/data_pipeline/identifiers/` package (a decoy — delete it
                             or leave inert). Use canonical _t{time_int:04d}.
2.  Scope 3  env.yaml       ← separate later track (config.yaml exists; env + path-decoupling
                             do not). Precedes the well-runner (runner takes output_root as a
                             param, never derives it). Also lands model_python_env for E1.
3.  pipeline_orchestrator/orchestration/paths.py + PIPELINE_STEPS
                          ← the artifact path registry (imports identifiers). Fanout-enforced
                             resolver. Depends on Scope 1. ⚠️ resolve F2 (sentinel convention)
                             here.
4.  Scope 2                 ← flip well/well_id semantics in schemas + call sites; regenerate.
                             Migration site: the discover_wells checkpoint reads well_index
                             (Snakefile:448,459) — see §5.
5.  lib/well_runner.py      ← select_run_wells (pure) + checkpoint glue + merge_trigger_inputs
                             + WellRun. Built on normalized IDs (needs Scope 1/2).
6.  Wire ONE stage E2E      ← through the registry as the proof, then replicate to the rest;
                             this is where Scope-5 per-well conversion + merge-wall rewires land.
7.  Embeddings (E1/E2)      ← the active frontier; needs paths.py + well-runner + Scope 3
                             (model_python_env). FEEDS analysis_ready (add latents input).
```

Independent of the chain: **Win 2 `tasks.py`** (verb dispatch) can land anytime and eases
step 6. Detail: [[well_id_throughline_refactor_plan]] (Scopes), [[per_well_throughline_findings]]
§WELL-RUNNER.

---

## 4. Per-stage status table (countable remaining work)

**Updated 2026-06-07 after audit on `20260222_docs_snakemake_remake` (the canonical code branch).**

| Bucket | Count | Stages / items |
|---|---|---|
| **built** (rule + code, this branch) | 17 | A1–A5, B0, B1, B2, B3, B4–B8 (features, F1 resolved), Q1, Q3, Q5, Q6*, Q7*, Q8, C1 *(Q6/Q7 are stubs)* |
| **rewire** (built, reads merged input — Scope 5) | 2 | Q2 viability_qc, Q4 surface_area_qc |
| **not-built** (no rule/code) | 2 | E1 compute_embeddings, E2 merge_embeddings |
| **infra — BUILT** (was "not-built" in prior version) | — | `shared/identifiers/` split ✅; `orchestration/paths.py` ✅; `orchestration/well_runner.py` ✅; `tasks.py` ✅; `env.yaml` + `env.example.yaml` ✅ |
| **YX1 Phase 1 recompose — BUILT + SMOKE RUN PASSED 2026-06-07** | — | `extract_yx1_scope_metadata.py` emits `raw_position_label`+`x_um`/`y_um`, no premature IDs; `map_yx1_series_to_wells.py` is CSV→CSV (no ND2 re-open), ref path config-sourced; Snakefile `map_series_to_wells` rule drops `raw_images_dir`; schema updated. Smoke run on `20250912` (95 wells): all 4 stages clean, `well_id=20250912_A01` global IDs confirmed. |
| **scaffolding** | 1 pkg | `embeddings/` (0-byte `__init__.py`, **not-built**) |

*(Count note: A2 and B0 each have microscope variants; counted once.)*

**Headline:** YX1 Phase 1 recompose and smoke run are complete. Next: **Scope 2** global `well_id` flip (`build_well_id(exp, well)` 2-arg; 6 mint sites) → wire one stage E2E through `paths.py` registry → embeddings (E1/E2).

---

## 5. Open items — blocks-build vs defer

| Item | Disposition | Reason |
|---|---|---|
| **F1: feature entrypoints (branch artifact)** | **RESOLVED** | Absent on the `20260605` orientation worktree where the audit ran; **present on this branch** (`feature_extraction/entrypoints/`, 7 files). Resolved by consolidating docs onto the code branch. (§6 F1) |
| Exhaustive Zone-C grep-audit | **DONE** (safe) | Ran it — no cohort/cross-well statistic anywhere in features+QC+joins. (§6 F3) |
| **F2: `.validated` sentinel convention not uniform** | **blocks-build** (for `paths.py`) · **DECIDED** | Two conventions coexist (28 dot-prefixed `.X.validated` vs 30 suffix `X.csv.validated`). **Decision (2026-06-05): standardize on dot-prefixed hidden `.{filename}.validated`** (matches the spine artifacts; `validated_path = parent / f".{name}.validated"`, no per-stage flag). Action: rename the ~30 suffix sentinels in feature/QC rules during Scope-5 wiring. |
| `models_root` location | safe-to-defer | Leaning Scope-3 `env.yaml` reusing legacy `models/legacy/<name>` layout; confirm when wiring E1. |
| `mseq_pipeline_py3.9` env name | **confirmed** (safe) | Env exists (`…/mamba/envs/mseq_pipeline_py3.9`); switch at `legacy_model_utils.py:204`. (§6 F5) |
| Stitching `execution` (single vs per_well) | safe-to-defer | Stays `single`; revisit only for a real IO/compute/debug win. |
| Merge cadence (per-stage vs stage-group) | safe-to-defer | Performance tuning, not architecture. |
| Scope-2 migration site (`discover_wells` reads `well_index`) | safe-to-defer (track) | Confirmed: Snakefile:448,459 read `well_index` from the contract; the `materialize_selected_wells`/`_wells_from_mapping` path reads it too (:240). Scope 2 must update both; order is safe (Scope 2 precedes well-runner). |
| **F7: Scope 1 is a SPLIT, not a create** | **plan-corrected** (2026-06-05) | `shared/identifiers.py` already exists (58 lines) and is imported by **10 modules**; `build_well_id(well_index)` returns the bare local `A01` (so `well_id==well_index` today). Per-well tree keys `per_well/{well_id}/` on this LOCAL value (the 🔵→🟢 flip is unbuilt). The well-runner needs `well_id` GLOBAL → that is the Scope-2 2-arg flip + `well_index→well` rename, breaking 6 mint sites. Scope 1 stays zero-risk by keeping signatures. (§6 F7) |
| Contract doc `identifier_and_wildcard_contract.md` was deleted | **RESOLVED** (2026-06-05) | Dropped in `8d680daa` doc-reorg but still referenced by code (`shared/identifiers.py:3` + 2 more) and the well_id plan. **Recovered verbatim** from `8e1764b6` + status banner (documents 🔵 CURRENT local-`well_id`). The other 9 docs that commit deleted have **0 live refs** (superseded cleanly). |

---

## 6. Audit findings (what did / did NOT hold up)

Audited the five spine claims (AUDIT_PASSOFF §2), the "settled decisions" (§3), and the
open-items (§4), grounding every claim against the Snakefile + working tree.

### Spine claims — verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | Fan is a checkpoint; per-well staleness needs a per-well spine | ✅ **HOLDS.** `checkpoint discover_wells` (Snakefile:427) reads `frame_contract.csv` (:434) → `wells.txt` (:438); `wells_for_experiment` expands the per-well DAG off it (:476–480). 🔵 CURRENT tag accurate. |
| 2 | Zone C is per-well-able (no cohort math) | ✅ **HOLDS — now exhaustively verified.** Grepped all QC + join modules **and** the feature modules: every aggregate is **within-embryo / within-Z-stack** (`embryo_qc` `np.percentile(...,5)` is "across all Z-pairs" of one embryo; `surface_area_qc` compares to the external `sa_reference_curves.csv`, no quantile). No cross-well statistic exists. Upgrades the doc's "spot-check" to "exhaustive." *(Feature modules were audited from git blobs during the orientation-worktree pass; they are present on this branch — F1.)* |
| 3 | The merge wall is rewiring, not algorithm change | ✅ **HOLDS, but doc UNDERCOUNTS.** Doc names only `surface_area_qc`. Actually **two** Zone-B QC rules read the merged `consolidated_features`: `compute_viability_qc` (Snakefile:753) **and** `compute_surface_area_qc` (:799). Both do per-snip work → both rewire cleanly to per-well shards. (F4) |
| 4 | `fanout` vs `execution` holds for embeddings | ✅ **HOLDS** (as design; unbuilt). Reasoning verified: per-well `conda run` model reload would dominate → one batched job, per-well shards. Contract §6.2 and findings doc agree. Same shape as stitching B0 (which the Snakefile confirms is per-well-tree + single-job). |
| 5 | Embeddings FEEDS analysis_ready; gate not circular | ✅ **HOLDS.** No embeddings rule in the Snakefile (unbuilt, active frontier). `assemble_analysis_ready` reads only features + qc_flags (Snakefile:937–940), **not** latents. `embedding_calculated` hardcoded `False` (`assemble.py:65`) reserves the seam. Gate comes from QC consolidation, not analysis_ready — non-circular. |

### Findings (named, not silently resolved)

- **F1 — RESOLVED (branch artifact): feature-extraction entrypoints.** During the audit
  (run on the `20260605` orientation worktree), `src/data_pipeline/feature_extraction/`
  contained **only `__init__.py`** — so the Snakefile's calls to
  `feature_extraction.entrypoints.compute_*` (mask_geometry / pose_kinematics /
  fraction_alive / stage_predictions / consolidate_features) would `ModuleNotFoundError`,
  contradicting the docs' "wired end-to-end / ✅ ported." **Root cause:** the orientation +
  audit docs had been committed on `20260605` (an *older* fork that predates the
  `entrypoints/` submodule and also carries unrelated CEP290/label-transfer work), while the
  real refactor code — including the 7 `entrypoints/compute_*.py` — lives on
  `mdcolon/20260222_docs_snakemake_remake`. **Resolution:** the docs were consolidated onto
  `20260222` (this branch), so docs and the entrypoint code now live together. B4–B8 are
  **built** here. Verify: `ls src/data_pipeline/feature_extraction/entrypoints/`.

- **F2 — BUILD-BLOCKING for the registry: `.validated` sentinel convention not uniform.
  DECIDED 2026-06-05 → dot-prefixed hidden.**
  The Snakefile mixes **dot-prefixed hidden** sentinels (`.frame_contract.validated`,
  `.segmentation_tracking.validated`, `.analysis_ready.validated` — 28 occurrences) with
  **suffix** sentinels (`mask_geometry_metrics.csv.validated`, `qc_flags.csv.validated` —
  30 occurrences). The MVP `validated_path()` helper assumes a single form
  ([[per_well_throughline_findings]] AUDIT TODO flagged this as unverified — now
  **confirmed non-uniform**). **Decision: standardize on `.{filename}.validated`** (the
  dot-prefixed hidden form already used by the load-bearing spine artifacts the checkpoint
  and merges depend on). Then `validated_path(...) = artifact_path(...).parent /
  f".{name}.validated"` — a one-liner, no per-stage convention flag (this kills the F2
  branching entirely). **Action:** rename the ~30 suffix sentinels (the feature/QC
  `*.csv.validated` outputs) to the dot-prefixed form during Scope-5 wiring; update the
  corresponding `done = ...` lines in those rules.

- **F3 — RESOLVES open question #2 (the #1 audit item): Zone-C cohort audit is clean.**
  See spine claim 2. The exhaustive grep the docs marked "pending" is done; no cohort math.

- **F4 — Doc undercounts the merge-wall.** `compute_viability_qc` also reads
  `consolidated_features` (Snakefile:753), not just `surface_area_qc`. Both are Scope-5
  rewires. Update the findings doc's Scope-5 list to name both. (Not blocking; tracking.)

- **F5 — CONFIRMS: model sub-env exists.** `mseq_pipeline_py3.9` is present
  (`…/mamba/envs/mseq_pipeline_py3.9`); the `conda run -n mseq_pipeline_py3.9` switch is at
  `legacy_model_utils.py:204`. Contract §8 Q5 answered: yes, still current.

- **F6 — "Merge never shrinks" is 🟢 TARGET, not 🔵 CURRENT (correctly tagged, worth
  flagging).** The current merges (`merge_snip_manifests`, `merge_segmentation_tracking`)
  take `--inputs` = the **declared** per-well shard paths from `wells_for_experiment` and
  merge **only those** — they do **not** scan-all-present-shards. So with a subset
  `target_wells`, today's merged file *would* shrink to the run subset. The doc's
  "triggers narrowly, composes broadly (execution-time scan of all `.validated` shards)" is
  an unbuilt TARGET behavior ([[per_well_throughline_findings]] §GUIDING PRINCIPLE). The doc
  tags it 🟢 correctly; this finding just makes the CURRENT≠TARGET gap explicit so the
  well-runner build (step 5) knows the scan is *new* work, not a rewire.

- **F7 — Scope 1 is a SPLIT of an existing module, not a greenfield create (2026-06-05
  audit on `20260222`).** The earlier framing ("create empty `identifiers/`, zero-risk")
  was stale on two counts. (a) `src/data_pipeline/shared/identifiers.py` **already exists**
  (58 lines: `build_well_id`/`build_image_id`/`build_embryo_id`/`build_snip_id`/
  `normalize_embryo_local_track_id`/`sanitize_experiment_id`) and is **imported by 10
  modules** — so Scope 1 is the *split* the well_id plan describes (lines 173–182), into
  `constructors.py` + new `parsers.py`/`validators.py`, behind a re-export shim. The empty
  top-level `src/data_pipeline/identifiers/` package is a **decoy** — the real home is
  `shared/identifiers/`. (b) `build_well_id(well_index)` today returns the **bare local
  label `A01`**, so `well_id == well_index` and the per-well tree (`per_well/{well_id}/`)
  keys on a LOCAL id (matches B0's 🔵 tag). **The well-runner needs a GLOBAL `well_id`**
  (`{experiment_id}_{well}`) — that is the **Scope-2** 2-arg signature flip
  (`build_well_id(experiment_id, well)`, sanitize moved up) + `well_index→well` column
  rename, which **breaks 6 mint sites** (`build_well_id`×3 keyence/map_series/yx1,
  `build_image_id`×5, `build_embryo_id`×1 sam2_ingestor). Keeping Scope 1 to the split-only
  (signatures unchanged) preserves its zero-risk character; the flip that unblocks the
  well-runner is owned by Scope 2. (§5 F7)

### Decisions confirmed settled (AUDIT_PASSOFF §3)

Frame-contract → dedicated `frame_contracts/` family (findings §1002, DECIDED 2026-06-05);
multi-channel → channel as rows (findings §1005); `well_id` canonical everywhere incl. image
trees (findings §347); `target_wells` = compute filter never merge-membership (verified: merge
inputs derive from the checkpoint via `wells_for_experiment`, not `TARGET_WELLS`); registry
governs tabular artifacts only (image trees off-registry); model interpreter → `env.yaml`
(contract §6.5, reconciled). **No stale "leaning/open" text blocks the build** — the two spots
the pass-off flagged (findings #5 frame-contract family, stitched-contract §193) read as
resolved in the current `target/` docs.

---

## 7. Bottom line

The **design is coherent and all five spine claims hold** — the per-well architecture, the fan
checkpoint, the no-cohort-math guarantee, and the embeddings-feeds-analysis_ready ordering are
all verified against code. **No design blockers remain.** F1 (missing feature entrypoints) was
a branch artifact, now resolved by consolidating docs onto the code branch. F2 (sentinel
convention) is DECIDED (dot-prefixed hidden) — an implementation detail for `paths.py`. The
real remaining work is **building**, not fixing: **Scope 1 (`shared/identifiers/` split) → `paths.py`
(apply F2) → Scope 2/3 → `well_runner.py` → wire one stage → embeddings (E1/E2, the active
frontier)**, plus the 2 merge-wall QC rewires (Scope 5).
