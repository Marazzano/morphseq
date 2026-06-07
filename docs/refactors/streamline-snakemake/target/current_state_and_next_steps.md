# Current State & Next Steps (🔵 verified 2026-06-04)

**Status:** a snapshot of where the refactor actually stands on disk, plus the recommended next
move. Verified against code on 2026-06-04. This is the "where are we" anchor; the design lives in
the other `target/` docs.
**Companion to:** `per_well_throughline_findings.md` (north star), `well_id_throughline_refactor_plan.md`
(the formal Scopes), `front_end_naming_and_frame_inventory_flow.md` (front-end), `frame_inventory_handoff_contract.md`
(the drop-in seam).

---

## 📂 Doc coverage — what's in `target/` now

All refactor docs live in `target/` (the two throughline docs were `git mv`'d here 2026-06-04):

| Doc | Covers | Status |
|---|---|---|
| `per_well_throughline_findings.md` | north star: grain model, registry, well-runner, DAG mechanics | design complete |
| `well_id_throughline_refactor_plan.md` | the formal Scopes 1–5 | design complete |
| `front_end_naming_and_frame_inventory_flow.md` | ingest lineages, fan, post-fan tail | design complete |
| `frame_inventory_handoff_contract.md` | the stitched drop-in seam + frame_inventory contract | design complete |
| `model_input_handoff_contract.md` | the model/embedding seam (legacy build_06): inference-first encode of snips → latents; symlink view of `processed_snip_path`; reuses `gen_embeddings` + the `mseq_pipeline_py3.9` sub-env | design (inference) complete; training deferred |
| `current_state_and_next_steps.md` | **this doc** — verified on-disk state | living |
| `OVERALL_PLAN.md` | top-level plan/index: goal, the ordered spine (family/fanout/execution/status per stage), build order, per-stage status table, open items, **audit findings** (2026-06-05) | living |

**Not yet a dedicated target doc** (specified *inside* the findings doc, not broken out):
- the **per-well stage pattern** for features + QC (Scope 5) — see "The Per-Well Stage Pattern" below
- the **registry / well-runner code** (`lib/paths.py`, `lib/well_runner.py`) — designed, not built

---

## 🆔 Scope 1 — Identifiers: EMPTY (create, not split)

**On disk:** `src/data_pipeline/identifiers/__init__.py` is **0 bytes**; nothing else in the package.

**IDs are minted inline as f-strings today** (verified 2026-06-04):
- `metadata_ingest/scope/yx1_scope_metadata.py:208` → `well_id = f"{experiment_id}_{well_index}"`
- `metadata_ingest/scope/keyence_scope_metadata.py:261` → `well_id`; `:265` → `image_id` (uses `_f{time_int:04d}`)
- `metadata_ingest/mapping/series_well_mapper_keyence.py:177` → `well_id`

**What goes in it** (per the plan + the immutable-key decision in `frame_inventory_handoff_contract.md`):
```
identifiers/
    __init__.py     re-exports the public names
    constructors.py build_well_id(experiment_id, well_index)  → sanitize ONCE here
                    build_image_id(well_id, channel_id, time_index)
                    build_embryo_id, build_snip_id            — compose from ATOMS
    parsers.py      split_well_id(well_id) -> (experiment_id, well_index); parse_image_id; _LOCAL_ID_RE
    validators.py   validate_well_id (fail loud on bare A01); recompute_and_check(atoms, supplied_id)
    README.md       the "sign on the door": atoms are truth; well_id/image_id are DERIVED
```

- This is the layer the handoff contract leans on: the validator's "recompute well_id/image_id from
  atoms and fail loud on disagreement" = `validators.recompute_and_check()`.
- **Reconcile on build:** the handoff doc standardized `time_index` + `_t{time_index:04d}`, but the
  Keyence inline code uses `_f{time_int:04d}`. `build_image_id` must use the canonical `_t####` form
  (part of the `frame_index`/`time_int` → `time_index` collapse).
- **Zero-risk / additive** — no existing importers of these names to preserve (the package is empty),
  so Scope 1 is purely "write the constructors + optionally repoint the inline f-strings."

---

## ⚙️ Scope 3 — Config + Environment: HALF DONE

| Piece | State |
|---|---|
| `config.yaml` (science knobs) | ✅ **exists** — `src/data_pipeline/pipeline_orchestrator/config.yaml` |
| `env.yaml` (machine/paths) | ❌ **does not exist** |
| `env.example.yaml` (template) | ❌ **does not exist** |
| path routing decoupled from code location | ❌ **still welded** |

**The welding, verified (Snakefile):**
- `:17` `PROJECT_ROOT = WORKFLOW_DIR.parent.parent.parent` — data location derived from code location.
- `:34-36` `DATA_ROOT = ... PROJECT_ROOT / "data_pipeline_output"` — outputs pile up inside the repo.
- `:23` `PYTHON = "/net/.../mdcolon/.../bin/python"` — **interpreter hardcoded to a home dir**; no
  other machine/user can run without editing the Snakefile.

**Also:** `config.yaml` mixes machine concerns into the science config — `device: "cuda"` (×2),
absolute model checkpoint paths (`weights_path`, `checkpoint_path`). Scope 3 moves those to `env.yaml`.

**What Scope 3 builds** (per findings doc / refactor plan): `env.yaml` (gitignored) +
`env.example.yaml` (committed) beside the Snakefile, with `input_root` / `output_root` / `models_root`
as first-class absolute paths and `python` / `device` moved out of `config.yaml`. `project_root` keeps
being derived (works with zero setup). This is a **separate, later track** from identifiers.

---

## 🔁 The Per-Well Stage Pattern (features + QC) — ONE pattern, not two

**Verified 2026-06-04: features and QC rules are the SAME template.** `compute_mask_geometry` and
`compute_pose_kinematics` (Snakefile:618, :635) are **byte-identical except three tokens**: the stage
name, the module path, and the output filename. All 13 stages follow it:
- Features: `compute_mask_geometry`, `compute_pose_kinematics`, `compute_fraction_alive`,
  `compute_stage_predictions`, `consolidate_features`.
- QC: `compute_segmentation_qc`, `compute_viability_qc`, `compute_death_detection`,
  `compute_surface_area_qc`, `compute_auxiliary_mask_qc`, `compute_focus_qc`, `compute_motion_qc`,
  `consolidate_qc`.

**The three things that vary per stage** (everything else is the template):
```
1. stage name      compute_mask_geometry        vs  compute_pose_kinematics
2. module path     ...entrypoints.compute_mask_geometry  vs  ...compute_pose_kinematics
3. output file     mask_geometry/mask_geometry_metrics.csv  vs  pose_kinematics/pose_kinematics_metrics.csv
```
Features vs QC differ only in **`family`** (`computed_features/` vs `quality_control/`) and inputs.

**This is exactly the registry's job** (findings doc): one `STAGES` row per stage drives the path,
the rule body is a copy-paste template. "Add a feature / a QC metric = **one registry row + one
compute fn + one rule from the template**." Features and QC are the *same* recipe with a different
`family` field.

**The one structural difference to handle (Scope 5 wiring):** QC stages produce **flags**, and a few
currently read the **merged/consolidated** contract (e.g. `surface_area_qc` reads
`consolidated_features`) — the accidental merge-wall the findings doc flagged. The math is per-snip /
per-embryo (no cohort stat — findings doc audit), so the conversion is **rewiring those rules to read
per-well shards**, not rewriting algorithms.

> **Recommendation (mdcolon asked: "should I flesh out the features/QC pattern? they're the same"):**
> **YES — as ONE documented pattern.** The code proves they're copy-paste-with-find-replace, so the
> valuable artifact is a single **"add a per-well stage" recipe**: (1) the template rule, (2) the
> registry row, (3) the three tokens that vary, (4) the per-well conversion checklist (incl. the few
> QC merge-wall reads to rewire). This turns 13 near-identical rules into "one pattern + a table" —
> the whole point of the registry. Write it as a sibling doc (`per_well_stage_pattern.md`) once
> `lib/paths.py` exists (the pattern *imports* the registry), OR sketch it now and wire it when the
> registry lands.

---

## 🧭 Recommended order

```
Scope 1  identifiers/       ← DO NEXT. Empty, additive, zero-risk; unblocks everything keying on well_id;
                              the layer the handoff validator depends on.
Scope 3  env.yaml           ← separate later track (config exists; env + path-decoupling do not).
lib/paths.py + STAGES       ← the registry; precedes the well-runner.
Scope 5  per-well features/QC ← the ONE pattern above; mostly rewiring (merge-wall reads), not new math.
```

**Next concrete action:** build Scope 1 (`identifiers/`) — `constructors.py` / `parsers.py` /
`validators.py` / `README.md` — using the canonical `_t{time_index:04d}` form, against the four inline
f-string call-sites above.
