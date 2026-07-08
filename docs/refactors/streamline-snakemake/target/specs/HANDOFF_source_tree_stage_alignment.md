# HANDOFF: Source Tree ↔ Output Stage Alignment

**Status:** Phases 1-3 of 5 DONE and committed. Phases 4-5 (the big ones) NOT started.
**Plan doc:** `docs/refactors/streamline-snakemake/target/specs/source_tree_stage_alignment.md` —
read that first, it has the full target layout, the mapping table, and the "why" for every decision.
This handoff is the "what's left and how to pick it back up" companion.

**Branch:** `mdcolon/20260222_docs_snakemake_remake`. All work so far is in 3 real commits (see below) —
nothing squashed, nothing amended. Safe to `git log -p` any of them to see exactly what moved.

---

## Do this first, before touching anything

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs
git log --oneline -5   # confirm you're on top of 3fa29954 (or later)
git status --short | grep -v '^??' | grep -v data_pipeline_output   # should be EMPTY
```

If that status check isn't empty, STOP and figure out what changed before continuing — the last
session ended with a fully clean, fully-committed working tree.

Then re-establish the test baseline (this is the one command you'll run after every change):

```bash
PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output python -m pytest \
  src/data_pipeline tests --import-mode=importlib -q \
  --ignore=tests/test_build04_bootstrap.py \
  --ignore=tests/test_sam2_staleness.py \
  --ignore=tests/test_stage_predictions_validator.py \
  --ignore=tests/test_time_helpers.py \
  --ignore=tests/test_timestamp_extraction.py \
  --ignore=tests/test_tracking_logic.py
```

Expected result right now: **928 passed, 2 skipped, 3 deselected.** If you see anything else, the
tree has drifted since this handoff was written — treat that as new information, not as this
handoff being wrong.

The 6 `--ignore`d files are **pre-existing breakage unrelated to this refactor** (confirmed at the
start of this work): `src.build.*` legacy imports, a missing `compute_dead_flag2_persistence`
symbol, missing pytest fixtures on what are really standalone scripts. Don't try to fix them as
part of this refactor — they were broken before this work started and are out of scope.

Always use `conda run -n segmentation_grounded_sam --no-capture-output python ...` — never bare
`python`/`python3`, never `conda activate`. Always `PYTHONPATH=src:$PYTHONPATH` and
`--import-mode=importlib` (the `tests/` namespace shadows `src/`, imports break without it).

---

## What's done (Phases 1-3, commits 407a6d74 → 3fa29954)

### Commit `407a6d74` — channel-map refactor
- Split `schemas/channel_normalization.py` → `shared/channel_vocabulary.py` (pure vocabulary +
  `validate_channel_id`) — cross-cutting, stays outside any stage.
- New `ScopeChannelMap` class (`metadata_ingest/scope/shared/channel_map_contract.py`): a per-scope
  dialect map that **validates itself at construction** and raises a detailed fail-loud from
  `.to_canonical()`, instead of relying on an external test to catch a bad map. Written
  beginner-readable per the user's explicit ask (heavy comments, one dict entry per line, no
  cleverness) — if you touch this file, keep that style.
- `scope/{yx1,keyence}/mappings.py` → `channel_map.py`, now built on `ScopeChannelMap`.
- Deleted `canonical_mapper.py` (the old generic `apply_canonical_mapping` — zero real callers left
  after the `ScopeChannelMap` migration) and `schemas/` (now empty).
- `metadata_ingest/scope/tests/test_canonical_mapping.py` rewritten to verify the guard *exists*
  rather than recomputing it (the snip_qc parallel — see the plan doc's "vocab" section).

### Commit `ef064b96` — delete empty package stubs
- Deleted `config/`, `identifiers/` (top-level), `features/`, `embeddings/` — each was a single
  0-byte `__init__.py` with **zero real importers**. Not shims needing rewiring, just dead stubs.
- Also cleaned stale `__pycache__`-only dirs (`schemas/`, `feature_extraction/{core,consolidated_features}/`)
  whose tracked contents were already deleted in prior (pre-this-session) commits.

### Commits `b0c4b90e` + `3fa29954` — archived pre-disentanglement segmentation cluster
**This is the one with a real surprise in it — read this before doing anything similar elsewhere.**

`segmentation_and_tracking/` (27 files) looked in the plan doc like "untangle a live subsystem."
Investigation showed it's actually a **dead "Phase 3" draft**: mutually references
`segmentation/grounded_sam2/` and `segmentation/video_generation/`, none of the three is wired into
`PIPELINE_STEPS`, `tasks.py`, or any Snakemake rule, and its own tests were already broken by API
drift (`normalize_frame_detections() got an unexpected keyword argument 'video_id'`) — evidence
nobody had been running them.

**Per user decision, archived (not deleted) all three clusters** into `segmentation/_archive/`:
```
segmentation/_archive/
  grounded_sam2/
  video_generation/
  segmentation_and_tracking/
```
Archiving (not deleting) means it's still reviewable/diffable; parking it in `_archive/` signals
"candidate for removal" for a later pass, same pattern already used for `schemas/_archive/`
elsewhere in this codebase.

**Two pieces were genuinely live and got extracted BEFORE archiving** (don't re-archive these if you
see them referenced — they're intentionally outside `_archive/` now):
- `clean_embryo_mask()` + its helpers → `segmentation/shared/mask_processing.py`. Used by 4 real
  `feature_extraction` metrics (`mask_geometry_metrics.py`, `curvature_metrics/skeletonization.py`,
  `pose_kinematics_metrics.py`, `fraction_alive/_legacy_compute.py`) — all real `PIPELINE_STEPS`
  entries (`mask_geometry`, `curvature_metrics`, `pose_kinematics`, `fraction_alive`).
- `gdino_detection.py` → `detection/backends/groundingdino/gdino_detection.py`. The **live**
  GroundingDINO adapter (`run_groundingdino_detection.py` — the actual backend behind the real
  `frame_detections` pipeline step) reuses `detect_embryos`/`filter_detections` from this file
  verbatim. Its own docstring said so; don't assume a file is dead just because its parent folder is.

**Lesson for the rest of this plan:** "not wired into `PIPELINE_STEPS`" is the right test for
dead-vs-live, but check it **per file**, not per folder — a live file can hide inside an otherwise
dead folder (and vice versa: a nominally "live" folder can have long-abandoned files in it).
`grep -rln "<module.dotted.path>" --include=*.py src tests` for the SPECIFIC symbol, not just the
folder name, before assuming.

**A commit-hygiene mistake happened here, now fixed but worth knowing:** the first commit
(`b0c4b90e`) captured the file *moves* but missed staging the *content edits* (import-path fixups,
docstring updates) needed to make the moved code actually resolve. Caught it by re-diffing `HEAD`
against the working tree after committing — always do that after a big `git mv`-heavy commit:
```bash
git status --short | grep -v '^??'   # should be empty right after a commit
```
The fixup landed in `3fa29954`. Both commits together are correct; don't try to squash/amend them.

---

## What's left — Phases 4-5 (NOT started)

Read `source_tree_stage_alignment.md` for the full target layout and reasoning. Short version:

### Phase 4 (Task #5 in the plan doc): regroup modules under the 5 stage folders
```
src/data_pipeline/
  acquisition/         <- metadata_ingest/, image_building/, image_materialization/
  object_extraction/   <- detection/, segmentation/, auxiliary_masks/, snip_processing/
  feature_extraction/  <- already named right, stays put
  quality_control/     <- already named right, stays put
  analysis_ready/       <- already named right, stays put
```
This is a **wide, tree-spanning `git mv` + import sweep** — every
`from data_pipeline.metadata_ingest...` becomes `from data_pipeline.acquisition.metadata_ingest...`,
repeated across `src/data_pipeline`, `src/analyze`, `results/`, and `tests/`. Do it **one stage at a
time**, one commit per stage, tests green before moving to the next — same discipline as Phases 1-3.
Recommended order (per the plan doc, DAG-earliest first): `acquisition` → `object_extraction` →
leave `feature_extraction`/`quality_control`/`analysis_ready` as pure renames-in-place (they're
already correctly named, this step is really just confirming nothing needs to move).

**Before starting:** re-run the same kind of "is this actually live" check that caught the
`segmentation_and_tracking/` surprise. Don't assume every file in `metadata_ingest/`, `detection/`,
etc. is live just because the folder maps to a real stage — spot-check with
`grep -rln` against `PIPELINE_STEPS`/`tasks.py`/`rules/` the way this session did.

**Watch for:** `shared/`, `io/`, `utils/`, `viz/`, `models/`, `pipeline_orchestrator/` are
deliberately **cross-cutting and do NOT move** into any stage folder — they're imported *by* stages.
The plan doc has an explicit "why" for this; don't fold them in for tidiness.

**Open sub-decision, still unresolved:** `utils/` holds only `cuda_diagnostics.py` (7 importers) —
fold into `shared/` (no bare one-file folder) or leave as-is (conventional name)? Low stakes, ask the
user or use judgment.

### Phase 5 (Task #6 in the plan doc): reconcile paths.py + doctrine vocabulary
- Make source folder names *equal* the `stage` strings already in
  `pipeline_orchestrator/orchestration/paths.py::PIPELINE_STEPS` (this should mostly already be true
  once Phase 4 lands — this phase is the verification + cleanup pass).
- Fix `docs/refactors/streamline-snakemake/target/specs/output_tree_doctrine.md`'s river prose:
  it currently says `... → features → ...`, should say `... → feature_extraction → ...` (decided
  2026-07-01: `feature_extraction` because it mirrors `object_extraction` — the user's own reasoning,
  don't relitigate this one).
- Update the `paths.py` docstring so code, docs, and disk layout all agree.

---

## Standing rules for the rest of this work (don't relitigate these)

- **Hard cutover, no compat shims.** No re-export shims at old import paths during the Phase 4 move —
  fix every importer directly. User explicitly chose this over a softer shim-based migration.
- **Commit after every phase, tests green before each commit.** The user's own words: "go through
  each folder, do the mapping, and then validate the test. Rinse, wash, repeat, and commit each
  time." Don't batch multiple stages into one commit.
- **Archive, don't delete, when something looks dead but you're not 100% sure.** Matches the
  `segmentation/_archive/` and pre-existing `schemas/_archive/` pattern. Deletion is a separate,
  later, explicit decision.
- **Don't guess on architectural forks — ask.** This work involved several `AskUserQuestion` calls
  on real forks (identifier grammar vs minting home, channel vocab home, archive-vs-fix-in-place for
  segmentation_and_tracking). If Phase 4/5 surfaces a similar fork (e.g. "does this file really
  belong in acquisition or is it cross-cutting?"), ask rather than picking the shape that seems
  cleanest.
