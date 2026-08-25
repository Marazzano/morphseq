# UNet backend split map — Phase 2 audit (NO MOVES YET)

**Status:** AUDIT (2026-07-01). Confirms the inference/adapter seam for the `unet_snip` backend and
lists exact file destinations + import repoints. Nothing moved. Feeds §7 Phase 2 of
`mophseq_env_pixi_orchestrator.md`.

Verified against: existing conda envs (`segmentation_grounded_sam` = pipeline/RUN,
`mseq_pipeline_py3.9` = MODEL_RUN). Pixi deferred — this split is env-agnostic.

---

## Current reality (the "before")

`snip_auxiliary_masks` runs as ONE `{RUN}` subprocess in the pipeline env
(`rules/snip_auxiliary_masks.smk` → `tasks.py::cmd_snip_auxiliary_masks` →
`unet_snip/entrypoint.py`). Torch is imported *inside that same process* via
`model_loader.load_unet_snip_predictors`. There is **no raw-sidecar seam today** — one process
loads the model, runs inference, writes PNGs, AND writes the final contract CSV.

**DECISION (2026-07-01): UNet is an INTEGRATED backend (Class A) — keep it in-process, NO sidecar.**
The whole "split into backend subprocess + sidecar + adapter rule" apparatus (earlier drafts of this
file) is ABANDONED. Rationale settled in `mophseq_env_pixi_orchestrator.md` §2: MorphSeq is the
published unit, not UNet-as-a-service; the pure adapter surface is verified pure so the backend env
importing it costs nothing; and no env boundary forces a sidecar here (both halves run 3.10 today).
The refactor is therefore a **file relocation**, not a DAG change.

## Responsibility layering (already clean by responsibility, not by process)

| file | responsibility | torch? | destination |
|---|---|---|---|
| `backends/unet_snip/model_loader.py` | `FishModelSnipPredictor`, config parse, `load_unet_snip_predictors` | **YES** | `external_backends/unet_backend/` |
| `models/unet/*` (`load_fish_unet_model`) | the UNet model itself | **YES** | `external_backends/unet_backend/` |
| `backends/unet_snip/run_unet_snip.py` | run predictors over inventory, build rows, write PNGs, validate | no | SPLIT: predictor loop → backend; row-build/write/validate → `segmentation/adapt_unet_output.py` |
| `backends/unet_snip/entrypoint.py` | fs glue: read inventory, wire config→predictors→runner, imports adapter, write CSV | no* | backend CLI (imports pipeline-side adapter in-process) |
| `backends/unet_snip/snip_auxiliary_masks_contract.py` | the contract | no | pipeline-side contract home (NOT under backend/) |

\* `entrypoint.py` is torch-free but *calls* `load_unet_snip_predictors` (which imports torch) — that
call is the seam.

## The one import to resolve: FishModelSnipPredictor imports data_pipeline

`FishModelSnipPredictor.__call__` (`model_loader.py:115-135`) imports `snip_image_to_model_grid`,
`back_to_snip_frame`, `assert_on_snip_frame` from `data_pipeline.snip_processing.snip_frame_masks`.
Under the INTEGRATED model this is **fine** — the backend CLI runs in an env that has `data_pipeline`
importable, and these helpers are pure numpy. No sidecar, no coordinate-space gate needed. The round
trip stays in RAM exactly as today; "only the promise gets written" (`model_loader.py:36-41`) is
preserved. The only change is WHERE the code lives, not how it runs.

## The move (integrated, in-process)

```txt
external_backends/unet_backend/src/morphseq_unet_backend/   # heavy env (has data_pipeline importable)
  model_loader.py        # FishModelSnipPredictor, load_unet_snip_predictors  (torch)
  models/unet/*          # load_fish_unet_model
  inference.py           # from segmentation/unet/
  entrypoint.py          # CLI: read inventory -> load predictors -> run -> IMPORT adapter -> write CSV
  run_unet_snip.py       # predictor-orchestration loop

src/data_pipeline/segmentation/                             # pipeline-owned, PURE adapter surface
  adapt_unet_output.py   # row-building + write_auxiliary_mask_png + assert_on_snip_frame
                         #   + validate_snip_auxiliary_masks  (the entrypoint imports THIS)
  # snip_auxiliary_masks_contract.py stays under its owning snip/segmentation contract home
```

The backend entrypoint does inference then calls `adapt_unet_output` in-process and writes the
canonical CSV directly — same as today, just relocated. `snip_image_to_model_grid` /
`back_to_snip_frame` stay in `data_pipeline.snip_processing.snip_frame_masks`; the backend imports
them (pure, no cost).

## Import repoints (after the move)

- `tasks.py:581` `cmd_snip_auxiliary_masks` — import the relocated backend entrypoint
  (`morphseq_unet_backend.entrypoint`); it in turn imports the pipeline-side adapter.
- `tasks.py:610` `cmd_validate_snip_auxiliary_masks` — repoint contract import to its pipeline-side
  home (torch-free, trivial).
- `viz/render_snip.py:19` — repoint `snip_auxiliary_masks_contract` import to pipeline-side home.
- `rules/snip_auxiliary_masks.smk` — runner string only (env swap later); the single rule shape is
  UNCHANGED. validate/merge rules untouched.

## Runner + CPU

- **Runner → `conda run -n segmentation_grounded_sam` now, pixi later.** Prove the move under
  known-good runtime so a B01 divergence can't be blamed on pixi. Pixi is a later runner-string swap.
- **CPU is fine.** UNet runs on `device="cpu"`; the wire-through + contract gate need NO GPU node.

## Baseline (captured)

- **Contract baseline SNAPSHOTTED:** `docs/data_pipeline/specs/target/specs/baselines/
  B01_snip_auxiliary_masks.BASELINE.csv` (copied from the existing `.validated` B01 run).
- **Gate fingerprint:** rows=4, snips=1, mask_types=[bubble,focus,via,yolk], all_valid=True,
  mask_shape=(576,256), schema_sha1=060619fecb9b.
- **CAUTION — on-disk PNGs are STALE:** the B01 well dir has PNGs for t0000/t0001/t0002… (leftover
  from a prior multi-tp run) but the manifest is 1 snip. Do NOT pixel-diff the directory; the CSV
  manifest is the trustworthy reference. For pixel parity, regenerate cleanly on CPU and compare with
  geometry tolerance.

## Gate sequence after the move (CPU)

1. Contract gate (no GPU): re-run B01 aux-masks on CPU post-move → new manifest matches the gate
   fingerprint above (schema + rows + snips + mask_types + all_valid + mask_shape).
2. Install guardrail (§2): `python -c "import morphseq_unet_backend.entrypoint"` resolves in the unet
   env (proves the integrated seam — env resolved AND the adapter surface imports there).
3. Pixel gate (optional, CPU): compare post-move mask PNGs to a clean pre-move CPU regen, geometry
   tolerance (not byte-diff).
