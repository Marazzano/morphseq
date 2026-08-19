# Tonight's agent prompt — Phase 0: restore one executable core stack

Paste from the line below. Fill `<TRAINING_ENV>` first. Run on branch `slice/phase0`.

---

You are working in the MorphSeq latent-morphology-model repo. Read `docs/refactor/AGENTS.md` and `docs/refactor/DECISIONS.md`
before making changes; they are binding. The integration audit is at
`docs/refactor/NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md`. This task makes **no design decisions** —
everything below is already ratified. If something here conflicts with what you find in the code,
report the conflict in your summary rather than resolving it yourself.

**Goal.** Make the core stack importable, runnable, and provably metric-capable on the *existing*
legacy data, before we change the data layer. Proving the metric machinery works on unchanged data
is the point — it de-risks the pipeline-manifest migration that follows.

## 1. Packaging and imports

`src/core/models/model_configs.py:8` does `from data.dataset_configs import NTXentDataConfig`.
That is not `src.core.data.dataset_configs`. With `src` on `PYTHONPATH` it silently resolves to
`src/data/dataset_configs.py`, an empty downstream-analysis shim whose `NTXentDataConfig` has no
machinery — so the default dataconfig is inert.

- Make the repo a properly installed package (`pip install -e .`) with a declared package root and
  absolute `src.core.*` imports throughout. Do **not** fix this with `sys.path` or `PYTHONPATH`
  manipulation anywhere — that is the bug, not the fix. Do not mix `src.core...` and `core...`.
- **Do not move, rename, or restructure any file or directory.** The package tree stays exactly as it
  is. This task changes import statements, adds a packaging file, and fixes config paths — nothing
  else. If you believe a move is required, stop and report it instead of doing it.
- Do **not** delete or modify `src/data` (may serve downstream analysis) or `src/vae` (dead for
  `src/core` but **live for `src/analyze`**). If anything depends on the old resolution, report it.

## 2. Entrypoints

`src/core/run/training.py` uses an absolute Hydra path `/src/core/hydra_configs`;
`training_cluster.py` points at `src/hydra_configs` while live configs are under
`src/core/hydra_configs`; scripts under `src/core/run` invoke `src.run.*` rather than `src.core.run.*`.
Pick **one** supported entrypoint, make its Hydra path package-relative, update or retire the rest.

## 3. The second metric mis-wiring cause

The metric hydra config sets `dataconfig.target: "BasicDataset"` (single image), so
`metricVAE.forward` takes the vanilla branch and the loss's `unbind(dim=1)` fails. Fix it to select
the NTXent dataset. **Fixing the import in §1 without this leaves the metric path dead**, and it
will look like it worked.

## 4. Image size becomes configuration, and the pixel-scale literal must follow

`loss_functions.py:185-187` normalises reconstruction by a literal `128*288`, independent of
`input_dim`. Make model input size a config parameter (default `(1, 288, 128)`) and derive
`_pixel_scale` from `input_dim`. At 288×128 the derived value equals the literal, so no existing run
changes — but the moment size is tunable, the literal silently mis-scales reconstruction against KL
and the metric term. Add a test asserting derived == literal at the default geometry.

Verify whether `_pixel_scale` divides reconstruction or multiplies KL, and say which in your summary.

## 5. Ratified hygiene

- Delete the **margin term** from the metric loss entirely (not disable). It is shared across all
  logits and cancels in the softmax — confirmed inert, no experiments invalidated.
- Delete `accumulate_grad_batches` (never passed to Trainer, ignored under manual optimization).
- Add the missing `logvar` clamp to `metricVAE` for parity with `VAE`.
- `tv_weight` is computed but never summed into the loss — report it, don't remove it yet.

## 6. Tests

- `tests/core/test_imports.py`: import the model config module, compose the metric Hydra config, and
  **assert the concrete class** of the resulting data config is the training implementation from
  `src.core.data`. Asserting that composition merely succeeded is exactly what failed to catch this.
- A metric forward/backward smoke test on CPU that goes down the **metric** branch: batch shape
  `[2, 1, 288, 128]`, `unbind(dim=1)` succeeds, and reconstruction, KL, and metric terms are all finite.
- A basic VAE forward/backward smoke test on CPU.
- Make trainer hardware configurable enough to run these on CPU. Currently hardcoded to GPU, FP16,
  all devices, DDP.
- Report whether `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` imports cleanly in
  `<TRAINING_ENV>` with only training dependencies. If it does not, **do not install the pipeline's
  dependencies** — report the transitive import chain instead.

## Constraints

Do not touch the model architecture, latent partitioning, or any loss formula beyond the margin
deletion in §5. Do not touch `src/core/data/pipeline_manifest.py` (does not exist yet), the dataset
classes, or the Lightning wrappers — a later slice owns those. Do not weaken an assertion to make it
pass; report it failing instead.

## Done when

`pip install -e .` from a clean checkout, the chosen entrypoint runs `--help`, all smoke tests pass
on CPU, and no entrypoint or script mutates `sys.path`.

## In your summary, state

The packaging choice and why; the entrypoint you kept and what you retired; whether `_pixel_scale`
divides recon or multiplies KL; the result of the `src.data_pipeline` import check with the chain if
it failed; and anything you found that depended on the old import resolution.
