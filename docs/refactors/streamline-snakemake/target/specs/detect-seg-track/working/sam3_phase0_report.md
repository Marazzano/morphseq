# SAM3.1 Phase-0 Beachhead — Empirical Report

**Status:** GREEN. Proven on a real GPU node on 2026-07-01. This report records the facts that should
now be codified in `install_backend.py`, `install_model.py`, `loader.py`, and `verify_backend.py`.
It supersedes the older transformers/HF-snapshot assumptions in `targets/sam3_spec.md`.

## Current decision

Use **native `facebookresearch/sam3` only**:

- `device="cuda"` -> `build_sam3_predictor(version="sam3.1")` as the canonical production path.
- `device="cpu"` -> `build_sam3_predictor(version="sam3")` as a dev/smoke path.
- Both versions use the same native request API:

```python
predictor = build_sam3_predictor(version="sam3")  # or "sam3.1"
response = predictor.handle_request({"type": "start_session", ...})
predictor.handle_request({"type": "add_prompt", ...})
for out in predictor.handle_stream_request({"type": "propagate_in_video", ...}):
    ...
```

This removes the old framework confound. CPU/dev and CUDA/canonical legs still use different model
versions and weights, but they no longer use two separate inference APIs.

## Decisive corrections

- **SAM3.1 is not a transformers model today.** `facebook/sam3.1` ships a raw
  `sam3.1_multiplex.pt` checkpoint and no Hugging Face Transformers integration. SAM3.1 loads through
  Meta's native `facebookresearch/sam3` package.
- **Plain SAM3.0 is CPU-safe in the native package.** `build_sam3_predictor(version="sam3")` avoids
  the SAM3.1 multiplex import-time CUDA probe and honors CPU fallback.
- **SAM3.1 remains GPU-only for MorphSeq.** The multiplex path imports code with a top-level CUDA
  capability probe (`sam3/model/sam3_multiplex_base.py:36`). With no visible GPU, that import fails
  before a `device` argument can help.
- **`use_rope_real=True` is load-bearing.** The predictor wrapper is the safe target interface; lower
  level model builders can hit checkpoint key-layout mismatches.
- **FA3 is a production performance option, not a correctness dependency.** Do not put FA3 inside the
  equivalence harness.

## Proven environment

Committed stable target environment in `pixi.toml`:

```txt
python 3.12
torch / torchvision CUDA wheel stack
native SAM3 runtime deps
native facebookresearch/sam3 checkout, pinned externally by install_backend.py
```

Empirical node:

```txt
GPU: NVIDIA L40S
driver: 560.35
system CUDA: 12.6
cu128 torch runtime: cuda_available=True via forward compatibility
```

Observed results for the native SAM3 path:

- Native SAM3.1 predictor load succeeded on the L40S with `use_fa3=False, use_rope_real=True`.
- The loaded checkpoint was `facebook/sam3.1` / `sam3.1_multiplex.pt`.
- The setup report correctly classifies this state as `backend_class=detached` until the detection
  adapter/contract smoke exists.
- The FA3/cu128 lane was resolved/exercised during Phase 0, but after the `sam3` / `sam3-fa3` split it
  is treated as an unverified performance lane rather than the MVP install path.

## Runtime dependency findings

These were found empirically and are encoded in the stable `sam3-backend` pixi feature:

1. **torch / torchvision** need the CUDA wheel stack; avoid accidentally resolving CPU-only builds.
2. **Native SAM3 deps:** `timm`, `numpy<2`, `ftfy`, `regex`, `iopath`, `tqdm`.
3. **setuptools<81** is required because native SAM3 imports `pkg_resources`.
4. **einops** is imported unconditionally by `sam3/sam/rope.py`.
5. **pycocotools** is imported through training/collator modules even for inference-only use.
6. **psutil** is also required by the native package path exercised during load.

Transformers is intentionally **not** a dependency of the current `sam3` env.

## Install and load facts to codify

- The upstream runtime lives outside git, e.g. `~/.cache/morphseq/vendor/sam3`, and is exposed through
  `PYTHONPATH`. Editable installs are not the preferred mechanism because a pixi re-resolve can break
  them.
- `install_backend.py` owns cloning/updating that checkout to the pinned commit and reporting:
  vendor path, git URL, commit, dirty state, and importability.
- `install_model.py` owns fetching/checking the gated SAM3.1 checkpoint and writing a checkpoint
  manifest.
- The native checkpoint helper uses Hugging Face internals and does not provide the same
  `from_pretrained(cache_dir=...)` control surface. Prefer:
  - `HF_HUB_CACHE` to steer artifact placement under `$MORPHSEQ_MODEL_CACHE/sam3/hf/`.
  - `HF_TOKEN` for gated access, so moving cache directories does not silently lose auth.
  - Avoid relocating `HF_HOME` unless token handling is explicit.
- `loader.py` should route by config/device:
  - CUDA canonical: `version="sam3.1"`, `use_rope_real=True`, default `use_fa3=False`.
  - CPU/dev: `version="sam3"`, same native request API.

## Equivalence gate

First Phase-1 task:

```txt
sam3.0 CPU  vs  sam3.0 CUDA  vs  sam3.1 CUDA
```

Run all three through the native `handle_request` API on the same tiny fixture, with `use_fa3=False`.
This separates:

- device effects (`sam3.0 CPU` vs `sam3.0 CUDA`)
- model-version effects (`sam3.0 CUDA` vs `sam3.1 CUDA`)

Until this passes, CPU SAM3 remains a dev/smoke path and should not produce canonical outputs.

## SAM3-FA3 lane

`sam3-fa3` is reserved for production performance. It is not part of the MVP and is not required for
correctness/equivalence. Before enabling it, run a separate CUDA smoke requiring:

- CUDA available.
- NVIDIA GPU compute capability >= 8.0.
- `flash_attn` / native SAM3 FA3 imports succeed.
- bf16 or fp16 autocast is active; fp32 is not acceptable for the FA3 runtime path on Ampere/Ada.
- One tiny SAM3.1 `use_fa3=True` inference completes without crashing.

If the smoke passes, set `fa3_available=true` and `production_use_fa3=true`. If any gate fails, set
`fa3_available=false`, keep `production_use_fa3=false`, and fall back to the stable `sam3` env with
`use_fa3=False`.

## Backend report fields to preserve

`verify-backend sam3` should write enough detail that Snakemake and future maintainers do not infer
readiness from hardware alone:

```txt
sam3_runtime_kind = native_facebookresearch_sam3
sam3_vendor_path
sam3_vendor_commit
sam3_version
sam3_device
torch_version
torchvision_version
cuda_available
cuda_version
gpu_name
compute_capability
bf16_supported
recommended_precision
artifact_kind = hf_raw_checkpoint
hf_model_id
hf_revision
hf_cache_dir
checkpoint_path
checkpoint_sha256
hf_access_ok
real_model_load_ok
exemplar_prompt_smoke_ok
adapter_import_ok
forbidden_imports_ok
contract_write_smoke_ok
sam3_detection_ok
flash_attn_import_ok
fa3_available
fa3_smoke_test
production_use_fa3
equivalence_status
equivalence_report_path
```

## Open follow-ups

- Re-solve `pixi.lock` on a pixi-enabled node if it still carries packages from the old transformers
  experiment.
- Implement the backend scripts and loader from the facts above.
- Add the three-way equivalence smoke before treating CPU/dev outputs as comparable.
- Add the FA3 smoke only as a production performance gate.
- Add Snakemake runner wiring for SAM3 GPU execution, analogous to the existing specialized runners.
