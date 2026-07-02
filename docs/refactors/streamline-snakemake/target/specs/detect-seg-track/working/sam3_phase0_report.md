# SAM3.1 Phase-0 Beachhead — Empirical Report

**Status:** GREEN. Proven on a real GPU node 2026-07-01. This records what actually worked, so
`install_backend.py` / `install_model.py` / `loader.py` can be codified from fact, not the spec's
guesses. Supersedes the class-name and load-path assumptions in `targets/sam3_spec.md` §2–§5.

## The decisive correction to the spec

`sam3_spec.md` assumes SAM3 loads via `transformers` (`Sam3Model.from_pretrained("facebook/sam3")`).
That is **wrong for SAM 3.1**. Audited facts:

- `facebook/sam3.1` (HF, gated=manual) ships **only a raw checkpoint** `sam3.1_multiplex.pt`. Its
  README states verbatim: *"there is no Hugging Face Transformers integration."* → 3.1 loads through
  Meta's **native `facebookresearch/sam3` package**, not transformers.
- Plain `facebook/sam3` DOES have a transformers-format checkpoint (`model.safetensors`) and loads via
  `Sam3VideoModel.from_pretrained` (also proven green). It is the transformers-path fallback, not 3.1.
- transformers class names, when used, are `Sam3VideoModel` / `Sam3VideoProcessor` (model_type
  `sam3_video`), NOT the spec's `Sam3Model`/`Sam3Processor`.

**Decision (user):** use **SAM 3.1** via the native vendored package. This is the spec's §11 "vendored
checkout" path — normally debug-only, but the only path for 3.1 until Meta ships transformers weights.

## Proven environment (pixi `sam3` env, L40S / driver 560.35 / CUDA 12.6)

```
python 3.12.13   torch 2.7.1 (cuda 12.6, cuda_available=True)   torchvision 0.22.0
timm 1.0.27      einops 0.8.2     setuptools 80.10.2 (<81)       gpu NVIDIA L40S
```

Result: `Sam3MultiplexVideoPredictor` built + loaded in ~45s. `build_sam3_multiplex_video_predictor(
use_fa3=False, use_rope_real=True)` → `SAM3.1 MULTIPLEX PREDICTOR READY`.

## The empirical dep chain (each found by running on the node, now codified in pixi.toml)

1. **torchvision** — `Sam3VideoProcessor` / native ops require it. cuda126 build to avoid CPU-only.
2. **native sam3 runtime deps** — `timm, numpy<2, ftfy==6.1.1, regex, iopath, tqdm` (from its pyproject).
3. **setuptools<81** — sam3 does `import pkg_resources`; setuptools 81+ removed it. Pins cleanly ONLY
   because transformers (which forced setuptools 82) is not in the env.
4. **einops** — imported unconditionally by `sam3/sam/rope.py` (upstream lists it optional; it isn't).
5. **pycocotools** — dragged in at import via `train.data.collator` → `coco_json_loaders`, even for
   inference-only use.

## Load-path facts for loader.py / install_model.py

- **Vendored package** at `~/.cache/morphseq/vendor/sam3` (git commit `5dd401d`), reached via
  **PYTHONPATH**, NOT `pip install -e` (editable installs do NOT survive a `pixi install` re-resolve).
- **Correct entry point:** `build_sam3_multiplex_video_predictor(use_fa3=False, use_rope_real=True)`.
  - `use_rope_real=True` is load-bearing: the low-level `build_sam3_multiplex_video_model` defaults it
    False → fatal state_dict key mismatch (wrong RoPE key layout). The predictor wrapper defaults it
    True. Use the predictor, not the raw model builder.
  - `use_fa3=False` because FlashAttention-3 is not installed (optional, perf-only). Yields 64 benign
    "Missing keys" (`freqs_cis_real/imag` — derived RoPE buffers, recomputed at runtime, non-fatal).
- **Checkpoint fetch:** `download_ckpt_from_hf(version="sam3.1")` → `sam3.1_multiplex.pt` (3.3G). It
  uses `hf_hub_download` with **no `cache_dir` param**, so it lands in the default HF hub cache
  (`~/.cache/huggingface/hub`), NOT the spec's `$MORPHSEQ_MODEL_CACHE/sam3/hf/`. To honor spec §2.1,
  `install_model.py` must steer via **`HF_HOME` / `HF_HUB_CACHE` env vars**, not a function arg.
- **Auth:** gated repo → needs a logged-in token. `HF_HOME` must NOT be relocated away from where
  `hf auth login` stored the token (`~/.cache/huggingface/`), or auth silently breaks with a 401.
  Prefer exporting `HF_TOKEN` (discoverable regardless of HF_HOME) + steering only `HF_HUB_CACHE`.

## DUAL-STACK by device (adopted 2026-07-01) — supersedes "GPU-only / CPU rejected" below

Rather than accept GPU-only, the `sam3` env carries BOTH stacks and the loader routes by device:

- **`device="cuda"`** → NATIVE `facebookresearch/sam3`, SAM3.1 multiplex, FlashAttention-3.
  `build_sam3_multiplex_video_predictor(use_fa3=True, use_rope_real=True)`. Proven: loads on L40S ~43s.
- **`device="cpu"`** → transformers `Sam3VideoModel.from_pretrained("facebook/sam3")` (plain sam3;
  no CUDA probe). Proven: loads on CPU / fp32 in ~49s with the GPU hidden.

Proven committed env (one pixi lock, `[feature.sam3-backend]`, channels=["conda-forge"], cu128 index,
index-strategy unsafe-best-match): **python 3.12 · torch 2.10.0+cu128 · torchvision 0.25.0+cu128 ·
flash-attn-3 3.0.0 · transformers 5.12.1**, `torch.cuda.is_available()=True` on the L40S (CUDA 12.6
driver, cu128 runtime via forward-compat). Native deps (all found empirically): timm, numpy<2, ftfy,
regex, iopath, tqdm, einops, pycocotools, psutil, setuptools<81.

**KNOWN DEVIATION (user-accepted, must be gated):** CPU and CUDA run DIFFERENT models — CPU=sam3.0
(transformers safetensors), CUDA=sam3.1-multiplex (native .pt), different weights + tracking algo. The
loader must record which variant ran, and CPU is a nice-to-have (dev/smoke), NOT for canonical outputs
until an **output-equivalence check** confirms the two agree closely on the same image. That check is
the FIRST Phase-1 task (not yet done — the two stacks have different inference APIs: native
`handle_request` prompt API vs transformers processor `postprocess_outputs`).

## (Historical) CPU load: NOT supported for SAM3.1 — why the dual-stack was needed

SAM3.1 multiplex is architecturally GPU-committed. Two independent walls, both confirmed on the node:

1. **Import-time CUDA probe.** `sam3/model/sam3_multiplex_base.py:36` runs a bare module-level
   `if torch.cuda.get_device_properties(0).major >= 8:` at import. With the GPU hidden
   (`CUDA_VISIBLE_DEVICES=""`) this raises `RuntimeError: No CUDA GPUs are available` before any
   `device` argument is consulted — so the recommended `build_sam3_multiplex_video_predictor` entry
   point cannot even be imported on CPU.
2. **CPU builder can't load the checkpoint.** The lower-level `build_sam3_multiplex_video_model(
   device="cpu")` avoids that probe but does its own `load_state_dict(strict=True)` with a key layout
   that does NOT match `sam3.1_multiplex.pt` (fails even with `use_rope_real=True`). Only the
   predictor wrapper does the correct key remap/merge — and that wrapper is the CUDA-gated one.

The non-multiplex `build_sam3_video_predictor` returns `Sam3VideoPredictorMultiGPU` (also GPU-centric).

**Decision:** SAM3.1 is **GPU-only** in MorphSeq. Run it in the `sam3` pixi env on a GPU node (a
`SAM3_RUN` Snakemake runner, mirroring `MATERIALIZATION_RUN`), same as `materialization`/`sam2`.
A CPU SAM3.1 would require a maintained fork (patch the probe AND fix the CPU state_dict remap) and
would be unusably slow regardless. If a CPU-loadable SAM3 is ever needed (CI smoke, dev boxes), the
clean path is plain `facebook/sam3` via transformers `Sam3VideoModel.from_pretrained` (no CUDA probe;
proven to load earlier) as a SEPARATE additive feature — not a patch to the vendored 3.1 package.

## FlashAttention-3 (perf, deferred)

`use_fa3=True` imports `flash_attn_interface` lazily via `sam3/perflib/fa3.py` (only when enabled — so
`use_fa3=False` is clean). It powers 3.1's headline ~7x-at-128-objects speedup, but: CUDA-only,
upstream ships a **cu128** wheel (`pip install flash-attn-3 --no-deps --index-url .../cu128`) that
mismatches our **cu126** env (fallback = a `ninja`+CUDA-toolkit source build), and the model already
loads/runs with `use_fa3=False`. For embryo images (few objects/frame) the gain is modest. **Deferred**
as a later perf item, not a correctness blocker.

## Open follow-ups (not Phase-0 blockers)

- FlashAttention-3 build for `use_fa3=True` (perf; ~7x multi-object speed is the 3.1 selling point).
- Snakemake wiring: a `SAM3_RUN` runner (`pixi run -e sam3 …`) + vendored PYTHONPATH injection.
- The pixi.toml `sam3-backend` feature currently declares only the vendored package's DEPS; the
  vendored `sam3` package itself is not pinned (gitignored checkout) — install_backend.py owns cloning
  it to `~/.cache/morphseq/vendor/sam3` at the recorded commit.
