# SAM3 Integration — Target Spec

**Status:** TARGET SPEC (aspirational; refined against Phase-0 facts recorded 2026-07-01 in
`../working/sam3_phase0_report.md`). The original transformers/HF-snapshot assumptions were
superseded by the empirical native-SAM3 path; audit every path/command against the real tree before
treating as implemented.

**Purpose:** Wire SAM3 into MorphSeq as an **exemplar-driven detection backend first**, and a
**seg-track backend later — as two SEPARATE integrations**, in a way a biologist can install with one
command and a maintainer can fix when it breaks.

**Related:** [[mophseq_env_pixi_orchestrator]] (env + backend split), [[detection_world]]
(`frame_detections` contract), [[segmentation_world]] (`frame_masks` contract),
[[source_tree_stage_alignment]] (stage homes).

---

## 0. Tattoo

> **Two bridges, different rivers.** The **loader** makes the model *usable*
> (installed package + checkpoint → ready model object). The **adapter** makes the output *canonical*
> (raw model output → pipeline contract). Do not mix them.
>
> **We ship the *way to install*, not the model.** Upstream SAM3 code and weights are installed and
> fetched, never vendored into git.
>
> **`MORPHSEQ_MODEL_CACHE` is one gate into the artifact kingdom, many species inside it.** The cache
> root is a single MorphSeq-controlled abstraction; the artifact *layout* under it is backend-specific
> (SAM2 = explicit `.pt` files; SAM3.1 = native upstream checkout + HF-hosted raw checkpoint). Each backend's
> `install_model.py` owns the ritual for making its upstream artifacts land under that root.

### Vocabulary

This repo already uses **backend** for pipeline-side detection code under
`src/data_pipeline/object_extraction/detection/backends/<backend>/`. Keep that naming to avoid
churn. When ambiguity matters, use **model backend** for the heavy runtime integration under
`model_backends/<model>/`, and **detection backend** for the pipeline adapter/router/filter under
`src/data_pipeline/object_extraction/detection/backends/<backend>/`.

> The SAM3 **model backend** powers the SAM3 **detection backend**.

---

## 1. Aspiration & scope (the reframe)

SAM3 enters MorphSeq as **two separate integrations**, not one fused detect+segment+track stack:

```txt
SAM3 detection backend  -> frame_detections   (FIRST; this spec's main body)
SAM3 seg-track backend  -> frame_masks         (LATER; parallels SAM2, reserved §6)
```

The two share the **model bundle** (one env, one loader, one weights cache) but NOT the pipeline
seam. Each has its own CLI, its own adapter, its own contract, and can land in a different phase.

**Explicitly killed** (versus an earlier draft of this spec):

- **Mode 3 (fused detect+segment+track in one backend call).** It collapses the
  `frame_detections → frame_masks` river into one GPU pass, which breaks the detection→segmentation
  DAG split and destroys the `prompt_detection_id → track_id → mask_id` provenance chain (a text
  prompt has no `detection_id`). Not built. See §11.
- **The fat `detection_prompt_manifest.csv`** (15 columns, seed frames, tracking direction,
  positive/negative). Its seed-frame and tracking-direction columns belong to *segmentation*, which
  stays on the existing SAM2 seed-selection path. Replaced by the leaner **exemplar library** (§4).

**Untouched downstream** — the whole point of keeping the split: seed selection, SAM2 tracking, and
`frame_masks` do not change. SAM3-detection emits `frame_detections` rows like any other detector;
`kept_frame_detections(...)` and everything after it is identical.

### 1.1 Phase-0 developments to carry forward

These are the empirical findings from the 2026-07-01 beachhead that must be preserved in code,
reports, and future handoffs:

- **SAM3.1 is native-only today.** `facebook/sam3.1` provides a raw `sam3.1_multiplex.pt` checkpoint,
  not a transformers model. MorphSeq loads 3.1 through a pinned `facebookresearch/sam3` checkout.
- **One native API covers both useful legs.** `build_sam3_predictor(version="sam3.1")` is the
  canonical CUDA path; `build_sam3_predictor(version="sam3")` is the CPU/dev path. Both use
  `handle_request` / `handle_stream_request`, so the equivalence check no longer has a framework
  confound.
- **SAM3.1 remains GPU-only.** The multiplex path has an import-time CUDA capability probe and should
  run through a `SAM3_RUN`-style GPU runner, not on CPU/login nodes.
- **`use_rope_real=True` is load-bearing.** The predictor wrapper's key layout/remap is the safe load
  path; the raw lower-level model builder is not the target interface.
- **FA3 is production performance, not correctness.** The base `sam3` env is the stable non-FA3
  runtime. `sam3-fa3` is a reserved performance lane and must pass a separate CUDA half-precision
  smoke before production enables `use_fa3=True`. The equivalence harness should use `use_fa3=False`.
- **HF cache/auth steering is subtle.** The native checkpoint helper is controlled by HF environment
  variables, not a `from_pretrained(cache_dir=...)` call. Prefer steering `HF_HUB_CACHE` for artifacts
  and using `HF_TOKEN` for auth so relocating `HF_HOME` does not break gated access.
- **The backend report is the truth artifact.** It must record hardware, dtype recommendation,
  vendor commit, checkpoint identity, SAM3 version, FA3 availability, and equivalence status so
  Snakemake does not infer production readiness from hardware alone.

---

## 2. Three homes, three lifecycles (the "where does it land" answer)

The vendor ships two separable things — architecture-as-code and parameters-as-weights — and MorphSeq
routes them to **different homes with different lifecycles**. Plus MorphSeq's own integration code.
Three locations, three owners:

| what | where | in git? | installed by | lifecycle |
|------|-------|---------|--------------|-----------|
| MorphSeq SAM3 integration (install scripts, loader, config, CLI) | `model_backends/sam3/` | **yes** (tiny) | authored | changes when *we* edit |
| upstream SAM3 runtime (`facebookresearch/sam3`) | gitignored vendor cache, exposed to the `sam3` env via `PYTHONPATH` | no | `install-backend` | clone/update to pinned commit |
| SAM3 model artifacts (`sam3.1_multiplex.pt`, plus optional `sam3` dev weights) | `$MORPHSEQ_MODEL_CACHE/sam3/` via steered HF hub cache | no | `install-model` | fetch once; big; shared |

Keeping code and artifacts in separate homes is *why* iterating is cheap: bump the upstream SAM3
runtime pin and re-run `install-backend` **without** re-downloading weights; re-run `install-model`
**without** re-resolving the env. This is the physical basis of the "fixable when it breaks" property
(§7).

**SAM2 vs SAM3 load — a decisive asymmetry (audited 2026-07-01).** The two models do NOT load the
same way, and the doctrine must not pretend they do:

- **SAM2** (`facebookresearch/sam2`): `build_sam2_video_predictor(config_file, ckpt_path)`. The config
  is a **Hydra** YAML that travels *inside the installed package* (`compose(config_name=...)` finds it
  on the package search path — no separate config file to manage). The **checkpoint is an explicit
  `ckpt_path` argument** you manage; upstream's README defaults it to a repo-local `checkpoints/` dir
  (the wrinkle — see §2.1).
- **SAM3.1** (`facebookresearch/sam3`): `build_sam3_predictor(version="sam3.1")` /
  `build_sam3_multiplex_video_predictor(use_rope_real=True, use_fa3=<bool>)`. The 3.1 HF repo ships a
  raw `sam3.1_multiplex.pt` checkpoint and explicitly has **no transformers integration**. The native
  package downloads the checkpoint through Hugging Face helper code; MorphSeq must steer the HF hub
  cache via `HF_HUB_CACHE`/related environment rather than by passing a `cache_dir` to
  `from_pretrained`.
- **SAM3.0 native** (`version="sam3"`): the same native package exposes a CPU-safe development/smoke
  path with the same `handle_request` / `handle_stream_request` API. It is not canonical for
  production outputs until the Phase-1 equivalence check says otherwise.

Consequence: for SAM3.1, a gitignored upstream checkout is **not a debug fallback**; it is the expected
runtime path until Meta ships a maintained package/transformers integration for 3.1. It must be
installed outside git, pinned by commit, and recorded in the setup report.

**Naming:** the backend home is `model_backends/` (renamed from the pixi spec's `external_backends/`).
`loader.py` is the only module that touches all three homes; its *shape differs per backend* (§3):
SAM2's loader passes an explicit `ckpt_path`, SAM3's loader injects/uses the vendored native package
and selects `version="sam3.1"` or `version="sam3"` by device/config.

### 2.1 Model cache resolution

Model checkpoints live in a MorphSeq model cache, **not** in the source repo. The repo and the cache
are different disk kingdoms (same name, different filesystem location):

```txt
/path/to/morphseq/                 # source repo (installer, loader, CLI, verify code)
~/.cache/morphseq/models/          # default model cache (downloaded checkpoints + metadata)
$MORPHSEQ_MODEL_CACHE/sam3/        # SAM3 artifacts under an overridden cache root
```

Cache root resolution order (three tiers; **no `env.yaml`** — that file is gitignored/conda-era and
must not sprout roots under the new torch-free config; see [[project_two_tree_reconciliation]]):

```txt
1. --model-cache        (best for one-off debugging)
2. MORPHSEQ_MODEL_CACHE  (best for cluster/shared config)
3. ~/.cache/morphseq/models   (default; local, no-config)
```

Backend artifacts live below the resolved root, but **the layout is backend-specific** — the cache
root is a MorphSeq-controlled artifact root, NOT a promise that every backend uses explicit `.pt`
checkpoints:

```txt
{model_cache_root}/{backend_name}/
  <backend-specific artifact layout>
  metadata/
```

SAM2 (explicit checkpoint files):

```txt
$MORPHSEQ_MODEL_CACHE/sam2/
  checkpoints/
    sam2.1_hiera_large.pt
  metadata/
    checkpoint_manifest.json
```

SAM3.1 (native checkpoint fetched through Hugging Face helper code — MorphSeq steers the HF hub
cache, but does not pretend this is a transformers snapshot):

```txt
$MORPHSEQ_MODEL_CACHE/sam3/
  hf/                              # HF hub cache controlled by MorphSeq (prefer HF_HUB_CACHE)
    models--facebook--sam3.1/
      snapshots/ refs/ blobs/
  metadata/
    checkpoint_manifest.json
    vendor_manifest.json
```

`install-model sam3` prefetches the gated `facebook/sam3.1` checkpoint into the steered HF hub cache.
The native upstream helper currently does not accept a `cache_dir` argument, so the reliable control
surface is environment steering: prefer `HF_HUB_CACHE` for artifact placement and `HF_TOKEN` for auth.
Avoid moving `HF_HOME` away from the user's login cache unless token handling is explicit. The setup
report records the resolved checkpoint path/commit; callers should not hard-code an upstream
cache-internal path.

Rules:

- The resolved cache root must be **absolute** after `~` expansion. `~` is allowed and expanded.
- **Relative paths are rejected in v1.** No repo-root discovery, no `.git` walking, no CWD (CWD is a
  frequent source of ambiguous behavior). A relative `--model-cache` fails clearly: *"Relative model cache paths are not
  supported. Pass an absolute path or use ~/..."*. Dev-relative support can be added later if anyone
  asks; it is deliberately out of scope so the resolver stays pure.

### Cache resolver home — pipeline-side, verified-bounded

The resolver is **not** model-heavy, **not** SAM3-specific, and **not** really backend code — it is
shared MorphSeq infrastructure. It lives pipeline-side, alongside existing shared infra
(`shared/identifiers/`, `shared/channel_vocabulary.py`, `shared/path_contracts.py`):

```txt
src/data_pipeline/shared/model_cache.py
  resolve_model_cache_root(cli_model_cache=None) -> Path
  resolve_backend_cache_root(backend_name, cli_model_cache=None) -> Path
```

It is part of the **verified-bounded adapter surface** that integrated backend CLIs may import
in-process inside their own pixi env — the *same* blessed bridge already proven at install time by the
adapter guardrail (§8). Every backend `config.py` calls it with its own name:

```python
from data_pipeline.shared.model_cache import resolve_backend_cache_root
sam3_cache = resolve_backend_cache_root("sam3", cli_model_cache=args.model_cache)
```

The resolver is torch-free and must not import Snakemake, `pipeline_orchestrator`, model stacks,
detection backends, or result scripts.

- **Not** in `model_backends/_shared/` — that sibling dir is not on the sam3 env's path without a new
  installable, a PYTHONPATH hack, or per-backend copies (all worse; a path resolver does not deserve
  its own installable package).
- **Not** duplicated into each backend — 30 lines today becomes four divergent path resolvers once
  metadata/manifest/HF conventions arrive.

---

## 3. Loader vs adapter doctrine (don't mix the bridges)

```txt
loader:   installed SAM3 package + checkpoint  ->  ready model/predictor object
adapter:  raw SAM3 output                       ->  canonical frame_detections / frame_masks
```

They cross different rivers. Keep them in different homes. **model backend** (runtime) vs **detection
backend** (pipeline-side):

```txt
model_backends/sam3/                         # MODEL BACKEND. HEAVY. SAM3.1 import is GPU-only.
  install_backend.py    # the WAY to install code: pins deps + vendored native SAM3 commit
  install_model.py      # the WAY to prefetch/checkpoint-manifest SAM3 weights into the cache
  verify_backend.py     # the CHECK (§8)
  backend.toml          # backend metadata (source, pin, variants)
  src/morphseq_sam3/
    config.py           # torch-free: variant, calls resolve_backend_cache_root("sam3")
    loader.py           # heavy: native build_sam3_predictor(...) -> predictor
    cli_detect.py       # the detection CLI Snakemake shells out to
    cli_track.py        # LATER: the seg-track CLI (§6)
    smoke.py

src/data_pipeline/object_extraction/detection/     # PIPELINE. contracts + backends stay here.
  frame_detections_contract.py    # EXISTS, unchanged
  validate_frame_detections.py    # EXISTS, unchanged
  kept_frame_detections.py        # EXISTS, unchanged
  exemplar_library.py             # NEW (§4)
  backends/
    groundingdino/                #   existing dir
    detectron2/                   #   existing dir; not a prerequisite for SAM3 (see §5 note)
    sam3/                         # NEW detection backend
      config.py                   #   is_kept thresholds, exemplar_set selection
      run_sam3_detection.py       #   backend router glue for the stage
      filter_sam3_detections.py   #   SAM3 scores -> is_kept  (the groundingdino-parallel seam)
      adapt_sam3_detections.py    #   PIPELINE-OWNED: raw -> frame_detections rows
```

`detection/backends/sam3/` is the **pipeline-side detection backend** (adapter/router/filter), not the
SAM3 model runtime. It must stay torch-free except where explicitly verified; model loading and
upstream imports live only in `model_backends/sam3/`.

Backend-specific adapters live under `detection/backends/<backend>/` when they are pipeline-owned and
contract-facing. They are **not** part of `model_backends/<backend>/`. This preserves the stage
boundary — `frame_detections` is shared across GroundingDINO / Detectron2 / SAM3.

**config.py is torch-free / loader.py is heavy / adapter is pipeline-owned.** The cache resolver (§2.1)
is the concrete proof that config.py is torch-free and liftable: cache-path resolution genuinely needs
no torch, which is exactly why it can be shared pipeline-side.

**The loader's shape differs per backend** (from the load-asymmetry in §2):

```python
# SAM2 loader — explicit checkpoint file; config travels with the package
sam2_cache = resolve_backend_cache_root("sam2")
ckpt_path  = sam2_cache / "checkpoints" / "sam2.1_hiera_large.pt"
predictor  = build_sam2_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_l.yaml",   # Hydra name, resolved inside the package
    ckpt_path=str(ckpt_path),
)

# SAM3 loader — native package + version/device routing; checkpoint helper uses steered HF hub cache
sam3_cache = resolve_backend_cache_root("sam3")
predictor  = build_sam3_predictor(
    version="sam3.1",        # canonical CUDA production path
    use_rope_real=True,      # load-bearing for checkpoint key layout
    use_fa3=False,           # correctness/equivalence default; production may enable after FA3 smoke
)
```

Use `version="sam3"` for CPU/dev smoke or for the Phase-1 three-way equivalence check. It shares the
same native request API, but uses different weights/algorithm, so the setup report must record the
actual `sam3_version` used for every smoke and production run.

The detection CLI imports **both** the loader and the pipeline adapter:

```txt
morphseq_sam3.loader                                                       (model-backend-side)
data_pipeline.object_extraction.detection.backends.sam3.adapt_sam3_detections   (pipeline-side)
```

This is allowed under the integrated-backend policy — **but proven at install time, not asserted.**
`verify-backend sam3` imports `adapt_sam3_detections` in the sam3 env and asserts no forbidden
transitive imports (§8). This matters because the adapter pulls in `frame_detections_contract` +
`exemplar_library` + `identifiers`, and any of those could drag in something heavy; the guardrail is
what catches it.

---

## 4. The exemplar library (the one genuinely new validated input asset)

Exemplars are **prompt sources, not detections.** A curated, on-disk, versioned set of example
images with boxes that tells the SAM3 concept detector what an embryo looks like *for this dataset's
visual quality*. Decoupled from seed selection (that stays SAM2's job — see §1).

The ergonomic (the whole reason this exists): a new dataset with a different look gets a new exemplar
set with a couple of examples. Point the detector at it via config. Adapt to new data quality without
touching prompt strings, retraining, or any other stage.

### Layout

```txt
exemplar_library/
  zebrafish_default/
    manifest.csv          # rows below
    images/
      zf_default_early_001.png
      zf_default_mid_001.png
      zf_default_late_001.png
  low_contrast_keyence/   # the "new dataset, different quality" case
    manifest.csv
    ...
```

Minimum manifest columns (small on purpose — no seed/tracking columns; those are segmentation's):

```txt
exemplar_id
concept_label            # e.g. "zebrafish embryo"
prompt_role              # positive / negative
prompt_type              # v1: box
reference_image_path
bbox_x_min_px
bbox_y_min_px
bbox_x_max_px
bbox_y_max_px
notes                    # optional
```

Example rows:

```csv
exemplar_id,concept_label,prompt_role,prompt_type,reference_image_path,bbox_x_min_px,bbox_y_min_px,bbox_x_max_px,bbox_y_max_px,notes
zf_default_early_001,zebrafish embryo,positive,box,images/zf_default_early_001.png,120,90,360,260,legacy MorphSeq Playground; early timepoint; multi-embryo well
zf_default_mid_001,zebrafish embryo,positive,box,images/zf_default_mid_001.png,115,88,370,285,current experiment; middle timepoint; multi-embryo well
zf_default_late_001,zebrafish embryo,positive,box,images/zf_default_late_001.png,105,80,390,310,current experiment; later timepoint; multi-embryo well
zf_default_neg_001,zebrafish embryo,negative,box,images/zf_default_neg_001.png,10,10,90,80,bubble/debris/background
```

The v1 manifest deliberately avoids a polymorphic `reference_box_or_mask` field. Masks can be added
later as an optional `reference_mask_path` only if native SAM3 exemplar masks are empirically needed;
the pipeline contract should not grow mask semantics until then.

For automated tests, keep a tiny synthetic fixture under `tests/improvements/...` to exercise parsing
and contract-writing. For production detection, curate the real `zebrafish_default` set from:

- legacy `morphseq_playground` images/masks, especially an early multi-embryo well frame;
- the current experiment, with one middle and one later multi-embryo frame;
- optional negative boxes for debris, bubbles, empty background, or edge artifacts.

### Selection & provenance

The detector config names one set:

```yaml
frame_detection:
  backend: sam3
  exemplar_set: low_contrast_keyence
  concept_label: "zebrafish embryo"
```

The shared `frame_detections` contract already carries a required **`detector_backend`** column
(a data value, e.g. `"groundingdino"` / `"sam3"`, in `frame_detections_contract.py`). Keep that column
name — it is a contract field, not a routing key, and renaming it would churn the contract, validator,
and every backend. The emitted row value stays under `detector_backend`.

- Home for the resolver: `data_pipeline/object_extraction/detection/exemplar_library.py`
  (`resolve_exemplar_set(name) -> validated set`). It defines *what an exemplar means to a detection*
  — pipeline-side, model-neutral.
- An exemplar set is a **versioned input asset**, like a checkpoint: it may live under the model cache
  or a curated data dir, not necessarily in git if large.
- Which set was used is recorded in the **detection provenance sidecar**, never as a column in the
  shared `frame_detections` table. Exemplar/prompt details are backend-specific, exactly like
  GroundingDINO's `box_threshold` ([[detection_world]] "shared table stays model-neutral").

**Rule:** a prelabeled image is a prompt source, not a canonical detection. It never masquerades as a
pipeline output.

---

## 5. Detection integration (FIRST)

SAM3 slots into [[detection_world]] as just another `detector_backend`. The river is unchanged:

```txt
frame_inventory[well]
  -> run_frame_detection  (backend: sam3, exemplar_set: <name>)
  -> frame_detections[well]        # SAM3 is one detector_backend value
  -> kept_frame_detections(...)    # UNCHANGED
  -> (existing seed selection -> SAM2 tracking -> frame_masks)   # UNCHANGED
```

Backend region mirrors the existing detection backend layout
(`backends/{detectron2,groundingdino}/`). SAM3 lands directly in `detection/backends/sam3/`:

```txt
detection/backends/sam3/
  config.py                   # exemplar_set, concept_label, score thresholds
  run_sam3_detection.py       # backend router glue
  filter_sam3_detections.py   # SAM3 concept scores -> is_kept (the seam; parallels
                              #   filter_groundingdino_detections.py)
  adapt_sam3_detections.py    # raw SAM3 output -> shared frame_detections rows (pipeline-owned)
```

`filter_sam3_detections.py` is the seam where SAM3 config becomes the shared `is_kept` outcome; the
shared runner/validator require `is_kept` and do not know how SAM3 decided it. The shared
`frame_detections` contract, its validator, and `kept_frame_detections()` are untouched. SAM3-only
fields (concept scores, exemplar set, raw phrases) go in optional audit columns / the provenance
sidecar, not the shared schema.

The detection CLI flow inside the bundle (native SAM3 request API shape — audited 2026-07-01):

```python
config = resolve_sam3_config(...)                       # config.py (torch-free)
exemplars = resolve_exemplar_set(config.exemplar_set)   # pipeline-side
predictor = load_sam3_detector(config)                  # loader.py (heavy): build_sam3_predictor(...)

# Exact request payloads are implementation-owned, but the boundary is this shape:
session = predictor.handle_request({
    "type": "start_session",
    "frames": frame_or_tiny_video,
})
response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session["session_id"],
    "concept_label": config.concept_label,
    "exemplars": exemplars,  # positive/negative boxes or masks
})
results = normalize_native_sam3_response(response)       # boxes + scores, masks ignored here

adapt_sam3_detections(results, exemplars, output_csv)   # pipeline-owned adapter -> frame_detections
```

Note SAM3 returns masks *and* boxes for the detection stage; MorphSeq takes the **boxes+scores** into
`frame_detections` and discards masks here (masks are a seg-track concern, §6). The exemplar
positive/negative boxes are exactly what §4's manifest `prompt_role` encodes.

---

## 6. Seg-track integration (LATER — reserved)

A *separate* integration, deferred until detection is proven. Same model bundle, second CLI
(`cli_track.py`), second adapter (`adapt_sam3_output.py` → `frame_masks`). Parallels SAM2 and reuses
the existing `prompt_detection_id → sam2_object_id-style → track_id` provenance chain from
[[segmentation_world]].

**Bidirectional tracking is a seg-track concern, not detection.** SAM3's video API exposes
`reverse: bool` and `start_frame_idx`, the same shape SAM2 already uses
(`propagate_in_video(start_frame_idx=seed_idx, reverse=True/False)`). Its smoke belongs to *this*
phase, not the detection phase. Kept thin on purpose — do not over-design the tracker before the
detection path lands.

---

## 7. Install & setup: one command, three idempotent steps

The biologist runs one command; the maintainer gets three handholds. `setup-sam3` is a **thin,
self-reporting wrapper** over three **independently re-runnable, idempotent** steps. Idempotency is
what makes "one command" and "fixable when it breaks" both true instead of in tension.

```txt
pixi run setup-sam3         # convenience wrapper; prints the step it is on

  = pixi run install-backend sam3            # step 1: runtime/code
      env resolves (torch/torchvision + native SAM3 deps pinned into the stable sam3 env)
      + gitignored upstream SAM3 checkout exists at the pinned commit
      + MorphSeq integration code present
      + record package source/pin/vendor path

    pixi run install-model  sam3 --variant <v>   # step 2: weights/artifacts
      prefetch the gated facebook/sam3.1 checkpoint into $MORPHSEQ_MODEL_CACHE/sam3/hf/
      (steer HF_HUB_CACHE; prefer HF_TOKEN for auth)
      + record hf_model_id / hf_revision / checkpoint_path / checkpoint_sha256 / hf_access_ok

    pixi run verify-backend sam3               # step 3: truth/report (§8)
```

Signpost-on-failure — the wrapper names the failing sub-step and the one command that retries it:

```txt
$ pixi run setup-sam3
[1/3] install-backend sam3 ...... ok
[2/3] install-model   sam3 ...... FAILED: HF download timed out
      -> fix: check `huggingface-cli whoami`, then re-run just this step:
         pixi run install-model sam3 --variant large
```

Each sub-step checks its own preconditions and points at the step that fills them. Re-running a step
whose work is already done must be safe and skip the completed work (idempotent). Weights are never
re-downloaded to fix a code edit; the env is never re-resolved to fix a missing weight.

---

## 8. `verify-backend sam3`: "loads *into the pipeline*"

The check that matters is not "does the model load in a vacuum" — it is **"can it be loaded into the
pipeline and write a canonical output."** Because we ship the *way to install* (not the artifacts),
verify assumes nothing is present and gates on it:

```txt
0. presence gate
   native SAM3 checkout present at pinned commit? -> if not: "run install-backend sam3 first", stop
   SAM3.1 checkpoint present in steered HF cache? -> if not: "run install-model sam3 first", stop
   torch/SAM3 importable in the sam3 env? -> if not: "run install-backend sam3 first", stop

1. real-model load               # proves ARTIFACTS load. NOT "or a fake."
   loader.load_sam3_detector(config) builds the native predictor on the real checkpoint
   CUDA canonical path: version="sam3.1", device="cuda", use_rope_real=True, use_fa3=False

2. exemplar-driven detection smoke
   resolve a tiny exemplar set from disk -> feed it -> get boxes/scores
   (proves SAM3 actually detects on our shape; a detector with no exemplar config detects nothing)

3. adapter guardrail
   import adapt_sam3_detections IN the sam3 env
   assert no forbidden transitive imports (snakemake, pipeline_orchestrator, foreign model stacks,
     detectron2/groundingdino unless this is that backend, notebook/result scripts)

4. contract write
   adapter writes canonical frame_detections -> validate_frame_detections passes

5. production FA3 smoke (optional for correctness, required before enabling FA3 in production)
   hardware gate: CUDA available + compatible NVIDIA GPU (compute capability >= 8.0)
   software gate: flash-attn / native SAM3 FA3 imports succeed
   runtime gate: one tiny SAM3.1 inference runs under fp16 or bf16 autocast, not fp32
   failure sets fa3_available=false and production_use_fa3=false; it does NOT fail
   sam3_detection_ok and does NOT belong in the equivalence harness
```

**Real-vs-fake smoke split (do not "or" them):** a `fake_predictor` smoke is allowed *only* for the
contract-write path (proving the adapter/contract seam); a **real** checkpoint-backed native predictor
load is *required* for model-artifact viability (step 1). They are different smokes with different
report fields.

**No-fake-success invariant:** `sam3_detection_ok` is true only if ALL of `real_model_load_ok`,
`exemplar_prompt_smoke_ok`, `adapter_import_ok`, `forbidden_imports_ok`, and `contract_write_smoke_ok`
are true. A fake predictor can never set `sam3_detection_ok = true` by itself — this keeps the setup
report from becoming ceremonial confetti.

Integrated mode is allowed only if these pass. Failure → narrow the adapter surface or classify SAM3
detached. Backend class is discovered here, not asserted in advance.

---

## 9. Setup report (the audit artifact)

`verify-backend` writes a machine-readable report so Snakemake never guesses whether SAM3 is usable:

```txt
data_pipeline_output/backend_setup/sam3/setup_report.json
```

Fields (re-tuned from the earlier draft — detection-first):

Fields are grouped: shared, then runtime, then artifact, then smoke results. SAM3.1 is a native
checkpoint backend whose artifacts are fetched through Hugging Face, not a transformers snapshot
backend.

```txt
# --- shared ---
backend_name
backend_class                 # integrated / detached / unavailable  (discovered, not asserted)
python_version
torch_version
torchvision_version
cuda_available
cuda_version
gpu_name
compute_capability
bf16_supported
model_cache_root              # the resolved absolute cache root (§2.1)

# --- runtime ---
sam3_runtime_kind             # "native_facebookresearch_sam3"
sam3_vendor_path
sam3_vendor_commit
sam3_version                  # "sam3.1" for canonical CUDA, "sam3" for CPU/dev/equivalence leg
sam3_device                   # cuda / cpu
recommended_precision         # bf16 on compatible CUDA, fp16 fallback if needed, fp32 for CPU/dev
flash_attn_import_ok
fa3_available
fa3_smoke_test                # passed / failed / skipped
production_use_fa3            # true only if the FA3 smoke passed

# --- artifact (backend-specific) ---
artifact_kind                 # for SAM3.1: "hf_raw_checkpoint"
artifact_root                 # $MORPHSEQ_MODEL_CACHE/sam3/
artifact_manifest_path        # metadata/checkpoint_manifest.json
hf_model_id                   # facebook/sam3.1 for canonical CUDA
hf_revision
hf_cache_dir                  # {artifact_root}/hf
checkpoint_path
checkpoint_sha256
hf_access_ok

# --- smokes / guardrail ---
adapter_import_ok             # adapter imported in sam3 env
forbidden_imports_ok          # dependency-boundary guardrail
real_model_load_ok            # step 1 (native predictor + checkpoint load)  -- SPLIT from contract smoke
exemplar_prompt_smoke_ok      # step 2  -- the detector actually detects
contract_write_smoke_ok       # step 4 (fake predictor allowed here)
sam3_detection_ok             # AND of the five above (no-fake-success invariant, §8)
equivalence_status            # not_started / passed / failed; sam3.0 CPU is not canonical until passed
equivalence_report_path
# bidirectional_tracking_ok   -- MOVED to the seg-track phase (§6), not part of detection verify
```

---

## 10. Phasing & gates

**SAM3 Phase 0 — env + native predictor beachhead.** Completed empirically on an L40S on 2026-07-01
and recorded in `../working/sam3_phase0_report.md` plus `pixi.toml`: the stable `sam3` lane is the
non-FA3 correctness path, the native upstream checkout loads SAM3.1 on CUDA, and `version="sam3"`
provides a CPU/dev router through the same native request API. The `sam3-fa3` lane is reserved for
later performance work and is not part of the MVP. The remaining Phase-0 cleanup is to codify those
manual findings in `install_backend.py`, `install_model.py`, `loader.py`, and `verify_backend.py`.

**Phase 1a — equivalence check, separate from FA3.** Run `sam3.0-CPU`, `sam3.0-CUDA`, and
`sam3.1-CUDA` on the same tiny fixture through the native `handle_request` API. Use `use_fa3=False`
for this comparison so precision/performance kernels do not confound model/device differences. Until
this passes, CPU SAM3 remains a dev/smoke path and not a source of canonical outputs.

**Phase 1b — detection integration** (this spec's main body): exemplar library +
`backends/sam3/{run,filter,adapt}` → gate on `verify-backend sam3` green (§8), against the tier-1
smoke fixture where applicable.

**Phase 1c — FA3 production enablement.** After correctness is established, run the FA3 production
smoke under bf16/fp16 autocast. If it passes, set `production_use_fa3=true`; if it fails, keep
`use_fa3=False`. This is a performance gate, not a correctness gate.

**Phase 2 — seg-track integration** (§6, separate/later): `cli_track.py` +
`adapt_sam3_output.py` + bidirectional smoke → gate on `frame_masks` contract-identity vs the SAM2
path.

**Operational precondition (blocks canonical SAM3.1):** HF gated access to `facebook/sam3.1` requested
and approved, with token handling that still works when HF hub cache placement is steered. Confirm
before building production setup.

---

## 11. Non-goals (first pass)

- Mode 3 — fused detect+segment+track in one backend call (breaks the DAG split + provenance chain).
- The fat `detection_prompt_manifest.csv` (its seed/tracking columns belong to segmentation).
- Vendoring the upstream SAM3 repo or its weights into git. A gitignored native upstream checkout is
  expected for SAM3.1 today, but it lives outside git, is installed by `install-backend`, and is pinned
  in the setup report.
- A weights/artifact path under `model_backends/` (silently re-invents vendoring; the cache
  indirection is the point — §2.1).
- Assuming SAM3.1 has a transformers `from_pretrained` path. It does not today; 3.1 uses the native
  package and a raw HF-hosted checkpoint.
- Putting FA3 inside the equivalence harness. FA3 is a production performance option with fp16/bf16
  runtime requirements; correctness/equivalence should use the same non-FA3 path where possible.
- Running an upstream checkpoint downloader **in-place** under a source/vendor checkout (SAM2's README
  defaults weights to a repo-local `checkpoints/`; `install-model sam2` must instead download into
  `$MORPHSEQ_MODEL_CACHE/sam2/checkpoints/` or run the script with CWD redirected there — see §2.1).
- Putting pipeline adapter code in `model_backends/sam3/`; pipeline-side detection code belongs under
  `detection/backends/sam3/`, while heavy runtime code belongs under `model_backends/sam3/`.
- `model_backends/sam3/` owning pipeline contracts/adapters/validators (breaks the stage boundary;
  `frame_detections` is shared with GroundingDINO/Detectron2).
- A monolithic, non-re-runnable `setup-sam3` (idempotent sub-steps are the fixable-when-broken
  property).
- `env.yaml` in the cache-resolution chain (conda-era, gitignored; keeps the resolver impure).
- Repo-relative / CWD-relative cache paths in v1 (absolute + `~` only).
- Exemplars masquerading as canonical detections.
```
