# SAM3 Integration — Target Spec

**Status:** TARGET SPEC (aspirational; refined against decisions in-thread 2026-07-01). Assumes the
pixi backend plan from [[mophseq_env_pixi_orchestrator]] is finished. Audit every path/command
against the real tree before treating as implemented.

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
> (SAM2 = explicit `.pt` files; SAM3 = a Hugging Face snapshot cache). Each backend's
> `install_model.py` owns the ritual for making its upstream artifacts land under that root.

### Vocabulary (reserve "backend" for one thing)

To avoid overloading "backend," this spec reserves **model backend** for `model_backends/<model>/` —
the heavy model-runtime integration that installs, loads, and invokes an upstream model. Pipeline-side
detection implementations are **detection providers** and live under
`src/data_pipeline/object_extraction/detection/providers/<provider>/`. A detection provider owns
provider-specific routing, filtering, and adaptation into the shared `frame_detections` contract, and
must stay torch-free except where explicitly verified. Model loading and upstream imports stay in the
model backend.

> The SAM3 **model backend** powers the SAM3 **detection provider**.

---

## 1. Aspiration & scope (the reframe)

SAM3 enters MorphSeq as **two separate integrations**, not one fused detect+segment+track stack:

```txt
SAM3 detection provider  -> frame_detections   (FIRST; this spec's main body)
SAM3 seg-track provider  -> frame_masks         (LATER; parallels SAM2, reserved §6)
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

---

## 2. Three homes, three lifecycles (the "where does it land" answer)

The vendor ships two separable things — architecture-as-code and parameters-as-weights — and MorphSeq
routes them to **different homes with different lifecycles**. Plus MorphSeq's own integration code.
Three locations, three owners:

| what | where | in git? | installed by | lifecycle |
|------|-------|---------|--------------|-----------|
| MorphSeq SAM3 integration (install scripts, loader, config, CLI) | `model_backends/sam3/` | **yes** (tiny) | authored | changes when *we* edit |
| upstream SAM3 runtime (`transformers` with SAM3 classes) | Pixi-managed `sam3` env site-packages | no (gitignored, like torch) | `install-backend` | re-pip; pinned |
| SAM3 model artifacts (HF snapshot of `facebook/sam3`) | `$MORPHSEQ_MODEL_CACHE/sam3/hf/` | no | `install-model` | fetch once; big; shared |

Keeping code and artifacts in separate homes is *why* iterating is cheap: bump the transformers pin and
re-run `install-backend` **without** re-downloading the snapshot; re-run `install-model` **without**
re-resolving the env. This is the physical basis of the "fixable when it breaks" property (§7).

**SAM2 vs SAM3 load — a decisive asymmetry (audited 2026-07-01).** The two models do NOT load the
same way, and the doctrine must not pretend they do:

- **SAM2** (`facebookresearch/sam2`): `build_sam2_video_predictor(config_file, ckpt_path)`. The config
  is a **Hydra** YAML that travels *inside the installed package* (`compose(config_name=...)` finds it
  on the package search path — no separate config file to manage). The **checkpoint is an explicit
  `ckpt_path` argument** you manage; upstream's README defaults it to a repo-local `checkpoints/` dir
  (the wrinkle — see §2.1).
- **SAM3** (HF `transformers`): `Sam3Model.from_pretrained("facebook/sam3")`. Config + weights are one
  HF snapshot; **HF downloads and caches internally** (`HF_HOME` / `cache_dir` → normally
  `~/.cache/huggingface/hub`). There is **no user-managed `.pt`** and no separate config file. MorphSeq
  *steers* the HF cache into our root; it does not own the file layout.

Consequence: an editable/vendored upstream checkout is **not** the normal path for either. SAM2's
configs travel with the pip-installed package; SAM3 needs only `transformers` + `from_pretrained`. A
gitignored `vendor/<model>/` checkout is a debug-only fallback, recorded in the setup report if used,
never the expected plan (§11).

**Naming:** the backend home is `model_backends/` (renamed from the pixi spec's `external_backends/`).
`loader.py` is the only module that touches all three homes; its *shape differs per backend* (§3):
SAM2's loader passes an explicit `ckpt_path`, SAM3's loader steers `cache_dir`.

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

SAM3 (Hugging Face snapshot cache — MorphSeq steers HF into `hf/`):

```txt
$MORPHSEQ_MODEL_CACHE/sam3/
  hf/                              # HF cache root controlled by MorphSeq (HF_HOME / cache_dir)
    hub/models--facebook--sam3/
      snapshots/ refs/ blobs/
  metadata/
    hf_snapshot_manifest.json
```

`install-model sam3` prefetches the gated `facebook/sam3` snapshot into `hf/` by pointing
`HF_HOME`/`cache_dir` at it (or `snapshot_download`). The SAM3 loader calls `from_pretrained(...)` with
that `cache_dir`. **No SAM3 `.pt` path is a MorphSeq contract.**

Rules:

- The resolved cache root must be **absolute** after `~` expansion. `~` is allowed and expanded.
- **Relative paths are rejected in v1.** No repo-root discovery, no `.git` walking, no CWD (CWD is a
  trickster goblin). A relative `--model-cache` fails clearly: *"Relative model cache paths are not
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
- **Not** duplicated into each backend — 30 lines today becomes four path-goblins in different hats
  once metadata/manifest/HF conventions arrive.

---

## 3. Loader vs adapter doctrine (don't mix the bridges)

```txt
loader:   installed SAM3 package + checkpoint  ->  ready model/predictor object
adapter:  raw SAM3 output                       ->  canonical frame_detections / frame_masks
```

They cross different rivers. Keep them in different homes. **model backend** (runtime) vs **detection
provider** (pipeline-side) — never overload "backend":

```txt
model_backends/sam3/                         # MODEL BACKEND. HEAVY. GPU-only import.
  install_backend.py    # the WAY to install code: pins transformers (+ SAM3 deps) into the env
  install_model.py      # the WAY to prefetch the HF snapshot into the cache
  verify_backend.py     # the CHECK (§8)
  backend.toml          # backend metadata (source, pin, variants)
  src/morphseq_sam3/
    config.py           # torch-free: variant, calls resolve_backend_cache_root("sam3")
    loader.py           # heavy: from_pretrained(cache_dir=...) -> model + processor
    cli_detect.py       # the detection CLI Snakemake shells out to
    cli_track.py        # LATER: the seg-track CLI (§6)
    smoke.py

src/data_pipeline/object_extraction/detection/     # PIPELINE. contracts + providers stay here.
  frame_detections_contract.py    # EXISTS, unchanged
  validate_frame_detections.py    # EXISTS, unchanged
  kept_frame_detections.py        # EXISTS, unchanged
  exemplar_library.py             # NEW (§4)
  providers/                      # target dir (de-overload "backend"; §0 vocabulary)
    groundingdino/                #   existing dir; rename backends/->providers/ is a SEPARATE migration
    detectron2/                   #   existing dir; not a prerequisite for SAM3 (see §5 note)
    sam3/                         # NEW detection provider (lands here directly)
      config.py                   #   is_kept thresholds, exemplar_set selection
      run_sam3_detection.py       #   provider router glue for the stage
      filter_sam3_detections.py   #   SAM3 scores -> is_kept  (the groundingdino-parallel seam)
      adapt_sam3_detections.py    #   PIPELINE-OWNED: raw -> frame_detections rows
```

`detection/providers/sam3/` is the **pipeline-side detection provider** (adapter/router/filter), not
the SAM3 model runtime. It must stay torch-free except where explicitly verified; model loading and
upstream imports live only in `model_backends/sam3/`.

Backend-specific adapters live under `detection/providers/<provider>/` when they are pipeline-owned
and contract-facing. They are **not** part of `model_backends/<backend>/`. This preserves the stage
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

# SAM3 loader — steer HF cache_dir; no explicit .pt, no separate config file
sam3_cache = resolve_backend_cache_root("sam3")
model      = Sam3Model.from_pretrained("facebook/sam3", cache_dir=str(sam3_cache / "hf"))
processor  = Sam3Processor.from_pretrained("facebook/sam3", cache_dir=str(sam3_cache / "hf"))
```

(Exact SAM3 class names — `Sam3Model` / `Sam3VideoModel` / `Sam3Tracker` / `Sam3Processor` — audit
against the installed transformers version; the architectural distinction is what's load-bearing:
**SAM2 passes explicit files; SAM3 steers HF cache.**)

The detection CLI imports **both** the loader and the pipeline adapter:

```txt
morphseq_sam3.loader                                                       (model-backend-side)
data_pipeline.object_extraction.detection.providers.sam3.adapt_sam3_detections   (pipeline-side)
```

This is allowed under the integrated-backend policy — **but proven at install time, not asserted.**
`verify-backend sam3` imports `adapt_sam3_detections` in the sam3 env and asserts no forbidden
transitive imports (§8). This matters because the adapter pulls in `frame_detections_contract` +
`exemplar_library` + `identifiers`, and any of those could drag in something heavy; the guardrail is
what catches it.

---

## 4. The exemplar library (the one genuinely new validated input asset)

Exemplars are **prompt sources, not detections.** A curated, on-disk, versioned set of example
images with boxes/masks that tells the SAM3 concept detector what an embryo looks like *for this
dataset's visual quality*. Decoupled from seed selection (that stays SAM2's job — see §1).

The ergonomic (the whole reason this exists): a new dataset with a different look gets a new exemplar
set with a couple of examples. Point the detector at it via config. Adapt to new data quality without
touching prompt strings, retraining, or any other stage.

### Layout

```txt
exemplar_library/
  zebrafish_default/
    manifest.csv          # rows below
    img_001.jpg
    img_001.json          # box(es)/mask ref for img_001
    ...
  low_contrast_keyence/   # the "new dataset, different quality" case
    manifest.csv
    ...
```

Minimum manifest columns (small on purpose — no seed/tracking columns; those are segmentation's):

```txt
exemplar_id
concept_label            # e.g. "zebrafish embryo"
prompt_role              # positive / negative
prompt_type              # box / mask / exemplar_image
reference_image_path
reference_box_or_mask    # json/path; box xyxy or mask ref
```

### Selection & provenance

The detector config names one set:

```yaml
frame_detection:
  provider: sam3
  exemplar_set: low_contrast_keyence
  concept_label: "zebrafish embryo"
```

The shared `frame_detections` contract already carries a required **`detector_backend`** column
(a data value, e.g. `"groundingdino"` / `"sam3"`, in `frame_detections_contract.py`). Keep that column
name — it is a contract field, not a routing key, and renaming it would churn the contract, validator,
and every provider. Prose uses "provider" to disambiguate from `model_backends/`; the emitted row value
stays under `detector_backend`.

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

Provider region mirrors the existing detection backend layout (currently
`backends/{detectron2,groundingdino}/`). SAM3 lands directly in `detection/providers/sam3/`; renaming
the existing `backends/` dir to `providers/` is a **separate, non-blocking migration** (it would churn
GroundingDINO/Detectron2 importers), not a prerequisite for SAM3:

```txt
detection/providers/sam3/
  config.py                   # exemplar_set, concept_label, score thresholds
  run_sam3_detection.py       # provider router glue
  filter_sam3_detections.py   # SAM3 concept scores -> is_kept (the seam; parallels
                              #   filter_groundingdino_detections.py)
  adapt_sam3_detections.py    # raw SAM3 output -> shared frame_detections rows (pipeline-owned)
```

`filter_sam3_detections.py` is the seam where SAM3 config becomes the shared `is_kept` outcome; the
shared runner/validator require `is_kept` and do not know how SAM3 decided it. The shared
`frame_detections` contract, its validator, and `kept_frame_detections()` are untouched. SAM3-only
fields (concept scores, exemplar set, raw phrases) go in optional audit columns / the provenance
sidecar, not the shared schema.

The detection CLI flow inside the bundle (real SAM3 HF API shape — audited 2026-07-01):

```python
config = resolve_sam3_config(...)                       # config.py (torch-free)
exemplars = resolve_exemplar_set(config.exemplar_set)   # pipeline-side
model, processor = load_sam3_detector(config)           # loader.py (heavy): from_pretrained(cache_dir=)

# text concept + exemplar boxes drive the concept detector:
inputs = processor(
    images=frame, text=config.concept_label,
    input_boxes=exemplar_boxes, input_boxes_labels=exemplar_labels,  # pos/neg exemplars
    return_tensors="pt",
).to(model.device)
outputs = model(**inputs)
results = processor.post_process_instance_segmentation(
    outputs, target_sizes=inputs["original_sizes"].tolist(),
)[0]                                                     # masks + boxes + scores

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

  = pixi run install-backend sam3            # step 1: the CODE
      env resolves (transformers + SAM3 deps pinned into the sam3 env)
      + MorphSeq integration code present
      + record package source/pin

    pixi run install-model  sam3 --variant <v>   # step 2: the ARTIFACTS
      prefetch the gated facebook/sam3 HF snapshot into $MORPHSEQ_MODEL_CACHE/sam3/hf/
      (point HF_HOME/cache_dir at it, or snapshot_download)
      + record hf_model_id / hf_revision / hf_snapshot_dir / hf_snapshot_commit / hf_access_ok

    pixi run verify-backend sam3               # step 3: the CHECK (§8)
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
   HF snapshot present in cache?  -> if not: "run install-model sam3 first", stop
   transformers + SAM3 importable? -> if not: "run install-backend sam3 first", stop

1. real-model load               # proves ARTIFACTS load. NOT "or a fake."
   loader.load_sam3_detector(config) runs from_pretrained(cache_dir=...) on the real snapshot

2. exemplar-driven detection smoke
   resolve a tiny exemplar set from disk -> feed it -> get boxes/scores
   (proves SAM3 actually detects on our shape; a detector with no exemplar config detects nothing)

3. adapter guardrail
   import adapt_sam3_detections IN the sam3 env
   assert no forbidden transitive imports (snakemake, pipeline_orchestrator, foreign model stacks,
     detectron2/groundingdino unless this is that backend, notebook/result scripts)

4. contract write
   adapter writes canonical frame_detections -> validate_frame_detections passes
```

**Real-vs-fake smoke split (do not "or" them):** a `fake_predictor` smoke is allowed *only* for the
contract-write path (proving the adapter/contract seam); a **real** snapshot load is *required* for
model-artifact viability (step 1). They are different smokes with different report fields.

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

Fields (retuned from the earlier draft — detection-first):

Fields are grouped: shared, then artifact (backend-specific — SAM3 is an HF-snapshot backend, not a
`.pt` backend), then the smoke results.

```txt
# --- shared ---
backend_name
backend_class                 # integrated / detached / unavailable  (discovered, not asserted)
python_version
torch_version
transformers_version
cuda_available
cuda_version
gpu_name
model_cache_root              # the resolved absolute cache root (§2.1)

# --- artifact (backend-specific) ---
artifact_kind                 # for SAM3: "hf_snapshot"  (SAM2 would be "explicit_checkpoint")
artifact_root                 # $MORPHSEQ_MODEL_CACHE/sam3/
artifact_manifest_path        # metadata/hf_snapshot_manifest.json
hf_model_id                   # facebook/sam3
hf_revision
hf_cache_dir                  # {artifact_root}/hf
hf_snapshot_dir
hf_snapshot_commit
hf_access_ok

# --- smokes / guardrail ---
adapter_import_ok             # adapter imported in sam3 env
forbidden_imports_ok          # dependency-boundary guardrail
real_model_load_ok            # step 1 (snapshot load)  -- SPLIT from the contract smoke
exemplar_prompt_smoke_ok      # step 2  -- the detector actually detects
contract_write_smoke_ok       # step 4 (fake predictor allowed here)
sam3_detection_ok             # AND of the five above (no-fake-success invariant, §8)
# bidirectional_tracking_ok   -- MOVED to the seg-track phase (§6), not part of detection verify
```

---

## 10. Phasing & gates

**SAM3 Phase 0 — does the env resolve + a real snapshot load on a GPU node?** This is the gating
unknown (the CUDA ≥12.6 / torch ≥2.7 / py ≥3.12 floor can only be proven on a GPU compute node via
SGE, not a login node). Mirror the pixi Phase-0 beachhead logic: write an SGE probe that runs
`install-backend` + a real `from_pretrained` load. If it fails, STOP — SAM3 is detached-only or
blocked; reframe before writing adapter code. (Cluster GPU nodes are believed to support 12.6+; still
prove it.)

**Phase 1 — detection integration** (this spec's main body): exemplar library +
`providers/sam3/{run,filter,adapt}` → gate on `verify-backend sam3` green (§8), against the tier-1
smoke fixture where applicable.

**Phase 2 — seg-track integration** (§6, separate/later): `cli_track.py` +
`adapt_sam3_output.py` + bidirectional smoke → gate on `frame_masks` contract-identity vs the SAM2
path.

**Operational precondition (blocks everything):** HF gated access to `facebook/sam3` requested and
approved. Confirm before building.

---

## 11. Non-goals (first pass)

- Mode 3 — fused detect+segment+track in one backend call (breaks the DAG split + provenance chain).
- The fat `detection_prompt_manifest.csv` (its seed/tracking columns belong to segmentation).
- Vendoring the upstream SAM3 repo or its weights into git ("that's what gets ignored"; ship the
  *install method*, pin the runtime). A gitignored `vendor/<model>/` checkout is debug-only, recorded
  in the setup report if used — never the expected plan.
- A weights/artifact path under `model_backends/` (silently re-invents vendoring; the cache
  indirection is the point — §2.1).
- Assuming SAM3 has a user-managed `.pt` checkpoint (it doesn't — HF `from_pretrained` owns the
  snapshot; a `checkpoint_path` contract is SAM2-only).
- Running an upstream checkpoint downloader **in-place** under a source/vendor checkout (SAM2's README
  defaults weights to a repo-local `checkpoints/`; `install-model sam2` must instead download into
  `$MORPHSEQ_MODEL_CACHE/sam2/checkpoints/` or run the script with CWD redirected there — see §2.1).
- Overloading "backend" — pipeline-side detection code is a **provider** under `detection/providers/`;
  "model backend" is reserved for `model_backends/` (§0 vocabulary).
- `model_backends/sam3/` owning pipeline contracts/adapters/validators (breaks the stage boundary;
  `frame_detections` is shared with GroundingDINO/Detectron2).
- A monolithic, non-re-runnable `setup-sam3` (idempotent sub-steps are the fixable-when-broken
  property).
- `env.yaml` in the cache-resolution chain (conda-era, gitignored; keeps the resolver impure).
- Repo-relative / CWD-relative cache paths in v1 (absolute + `~` only).
- Exemplars masquerading as canonical detections.
```
