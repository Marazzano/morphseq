# MorphSeq Pixi Environment & Backend Split — Target Spec

**Status:** TARGET SPEC (locked-in core; still being fleshed out).
**Purpose:** Define the pixi environment model and the backend/pipeline split for MorphSeq publication.
**Audit:** Every pixi command and layout below is a *target*. Audit against the real tree before treating as implemented.

Related specs: [[source_tree_stage_alignment]] owns the `src/data_pipeline/` stage reorg. This spec owns
environments and the model-backend boundary only.

---

## 0. Tattoo

> **Model outputs are raw facts. Pipeline-owned adapters turn facts into contracts.**
> **Adapters may run in-process or across a file seam; contracts stay pipeline-owned.**

Backends execute models. Adapters translate raw model output into pipeline contracts.
Contracts stay pipeline-owned. Nothing gets siphoned into a fake shared registry. Raw sidecars
exist only when an environment boundary forces them, not for architectural purity.

---

## 1. Pixi environment profiles

Pixi is the official front door. Environment profiles are installed independently:

| profile      | contents                                    | torch? | status   |
|--------------|---------------------------------------------|--------|----------|
| `analysis`   | lightweight read/plot; may import bounded leaves like `data_pipeline.object_extraction.segmentation.masks.*` | no | active |
| `pipeline`   | `data_pipeline` + Snakemake + adapters      | no     | active   |
| `sam2`       | SAM2 backend, own python pin                | yes    | active   |
| `unet`       | UNet backend, own python pin                | yes    | active   |
| `legacy-vae` | 3.9-pinned VAE embeddings backend           | yes    | active   |
| `sam3`       | SAM3 backend, own python pin                | yes    | reserved |

Design target:
- Pixi installs environments.
- Pixi tasks install/verify backend model artifacts.
- Snakemake runs the DAG in the `pipeline` env.

**Feature naming — engine-tagged orchestration.** The orchestration feature is `pipeline_sm`
(snakemake), *not* a bare `orchestration` or `workflow`. Rationale: the name must answer both *of
what* (the pipeline) and *which engine* (`_sm` = snakemake), because the engine is a swappable
implementation of a fixed role. A future Nextflow port is a sibling feature `pipeline_nf` composed
into a parallel `pipeline` env; the `_sm`/`_nf` suffix is the disambiguator. Everything else in the
DAG-driving env (contracts, image-io, plotting) is engine-agnostic and shared across both.
- Backends run inference in their own env. Integrated backends exchange input manifests and canonical
  outputs; detached backends additionally exchange raw sidecars/parquet through a file seam.
- The pipeline env never imports a heavyweight model stack.

Failure of an optional backend (`sam2`/`sam3`/`legacy-vae`) must not imply the `analysis` or
`pipeline` profiles are broken.

**Phase 0 finding (2026-07-01) — materialization is a THIRD compute zone, not part of `pipeline`.**
The `pipeline` env's "no torch" invariant was audited by actually resolving it and running the
tier-1 smoke DAG. `tasks.py` (imported once, module-wide, by every Snakemake rule) turned out to
import `materialize_stitched_images.py` at module level, which imports `torch` at module level,
which broke the torch-free resolve for the whole task dispatcher — not because of an incidental
import, but because `LoG_focus_stacker` (`acquisition/image_building/shared/log_focus.py`) runs a
**deliberate GPU conv2d** over each Z-stack. This isn't an accident to clean up; focus-stacking is
legitimately GPU compute and stays that way.

Naming note: this zone is **materialization**, not the broader **acquisition** stage. `acquisition`
also covers metadata ingest, scope resolution, well discovery, and position mapping — all confirmed
torch-free under Phase 0 (`validate-frame-inventory` and the discovery/config tasks run clean under
pixi). Only the narrow slice that reads ND2 -> stitches -> focus-stacks -> writes the built image
product (`data_pipeline.acquisition.image_materialization.*`, already its own subpackage in the
source tree) is torch-coupled. Calling the whole zone "acquisition" would wrongly imply the entire
stage needs torch; "materialization" names the actual coupled slice.

Working backward from the actual goals of "torch-free pipeline env" (fast/portable resolve,
CI/laptop-runnable orchestration, a clean backend-import boundary) — none of those goals require
*materialization* specifically to be torch-free. They require the **contract/orchestration layer**
(Snakemake, `tasks.py` dispatch, `*_contract.py` schemas, validators, identifiers) to be torch-free.
Materialization (ND2 read, stitching, `LoG_focus_stacker`) was swept into "pipeline" only because
that's where it lives in the source tree today, not because it belongs there conceptually —
structurally it is the same shape as a backend (GPU compute producing a canonical artifact), just
not "a model" in the ML sense.

**Resolution:** `pipeline` stays contracts+orchestration only (torch-free, proven — see Phase 0 in
§8). Materialization is a fourth zone that needs its own pixi env eventually (same family as
`sam2`/`unet`: GPU compute, own env, contract-writing adapter at the seam) — NOT solved in Phase 0.
The immediate unblock was function-local, not architectural: `cmd_materialize_stitched` in
`tasks.py` now imports `materialize_stitched_images` inside the function body (matching the
existing UNet/SAM2 pattern at `tasks.py` ~581/~856) so `tasks.py` itself stays importable in the
torch-free env — the command still requires torch to actually run.

| profile          | contents                                              | torch? | status    |
|-------------------|-------------------------------------------------------|--------|-----------|
| `materialization` | ND2 read, stitching, `LoG_focus_stacker` (GPU conv2d)  | yes    | **active** — wired + proven live on A100 GPU node (2026-07-01): full `write_resolved_product_plan_for_well` -> `materialize_image_product_for_well` -> `validate_frame_inventory_for_well` chain, 5/5 steps green via `MATERIALIZATION_RUN` / `pixi run -e materialization` |

---

## 2. Backend integration doctrine (two-tier)

Backends are model-execution environments. They may be **integrated** or **detached**.

- **Integrated backend** — runs inference and calls the exact pipeline-owned adapter path **in the
  same process, inside the backend env**. It may import the *verified-bounded adapter surface*
  (defined below), adapts in memory, and writes canonical outputs directly. No sidecar.
- **Detached backend** — writes raw sidecars only; a pipeline-env adapter reads them and writes
  canonical outputs. Used when the environment boundary forces it.

> **Do not introduce raw sidecars unless the environment boundary requires them.**

The governing phrase: **MorphSeq is the published unit of reproducibility, not the backend as a
standalone library (not "SAM2-as-a-service").** Pixi reproducibility does not require backend import
purity; it requires:
1. each environment resolves reproducibly,
2. each CLI runs in the intended pixi env,
3. model artifacts are installed/verified,
4. canonical outputs validate against MorphSeq contracts.

**Backend classification:**

| backend    | class     | seam                                    |
|------------|-----------|-----------------------------------------|
| SAM2       | integrated | in-process adapter, no sidecar          |
| UNet       | integrated | in-process adapter, no sidecar          |
| legacy-VAE | detached  | existing parquet/env seam (3.9 pickles) |
| SAM3       | undecided | discovered at install (see guardrail)   |

**Verified-bounded adapter surface** — the term is deliberate: NOT "pure" and NOT dependency-free.
The requirement is bounded, expected, non-orchestration, non-model-stack dependencies. Allowed
dependency classes:
- stdlib
- numpy / pandas
- opencv / `cv2` where mask resizing requires it
- skimage / PIL / imageio where adapter-side image I/O requires it
- local contract, identifier, geometry, mask, and snip-frame utilities

Forbidden dependency classes:
- Snakemake / rule orchestration
- `data_pipeline.pipeline_orchestrator` and other cross-stage task orchestration
- torch model stacks not belonging to the active backend
- detectron2 / groundingdino unless verifying that detection backend specifically
- training libraries, notebook code, result scripts, or unrelated model stacks

An integrated backend env may therefore have `data_pipeline` importable when doing so preserves the
in-memory adapter path and avoids unnecessary intermediates, but the imported surface must stay
bounded and contract-facing.

**Verified-bounded surface gate:** importing the exact adapter module in the backend env must NOT
transitively import forbidden modules. Failure -> either narrow the adapter imports, or classify the
backend as detached.

**Template — the `feature_extraction/legacy_embeddings` pattern:** a backend is a **self-contained
subpackage** that imports nothing outside its own folder except third-party + the *verified-bounded
adapter surface*. legacy-VAE goes further (co-located local contract + a *standalone inference loader
that re-declares the thin model slice*, importing no heavy training library — "safe under 3.9 or
3.10"). SAM2 shares the pipeline-wide `frame_masks` contract instead of co-locating one, allowed
*because that surface is verified-bounded*.

**Guardrail — discover the class at INSTALL time, don't assert it in advance.** After
`pixi install -e <backend>`, the verify task imports the **exact adapter module the backend CLI will
use in production** inside the backend's own resolved env. If the adapter is pipeline-owned, import
the pipeline adapter path, not a backend-package copy:

    pixi run -e sam2 python -c "import data_pipeline.object_extraction.segmentation.adapt_sam2_output"

This is one part of a runtime resolution check, stronger than a static import-pattern test: it proves
the env resolved and the adapter's cross-boundary imports work *in that env*. Import success alone
does not prove integrated mode. `verify-backend` must also inspect the dependency boundary and run a
tiny runtime/output smoke (see §3). If the full check passes -> integrated seam is proven. If it
fails -> either narrow the adapter surface, or classify the backend as detached.

---

## 3. Model Backend Policy

A backend is usable only if the model can run where Snakemake calls it. For each backend,
`verify-backend` must prove these checks inside the backend pixi environment:

1. **Environment resolution** — the pixi environment resolves and installs.
2. **Adapter viability** — the backend env can import the exact adapter module used by the backend CLI.
3. **Dependency boundary** — after adapter import, forbidden transitive imports are absent.
4. **Model artifact viability** — required checkpoints are present, readable, and their
   version/variant metadata match expectations.
5. **Runtime viability** — a tiny CLI smoke can load the model or fake predictor, run inference,
   write output, and validate the canonical contract.

Integrated mode is allowed only if these checks pass. If they fail, the backend must either narrow
its adapter surface or move to detached mode.

For SAM2, prefer a smoke through `fake_predictor.py` when possible, plus a real checkpoint load test
for model artifact viability. For UNet, a CPU smoke is sufficient. The policy is simple: a backend is
not installed merely because its dependencies resolved; it is installed only when the model, runtime,
and contract-writing path all work in the environment where Snakemake will call it.

---

## 4. Three-zone layout

Canonical source paths in this section follow [[source_tree_stage_alignment]]. If that spec changes
the stage path, this spec inherits the new path rather than defining a competing home.

```txt
external_backends/            # HEAVY envs. own pyproject.toml + python pin each.
  sam2_backend/               #   integrated: inference here; imports the verified-bounded surface
  unet_backend/               #   integrated
  legacy_vae_backend/         #   detached: standalone loader, existing parquet seam
  sam3_backend/               # reserved; class discovered at install

src/data_pipeline/            # PIPELINE env. no torch.
  object_extraction/
    segmentation/
      adapt_sam2_output.py      # pipeline-owned adapter; integrated backend imports it in-process
      adapt_unet_output.py      # raw model output -> canonical aux masks (in-process)
      frame_masks_contract.py   # STAYS. contract lives at home.
      masks/                    # STAYS. bounded mask utils; analysis imports directly (see below).
        mask_rle.py             #   numpy-only
        mask_geometry.py        #   numpy-only
        mask_resize.py          #   numpy + cv2
    snip_processing/
      snip_auxiliary_masks_contract.py
  shared/identifiers/         # STAYS. adapters mint ids here.
```

Mask utils — decision (audited 2026-07-01): **do NOT promote to a top-level package.** Analysis
does mask analysis by importing `data_pipeline.object_extraction.segmentation.masks.*` directly.
- Rationale: import footprint is `numpy` + `cv2` only — no torch/snakemake/pipeline internals.
  Package initializers in this path must stay cheap, so importing the leaf triggers no model stack.
  Measured in-env: HEAVY deps pulled in = NONE.
- Since it is not heavy on import, a standalone package buys nothing but rename churn (18 importers,
  and a second rename when a real `morphseq` analysis package eventually lands). Leave it put.
- Requirement on the `analysis` profile: numpy (given) + opencv (`cv2`, needed only by `mask_resize`;
  rle/geometry are numpy-only).
- Integrated backends MAY import these utils — they are part of the verified-bounded adapter surface (see §2).
- Detection utils are NOT exported to analysis: `data_pipeline.object_extraction.detection` is torch-coupled at module
  level (imports detectron2/groundingdino backends) and has no analysis-side demand. Detection
  follows the backend split — inference to `external_backends/`, contract + adapter stay pipeline-side.

Adapter placement rule: adapters live in the **stage that produces their contract**, next to that
contract and the rule that calls them (`object_extraction/segmentation/adapt_sam2_output.py`,
`adapt_unet_output.py`;
VAE latents adapter under `feature_extraction/`). Group by stage, never globally — there is **no
`data_pipeline/adapters/`** (a global folder cross-cuts the stage river and re-creates the
"duplicate homes" drift). At ≤2 adapters per stage, keep them as flat `adapt_*_output.py` files (the
prefix already groups them in a listing). Only at ~3+ in one stage, promote to a **per-stage**
`<stage>/adapters/` subfolder.

Dependency arrows (one-directional; the pipeline never imports a backend's model code):

```txt
analysis                     --> data_pipeline.object_extraction.segmentation.masks
data_pipeline                --> data_pipeline.object_extraction.segmentation.masks

integrated backend (sam2/unet) --> verified-bounded adapter surface
detached backend (legacy-vae)  --> writes parquet/sidecar; pipeline adapter reads it

FORBIDDEN for ANY backend --> data_pipeline orchestration / snakemake / cross-stage code
  (guardrail = install-time import of the exact production adapter module in the backend env, §2/§3)
```

---

## 5. Runtime seam (per class)

**Integrated (SAM2, UNet)** — one process in the backend env; no sidecar:

```txt
[pipeline env] build input manifest (frame_inventory / prompts / snip_inventory)
      | subprocess: pixi run -e <backend> morphseq-run-<backend> ...
[backend env]  run inference -> call pipeline-owned adapter IN-PROCESS
               (adapter imports the verified-bounded surface: contracts + masks + identifiers + bounded I/O)
               -> write canonical contract file directly
[pipeline env] validate against contract (existing validate_* rule, unchanged)
```

This matches today's behavior — inference + in-memory adaptation + direct canonical write — with the
model code relocated to `external_backends/` and the adapter still pipeline-owned. No new artifact,
no new rule edge.

**Detached (legacy-VAE; future SAM3 iff install guardrail fails)** — the existing seam:

```txt
[pipeline env] build manifest
      | subprocess: MODEL_RUN interpreter (own env)
[backend env]  standalone loader -> encode -> write parquet/sidecar (no cross-import)
      | (files cross the boundary)
[pipeline env] adapter reads parquet/sidecar -> validate -> canonical
```

Reserve the detached seam for real env incompatibility (3.9 pickles today). Do not pay its I/O +
extra-rule cost where an integrated in-process adapter works.

---

## 6. Pixi task targets (audit before implementing)

```txt
pixi install -e analysis   && pixi run demo-analysis
pixi install -e pipeline   && pixi run demo-pipeline

pixi run install-backend sam2         # install env + wrapper + prove Snakemake-call viability
pixi run install-model   sam2 --variant <v>   # download/verify checkpoint
pixi run verify-backend  sam2         # import + dependency-boundary check + tiny output smoke
```

- Backend setup separates three concerns: (1) code deps, (2) checkpoint availability, (3) runtime viability.
- **`verify-backend` is the class guardrail (§2/§3):** it imports the exact production adapter
  module IN the backend env
  (`pixi run -e sam2 python -c "import data_pipeline.object_extraction.segmentation.adapt_sam2_output"`),
  inspects forbidden transitive imports, then runs a tiny backend CLI/output smoke. Full check OK ->
  integrated seam proven. Failure -> narrow the adapter surface or use the detached seam.
- Model weights live in a cache, never git. Default `~/.cache/morphseq/models/`,
  override `MORPHSEQ_MODEL_CACHE`.
- Do not vendor SAM2/SAM3/VAE model repos into MorphSeq source control.

---

## 7. Non-goals (first pass)

- one env containing every dependency
- raw sidecars for integrated backends where no env boundary forces them ("sidecars for purity's sake")
- contracts relocated into a shared/registry package
- backends importing pipeline *orchestration/snakemake/cross-stage* code (the verified-bounded
  adapter surface IS allowed)
- checkpoints committed to git
- a `morphseq` lightweight analysis package (proposed, not built — deferred)

---

## 8. Migration plan (phased, gated)

**Framing:** this is *backend refinement*, not a rewrite. The seam already exists — `frame_masks.smk`
and `snip_auxiliary_masks.smk` already `shell:` out to a CLI, and heavy imports in `tasks.py` are
already function-local (lines ~581, ~856), and the pipeline already runs one env calling another via
`conda run -n mseq_pipeline_py3` + the `MODEL_RUN` 3.9 VAE interpreter. So most phases are
*renaming conda→pixi* and *re-homing already-shelling-out code*, not new architecture.

**No new DAG shape for SAM2/UNet (integrated).** Per §2 they stay in-process: inference + in-memory
adaptation + direct canonical write, exactly as today. The refactor MOVES the model code to
`external_backends/` and keeps the adapter pipeline-owned (the backend imports it). No sidecar, no
new rule edge. legacy-VAE keeps its EXISTING detached parquet seam (not new). So every phase is moves
+ conda→pixi renaming — no invented DAG layer.

**Gate discipline:** every phase is verified against the existing tier-1 smoke fixture
(`config_smoke_through_line_20250912_B01.yaml`, B01 / 1tp — the proven-green [[project_tier1_through_line_green]]
run). Model steps use contract-validation + row-count/geometry-tolerance equality, NOT byte-`diff`
(GPU is nondeterministic). Each phase is independently revertable.

### Phase 0 — Pixi beachhead. ZERO source moves.
- **Move:** nothing. **Add:** root `pixi.toml` with only the `pipeline` env, mapping the current
  `mseq_pipeline_py3` dependency set.
- **Gate:** `pixi run -e pipeline snakemake … through_line` reproduces the tier-1 green output.
- **Why first:** proves pixi can *resolve* the env on the cluster. This is the gating unknown, not
  the code. If pixi can't reproduce the existing green run, STOP — the pixi premise needs rework
  before any file moves. Revert = delete one file.

### Phase 1 — one backend env, still in place. ZERO source moves.
- **Move:** nothing. **Change:** in `rules/frame_masks.smk` (and later `snip_auxiliary_masks.smk`),
  swap the runner string `conda run -n … python -m …` → `pixi run -e sam2 …`. Add the `sam2` (and
  `legacy-vae`) pixi env.
- **Gate:** frame_masks for B01 is contract-identical to the conda run.
- **Why:** proves pixi hosts a heavy env and the subprocess seam works under pixi, with no files
  moved. Revert = one-line rule string.

### Phase 2 — relocate ONE backend's model code out; keep adapter in-process. (First real moves.)
Do **UNet first** — cleanest single-CLI backend. INTEGRATED: no sidecar, no new rule.
- **MOVE OUT to `external_backends/unet_backend/src/morphseq_unet_backend/`** (heavy env):
  - `object_extraction/segmentation/backends/unet_snip/model_loader.py` (`FishModelSnipPredictor`, torch)
  - `models/unet/*`, `object_extraction/segmentation/unet/inference.py`,
    `object_extraction/segmentation/unet/__init__.py`
  - `entrypoint.py` + `run_unet_snip.py` orchestration — the backend CLI. It IMPORTS the
    pipeline-owned adapter in-process and writes the canonical CSV directly (as today).
- **STAY pipeline-side** (the verified-bounded adapter surface the backend imports):
  - `adapt_unet_output.py` (new home for the row-building + `write_auxiliary_mask_png` +
    `assert_on_snip_frame` + `validate_snip_auxiliary_masks`), and
    `snip_auxiliary_masks_contract.py` — under the owning snip/segmentation contract home.
- **UNet bounded-surface note:** UNet integrated mode imports more than a minimal contract module.
  It includes row construction, snip-frame assertion, homecoming resize, canonical PNG writing, and
  contract validation. This is allowed because those operations are contract-facing and preserve the
  existing in-memory path. The verify task must prove those imports resolve in the UNet env.
- **Repoint:** `tasks.py` (~581, ~610) and `viz/render_snip.py` at the pipeline-side adapter/contract.
- **Gate:** B01 smoke green (contract fingerprint, see `unet_split_map.md` — CPU is fine) **and**
  install guardrail: `pixi run -e unet python -c "import data_pipeline.object_extraction.segmentation.adapt_unet_output"`
  resolves in the `unet` env, followed by the dependency-boundary and tiny-output smoke checks.

### Phase 3 — repeat for SAM2 (integrated), then legacy-VAE (detached).
- **SAM2 → `external_backends/sam2_backend/src/morphseq_sam2_backend/`** — INTEGRATED, no sidecar
  (files scattered across THREE homes today; consolidate on the way out):
  - from `object_extraction/segmentation/sam2_video/`: `run_sam2_video.py`, `model_loader.py`,
    `sam2_frame_view.py`
  - from `models/`: `sam2.py`; from `backends/sam2_video/`: `prompt_detections.py`, `fake_predictor.py`
  - **STAY pipeline-side:** `adapt_sam2_output.py` →
    `data_pipeline/object_extraction/segmentation/` (imports the verified-bounded surface:
    `frame_masks_contract`, `masks/`, `identifiers`). The SAM2 CLI imports it in-process. Repoint
    `tasks.py` (~856–860).
  - **Gate:** B01 frame_masks contract-identical + install guardrail:
    `pixi run -e sam2 python -c "import data_pipeline.object_extraction.segmentation.adapt_sam2_output"`
    resolves in the `sam2` env, followed by the dependency-boundary and tiny-output smoke checks.
- **legacy-VAE → `external_backends/legacy_vae_backend/`:** DETACHED, keeps its existing seam. It
  already runs under `MODEL_RUN` via `-m data_pipeline.feature_extraction.legacy_embeddings.entrypoint`
  and the loader is already a *standalone re-declared model slice* (no `legacy.vae.*` import, "safe
  under 3.9 or 3.10"). Move `src/legacy/vae/*` into the backend package. Mostly a *home change* — the
  detached split already exists and is the TEMPLATE for the two-tier doctrine.

### Phase 4 — `analysis` env + mask leaf verification.
- **Move:** nothing (masks stay at `data_pipeline.object_extraction.segmentation.masks.*` per §4 audit).
- **Add:** `analysis` pixi env (numpy + cv2 + plotting). Confirm `import
  data_pipeline.object_extraction.segmentation.masks` resolves there with no heavy deps. Add the
  per-backend adapter guardrail (§2/§3) to CI as the permanent invariant.
- **Gate:** a representative `results/` mask-analysis snippet runs under `pixi run -e analysis`.

### Detection (deferred, same recipe)
Not on the critical path. When done: `detection/backends/{detectron2,groundingdino}/*` →
`external_backends/*`; `detection/frame_detections_contract.py` + a new `adapt_*_detections.py` stay
pipeline-side. No analysis export (§4).

---

## 9. Open / to flesh out

- exact `pyproject.toml` deps per backend, including model deps plus any verified-bounded
  adapter-surface deps needed by integrated backends
- raw sidecar format for detached backends only, if/when a backend cannot pass the integrated guardrail
- python pins per backend after real dependency audit
- whether `analysis` env installs all of `data_pipeline` or only a minimal subset sufficient for
  `data_pipeline.object_extraction.segmentation.masks.*`
- unify import rooting (`data_pipeline.x` vs `src.analyze.x`) so analysis imports have one clean package path
