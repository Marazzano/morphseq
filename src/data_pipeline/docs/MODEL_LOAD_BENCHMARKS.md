# Model Load-Time vs Per-Well Work-Time Benchmarks

**Purpose.** The pipeline currently runs one Snakemake job per well for each GPU-model
family; each job cold-starts a fresh Python process, loads model weights, does GPU work on
ONE well, and exits. A proposed "model server" (load once per run, per-well clients talk to
it over a socket) is only worth building where **load time is a meaningful fraction of
per-well work time**. This doc measures that fraction for the four GPU-model families in the
pipeline, on real data, on a real GPU node in this cluster.

**Date measured:** 2026-07-25. **Measured by:** direct SGE batch jobs (see "Exact commands"
below) — not simulated, not estimated.

## Environment

- Cluster: UW GS SGE grid. GPU node used: `t005`/`t008`/`t010.grid.gs.washington.edu`
  (trapnell-short.q), each carrying **2x NVIDIA L40S, 46068 MiB (~46GB) each**, driver
  560.35.05 / CUDA 12.6. SGE's `cuda` consumable resource assigns **one exclusive GPU per
  job** — `nvidia-smi` inside each job showed "No running processes found" before our job
  started, i.e. no other tenant sharing the specific device our job got. Confirmed no other
  compute jobs were scheduled on trapnell-short.q's GPU nodes at measurement time
  (`qstat -f -q trapnell-short.q@t005,...` showed 0 used slots beforehand).
- Model weights live on **NFS** (`/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/models`,
  network-mounted, not local disk) — confirmed via `env.yaml.paths.models_root`. This matters:
  the first read of any checkpoint in a benchmark run pays full NFS latency; repeated reads in
  the same run are dominated by whatever page cache the node retained. We report both.
- Conda env: `segmentation_grounded_sam` (GroundingDINO, SAM2, UNet — all torch 2.7.1+cu118,
  GPU-capable). The `legacy_embeddings` (VAE) stage runs under a **separate Python-3.9
  interpreter** (`mseq_pipeline_py3.9`, torch 2.5.1) per `model_input_handoff_contract.md`.
- Real experiment used: **`20260702_hotchem_36hpf_plate02`** (96 wells), confirmed on disk
  with complete artifact chains (frame_inventory → frame_detections → frame_masks → snips →
  snip_auxiliary_masks, and snips → latent_embeddings) for all 96 wells. Wells `A01`, `D07`,
  `F08` used as the representative sample. This plate is uniform in size — every well checked
  has **15 acquisition frames, 1 embryo/snip per well** — so no well-size variation could be
  demonstrated within this plate; this is reported honestly rather than invented.

## Headline ratios

Ratio = `load / (load + work)`, computed the way it actually happens in production: **each
Snakemake job is a fresh process**, so "load" for that job = Python/library import time +
first model-load call; "work" = the first (only) forward pass that process ever does for its
one well. A resident server pays warm-load + steady-state work instead — both rows are shown.

| Family | Fresh-process ratio (today's reality) | Warm-load/steady-work ratio (server-world) |
|---|---|---|
| frame_detections (GroundingDINO) | **0.90** | 0.86 |
| frame_masks (SAM2) | **0.80** | 0.80 |
| snip_auxiliary_masks (4x UNet) | **0.92** | 0.96 |
| latent_embeddings (VAE, CPU) | **0.97** | 0.49 (see caveat) |

All four ratios are high under the fresh-process framing that matches current production
behavior — **load dominates for all four families**. The bottom row's warm-load ratio drops
because at steady state the *loaded* VAE forward pass is fast (~20ms) relative to import
overhead that doesn't shrink; see the per-family notes for why this number is less decisive
than the others.

## Per-family results

### 1. frame_detections — GroundingDINO

| Metric | Value |
|---|---|
| Import time | 10.6 s |
| Load time, cold (1st call in process) | **50.3 s** |
| Load time, warm (2nd/3rd call, same process) | 0.84 s, 1.00 s (avg 0.92 s) |
| Work, well A01 (15 frames), 1st call | 7.03 s |
| Work, well D07 (15 frames) | 0.146 s |
| Work, well F08 (15 frames) | 0.149 s |
| Peak GPU mem after load only | alloc 0 MB / reserved 0 MB (lazy CUDA alloc — GroundingDINO doesn't touch the GPU until first forward) |
| Peak GPU mem after 1st forward (well A01) | alloc 1689 MB / **reserved 1879 MB** |

**Note on the 50s cold load:** first call in the process both deserializes the checkpoint
from NFS *and* pays one-time CUDA context init + kernel JIT compilation the repeat calls
don't pay — we did not attempt to separate these two effects (would need `nvidia-smi` process
tracing or `CUDA_MODULE_LOADING=EAGER` timing, out of scope here); reported as one number,
honestly labeled "cold." Warm-in-process loads (0.84–1.00s) show what a resident server pays
per *subsequent* load only in the pathological case where it reloads weights but keeps the
CUDA context — the real server design reloads neither, so per-well cost would be ~0.15s only.

**Verdict: SERVE.** Even the best-case per-well work (0.15s once GPU is warm) is dwarfed by
the 50s cold-start (network + CUDA init) every fresh Snakemake job pays today.

### 2. frame_masks — SAM2 video predictor

| Metric | Value |
|---|---|
| Import time | 8.1 s |
| Load time, cold | 3.22 s |
| Load time, warm | 0.357 s, 0.360 s (avg 0.358 s) |
| Work, well A01 (1 segmentation frame, 1 prompt box), 1st call | 2.91 s |
| Work, well D07 | 0.100 s |
| Work, well F08 | 0.076 s |
| Peak GPU mem after load only | alloc 668 MB / reserved 684 MB |
| Peak GPU mem after 1st forward (well A01) | alloc 658 MB / **reserved 1112 MB** |

**Note on n_frames=1:** `select_segmentation_frame_view` (the function `cmd_frame_masks`
itself calls) selected only **1** frame for SAM2's model view in this well, not all 15
acquisition frames — this experiment's frame_detections table has exactly one kept detection
(at t0000), so SAM2 seeds+propagates over a 1-frame "video." This is what the real per-well
job does for this plate; we did not force a longer synthetic video. Because of this, "work"
here is closer to a lower bound — a well with more kept detection frames would show more
propagation work, which we could not demonstrate from this plate's actual data.

**Verdict: SERVE.** Load (cold 3.2s, or 8.1s import + 3.2s load = 11.3s per fresh job) exceeds
work (well under 1s once warm) by a wide margin.

### 3. snip_auxiliary_masks — 4x UNet (via/yolk/focus/bubble)

| Metric | Value |
|---|---|
| Import time | 33.1 s |
| Load time, cold (all 4 UNets, first attempt) | 73.3 s |
| Load time, cold (all 4 UNets, second attempt, different node) | 80.7 s |
| Load time, warm (2nd/3rd call, same process) | 0.90–0.92 s |
| Work, well A01 (1 snip, all 4 mask types), 1st call | 10.25 s |
| Work, well D07 (1 snip) | 0.043 s |
| Work, well F08 (1 snip) | 0.036 s |
| Peak GPU mem after load only (all 4 UNets resident) | alloc 868 MB / reserved 879 MB |
| Peak GPU mem after 1st forward (well A01) | alloc 462 MB / **reserved 879 MB** |

The cold load ran twice (73.3s and 80.7s, on two different GPU nodes) — the ~10% spread is
consistent with NFS latency variance rather than measurement noise. Loading 4 separate
FishModel (FPN+ResNet34) checkpoints sequentially is the single most expensive load of the
four families measured.

**Verdict: SERVE — the strongest case of the four.** A 73–81s cold load against <50ms
steady-state work per well (1 snip/well in this dataset) is the most lopsided ratio measured.
Note this dataset has only 1 snip/well; wells with more embryos/snips would raise the
per-well work time somewhat, but would need an order of magnitude more snips per well before
work would rival the observed load cost.

### 4. latent_embeddings — legacy VAE

| Metric | Value |
|---|---|
| Runs under | Python 3.9 (`mseq_pipeline_py3.9`), **not** the GPU-capable conda env |
| Import time | 37.3 s |
| Load time, cold | 0.47 s |
| Load time, warm | 0.020 s, 0.018 s (avg 0.019 s) |
| Work, well A01 (1 snip), 1st call | 1.03 s |
| Work, well D07 (1 snip) | 0.019 s |
| Work, well F08 (1 snip) | 0.021 s |
| GPU memory | **not measured — see below** |

**Critical finding: this family cannot currently use the GPU at all.** Production config
(`config.yaml`'s `legacy_embeddings` block) has no `device` key, and
`rules/latent_embeddings.smk` defaults to `_LE_CFG.get("device", "cpu")` — so **today this
stage runs on CPU in production**. We additionally confirmed the `mseq_pipeline_py3.9`
interpreter's torch build reports `cuda.is_available() == False` **even when run on the GPU
node with a free L40S visible to `nvidia-smi`** — i.e. that Python 3.9 environment's torch
2.5.1 was not built/installed with CUDA support. Requesting `--device cuda` in our benchmark
against that interpreter produced no error but also no GPU execution (correctly skipped, not
faked). **We could not produce a GPU memory number for this family and are not reporting
one** — CPU work has no `torch.cuda.max_memory_*` analog.

We separately confirmed (by reading `legacy_embeddings/entrypoint.py`) that
`run_legacy_embeddings()` already supports loading the encoder **once** and writing multiple
wells' latents in one process — the CLI takes `--snip-inventory-csv` / `--output-parquet` as
`nargs="+"`, paired positionally. We exercised this directly: loading once and encoding all 3
wells in one call took **0.338s total** vs. ~0.47s (cold) + ~1.0s (first work) ≈ 1.5s if that
same work were done as 3 separate fresh processes each paying import+load again. The
Snakemake **rule**, however, still declares this as one job per well (comment in
`latent_embeddings.smk`: "the batch optimization is deferred") — so production pays
import+load fresh per well today despite the code already supporting the batched shape.

**Verdict: mixed — the *rule* wiring needs the batch fix more urgently than a socket server
does.** The `import_time` (37s, driven by Python 3.9 + numpy/torch/pandas startup cost) is by
far the largest cost in this family's per-job total, and it is **not** something a model
server eliminates by itself if it still spawns a full new 3.9 interpreter per request — the
socket-server pattern helps here only if the 3.9 process itself stays resident and answers
requests without re-exec'ing. Simply wiring the existing batch-capable entrypoint into the
Snakemake rule (call `run_legacy_embeddings` with the whole run's wells, as intended by
`execution=RUN_BATCH_WRITES_PER_WELL_SHARDS`) captures nearly all of this gain without a new
server component. Separately: since this stage runs on CPU only today, it is not a candidate
for the GPU co-residency question below.

## Peak GPU memory summary (reserved — the number that matters for co-residency)

| Family | Reserved after load | Reserved after 1st forward (production-representative) |
|---|---|---|
| frame_detections | 0 MB (lazy) | 1879 MB |
| frame_masks | 684 MB | 1112 MB |
| snip_auxiliary_masks (4 UNets) | 879 MB | 879 MB |
| latent_embeddings | not measured (CPU-only today) | not measured |

**Sum of the three GPU-resident families' peak reserved memory:** 1879 + 1112 + 879 =
**3870 MB (~3.9 GB)**, against a **46068 MiB (~46 GB)** L40S. That is roughly **8.4% of one
GPU's capacity**, with ~42 GB of headroom to spare — and this cluster hands out full GPUs
per job (not fractional/MPS-shared), so in practice each family could be given an entire L40S
with room to spare, or all three could share a single L40S with a >10x safety margin.

## Co-residency verdict

**Yes — all three GPU-using families (frame_detections, frame_masks, snip_auxiliary_masks)
fit comfortably on one L40S simultaneously, with wide headroom (~42 GB free out of 46 GB).**
`latent_embeddings` is CPU-only in production today and is out of scope for GPU co-residency
until/unless it's moved onto a CUDA-capable Python 3.9 build.

This supports **Mode A** (all three GPU servers co-resident on one GPU, let Snakemake pack
jobs onto it) over **Mode B** (forced stage separation via a DAG barrier) *for memory
capacity reasons*. We did not measure compute-contention effects (e.g. whether GroundingDINO
and SAM2 forward passes interleaved on the same device slow each other down under concurrent
load) — that would require a follow-up concurrent-load test, which was out of scope for this
load-vs-work measurement. Note also that this cluster's SGE `cuda` resource currently hands
out **exclusive whole GPUs per job** — building a co-resident Mode-A server would need those
jobs to explicitly share one GPU allocation (e.g. one qsub job hosting all three servers, or
an MPS/time-slicing setup), which is an operational change beyond just "the memory fits."

## Per-model recommendation

| Family | Recommendation | Why |
|---|---|---|
| frame_detections (GroundingDINO) | **SERVE** | Cold load 50s vs. steady work 0.15s/well — load totally dominates. |
| frame_masks (SAM2) | **SERVE** | Cold load 3.2s (11.3s incl. import) vs. steady work <0.1s/well. |
| snip_auxiliary_masks (4x UNet) | **SERVE** | Cold load 73–81s vs. steady work <50ms/well — the largest load cost measured, most clear-cut case. |
| latent_embeddings (VAE) | **FIX THE RULE FIRST, THEN RE-EVALUATE** | The code already supports one-load-many-wells; the Snakemake rule doesn't use it. Wiring the existing batch entrypoint into the rule (no new server needed) captures most of the win. A resident server only helps further if it keeps the Python 3.9 process itself alive across requests — worth revisiting once (a) the batch rule fix lands and (b) whether this stage should move to GPU is decided (today: CPU-only, no CUDA-capable py3.9 torch build available). |

## Honest limitations / what we did NOT measure

- **True cold page-cache vs warm page-cache** could not be cleanly separated — we have no
  root access to drop OS page cache (`echo 3 > /proc/sys/vm/drop_caches`) on these shared
  nodes. Our "cold" number is "first read in a fresh process on a node we didn't just use";
  our "warm" number is "repeat read in the same process." A truly cold NFS read on a node
  that has never touched these weights could be slower or faster than what we saw — we did
  not attempt to force that state and are not claiming a number for it.
- **Well-size variation**: every well checked in `20260702_hotchem_36hpf_plate02` has
  identically 15 frames / 1 snip. We could not demonstrate how per-well work scales with more
  frames/snips from this plate's real data; we did not fabricate a scaling estimate.
- **GPU compute contention** under concurrent multi-model load (the actual Mode-A operating
  condition) was not measured — only static memory sums.
- **latent_embeddings GPU memory**: not measured, because this stage does not run on GPU in
  production and the available Python-3.9 torch build has no CUDA support to measure against.
- Import time (8–37s depending on family) is real per-fresh-process cost but is **not** solely
  "model load" — it's mostly Python/library import (torch, transformers/detectron-adjacent
  deps, pandas). A resident server pays this once regardless; a rule that merely batches
  wells within the *same* Snakemake job (like the deferred `latent_embeddings` batch design)
  would also amortize it without a socket server, which is why the `latent_embeddings` verdict
  above is call out separately.

## Exact commands (reproducibility)

All benchmark scripts are throwaway and live in a scratch dir (not committed, not part of the
pipeline): `/net/trapnell/vol1/home/mdcolon/proj/morphseq/.gpu_bench_scratch/`.

GPU check (confirms exclusive single-GPU-per-job allocation, no other tenants):
```bash
qsub -N gpu_tenants -l gpgpu=1 -l cuda=1 -l h_rt=0:5:0 -pe serial 1 -l mfree=2G \
     -q trapnell-short.q <script that runs: hostname; nvidia-smi>
```

frame_detections, frame_masks, snip_auxiliary_masks (all under `segmentation_grounded_sam`):
```bash
conda activate segmentation_grounded_sam
python .gpu_bench_scratch/bench_frame_detections.py
python .gpu_bench_scratch/bench_frame_masks.py
python .gpu_bench_scratch/bench_snip_auxiliary_masks.py
```
Submitted as one SGE job: `-l gpgpu=1 -l cuda=1 -l h_rt=1:0:0 -pe serial 4 -l mfree=32G -q trapnell-short.q`.

latent_embeddings (Python 3.9 model interpreter, per `env.yaml.runtime.model_python_executable`):
```bash
/net/trapnell/vol1/home/mdcolon/.local/share/mamba/envs/mseq_pipeline_py3.9/bin/python \
    .gpu_bench_scratch/bench_latent_embeddings.py --device cpu
/net/trapnell/vol1/home/mdcolon/.local/share/mamba/envs/mseq_pipeline_py3.9/bin/python \
    .gpu_bench_scratch/bench_latent_embeddings.py --device cuda   # confirmed no-op: cuda unavailable in this interpreter
```
Submitted as an SGE job with the same `-l gpgpu=1 -l cuda=1` resources (GPU requested so the
node is available for the `--device cuda` attempt, even though that attempt reported
`cuda_available: False` and was skipped).

Each benchmark script:
- imports the exact production loader/runner functions used by `tasks.py`'s `cmd_*` functions
  (`load_groundingdino_model`, `load_sam2_video_predictor`, `load_unet_snip_predictors` +
  `run_unet_for_snip_inventory`, `load_legacy_vae_encoder` + `encode_snips` +
  `run_legacy_embeddings`) — no reimplementation of model-loading logic;
- loads the model 3x in the same process (`load_times_s`: [cold, warm, warm]) and calls
  `torch.cuda.reset_peak_memory_stats()` before each phase so allocated/reserved figures are
  phase-scoped, not cumulative across the whole script;
- runs the real per-well work on wells `A01`, `D07`, `F08` of
  `20260702_hotchem_36hpf_plate02`, reading the actual on-disk frame_inventory /
  frame_detections / snip_inventory shards for those wells;
- writes a JSON of all timings/memory to `.gpu_bench_scratch/results_<family>.json`.

Full scripts (for exact reproduction): `bench_frame_detections.py`, `bench_frame_masks.py`,
`bench_snip_auxiliary_masks.py`, `bench_latent_embeddings.py`, all in
`/net/trapnell/vol1/home/mdcolon/proj/morphseq/.gpu_bench_scratch/`.

**No pipeline code, `.smk` rules, or `tasks.py` were modified to produce these numbers** —
per constraint, this was measurement-only. One genuine latent bug was discovered along the
way and is called out for a maintainer, not fixed here: `data_pipeline/models/unet.py`'s
`_ensure_src_on_path()` puts `.../morphseq/src` on `sys.path`, but the subsequent `from
src.core.functions.core_utils_segmentation import FishModel` needs `.../morphseq` (src's
*parent*) on the path instead — our benchmark script worked around this locally (added the
repo root too) rather than patching the pipeline file.
