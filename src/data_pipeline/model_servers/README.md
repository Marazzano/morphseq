# model_servers

Resident model-server harness. Loads a model **once** in a long-lived process and
serves per-well requests over a Unix domain socket, instead of paying model-load
cost in every Snakemake per-well job (today: import torch -> load weights -> do a
few seconds/minutes of work -> exit, repeated once per well).

**WIRED AND PROVEN** (2026-07-26), opt-in per step, default off:

| toggle | service |
|---|---|
| `frame_detections.use_model_server` | `service_grounding_dino` |
| `frame_masks.use_model_server` | `service_sam2` |
| `unet_snip.use_model_server` | `service_unet_aux_masks` |

The DAG shape does not change — a served rule's `shell:` calls `client.py` instead
of `tasks.py`, and Snakemake still checks for the same output file after the job
exits. Full-plate verification (job 22833499, 96 wells of 20260408_pbx): 866/866
steps, **2 adapter loads** (GroundingDINO 24.46s, 4x UNet 68.43s), 192 requests
served, 0 transport/group/rule errors. Per-well reloading would have cost ~2.5h;
it cost 93s.

Before changing a served rule, read `socket_paths.py` — it lists the five traps
this wiring cost us, every one of which fails silently or late.

## Why paths, not payloads

Client and server share a filesystem. A request carries **input/output file
paths**; the server does the inference and writes the result itself. The client
never sees pixel/array data and never imports torch or any model library — it is
a ~15-line generic program that connects, sends a small JSON request, blocks, and
exits 0 or nonzero. This is also what preserves Snakemake's contract: Snakemake
checks that the output file exists after the client process exits, and doesn't
care which process actually wrote it.

## Layout

```
protocol.py       Wire format (length-prefixed JSON). Stdlib only. No torch.
harness.py        Generic server: socket bind, accept loop, dispatch, signals,
                  logging. NEVER imports torch. Knows nothing about tensors,
                  CUDA, wells, or biology.
client.py         Generic, tiny client. Connect -> send request -> block for
                  reply -> exit 0/nonzero. Never imports torch.
adapter_base.py   The ModelAdapter interface (load/handle) + name->class
                  registry. The ONLY seam where model-specific code is allowed.
atomic_write.py   Small temp-file+rename helper adapters use to finalize output
                  files so a partial write is never visible at the final path.
adapters/
  fake.py         Trivial no-GPU adapter used by the harness/client tests.
  sam2.py         Real SAM2 video-predictor adapter (frame_masks product).
tests/
  test_harness.py Generic-parts tests (dispatch, round-trip, atomic write,
                  error propagation, concurrency) using the fake adapter --
                  no GPU required.
```

## Starting a server

```bash
conda run -n segmentation_grounded_sam --no-capture-output python \
  -m data_pipeline.model_servers.harness \
  --adapter sam2 \
  --socket-path /path/to/sam2.sock \
  --adapter-arg sam2_models_root=/path/to/models_root/sam2 \
  --adapter-arg sam2_config=configs/sam2.1/sam2.1_hiera_s.yaml \
  --adapter-arg sam2_checkpoint=checkpoints/sam2.1_hiera_small.pt \
  --adapter-arg sam2_model_id=sam2.1_hiera_s \
  --adapter-arg device=cuda
```

`--adapter-arg KEY=VALUE` is repeatable; each pair becomes a keyword argument to
the adapter class's `__init__`. All adapter-arg values arrive as **strings** (the
harness does not know or care about an adapter's constructor signature beyond
that).

**Readiness contract:** the harness creates the socket file only *after*
`adapter.load()` returns successfully. A client that manages to connect
therefore implies the model is loaded and ready to serve — there is no separate
handshake or health check.

**Shutdown:** SIGTERM (or SIGINT) triggers a clean exit — the accept loop stops,
in-flight handlers get a bounded join, and the socket file is removed. (Signal
handlers can only be installed on the main thread of the main interpreter; if you
embed `ModelServer` in a background thread — as the tests do — install your own
shutdown trigger via `server._stop_event.set()` instead.)

## Calling it from a client

```bash
conda run -n segmentation_grounded_sam --no-capture-output python \
  -m data_pipeline.model_servers.client \
  --socket-path /path/to/sam2.sock \
  --payload-json '{"frame_inventory_csv": "...", "frame_detections_csv": "...", "output_csv": "...", "prompt_seeds_csv": "..."}' \
  --timeout 600
```

Exit code 0 = success (server wrote the output file(s)); nonzero = failure. On
failure, stderr prints the server's error string. `--payload-json` is passed
through untouched to the adapter's `handle()` as a dict — the client does not
interpret it.

You can also call `data_pipeline.model_servers.client.call_server(socket_path,
payload, timeout=...)` directly from Python; it returns a `Response(request_id,
ok, error)`.

## What an adapter author implements

Exactly two methods (see `adapter_base.py` for the full interface docstring):

```python
from data_pipeline.model_servers.adapter_base import register_adapter

@register_adapter("my_model")
class MyModelAdapter:
    def __init__(self, **kwargs):
        ...  # adapter-arg strings land here

    def load(self) -> None:
        """Called once, before the socket file is created. Load weights here."""
        ...

    def handle(self, payload: dict) -> None:
        """Called once per request. `payload` is your own request shape (paths
        in, paths out). Do the work and write your own output file(s). Raise on
        failure -- the harness converts any exception into an error Response
        without killing the process or corrupting state for the next request.
        Return value is ignored."""
        ...
```

Responsibilities that belong to the adapter, not the harness:
- **Output atomicity.** Use `atomic_write.atomic_write_via(path, writer_fn)` (or
  `atomic_write_bytes`) so a partial write is never observable at the final path.
- **Per-request isolation.** If your model holds mutable state across a call
  (see SAM2 below), you are responsible for guaranteeing that state from one
  request cannot leak into or corrupt the next. The harness's only guarantee is
  that an exception in `handle()` doesn't crash the process or skip cleanup of
  *its own* dispatch loop -- it has no visibility into your model's internals.
- **Registering yourself** in `adapters/__init__.py` (import your module there;
  the `@register_adapter("name")` decorator does the rest) so `--adapter <name>`
  can find you.

Import your model library (torch, etc.) freely inside your adapter module — that
is the whole point of isolating model-specific code behind this interface.
`adapters/__init__.py` wraps the SAM2 import in a `try/except ImportError` so the
harness and the fake-adapter tests stay importable in environments without torch.

### Resist adding new hooks to the interface

The interface is intentionally just `load()` + `handle()`. If a new adapter seems
to need a third method (a pre-flight check, a mid-batch callback, a teardown
hook), treat that as a signal the seam is drawn wrong for that model family, not
as a reason to grow this interface — solve it inside that adapter's `handle()`
instead. See `adapter_base.py`'s docstring for the reasoning: the three model
families this harness targets (stateful video predictor, stateless per-frame
detector, multi-model UNet ensemble) have genuinely different call shapes, and a
common interface only helps if it stays this narrow.

## The SAM2 adapter (`adapters/sam2.py`)

SAM2 is the **stateful** model family: `predictor.init_state(video_path=...)`
returns a fresh `inference_state` dict scoped to that call (verified by reading
`sam2_video_predictor.py` -- all mutable tracking state lives in that returned
dict, not on the predictor object itself). The adapter's isolation guarantee is:
call `init_state()` fresh for every request, never reuse a previous well's state
object, and drop the reference as soon as `handle()` is done with it. There is no
separate `predictor.reset()` call needed as long as that discipline holds.

The per-well control flow (read frame_inventory/frame_detections shards -> build
RGB JPEG frames in a temp dir -> seed from kept detections -> propagate -> adapt
+ validate -> write frame_masks + prompt_seeds) is duplicated from
`cmd_frame_masks` in `pipeline_orchestrator/tasks.py` rather than imported,
because that function does load+run+write as one inline block with no load/run
split. Factoring a shared load/run split out of `cmd_frame_masks` is natural
follow-up work but was out of scope for this prototype pass (see the
`model_servers/__init__.py` and `adapters/sam2.py` docstrings for the same note).
Everything the adapter calls into beyond that top-level control flow (the SAM2
loader, frame-view selection, output adapter, validator) is imported from the
existing modules, not duplicated.

## Running the tests

```bash
PYTHONPATH=src conda run -n segmentation_grounded_sam --no-capture-output python \
  -m pytest src/data_pipeline/model_servers/tests/test_harness.py -v
```

These use the `fake` adapter and need no GPU. They cover: readiness ordering
(socket file appears only after `load()`), client/server round-trip, atomic
write (no leftover temp file, no truncated content), error propagation from a
failing request without corrupting the server for subsequent requests, and
concurrent requests staying isolated from each other.
