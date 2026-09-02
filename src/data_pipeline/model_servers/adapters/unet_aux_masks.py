"""UNet auxiliary-mask adapter for the resident model-server harness (PROTOTYPE).

This is the HIGHEST-VALUE adapter in the set: the per-well `snip-auxiliary-masks`
task loads FOUR separate UNet checkpoints (via / yolk / focus / bubble) every time
it runs, measured at 73-81s of pure model loading before any inference happens
(see `docs/data_pipeline/MODEL_LOAD_BENCHMARKS.md`). This adapter loads all
four once in `load()` and reuses them across every well the resident server serves.

WHAT THIS DUPLICATES VS. IMPORTS:
This adapter does NOT duplicate any model-specific logic. Unlike the SAM2 adapter
(which has no load/run split in its per-well entrypoint and so has to re-implement
the per-well body), the unet_snip backend was ALREADY split cleanly:

    parse_unet_snip_model_config()   -- config -> list[UNetSnipPredictorConfig]
    load_unet_snip_predictors()      -- load()-time: build the 4 FishModel predictors
    run_unet_for_snip_inventory()    -- handle()-time: run predictors over one well's
                                         snip_inventory, write mask PNGs, return the
                                         manifest DataFrame
    validate_snip_auxiliary_masks(), validate_snip_auxiliary_masks_against_snip_inventory()

This adapter imports all four and calls them in the same order and with the same
arguments as `run_snip_auxiliary_masks()` in
`data_pipeline.object_extraction.segmentation.backends.unet_snip.entrypoint`
(the function `cmd_snip_auxiliary_masks` in pipeline_orchestrator/tasks.py calls).
The only thing NOT reused is the entrypoint function's own body, because that
function does config-parsing + predictor-loading + running as one call, and this
adapter needs to split "parse config + load predictors" (load()) from "run for one
well" (handle()). The split is mechanical: `run_snip_auxiliary_masks()`'s source
is copied here across the load()/handle() boundary line-for-line, not rewritten.

Because handle() calls the exact same `run_unet_for_snip_inventory` +
`validate_snip_auxiliary_masks` + `validate_snip_auxiliary_masks_against_snip_inventory`
functions as the per-well path, with the same inputs, output is byte-identical to
the existing per-well path by construction (same code path from that point down) --
see the equivalence-proof script for a concrete comparison rather than just this
argument.

VERIFIED MODEL INDEPENDENCE (per task instructions: "don't assume, verify"):
  - Read `model_loader.py`: `load_unet_snip_predictors()` calls
    `load_fish_unet_model(cfg.checkpoint_path, device=...)` once per mask family.
    Each call constructs a BRAND NEW `FishModel("FPN", "resnet34", ..., encoder_weights=None)`
    and loads that family's own state dict into it. There is no shared nn.Module,
    no shared encoder, and no weight-tying between the four models -- four
    independent instances of the same architecture with four different checkpoints.
  - Read `run_unet_snip.py`: `run_auxiliary_mask_predictors_for_snip()` calls each
    predictor independently on the SAME input snip image
    (`predictor(snip_image) for mask_type, predictor in predictors.items()`).
    No mask family's output is fed into another's input -- all four run in
    parallel-conceptually (sequentially in this implementation) off one shared
    read-only input, not a pipeline.
  - Per-well state: `FishModelSnipPredictor.__call__` is a pure function of its
    input image plus the frozen model weights/shapes captured at construction
    (`self._model`, `self._model_input_shape`, `self._artifact_shape`, `self._device`).
    No normalization statistics, running caches, or mutable buffers are computed
    from one snip and reused for another (each call resizes + stacks + runs
    `torch.no_grad()` fresh). `model.eval()` is set once at load time (in
    `load_fish_unet_model`) and never toggled per-request, so no train/eval or
    dropout/batchnorm state can leak between wells either.
  - Net: the four models are safe to load once and reuse across every well's
    `handle()` call. There is no cross-well or cross-model state to reset.

Request payload contract (paths, not payloads -- see protocol.py):
    {
      "snip_inventory_csv": str,   # per-well snip_inventory shard (already validated)
      "output_root": str,          # data root; masks + manifest paths are written under here
      "output_csv": str,           # where to write the per-well snip_auxiliary_masks manifest
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.model_servers.adapter_base import register_adapter
from data_pipeline.model_servers.atomic_write import atomic_write_via
from data_pipeline.object_extraction.segmentation.backends.unet_snip.model_loader import (
    load_unet_snip_predictors,
    parse_unet_snip_model_config,
)
from data_pipeline.object_extraction.segmentation.backends.unet_snip.run_unet_snip import (
    UNET_SNIP_BACKEND_LABEL,
    run_unet_for_snip_inventory,
)
from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    validate_snip_auxiliary_masks,
    validate_snip_auxiliary_masks_against_snip_inventory,
)
from data_pipeline.object_extraction.snip_processing.io import resolve_snip_inventory_image_paths
from data_pipeline.object_extraction.snip_processing.snip_frame_shape import resolve_snip_frame_shape


@register_adapter("snip_auxiliary_masks")
class SnipAuxiliaryMasksAdapter:
    """Resident-server adapter wrapping the four unet_snip auxiliary-mask predictors.

    Constructor args come from the harness's `--adapter-arg KEY=VALUE` flags, so they
    arrive as strings. `unet_snip_config_json` carries the resolved `unet_snip` config
    block (models_root + per-family checkpoint/model_input_shape) as a JSON string,
    since the harness's CLI only supports flat KEY=VALUE pairs, not nested config --
    the caller is expected to `json.dumps(unet_snip_config)` the same dict that
    `cmd_snip_auxiliary_masks` builds from config.yaml + --models-root today.
    `snip_frame_shape_json` similarly carries the resolved `[height, width]` pair
    (see `resolve_snip_frame_shape`), so this adapter never re-reads config.yaml
    itself -- the caller resolves both once, at server-start time, exactly as
    `cmd_snip_auxiliary_masks` does per-well today.
    """

    def __init__(
        self,
        *,
        unet_snip_config_json: str,
        snip_frame_shape_json: str = "[576, 256]",
        device: str = "cuda",
    ) -> None:
        import json

        self.unet_snip_config: dict[str, Any] = json.loads(unet_snip_config_json)
        # device passed explicitly wins over any device embedded in the config block,
        # mirroring the per-well path where unet_snip_config["device"] is the only
        # source -- but the harness's own --adapter-arg device=... convention (see
        # sam2.py) is honored too, so both call styles behave.
        if device and "device" not in self.unet_snip_config:
            self.unet_snip_config["device"] = device
        self.snip_frame_shape = tuple(json.loads(snip_frame_shape_json))
        self.predictors: dict[str, Any] | None = None
        self.checkpoint_paths: dict[str, str] | None = None
        self.model_id: str | None = None

    def load(self) -> None:
        artifact_shape = (int(self.snip_frame_shape[0]), int(self.snip_frame_shape[1]))
        specs = parse_unet_snip_model_config(self.unet_snip_config, artifact_shape=artifact_shape)
        device = str(self.unet_snip_config.get("device", "cuda"))
        # Loads all FOUR FishModel checkpoints (via/yolk/focus/bubble) once, before the
        # harness creates the socket file. This IS the amortization: the 73-81s cost
        # measured per-well happens exactly once for the life of the server process.
        self.predictors = load_unet_snip_predictors(specs, device=device)
        self.checkpoint_paths = {spec.mask_type: str(spec.checkpoint_path) for spec in specs}
        self.model_id = str(self.unet_snip_config.get("model_id", UNET_SNIP_BACKEND_LABEL))
        self._artifact_shape = artifact_shape

    def handle(self, payload: dict[str, Any]) -> None:
        if self.predictors is None:
            raise RuntimeError("SnipAuxiliaryMasksAdapter.handle() called before load()")

        snip_inventory_csv = Path(payload["snip_inventory_csv"])
        output_root = Path(payload["output_root"])
        output_csv = Path(payload["output_csv"])

        # Below this line, the control flow mirrors
        # object_extraction/segmentation/backends/unet_snip/entrypoint.py::run_snip_auxiliary_masks
        # line-for-line (from the point predictors already exist), so per-well output
        # is produced by the identical code path as the existing per-well task.
        snip_inventory = pd.read_csv(snip_inventory_csv)

        resolved_inventory = resolve_snip_inventory_image_paths(snip_inventory, output_root=output_root)
        resolved_inventory = snip_inventory.merge(resolved_inventory, on="snip_id", how="left")
        resolved_inventory["processed_snip_path"] = resolved_inventory["resolved_image_path"]
        resolved_inventory = resolved_inventory.drop(columns=["resolved_image_path"])

        masks_dir = output_root / "object_extraction"
        df = run_unet_for_snip_inventory(
            resolved_inventory,
            self.predictors,
            output_dir=masks_dir,
            model_id=self.model_id,
            model_backend=UNET_SNIP_BACKEND_LABEL,
            checkpoint_paths=self.checkpoint_paths,
            artifact_shape=self._artifact_shape,
        )

        # Contract (self) + cross-check identity against the snip_inventory it was built from --
        # same two calls, same order, as the per-well entrypoint.
        validate_snip_auxiliary_masks(df)
        validate_snip_auxiliary_masks_against_snip_inventory(df, snip_inventory)

        atomic_write_via(output_csv, lambda tmp: df.to_csv(tmp, index=False))
