"""GroundingDINO adapter for the resident model-server harness (PROTOTYPE).

This is the ONLY torch/GroundingDINO-aware file in this package (besides the tests that exercise
it). Unlike `adapters/sam2.py` -- which duplicates `cmd_frame_masks`'s per-well control flow because
that function has no load/run split -- the GroundingDINO per-well path already IS load/run split:
`cmd_frame_detections` in `data_pipeline.pipeline_orchestrator.tasks` (~line 379) does exactly two
things: call `load_groundingdino_model(...)` then call `run_frame_detection(...)`. Both are ordinary
importable functions with no inline duplication of internal logic. So this adapter IMPORTS both
rather than re-implementing anything:

    load()   -> data_pipeline.models.groundingdino.load_groundingdino_model
    handle() -> data_pipeline.object_extraction.detection.run_frame_detection.run_frame_detection

This keeps this file a thin wrapper: convert harness path-payload <-> existing function signatures,
plus the one behavior the existing CSV-path function does not give us for free -- atomic output
writes (`run_frame_detection` calls `df.to_csv(output_csv, ...)` directly, which is fine for a
one-shot process but not safe for a long-lived server where a bad write must never leave a partial
file visible at the final path). We reuse `run_frame_detection_df` (the in-memory core) and do our
own atomic_write_via(...) of the resulting DataFrame instead of calling the CSV-path wrapper, so we
get IDENTICAL output content without duplicating any detection logic.

Request payload contract (paths, not payloads -- see protocol.py):
    {
      "frame_inventory_csv": str,
      "output_csv": str,
      "detector_model_id": str,   # optional, defaults to the value passed at adapter construction
    }

GDINO is the STATELESS model family (a per-frame detector, not a video predictor like SAM2).
Verified by reading the full call chain this adapter exercises:
  - `run_frame_detection_df` loops over BF frames and calls the backend's `detect_frame` per frame,
    accumulating rows in a local list. No object here holds a reference to a previous call's result.
  - `detect_frame` (groundingdino backend) calls `detect_embryos` then `filter_detections`, both of
    which build and return fresh local Python lists every call. Nothing is attached to `model`.
  - `detect_embryos` calls `groundingdino.util.inference.predict`, which does `model = model.to(device)`
    (rebinding the LOCAL name, not mutating a persistent attribute) and runs the forward pass under
    `torch.no_grad()`. The only model attribute it reads is `model.tokenizer`, which is immutable
    per-instance state set at construction, not written by `predict()`.
  - Nowhere in this path does any code do `model.<attr> = ...`. Grepped
    `object_extraction/detection/` for that pattern and found none.
So per-request isolation is automatic here: every `handle()` call is a fresh read -> pure functional
detection loop -> write, with no request-scoped state to reset (unlike SAM2's `inference_state`,
which this file therefore has no analog of).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.model_servers.adapter_base import register_adapter
from data_pipeline.model_servers.atomic_write import atomic_write_via
from data_pipeline.models.groundingdino import load_groundingdino_model
from data_pipeline.object_extraction.detection.backends.groundingdino.config import (
    GroundingDinoDetectionConfig,
)
from data_pipeline.object_extraction.detection.run_frame_detection import run_frame_detection_df


@register_adapter("grounding_dino")
class GroundingDinoAdapter:
    """Resident-server adapter wrapping the (stateless) GroundingDINO detector.

    Constructor args come from the harness's `--adapter-arg KEY=VALUE` flags, so they arrive as
    strings; Path-typed args are wrapped in Path() the same way Sam2Adapter does for its model paths.
    """

    def __init__(
        self,
        *,
        gdino_repo_dir: str,
        gdino_config: str,
        gdino_weights: str,
        detector_model_id: str = "SwinT_OGC",
        device: str = "cuda",
    ) -> None:
        self.gdino_repo_dir = gdino_repo_dir
        self.gdino_config = gdino_config
        self.gdino_weights = gdino_weights
        self.default_detector_model_id = detector_model_id
        self.device = device
        self.model = None

    def load(self) -> None:
        # Loaded once, before the harness creates the socket file. Mirrors
        # cmd_frame_detections's load call exactly (same function, same kwargs).
        self.model = load_groundingdino_model(
            repo_dir=Path(self.gdino_repo_dir),
            config_path=Path(self.gdino_config),
            weights_path=Path(self.gdino_weights),
            device=self.device,
        )

    def handle(self, payload: dict[str, Any]) -> None:
        if self.model is None:
            raise RuntimeError("GroundingDinoAdapter.handle() called before load()")

        frame_inventory_csv = Path(payload["frame_inventory_csv"])
        output_csv = Path(payload["output_csv"])
        detector_model_id = str(payload.get("detector_model_id", self.default_detector_model_id))

        # Same config construction as cmd_frame_detections (device is the only knob threaded
        # through today; text_prompt/box_threshold/text_threshold/iou_threshold stay at their
        # dataclass defaults, matching the per-well CLI path exactly).
        config = GroundingDinoDetectionConfig(device=self.device)

        reference_frame_inventory = pd.read_csv(frame_inventory_csv)
        # Use the in-memory core (not the CSV-path wrapper run_frame_detection) so we control the
        # write ourselves and can make it atomic -- the computed DataFrame is byte-for-byte the same
        # either way, since run_frame_detection is a thin read/write shell around this same function.
        frame_detections = run_frame_detection_df(
            reference_frame_inventory,
            backend="groundingdino",
            model=self.model,
            detector_model_id=detector_model_id,
            config=config,
        )

        atomic_write_via(output_csv, lambda tmp: frame_detections.to_csv(tmp, index=False))
