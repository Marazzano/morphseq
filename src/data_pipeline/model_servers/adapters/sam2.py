"""SAM2 adapter for the resident model-server harness (PROTOTYPE).

This is the ONLY torch-aware / SAM2-aware file in this package (besides the tests
that exercise it). It duplicates the per-well body of `cmd_frame_masks` in
`data_pipeline.pipeline_orchestrator.tasks` (~line 1043) rather than importing it,
because that function does load+run+write as one inline block with no load/run
split — factoring that apart is in-scope future cleanup, NOOT this prototype pass
(see the package README's "what should later be shared" note). Everything this
adapter calls out to (loader, frame-view selection, output adapter, validator) IS
imported from the existing modules; only the top-level control flow is duplicated.

Request payload contract (paths, not payloads — see protocol.py):
    {
      "frame_inventory_csv": str,
      "frame_detections_csv": str,
      "output_csv": str,
      "prompt_seeds_csv": str,
      "sam2_model_id": str,        # optional, defaults to the value passed at adapter construction
    }

SAM2 is the STATEFUL model family: `predictor.init_state(video_path=...)` builds a
fresh `inference_state` dict scoped to that call (verified against
sam2_video_predictor.py — all mutable tracking state lives in that returned dict,
not on the predictor object). This adapter's isolation guarantee is: never reuse an
`inference_state` across requests, and never hold a reference to the previous
well's state after `handle()` returns. That is the whole "reset between wells"
story for this model family — there is no separate predictor.reset() call needed
as long as we always call init_state() fresh and let the old state object be
garbage collected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from data_pipeline.model_servers.adapter_base import register_adapter
from data_pipeline.model_servers.atomic_write import atomic_write_via
from data_pipeline.models.sam2 import load_sam2_video_predictor
from data_pipeline.object_extraction.segmentation.backends.sam2_video.adapt_sam2_output import (
    adapt_sam2_well_output,
)
from data_pipeline.object_extraction.segmentation.backends.sam2_video.prompt_detections import (
    select_segmentation_frame_view,
    validate_sam2_prompts,
)
from data_pipeline.object_extraction.segmentation.validate_frame_masks import validate_frame_masks


def _to_rgb_jpeg(src: Path, dst: Path) -> None:
    Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)


@register_adapter("sam2")
class Sam2Adapter:
    """Resident-server adapter wrapping the SAM2 video predictor.

    Constructor args come from the harness's `--adapter-arg KEY=VALUE` flags, so
    they arrive as strings; this adapter converts what it needs (device is already
    a string, model paths are strings that get wrapped in Path()).
    """

    def __init__(
        self,
        *,
        sam2_models_root: str,
        sam2_config: str,
        sam2_checkpoint: str,
        sam2_model_id: str = "sam2_video",
        device: str = "cuda",
    ) -> None:
        self.sam2_models_root = sam2_models_root
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.default_model_id = sam2_model_id
        self.device = device
        self.predictor = None

    def load(self) -> None:
        # Loaded once, before the harness creates the socket file. Never expand
        # sam2_config / sam2_checkpoint to absolute paths — see models/sam2.py and
        # frame_masks.smk docstrings for why the relative form is required by the
        # Hydra chdir dance.
        self.predictor = load_sam2_video_predictor(
            sam2_models_root=Path(self.sam2_models_root),
            config_path=Path(self.sam2_config),
            checkpoint_path=Path(self.sam2_checkpoint),
            device=self.device,
        )

    def handle(self, payload: dict[str, Any]) -> None:
        if self.predictor is None:
            raise RuntimeError("Sam2Adapter.handle() called before load()")

        frame_inventory_csv = Path(payload["frame_inventory_csv"])
        frame_detections_csv = Path(payload["frame_detections_csv"])
        output_csv = Path(payload["output_csv"])
        prompt_seeds_csv = Path(payload["prompt_seeds_csv"])
        model_id = str(payload.get("sam2_model_id", self.default_model_id))

        frame_inventory = pd.read_csv(frame_inventory_csv)
        frame_detections = pd.read_csv(frame_detections_csv)
        well_id = str(frame_inventory["well_id"].iloc[0])
        model_inventory = select_segmentation_frame_view(frame_inventory, frame_detections)

        kept = frame_detections[frame_detections["is_kept"].astype(bool)].copy()
        prompt_detections = kept.rename(columns={"detection_id": "prompt_detection_id"})[
            [
                "prompt_detection_id", "image_id", "time_index",
                "bbox_x_min_px", "bbox_y_min_px", "bbox_x_max_px", "bbox_y_max_px",
                "is_kept",
            ]
        ]
        validate_sam2_prompts(prompt_detections, model_inventory)

        ordered = (
            model_inventory.sort_values(["time_index", "image_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        model_frame_view = ordered.copy()
        model_frame_view["sam2_frame_index"] = model_frame_view.index

        with tempfile.TemporaryDirectory(prefix=f"sam2_frames_{well_id}_") as tmpdir:
            rgb_dir = Path(tmpdir)
            for _, row in ordered.iterrows():
                dst = rgb_dir / f"{int(row['time_index']):05d}.jpg"
                _to_rgb_jpeg(Path(str(row["image_path"])), dst)

            seed_time = int(prompt_detections["time_index"].min())
            seed_sam2_idx = int(
                model_frame_view[model_frame_view["time_index"] == seed_time]["sam2_frame_index"].iloc[0]
            )

            # Fresh inference_state per request — this IS the reset. init_state()
            # returns a new dict scoped to this call; nothing from a previous
            # well's state is referenced or reused below, and the object is
            # dropped (eligible for GC) as soon as this `with` block/method exits.
            inference_state = self.predictor.init_state(video_path=str(rgb_dir))
            try:
                seed_prompts = prompt_detections[prompt_detections["time_index"] == seed_time].reset_index(
                    drop=True
                )
                for i, (_, row) in enumerate(seed_prompts.iterrows()):
                    box = np.array(
                        [
                            row["bbox_x_min_px"], row["bbox_y_min_px"],
                            row["bbox_x_max_px"], row["bbox_y_max_px"],
                        ],
                        dtype=np.float32,
                    )
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=seed_sam2_idx,
                        obj_id=i,
                        box=box,
                    )

                sam2_raw_output: dict[int, dict[int, np.ndarray]] = {}
                for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
                    masks: dict[int, np.ndarray] = {}
                    for obj_id, logit in zip(obj_ids, mask_logits):
                        arr = logit.squeeze().cpu().numpy() if hasattr(logit, "cpu") else np.squeeze(np.asarray(logit))
                        masks[int(obj_id)] = (arr > 0).astype(bool)
                    sam2_raw_output[int(frame_idx)] = masks
            finally:
                # Explicitly drop the state reference. Combined with never reusing
                # it for another request, this is the isolation guarantee: no
                # mutable SAM2 tracking state survives across handle() calls.
                del inference_state

        frame_masks = adapt_sam2_well_output(
            well_id,
            sam2_raw_output,
            model_frame_view,
            prompt_detections,
            model_id=model_id,
        )
        validate_frame_masks(frame_masks, model_inventory)

        atomic_write_via(output_csv, lambda tmp: frame_masks.to_csv(tmp, index=False))
        atomic_write_via(prompt_seeds_csv, lambda tmp: prompt_detections.to_csv(tmp, index=False))
