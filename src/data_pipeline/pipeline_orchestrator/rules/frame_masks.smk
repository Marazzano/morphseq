"""Frame-masks product-family rules.

Consumes the per-well frame_inventory shard and the per-well frame_detections shard and
emits the per-well frame_masks shard. In the default ``model`` mode it runs SAM2 video
segmentation. In ``precomputed`` mode it selects authoritative canonical rows from an
experiment-level ingress CSV; detections remain an independent audit input and can never
replace those masks. The merged experiment-level table is an aggregate view for audit/reporting.

SAM2 path contract — see models/sam2.py for the full explanation. The short version:
  - sam2_models_root must point to the directory that *contains* the sam2/ package subdir
    (typically MODELS_DIR / "sam2", which is a symlink to the checkout).
  - sam2_config is a *relative* path like "configs/sam2.1/sam2.1_hiera_s.yaml" — the
    loader resolves it against the package subdir and passes it to Hydra in that form.
  - sam2_checkpoint is a *relative* path like "checkpoints/sam2.1_hiera_small.pt" —
    resolved against sam2_models_root by the loader.
  Never expand these to absolute paths before passing; the loader's chdir+Hydra dance
  requires the relative form.
"""

from data_pipeline.model_servers.socket_paths import service_socket_pattern

FRAME_MASKS_STEP = "frame_masks"

# Producer selection is mode-exclusive. Missing ``mode`` means the historical model path, so all
# non-SeaHub scopes retain their current behavior. ``precomputed`` takes precedence over
# ``use_model_server`` deliberately: a runtime may inherit use_model_server=true, but an
# authoritative handoff must never start SAM2 or let it overwrite imported masks.
FRAME_MASKS_CONFIG = config.get("frame_masks", {}) or {}
FRAME_MASKS_MODE = str(FRAME_MASKS_CONFIG.get("mode", "model")).strip().lower()
if FRAME_MASKS_MODE not in ("model", "precomputed"):
    raise ValueError(
        "config frame_masks.mode must be 'model' or 'precomputed', "
        f"got {FRAME_MASKS_MODE!r}."
    )

FRAME_MASKS_PRECOMPUTED_CSV = str(FRAME_MASKS_CONFIG.get("precomputed_csv", "") or "")
FRAME_MASKS_REQUIRE_DETECTION_AUDIT = bool(
    FRAME_MASKS_CONFIG.get("require_detection_audit", False)
)
if FRAME_MASKS_MODE == "precomputed":
    if not FRAME_MASKS_PRECOMPUTED_CSV:
        raise ValueError(
            "frame_masks.mode='precomputed' requires frame_masks.precomputed_csv."
        )
    if not Path(FRAME_MASKS_PRECOMPUTED_CSV).is_absolute():
        raise ValueError(
            "frame_masks.precomputed_csv must be an absolute per-experiment CSV path."
        )
    if not FRAME_MASKS_REQUIRE_DETECTION_AUDIT:
        raise ValueError(
            "frame_masks.mode='precomputed' requires "
            "frame_masks.require_detection_audit=true."
        )

# Opt-in resident SAM2 process for model mode only. The per-well client rules and their outputs
# stay unchanged; only model ownership moves from each client process into one service.
FRAME_MASKS_SERVED = bool(FRAME_MASKS_CONFIG.get("use_model_server", False))


def _frame_masks_socket_pattern() -> str:
    return service_socket_pattern("sam2")


def _frame_masks_artifact(experiment: str, artifact: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(FRAME_MASKS_STEP, artifact, experiment, path_mode=path_mode, well_id=well_id)

def _frame_masks_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(FRAME_MASKS_STEP, "frame_masks", experiment, path_mode=path_mode, well_id=well_id)

def _frame_masks_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, FRAME_MASKS_STEP, "frame_masks", wc.experiment, wells_for_experiment(wc))

def _frame_masks_validated_for_run(wc):
    return [_frame_masks_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w) for w in wells_for_experiment(wc)]


if FRAME_MASKS_MODE == "precomputed":
    rule frame_masks_per_well_precomputed:
        """Select authoritative precomputed masks; retain redundant detections as audit only."""
        input:
            precomputed=FRAME_MASKS_PRECOMPUTED_CSV,
            frame_inventory=str(_frame_inventory_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            frame_inventory_validated=str(_frame_inventory_validated(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            # Requiring both files proves the redundant detector ran and passed its own canonical
            # validation. The ingest task writes only rows selected from input.precomputed.
            frame_detections=str(_frame_detections_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            frame_detections_validated=str(_frame_detections_validated(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
        output:
            frame_masks=str(_frame_masks_artifact(
                "{experiment}", "frame_masks",
                path_mode=PATH_MODE_PER_WELL,
                well_id="{well_id}",
            )),
            prompt_seeds=str(_frame_masks_artifact(
                "{experiment}", "prompt_seeds",
                path_mode=PATH_MODE_PER_WELL,
                well_id="{well_id}",
            )),
        shell:
            """
            {RUN} -m data_pipeline.pipeline_orchestrator.tasks ingest-precomputed-frame-masks \
              --precomputed-frame-masks-csv "{input.precomputed}" \
              --frame-inventory-csv "{input.frame_inventory}" \
              --frame-detections-csv "{input.frame_detections}" \
              --well-id "{wildcards.well_id}" \
              --output-csv "{output.frame_masks}" \
              --prompt-seeds-csv "{output.prompt_seeds}" \
              --require-detection-audit true
            """


elif FRAME_MASKS_SERVED:
    rule service_sam2:
        """Load SAM2 once and serve every per-well segmentation request serially."""
        output:
            socket=service(_frame_masks_socket_pattern()),
        params:
            device=lambda wc: str(config.get("frame_masks", {}).get("device", DEVICE)),
            sam2_models_root=lambda wc: str(MODELS_DIR / "sam2"),
            sam2_config=lambda wc: str(config.get("frame_masks", {}).get(
                "sam2_config", "configs/sam2.1/sam2.1_hiera_s.yaml"
            )),
            sam2_checkpoint=lambda wc: str(config.get("frame_masks", {}).get(
                "sam2_checkpoint", "checkpoints/sam2.1_hiera_small.pt"
            )),
            sam2_model_id=lambda wc: str(config.get("frame_masks", {}).get(
                "sam2_model_id", "sam2.1_hiera_s"
            )),
        resources:
            gpu=1,
        shell:
            """
            {RUN} -m data_pipeline.model_servers.harness \
              --adapter sam2 \
              --socket-path "{output.socket}" \
              --adapter-arg sam2_models_root="{params.sam2_models_root}" \
              --adapter-arg sam2_config="{params.sam2_config}" \
              --adapter-arg sam2_checkpoint="{params.sam2_checkpoint}" \
              --adapter-arg sam2_model_id="{params.sam2_model_id}" \
              --adapter-arg device="{params.device}"
            """


    rule frame_masks_per_well_served:
        """Run one well through the resident SAM2 process, retaining ordinary outputs."""
        input:
            frame_inventory=str(_frame_inventory_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            frame_inventory_validated=str(_frame_inventory_validated(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            frame_detections=str(_frame_detections_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            socket=_frame_masks_socket_pattern(),
        threads: 0  # service groups sum client cores; blocked clients consume none
        output:
            frame_masks=str(_frame_masks_artifact(
                "{experiment}", "frame_masks",
                path_mode=PATH_MODE_PER_WELL,
                well_id="{well_id}",
            )),
            prompt_seeds=str(_frame_masks_artifact(
                "{experiment}", "prompt_seeds",
                path_mode=PATH_MODE_PER_WELL,
                well_id="{well_id}",
            )),
        params:
            sam2_model_id=lambda wc: str(config.get("frame_masks", {}).get(
                "sam2_model_id", "sam2.1_hiera_s"
            )),
        shell:
            """
            {RUN} -m data_pipeline.model_servers.client \
              --socket-path "{input.socket}" \
              --payload-json '{{"frame_inventory_csv": "{input.frame_inventory}", "frame_detections_csv": "{input.frame_detections}", "output_csv": "{output.frame_masks}", "prompt_seeds_csv": "{output.prompt_seeds}", "sam2_model_id": "{params.sam2_model_id}"}}'
            """


else:
    rule frame_masks_per_well:
        """Run SAM2 in-process for one well (the default compatibility path)."""
        input:
            frame_inventory=str(_frame_inventory_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            frame_inventory_validated=str(_frame_inventory_validated(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
            frame_detections=str(_frame_detections_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
        output:
            frame_masks=str(_frame_masks_artifact(
                "{experiment}", "frame_masks",
                path_mode=PATH_MODE_PER_WELL,
                well_id="{well_id}",
            )),
            prompt_seeds=str(_frame_masks_artifact(
                "{experiment}", "prompt_seeds",
                path_mode=PATH_MODE_PER_WELL,
                well_id="{well_id}",
            )),
        params:
            device=lambda wc: str(config.get("frame_masks", {}).get("device", DEVICE)),
            sam2_models_root=lambda wc: str(MODELS_DIR / "sam2"),
            sam2_config=lambda wc: str(config.get("frame_masks", {}).get(
                "sam2_config", "configs/sam2.1/sam2.1_hiera_s.yaml"
            )),
            sam2_checkpoint=lambda wc: str(config.get("frame_masks", {}).get(
                "sam2_checkpoint", "checkpoints/sam2.1_hiera_small.pt"
            )),
            sam2_model_id=lambda wc: str(config.get("frame_masks", {}).get(
                "sam2_model_id", "sam2.1_hiera_s"
            )),
        resources:
            gpu=1,
        shell:
            """
            {RUN} -m data_pipeline.pipeline_orchestrator.tasks frame-masks \
              --frame-inventory-csv "{input.frame_inventory}" \
              --frame-detections-csv "{input.frame_detections}" \
              --output-csv "{output.frame_masks}" \
              --prompt-seeds-csv "{output.prompt_seeds}" \
              --sam2-models-root "{params.sam2_models_root}" \
              --sam2-config "{params.sam2_config}" \
              --sam2-checkpoint "{params.sam2_checkpoint}" \
              --sam2-model-id "{params.sam2_model_id}" \
              --device "{params.device}"
            """


rule validate_frame_masks_for_well:
    """Validate the per-well frame_masks shard against its frame_inventory; write the .validated sentinel.

    The merge (collect_well_shard_paths) only picks up shards that carry a .validated sentinel, so
    this rule is what makes a built frame_masks shard eligible for the merged table — and what lets
    downstream consumers (e.g. physical_embryo_registry, snip_processing) depend on a *validated*
    frame_masks shard rather than a raw one.
    """
    input:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated=str(_frame_masks_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-masks \
          --input-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_frame_masks:
    """Row-stack per-well frame_masks shards into the experiment-level merged table."""
    input:
        per_well=_frame_masks_artifacts_for_run,
        per_well_validated=_frame_masks_validated_for_run,
    output:
        merged=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.object_extraction.segmentation.frame_masks_contract import FRAME_MASKS_REQUIRED_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'frame_masks', 'frame_masks', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=list(FRAME_MASKS_REQUIRED_COLUMNS), sort_columns=['experiment_id', 'well_id', 'time_index'])
"
        """
