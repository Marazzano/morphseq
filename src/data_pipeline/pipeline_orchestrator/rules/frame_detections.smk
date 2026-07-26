"""Frame-detections product-family rules.

Consumes the per-well frame_inventory shard (the microscope-agnostic handoff seam) and runs
GroundingDINO to produce the per-well frame_detections shard. The merged experiment-level table
is an aggregate view for audit/reporting; downstream per-well consumers (frame_masks — not yet
wired) depend on the per-well shard directly.
"""

FRAME_DETECTIONS_STEP = "frame_detections"
FRAME_DETECTIONS_ARTIFACT = "frame_detections"

# --- resident model server (opt-in) -----------------------------------------------------------
# When enabled, GroundingDINO is loaded ONCE by rule service_grounding_dino and the per-well jobs
# become thin socket clients. Measured: ~50s of model load per well today, ~8h across a 576-well
# run. The DAG is unchanged either way -- still one job per well, same shards -- because Snakemake
# only checks that `output` exists and the command exited 0; it does not care which process wrote
# the file. See docs/MODEL_SERVER_WIRING.md.
#
# TWO THINGS THAT WILL BITE YOU:
#   1. The client rule must NOT declare `resources: gpu=1`. The client holds no GPU memory -- it
#      sends two paths over a socket and blocks. The SERVICE holds the GPU. If both declare gpu=1
#      the service takes the only unit, no client is ever schedulable, and Snakemake blocks
#      forever WITHOUT an error (it correctly concludes nothing is runnable).
#   2. Services are incompatible with --cores 1: the service job occupies a core for the whole
#      run, so service + client needs >= 2 cores, else "Excess Resources: _cores: 2/1".
FRAME_DETECTIONS_SERVED = bool(config.get("frame_detections", {}).get("use_model_server", False))

def _frame_detections_socket(experiment: str) -> Path:
    """Socket path for this experiment's GroundingDINO service.

    NOT under DATA_ROOT. AF_UNIX socket paths are capped at ~108 bytes by the kernel
    (sockaddr_un.sun_path), and DATA_ROOT alone is ~90 bytes on the shared nlammers tree --
    the natural path
        {DATA_ROOT}/object_extraction/frame_detections/{experiment}/grounding_dino.sock
    is 157 bytes and fails at bind() with an unhelpful error. The DAG builds fine; the service
    just dies on startup, so this is a runtime-only trap.

    A short /tmp path avoids it. This is sound because the socket is a transient IPC endpoint,
    not a data artifact: server and client always run on the same node (AF_UNIX cannot cross
    nodes anyway), and the file is recreated per run. Hashing the experiment keeps the name
    short and collision-free while staying per-experiment, so concurrent runs on different
    experiments cannot share a socket.
    """
    digest = hashlib.sha1(str(experiment).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"morphseq_gdino_{digest}.sock"


def _frame_detections_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(FRAME_DETECTIONS_STEP, FRAME_DETECTIONS_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _frame_detections_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(FRAME_DETECTIONS_STEP, FRAME_DETECTIONS_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _frame_detections_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, FRAME_DETECTIONS_STEP, FRAME_DETECTIONS_ARTIFACT, wc.experiment, wells_for_experiment(wc))

def _frame_detections_validated_for_run(wc):
    return [_frame_detections_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w) for w in wells_for_experiment(wc)]


rule service_grounding_dino:
    """Resident GroundingDINO server: load the model ONCE, serve per-well requests over a socket.

    Only instantiated when frame_detections.use_model_server is true. `service(...)` marks the
    socket as a service output, so Snakemake starts this job when the first consumer needs it and
    tears it down (SIGTERM) after the last one finishes -- exactly the lifetime we want, with no
    sidecar machinery of our own.

    This rule holds the GPU for its whole lifetime, which is why it -- and NOT the per-well client
    rule -- declares gpu=1. See the header note.

    The harness creates the socket file only AFTER the adapter finishes loading, so a client that
    connects successfully is guaranteed to be talking to a ready model. That ordering IS the
    readiness handshake; there is no other one.
    """
    output:
        socket=service(str(_frame_detections_socket("{experiment}"))),
    params:
        device=lambda wc: str(config.get("frame_detections", {}).get("device", "cuda")),
        gdino_repo=lambda wc: str(MODELS_DIR / "GroundingDINO"),
        gdino_config=lambda wc: str(MODELS_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
        gdino_weights=lambda wc: str(MODELS_DIR / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"),
    resources:
        gpu=1,
    shell:
        """
        {RUN} -m data_pipeline.model_servers.harness \
          --adapter grounding_dino \
          --socket-path "{output.socket}" \
          --adapter-arg gdino_repo_dir="{params.gdino_repo}" \
          --adapter-arg gdino_config="{params.gdino_config}" \
          --adapter-arg gdino_weights="{params.gdino_weights}" \
          --adapter-arg device="{params.device}"
        """


rule frame_detections_per_well_served:
    """Per-well frame_detections via the resident GroundingDINO server (thin socket client).

    Same inputs, same output shard, same DAG position as frame_detections_per_well -- the only
    difference is that inference happens in the resident service process instead of here. The
    client sends PATHS and blocks until the server has written the output, so Snakemake's
    file-exists contract is preserved unchanged.

    Deliberately declares NO gpu resource: this process never touches the GPU. See the header.
    """
    input:
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        socket=lambda wc: str(_frame_detections_socket(wc.experiment)),
    output:
        detections=str(_frame_detections_artifact(
            "{experiment}",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    params:
        detector_model_id=lambda wc: str(
            config.get("frame_detections", {}).get("detector_model_id", "groundingdino_swint_ogc")
        ),
    shell:
        """
        {RUN} -m data_pipeline.model_servers.client \
          --socket-path "{input.socket}" \
          --payload-json '{{"frame_inventory_csv": "{input.frame_inventory}", "output_csv": "{output.detections}", "detector_model_id": "{params.detector_model_id}"}}'
        """


rule frame_detections_per_well:
    """Run GroundingDINO detection over a validated per-well frame_inventory shard.

    One job per well, loading the model in-process. This is the DEFAULT path; set
    frame_detections.use_model_server to route through the resident server instead
    (frame_detections_per_well_served), which loads the model once per run rather than once per
    well. Both write the identical shard -- proven cell-for-cell equivalent on 3 real wells.
    """
    input:
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        detections=str(_frame_detections_artifact(
            "{experiment}",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    params:
        device=lambda wc: str(config.get("frame_detections", {}).get("device", "cuda")),
        gdino_repo=lambda wc: str(MODELS_DIR / "GroundingDINO"),
        gdino_config=lambda wc: str(MODELS_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
        gdino_weights=lambda wc: str(MODELS_DIR / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"),
    # gpu=1: this process itself loads GroundingDINO onto the GPU and holds that memory for the
    # job's duration. Caps concurrent GPU jobs to 1 (pass --resources gpu=1 at the CLI to enforce
    # it; see rules/frame_masks.smk for the fuller note). If a future model-server design puts the
    # model in a resident process instead, do NOT copy this onto the per-well client rule — the
    # client no longer holds GPU memory itself, and double-claiming the slot deadlocks scheduling.
    resources:
        gpu=1,
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks frame-detections \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-csv "{output.detections}" \
          --gdino-repo-dir "{params.gdino_repo}" \
          --gdino-config "{params.gdino_config}" \
          --gdino-weights "{params.gdino_weights}" \
          --device "{params.device}"
        """


# Both frame_detections_per_well and frame_detections_per_well_served produce the same per-well
# shard, which is ambiguous to the DAG resolver. ruleorder picks the winner from the config gate;
# the losing rule stays defined but is never selected. (Defining only one of them conditionally
# would also work, but keeping both defined means `snakemake --list` and the test suite can see
# the served rule regardless of the current toggle.)
if FRAME_DETECTIONS_SERVED:
    ruleorder: frame_detections_per_well_served > frame_detections_per_well
else:
    ruleorder: frame_detections_per_well > frame_detections_per_well_served


rule validate_frame_detections_for_well:
    """Validate the per-well frame_detections shard against its frame_inventory; write the .validated sentinel.

    Symmetric with validate_frame_masks_for_well. The merge (collect_well_shard_paths) only picks up
    shards that carry a .validated sentinel, so this rule is what makes a built frame_detections shard
    eligible for the merged table. (frame_masks_per_well consumes the per-well detection shard
    directly, so it does not require this sentinel — only the merge does.)
    """
    input:
        detections=str(_frame_detections_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}",
        )),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated=str(_frame_detections_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-detections \
          --input-csv "{input.detections}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_frame_detections:
    """Row-stack per-well frame_detections shards into the experiment-level merged table."""
    input:
        per_well=_frame_detections_artifacts_for_run,
        # Wait on each shard's .validated sentinel too (written by validate_frame_detections_for_well),
        # matching the other merge rules — collect_well_shard_paths only collects validated shards.
        per_well_validated=_frame_detections_validated_for_run,
    output:
        merged=str(_frame_detections_artifact(
            "{experiment}",
            path_mode=PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.object_extraction.detection.frame_detections_contract import REQUIRED_FRAME_DETECTIONS_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'frame_detections', 'frame_detections', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=list(REQUIRED_FRAME_DETECTIONS_COLUMNS), sort_columns=['experiment_id', 'well_id', 'time_index'])
"
        """
