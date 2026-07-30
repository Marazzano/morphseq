"""Physical-embryo-registry product-family rules.

The identity-origination boundary: the one place track_id -> physical_embryo_id is resolved.
Consumes the per-well ``frame_masks`` shard (the DETECTED set, upstream of the valid/invalid QC
split) and mints ONE registry row per distinct ``(well_id, track_id)``. PER_WELL_THEN_MERGE: a
per-well shard is built + validated, then the shards are concatenated into the experiment-level
table whose validator enforces GLOBAL physical_embryo_id uniqueness.

Validation fires at BOTH grains (per-well and merged), each emitting its own ``.validated``
sentinel — mirroring frame_inventory's build+validate+merge template (the conformant pattern),
not frame_masks' leaner inline-shell merge.

DOCTRINE: no raw artifact-path strings (all via paths.py); the merge input set is resolved by the
planning-time, disk-blind ``run_well_shard_paths`` (well_runner); the merge concat-with-uniqueness
lives in the product's own Stage-2 ``merge_physical_embryo_registry`` (called by the task verb),
which is where the global-uniqueness law belongs.
"""

PHYSICAL_EMBRYO_REGISTRY_STEP = "physical_embryo_registry"
PHYSICAL_EMBRYO_REGISTRY_ARTIFACT = "physical_embryo_registry"

# The upstream source: frame_masks owns this product/path. We name the per-well shard locally
# (rather than reaching into frame_masks.smk's helper) so this file's dependency on the upstream
# location reads in one place.
FRAME_MASKS_STEP = "frame_masks"


def _frame_masks_per_well_csv(experiment: str, *, well_id: str):
    return rule_artifact(FRAME_MASKS_STEP, "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _frame_masks_per_well_validated(experiment: str, *, well_id: str):
    return rule_validated(FRAME_MASKS_STEP, "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _registry_collection_provenance(experiment: str):
    # The DECLARED collection fact (experiment-grain). OWNS the n_sources merge count that selects
    # the EmbryoMergePolicy; every experiment declares one (a single experiment is a collection of
    # ONE source), so the count is always available at its source.
    return rule_artifact("collection_provenance", "provenance", experiment, path_mode=PATH_MODE_EXPERIMENT)


def _registry_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(PHYSICAL_EMBRYO_REGISTRY_STEP, PHYSICAL_EMBRYO_REGISTRY_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _registry_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(PHYSICAL_EMBRYO_REGISTRY_STEP, PHYSICAL_EMBRYO_REGISTRY_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _registry_shards_for_run(wc):
    # Planning-time, disk-blind: the merge declares its deps on the RUN wells' shard paths.
    return run_well_shard_paths(DATA_ROOT, PHYSICAL_EMBRYO_REGISTRY_STEP, PHYSICAL_EMBRYO_REGISTRY_ARTIFACT, wc.experiment, wells_for_experiment(wc))

def _registry_validated_for_run(wc):
    return [_registry_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id) for well_id in wells_for_experiment(wc)]


rule build_physical_embryo_registry_for_well:
    """Mint one per-well registry shard from the per-well frame_masks shard + collection provenance.

    One job per well (cheap CPU: drop-duplicates + the track_id -> physical_embryo_id mint chain).
    The task verb validates the shard before writing.
    """
    input:
        # Depend on the VALIDATED per-well frame_masks shard (mirroring frame_masks_per_well's
        # dependence on frame_inventory + its validated sentinel). The frame_masks validate rule
        # writes this sentinel; the registry is built only from a frame_masks shard that passed
        # its contract.
        frame_masks=str(_frame_masks_per_well_csv("{experiment}", well_id="{well_id}")),
        frame_masks_validated=str(_frame_masks_per_well_validated("{experiment}", well_id="{well_id}")),
        # The collection provenance artifact OWNS the n_sources merge count (len(sources)) that
        # selects the EmbryoMergePolicy (NORMAL / BRIDGE / FRACTURE). Read from its owner rather
        # than from a per-frame column: n_sources is an experiment-grain constant, so carrying it
        # on every frame row made frame_inventory a courier for a fact it does not own.
        collection_provenance=str(_registry_collection_provenance("{experiment}")),
    output:
        registry=str(_registry_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks build-physical-embryo-registry \
          --frame-masks-csv "{input.frame_masks}" \
          --collection-provenance-json "{input.collection_provenance}" \
          --output-csv "{output.registry}"
        """


rule validate_physical_embryo_registry_for_well:
    input:
        registry=str(_registry_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated=str(_registry_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-physical-embryo-registry \
          --input-csv "{input.registry}" \
          --output-flag "{output.validated}"
        """


rule merge_physical_embryo_registry:
    """Concat the validated per-well shards; the task verb re-validates GLOBAL uniqueness."""
    input:
        per_well=_registry_shards_for_run,
        per_well_validated=_registry_validated_for_run,
    output:
        merged=str(_registry_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks merge-physical-embryo-registry \
          --inputs {input.per_well} \
          --output-csv "{output.merged}"
        """


rule validate_physical_embryo_registry:
    input:
        merged=str(_registry_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        validated=str(_registry_validated("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-physical-embryo-registry \
          --input-csv "{input.merged}" \
          --output-flag "{output.validated}"
        """


rule physical_embryo_registry_report:
    """TERMINAL: embryos-per-well distribution + plate heatmap + embryos-per-well OVER TIME (death
    proxy). Consumed by nothing — only the `reports` aggregate target requests this. Reads the
    merged frame_masks (sibling object_extraction artifact) for the per-frame presence that the
    animal-grain registry table lacks; both inputs are object_extraction, so this stays a stage-local
    leaf. See viz/report_world.md."""
    input:
        registry=str(_registry_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
        frame_masks=str(_frame_masks_artifact("{experiment}", "frame_masks", path_mode=PATH_MODE_MERGED)),
    output:
        embryos_per_well_png=str(rule_artifact("physical_embryo_registry_report", "embryos_per_well_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        embryos_per_well_plate_png=str(rule_artifact("physical_embryo_registry_report", "embryos_per_well_plate_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        embryos_per_well_over_time_png=str(rule_artifact("physical_embryo_registry_report", "embryos_per_well_over_time_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks physical-embryo-registry-report \
          --physical-embryo-registry-csv "{input.registry}" \
          --frame-masks-csv "{input.frame_masks}" \
          --output-embryos-per-well-png "{output.embryos_per_well_png}" \
          --output-embryos-per-well-plate-png "{output.embryos_per_well_plate_png}" \
          --output-embryos-per-well-over-time-png "{output.embryos_per_well_over_time_png}"
        """
