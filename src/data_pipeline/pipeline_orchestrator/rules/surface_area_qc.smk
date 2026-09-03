"""surface_area_qc quality-control rules.

First feature-derived QC product: one sa_outlier_flag per snip_id. Stage-binned two-sided
outlier flag — area_um2 (from the mask_geometry feature shard) judged against a packaged
wildtype p5/p95 reference interpolated at predicted_stage_hpf (from the stage_predictions
shard). Consumes the validated snip_inventory (the universe) and the per-well
physical_embryo_registry shard (identity verifier). Per-well build -> validate -> merge,
mirroring the feature-product template.
"""

SURFACE_AREA_QC_STEP = "surface_area_qc"


def _saqc_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(SURFACE_AREA_QC_STEP, "surface_area_qc", experiment, path_mode=path_mode, well_id=well_id)

def _saqc_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(SURFACE_AREA_QC_STEP, "surface_area_qc", experiment, path_mode=path_mode, well_id=well_id)

def _saqc_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "legacy_default_snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "legacy_default_snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_frame_inventory(experiment, *, well_id):
    return rule_artifact("frame_inventory", "inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_mask_geometry(experiment, *, well_id):
    return rule_artifact("mask_geometry", "mask_geometry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_mask_geometry_validated(experiment, *, well_id):
    return rule_validated("mask_geometry", "mask_geometry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_stage_predictions(experiment, *, well_id):
    return rule_artifact("stage_predictions", "stage_predictions", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_stage_predictions_validated(experiment, *, well_id):
    return rule_validated("stage_predictions", "stage_predictions", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _saqc_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, SURFACE_AREA_QC_STEP, "surface_area_qc", wc.experiment, wells_for_experiment(wc))


rule build_surface_area_qc_for_well:
    """Compute the per-well surface_area_qc shard from mask_geometry + stage_predictions + universe."""
    input:
        snip_inventory=str(_saqc_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_saqc_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_saqc_frame_inventory("{experiment}", well_id="{well_id}")),
        mask_geometry=str(_saqc_mask_geometry("{experiment}", well_id="{well_id}")),
        mask_geometry_validated=str(_saqc_mask_geometry_validated("{experiment}", well_id="{well_id}")),
        stage_predictions=str(_saqc_stage_predictions("{experiment}", well_id="{well_id}")),
        stage_predictions_validated=str(_saqc_stage_predictions_validated("{experiment}", well_id="{well_id}")),
        physical_embryo_registry=str(_saqc_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_saqc_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        surface_area_qc=str(_saqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    params:
        # The MERGED config (base + runtime overlay), so a per-experiment
        # quality_control.surface_area_qc block can retune the band. params: not input: —
        # read lazily at execution time, never tracked or locked (see CONFIG_YAML in the Snakefile).
        config_yaml=str(CONFIG_YAML),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks surface-area-qc \
          --mask-geometry-csv "{input.mask_geometry}" \
          --stage-predictions-csv "{input.stage_predictions}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.surface_area_qc}" \
          --config-yaml "{params.config_yaml}"
        """


rule validate_surface_area_qc_for_well:
    """Validate the per-well surface_area_qc shard (spine + flag, registry as verifier) and write .validated."""
    input:
        surface_area_qc=str(_saqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_saqc_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_saqc_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-surface-area-qc \
          --input-csv "{input.surface_area_qc}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_surface_area_qc:
    """Row-stack per-well surface_area_qc shards into the experiment-level merged table."""
    input:
        per_well=_saqc_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_saqc_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_saqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.quality_control.surface_area_qc.contract import SURFACE_AREA_QC_TABLE_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'surface_area_qc', 'surface_area_qc', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=SURFACE_AREA_QC_TABLE_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """


rule surface_area_qc_report:
    """TERMINAL: area-vs-stage scatter against the reference band + stage-banded quartile gallery.
    Consumed by nothing — only the `reports` aggregate target requests this. See viz/report_world.md."""
    input:
        surface_area_qc=str(_saqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
        mask_geometry=str(rule_artifact("mask_geometry", "mask_geometry", "{experiment}", path_mode=PATH_MODE_MERGED)),
        stage_predictions=str(rule_artifact("stage_predictions", "stage_predictions", "{experiment}", path_mode=PATH_MODE_MERGED)),
        snip_inventory=str(rule_artifact("snip_inventory", "snip_inventory", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        vs_stage_png=str(rule_artifact("surface_area_qc_report", "vs_stage_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        gallery_png=str(rule_artifact("surface_area_qc_report", "gallery_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks surface-area-qc-report \
          --surface-area-qc-csv "{input.surface_area_qc}" \
          --mask-geometry-csv "{input.mask_geometry}" \
          --stage-predictions-csv "{input.stage_predictions}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --output-root "{DATA_ROOT}" \
          --output-vs-stage-png "{output.vs_stage_png}" \
          --output-gallery-png "{output.gallery_png}"
        """
