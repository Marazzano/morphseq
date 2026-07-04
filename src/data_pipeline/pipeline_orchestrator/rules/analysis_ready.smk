"""analysis_ready — merged-level fan-in join (OPTIONAL downstream product).

One row per snip_id: the identity spine + every per-snip feature payload + the snip_qc verdict +
the per-well plate_metadata broadcast by well_id. Every input here is already a MERGED,
experiment-grain artifact (or the experiment-grain PLATE_METADATA_CSV), so this is a single rule —
no per-well build/validate/merge chain of its own. snip_qc remains the proven through-line
terminal; analysis_ready is NOT part of `through_line` or `rule all` — it is requested by its own
named target (see below), mirroring how `reports` is opt-in.

See docs/refactors/streamline-snakemake/target/specs/analysis_ready_assemble_plan.md.
"""

ANALYSIS_READY_STEP = "analysis_ready"


def _analysis_ready_artifact(experiment, *, path_mode):
    return rule_artifact(ANALYSIS_READY_STEP, "analysis_ready", experiment, path_mode=path_mode)


rule build_analysis_ready:
    """Fan-in join: merged curvature/stage/mask_geometry/pose/fraction_alive + latents + snip_qc
    + plate_metadata -> one analysis_ready parquet, one row per snip_id."""
    input:
        curvature_metrics=str(rule_artifact("curvature_metrics", "curvature_metrics", "{experiment}", path_mode=PATH_MODE_MERGED)),
        stage_predictions=str(rule_artifact("stage_predictions", "stage_predictions", "{experiment}", path_mode=PATH_MODE_MERGED)),
        mask_geometry=str(rule_artifact("mask_geometry", "mask_geometry", "{experiment}", path_mode=PATH_MODE_MERGED)),
        pose_kinematics=str(rule_artifact("pose_kinematics", "pose_kinematics", "{experiment}", path_mode=PATH_MODE_MERGED)),
        fraction_alive=str(rule_artifact("fraction_alive", "fraction_alive", "{experiment}", path_mode=PATH_MODE_MERGED)),
        latents=str(rule_artifact("latent_embeddings", "latents", "{experiment}", path_mode=PATH_MODE_MERGED)),
        snip_qc=str(rule_artifact("snip_qc", "verdict", "{experiment}", path_mode=PATH_MODE_MERGED)),
        plate_metadata=str(PLATE_METADATA_CSV),
    output:
        analysis_ready=str(_analysis_ready_artifact("{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks analysis-ready \
          --curvature-metrics-csv "{input.curvature_metrics}" \
          --stage-predictions-csv "{input.stage_predictions}" \
          --mask-geometry-csv "{input.mask_geometry}" \
          --pose-kinematics-csv "{input.pose_kinematics}" \
          --fraction-alive-csv "{input.fraction_alive}" \
          --latents-parquet "{input.latents}" \
          --snip-qc-parquet "{input.snip_qc}" \
          --plate-metadata-csv "{input.plate_metadata}" \
          --output-parquet "{output.analysis_ready}"
        """


rule analysis_ready_report:
    """TERMINAL: the one report whose input surface is the whole joined DAG (embeddings + plate
    metadata + predicted_stage_hpf + genotype) — latent PCA/UMAP + survival-over-stage panels.
    Consumed by nothing; requested only by the `reports` aggregate target (auto-discovered via the
    `*_report` step-name convention) or directly: `snakemake analysis_ready_report`."""
    input:
        analysis_ready=str(_analysis_ready_artifact("{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        death_event=str(rule_artifact("death_event", "death_event", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        latent_projection_png=str(rule_artifact("analysis_ready_report", "latent_projection_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        survival_over_stage_png=str(rule_artifact("analysis_ready_report", "survival_over_stage_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        genotype_survival_panel_png=str(rule_artifact("analysis_ready_report", "genotype_survival_panel_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        well_survival_over_stage_png=str(rule_artifact("analysis_ready_report", "well_survival_over_stage_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks analysis-ready-report \
          --analysis-ready-parquet "{input.analysis_ready}" \
          --death-event-csv "{input.death_event}" \
          --output-latent-projection-png "{output.latent_projection_png}" \
          --output-survival-over-stage-png "{output.survival_over_stage_png}" \
          --output-genotype-survival-panel-png "{output.genotype_survival_panel_png}" \
          --output-well-survival-over-stage-png "{output.well_survival_over_stage_png}"
        """
