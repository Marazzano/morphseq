"""Stage-level report rollups — one HTML+PDF page per stage.

Each rule gathers every per-step report PNG produced under one stage onto a single page (via
viz/stage_report.py). TERMINAL: consumed by nothing, requested only by the `reports` aggregate
target. The rule's inputs are that stage's per-step report PNGs (discovered from the registry),
so the rollup rebuilds whenever a child report changes and Snakemake builds the children first.

The step->PNG mapping lives in exactly one place — report_steps_for_stage / stage_report_pngs
in viz/stage_report.py — reused here so the rule body carries no hard-coded step list.
"""

from data_pipeline.viz.stage_report import report_steps_for_stage


def _stage_rollup_pngs(stage, experiment):
    """Every per-step report PNG under `stage` for `experiment`, as rule input strings."""
    paths = []
    for step in report_steps_for_stage(stage):
        for artifact in known_artifacts(step):
            paths.append(str(rule_artifact(step, artifact, experiment, path_mode=PATH_MODE_EXPERIMENT)))
    return paths


def _rollup_outputs(stage):
    step = f"{stage}_rollup_report"
    return {
        "html": str(rule_artifact(step, "index_html", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        "pdf": str(rule_artifact(step, "index_pdf", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    }


rule object_extraction_rollup_report:
    """TERMINAL stage rollup: every object_extraction per-step report PNG on one HTML+PDF page."""
    input:
        lambda wc: _stage_rollup_pngs("object_extraction", wc.experiment),
    output:
        html=_rollup_outputs("object_extraction")["html"],
        pdf=_rollup_outputs("object_extraction")["pdf"],
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks stage-rollup-report \
          --stage object_extraction --data-root "{DATA_ROOT}" --experiment "{wildcards.experiment}" \
          --output-html "{output.html}"
        """


rule feature_extraction_rollup_report:
    """TERMINAL stage rollup: every feature_extraction per-step report PNG on one HTML+PDF page."""
    input:
        lambda wc: _stage_rollup_pngs("feature_extraction", wc.experiment),
    output:
        html=_rollup_outputs("feature_extraction")["html"],
        pdf=_rollup_outputs("feature_extraction")["pdf"],
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks stage-rollup-report \
          --stage feature_extraction --data-root "{DATA_ROOT}" --experiment "{wildcards.experiment}" \
          --output-html "{output.html}"
        """


rule quality_control_rollup_report:
    """TERMINAL stage rollup: every quality_control per-step report PNG on one HTML+PDF page."""
    input:
        lambda wc: _stage_rollup_pngs("quality_control", wc.experiment),
    output:
        html=_rollup_outputs("quality_control")["html"],
        pdf=_rollup_outputs("quality_control")["pdf"],
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks stage-rollup-report \
          --stage quality_control --data-root "{DATA_ROOT}" --experiment "{wildcards.experiment}" \
          --output-html "{output.html}"
        """
