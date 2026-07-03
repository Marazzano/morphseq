"""Stage-level report rollup — one HTML+PDF page per pipeline stage.

A stage rollup gathers every per-step report PNG produced under one stage
(``physical_embryo_registry_report``, ``mask_geometry_report``, …) onto a single reviewable
page via :class:`data_pipeline.viz.HtmlReport`. It is a coarser terminal tier on top of the
per-step reports (see viz/report_world.md): it embeds the PNGs those reports already wrote —
it renders nothing itself and consumes nothing downstream.

The set of per-step reports per stage is DISCOVERED from the path registry
(:func:`report_steps_for_stage`), so a new ``*_report`` step is picked up without editing this
module or the rollup rule. Rollup steps themselves (``*_rollup_report``) are excluded so a
rollup never tries to embed its own output.

This module is pure layout glue over ``HtmlReport`` — no matplotlib / pandas.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PIPELINE_STEPS,
    artifact_path,
    known_artifacts,
)
from data_pipeline.viz import HtmlReport

_ROLLUP_SUFFIX = "_rollup_report"
_REPORT_SUFFIX = "_report"


def report_steps_for_stage(stage: str) -> list[str]:
    """Return every per-step ``*_report`` step registered under ``stage``, in registry order.

    Excludes rollup steps (``*_rollup_report``) so a stage rollup never embeds itself.
    """
    return [
        step
        for step, row in PIPELINE_STEPS.items()
        if row["stage"] == stage
        and step.endswith(_REPORT_SUFFIX)
        and not step.endswith(_ROLLUP_SUFFIX)
    ]


def _product_of(report_step: str) -> str:
    """``mask_geometry_report`` -> ``mask_geometry`` (the subsection title for its PNGs)."""
    return report_step[: -len(_REPORT_SUFFIX)]


def stage_report_pngs(data_root, stage: str, experiment_id: str) -> list[tuple[str, list[Path]]]:
    """Resolve, per per-step report under ``stage``, its ``(product, [png_path, …])``.

    Paths come straight from the registry (``artifact_path(..., path_mode="experiment")``),
    so they cannot drift from what the report rules write. Non-existent PNGs are dropped (a
    report that produced no artifact contributes an empty list, rendered as a stub note).
    """
    groups: list[tuple[str, list[Path]]] = []
    for step in report_steps_for_stage(stage):
        pngs = [
            Path(artifact_path(data_root, step, artifact, experiment_id, path_mode="experiment"))
            for artifact in known_artifacts(step)
        ]
        groups.append((_product_of(step), [p for p in pngs if p.exists()]))
    return groups


def build_stage_rollup_report(
    data_root,
    stage: str,
    experiment_id: str,
    *,
    output_html: Path,
) -> list[Path]:
    """Build one stage's rollup page (HTML + sibling PDF) and return the paths written.

    One section (the stage), one subsection per per-step report (its product), embedding that
    report's PNGs. A report with no PNG on disk renders a 'no report artifacts' stub. Returns
    ``[html_path, pdf_path]`` (see :meth:`HtmlReport.write`).
    """
    report = HtmlReport(f"{stage} — report", subtitle=str(experiment_id))
    for product, pngs in stage_report_pngs(data_root, stage, experiment_id):
        if pngs:
            report.add_images(pngs, section=stage, subsection=product)
        else:
            report.add_note("no report artifacts", section=stage, subsection=product)
    return report.write(Path(output_html))
