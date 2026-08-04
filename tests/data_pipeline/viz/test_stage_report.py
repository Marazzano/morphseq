from __future__ import annotations

from pathlib import Path

from PIL import Image

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PIPELINE_STEPS,
    artifact_path,
    known_artifacts,
)
from data_pipeline.viz.stage_report import (
    _product_of,
    build_stage_rollup_report,
    report_steps_for_stage,
    stage_report_pngs,
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(30, 60, 90)).save(path)


def test_report_steps_discovers_from_registry_in_order() -> None:
    """The report list is DERIVED from PIPELINE_STEPS, in registry order.

    Asserted against the registry rather than a frozen literal: this test is about the discovery
    and ordering rule, not about which QC reports happen to exist today. A hardcoded list here goes
    stale (and fails) the moment a real report step is registered.
    """
    expected = [
        step
        for step, meta in PIPELINE_STEPS.items()
        if meta["stage"] == "quality_control"
        and step.endswith("_report")
        and not step.endswith("_rollup_report")
    ]
    assert expected  # guard: the registry really does declare QC reports
    assert report_steps_for_stage("quality_control") == expected


def test_report_steps_excludes_rollup_reports() -> None:
    # every stage that has a *_rollup_report row must never surface it as a per-step report
    rollups = [s for s in PIPELINE_STEPS if s.endswith("_rollup_report")]
    assert rollups  # guard: registry actually has rollups to exclude

    for rollup in rollups:
        stage = PIPELINE_STEPS[rollup]["stage"]
        assert rollup not in report_steps_for_stage(stage)


def test_report_steps_only_returns_report_steps() -> None:
    for step in report_steps_for_stage("feature_extraction"):
        assert step.endswith("_report")
        assert not step.endswith("_rollup_report")


def test_report_steps_unknown_stage_is_empty() -> None:
    assert report_steps_for_stage("not_a_real_stage") == []


def test_stage_report_pngs_drops_nonexistent_paths(tmp_path: Path) -> None:
    # no files on disk -> every group present but with empty png lists
    groups = stage_report_pngs(tmp_path, "feature_extraction", "20250912")

    # Products are DERIVED from the registry's feature_extraction report steps, not hardcoded —
    # a frozen literal here goes stale as soon as another report step is registered.
    products = [product for product, _ in groups]
    assert products == [_product_of(s) for s in report_steps_for_stage("feature_extraction")]
    assert all(pngs == [] for _, pngs in groups)


def test_stage_report_pngs_resolves_existing_registry_paths(tmp_path: Path) -> None:
    step = "mask_geometry_report"
    written = []
    for artifact in known_artifacts(step):
        p = Path(artifact_path(tmp_path, step, artifact, "20250912", path_mode="experiment"))
        _write_png(p)
        written.append(p)

    groups = stage_report_pngs(tmp_path, "feature_extraction", "20250912")

    # Only THIS step's PNGs were written, so its group carries them and every sibling report step
    # in the stage contributes an empty group. Keyed by product, not by a hardcoded group count.
    by_product = dict(groups)
    assert sorted(by_product[_product_of(step)]) == sorted(written)
    for other in report_steps_for_stage("feature_extraction"):
        if other != step:
            assert by_product[_product_of(other)] == []


def test_build_stage_rollup_writes_html_and_pdf_with_images(tmp_path: Path) -> None:
    step = "mask_geometry_report"
    for artifact in known_artifacts(step):
        _write_png(Path(artifact_path(tmp_path, step, artifact, "20250912", path_mode="experiment")))

    out_html = tmp_path / "rollup.html"
    written = build_stage_rollup_report(
        tmp_path, "feature_extraction", "20250912", output_html=out_html
    )

    assert written[0] == out_html
    assert out_html.exists()
    pdf = out_html.with_suffix(".pdf")
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"
    assert "mask_geometry" in out_html.read_text()


def test_build_stage_rollup_stubs_reports_with_no_artifacts(tmp_path: Path) -> None:
    # nothing on disk -> HTML-only page (no images => no PDF pages), stub note per product
    out_html = tmp_path / "rollup.html"
    written = build_stage_rollup_report(
        tmp_path, "feature_extraction", "20250912", output_html=out_html
    )

    text = out_html.read_text()
    assert "no report artifacts" in text
    assert "mask_geometry" in text
