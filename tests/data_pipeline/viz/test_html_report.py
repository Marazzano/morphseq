from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data_pipeline.viz.html_report import HtmlReport


def _png(tmp_path: Path, name: str, color=(200, 40, 40), mode: str = "RGB") -> Path:
    path = tmp_path / name
    Image.new(mode, (24, 24), color=color if mode == "RGB" else None).save(path)
    return path


def test_render_embeds_images_base64_no_external_assets(tmp_path: Path) -> None:
    report = HtmlReport("Feature QC", subtitle="20250912")
    report.add_image(_png(tmp_path, "a.png"), section="feature_extraction", subsection="mask_geometry")

    out = report.render()

    assert "data:image/png;base64," in out
    assert "Feature QC" in out and "20250912" in out
    # self-contained: no <img src> pointing at a file path
    assert 'src="a.png"' not in out


def test_sections_and_subsections_preserve_first_seen_order(tmp_path: Path) -> None:
    report = HtmlReport("Ordered")
    report.add_image(_png(tmp_path, "1.png"), section="alpha", subsection="one")
    report.add_image(_png(tmp_path, "2.png"), section="alpha", subsection="two")
    report.add_image(_png(tmp_path, "3.png"), section="beta", subsection="one")

    out = report.render()

    assert out.index("alpha") < out.index("beta")
    assert out.index(">one<") < out.index(">two<")
    # nav lists subsection anchors in order
    assert out.index('href="#one"') < out.index('href="#two"')


def test_find_or_create_bucket_groups_repeated_calls(tmp_path: Path) -> None:
    report = HtmlReport("Grouping")
    report.add_image(_png(tmp_path, "a.png"), section="s", subsection="sub")
    report.add_image(_png(tmp_path, "b.png"), section="s", subsection="sub")

    # both images land under one <h3>, only one nav anchor
    assert report.render().count('id="sub"') == 1


def test_note_and_error_render_as_stub_and_err_blocks(tmp_path: Path) -> None:
    report = HtmlReport("States")
    report.add_note("no artifacts", section="s", subsection="empty")
    report.add_error("boom: traceback", section="s", subsection="broken")

    out = report.render()

    assert 'class="stub"' in out and "no artifacts" in out
    assert 'class="err"' in out and "boom: traceback" in out


def test_missing_image_renders_missing_note_not_crash(tmp_path: Path) -> None:
    report = HtmlReport("Missing")
    report.add_image(tmp_path / "does_not_exist.png", section="s")

    out = report.render()

    assert 'class="missing"' in out


def test_html_escaping_of_user_strings(tmp_path: Path) -> None:
    report = HtmlReport("A <script> title")
    report.add_error("<b>not html</b>", section="s")

    out = report.render()

    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&lt;b&gt;not html&lt;/b&gt;" in out


def test_write_emits_html_and_sibling_pdf(tmp_path: Path) -> None:
    report = HtmlReport("Both")
    report.add_image(_png(tmp_path, "a.png"), section="s")

    written = report.write(tmp_path / "index.html")

    html_path, pdf_path = written
    assert html_path == tmp_path / "index.html"
    assert pdf_path == tmp_path / "index.pdf"
    assert html_path.exists() and pdf_path.exists()
    assert pdf_path.read_bytes()[:4] == b"%PDF"


def test_write_html_only_skips_pdf(tmp_path: Path) -> None:
    report = HtmlReport("HtmlOnly")
    report.add_image(_png(tmp_path, "a.png"), section="s")

    written = report.write(tmp_path / "index.html", pdf=False)

    assert written == [tmp_path / "index.html"]
    assert not (tmp_path / "index.pdf").exists()


def test_write_pdf_one_page_per_image_flattens_alpha(tmp_path: Path) -> None:
    report = HtmlReport("Pdf")
    report.add_image(_png(tmp_path, "rgb.png"), section="s")
    report.add_image(_png(tmp_path, "rgba.png", mode="RGBA"), section="s")

    pdf_path = report.write_pdf(tmp_path / "out.pdf")

    assert pdf_path.exists()
    assert pdf_path.read_bytes()[:4] == b"%PDF"


def test_write_pdf_with_no_images_raises(tmp_path: Path) -> None:
    report = HtmlReport("Empty")
    report.add_note("nothing here", section="s")

    with pytest.raises(ValueError):
        report.write_pdf(tmp_path / "out.pdf")


def test_pdf_skips_notes_errors_and_missing_images(tmp_path: Path) -> None:
    report = HtmlReport("Mixed")
    report.add_image(_png(tmp_path, "real.png"), section="s")
    report.add_note("stub", section="s")
    report.add_error("err", section="s")
    report.add_image(tmp_path / "gone.png", section="s")

    # only the one real image should page; no crash on the non-image items
    pdf_path = report.write_pdf(tmp_path / "out.pdf")
    assert pdf_path.exists()
