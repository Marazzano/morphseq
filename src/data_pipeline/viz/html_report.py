"""Renderer-agnostic page assembler for image artifacts (HTML + PDF).

Takes labeled groups of image paths (report PNGs, plain images, click-throughs) and emits
ONE self-contained page — either a portable HTML file (every image base64-embedded, no
external assets) or a multi-page PDF (one image per page, easy to open on the cluster). It
has NO opinion about pipeline stages, steps, or the path registry: it is the page-layout
analog of the "dumb renderer" layer, one level below a step's report.py.

The API is flat — you never hold intermediate handles. Every ``add_*`` call names its
``section`` (and optional ``subsection``) by string; the report finds-or-creates that bucket,
preserving first-seen order::

    report = HtmlReport("Feature QC", subtitle="20250912")
    report.add_image(png1, section="feature_extraction", subsection="mask_geometry")
    report.add_image(png2, section="feature_extraction", subsection="mask_geometry")
    report.add_note("stub — no artifacts", section="feature_extraction", subsection="curvature")
    report.add_image(x, section="gallery")            # no subsection -> flat strip
    report.write(out_dir / "index.html")              # writes index.html + index.pdf by default
    # report.write(..., pdf=False)                    # HTML only
    # report.write_pdf(out_dir / "x.pdf")             # PDF only (multi-page, one image per page)

A section without subsections renders as a flat click-through strip. The HTML page has a
sticky auto-built nav over the subsection anchors.

HTML emit is stdlib-only. ``write_pdf`` imports Pillow lazily, so importing this module never
pulls a graphics stack.

The CSS + card/error/stub states are lifted from
results/mcolon/20260702_qc_reporting_v1/build_review_index.py, the first caller.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass, field
from pathlib import Path

_STYLE = """
body{font-family:system-ui,sans-serif;margin:0;background:#f5f1e9;color:#111}
header{position:sticky;top:0;background:#fff;padding:14px 24px;box-shadow:0 1px 4px #0002;z-index:9}
header h1{margin:0 0 6px;font-size:18px} nav{font-size:13px} nav a{margin-right:2px;color:#2166AC}
section{padding:8px 24px 32px} h2{border-bottom:2px solid #2166AC;padding-bottom:4px}
h3{margin-top:28px;color:#444} figure{margin:12px 0;background:#fff;padding:10px;border-radius:6px;
  box-shadow:0 1px 3px #0002;display:inline-block;max-width:100%}
figure img{max-width:100%;height:auto;display:block} figcaption{font-size:12px;color:#666;margin-top:6px}
pre.err{background:#fff4f4;color:#B2182B;padding:12px;border-radius:6px;overflow-x:auto;font-size:12px}
p.stub{color:#999;font-style:italic;font-size:13px}
p.missing{color:#B2182B;font-size:13px}
"""


@dataclass
class _Item:
    """One entry: an image, an error block, or a stub note. Exactly one of ``path`` /
    ``error`` / ``note`` is set. Kept structured (not pre-rendered HTML) so the HTML and PDF
    emitters read the same source of truth."""

    path: Path | None = None
    caption: str | None = None
    error: str | None = None
    note: str | None = None

    @property
    def label(self) -> str:
        if self.caption is not None:
            return self.caption
        return self.path.name if self.path is not None else ""

    @property
    def image_exists(self) -> bool:
        return self.path is not None and Path(self.path).exists()


@dataclass
class _Subsection:
    """A titled group of items under a section — becomes an <h3> anchor + nav entry. The empty
    string is the section's flat bucket (no <h3>, no nav entry)."""

    title: str
    items: list[_Item] = field(default_factory=list)


@dataclass
class _Section:
    """A top-level <h2> band holding ordered subsections (the "" one is the flat bucket)."""

    title: str
    subsections: dict[str, _Subsection] = field(default_factory=dict)

    def bucket(self, subsection: str) -> _Subsection:
        """Find-or-create the named subsection ("" = the flat bucket), preserving order."""
        if subsection not in self.subsections:
            self.subsections[subsection] = _Subsection(subsection)
        return self.subsections[subsection]


def _anchor(title: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in title.lower()).strip("-")


def _item_html(item: _Item) -> str:
    label = html.escape(item.label)
    if item.error is not None:
        return f'<pre class="err">{html.escape(item.error)}</pre>'
    if item.note is not None:
        return f'<p class="stub">{html.escape(item.note)}</p>'
    if not item.image_exists:
        return f'<p class="missing">image missing: {label}</p>'
    b64 = base64.b64encode(Path(item.path).read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{label}">'
            f'<figcaption>{label}</figcaption></figure>')


class HtmlReport:
    """Assemble a self-contained page from image artifacts via a flat, path-style API.

    Every ``add_*`` names its ``section`` (and optional ``subsection``) by string; the report
    finds-or-creates that bucket in first-seen order — you never hold a Section/Subsection
    handle. ``render()`` / ``write(path)`` emit self-contained HTML (plus a sibling PDF by
    default); ``write_pdf(path)`` emits a multi-page PDF (one image per page).
    """

    def __init__(self, title: str, *, subtitle: str | None = None) -> None:
        self.title = title
        self.subtitle = subtitle
        self._sections: dict[str, _Section] = {}

    # ── building (flat, find-or-create) ──────────────────────────────────────────────────
    def _bucket(self, section: str, subsection: str) -> _Subsection:
        if section not in self._sections:
            self._sections[section] = _Section(section)
        return self._sections[section].bucket(subsection)

    def add_image(self, path: Path, *, section: str, subsection: str = "",
                  caption: str | None = None) -> "HtmlReport":
        """Add one image to ``section`` (and optional ``subsection``). A missing file renders an
        'image missing' note in HTML and is skipped in the PDF. Returns self."""
        self._bucket(section, subsection).items.append(_Item(path=Path(path), caption=caption))
        return self

    def add_images(self, paths, *, section: str, subsection: str = "",
                   caption: str | None = None) -> "HtmlReport":
        """Add several images in order. ``caption`` (a str or None) applies to all."""
        for p in paths:
            self.add_image(p, section=section, subsection=subsection, caption=caption)
        return self

    def add_error(self, message: str, *, section: str, subsection: str = "") -> "HtmlReport":
        """Add an error block (e.g. a producer traceback) instead of an image."""
        self._bucket(section, subsection).items.append(_Item(error=message))
        return self

    def add_note(self, message: str, *, section: str, subsection: str = "") -> "HtmlReport":
        """Add an italic stub note (e.g. 'stub — no artifacts')."""
        self._bucket(section, subsection).items.append(_Item(note=message))
        return self

    # ── rendering ────────────────────────────────────────────────────────────────────────
    def _nav(self) -> str:
        links = [
            f'<a href="#{_anchor(sub.title)}">{html.escape(sub.title)}</a>'
            for section in self._sections.values()
            for sub in section.subsections.values()
            if sub.title  # the "" flat bucket has no nav entry
        ]
        return " · ".join(links)

    def _section_html(self, section: _Section) -> str:
        parts = []
        for sub in section.subsections.values():
            body = "".join(_item_html(it) for it in sub.items)
            if sub.title:  # titled subsection -> <h3> anchor
                body = body or '<p class="stub">no artifacts</p>'
                parts.append(f'<h3 id="{_anchor(sub.title)}">{html.escape(sub.title)}</h3>{body}')
            else:  # flat bucket -> items straight under the section
                parts.append(body)
        return f'<section><h2>{html.escape(section.title)}</h2>{"".join(parts)}</section>'

    def render(self) -> str:
        subtitle = f' — {html.escape(self.subtitle)}' if self.subtitle else ""
        header = f'<header><h1>{html.escape(self.title)}{subtitle}</h1><nav>{self._nav()}</nav></header>'
        body = header + "".join(self._section_html(s) for s in self._sections.values())
        return (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>{html.escape(self.title)}</title><style>{_STYLE}</style></head>"
                f"<body>{body}</body></html>")

    def _all_items(self):
        for section in self._sections.values():
            for sub in section.subsections.values():
                yield from sub.items

    # ── writing ──────────────────────────────────────────────────────────────────────────
    def write(self, output_path: Path, *, pdf: bool = True) -> list[Path]:
        """Write the report. By default emits BOTH the self-contained HTML page and a sibling
        ``.pdf`` (same stem), returning every path written (HTML first).

        ``output_path`` names the HTML file (e.g. ``index.html``); the PDF lands beside it as
        ``index.pdf``. Pass ``pdf=False`` for HTML only (e.g. a report with no images — the PDF
        is an image contact sheet and needs at least one page).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render())
        written = [output_path]
        if pdf:
            written.append(self.write_pdf(output_path.with_suffix(".pdf")))
        return written

    def write_html(self, output_path: Path) -> Path:
        """Write only the self-contained HTML page (no PDF)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render())
        return output_path

    def write_pdf(self, output_path: Path, *, dpi: int = 150) -> Path:
        """Write a multi-page PDF — one page per embedded image, in document order.

        The PDF is an image contact sheet, not a reflow of the HTML: each report figure becomes
        one page (RGB, white background), so it opens cleanly in any PDF viewer on the cluster.
        Error/stub items and missing images produce no page (HTML-only affordances). Requires
        Pillow; raises RuntimeError with the fix if unavailable, ValueError if no image pages.
        """
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "write_pdf needs Pillow (PIL). Install it, or use write(..., pdf=False) for HTML."
            ) from exc

        pages: list["Image.Image"] = []
        for item in self._all_items():
            if not item.image_exists:
                continue
            img = Image.open(Path(item.path))
            if img.mode != "RGB":  # PDF pages must be RGB (flatten alpha over white)
                background = Image.new("RGB", img.size, "white")
                background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
                img = background
            pages.append(img)

        if not pages:
            raise ValueError("No embeddable images — nothing to write to PDF.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pages[0].save(
            output_path, "PDF", resolution=float(dpi),
            save_all=True, append_images=pages[1:],
        )
        return output_path
