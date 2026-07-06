# `html_report` — assembling image artifacts into an HTML + PDF page

`data_pipeline.viz.html_report.HtmlReport` lays a set of image artifacts (report PNGs, plain
images, click-throughs) into **one self-contained page**: a portable HTML file (every image
base64-embedded — no external assets) and, by default, a sibling multi-page PDF (one image per
page, easy to open on the cluster).

It is a **dumb layout primitive**: it knows nothing about pipeline stages, steps, or the path
registry — it just takes labeled images and arranges them. It is the page-layout analog of the
"dumb renderer" layer in `viz/reporting.py`, one level below a step's `report.py`. Use it
whenever you want to bundle a bunch of PNGs into a reviewable page.

- **Import:** `from data_pipeline.viz import HtmlReport`
- **Deps:** HTML emit is stdlib-only. `write_pdf` imports Pillow lazily, so importing the
  module never pulls a graphics stack.

---

## The API is flat — you never hold handles

Every `add_*` call names its `section` (and optional `subsection`) **by string**. The report
**finds-or-creates** that bucket, so you never juggle intermediate objects:

```python
from data_pipeline.viz import HtmlReport

report = HtmlReport("Feature QC", subtitle="20250912")

report.add_image(png1, section="feature_extraction", subsection="mask_geometry")
report.add_image(png2, section="feature_extraction", subsection="mask_geometry")
report.add_note("stub — no artifacts", section="feature_extraction", subsection="curvature")
report.add_error(traceback_str, section="quality_control", subsection="focus_qc")
report.add_image(x, section="gallery")   # omit subsection -> flat strip, no <h3>

report.write(out_dir / "index.html")     # writes index.html AND index.pdf
```

### Layout model

```
HtmlReport (title, subtitle)
└── section   → <h2> band          (e.g. a pipeline stage)
    └── subsection → <h3> + nav anchor   (e.g. a product)
        └── item   → an image / error block / stub note
```

- Omit `subsection` (or pass `""`) to drop items **straight under the section** as a flat
  click-through strip (no `<h3>`, no nav entry).
- The HTML page gets a **sticky auto-built nav** linking every titled subsection.

---

## Ordering — it's just add-order

Sections and subsections render **in the order you first add them** (insertion order):

- the first `add_*` that mentions a new `section=` / `subsection=` fixes its position;
- re-adding to an existing bucket **appends** the item there and does **not** reorder anything.

So you control order purely by the sequence of calls — there is no separate ordering knob. If
you drive from a list, keep the list in the order you want on the page:

```python
# REPORTS is in pipeline order -> the page comes out in pipeline order.
for stage, product in REPORTS:
    pngs, err = run_report(product)
    if err:
        report.add_error(err, section=stage, subsection=product)
    elif not pngs:
        report.add_note("stub — no artifacts", section=stage, subsection=product)
    else:
        report.add_images(pngs, section=stage, subsection=product)
```

---

## Methods

### Building

| Method | Adds |
|---|---|
| `add_image(path, *, section, subsection="", caption=None)` | one embedded image; `caption` defaults to the file name |
| `add_images(paths, *, section, subsection="", caption=None)` | several images in order; `caption` (str or None) applies to all |
| `add_error(message, *, section, subsection="")` | a red error block (e.g. a producer traceback) instead of an image |
| `add_note(message, *, section, subsection="")` | an italic stub note (e.g. "stub — no artifacts") |

All builders return `self`, so they chain if you like:
`report.add_image(a, section="s").add_image(b, section="s")`.

**Item states (how each renders):**

- a real image → an embedded `<figure>` with a caption;
- a **missing** image path → an "image missing: …" note in the HTML, and **no page** in the PDF;
- an **error** → a `<pre class="err">` block (HTML only, no PDF page);
- a **note** → an italic stub line (HTML only, no PDF page).

All captions, titles, notes, and error text are **HTML-escaped**, so arbitrary labels /
tracebacks are safe to pass in.

### Writing

| Method | Writes | Returns |
|---|---|---|
| `write(path, *, pdf=True)` | the HTML page **and** a sibling `.pdf` (same stem) by default | `list[Path]` written, HTML first |
| `write_html(path)` | only the self-contained HTML page | `Path` |
| `write_pdf(path, *, dpi=150)` | only the multi-page PDF (one page per image, RGB, white bg) | `Path` |
| `render()` | — (returns the full `<!doctype html>` string, writes nothing) | `str` |

Notes:

- **`write()` emits both by default.** `path` names the HTML file (e.g. `index.html`); the PDF
  lands beside it as `index.pdf`. Pass `pdf=False` for HTML only.
- The **PDF is an image contact sheet**, not a reflow of the HTML: each embedded image becomes
  one page. Error/stub items and missing images produce no page. `write_pdf` raises
  `ValueError` if there are no images to page, and `RuntimeError` (with the fix) if Pillow is
  unavailable.

---

## Examples

### 1. Multi-stage review page (the QC/feature review driver)

```python
report = HtmlReport("QC / feature report review", subtitle="20250912")
for stage, product in REPORTS:            # pipeline order
    pngs, err = run_report(product)
    if err:
        report.add_error(err, section=stage, subsection=product)
    elif not pngs:
        report.add_note("stub — no artifacts", section=stage, subsection=product)
    else:
        report.add_images(pngs, section=stage, subsection=product)

for path in report.write(out_dir / "index.html"):   # -> index.html + index.pdf
    print(f"wrote {path}")
```

See `results/mcolon/20260702_qc_reporting_v1/build_review_index.py` for the live driver.

### 2. A flat click-through gallery (no subsections)

```python
report = HtmlReport("Embryo gallery")
for png in sorted(gallery_dir.glob("*.png")):
    report.add_image(png, section="all embryos")     # one flat section
report.write(out_dir / "gallery.html")               # gallery.html + gallery.pdf
```

### 3. HTML only (e.g. a report with no images to page)

```python
report = HtmlReport("Notes")
report.add_note("nothing to show yet", section="status")
report.write(out_dir / "notes.html", pdf=False)      # HTML only
```

### 4. PDF only

```python
report.write_pdf(out_dir / "review.pdf", dpi=200)
```
