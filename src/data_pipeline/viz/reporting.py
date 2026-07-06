"""Generic histogram + cutoff-relative quartile-gallery reporting toolkit.

Stage-agnostic: callers supply a DataFrame, a raw metric column, and the cutoff that judges it
(plus which side of the cutoff fails). This module owns rendering only — it has no opinion about
which pipeline stage or QC product it is being used for.

Card aesthetic (fixed-frame image fit, colored metric badge, styled border) is lifted from
results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3f_embryo_portfolio.py in the morphseq tree.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

BAND_NAMES: tuple[str, ...] = ("worst_of_worst", "borderline_fail", "borderline_pass", "clear_pass")

# Fixed cell frame — landscape (long side horizontal). Portrait snip images get rotated 90 deg
# to fill it (see _fit_into_frame), matching the horizontal card layout used here.
FRAME_W, FRAME_H = 420, 220

FAIL_COLOR = "#B2182B"
PASS_COLOR = "#2166AC"
CARD_BG = "#ffffff"
CARD_FAIL_BG = "#fff4f4"
PAGE_BG = "#f5f1e9"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(24, True)
FONT_HEADER = _font(13, True)
FONT_LABEL = _font(11)


def _fail_mask(metric: pd.Series, cutoff: float, fail_direction: Literal["below", "above"]) -> pd.Series:
    return metric < cutoff if fail_direction == "below" else metric > cutoff


def plot_metric_histogram(
    metric: pd.Series,
    cutoff: float,
    *,
    fail_direction: Literal["below", "above"],
    title: str,
    output_path: Path,
    xlabel: str | None = None,
) -> Path:
    """Histogram of the raw metric, with a vertical line at the actual cutoff value. Each bin is
    split into pass/fail sub-counts by the ACTUAL values inside it (not by which side of the
    cutoff the bin's edges fall on), so a bin straddling the cutoff — e.g. a wide auto-sized bin
    containing both a pile of exactly-passing zeros and a few genuinely-failing near-zero values —
    renders as a correctly proportioned pass/fail stack instead of being colored as one verdict.
    """
    values = metric.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    fail_values = _fail_mask(pd.Series(values), cutoff, fail_direction).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    _, edges = np.histogram(values, bins="auto")
    pass_counts, _ = np.histogram(values[~fail_values], bins=edges)
    fail_counts, _ = np.histogram(values[fail_values], bins=edges)
    bin_lefts, widths = edges[:-1], np.diff(edges)

    ax.bar(bin_lefts, pass_counts, width=widths, align="edge", color=PASS_COLOR, edgecolor="white")
    ax.bar(bin_lefts, fail_counts, width=widths, align="edge", bottom=pass_counts,
           color=FAIL_COLOR, edgecolor="white")
    ax.axvline(cutoff, color=FAIL_COLOR, linestyle="--", linewidth=1.5, label=f"cutoff = {cutoff:g}")

    fail = _fail_mask(pd.Series(values), cutoff, fail_direction)
    ax.set_title(f"{title}\n(n_fail={int(fail.sum())}, n_pass={int((~fail).sum())}, n_total={values.size})")
    ax.set_xlabel(xlabel or metric.name or "metric")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _signed_distance_to_cutoff(
    metric: pd.Series, cutoff: float, fail_direction: Literal["below", "above"]
) -> pd.Series:
    return metric - cutoff if fail_direction == "below" else cutoff - metric


def _signed_distance_to_band(metric: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Signed margin to the nearest violated bound of a per-row [lower, upper] band.

    Negative outside the band (magnitude = how far outside); non-negative inside (magnitude =
    distance to the nearer edge). Used when the "cutoff" is itself a function of another column
    (e.g. a stage-interpolated reference band) rather than one fixed scalar.
    """
    margin_to_lower = metric - lower
    margin_to_upper = upper - metric
    return pd.concat([margin_to_lower, margin_to_upper], axis=1).min(axis=1)


def select_quartile_bands(
    df: pd.DataFrame,
    metric_col: str,
    cutoff: float,
    *,
    fail_direction: Literal["below", "above"],
    n_per_band: int = 8,
) -> dict[str, pd.DataFrame]:
    """Split rows into four cutoff-relative bands, ranked by distance from ``cutoff``.

    - worst_of_worst: farthest past the cutoff on the fail side (clear fails)
    - borderline_fail: closest to the cutoff on the fail side
    - borderline_pass: closest to the cutoff on the pass side
    - clear_pass: farthest from the cutoff on the pass side (confident passes — spot-check that
      "obviously fine" snips actually look fine, not just the borderline ones)
    """
    ranked = df.dropna(subset=[metric_col]).copy()
    signed_distance = _signed_distance_to_cutoff(ranked[metric_col], cutoff, fail_direction)
    return _bands_from_signed_distance(ranked, signed_distance, n_per_band)


def select_quartile_bands_vs_band(
    df: pd.DataFrame,
    metric_col: str,
    lower_col: str,
    upper_col: str,
    *,
    n_per_band: int = 8,
) -> dict[str, pd.DataFrame]:
    """Same four bands as :func:`select_quartile_bands`, but judged against a per-row
    ``[lower_col, upper_col]`` band instead of one fixed scalar cutoff (e.g. a stage-interpolated
    reference band, where the pass/fail threshold is itself a function of another column).
    """
    ranked = df.dropna(subset=[metric_col, lower_col, upper_col]).copy()
    signed_distance = _signed_distance_to_band(ranked[metric_col], ranked[lower_col], ranked[upper_col])
    return _bands_from_signed_distance(ranked, signed_distance, n_per_band)


def _bands_from_signed_distance(
    ranked: pd.DataFrame, signed_distance: pd.Series, n_per_band: int
) -> dict[str, pd.DataFrame]:
    ranked = ranked.assign(_signed_distance=signed_distance).sort_values("_signed_distance")
    fail = ranked[ranked["_signed_distance"] < 0]
    passed = ranked[ranked["_signed_distance"] >= 0]

    return {
        "worst_of_worst": fail.head(n_per_band).drop(columns="_signed_distance"),
        "borderline_fail": fail.tail(n_per_band).iloc[::-1].drop(columns="_signed_distance"),
        "borderline_pass": passed.head(n_per_band).drop(columns="_signed_distance"),
        "clear_pass": passed.tail(n_per_band).iloc[::-1].drop(columns="_signed_distance"),
    }


def _fit_into_frame(img: Image.Image, frame_w: int, frame_h: int) -> Image.Image:
    """Snap the image's longest side to the frame's longest side, then scale-to-fit (no crop)."""
    src = img.convert("RGB")
    frame_portrait = frame_h >= frame_w
    img_portrait = src.height >= src.width
    if img_portrait != frame_portrait:
        src = src.rotate(90, expand=True)
    src = ImageOps.contain(src, (frame_w, frame_h), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (frame_w, frame_h), "#111111")
    canvas.paste(src, ((frame_w - src.width) // 2, (frame_h - src.height) // 2))
    return canvas


def _card(
    row: pd.Series,
    metric_col: str,
    image_path_col: str,
    label_col: str,
    card_w: int,
    card_h: int,
    head_h: int,
    *,
    is_fail: bool,
    badge_text: str,
    image_fn=None,
) -> Image.Image:
    """One gallery card. By default the cell image is loaded from ``row[image_path_col]``; pass
    ``image_fn(row) -> PIL.Image`` to supply an already-composited image instead (e.g. a snip with
    a centerline drawn on it). ``image_fn`` returning ``None`` falls back to the path.
    """
    badge_color = FAIL_COLOR if is_fail else PASS_COLOR
    bg = CARD_FAIL_BG if is_fail else CARD_BG

    im = Image.new("RGB", (card_w, card_h), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, card_w - 1, card_h - 1], outline=badge_color, width=2)

    y = 6
    d.text((8, y), str(row[label_col]), fill="#111111", font=FONT_LABEL)
    y += 16
    d.rectangle([8, y, 22, y + 14], fill=badge_color, outline="#222222")
    d.text((26, y - 1), badge_text, fill="#111111", font=FONT_HEADER)
    y = head_h

    box = (6, y, card_w - 6, card_h - 6)
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    try:
        cell = image_fn(row) if image_fn is not None else None
        if cell is None:
            cell = Image.open(Path(str(row[image_path_col])))
        framed = _fit_into_frame(cell, box_w, box_h)
        im.paste(framed, (box[0], box[1]))
    except (FileNotFoundError, OSError):
        d.rectangle(box, fill="#eeeeee", outline="#bbbbbb")
        d.text((box[0] + 10, box[1] + box_h // 2 - 8), "image missing", fill="#777777", font=FONT_HEADER)
    return im


def _blend_line(base: Image.Image, pts, color: tuple[int, int, int], width: int, alpha: float) -> None:
    """Draw a polyline at ``alpha`` opacity by compositing an RGBA overlay onto ``base`` in place.

    PIL's ``ImageDraw.line`` has no per-stroke alpha, so we draw the stroke opaque on a transparent
    layer and alpha-composite it — the only way to get the faint raw-centerline underlay.
    """
    if len(pts) < 2:
        return
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).line(pts, fill=(*color, int(round(alpha * 255))), width=width)
    base.alpha_composite(overlay)


def draw_centerline_on_snip(
    snip_path: Path,
    spline_xy_px: np.ndarray,
    *,
    raw_centerline_xy_px: np.ndarray | None = None,
    contour_mask: np.ndarray | None = None,
    spline_color: tuple[int, int, int] = (231, 76, 60),
    raw_color: tuple[int, int, int] = (255, 214, 10),
    contour_color: tuple[int, int, int] = (46, 204, 113),
    raw_alpha: float = 0.4,
) -> Image.Image:
    """Composite the curvature centerline onto a processed snip for the gallery, in up to 3 layers:

    1. mask outline (``contour_mask``), faint green — what the spine was fit inside;
    2. raw skeleton centerline (``raw_centerline_xy_px``), thin and low-alpha — the jagged geodesic
       path the B-spline was fit *to*;
    3. the B-spline centerline (``spline_xy_px``), bold red — the smoothed curve the curvature metric
       is actually computed from, with white endpoint dots so head/tail ordering is visible.

    Drawing the raw path under the spline lets a reviewer see both the fit and its input, and catch a
    spline that has smoothed away from a scrambled skeleton (a failure curvature summaries hide). All
    arrays are ordered (x, y) in the snip's own pixel frame (the snip mask is a same-size crop, so no
    rescale is needed).
    """
    base = Image.open(snip_path).convert("RGBA")
    draw = ImageDraw.Draw(base)

    if contour_mask is not None:
        from skimage.measure import find_contours  # local: only the overlay path needs it
        for contour in find_contours(np.asarray(contour_mask, dtype=float), level=0.5):
            pts = [(float(x), float(y)) for y, x in contour]  # (row, col) -> (x, y)
            if len(pts) >= 2:
                draw.line(pts, fill=contour_color, width=1)

    if raw_centerline_xy_px is not None and len(raw_centerline_xy_px) >= 2:
        raw_pts = [(float(x), float(y)) for x, y in raw_centerline_xy_px]
        _blend_line(base, raw_pts, raw_color, width=1, alpha=raw_alpha)

    if spline_xy_px is not None and len(spline_xy_px) >= 2:
        pts = [(float(x), float(y)) for x, y in spline_xy_px]
        draw.line(pts, fill=spline_color, width=2)
        for x, y in (pts[0], pts[-1]):  # mark the two endpoints so ordering is visible
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 255, 255), fill=spline_color)

    return base.convert("RGB")


def draw_mask_on_snip(
    snip_path: Path,
    mask: np.ndarray,
    *,
    fill_color: tuple[int, int, int] = (46, 204, 113),
    fill_alpha: float = 0.30,
    contour_color: tuple[int, int, int] = (27, 94, 32),
    contour_width: int = 1,
) -> Image.Image:
    """Composite a binary mask onto a snip image for gallery inspection.

    The mask is shown as a translucent fill plus a thin contour so reviewers can see both the
    occupied region and its boundary in the snip's native pixel frame.
    """
    base = Image.open(snip_path).convert("RGBA")
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.shape != (base.height, base.width):
        raise ValueError(
            f"mask overlay shape {mask_arr.shape!r} does not match snip {snip_path} shape "
            f"{(base.height, base.width)!r}."
        )

    if np.any(mask_arr):
        overlay_arr = np.zeros((base.height, base.width, 4), dtype=np.uint8)
        overlay_arr[mask_arr] = (*fill_color, int(round(fill_alpha * 255)))
        base.alpha_composite(Image.fromarray(overlay_arr, mode="RGBA"))

        from skimage.measure import find_contours  # local: only the overlay path needs it

        draw = ImageDraw.Draw(base)
        for contour in find_contours(mask_arr.astype(float), level=0.5):
            pts = [(float(x), float(y)) for y, x in contour]
            if len(pts) >= 2:
                draw.line(pts, fill=contour_color, width=contour_width)

    return base.convert("RGB")


def _render_band_page(
    bands: dict[str, pd.DataFrame],
    badge_text_fn,
    is_fail_fn,
    *,
    metric_col: str,
    image_path_col: str,
    label_col: str,
    title: str,
    output_path: Path,
    n_per_band: int,
    band_cols: int,
    band_names: tuple[str, ...] = BAND_NAMES,
    image_fn=None,
) -> Path:
    head_h = 40
    card_w, card_h = FRAME_W + 12, head_h + FRAME_H + 12
    row_header_w = 160
    band_title_h = 28
    margin, gap, title_h = 20, 8, 44

    band_cols = max(band_cols, 1)
    band_rows = math.ceil(n_per_band / band_cols)
    strip_h = band_title_h + band_rows * card_h + (band_rows - 1) * gap

    page_w = margin * 2 + row_header_w + band_cols * card_w + (band_cols - 1) * gap
    page_h = title_h + len(band_names) * (strip_h + gap) + margin

    page = Image.new("RGB", (page_w, page_h), PAGE_BG)
    d = ImageDraw.Draw(page)
    d.text((margin, 10), title, fill="#111111", font=FONT_TITLE)

    for band_idx, band_name in enumerate(band_names):
        rows = bands[band_name]
        strip_y0 = title_h + band_idx * (strip_h + gap)
        d.text((margin, strip_y0 + strip_h // 2 - 8), f"{band_name}\n(n={len(rows)})",
               fill="#111111", font=FONT_HEADER)

        for i, (_, row) in enumerate(rows.iterrows()):
            grid_row, grid_col = divmod(i, band_cols)
            x0 = margin + row_header_w + grid_col * (card_w + gap)
            y0 = strip_y0 + band_title_h + grid_row * (card_h + gap)
            card = _card(row, metric_col, image_path_col, label_col, card_w, card_h, head_h,
                         is_fail=is_fail_fn(row), badge_text=badge_text_fn(row), image_fn=image_fn)
            page.paste(card, (x0, y0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path)
    return output_path


def render_quartile_gallery(
    df: pd.DataFrame,
    metric_col: str,
    cutoff: float,
    *,
    fail_direction: Literal["below", "above"],
    image_path_col: str,
    label_col: str,
    title: str,
    output_path: Path,
    n_per_band: int = 8,
    band_cols: int = 4,
) -> Path:
    """Render one page, one horizontal strip per cutoff-relative band (worst_of_worst on top,
    clear_pass on the bottom), judged against a single fixed ``cutoff``. Within a band's strip,
    cards wrap into a ``band_cols``-wide grid (default 4 columns x 2 rows for n_per_band=8) so
    long snip_id/metric labels don't clip.

    ``image_path_col`` must already contain fully resolved, loadable paths. For a metric whose
    pass/fail threshold is itself a function of another column (e.g. a stage-interpolated
    reference band), use :func:`render_quartile_gallery_vs_band` instead.
    """
    bands = select_quartile_bands(df, metric_col, cutoff, fail_direction=fail_direction, n_per_band=n_per_band)
    return _render_band_page(
        bands,
        lambda row: f"{metric_col} = {row[metric_col]:.4g}  (cutoff {cutoff:g})",
        lambda row: bool(_fail_mask(pd.Series([row[metric_col]]), cutoff, fail_direction).iloc[0]),
        metric_col=metric_col, image_path_col=image_path_col, label_col=label_col,
        title=title, output_path=output_path, n_per_band=n_per_band, band_cols=band_cols,
    )


def render_quartile_gallery_with_overlay(
    df: pd.DataFrame,
    metric_col: str,
    cutoff: float,
    *,
    fail_direction: Literal["below", "above"],
    image_fn,
    image_path_col: str,
    label_col: str,
    title: str,
    output_path: Path,
    n_per_band: int = 8,
    band_cols: int = 4,
) -> Path:
    """Cutoff-relative quartile gallery whose cell image is drawn by ``image_fn(row) -> PIL
    image`` instead of loaded straight from disk — useful when the report wants an overlay on the
    snip but still needs the pass/fail-aware banding of :func:`render_quartile_gallery`.
    """
    bands = select_quartile_bands(df, metric_col, cutoff, fail_direction=fail_direction, n_per_band=n_per_band)

    def badge_text(row: pd.Series) -> str:
        return f"{metric_col} = {row[metric_col]:.4g}  (cutoff {cutoff:g})"

    def is_fail(row: pd.Series) -> bool:
        return bool(_fail_mask(pd.Series([row[metric_col]]), cutoff, fail_direction).iloc[0])

    return _render_band_page(
        bands,
        badge_text,
        is_fail,
        metric_col=metric_col,
        image_path_col=image_path_col,
        label_col=label_col,
        title=title,
        output_path=output_path,
        n_per_band=n_per_band,
        band_cols=band_cols,
        image_fn=image_fn,
    )


def render_quartile_gallery_vs_band(
    df: pd.DataFrame,
    metric_col: str,
    lower_col: str,
    upper_col: str,
    *,
    image_path_col: str,
    label_col: str,
    title: str,
    output_path: Path,
    n_per_band: int = 8,
    band_cols: int = 4,
) -> Path:
    """Same layout as :func:`render_quartile_gallery`, but judged against a per-row
    ``[lower_col, upper_col]`` band instead of one fixed scalar cutoff. Each card's badge shows
    the raw metric value alongside that row's own band, so the reader can see the actual
    reference (e.g. stage-interpolated p5/p95 x k) each embryo was judged against.
    """
    bands = select_quartile_bands_vs_band(df, metric_col, lower_col, upper_col, n_per_band=n_per_band)

    def badge_text(row: pd.Series) -> str:
        return f"{metric_col} = {row[metric_col]:.4g}  (band [{row[lower_col]:.4g}, {row[upper_col]:.4g}])"

    def is_fail(row: pd.Series) -> bool:
        return not (row[lower_col] <= row[metric_col] <= row[upper_col])

    return _render_band_page(
        bands, badge_text, is_fail,
        metric_col=metric_col, image_path_col=image_path_col, label_col=label_col,
        title=title, output_path=output_path, n_per_band=n_per_band, band_cols=band_cols,
    )


def plot_metric_vs_reference(
    df: pd.DataFrame,
    x_col: str,
    metric_col: str,
    *,
    fail_col: str,
    reference_df: pd.DataFrame,
    reference_x_col: str,
    reference_lower_col: str,
    reference_upper_col: str,
    title: str,
    output_path: Path,
    xlabel: str | None = None,
    ylabel: str | None = None,
    pass_alpha: float = 0.25,
    fail_alpha: float = 0.45,
    y_clip_quantile: float | None = 0.975,
    y_clip_headroom: float = 1.15,
) -> Path:
    """Scatter ``metric_col`` vs. a covariate (``x_col``, e.g. stage), with the reference band
    (``reference_lower_col``..``reference_upper_col`` over ``reference_x_col``) drawn as a shaded
    curve. For metrics whose pass/fail threshold is itself a function of another column — a
    single histogram with one fixed cutoff line is misleading for these; this plot shows the
    real time/covariate-dependent judgment instead.

    Extreme outliers (e.g. yolk-only / full-frame SAM2 mask blowups orders of magnitude above the
    real distribution) otherwise stretch the y-axis and squash the meaningful band into a thin
    strip. ``y_clip_quantile`` caps the y-axis at that quantile of ``metric_col`` (×
    ``y_clip_headroom`` for breathing room); points above the cap are NOT dropped — they are pinned
    to the top edge as upward-pointing markers so they stay visibly "off the chart, way over," and
    the cap + over-cap count are annotated. Set ``y_clip_quantile=None`` to disable clipping.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ref = reference_df.sort_values(reference_x_col)
    ax.fill_between(ref[reference_x_col], ref[reference_lower_col], ref[reference_upper_col],
                     color=PASS_COLOR, alpha=0.12, label="reference band")
    ax.plot(ref[reference_x_col], ref[reference_lower_col], color=PASS_COLOR, linewidth=1, linestyle="--")
    ax.plot(ref[reference_x_col], ref[reference_upper_col], color=PASS_COLOR, linewidth=1, linestyle="--")

    fail = df[fail_col].astype(bool)

    # Data-driven y-cap so a handful of blowup masks don't flatten the real distribution. Cap on the
    # metric's own high quantile (across all rows), never below the reference band's top.
    y_cap = None
    n_over_cap = 0
    if y_clip_quantile is not None:
        metric_vals = pd.to_numeric(df[metric_col], errors="coerce").dropna()
        if len(metric_vals):
            band_top = float(pd.to_numeric(ref[reference_upper_col], errors="coerce").max())
            q_cap = float(metric_vals.quantile(y_clip_quantile)) * y_clip_headroom
            y_cap = max(q_cap, band_top * y_clip_headroom)
            n_over_cap = int((metric_vals > y_cap).sum())

    def _draw(mask, *, color, alpha, size, label):
        x = df.loc[mask, x_col]
        y = pd.to_numeric(df.loc[mask, metric_col], errors="coerce")
        ax.scatter(x, y, s=size, color=color, alpha=alpha, label=label)
        if y_cap is not None:
            over = y > y_cap
            if over.any():
                # Pin over-cap points to the ceiling with an up-arrow so they read as "clipped".
                ax.scatter(x[over], [y_cap] * int(over.sum()), s=size + 12, color=color,
                           alpha=min(1.0, alpha + 0.35), marker="^")

    _draw(~fail, color=PASS_COLOR, alpha=pass_alpha, size=8, label="pass")
    _draw(fail, color=FAIL_COLOR, alpha=fail_alpha, size=10, label="fail")

    if y_cap is not None:
        # Pin the bottom at (a hair below) the data floor so autoscale doesn't leave dead negative
        # space — these metrics (areas, counts) are non-negative.
        metric_min = float(pd.to_numeric(df[metric_col], errors="coerce").min())
        ax.set_ylim(bottom=min(0.0, metric_min), top=y_cap)

    subtitle = f"n_fail={int(fail.sum())}, n_pass={int((~fail).sum())}, n_total={len(df)}"
    if n_over_cap:
        subtitle += f"; {n_over_cap} over-cap (▲ at y={y_cap:.2g})"
    ax.set_title(f"{title}\n({subtitle})")
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or metric_col)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────────────────
# Renderer A — grouped-trace-over-time (the "curtain")
# ──────────────────────────────────────────────────────────────────────────────────────────
def plot_grouped_traces(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: str,
    time_col: str = "time_index",
    title: str,
    output_path: Path,
    ylabel: str | None = None,
    alpha: float = 0.35,
    linewidth: float = 1.4,
    smoothing_window: int = 1,
    event_times: pd.DataFrame | None = None,
    event_time_col: str = "time_index",
    post_event_color: str = FAIL_COLOR,
    overlay: tuple[pd.Series, pd.Series, str] | None = None,
    overlay_on_secondary_axis: bool = False,
    overlay_ylabel: str = "",
    drawstyle: str = "default",
) -> Path:
    """One line per ``group_col`` (e.g. physical_embryo_id, well_id) tracing ``value_col`` over
    ``time_col``, all overplotted — the "curtain". A dense band means a coherent cohort; strands
    peeling away show *when* something happens per group.

    Grain is the point: ``group_col`` = the identity that owns the quantity (animal facts group by
    physical_embryo_id, plate facts by well_id). Traces are sorted by ``time_col`` within each group.

    Optional per-group EVENT rendering (``event_times``: one row per group with the event's
    ``event_time_col``): each trace is drawn in the base color up to its event and in
    ``post_event_color`` after it, with a black diamond at the event point — e.g. an embryo's
    fraction_alive trace turning red at its called death. ``smoothing_window`` > 1 applies a
    centered rolling mean per group first, to calm frame-to-frame jitter.

    Optional aggregate ``overlay`` (x, y, label): a bold line drawn on top of the ensemble — e.g.
    the whole-experiment alive-embryo survival curve laid over the per-embryo strands. When the
    overlay is on a different scale than the traces (a count over a 0–1 fraction curtain), set
    ``overlay_on_secondary_axis`` so it gets its own right-hand y-axis (``overlay_ylabel``).
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    ordered = df.sort_values([group_col, time_col])
    n_groups = ordered[group_col].nunique()
    event_lookup: dict[object, float] = {}
    if event_times is not None and len(event_times):
        event_lookup = dict(zip(event_times[group_col], event_times[event_time_col]))

    n_events = 0
    for group_key, trace in ordered.groupby(group_col, sort=False):
        t = trace[time_col].to_numpy()
        y = trace[value_col].to_numpy(dtype=float)
        if smoothing_window > 1:
            y = pd.Series(y).rolling(smoothing_window, center=True, min_periods=1).mean().to_numpy()

        event_t = event_lookup.get(group_key)
        if event_t is None:
            ax.plot(t, y, color=PASS_COLOR, alpha=alpha, linewidth=linewidth, drawstyle=drawstyle)
            continue

        # Split the trace at the event: base color before, post_event_color after (inclusive overlap
        # of one point so the segments join visually).
        pre = t <= event_t
        post = t >= event_t
        ax.plot(t[pre], y[pre], color=PASS_COLOR, alpha=alpha, linewidth=linewidth, drawstyle=drawstyle)
        ax.plot(t[post], y[post], color=post_event_color, alpha=min(alpha + 0.15, 1.0), linewidth=linewidth, drawstyle=drawstyle)
        # Black diamond at the event point (y taken from the smoothed trace at/near event_t).
        ev_idx = int(np.argmin(np.abs(t - event_t)))
        ax.scatter([t[ev_idx]], [y[ev_idx]], marker="D", s=22, color="#111111",
                   edgecolors="white", linewidths=0.5, zorder=4)
        n_events += 1

    legend_handles = []
    if n_events:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color=PASS_COLOR, lw=2, label="alive"),
            Line2D([0], [0], color=post_event_color, lw=2, label="after called death"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#111111", markersize=7,
                   label=f"called death ({n_events})"),
        ]

    if overlay is not None:
        ox, oy, olabel = overlay
        overlay_color = "#111111"
        if overlay_on_secondary_axis:
            ax2 = ax.twinx()
            ax2.plot(ox, oy, color=overlay_color, linewidth=2.6, zorder=5)
            # Right axis is a RAW EMBRYO COUNT: pin 0 -> full cohort so the survival line reads as
            # an honest count starting at the total, not a rescaled/normalized curve.
            ax2.set_ylim(0, float(pd.Series(oy).max()))
            ax2.set_ylabel(overlay_ylabel or olabel)
        else:
            ax.plot(ox, oy, color=overlay_color, linewidth=2.6, zorder=5)
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], color=overlay_color, lw=2.6, label=olabel))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", framealpha=0.9)

    ax.set_title(f"{title}\n(n_{group_col}={n_groups}, n_rows={len(df)}"
                 + (f", n_deaths={n_events}" if n_events else "") + ")")
    ax.set_xlabel(time_col)
    ax.set_ylabel(ylabel or value_col)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# Fixed categorical order (never cycled/reassigned) — see docs dataviz skill's palette.md.
# Light-mode 8-hue theme; worst adjacent CVD delta 24.2, well clear of the >=12 target.
CATEGORICAL_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)


def _plot_labeled_series_panel(ax, df: pd.DataFrame, value_col: str, *, series_col: str,
                                time_col: str, panel_title: str, categories: list,
                                smoothing_window: int) -> None:
    """Draw one panel's worth of labeled series onto an existing Axes (shared color assignment
    via the caller-supplied ``categories`` order, so the same reason gets the same color in
    every panel of a multi-panel figure — critical for a side-by-side comparison to be legible).
    ``smoothing_window`` > 1 applies a centered rolling mean per series first, to calm
    frame-to-frame jitter (the same smoothing ``plot_grouped_traces`` offers)."""
    ordered = df.sort_values([series_col, time_col])
    for i, category in enumerate(categories):
        trace = ordered[ordered[series_col] == category]
        if trace.empty:
            continue
        y = trace[value_col].to_numpy(dtype=float)
        if smoothing_window > 1:
            y = pd.Series(y).rolling(smoothing_window, center=True, min_periods=1).mean().to_numpy()
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        ax.plot(trace[time_col], y, color=color, linewidth=1.8, label=str(category))
    ax.set_title(f"{panel_title}\n(n_rows={len(df)})", fontsize=11)
    ax.set_xlabel(time_col)


def plot_labeled_series_over_time(
    df: pd.DataFrame,
    value_col: str,
    *,
    series_col: str,
    time_col: str = "time_index",
    title: str,
    output_path: Path,
    ylabel: str | None = None,
    panels: list[tuple[str, pd.DataFrame]] | None = None,
    smoothing_window: int = 1,
) -> Path:
    """One THIN, DISTINCTLY-COLORED, LEGENDED line per named category in ``series_col`` (e.g. a
    fixed vocabulary of QC exclusion reasons) tracing ``value_col`` over ``time_col``.

    Distinct from ``plot_grouped_traces`` (renderer A): that renderer overplots MANY same-colored,
    low-alpha strands where the ENSEMBLE shape is the point (one line per animal/well). This
    renderer is for a SMALL, NAMED set of categories where each line's IDENTITY matters — colors
    are assigned in a fixed categorical order (never cycled) and every line is legended, per the
    "legend always present for >=2 series" rule. Falls back to the categorical palette's fixed
    order, wrapping only if ``series_col`` has more distinct values than the palette (a smell —
    keep the vocabulary small; this is not a general many-series line chart).

    Pass ``panels`` — a list of ``(panel_title, panel_df)`` — to render several related views
    (e.g. "all snips" vs. "not-dead snips only") as SIDE-BY-SIDE subplots in one PNG instead of
    separate files, sharing one y-axis and one legend: the point of a comparison is reading two
    views at a glance, which a shared axis + single legend makes possible in a way two standalone
    files cannot. ``df``/``value_col`` are ignored when ``panels`` is given (panels carry their
    own frames); the category order/colors are still computed ONCE, from the union of all panels'
    categories, so the same reason is the same color in every panel.

    ``smoothing_window`` > 1 applies a centered rolling mean to each series first (same idea as
    ``plot_grouped_traces``'s smoothing) — a per-time_index fraction over a modest cohort is noisy
    frame to frame, and the trend across reasons is the point, not every jagged tick.
    """
    if panels is None:
        panels = [(title, df)]
        suptitle = None
    else:
        suptitle = title

    # ONE category->color assignment across all panels (union, first-seen order) so a reason's
    # color does not shift panel to panel — the whole reason a shared legend is trustworthy.
    categories: list = []
    for _, panel_df in panels:
        for cat in panel_df[series_col]:
            if cat not in categories:
                categories.append(cat)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (panel_title, panel_df) in zip(axes, panels):
        _plot_labeled_series_panel(
            ax, panel_df, value_col, series_col=series_col, time_col=time_col,
            panel_title=panel_title if suptitle else f"{title}\n(n_{series_col}={len(categories)}, n_rows={len(panel_df)})",
            categories=categories,
            smoothing_window=smoothing_window,
        )
    axes[0].set_ylabel(ylabel or value_col)

    handles = [
        plt.Line2D([0], [0], color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)], lw=1.8, label=str(cat))
        for i, cat in enumerate(categories)
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.95), framealpha=0.9, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.86, 0.94) if suptitle else (0, 0, 0.86, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight": the legend sits OUTSIDE the tight_layout rect (by design, so it never
    # overlaps a panel) — without this, savefig clips it at the figure's nominal canvas edge.
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_line(
    x: pd.Series,
    y: pd.Series,
    *,
    title: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    color: str = PASS_COLOR,
    annotate_start_final: bool = False,
) -> Path:
    """A single aggregate line — e.g. total alive embryos across the whole experiment over
    time_index (the one-number cohort survival curve). The whole-experiment counterpart to the
    per-group curtain; expect a monotone-ish decline with precipitous drops at mass-death events.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color=color, linewidth=2.2, marker="o", markersize=3)
    ax.fill_between(x, y, color=color, alpha=0.12)
    if annotate_start_final and len(x) and len(y):
        x_values = pd.Series(x).to_numpy(dtype=float)
        y_values = pd.Series(y).to_numpy(dtype=float)
        valid = np.where(np.isfinite(x_values) & np.isfinite(y_values))[0]
        if valid.size:
            start_i, final_i = valid[0], valid[-1]
            for label, idx in (("start", start_i), ("final *", final_i)):
                ax.scatter([x_values[idx]], [y_values[idx]], s=60, color=color, edgecolor="black", zorder=3)
                ax.annotate(
                    f"{label}: {int(y_values[idx])} embryos",
                    xy=(x_values[idx], y_values[idx]),
                    xytext=(6, 8),
                    textcoords="offset points",
                    fontsize=9,
                    weight="bold" if label.startswith("final") else "normal",
                )
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────────────────
# Renderer B — count-per-group distribution
# ──────────────────────────────────────────────────────────────────────────────────────────
def plot_count_per_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    output_path: Path,
    xlabel: str | None = None,
) -> Path:
    """Histogram of a per-group ROW COUNT — "how many rows per ``group_col``" (e.g. embryos per
    well, snips per embryo, detections per frame). This bins *counts of rows*, NOT a measured
    value, which is why it is a distinct renderer from ``plot_metric_histogram`` (a measured-value
    histogram). Integer-aligned bins so each bar is a discrete count.
    """
    counts = df.groupby(group_col).size().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = int(counts.min()), int(counts.max())
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0) if hi > lo else np.array([lo - 0.5, lo + 0.5])
    ax.hist(counts, bins=edges, color=PASS_COLOR, edgecolor="white")

    ax.set_title(f"{title}\n(n_{group_col}={counts.size}, "
                 f"mean={counts.mean():.2f}, median={np.median(counts):.0f}, max={hi})")
    ax.set_xlabel(xlabel or f"rows per {group_col}")
    ax.set_ylabel(f"count of {group_col}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────────────────
# Renderer D — histogram grid (one gridded PNG, one panel per column)
# ──────────────────────────────────────────────────────────────────────────────────────────
def plot_histogram_grid(
    df: pd.DataFrame,
    columns: list[str],
    *,
    title: str,
    output_path: Path,
    cutoffs: dict[str, tuple[float, Literal["below", "above"]]] | None = None,
    n_cols: int = 3,
) -> Path:
    """One figure with a grid of per-column histograms — the "look at the distribution of every
    feature" primitive, as a single PNG rather than N files. Where a column has an entry in
    ``cutoffs`` ({col: (cutoff, fail_direction)}), its bars are pass/fail split like
    ``plot_metric_histogram``; otherwise the column is drawn as a plain distribution. Non-finite
    values are dropped per column, so nullable columns degrade gracefully.
    """
    cutoffs = cutoffs or {}
    n = len(columns)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for idx, col in enumerate(columns):
        ax = axes[idx // n_cols][idx % n_cols]
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.set_title(f"{col}\n(no finite values)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        _, edges = np.histogram(values, bins="auto")
        if col in cutoffs:
            cutoff, fail_direction = cutoffs[col]
            fail_values = _fail_mask(pd.Series(values), cutoff, fail_direction).to_numpy()
            pass_counts, _ = np.histogram(values[~fail_values], bins=edges)
            fail_counts, _ = np.histogram(values[fail_values], bins=edges)
            lefts, widths = edges[:-1], np.diff(edges)
            ax.bar(lefts, pass_counts, width=widths, align="edge", color=PASS_COLOR, edgecolor="white")
            ax.bar(lefts, fail_counts, width=widths, align="edge", bottom=pass_counts,
                   color=FAIL_COLOR, edgecolor="white")
            ax.axvline(cutoff, color=FAIL_COLOR, linestyle="--", linewidth=1.2)
        else:
            ax.hist(values, bins=edges, color=PASS_COLOR, edgecolor="white")
        ax.set_title(f"{col}  (n={values.size})", fontsize=10)

    for idx in range(n, n_rows * n_cols):  # blank the unused cells
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────────────────
# Plain value-quartile gallery (NO threshold) — for distribution-only metrics
# ──────────────────────────────────────────────────────────────────────────────────────────
VALUE_QUARTILE_BANDS: tuple[str, ...] = ("Q4_highest", "Q3", "Q2", "Q1_lowest")


def select_value_quartiles(
    df: pd.DataFrame, metric_col: str, *, n_per_band: int = 8
) -> dict[str, pd.DataFrame]:
    """Split rows into four bands by VALUE quartile of ``metric_col`` — no pass/fail, no cutoff.
    For distribution-only metrics (no threshold) where the gallery just answers "what does a
    low/mid/high one look like?". Each band shows up to ``n_per_band`` example rows drawn from that
    quartile's extreme so the strip spans the quartile's range.
    """
    ranked = df.dropna(subset=[metric_col]).sort_values(metric_col)
    if ranked.empty:
        return {name: ranked for name in VALUE_QUARTILE_BANDS}
    q1, q2, q3 = ranked[metric_col].quantile([0.25, 0.5, 0.75])
    below = lambda s, hi: s[s[metric_col] <= hi]  # noqa: E731
    within = lambda s, lo, hi: s[(s[metric_col] > lo) & (s[metric_col] <= hi)]  # noqa: E731
    return {
        "Q4_highest": ranked[ranked[metric_col] > q3].tail(n_per_band).iloc[::-1],
        "Q3": within(ranked, q2, q3).head(n_per_band),
        "Q2": within(ranked, q1, q2).head(n_per_band),
        "Q1_lowest": below(ranked, q1).head(n_per_band),
    }


def render_value_quartile_gallery(
    df: pd.DataFrame,
    metric_col: str,
    *,
    image_path_col: str,
    label_col: str,
    title: str,
    output_path: Path,
    n_per_band: int = 8,
    band_cols: int = 4,
) -> Path:
    """Gallery of example images across VALUE quartiles of ``metric_col`` — the no-threshold
    counterpart to :func:`render_quartile_gallery`. Use this for a distribution-only metric (a
    histogram with no cutoff); use the cutoff/band galleries only when a real pass/fail threshold
    exists and the interesting region is *near* it.
    """
    bands = select_value_quartiles(df, metric_col, n_per_band=n_per_band)
    return _render_band_page(
        bands,
        lambda row: f"{metric_col} = {row[metric_col]:.4g}",
        lambda row: False,  # no fail concept for a pure distribution
        metric_col=metric_col, image_path_col=image_path_col, label_col=label_col,
        title=title, output_path=output_path, n_per_band=n_per_band, band_cols=band_cols,
        band_names=VALUE_QUARTILE_BANDS,
    )


def render_value_quartile_gallery_with_overlay(
    df: pd.DataFrame,
    metric_col: str,
    *,
    image_fn,
    image_path_col: str,
    label_col: str,
    title: str,
    output_path: Path,
    n_per_band: int = 8,
    band_cols: int = 4,
) -> Path:
    """VALUE-quartile gallery (no threshold) whose cell image is drawn by ``image_fn(row) -> PIL
    image`` instead of loaded straight from disk — for reports that overlay a derived geometry on
    the snip (e.g. the curvature centerline via :func:`draw_centerline_on_snip`). Bands span the
    ``metric_col`` value range so the reader sees a low / mid / high example of the metric with its
    overlay drawn on top. ``image_fn`` returning ``None`` for a row falls back to ``image_path_col``.
    """
    bands = select_value_quartiles(df, metric_col, n_per_band=n_per_band)
    return _render_band_page(
        bands,
        lambda row: f"{metric_col} = {row[metric_col]:.4g}",
        lambda row: False,
        metric_col=metric_col, image_path_col=image_path_col, label_col=label_col,
        title=title, output_path=output_path, n_per_band=n_per_band, band_cols=band_cols,
        band_names=VALUE_QUARTILE_BANDS, image_fn=image_fn,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────
# 96-well plate heatmap — a per-well scalar laid out in physical plate geometry (8x12)
# ──────────────────────────────────────────────────────────────────────────────────────────
# Plate geometry: rows A-H (8), columns 1-12 (12). Well-id parsing is NOT done here — the
# well_id -> (row, col) decode is owned by the shared identifiers package (the one sanctioned home
# for identifier grammar). This viz layer imports it; it never re-parses a well_id with ord()/split.
from data_pipeline.shared.identifiers import parse_well_row_col  # noqa: E402

PLATE_ROWS = 8   # A-H
PLATE_COLS = 12  # 1-12


def plot_plate_heatmap(
    df: pd.DataFrame,
    value_col: str,
    *,
    well_col: str = "well_id",
    title: str,
    output_path: Path,
    cbar_label: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    empty_color: str = "#dddddd",
) -> Path:
    """Lay a per-well scalar (``value_col``, one row per well) onto physical 8x12 plate geometry —
    rows A-H down, columns 1-12 across — so SPATIAL patterns (edge effects, dead quadrants, a bad
    column) are visible in a way a histogram cannot show. Wells absent from ``df`` render in
    ``empty_color``. Each cell is annotated with its value when ``annotate``.
    """
    grid = np.full((PLATE_ROWS, PLATE_COLS), np.nan)
    for _, r in df.iterrows():
        row, col = parse_well_row_col(str(r[well_col]))
        grid[row, col] = r[value_col]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    masked = np.ma.masked_invalid(grid)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(empty_color)
    im = ax.imshow(masked, cmap=cmap_obj, aspect="equal")

    ax.set_xticks(range(PLATE_COLS), labels=[str(c + 1) for c in range(PLATE_COLS)])
    ax.set_yticks(range(PLATE_ROWS), labels=[chr(ord("A") + rr) for rr in range(PLATE_ROWS)])
    ax.set_xticks(np.arange(-0.5, PLATE_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, PLATE_ROWS, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    if annotate:
        for rr in range(PLATE_ROWS):
            for cc in range(PLATE_COLS):
                if not np.isnan(grid[rr, cc]):
                    v = grid[rr, cc]
                    txt = f"{v:g}" if v == int(v) else f"{v:.2g}"
                    ax.text(cc, rr, txt, ha="center", va="center", fontsize=8, color="white")

    n_wells = int(np.isfinite(grid).sum())
    ax.set_title(f"{title}\n(n_wells={n_wells} / {PLATE_ROWS * PLATE_COLS})")
    fig.colorbar(im, ax=ax, label=cbar_label or value_col, shrink=0.8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────────────────
# Renderer F2 — well × time survival heatmap (SHARED across three reports)
# ──────────────────────────────────────────────────────────────────────────────────────────
def _well_time_survival_grid(
    df: pd.DataFrame,
    *,
    well_col: str,
    time_col: str,
    value_col: str,
) -> "tuple[list[str], np.ndarray, np.ndarray]":
    """Pivot a long (well, time, value) frame into a dense well × time matrix.

    Returns (well order [sorted], time axis values [sorted], value grid [n_wells x n_times]).
    Absent (well, time) cells are NaN (drawn as empty), so a well that vanishes mid-run reads as a
    gap rather than being silently dropped — the honest survival floor, same discipline as
    death_detection._alive_counts.
    """
    wells = sorted(df[well_col].astype(str).unique())
    times = np.sort(df[time_col].dropna().unique())
    well_ix = {w: i for i, w in enumerate(wells)}
    time_ix = {t: j for j, t in enumerate(times)}
    grid = np.full((len(wells), len(times)), np.nan)
    for _, r in df.iterrows():
        t = r[time_col]
        if pd.isna(t):
            continue
        grid[well_ix[str(r[well_col])], time_ix[t]] = r[value_col]
    return wells, times, grid


def _draw_well_survival_axes(
    ax_lead,
    ax_heat,
    *,
    wells: list[str],
    times: np.ndarray,
    grid: np.ndarray,
    time_label: str,
    value_label: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    lead_grid: np.ndarray | None = None,
) -> "matplotlib.image.AxesImage":
    """Draw the two leading annotation columns (starting cohort size + final value) and the
    well × time heatmap into a pre-made (ax_lead, ax_heat) pair. Returns the heatmap image (for a
    shared colorbar). Shared by both the global and per-genotype-facet code paths.

    Leading columns are derived from the grid itself so they never disagree with the heatmap:
      - starting cohort = first non-NaN value along each well's row (baseline the value is measured
        against — for a count grid this is the starting embryo count; for a fraction grid, 1.0);
      - final value      = last non-NaN value along each well's row (where the well ended up).
    """
    def _first_last(row: np.ndarray) -> "tuple[float, float]":
        valid = np.where(np.isfinite(row))[0]
        if valid.size == 0:
            return (np.nan, np.nan)
        return (row[valid[0]], row[valid[-1]])

    lead_source = lead_grid if lead_grid is not None else grid
    starts, finals = zip(*(_first_last(lead_source[i]) for i in range(len(wells)))) if wells else ((), ())
    lead = np.array([starts, finals], dtype=float).T  # n_wells x 2

    lead_masked = np.ma.masked_invalid(lead)
    lead_cmap = plt.get_cmap("Greys").copy()
    lead_cmap.set_bad("#dddddd")
    lead_vmax = float(np.nanmax(lead)) if np.isfinite(lead).any() else None
    ax_lead.imshow(lead_masked, cmap=lead_cmap, aspect="auto", vmin=0.0, vmax=lead_vmax)
    ax_lead.set_xticks([0, 1], labels=["start n", "final n"], fontsize=8)
    ax_lead.set_yticks(range(len(wells)), labels=wells, fontsize=7)
    for i in range(len(wells)):
        for j, v in enumerate(lead[i]):
            if np.isfinite(v):
                txt = f"{v:g}" if v == int(v) else f"{v:.2g}"
                ax_lead.text(j, i, txt, ha="center", va="center", fontsize=6, color="white")

    heat_masked = np.ma.masked_invalid(grid)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#dddddd")
    im = ax_heat.imshow(heat_masked, cmap=cmap_obj, aspect="auto", vmin=vmin, vmax=vmax)
    # Thin the x tick labels so a long time axis stays legible.
    step = max(1, len(times) // 20)
    xt = list(range(0, len(times), step))
    ax_heat.set_xticks(xt, labels=[f"{times[k]:g}" for k in xt], fontsize=7, rotation=90)
    ax_heat.set_yticks([])  # wells labeled on the leading-column axis
    ax_heat.set_xlabel(time_label)
    return im


def plot_well_survival_over_time(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    title: str,
    output_path: Path,
    well_col: str = "well_id",
    value_label: str | None = None,
    genotype_col: str | None = None,
    include_global: bool = True,
    lead_value_col: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Well × time survival heatmap with two leading annotation columns (starting cohort size +
    final value). SHARED by three reports (DRY): the plot is identical; only what is fed to it
    differs.

      - physical_embryo_registry_report: value = embryo count per well (death PROXY), time = time_index;
      - death_detection_report:          value = fraction alive (~persistence_dead_flag), time = time_index;
      - analysis_ready_report:           value = fraction alive, time = predicted_stage_hpf, faceted by genotype.

    ``df`` is long: one row per (well_col, time_col[, genotype_col]) carrying ``value_col``. When
    ``genotype_col`` is given, renders a GLOBAL panel (all wells) followed by one panel per genotype
    (small-multiple facets), mirroring plot_grouped_survival_panel. Otherwise a single global panel.

    ``lead_value_col`` lets callers color the main heatmap by one metric while showing start/final
    leading columns from another metric, e.g. relative survival in the body with actual embryo
    counts in the leading columns.

    ``vmin``/``vmax`` fix the color scale across facets so genotype panels are comparable (pass
    vmin=0, vmax=1 for a fraction; leave None for a count to autoscale per figure).
    """
    value_label = value_label or value_col
    time_label = time_col

    # Facet groups: the global "(all)" panel always first, then one per genotype if requested.
    if genotype_col is not None and genotype_col in df.columns:
        genos = sorted(df[genotype_col].dropna().astype(str).unique())
        panels: list["tuple[str, pd.DataFrame]"] = [("all genotypes", df)] if include_global else []
        panels += [(g, df[df[genotype_col].astype(str) == g]) for g in genos]
    else:
        panels = [("all wells", df)]

    grids = []
    for label, sub in panels:
        wells, times, grid = _well_time_survival_grid(
            sub, well_col=well_col, time_col=time_col, value_col=value_col
        )
        lead_grid = None
        if lead_value_col is not None:
            lead_wells, lead_times, lead_grid_candidate = _well_time_survival_grid(
                sub, well_col=well_col, time_col=time_col, value_col=lead_value_col
            )
            if lead_wells == wells and np.array_equal(lead_times, times):
                lead_grid = lead_grid_candidate
        if wells:
            grids.append((label, wells, times, grid, lead_grid))
    if not grids:
        raise ValueError(f"plot_well_survival_over_time: no wells to plot for {value_col!r}.")

    # One row per panel; each row is [leading columns | heatmap] via a nested gridspec. Row height
    # tracks well count so panels with more wells get more vertical room.
    total_wells = sum(len(wells) for _, wells, _, _, _ in grids)
    fig_h = max(3.0, 0.22 * total_wells + 1.2 * len(grids))
    fig = plt.figure(figsize=(13, fig_h))
    outer = fig.add_gridspec(
        len(grids), 1, height_ratios=[max(1, len(wells)) for _, wells, _, _, _ in grids], hspace=0.5
    )

    last_im = None
    for i, (label, wells, times, grid, lead_grid) in enumerate(grids):
        inner = outer[i].subgridspec(1, 2, width_ratios=[1, 14], wspace=0.02)
        ax_lead = fig.add_subplot(inner[0])
        ax_heat = fig.add_subplot(inner[1])
        ax_lead.set_title(label, fontsize=10, loc="left")
        last_im = _draw_well_survival_axes(
            ax_lead, ax_heat,
            wells=wells, times=times, grid=grid,
            time_label=time_label, value_label=value_label,
            cmap=cmap, vmin=vmin, vmax=vmax,
            lead_grid=lead_grid,
        )

    if last_im is not None:
        fig.colorbar(last_im, ax=fig.axes, label=value_label, shrink=0.6, pad=0.02)
    fig.suptitle(title, y=0.995)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────────────────
# Renderer G — latent PCA projections (analysis_ready)
# ──────────────────────────────────────────────────────────────────────────────────────────
def _latent_pca_coordinates(
    df: pd.DataFrame,
    latent_cols: list[str],
    *,
    random_state: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    work = df.dropna(subset=latent_cols).copy()
    if len(work) < 3:
        raise ValueError(
            f"latent PCA: only {len(work)} rows with complete latents "
            f"({len(latent_cols)} cols) — too few to project."
        )

    X = StandardScaler().fit_transform(work[latent_cols].to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=random_state)
    emb = pca.fit_transform(X)
    work["_pca_x"], work["_pca_y"] = emb[:, 0], emb[:, 1]
    return work, pca.explained_variance_ratio_


def _set_pca_axis_labels(ax, explained_variance: np.ndarray) -> None:
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% var)")


def plot_latent_pca_qc_state(
    df: pd.DataFrame,
    latent_cols: list[str],
    *,
    qc_pass_col: str,
    title: str,
    output_path: Path,
    random_state: int = 0,
) -> Path:
    """PCA all complete latent rows once, then color every point by whether the snip passed QC."""
    if qc_pass_col not in df.columns:
        raise ValueError(f"plot_latent_pca_qc_state: missing QC pass column {qc_pass_col!r}.")

    work, explained = _latent_pca_coordinates(df, latent_cols, random_state=random_state)
    pass_mask = work[qc_pass_col].fillna(False).astype(bool)

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    for label, mask, color in [
        ("not passing QC", ~pass_mask, FAIL_COLOR),
        ("passing QC", pass_mask, PASS_COLOR),
    ]:
        sub = work[mask]
        ax.scatter(
            sub["_pca_x"],
            sub["_pca_y"],
            s=8,
            alpha=0.55,
            c=color,
            label=f"{label} (n={len(sub)})",
        )
    _set_pca_axis_labels(ax, explained)
    ax.legend(title="QC state", loc="best", frameon=True)
    ax.set_title(title)
    fig.suptitle(
        "Purpose: check whether QC removes a distinct region of latent morphology space.\n"
        f"(n={len(work)} snips, latents={len(latent_cols)}, PCA on all complete rows)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_latent_pca_continuous(
    df: pd.DataFrame,
    latent_cols: list[str],
    *,
    continuous_col: str,
    title: str,
    output_path: Path,
    random_state: int = 0,
) -> Path:
    """PCA complete latent rows and color the projection by a continuous column."""
    work, explained = _latent_pca_coordinates(
        df.dropna(subset=[continuous_col]), latent_cols, random_state=random_state
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    sc = ax.scatter(
        work["_pca_x"],
        work["_pca_y"],
        c=work[continuous_col].to_numpy(dtype=float),
        cmap="viridis",
        s=9,
        alpha=0.75,
    )
    _set_pca_axis_labels(ax, explained)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=continuous_col, shrink=0.85)
    fig.suptitle(
        "Purpose: check whether latent morphology space tracks predicted developmental stage.\n"
        f"(n={len(work)} passing-QC snips, latents={len(latent_cols)}, PCA on plotted rows)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_latent_pca_categorical(
    df: pd.DataFrame,
    latent_cols: list[str],
    *,
    category_col: str,
    title: str,
    output_path: Path,
    category_colors: dict[str, str] | None = None,
    random_state: int = 0,
) -> Path:
    """PCA complete latent rows and color the projection by a categorical column."""
    work, explained = _latent_pca_coordinates(
        df.dropna(subset=[category_col]), latent_cols, random_state=random_state
    )
    work["_category"] = work[category_col].astype(str)

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    categories = sorted(work["_category"].dropna().unique())
    cmap = plt.get_cmap("tab20", max(1, len(categories)))
    for i, category in enumerate(categories):
        sub = work[work["_category"] == category]
        color = category_colors.get(category) if category_colors else None
        if color is None:
            color = cmap(i)
        ax.scatter(
            sub["_pca_x"],
            sub["_pca_y"],
            s=9,
            alpha=0.75,
            c=[color],
            label=f"{category} (n={len(sub)})",
        )
    _set_pca_axis_labels(ax, explained)
    ax.set_title(title)
    ax.legend(title=category_col, loc="best", frameon=True, fontsize=8)
    fig.suptitle(
        "Purpose: check whether passing-QC latent morphology space separates by genotype.\n"
        f"(n={len(work)} passing-QC snips, latents={len(latent_cols)}, PCA on plotted rows)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────────────────
# Renderer H — grouped survival panel (overlay header + per-group small-multiples)
# ──────────────────────────────────────────────────────────────────────────────────────────
def plot_grouped_survival_panel(
    curves: dict[str, "tuple[np.ndarray, np.ndarray]"],
    *,
    title: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    group_colors: dict[str, str] | None = None,
    n_cols: int = 3,
) -> Path:
    """One figure: a TOP overlay row with every group's survival curve layered on shared axes, then
    a small-multiples GALLERY below — one cell per group, same curve. ``curves`` maps group label ->
    (x, y). Layout: n_cols columns (default 3); a single group collapses the gallery to one column.

    x is whatever the caller binned on (here predicted_stage_hpf), y is total alive count.
    """
    groups = sorted(curves)
    n = len(groups)
    gallery_cols = 1 if n <= 1 else n_cols
    gallery_rows = math.ceil(n / gallery_cols)

    def _color(g: str) -> str | None:
        return (group_colors or {}).get(g)

    # Row 0 spans full width (the overlay); the gallery grid sits below it.
    fig = plt.figure(figsize=(5 * gallery_cols, 4 * (gallery_rows + 1)))
    gs = fig.add_gridspec(gallery_rows + 1, gallery_cols)

    ax_overlay = fig.add_subplot(gs[0, :])
    for g in groups:
        x, y = curves[g]
        ax_overlay.plot(x, y, marker="o", markersize=2, linewidth=1.8, label=g, color=_color(g))
    ax_overlay.set_title("all genotypes overlaid")
    ax_overlay.set_xlabel(xlabel)
    ax_overlay.set_ylabel(ylabel)
    ax_overlay.set_ylim(bottom=0)
    ax_overlay.legend(fontsize=8, loc="best", framealpha=0.9)

    for i, g in enumerate(groups):
        r, c = divmod(i, gallery_cols)
        ax = fig.add_subplot(gs[r + 1, c])
        x, y = curves[g]
        color = _color(g)
        ax.plot(x, y, marker="o", markersize=2, linewidth=1.8, color=color)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.set_title(g, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"{title}\n(n_genotypes={n})", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
