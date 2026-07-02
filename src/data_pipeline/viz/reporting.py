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
    """Histogram of the raw metric, with a vertical line at the actual cutoff value and each BAR
    colored by which side of the cutoff its bin falls on (fail-colored only if the whole bin is
    on the fail side, so a bin straddling the cutoff — e.g. mostly-passing values in a bin whose
    edge touches the cutoff — never reads as "failing" from a semi-transparent background wash).
    """
    values = metric.to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, edges = np.histogram(values, bins="auto")
    bin_lefts, bin_rights = edges[:-1], edges[1:]

    if fail_direction == "below":
        bin_is_fail = bin_rights <= cutoff
    else:
        bin_is_fail = bin_lefts >= cutoff
    colors = np.where(bin_is_fail, FAIL_COLOR, PASS_COLOR)

    ax.bar(bin_lefts, counts, width=np.diff(edges), align="edge", color=colors, edgecolor="white")
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
) -> Image.Image:
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
        image_path = Path(str(row[image_path_col]))
        framed = _fit_into_frame(Image.open(image_path), box_w, box_h)
        im.paste(framed, (box[0], box[1]))
    except (FileNotFoundError, OSError):
        d.rectangle(box, fill="#eeeeee", outline="#bbbbbb")
        d.text((box[0] + 10, box[1] + box_h // 2 - 8), "image missing", fill="#777777", font=FONT_HEADER)
    return im


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
    page_h = title_h + len(BAND_NAMES) * (strip_h + gap) + margin

    page = Image.new("RGB", (page_w, page_h), PAGE_BG)
    d = ImageDraw.Draw(page)
    d.text((margin, 10), title, fill="#111111", font=FONT_TITLE)

    for band_idx, band_name in enumerate(BAND_NAMES):
        rows = bands[band_name]
        strip_y0 = title_h + band_idx * (strip_h + gap)
        d.text((margin, strip_y0 + strip_h // 2 - 8), f"{band_name}\n(n={len(rows)})",
               fill="#111111", font=FONT_HEADER)

        for i, (_, row) in enumerate(rows.iterrows()):
            grid_row, grid_col = divmod(i, band_cols)
            x0 = margin + row_header_w + grid_col * (card_w + gap)
            y0 = strip_y0 + band_title_h + grid_row * (card_h + gap)
            card = _card(row, metric_col, image_path_col, label_col, card_w, card_h, head_h,
                         is_fail=is_fail_fn(row), badge_text=badge_text_fn(row))
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
) -> Path:
    """Scatter ``metric_col`` vs. a covariate (``x_col``, e.g. stage), with the reference band
    (``reference_lower_col``..``reference_upper_col`` over ``reference_x_col``) drawn as a shaded
    curve. For metrics whose pass/fail threshold is itself a function of another column — a
    single histogram with one fixed cutoff line is misleading for these; this plot shows the
    real time/covariate-dependent judgment instead.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ref = reference_df.sort_values(reference_x_col)
    ax.fill_between(ref[reference_x_col], ref[reference_lower_col], ref[reference_upper_col],
                     color=PASS_COLOR, alpha=0.12, label="reference band")
    ax.plot(ref[reference_x_col], ref[reference_lower_col], color=PASS_COLOR, linewidth=1, linestyle="--")
    ax.plot(ref[reference_x_col], ref[reference_upper_col], color=PASS_COLOR, linewidth=1, linestyle="--")

    fail = df[fail_col].astype(bool)
    ax.scatter(df.loc[~fail, x_col], df.loc[~fail, metric_col], s=8, color=PASS_COLOR, alpha=0.5, label="pass")
    ax.scatter(df.loc[fail, x_col], df.loc[fail, metric_col], s=10, color=FAIL_COLOR, alpha=0.7, label="fail")

    ax.set_title(f"{title}\n(n_fail={int(fail.sum())}, n_pass={int((~fail).sum())}, n_total={len(df)})")
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or metric_col)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
