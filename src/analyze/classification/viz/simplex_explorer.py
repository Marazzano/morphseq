"""Probability-simplex explorer: one threshold slider per class, ANDed together.

Because a multinomial model's class probabilities sum to 1, thresholding them
independently and ANDing the results carves out a *region of the simplex* rather
than slicing a single axis::

    P(CE) >= 0.7                          -> confidently CE
    P(CE) >= 0.3 AND P(CE) <= 0.6         -> boundary cases worth eyeballing
    P(Not Penetrant) >= 0.6, facet homo   -> candidate non-penetrant mutants

Impossible combinations (e.g. all classes ``>= 0.5``) are allowed — they simply
return nothing — and the page flags the constraint rather than silently showing
an empty plot.

Scope
-----
This module is for:
- Turning an already-scored, per-timepoint DataFrame into a JSON payload
- Rendering that payload into a self-contained interactive HTML page

Not for:
- Fitting models. Viz consumes probabilities; it does not produce them. Do the
  cross-validation / training in the calling script and pass the columns in.
- Experiment-specific loading, label curation, or color conventions.

Data flow
---------
1. ``build_simplex_payload(df, ...)`` — scored long DataFrame → payload dict.
2. ``render_simplex_explorer_html(payload, ...)`` — payload → HTML string.

Convenience: ``render_simplex_explorer_html_from_scores(df, ...)`` does both.

Taking pipeline output directly
-------------------------------
This module stays producer-agnostic: ``class_prob_cols`` is always explicit, and
nothing here knows what any upstream step names its columns. To feed it a known
producer's output without hand-writing that map, use the adapter module::

    from analyze.classification.viz.simplex_adapters import from_label_transfer

    from_label_transfer(
        transferred_df,                     # has prob_CE / prob_HTA / ...
        features=["baseline_deviation_normalized", "total_length_um"],
        output_path="explorer.html",
    )

See ``simplex_adapters`` for the available adapters and how to add one.

Class names vs zygosity
-----------------------
Class names are PHENOTYPE labels; the facet groups are GENOTYPES. The unaffected
class is called ``"Not Penetrant"``, never ``"wildtype"`` — an embryo of any
zygosity (het, homozygous, unknown) can be predicted Not Penetrant, which is the
whole point of the class. A ``wildtype`` *zygosity* facet and a ``Not Penetrant``
*class* reading differently is correct and intended. ``non_penetrant_class``
names which class the panel divider treats as unaffected.

Generalization
--------------
Nothing hardcodes three classes. The page builds one slider per entry in
``class_prob_cols``, so N=1, 2, 3, … all work; the impossible-region check and
the shade dropdown scale with N automatically.

Public API
----------
``build_simplex_payload``                  — scored DataFrame → payload dict
``render_simplex_explorer_html``           — payload → self-contained HTML
``render_simplex_explorer_html_from_scores`` — convenience: both in one call
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "build_simplex_payload",
    "render_simplex_explorer_html",
    "render_simplex_explorer_html_from_scores",
]

# ---------------------------------------------------------------------------
# Package asset paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_TEMPLATE_NAME = "_simplex_template.html"

# Fallback palette when the caller supplies no class colors. Deliberately not
# matplotlib's tab10 — these are the colorblind-safe hues already used across
# the project's genotype figures.
_DEFAULT_CLASS_COLORS = [
    "#1B9E77", "#B2182B", "#7F7F7F", "#2166AC", "#F7B267",
    "#9467bd", "#4C9F70", "#D95F02", "#7570B3", "#666666",
]

_DEFAULT_SUBTITLE = (
    "One slider per class, <strong>ANDed</strong>: an embryo is kept only if it satisfies "
    "every constraint at once. Because the class probabilities sum to 1, this carves out a "
    "region of the simplex rather than slicing a single axis. Each facet panel shows a dashed "
    "divider at the <em>classifier's</em> predicted-penetrant rate — that is a model output, "
    "not a measured biological penetrance."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Make a JS-safe, stable key from a class display name."""
    key = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return key or "class"


def _unique_keys(names: Sequence[str]) -> list[str]:
    """Slugified keys for ``names``, disambiguated on collision."""
    seen: dict[str, int] = {}
    keys: list[str] = []
    for n in names:
        base = _slugify(n)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        keys.append(base)
    return keys


def _round_or_none(val: object, ndigits: int) -> float | None:
    """Round to ``ndigits``, mapping NaN / non-numeric to None (JSON-safe)."""
    try:
        f = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, ndigits)


def _downsample(n: int, max_points: int) -> np.ndarray:
    """Indices of at most ``max_points`` evenly spaced samples out of ``n``."""
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------


def build_simplex_payload(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    features: Sequence[str],
    class_prob_cols: Mapping[str, str],
    group_col: Optional[str] = None,
    curated_col: Optional[str] = None,
    in_train_col: Optional[str] = None,
    class_colors: Optional[Mapping[str, str]] = None,
    class_aucs: Optional[Mapping[str, float]] = None,
    feature_labels: Optional[Mapping[str, str]] = None,
    group_labels: Optional[Mapping[str, str]] = None,
    group_order: Optional[Sequence[str]] = None,
    group_colors: Optional[Mapping[str, str]] = None,
    non_penetrant_class: Optional[str] = None,
    time_label: str = "Hours post fertilization",
    max_points: int = 60,
    default_color_by: str = "zyg",
) -> dict:
    """Build the JSON payload for the simplex explorer from a scored DataFrame.

    The input is *long*: one row per (embryo, timepoint). Class probabilities are
    read from the embryo-level columns named in ``class_prob_cols`` and are taken
    from the embryo's first row — they are expected to be constant within an
    embryo (e.g. a renormalized mean over timepoints).

    Parameters
    ----------
    df:
        Long, already-scored table. One row per embryo-timepoint.
    id_col:
        Embryo identifier column. Defines trace grouping.
    time_col:
        Numeric developmental-time column, the x axis of every panel.
    features:
        Ordered feature columns to plot, one panel row per feature.
    class_prob_cols:
        Ordered ``{class display name: probability column}``. Order defines
        slider order. Any N >= 1 is supported; nothing assumes three.
        Explicit by design — to take a known producer's output without writing
        this map, use ``analyze.classification.viz.simplex_adapters``.
    group_col:
        Categorical column used for faceting/coloring by group (e.g. zygosity).
        ``None`` → all embryos land in a single ``"all"`` group.
    curated_col:
        Optional manual-label column, shown in the hover tooltip only.
    in_train_col:
        Optional boolean-ish column marking embryos whose score is out-of-fold.
        ``None`` → every embryo is reported as "applied (not trained)".
    class_colors:
        ``{class name: hex}``. Missing classes fall back to a built-in palette.
    class_aucs:
        Optional ``{class name: one-vs-rest AUC}``, shown next to each slider so
        a weak class score is not read as if it were the strong one.
    feature_labels:
        ``{feature column: axis title}``. Missing → the column name.
    group_labels:
        ``{raw group value: display name}``. Missing → the raw value as string.
    group_order:
        Display order of groups. ``None`` → order of first appearance. Groups
        absent from the data are dropped.
    group_colors:
        ``{display group name: hex}``. Missing → gray.
    non_penetrant_class:
        Class whose prediction counts as NOT penetrant (typically the wildtype
        class). The panel divider reports the fraction of embryos predicted into
        any *other* class. ``None`` → the last class in ``class_prob_cols``.
    time_label:
        X-axis title.
    max_points:
        Max timepoints kept per trace; longer series are evenly downsampled.
    default_color_by:
        Initial "color by" mode: ``"zyg"`` (group) or ``"pred"``.

    Returns
    -------
    dict
        JSON-serializable payload consumed by ``render_simplex_explorer_html``.
    """
    class_names = list(class_prob_cols.keys())
    if not class_names:
        raise ValueError("class_prob_cols must name at least one class.")

    prob_cols = [class_prob_cols[c] for c in class_names]
    required = [id_col, time_col, *features, *prob_cols]
    for col in (group_col, curated_col, in_train_col):
        if col is not None:
            required.append(col)
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing required columns: {missing}. Available: {df.columns.tolist()}"
        )
    if default_color_by not in ("zyg", "pred"):
        raise ValueError(f"default_color_by must be 'zyg' or 'pred', got {default_color_by!r}")

    if non_penetrant_class is None:
        non_penetrant_class = class_names[-1]
    elif non_penetrant_class not in class_names:
        raise ValueError(
            f"non_penetrant_class={non_penetrant_class!r} is not one of {class_names}"
        )

    keys = _unique_keys(class_names)
    key_of = dict(zip(class_names, keys))

    colors = dict(class_colors) if class_colors else {}
    for i, name in enumerate(class_names):
        colors.setdefault(name, _DEFAULT_CLASS_COLORS[i % len(_DEFAULT_CLASS_COLORS)])

    flabels = dict(feature_labels) if feature_labels else {}
    glabels = dict(group_labels) if group_labels else {}
    gcolors = dict(group_colors) if group_colors else {}

    embryos: list[dict] = []
    seen_groups: list[str] = []
    # sort=False keeps embryo order stable w.r.t. the input, which keeps the
    # emitted HTML byte-reproducible for a given DataFrame.
    for eid, grp in df.groupby(id_col, sort=False):
        grp = grp.sort_values(time_col)
        row = grp.iloc[0]

        probs_raw = {name: float(row[class_prob_cols[name]]) for name in class_names}
        if not all(np.isfinite(v) for v in probs_raw.values()):
            continue  # an unscored embryo has nothing to threshold on

        sel = _downsample(len(grp), max_points)
        sub = grp.iloc[sel]

        group_raw = row[group_col] if group_col is not None else "all"
        group = glabels.get(group_raw, str(group_raw))
        if group not in seen_groups:
            seen_groups.append(group)

        pred = max(class_names, key=lambda c: probs_raw[c])
        embryos.append({
            "id": str(eid),
            "zyg": group,
            "pred": pred,
            "curated": (str(row[curated_col]) if curated_col is not None else ""),
            "inTrain": bool(row[in_train_col]) if in_train_col is not None else False,
            "p": {key_of[c]: round(probs_raw[c], 4) for c in class_names},
            "t": [_round_or_none(v, 2) for v in sub[time_col]],
            "y": [[_round_or_none(v, 4) for v in sub[f]] for f in features],
        })

    if group_order is not None:
        ordered_groups = [glabels.get(g, str(g)) for g in group_order]
        ordered_groups = [g for g in ordered_groups if g in seen_groups]
        # Anything present in the data but absent from group_order still needs a
        # panel, else its embryos vanish from the faceted view.
        ordered_groups += [g for g in seen_groups if g not in ordered_groups]
    else:
        ordered_groups = seen_groups

    return {
        "embryos": embryos,
        "classes": [
            {
                "name": name,
                "key": key_of[name],
                "color": colors[name],
                "auc": _round_or_none(class_aucs.get(name), 3) if class_aucs else None,
            }
            for name in class_names
        ],
        "nonPenetrantClass": non_penetrant_class,
        "zygOrder": ordered_groups,
        "zygColor": {g: gcolors.get(g, "#888888") for g in ordered_groups},
        "featureLabels": [flabels.get(f, f) for f in features],
        "timeLabel": time_label,
        "defaultColorBy": default_color_by,
    }


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------


def _load_asset(filename: str) -> str:
    """Load a package asset by filename, with editable-install fallback."""
    try:
        from importlib.resources import files as _res_files
        return (
            _res_files("analyze.classification.viz")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
    except Exception:
        return (_HERE / filename).read_text(encoding="utf-8")


def _escape_html(text: str) -> str:
    """Escape a plain-text string for safe substitution into markup."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _validate_payload(payload: Mapping[str, object]) -> None:
    """Raise ValueError if the payload cannot drive the template."""
    for field in ("embryos", "classes", "featureLabels", "zygOrder", "zygColor"):
        if field not in payload:
            raise ValueError(f"payload is missing required field {field!r}")
    classes = payload["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError("payload['classes'] must be a non-empty list.")
    keys = [c["key"] for c in classes]  # type: ignore[index]
    if len(set(keys)) != len(keys):
        raise ValueError(f"payload class keys must be unique, got {keys}")
    n_feat = len(payload["featureLabels"])  # type: ignore[arg-type]
    for e in payload["embryos"]:  # type: ignore[union-attr]
        if len(e["y"]) != n_feat:
            raise ValueError(
                f"embryo {e['id']!r} has {len(e['y'])} feature series "
                f"but featureLabels declares {n_feat}"
            )


def render_simplex_explorer_html(
    payload: Mapping[str, object],
    *,
    title: str = "Probability simplex explorer",
    subtitle: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Render a simplex-explorer payload to a self-contained interactive HTML page.

    The output has no external references — no CDN, no fonts, no images. It can
    be opened directly from disk in any modern browser, and follows the viewer's
    light/dark preference.

    Parameters
    ----------
    payload:
        Payload from ``build_simplex_payload`` (or an equivalent dict).
    title:
        Page ``<title>`` and headline. Escaped before substitution.
    subtitle:
        Intro paragraph under the headline. May contain inline HTML and is NOT
        escaped, so callers can emphasize example thresholds. ``None`` → a
        generic description of the ANDed-simplex semantics.
    output_path:
        If provided, the HTML is written here (UTF-8, parents created). The
        string is returned regardless.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    _validate_payload(payload)

    template = _load_asset(_TEMPLATE_NAME)
    # allow_nan=False so a stray NaN fails loudly here rather than emitting bare
    # `NaN` tokens that silently break JSON.parse-free inline evaluation later.
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)

    html = template.replace("__TITLE__", _escape_html(title))
    html = html.replace("__SUBTITLE__", subtitle if subtitle is not None else _DEFAULT_SUBTITLE)
    html = html.replace("__PAYLOAD__", data_json)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")

    return html


def render_simplex_explorer_html_from_scores(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    features: Sequence[str],
    class_prob_cols: Mapping[str, str],
    title: str = "Probability simplex explorer",
    subtitle: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    **payload_kwargs,
) -> str:
    """Build the payload and render it in one step.

    Convenience wrapper around ``build_simplex_payload`` +
    ``render_simplex_explorer_html``. Extra keyword arguments are forwarded to
    the payload builder.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    payload = build_simplex_payload(
        df,
        id_col=id_col,
        time_col=time_col,
        features=features,
        class_prob_cols=class_prob_cols,
        **payload_kwargs,
    )
    return render_simplex_explorer_html(
        payload,
        title=title,
        subtitle=subtitle,
        output_path=output_path,
    )
