"""Adapters: producer-specific tables → simplex-explorer payloads.

``simplex_explorer`` is deliberately producer-agnostic — it takes an explicit
``{class name: probability column}`` map and knows nothing about who produced
it. That keeps the viz primitive stable, but it means every caller has to
hand-write the same column map for the same few well-known producers.

This module is that translation layer, and *only* that. One adapter per
producer, each of which knows exactly one output convention, resolves it into
the explorer's explicit vocabulary, and delegates. Nothing here draws anything;
nothing in ``simplex_explorer`` imports this.

Why a module rather than a ``prefix=`` argument
-----------------------------------------------
Prefix sniffing inside the builder made a viz primitive responsible for knowing
that ``prob_`` means label-transfer and ``pred_proba_`` means
``run_classification``. That coupling grows with every new producer and fails
silently when two conventions coexist in one table. An adapter names its
producer in the function name, so the caller states which contract they are in
and a wrong guess is a loud error rather than a quietly mismapped column.

Adding a producer
-----------------
Write ``from_<producer>(df, ...)``, resolve its columns via
``resolve_prob_cols``, and call through to ``build_simplex_payload`` /
``render_simplex_explorer_html_from_scores``. Do not add prefixes to an
existing adapter to make it cover a second producer.

Public API
----------
``PROB_COL_CONVENTIONS``   — {producer name: probability-column prefix}
``resolve_prob_cols``      — explicit prefix + df → {class name: column}
``from_label_transfer``    — label_transfer output → HTML
``from_run_classification``— run_classification predictions → HTML
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from .simplex_explorer import render_simplex_explorer_html_from_scores

__all__ = [
    "PROB_COL_CONVENTIONS",
    "resolve_prob_cols",
    "from_label_transfer",
    "from_run_classification",
]

# The probability-column prefix each known producer writes. Explicit and
# closed: an unknown producer gets its own adapter, not a new guess here.
PROB_COL_CONVENTIONS: dict[str, str] = {
    "label_transfer": "prob_",
    "run_classification": "pred_proba_",
}


def resolve_prob_cols(
    df: pd.DataFrame,
    prefix: str,
    *,
    class_order: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Resolve ``{class name: column}`` for one known prefix.

    Unlike a sniffing helper, the prefix is required — the caller (an adapter)
    already knows which producer it is reading, so a miss is a real error.

    Parameters
    ----------
    df:
        Table to inspect.
    prefix:
        The producer's probability-column prefix, e.g. ``"prob_"``.
    class_order:
        Restrict to (and order by) these classes. Names with no matching column
        raise. ``None`` → all discovered classes, sorted for determinism.

    Returns
    -------
    dict[str, str]
        Ordered ``{class name: column}``.
    """
    found = {
        c[len(prefix):]: c
        for c in df.columns
        if c.startswith(prefix) and len(c) > len(prefix)
        and pd.api.types.is_numeric_dtype(df[c])
    }
    if not found:
        raise ValueError(
            f"No numeric {prefix!r} columns found. Available: {df.columns.tolist()}"
        )
    if class_order is None:
        return {c: found[c] for c in sorted(found)}

    missing = [c for c in class_order if c not in found]
    if missing:
        raise ValueError(
            f"class_order entries have no {prefix!r} column: {missing}. "
            f"Discovered: {sorted(found)}"
        )
    return {c: found[c] for c in class_order}


def _render(
    df: pd.DataFrame,
    prefix: str,
    *,
    id_col: str,
    time_col: str,
    features: Sequence[str],
    class_order: Optional[Sequence[str]],
    title: str,
    subtitle: Optional[str],
    output_path: Optional[Union[str, Path]],
    **payload_kwargs,
) -> str:
    """Shared adapter body: resolve columns for ``prefix``, then delegate."""
    return render_simplex_explorer_html_from_scores(
        df,
        id_col=id_col,
        time_col=time_col,
        features=features,
        class_prob_cols=resolve_prob_cols(df, prefix, class_order=class_order),
        title=title,
        subtitle=subtitle,
        output_path=output_path,
        **payload_kwargs,
    )


def from_label_transfer(
    df: pd.DataFrame,
    *,
    id_col: str = "query_embryo_id",
    time_col: str = "predicted_stage_hpf",
    features: Sequence[str],
    class_order: Optional[Sequence[str]] = None,
    title: str = "Probability simplex explorer",
    subtitle: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    **payload_kwargs,
) -> str:
    """Render a ``label_transfer`` output (``prob_<class>`` columns).

    Defaults match ``transfer_labels``, which keys embryos on
    ``query_embryo_id``. Extra keyword arguments are forwarded to
    ``build_simplex_payload`` (``group_col``, ``non_penetrant_class``, …).

    Returns
    -------
    str
        Self-contained HTML document.
    """
    return _render(
        df, PROB_COL_CONVENTIONS["label_transfer"],
        id_col=id_col, time_col=time_col, features=features,
        class_order=class_order, title=title, subtitle=subtitle,
        output_path=output_path, **payload_kwargs,
    )


def from_run_classification(
    df: pd.DataFrame,
    *,
    id_col: str = "embryo_id",
    time_col: str = "time_bin_center",
    features: Sequence[str],
    class_order: Optional[Sequence[str]] = None,
    title: str = "Probability simplex explorer",
    subtitle: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    **payload_kwargs,
) -> str:
    """Render a ``run_classification`` predictions layer (``pred_proba_<class>``).

    Defaults match ``result.layers["predictions"]``. Extra keyword arguments are
    forwarded to ``build_simplex_payload``.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    return _render(
        df, PROB_COL_CONVENTIONS["run_classification"],
        id_col=id_col, time_col=time_col, features=features,
        class_order=class_order, title=title, subtitle=subtitle,
        output_path=output_path, **payload_kwargs,
    )
