"""Collection TIME AXIS — the ONE place that maps source-native time onto merged time.

A collection plate is assembled from N independent source acquisitions. Each source numbers its
own frames from its own origin, so two snapshot sources both call their single frame ``0``. The
merged plate needs ONE monotonic temporal coordinate across all sources, while still being able
to say which source frame a merged frame came from.

That is exactly two columns, and this module is the only place allowed to produce them:

    raw_time_index   the source-native frame index, copied unchanged (provenance / audit)
    time_index       the canonical merged frame index (contiguous across the whole plate)

Both the scope-metadata union and the acquisition-inventory union call this helper, so the two
artifacts cannot drift apart on what ``time_index`` means — a drift that previously existed
(one block-offset, the other overwrote ``time_index`` with the per-source ordinal, which
destroyed intra-source time for a timelapse source).

**``time_index`` is NOT the source ordinal.** The source ordinal (which source) is a separate
column, ``source_ordinal``, and it is what the age map is keyed by. One ``source_ordinal`` may
span MANY ``time_index`` values (a timelapse source), so the two can never be the same column.
They coincide only in the all-snapshots case, which is why conflating them went unnoticed.

Import direction: pure pandas; imports nothing from the pipeline. Callers are the unions.
"""

from __future__ import annotations

import pandas as pd


def remap_source_time_indices(
    frame: pd.DataFrame,
    running_offset: int,
    *,
    scope_label: str = "remap_source_time_indices",
) -> tuple[pd.DataFrame, int]:
    """Preserve source-native time as ``raw_time_index``; emit contiguous merged ``time_index``.

    Args:
        frame: one SOURCE's rows, carrying that source's own ``time_index``.
        running_offset: the next free merged ``time_index`` (0 for the first source; thereafter
            the value returned by the previous call).
        scope_label: prefix for error messages, so a failure names its caller.

    Returns:
        ``(remapped_frame, next_offset)`` — the frame with ``raw_time_index`` (source-native) and
        ``time_index`` (merged, contiguous starting at ``running_offset``), plus the next free
        offset for the following source.

    Invariants (enforced here, pinned by tests):
      * **Rows sharing a ``raw_time_index`` share a ``time_index``.** A frame is many rows
        (z-planes, tiles, channels), and they must all land on the same merged timepoint.
      * **Merged ``time_index`` is CONTIGUOUS; source gaps are deliberately NOT preserved.**
        Source frame numbering can be sparse or 1-based — Keyence derives it from on-disk
        ``T####`` tokens, so a partial or resumed acquisition yields e.g. ``2, 4``. Naive
        ``time_index + offset`` would carry those gaps into the merged axis and mis-width the
        next source's block. Remapping by rank yields a dense axis; ``raw_time_index`` keeps the
        original value, so nothing is lost — only renumbered.
      * Ordering is by source-native value, so merged order follows real acquisition order.

    Raises:
        ValueError: if ``time_index`` is absent, holds missing values, or is non-numeric. Numeric
            coercion happens BEFORE ranking so mixed strings/numbers cannot silently mis-sort
            (``"10" < "9"`` lexically).
    """
    if "time_index" not in frame.columns:
        raise ValueError(
            f"[{scope_label}] expected a source-native 'time_index' column to remap; got columns "
            f"{sorted(frame.columns)}. This helper turns per-source time into merged plate time; "
            "the source's own time_index is its input."
        )
    if frame["time_index"].isna().any():
        n_missing = int(frame["time_index"].isna().sum())
        raise ValueError(
            f"[{scope_label}] source 'time_index' contains {n_missing} missing value(s). Every "
            "frame must declare its source-native timepoint — a null cannot be placed on the "
            "merged time axis."
        )

    out = frame.copy()
    # Coerce BEFORE ranking: errors="raise" so a non-numeric value fails loud here rather than
    # sorting lexically (which would order "10" before "9") or becoming a silent NaN.
    try:
        native = pd.to_numeric(out["time_index"], errors="raise").astype(int)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"[{scope_label}] source 'time_index' must be integer-valued; could not coerce "
            f"({exc}). Fix the source reader rather than coercing here — a mis-typed time axis "
            "silently reorders frames."
        ) from exc

    out["raw_time_index"] = native
    # Rank-based remap: the i-th distinct source timepoint becomes running_offset + i. This is
    # what makes the merged axis dense regardless of sparse/1-based source numbering, and it
    # guarantees every row sharing a raw value gets the same merged value (a dict lookup).
    distinct_native = sorted(native.unique())
    merged_by_native = {
        native_value: running_offset + i for i, native_value in enumerate(distinct_native)
    }
    out["time_index"] = out["raw_time_index"].map(merged_by_native).astype(int)

    return out, running_offset + len(distinct_native)
