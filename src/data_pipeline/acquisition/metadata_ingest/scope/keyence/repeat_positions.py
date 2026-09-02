"""One policy for the repeat stage positions a Keyence run occasionally produces.

WHAT HAPPENS ON THE SCOPE. A stage position sometimes gets set twice for the same well during
setup, so the export carries one more XY directory than there are wells -- e.g. 97 XY directories
for 96 wells, with both ``XY06`` and ``XY97`` holding the ``_A06`` marker. Both are real, complete
acquisitions of the same physical well; neither is corrupt, and no well is missing.

WHY IT MATTERS. Nothing downstream expects two acquisitions of one well. They union into a single
per-well inventory and ``materialize_well_keyence`` refuses the well outright::

    ValueError: Duplicate Keyence time/tile/z inventory cells for well=..._A06

which fails the entire experiment over an operator-side setup artifact. Observed 2026-08-30 on four
experiments (cep290 x2, b9d2 x2).

WHY THE LOWEST INDEX. The re-set position is appended at the END of the position list, so the lowest
index is the original acquisition. This is a POLICY choice, not a quality judgement -- nothing here
compares the two captures' pixels. Callers record what was dropped so the discarded acquisition
stays findable.

WHY IT LIVES HERE AND NOT IN THE CONTRACT. ``validate_position_well_mapping`` keys uniqueness on
(experiment_id, position_index) and deliberately PERMITS two positions per well, because a
collection plate concatenates one mapping block per raw source. De-duplicating by well is a Keyence
ingest policy, not a universal contract rule.

WHY TWO CALL SITES. The position mapping and the acquisition inventory are built independently, each
walking the raw tree itself; the inventory never reads the mapping. Applying the rule in only one of
them leaves the other still carrying both captures, so both call this -- and must agree, which is
why the rule is defined once here.
"""

from __future__ import annotations

import logging
from typing import Hashable, Iterable

import pandas as pd

log = logging.getLogger(__name__)


def kept_position_index_by_well(
    pairs: Iterable[tuple[Hashable, object]],
) -> dict[Hashable, int]:
    """Map each well to the ONE ``position_index`` this run keeps: the lowest.

    ``pairs`` is ``(well, position_index)``. Rows whose position_index is missing or non-numeric are
    ignored rather than guessed at -- a W0-style export has no XY position to order by, and there the
    caller must leave the data alone.
    """
    kept: dict[Hashable, int] = {}
    for well, position_index in pairs:
        if position_index is None or pd.isna(position_index):
            continue
        try:
            value = int(position_index)
        except (TypeError, ValueError):
            continue
        current = kept.get(well)
        if current is None or value < current:
            kept[well] = value
    return kept


def drop_repeat_well_positions(
    df: pd.DataFrame,
    *,
    well_column: str,
    warnings: list[str] | None = None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return ``df`` reduced to one position per well, plus the rows that were set aside.

    ``label_column`` names a human-readable position column (e.g. ``source_position_name``) used in
    the warning text; it is optional so callers without one still work.
    """
    if well_column not in df.columns or "position_index" not in df.columns:
        return df, []

    kept = kept_position_index_by_well(zip(df[well_column], df["position_index"]))
    if not kept:
        return df, []

    keep_mask = pd.Series(
        [
            (well in kept) and (not pd.isna(pos)) and int(pos) == kept[well]
            for well, pos in zip(df[well_column], df["position_index"])
        ],
        index=df.index,
    )
    if bool(keep_mask.all()):
        return df, []

    dropped = df.loc[~keep_mask].to_dict("records")
    if warnings is not None:
        for well in sorted({str(row[well_column]) for row in dropped}):
            dropped_here = [r for r in dropped if str(r[well_column]) == well]
            names = sorted(
                {
                    str(r[label_column]) if label_column else str(r["position_index"])
                    for r in dropped_here
                }
            )
            kept_row = df.loc[keep_mask & (df[well_column].astype(str) == well)]
            kept_name = (
                str(kept_row[label_column].iloc[0])
                if label_column and len(kept_row)
                else str(kept.get(well, "<none>"))
            )
            message = (
                f"Well {well}: repeat stage position(s) {', '.join(names)} dropped; "
                f"keeping {kept_name}."
            )
            warnings.append(message)
            log.warning(message)

    return df.loc[keep_mask].reset_index(drop=True), dropped
