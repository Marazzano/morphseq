"""Curated well exclusions — the single source of truth for "do not analyze this well".

Exclusions here come from **human image inspection**, not from the pipeline's automated QC. That is
deliberate and is the standing policy for this project: the current pipeline's ``use_snip`` /
``sa_outlier_flag`` were computed on a degraded snip raster (coarser pixel size, elevated
saturation) and against a reference population that mis-calibrates cold-reared embryos, so they are
not applicable to the legacy morphology this analysis is built on. Wells are dropped because someone
looked at the image and rejected it.

The table lives in ``excluded_wells.csv`` beside this module so it is diffable, reviewable, and
attributable — each row carries ``reason``, ``curated_by``, and ``curated_on``.

``well_id`` is the operative key; the ``experiment_date`` / ``collection_time_hpf`` /
``perturbation`` / ``well_index`` columns are the human-facing spelling the curator works in, and are
cross-checked against ``well_id`` on load so the two cannot silently drift apart.

Applied consistently across BOTH modalities — morphology and sequencing — so cohort membership is
identical everywhere. ``apply_exclusions`` is the one function analyses should call.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXCLUSIONS_FILENAME = "excluded_wells.csv"

EXCLUSION_COLUMNS: tuple[str, ...] = (
    "experiment_date",
    "collection_time_hpf",
    "perturbation",
    "well_index",
    "well_id",
    "reason",
    "curated_by",
    "curated_on",
)


def default_exclusions_path() -> Path:
    """Path to the curated exclusion table shipped alongside this module."""
    return Path(__file__).resolve().parent / EXCLUSIONS_FILENAME


def load_excluded_wells(path: str | Path | None = None) -> pd.DataFrame:
    """Read and validate the curated exclusion table.

    Raises:
        FileNotFoundError: if the table is missing.
        ValueError: on a missing column, a duplicate ``well_id``, a blank ``well_id``, or a row whose
            ``well_id`` disagrees with its own ``experiment_date``/``collection_time_hpf``/
            ``perturbation``/``well_index`` spelling.
    """
    resolved = Path(path) if path is not None else default_exclusions_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"[morphseq_integration] exclusion table not found: {resolved}")

    table = pd.read_csv(resolved, dtype=str, keep_default_na=False)
    missing = [column for column in EXCLUSION_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"[morphseq_integration] {resolved.name} is missing column(s) {missing}. "
            f"Expected header: {','.join(EXCLUSION_COLUMNS)}"
        )

    table = table.loc[:, list(EXCLUSION_COLUMNS)].copy()
    for column in EXCLUSION_COLUMNS:
        table[column] = table[column].astype(str).str.strip()
    table = table.loc[table["well_id"] != ""].reset_index(drop=True)

    duplicated = table.loc[table.duplicated(subset=["well_id"], keep=False), "well_id"]
    if not duplicated.empty:
        raise ValueError(
            f"[morphseq_integration] duplicate well_id in {resolved.name}: "
            f"{sorted(set(duplicated))}. Each well is excluded once."
        )

    # The redundant spelling is the point: it is what a curator reads and edits. Verify it agrees
    # with well_id rather than trusting that both were updated together.
    expected = (
        table["experiment_date"]
        + "_"
        + table["collection_time_hpf"]
        + "hpf_"
        + table["perturbation"]
        + "_"
        + table["well_index"]
    )
    disagrees = expected != table["well_id"]
    mismatched = pd.DataFrame(
        {"well_id": table.loc[disagrees, "well_id"], "expected": expected.loc[disagrees]}
    )
    if not mismatched.empty:
        raise ValueError(
            f"[morphseq_integration] {resolved.name} has rows whose well_id disagrees with its "
            f"experiment_date/collection_time_hpf/perturbation/well_index columns: "
            f"{mismatched.head(5).to_dict(orient='records')}."
        )
    return table


def excluded_well_ids(path: str | Path | None = None) -> set[str]:
    """The set of curated-excluded ``well_id``s."""
    return set(load_excluded_wells(path)["well_id"])


def apply_exclusions(
    frame: pd.DataFrame,
    *,
    well_id_column: str = "well_id",
    path: str | Path | None = None,
    mode: str = "drop",
    verbose: bool = False,
) -> pd.DataFrame:
    """Remove (or flag) curated-excluded wells.

    Args:
        mode: ``"drop"`` removes the rows; ``"flag"`` keeps them and adds a boolean
            ``curated_excluded`` column. Use ``"flag"`` when a plot needs to show what was dropped.
        verbose: print how many rows were removed, per experiment.

    Returns:
        The filtered (or flagged) frame. Always a copy.

    Note:
        Exclusions absent from ``frame`` are not an error — a table restricted to one plate will
        legitimately match only some of them.
    """
    if mode not in ("drop", "flag"):
        raise ValueError(f"[morphseq_integration] mode must be 'drop' or 'flag', got {mode!r}.")
    if well_id_column not in frame.columns:
        raise ValueError(
            f"[morphseq_integration] frame has no {well_id_column!r} column to exclude on."
        )

    excluded = excluded_well_ids(path)
    hit = frame[well_id_column].isin(excluded)

    if verbose:
        print(f"[exclusions] {int(hit.sum())} of {len(frame)} rows are curated-excluded")
        if hit.any() and "experiment_id" in frame.columns:
            print(frame.loc[hit].groupby("experiment_id").size().to_string())

    if mode == "flag":
        out = frame.copy()
        out["curated_excluded"] = hit.to_numpy()
        return out
    return frame.loc[~hit].reset_index(drop=True)


__all__ = [
    "EXCLUSION_COLUMNS",
    "apply_exclusions",
    "default_exclusions_path",
    "excluded_well_ids",
    "load_excluded_wells",
]
