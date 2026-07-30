"""Read one experiment's image-well -> hash-well map out of morphseq plate metadata.

Two sources, in preference order:

1. ``output/acquisition/{experiment_id}/ingest_metadata/plate_metadata.csv`` — the pipeline's own
   validated plate table. Preferred: it is what every other consumer reads, and its ``well_id`` is
   already minted.
2. ``input/plate_metadata/{experiment_id}_well_metadata.xlsx`` — read through the pipeline's own
   ``load_plate_metadata_pages``, never a private Excel parser.

The fallback exists because the hash sheets can be authored into a workbook after that experiment's
last ``ingest_plate_metadata`` run, in which case the CSV is real but predates the sheets. Falling
back keeps the crosswalk buildable without forcing a pipeline re-run first; ``source`` on the
returned frame records which one was used so a stale CSV is never invisible.

Sheet-name aliases are tolerated because the same grid has been authored under three names across
generations of the workbooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_loader import (
    load_plate_metadata_pages,
)
from data_pipeline.shared.identifiers import build_well_id

from .paths import PipelinePaths, default_paths

# Canonical column, then legacy aliases, for each fact we need. Order is preference order.
#
# Only the FORWARD map is accepted here. Some workbooks also carry a ``hash_to_image_map`` sheet,
# which is the inverse (hash well -> image well); reading it as if it were the forward map would
# silently transpose the pairing, so it is deliberately not an alias.
HASH_WELL_COLUMNS: tuple[str, ...] = ("image_to_hash_map",)
HASH_PLATE_COLUMNS: tuple[str, ...] = (
    "hash_plate_num",
    "image_to_hash_plate_num",
    "image_to_hash_plate_map",
)

SOURCE_PLATE_CSV = "plate_metadata_csv"
SOURCE_WORKBOOK = "plate_workbook"


@dataclass(frozen=True)
class ExperimentHashMap:
    """One experiment's per-well hash assignment, as authored (not yet normalized).

    ``table`` has columns ``well_id``, ``well_index``, ``hash_plate_raw``, ``hash_well_raw``.
    ``source`` is one of ``plate_metadata_csv`` / ``plate_workbook``.
    """

    experiment_id: str
    table: pd.DataFrame
    source: str


def _first_present(columns: "list[str]", candidates: "tuple[str, ...]") -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def _extract(
    df: pd.DataFrame, experiment_id: str, source: str
) -> ExperimentHashMap | None:
    """Pull the two hash columns out of a well-keyed table, or ``None`` if either is absent."""
    columns = list(df.columns)
    well_column = _first_present(columns, HASH_WELL_COLUMNS)
    plate_column = _first_present(columns, HASH_PLATE_COLUMNS)
    if well_column is None or plate_column is None:
        return None

    out = pd.DataFrame(
        {
            "well_index": df["well_index"].astype(str).str.strip(),
            "hash_plate_raw": df[plate_column],
            "hash_well_raw": df[well_column],
        }
    )
    out["well_id"] = [build_well_id(experiment_id, well) for well in out["well_index"]]
    out = out.loc[:, ["well_id", "well_index", "hash_plate_raw", "hash_well_raw"]]
    return ExperimentHashMap(
        experiment_id=experiment_id,
        table=out.reset_index(drop=True),
        source=source,
    )


def load_experiment_hash_map(
    experiment_id: str,
    *,
    paths: PipelinePaths | None = None,
) -> ExperimentHashMap | None:
    """Load ``experiment_id``'s hash map, or ``None`` if neither source carries one.

    ``None`` is a legitimate answer — most imaging experiments are not paired with sequencing and
    have no hash sheets. The caller decides whether that is an error.

    Raises:
        ValueError: propagated from the plate loader on a malformed grid sheet. A workbook whose
            hash grid is broken is a data-entry error worth surfacing, not worth skipping.
    """
    resolved = paths if paths is not None else default_paths()

    csv_path = resolved.plate_metadata_csv(experiment_id)
    if csv_path.is_file():
        found = _extract(pd.read_csv(csv_path), experiment_id, SOURCE_PLATE_CSV)
        if found is not None:
            return found

    workbook = resolved.plate_workbook(experiment_id)
    if workbook.is_file():
        pages = load_plate_metadata_pages(workbook)
        return _extract(pages.table, experiment_id, SOURCE_WORKBOOK)

    return None


def experiments_with_hash_map(
    *, paths: PipelinePaths | None = None
) -> list[str]:
    """Every imaging experiment whose plate workbook carries a hash map.

    This is the population that *should* have a key entry — i.e. the answer to "which experiments
    are paired with sequencing", read off morphseq's own metadata rather than an external table.
    """
    resolved = paths if paths is not None else default_paths()
    found: list[str] = []
    for workbook in sorted(resolved.plate_metadata_dir.glob("*_well_metadata.xlsx")):
        experiment_id = workbook.name.replace("_well_metadata.xlsx", "")
        with pd.ExcelFile(workbook) as excel:
            sheets = {str(name).strip().lower().replace(" ", "_").replace("-", "_") for name in excel.sheet_names}
        if sheets & set(HASH_WELL_COLUMNS) and sheets & set(HASH_PLATE_COLUMNS):
            found.append(experiment_id)
    return found
