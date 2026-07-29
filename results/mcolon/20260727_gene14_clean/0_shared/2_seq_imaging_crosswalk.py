"""Map imaging wells to GENE14 sequencing embryo_IDs.

*** TEMPORARY (2026-07-28): WE PARSE embryo_ID, WE DO NOT READ THE COORDINATE COLUMNS. ***
    The hash_plate / hash_well / rt_block COLUMNS in GENE14_embryo_metadata.tsv have carried
    errors that the embryo_ID STRING did not -- 11 cep290 embryos had hash_plate corrected
    P18->P02 while their ID still said P18, which collided two distinct embryos onto one
    coordinate. A fix is coming from the sequencing side. Until it lands and is verified,
    build_seq_index() takes the coordinate from parse_embryo_id() only.
    REMOVE THIS NOTE (and reconsider reading the columns) once the metadata is confirmed good.

Resolution:

    imaging (experiment, well)
        -> hash_well, hash_plate, rt_block   from the plate's Excel sheets
        -> embryo_ID                         from (hash_plate, hash_well, rt_block)

The invariants that make this work, each of which cost real debugging time:

- A sequencing embryo_ID IS `GENE14_{hash_plate}_{hash_well}_{rt_block}` and is unique, so that
  triple resolves an embryo exactly -- no gene, no timepoint, no tie-break.
- All three coordinate parts come from the plate's own Excel sheets, one sheet each, all keyed
  by imaging well: hash_plate_num, image_to_hash_map, rt_block.
- `hash_plate_num` decides WHICH wells a plate placed; a well with no hash plate has no
  sequencing coordinate. `image_to_hash_map` only translates wells that moved.
- "30to48" experiments have sequencing collection time 48 hpf, not their plate start age.
- One physical embryo is often imaged several times (_sci timeseries + _t01/_t02 snapshots), so
  several imaging rows legitimately resolve to the SAME embryo_ID. The caller collapses them.

Background, failure modes and history: see SEQ_IMAGING_CROSSWALK.md next to this file.

Public API (all used by 3_attach_morphseq_labels.py; nothing here calls itself):
    load_plate_maps()  -> {experiment: {imaging_well: HashLocation}}
    build_crosswalk()  -> DataFrame[experiment, imaging_well, hash_well, hash_plate, rt_block]
    collection_time_hpf(experiment, plate_start_age) -> sequencing timepoint (30to48 -> 48)
    rt_block_for(gene, collection_time_hpf)          -> 'Bl2', ...  (cross-check only)
    parse_embryo_id(embryo_ID)                       -> HashCoordinate | None
    build_seq_index()  -> {HashCoordinate: embryo_ID}
    resolve_seq_embryo_id(hash_plate, hash_well, rt_block, seq_index) -> embryo_ID | None

Pure lookup logic -- no I/O of its own beyond reading the plate Excels and the metadata TSV.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ------------------------------------------------------------------ paths
PROJECT_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
QC_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
EXCEL_DIR = QC_DIR / "source_plate_metadata_excels"
SEQ_METADATA_TSV = Path("/net/seahub_zfish/vol1/data/preprocessed/GENE14/GENE14_embryo_metadata.tsv")

# CROSS-CHECK ONLY (2026-07-28) -- the pipeline now reads rt_block from each plate's Excel
# `rt_block` sheet, so this table is no longer on the resolution path. Kept one more cycle
# because it agrees with the sheets on all 589 resolved rows; delete it, and rt_block_for(),
# once that has held for a session or two.
#
# The bench RT plate map: {gene: {collection_time_hpf: rt_block}}. An RT block covers one
# contiguous run of hash columns = one gene at one collection time, so block does NOT depend
# on hash plate (cep290@18 is Bl2 on both P02 and P18). A timepoint absent from a gene's dict
# means that gene was not collected then (cep290 has no 14hpf; b9d2 has no 24hpf).
# Verified against the sequencing metadata: all 27 (experimental gene, timepoint) groups are
# single-block.
#
# The three crispant targets were pooled into ONE block, so they all read Bl1 at every
# timepoint; "crispant" is the name the imaging labels use when the specific target is not
# broken out, and it maps to that same pooled block.
RT_BLOCK_BY_GENE_AND_TIMEPOINT: dict[str, dict[int, str]] = {
    "cep290": {
        18: "Bl2",
        24: "Bl3",
        30: "Bl4",
        48: "Bl5",
    },
    "b9d2": {
        14: "Bl6",
        18: "Bl7",
        30: "Bl8",
        48: "Bl9",
    },
    "foxj1a": {
        18: "Bl1",
        24: "Bl1",
        30: "Bl1",
        48: "Bl1",
    },
    "ift88": {
        18: "Bl1",
        24: "Bl1",
        30: "Bl1",
        48: "Bl1",
    },
    "sspo": {
        18: "Bl1",
        24: "Bl1",
        30: "Bl1",
        48: "Bl1",
    },
    "crispant": {
        18: "Bl1",
        24: "Bl1",
        30: "Bl1",
        48: "Bl1",
    },
}


# ------------------------------------------------------------------ small named types
class HashLocation(NamedTuple):
    """Where an imaging well ended up in hash space -- the full sequencing coordinate.

    All three come from the plate's own Excel sheets, so an imaging well maps to a sequencing
    embryo without consulting gene or timepoint at all.
    """
    hash_well: str
    hash_plate: str | None
    rt_block: str | None


class HashCoordinate(NamedTuple):
    """The three things an embryo_ID is made of: GENE14_{plate}_{well}_{rt_block}.

    This triple is the sequencing embryo's identity, so it is unique by construction -- which
    is the whole reason the lookup below needs no gene, no timepoint, and no tie-break.
    """
    plate: str
    well: str
    rt_block: str




# ------------------------------------------------------------------ helpers
def norm_well(value: object) -> str | None:
    """'A09' / 'a9' / ' A9 ' -> 'A9' (bare, uppercased). Missing -> None.

    Anything that does not look like a well is passed through cleaned-but-unchanged, so an
    oddity in a sheet stays visible instead of silently becoming None.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().upper()
    match = re.fullmatch(r"([A-H])0*(\d{1,2})", text)
    if match is None:
        return text

    row_letter, column_number = match.groups()
    return f"{row_letter}{int(column_number)}"


def fmt_plate(value: object) -> str | None:
    """18 / '18' / '18.0' -> 'P18'. Missing or blank -> None."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    return f"P{int(float(text)):02d}"


def _grid_to_map(df: pd.DataFrame) -> dict[str, str]:
    """8x12 plate grid (row0 = column numbers, col0 = row letters) -> {imaging_well: value}.

    Reads whatever cells are present, on the sheet's own terms — a deliberately skipped
    column (e.g. plate03's column 4) is simply absent, not inferred.
    """
    well_map: dict[str, str] = {}

    column_headers = df.iloc[0, 1:]      # the plate's column numbers, 1..12
    plate_rows = df.iloc[1:]             # everything below the header row

    for _, plate_row in plate_rows.iterrows():
        row_label = plate_row.iloc[0]
        if pd.isna(row_label):
            continue
        row_letter = str(row_label).strip().upper()

        for column_position, column_header in enumerate(column_headers, start=1):
            value = plate_row.iloc[column_position]
            if pd.isna(value) or not str(value).strip():
                continue
            try:
                column_number = int(float(column_header))
            except (ValueError, TypeError):
                continue
            well_map[f"{row_letter}{column_number}"] = str(value).strip()

    return well_map


def experiment_from_excel(path: Path) -> str:
    name = path.stem
    return name[: -len("_well_metadata")] if name.endswith("_well_metadata") else name


# ------------------------------------------------------------------ 1. per-plate map
def _read_grid_sheet(workbook: pd.ExcelFile, sheet_name: str) -> dict[str, str]:
    """One plate-grid sheet as {imaging_well: value}. Missing sheet -> empty (same as blank)."""
    if sheet_name not in workbook.sheet_names:
        return {}
    return _grid_to_map(workbook.parse(sheet_name, header=None))


def load_plate_maps(excel_dir: Path = EXCEL_DIR) -> dict[str, dict[str, HashLocation]]:
    """{experiment: {imaging_well: HashLocation}} for EVERY plate.

    Two regimes (see module docstring #1, #2):
      * populated image_to_hash_map -> reformatted: imaging_well -> hash_well from the sheet.
      * blank image_to_hash_map     -> identity:    hash_well == imaging_well.
    hash_plate ALWAYS comes from hash_plate_num (populated for both regimes).
    """
    plate_maps: dict[str, dict[str, HashLocation]] = {}

    for excel_path in sorted(excel_dir.glob("*_well_metadata.xlsx")):
        workbook = pd.ExcelFile(excel_path)
        hash_wells = _read_grid_sheet(workbook, "image_to_hash_map")
        hash_plates = _read_grid_sheet(workbook, "hash_plate_num")
        rt_blocks = _read_grid_sheet(workbook, "rt_block")

        if not hash_plates:              # no plate info -> cannot place these embryos
            continue

        # One sheet per coordinate part, all keyed by imaging well:
        #   hash_plate_num    -> hash plate   (also decides WHICH wells this plate placed;
        #                                      no plate means no sequencing coordinate)
        #   image_to_hash_map -> hash well    (absent = the well did not move)
        #   rt_block          -> RT block
        locations: dict[str, HashLocation] = {}
        for imaging_well in hash_plates:
            hash_well = hash_wells.get(imaging_well, imaging_well)

            locations[norm_well(imaging_well)] = HashLocation(
                hash_well=norm_well(hash_well),
                hash_plate=fmt_plate(hash_plates.get(imaging_well)),
                rt_block=rt_blocks.get(imaging_well),
            )

        plate_maps[experiment_from_excel(excel_path)] = locations

    return plate_maps


def build_crosswalk(excel_dir: Path = EXCEL_DIR) -> pd.DataFrame:
    """Flat table: one row per (experiment, imaging_well) -> hash_well, hash_plate."""
    rows = [
        {"experiment": experiment,
         "imaging_well": imaging_well,
         "hash_well": location.hash_well,
         "hash_plate": location.hash_plate,
         "rt_block": location.rt_block}
        for experiment, locations in load_plate_maps(excel_dir).items()
        for imaging_well, location in locations.items()
    ]
    return pd.DataFrame(rows).sort_values(["experiment", "imaging_well"]).reset_index(drop=True)


# ------------------------------------------------------------------ 2. collection time
def collection_time_hpf(experiment: str, plate_start_age: float | int | None) -> float | int | None:
    """Sequencing collection time. Plate start age, EXCEPT '30to48' plates -> 48 (docstring #5).

    Nothing in this module calls this — it is the `timepoint` you pass to resolve_seq_embryo_id.
    Note the argument is the RAW plate start age, which callers often read from a column that
    happens to share this function's name; the two are not the same number for 30to48 plates.
    """
    if "30to48" in str(experiment):
        return 48
    return plate_start_age


# ------------------------------------------------------------------ 3. sequencing lookup
EMBRYO_ID_RE = re.compile(r"^GENE14_(?P<plate>P\d+)_(?P<well>[A-H]\d+)_(?P<rt_block>Bl\d+)$")


def parse_embryo_id(embryo_id: str) -> HashCoordinate | None:
    """'GENE14_P18_F10_Bl2' -> HashCoordinate('P18', 'F10', 'Bl2'). Unparseable -> None.

    TEMPORARY (2026-07-28) -- see the note at the top of this module. The ID STRING is the
    identity, and it has stayed correct through metadata column errors that it did not share,
    so we take the coordinate from here rather than from hash_plate/hash_well/rt_block. Revisit
    once the sequencing-side metadata fix has landed and been verified.
    """
    m = EMBRYO_ID_RE.match(str(embryo_id).strip())
    if m is None:
        return None
    return HashCoordinate(plate=m.group("plate"),
                          well=norm_well(m.group("well")),
                          rt_block=m.group("rt_block"))


def build_seq_index(meta: pd.DataFrame | None = None) -> dict[HashCoordinate, str]:
    """Index the sequencing metadata by the coordinate PARSED OUT OF each embryo_ID.

    (plate, well, rt_block) is exactly what embryo_ID is made of, so this key is unique by
    construction -- one embryo per coordinate, no candidate lists and no tie-break. Raises if
    that ever stops holding, rather than resolving embryos to a silently-arbitrary twin.
    """
    if meta is None:
        meta = pd.read_csv(SEQ_METADATA_TSV, sep="\t")

    seq_index: dict[HashCoordinate, str] = {}
    unparsed: list[str] = []

    for embryo_id in meta["embryo_ID"].astype(str):
        coordinate = parse_embryo_id(embryo_id)
        if coordinate is None:
            unparsed.append(embryo_id)
            continue
        if coordinate in seq_index:
            raise ValueError(
                f"embryo_ID is not unique per coordinate: {embryo_id} collides with "
                f"{seq_index[coordinate]} at {coordinate}"
            )
        seq_index[coordinate] = embryo_id

    if unparsed:
        raise ValueError(
            f"{len(unparsed)} embryo_ID(s) do not match {EMBRYO_ID_RE.pattern!r}, so their "
            f"coordinate cannot be trusted: {unparsed[:5]}"
        )

    return seq_index


def rt_block_for(gene: str, collection_time_hpf) -> str | None:
    """(gene, collection time) -> RT block, per the bench RT plate map.

    CROSS-CHECK ONLY -- the plate Excels now carry an `rt_block` sheet, so resolution reads the
    block from there. Kept to verify the sheets against the bench map; see the note on
    RT_BLOCK_BY_GENE_AND_TIMEPOINT.
    """
    if gene is None or collection_time_hpf is None or pd.isna(collection_time_hpf):
        return None
    blocks_by_timepoint = RT_BLOCK_BY_GENE_AND_TIMEPOINT.get(str(gene).lower(), {})
    return blocks_by_timepoint.get(int(collection_time_hpf))


def resolve_seq_embryo_id(hash_plate: str | None, hash_well: str | None, rt_block: str | None,
                          seq_index: dict[HashCoordinate, str]) -> str | None:
    """Look up one fully-resolved coordinate -> its sequencing embryo_ID, else None.

    An exact lookup and nothing else: the coordinate IS what embryo_ID is made of, so a hit is
    exact and a miss is honest. Deriving the rt_block is the caller's job (rt_block_for), which
    keeps "which block is this?" separate from "which embryo is at this coordinate?".
    """
    if not hash_plate or not hash_well or not rt_block:
        return None

    coordinate = HashCoordinate(plate=hash_plate, well=norm_well(hash_well), rt_block=rt_block)
    return seq_index.get(coordinate)


if __name__ == "__main__":
    crosswalk = build_crosswalk()
    print(f"Crosswalk: {len(crosswalk)} (experiment, imaging_well) rows across "
          f"{crosswalk['experiment'].nunique()} experiments")
    print(crosswalk.head(12).to_string(index=False))
