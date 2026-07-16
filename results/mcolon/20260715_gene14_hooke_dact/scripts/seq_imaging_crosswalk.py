"""
seq_imaging_crosswalk.py — the ONE documented bridge between the imaging world and the
sequencing world for GENE14, so the mapping never has to be re-derived by hand again.

=============================================================================================
TWO ID WORLDS (they share NO id column)
=============================================================================================
  IMAGING  embryo_id      e.g. 20260415_cep290_18hpf_plate03_C01_e01   (date_gene_stage_plate_well_e##)
           physical_embryo_id  e.g. cep290_18hpf_plate01_F10           (gene_stage_plate_well; biology-scoped)
  SEQUENCING embryo_ID    e.g. GENE14_P18_F10_Bl2                      (GENE14_P{hashplate}_{hashwell}_Bl{rtblock})

A sequencing embryo_ID is DEFINED by (hash_plate, hash_well, RT block). It is the physical
identity — "you can only destroy an embryo once", so physical_embryo_id <-> sequencing embryo
is 1:1. The imaging side has redundant images (snapshot + _t01/_t02 backups + _sci timeseries,
up to 3 imaging embryo_ids) that all collapse to ONE physical embryo -> ONE sequencing embryo.

=============================================================================================
WHAT MAPS CLEANLY  (the easy 90%)
=============================================================================================
  * NORMAL plates: the imaging well IS the hash well (no reformatting). image_to_hash_map sheet
    is BLANK. Join is identity: hash_well == imaging_well; hash_plate from the hash_plate_num
    sheet. 15 of 16 phenotyped experiments are this case.
  * embryo_ID strings are IDENTICAL across every table (metadata, QC list, McClintock CDS,
    File A) -> joins on the id string match with zero orphans.

=============================================================================================
THE TRICKY THINGS  (why a naive join silently mismatches ~30% of the interesting embryos)
=============================================================================================
1. REFORMATTED PLATES (imaging well != hash well). Embryos were physically re-pipetted from
   the imaging plate onto a hash plate, so the well changes (C1->C6, A3->A9, ...). This is
   recorded PER PLATE in the Excel `image_to_hash_map` sheet (position = imaging well, value =
   hash well) + `hash_plate_num` sheet (value = hash plate). Only these plates are reformatted:
       crispants 260319_p1/p2, 260320_p4;  cep290 260414_plate_3 (=plate03);  b9d2 14hpf_plate02.
   RULE: populated image_to_hash_map -> use it; blank -> identity.

2. `hash_plate_num` IS POPULATED FOR *BOTH* REGIMES. It gives the hash plate for every embryo,
   normal OR reformatted. You MUST read it for identity plates too — it is what tells P02 from
   P18 and disambiguates the dual-hashing collisions (see #4). (Original bug: only reading it
   for reformatted plates -> 15 embryos unresolved.)

3. plate03 (cep290 18hpf) is IRREGULAR: 8 rows x 5 imaging columns with imaging COLUMN 4
   DELIBERATELY SKIPPED (cols 1,2,3,5,6 -> hash *06,*07,*09,*10,*11). There is NO imaging F4.
   Read the sheet on ITS OWN TERMS. Do NOT validate plate03 against File A: File A recorded
   imaging_well=F4 (etc.) for the rescued embryos, but those File-A imaging wells are part of
   the ORIGINAL error. The Excel is truth. (plate03 has 0 phenotype predictions anyway.)

4. P18->P02 "WRONG PLATE" CORRECTION -> HASH-SPACE COLLISIONS. 11 rescued cep290 embryos had
   their hash_plate corrected P18->P02 in the METADATA COLUMN, but their embryo_ID STRING still
   says P18 (id_plate=P18, hash_plate=P02, cross_batch=5). This pushed each onto a hash
   coordinate ALREADY OCCUPIED by a real plate01/P02 embryo. Result: two DISTINCT physical
   embryos share (gene, timepoint, hash_plate=P02, hash_well) — a naive key returns both.
   TIE-BREAK: the embryo_ID STRING preserves each embryo's ORIGINAL plate (id_plate). Pick the
   twin whose id_plate == the imaging plate's hash_plate (from hash_plate_num). The phenotyped
   embryo came from a specific imaging plate; that routes it to the correct twin.
       (cross_batch is the sequencing-side witness of this — 5 = the rescued group, exactly 11
        cep290 rows — but the imaging side has no cross_batch, so id_plate is the usable key.)

5. COLLECTION TIME != IMAGING TIME for "30to48" plates. plate01/plate02 of cep290 & b9d2 carry
   BOTH a 30hpf and a 48hpf snapshot (_t01/_t02) plus an _sci timeseries — redundant backups of
   the SAME embryos, collected across 30-48 hpf. The sequencing `timepoint` for these is 48.
   RULE: collection_time = plate start age, EXCEPT experiments containing "30to48" -> 48.
   (predicted_stage_hpf stays each embryo's TRUE age; collection_time is only for structure +
   for keying to the sequencing side.) Without this, a _t01 snapshot labeled 30 fails to match
   its embryo, which sequencing filed under collection 48.

=============================================================================================
THE RESOLUTION KEY (put together)
=============================================================================================
  imaging (experiment, well)
    --image_to_hash_map / identity-->  hash_well
    --hash_plate_num-->                hash_plate
    collection_time (30to48 -> 48)-->  timepoint
  Look up sequencing metadata by (gene, timepoint, hash_plate, hash_well); if >1 twin (the
  P18->P02 collisions), tie-break by id_plate == imaging hash_plate.  => one sequencing embryo_ID.

Result on the real data: 169/169 phenotyped embryos resolved.
=============================================================================================

Public API:
    load_plate_maps()  -> {experiment: {imaging_well: (hash_well, hash_plate)}}
    build_crosswalk()  -> DataFrame[experiment, imaging_well, hash_well, hash_plate]
    resolve_seq_embryo_id(gene, timepoint, hash_well, hash_plate, meta_index) -> embryo_ID | None
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------ paths
PROJECT_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
QC_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
EXCEL_DIR = QC_DIR / "source_plate_metadata_excels"
SEQ_METADATA_TSV = Path("/net/seahub_zfish/vol1/data/preprocessed/GENE14/GENE14_embryo_metadata.tsv")

# Reformatted plates carry a populated image_to_hash_map. Listed for documentation; the code
# detects them by whether that sheet has any cells, so this stays in sync automatically.
KNOWN_REFORMATTED = {
    "20260319_cilia_crispant_24hpf", "20260319_cilia_crispant_30hpf",
    "20260320_cilia_crispant_48hpf", "20260324_cep290_18hpf_plate01",
    "20260414_b9d2_14hpf_plate02", "20260415_cep290_18hpf_plate03",
}


# ------------------------------------------------------------------ helpers
def norm_well(w: object) -> str | None:
    """'A09' / 'a9' / ' A9 ' -> 'A9' (bare, uppercased). NaN/empty -> None."""
    if w is None or (isinstance(w, float) and pd.isna(w)):
        return None
    m = re.match(r"^([A-Ha-h])0*([0-9]{1,2})$", str(w).strip())
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else str(w).strip()


def fmt_plate(pnum: object) -> str | None:
    """18 / '18' / '18.0' -> 'P18'. Empty -> None."""
    if pnum is None or (isinstance(pnum, float) and pd.isna(pnum)) or str(pnum).strip() == "":
        return None
    return f"P{int(float(str(pnum).strip())):02d}"


def _grid_to_map(df: pd.DataFrame) -> dict[str, str]:
    """8x12 plate grid (row0 = column numbers, col0 = row letters) -> {imaging_well: value}.

    Reads whatever cells are present, on the sheet's own terms — a deliberately skipped
    column (e.g. plate03's column 4) is simply absent, not inferred.
    """
    out: dict[str, str] = {}
    row_labels = df.iloc[1:, 0].tolist()
    col_labels = df.iloc[0, 1:].tolist()
    for i, rlab in enumerate(row_labels):
        if pd.isna(rlab):
            continue
        for j, clab in enumerate(col_labels):
            val = df.iloc[1 + i, 1 + j]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                col = int(float(clab))
            except (ValueError, TypeError):
                continue
            out[f"{str(rlab).strip().upper()}{col}"] = str(val).strip()
    return out


def experiment_from_excel(path: Path) -> str:
    name = path.stem
    return name[: -len("_well_metadata")] if name.endswith("_well_metadata") else name


# ------------------------------------------------------------------ 1. per-plate map
def load_plate_maps(excel_dir: Path = EXCEL_DIR) -> dict[str, dict[str, tuple[str, str | None]]]:
    """{experiment: {imaging_well: (hash_well, hash_plate)}} for EVERY plate.

    Two regimes (see module docstring #1, #2):
      * populated image_to_hash_map -> reformatted: imaging_well -> hash_well from the sheet.
      * blank image_to_hash_map     -> identity:    hash_well == imaging_well.
    hash_plate ALWAYS comes from hash_plate_num (populated for both regimes).
    """
    maps: dict[str, dict[str, tuple[str, str | None]]] = {}
    reformatted: list[str] = []
    for path in sorted(excel_dir.glob("*_well_metadata.xlsx")):
        experiment = experiment_from_excel(path)
        xl = pd.ExcelFile(path)
        i2h = _grid_to_map(xl.parse("image_to_hash_map", header=None)) \
            if "image_to_hash_map" in xl.sheet_names else {}
        hpn = _grid_to_map(xl.parse("hash_plate_num", header=None)) \
            if "hash_plate_num" in xl.sheet_names else {}
        if not hpn:                      # no plate info -> cannot place embryos
            continue
        entry: dict[str, tuple[str, str | None]] = {}
        if i2h:                          # reformatted
            reformatted.append(experiment)
            for iw, hw in i2h.items():
                entry[norm_well(iw)] = (norm_well(hw), fmt_plate(hpn.get(iw)))
        else:                            # identity
            for iw, pnum in hpn.items():
                entry[norm_well(iw)] = (norm_well(iw), fmt_plate(pnum))
        maps[experiment] = entry
    return maps


def build_crosswalk(excel_dir: Path = EXCEL_DIR) -> pd.DataFrame:
    """Flat table: one row per (experiment, imaging_well) -> hash_well, hash_plate."""
    rows = []
    for experiment, entry in load_plate_maps(excel_dir).items():
        for iw, (hw, hp) in entry.items():
            rows.append({"experiment": experiment, "imaging_well": iw,
                         "hash_well": hw, "hash_plate": hp})
    return pd.DataFrame(rows).sort_values(["experiment", "imaging_well"]).reset_index(drop=True)


# ------------------------------------------------------------------ 2. collection time
def collection_time_hpf(experiment: str, plate_start_age: float | int | None) -> float | int | None:
    """Sequencing collection time. Plate start age, EXCEPT '30to48' plates -> 48 (docstring #5)."""
    if "30to48" in str(experiment):
        return 48
    return plate_start_age


# ------------------------------------------------------------------ 3. sequencing lookup
def build_seq_index(meta: pd.DataFrame | None = None):
    """Index the sequencing metadata for resolution + tie-break.

    Returns (by_full, by_part): dicts keyed on (gene, timepoint, hash_plate, hash_well) and
    (gene, timepoint, hash_well) -> list of {embryo_ID, id_plate}. Multiple entries occur only
    for the P18->P02 collisions (docstring #4).
    """
    if meta is None:
        meta = pd.read_csv(SEQ_METADATA_TSV, sep="\t")
    meta = meta.copy()
    meta["hash_well_n"] = meta["hash_well"].map(norm_well)
    meta["hash_plate_n"] = meta["hash_plate"].astype(str).str.strip()
    meta["gene_key"] = meta["target"].str.lower()
    meta["id_plate"] = meta["embryo_ID"].str.extract(r"GENE14_(P\d+)_")

    by_full: dict[tuple, list] = defaultdict(list)
    by_part: dict[tuple, list] = defaultdict(list)
    for _, m in meta.iterrows():
        tp = int(m["timepoint"]) if pd.notna(m["timepoint"]) else None
        rec = {"embryo_ID": m["embryo_ID"], "id_plate": m["id_plate"]}
        by_full[(m["gene_key"], tp, m["hash_plate_n"], m["hash_well_n"])].append(rec)
        by_part[(m["gene_key"], tp, m["hash_well_n"])].append(rec)
    return by_full, by_part


def _pick(cands: list, imaging_plate: str | None) -> str | None:
    """From candidate metadata rows, pick the twin whose ORIGINAL plate (id_plate) matches the
    imaging plate's hash plate (docstring #4). One candidate -> take it; else tie-break; else None."""
    if len(cands) == 1:
        return cands[0]["embryo_ID"]
    if imaging_plate is not None:
        match = [c for c in cands if c["id_plate"] == imaging_plate]
        if len(match) == 1:
            return match[0]["embryo_ID"]
    return None  # still ambiguous — do not guess


def resolve_seq_embryo_id(gene: str, timepoint, hash_well: str, hash_plate: str | None,
                          by_full, by_part) -> str | None:
    """Map one imaging-resolved embryo -> its sequencing embryo_ID, or None if unresolvable."""
    tp = int(timepoint) if timepoint is not None and not pd.isna(timepoint) else None
    if hash_plate is not None:
        cands = by_full.get((gene, tp, hash_plate, hash_well))
        if cands:
            return _pick(cands, hash_plate)
    cands = by_part.get((gene, tp, hash_well))
    if cands:
        return _pick(cands, hash_plate)
    return None


if __name__ == "__main__":
    cw = build_crosswalk()
    reformatted = sorted(set(cw["experiment"]) & KNOWN_REFORMATTED)
    print(f"Crosswalk: {len(cw)} (experiment, imaging_well) rows across "
          f"{cw['experiment'].nunique()} experiments; reformatted = {reformatted}")
    print(cw.head(12).to_string(index=False))
