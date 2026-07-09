"""
0 - Load and clean datasets for sequenced-only SCI cilia QC.

This script does not fit models and does not make plots.

It answers:
    1. Which reference files are we using?
    2. Which query plates are we using?
    3. Which plate metadata Excel files were copied here for provenance?
    4. Which embryos are sequenced?
    5. What labels will later scripts use?

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/0_load_and_clean_datasets.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the local cilia_qc_helpers module
from cilia_qc_helpers import select_for_label_transfer  # noqa: E402,F401


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]

BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
BUILD04_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build04_output"
STITCHED_FF  = PROJECT_ROOT / "morphseq_playground" / "built_image_data" / "stitched_FF_images"
PLATE_METADATA_DIR = PROJECT_ROOT / "metadata" / "plate_metadata"
EXPERIMENT_LIST = PROJECT_ROOT / "src/run_morphseq_pipeline/run_experiment_lists/20260605_sci_cilia_qc_first_pass.txt"

TABLE_DIR = RUN_DIR / "tables"
EXCEL_COPY_DIR = RUN_DIR / "source_plate_metadata_excels"
TABLE_DIR.mkdir(exist_ok=True)
EXCEL_COPY_DIR.mkdir(exist_ok=True)

SEQ_QC_FAILURES_CSV = TABLE_DIR / "failed_sequencing_qc" / "missing_embryos_due_to_sequencing.csv"

# This is intentionally a plain string so provenance is stable in the output table.
COPIED_ON = "2026-06-09"
COPY_EXCEL_FILES = False  # Set True only when intentionally refreshing provenance copies.


# -----------------------------------------------------------------------------
# Reference datasets
# -----------------------------------------------------------------------------

REFERENCE_FILES = {
    "b9d2": PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv",
    "cep290": PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv",
    "crispant": BUILD06_DIR / "df03_final_output_with_latents_20260202.csv",
}


# -----------------------------------------------------------------------------
# Label conventions
# -----------------------------------------------------------------------------

GENOTYPE_RENAME = {
    "b9d2_homo": "b9d2_homozygous",
    "b9d2_homozygous": "b9d2_homozygous",
    "b9d2_het": "b9d2_heterozygous",
    "b9d2_heterozygous": "b9d2_heterozygous",
    "b9d2_wt": "b9d2_wildtype",
    "b9d2_wildtype": "b9d2_wildtype",
    "b9d2_unknown": "b9d2_unknown",
    "b9d2_uncertain": "b9d2_unknown",
    "cep290_homo": "cep290_homozygous",
    "cep290_homozygous": "cep290_homozygous",
    "cep290_het": "cep290_heterozygous",
    "cep290_heterozygous": "cep290_heterozygous",
    "cep290_wt": "cep290_wildtype",
    "cep290_wildtype": "cep290_wildtype",
    "cep290_unknown": "cep290_unknown",
    "cep290_uncertain": "cep290_unknown",
    "foxj1a_crispant": "foxj1a_crispant",
    "ift88_crispant": "ift88_crispant",
    "sspo_crispant": "sspo_crispant",
    "scospondin_crispant": "sspo_crispant",
    "if88_ift74_crispant": "ift88_ift74_crispant",
    "ab": "ab_wildtype",
    "inj-ctrl": "injection_control",
    "unknown": "unknown",
    "uncertain": "unknown",
}

HOMO_PHENOTYPE_LABELS = {
    "b9d2": ["CE", "HTA"],
    "cep290": ["High_to_Low", "Low_to_High"],
}

WELLS = [f"{row}{col:02}" for row in "ABCDEFGH" for col in range(1, 13)]


# -----------------------------------------------------------------------------
# Imaging QC constants
# -----------------------------------------------------------------------------

REAL_QC_FLAGS = [
    "frame_flag", "sam2_qc_flag", "no_yolk_flag", "focus_flag",
    "bubble_flag", "dead_flag", "dead_flag2", "sa_outlier_flag",
]


# -----------------------------------------------------------------------------
# Sequencing QC: imaging-plate → experiment_id mapping
# -----------------------------------------------------------------------------
# (imaging_plate, perturbation) -> list[experiment_id]
# Copied verbatim from 3a_audit_sequenced_coverage.py (verified correct).
SEQ_QC_IMAGING_PLATE_MAP: dict[tuple[str, str], list[str]] = {
    # crispant
    ("260319_plate_1", "crispants"):         ["20260319_cilia_crispant_18hpf"],
    ("260319_plate_2", "crispants"):         ["20260319_cilia_crispant_24hpf"],
    ("260319_plate_3", "crispants"):         ["20260319_cilia_crispant_30hpf"],
    ("260320_plate_4", "crispants"):         ["20260320_cilia_crispant_48hpf"],
    # cep290
    ("260414_plate_3", "18hpf_cep290_mut"):  ["20260415_cep290_18hpf_plate03"],
    ("260414_plate_3", "18hpf_cep290_wt"):   ["20260415_cep290_18hpf_plate03"],
    ("260401_plate_2", "24hpf_cep290_wt"):   ["20260324_cep290_18hpf_24hpf_plate02"],
    ("260324_plate_1", "30hpf_cep290_mut"):  ["20260324_cep290_30hpf_plate01"],
    # 48hpf cep290: snapshot + sci timeseries
    ("260416_plate_1", "48hpf_cep290_mut"):  ["20260416_cep290_30to48hpf_plate01_t02", "20260415_sci_cep290_48hpf_plate01"],
    ("260416_plate_1", "48hpf_cep290_wt"):   ["20260416_cep290_30to48hpf_plate01_t02", "20260415_sci_cep290_48hpf_plate01"],
    ("260416_plate_1", "48hpf_cep290_AB"):   ["20260416_cep290_30to48hpf_plate01_t02", "20260415_sci_cep290_48hpf_plate01"],
    # b9d2
    ("260414_plate_1", "14hpf_b9d2_mut"):    ["20260414_b9d2_14hpf_plate01"],
    ("260414_plate_1", "14hpf_b9d2_wt"):     ["20260414_b9d2_14hpf_plate01"],
    ("260414_plate_1", "14hpf_b9d2_AB"):     ["20260414_b9d2_14hpf_plate01"],
    ("260415_plate_1", "30hpf_b9d2_mut"):    ["20260414_b9d2_30hpf_plate01"],
    ("260415_plate_1", "30hpf_b9d2_wt"):     ["20260414_b9d2_30hpf_plate01"],
    # 48hpf b9d2: snapshot + sci timeseries
    ("260414_plate_1", "48hpf_b9d2_mut"):    ["20260415_b9d2_30to48hpf_plate01_t02", "20260414_sci_b9d2_48hpf_plate01"],
    ("260414_plate_1", "48hpf_b9d2_wt"):     ["20260415_b9d2_30to48hpf_plate01_t02", "20260414_sci_b9d2_48hpf_plate01"],
    ("260414_plate_1", "48hpf_b9d2_AB"):     ["20260415_b9d2_30to48hpf_plate01_t02", "20260414_sci_b9d2_48hpf_plate01"],
    ("260414_plate_2", "48hpf_b9d2_mut"):    ["20260415_b9d2_30to48hpf_plate02_t02"],
}

SEQ_QC_DATE_WARNINGS: dict[tuple[str, str], str] = {
    ("260401_plate_2", "24hpf_cep290_wt"):   "date 260401 has no manifest experiment; mapped to 20260324_cep290_18hpf_24hpf_plate02",
    ("260414_plate_3", "18hpf_cep290_mut"):  "date 260414 vs experiment date 260415 for cep290_18hpf_plate03",
    ("260414_plate_3", "18hpf_cep290_wt"):   "date 260414 vs experiment date 260415 for cep290_18hpf_plate03",
    ("260414_plate_2", "48hpf_b9d2_mut"):    "date 260414 vs experiment date 260415 for b9d2_30to48hpf_plate02_t02",
}


# -----------------------------------------------------------------------------
# Collection time and data-source bookkeeping
# -----------------------------------------------------------------------------
#
# LOUD plate01 rule (cep290 & b9d2): plate01 carries BOTH the `_sci_` timeseries AND
# redundant `_t01`/`_t02` 48 hpf snapshots for the SAME physical embryos. The snapshots
# were backups only.
#   - Label transfer uses the `_sci_` TIMESERIES for plate01.
#   - `_t01`/`_t02` snapshots are for portfolio / QC spot-checking ONLY.
#   - For label PREDICTION, plate01 uses only the `_t02` (48 hpf) snapshot.
# `data_source` below encodes which experiments are timeseries vs snapshot so downstream
# scripts can honor this without re-deriving it.
#
# `collection_time_hpf` is a FACT about each experiment (when the cohort was collected),
# NOT something inferred from imaging. It is a manual per-experiment map. The `_sci_`
# timeseries and `30to48` plates are collected at 48 hpf. The ONE genuinely ambiguous
# plate `20260324_cep290_18hpf_24hpf_plate02` holds two collections (18 and 24 hpf) in
# one plate, so it is resolved PER EMBRYO from predicted_stage_hpf (see below).

# Experiments that are continuous timeseries (the plate01 label-transfer source).
TIMESERIES_DATA = [
    "20260414_sci_b9d2_48hpf_plate01",
    "20260415_sci_cep290_48hpf_plate01",
]

# Everything else collected for this cohort is a single-stage snapshot.
SNAPSHOT_DATA = [
    "20260319_cilia_crispant_18hpf",
    "20260319_cilia_crispant_24hpf",
    "20260319_cilia_crispant_30hpf",
    "20260320_cilia_crispant_48hpf",
    "20260324_cep290_18hpf_24hpf_plate02",
    "20260324_cep290_18hpf_plate01",
    "20260324_cep290_24hpf_plate01",
    "20260324_cep290_24hpf_plate02",
    "20260324_cep290_30hpf_plate01",
    "20260324_cep290_30hpf_plate02",
    "20260331_b9d2_18hpf_plate01",
    "20260331_b9d2_18hpf_plate02",
    "20260414_b9d2_14hpf_plate01",
    "20260414_b9d2_30hpf_plate01",
    "20260414_b9d2_30hpf_plate02",
    "20260415_b9d2_30to48hpf_plate01_t02",
    "20260415_b9d2_30to48hpf_plate02_t02",
    "20260415_cep290_18hpf_plate03",
    "20260415_cep290_30to48hpf_plate02_t01",
    "20260416_cep290_30to48hpf_plate01_t02",
    "20260416_cep290_30to48hpf_plate02_t02",
]

# Manual collection-time map (hpf) per query experiment_id. `_sci_`/`30to48` -> 48.
# The split-age plate is intentionally absent here -> resolved per-embryo from stage.
COLLECTION_TIME_HPF = {
    "20260319_cilia_crispant_18hpf": 18,
    "20260319_cilia_crispant_24hpf": 24,
    "20260319_cilia_crispant_30hpf": 30,
    "20260320_cilia_crispant_48hpf": 48,
    "20260324_cep290_18hpf_plate01": 18,
    "20260324_cep290_24hpf_plate01": 24,
    "20260324_cep290_24hpf_plate02": 24,
    "20260324_cep290_30hpf_plate01": 30,
    "20260324_cep290_30hpf_plate02": 30,
    "20260331_b9d2_18hpf_plate01": 18,
    "20260331_b9d2_18hpf_plate02": 18,
    "20260414_b9d2_14hpf_plate01": 14,
    "20260414_b9d2_14hpf_plate02": 14,
    "20260414_b9d2_30hpf_plate01": 30,
    "20260414_b9d2_30hpf_plate02": 30,
    "20260414_sci_b9d2_48hpf_plate01": 48,        # timeseries, collected at 48 hpf
    "20260415_b9d2_30to48hpf_plate01_t02": 48,
    "20260415_b9d2_30to48hpf_plate02_t02": 48,
    "20260415_cep290_18hpf_plate03": 18,
    "20260415_cep290_30to48hpf_plate02_t01": 30,  # _t01 snapshot = 30 hpf
    "20260415_sci_cep290_48hpf_plate01": 48,      # timeseries, collected at 48 hpf
    "20260416_cep290_30to48hpf_plate01_t02": 48,
    "20260416_cep290_30to48hpf_plate02_t02": 48,
}

# The single plate whose name carries two collection ages (18 and 24 hpf only);
# resolved per embryo from predicted_stage_hpf.
SPLIT_AGE_EXPERIMENTS = {"20260324_cep290_18hpf_24hpf_plate02"}


def data_source_for_experiment(experiment_id: str) -> str:
    """timeseries vs snapshot, from the manually-encoded lists. Unknown -> 'snapshot'."""
    if experiment_id in TIMESERIES_DATA:
        return "timeseries"
    if experiment_id in SNAPSHOT_DATA or experiment_id in SPLIT_AGE_EXPERIMENTS:
        return "snapshot"
    return "snapshot"


def split_age_collection_hpf(stage: float) -> int | float:
    """Split-age plate holds ONLY 18 and 24 hpf collections. Match exactly; NaN otherwise."""
    rounded = int(round(stage))
    if rounded == 18:
        return 18
    if rounded == 24:
        return 24
    return np.nan


def plate_token(experiment_id: str) -> str:
    """Pull the plateNN token from an experiment_id; default plate01 if absent."""
    for part in experiment_id.split("_"):
        if part.startswith("plate"):
            return part
    return "plate01"




def gene_for_experiment(experiment: str) -> str | None:
    """Route an experiment name to the gene-specific analysis group."""
    if "b9d2" in experiment:
        return "b9d2"
    if "cep290" in experiment:
        return "cep290"
    if "crispant" in experiment:
        return "crispant"
    return None


def zygosity_from_genotype(genotype: str) -> str | None:
    """Extract the zygosity-only label from a gene-specific genotype label."""
    genotype = str(genotype)
    for label in ["homozygous", "heterozygous", "wildtype", "unknown"]:
        if genotype.endswith(label):
            return label
    return None


def standardize_genotype(raw_genotype: object, gene: str | None) -> str:
    """Convert raw genotype calls into gene-specific labels like b9d2_homozygous."""
    value = str(raw_genotype).strip()
    if gene in {"b9d2", "cep290"} and value in {"unknown", "uncertain"}:
        value = f"{gene}_unknown"
    return GENOTYPE_RENAME.get(value, value)


def clean_phenotype(gene: str, phenotype: object) -> object:
    """Pool phenotype labels used downstream: b9d2 BA_rescue -> HTA; cep290 Intermediate -> Low_to_High."""
    if pd.isna(phenotype):
        return phenotype
    value = str(phenotype)
    if gene == "b9d2" and value == "BA_rescue":
        return "HTA"
    if gene == "cep290" and value == "Intermediate":
        return "Low_to_High"
    return value


def read_experiment_list() -> pd.DataFrame:
    """Read the query plate list and mark gene, sci timelapse status, and build06 availability."""
    rows = []
    for line in EXPERIMENT_LIST.read_text().splitlines():
        experiment = line.strip()
        if not experiment:
            continue
        gene = gene_for_experiment(experiment)
        if gene is None:
            continue
        build06_path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        rows.append(
            {
                "experiment": experiment,
                "gene": gene,
                "is_sci_timelapse": experiment.startswith("sci_") or "_sci_" in experiment,
                "build06_path": str(build06_path),
                "build06_exists": build06_path.exists(),
            }
        )
    return pd.DataFrame(rows)


def find_plate_metadata_excel(experiment: str) -> Path | None:
    """Find the source plate metadata Excel for one experiment."""
    for path in [
        PLATE_METADATA_DIR / f"{experiment}_well_metadata.xlsx",
        PLATE_METADATA_DIR / f"{experiment}.xlsx",
    ]:
        if path.exists():
            return path
    return None


def copy_plate_metadata_excels(experiment_manifest: pd.DataFrame) -> pd.DataFrame:
    """Copy source plate metadata Excels into this folder and return a provenance table."""
    rows = []
    for experiment in experiment_manifest["experiment"]:
        source = find_plate_metadata_excel(experiment)
        if source is None:
            rows.append(
                {
                    "experiment": experiment,
                    "source_excel": "",
                    "copied_excel": "",
                    "copied_on": COPIED_ON,
                    "status": "missing",
                }
            )
            continue
        copied = EXCEL_COPY_DIR / source.name
        shutil.copy2(source, copied)
        rows.append(
            {
                "experiment": experiment,
                "source_excel": str(source),
                "copied_excel": str(copied),
                "copied_on": COPIED_ON,
                "status": "copied",
            }
        )
    return pd.DataFrame(rows)


def read_sequenced_grid(experiment: str) -> dict[str, int] | None:
    """Read the Excel sequenced sheet as well -> code, where 0 is not sequenced and 1 or 2 is sequenced."""
    path = find_plate_metadata_excel(experiment)
    if path is None:
        return None
    with pd.ExcelFile(path) as xlf:
        if "sequenced" not in xlf.sheet_names:
            return None
        sheet = xlf.parse("sequenced", header=0)
    block = sheet.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
    values = block.to_numpy(dtype=str).ravel()
    out = {}
    for well, raw in zip(WELLS, values):
        text = raw.strip()
        try:
            out[well] = int(float(text)) if text not in {"", "nan"} else 0
        except ValueError:
            out[well] = 0
    return out


def load_query_table(row: pd.Series) -> pd.DataFrame:
    """Load one query build06 table and add genotype_clean, zygosity, phenotype_clean, and sequencing labels."""
    experiment = row["experiment"]
    gene = row["gene"]
    path = Path(row["build06_path"])
    df = pd.read_csv(path, low_memory=False)
    df["experiment"] = experiment
    df["gene"] = gene
    # genotype_clean keeps the gene name plus zygosity, e.g. b9d2_homozygous.
    # zygosity is the zygosity-only helper column used for genotype QC.
    df["genotype_clean"] = df["genotype"].map(lambda x: standardize_genotype(x, gene))
    df["zygosity"] = df["genotype_clean"].map(zygosity_from_genotype)
    if "cluster_categories" in df.columns:
        df["phenotype_clean"] = df["cluster_categories"].map(lambda x: clean_phenotype(gene, x))
    else:
        df["phenotype_clean"] = np.nan

    # collection_time_hpf: manual per-experiment fact, EXCEPT the split-age plate which
    # holds two collections (18 & 24 hpf) and is resolved per embryo from stage.
    df["data_source"] = data_source_for_experiment(experiment)
    if experiment in SPLIT_AGE_EXPERIMENTS:
        emb_stage = df.groupby("embryo_id")["predicted_stage_hpf"].transform("median")
        df["collection_time_hpf"] = emb_stage.map(
            lambda s: split_age_collection_hpf(s) if pd.notna(s) else np.nan
        )
        # Every embryo on this plate must resolve to exactly 18 or 24; surface any that didn't.
        unassigned = df.loc[df["collection_time_hpf"].isna(), "embryo_id"].unique()
        if len(unassigned):
            print(
                f"    WARNING: {experiment}: {len(unassigned)} embryo(s) matched neither "
                f"18 nor 24 hpf -> NaN: {list(unassigned)[:5]}"
            )
        print(
            f"    {experiment}: split-age plate -> collection_time_hpf resolved per embryo "
            f"from predicted_stage_hpf ({df['collection_time_hpf'].dropna().astype(int).value_counts().to_dict()})"
        )
    else:
        ct = COLLECTION_TIME_HPF.get(experiment)
        if ct is None:
            print(f"    WARNING: {experiment} has no collection_time_hpf mapping -> NaN")
            df["collection_time_hpf"] = np.nan
        else:
            df["collection_time_hpf"] = ct

    # physical_embryo_id keys on biological collection time (NOT experiment_id), so the
    # plate01 timeseries row and its _t02 48 hpf snapshot row for one physical embryo join.
    ct_token = df["collection_time_hpf"].map(
        lambda x: f"{int(x)}hpf" if pd.notna(x) else "naHpf"
    )
    plate = plate_token(experiment)
    df["physical_embryo_id"] = (
        gene + "_" + ct_token + "_" + plate + "_" + df["well"].astype(str)
    )

    sequenced_grid = read_sequenced_grid(experiment)
    if sequenced_grid is None:
        df["sequenced"] = np.nan
    else:
        df["sequenced"] = df["well"].map(sequenced_grid).fillna(0).astype(int)

    df["sequenced_stratum"] = ""
    df.loc[(df["sequenced"] > 0) & (df["genotype_clean"] == "ab_wildtype"), "sequenced_stratum"] = "AB"
    df.loc[(df["sequenced"] == 1) & (df["sequenced_stratum"] == ""), "sequenced_stratum"] = "wildtype_sibling"
    df.loc[(df["sequenced"] == 2) & (df["zygosity"] == "heterozygous"), "sequenced_stratum"] = "heterozygous"
    df.loc[(df["sequenced"] == 2) & (df["zygosity"] == "homozygous"), "sequenced_stratum"] = "homozygous"
    df.loc[(df["sequenced"] == 2) & (df["sequenced_stratum"] == ""), "sequenced_stratum"] = "mutant_unresolved"
    return df


def load_reference_table(gene: str, path: Path) -> pd.DataFrame:
    """Load one reference table and apply the same genotype and phenotype label cleanup."""
    df = pd.read_csv(path, low_memory=False)
    df["gene"] = gene
    # genotype_clean keeps the gene name plus zygosity, e.g. b9d2_homozygous.
    # zygosity is the zygosity-only helper column used for genotype QC.
    df["genotype_clean"] = df["genotype"].map(lambda x: standardize_genotype(x, gene))
    df["zygosity"] = df["genotype_clean"].map(zygosity_from_genotype)
    if "cluster_categories" in df.columns:
        df["phenotype_clean"] = df["cluster_categories"].map(lambda x: clean_phenotype(gene, x))
    else:
        df["phenotype_clean"] = np.nan
    # The collection bookkeeping is a query-cohort concern; reference experiment_ids are
    # bare dates with no age/plate/well structure. Emit the columns for schema parity but
    # leave them unset (label transfer trains on labels + features, not collection time).
    df["data_source"] = "reference"
    df["collection_time_hpf"] = np.nan
    df["physical_embryo_id"] = np.nan
    return df


# -----------------------------------------------------------------------------
# QC helpers
# -----------------------------------------------------------------------------

def _norm_well(w: str) -> str:
    """Normalize bare well format (E3) to zero-padded build06 format (E03)."""
    return re.sub(r"([A-Ha-h])(\d)$", lambda m: m.group(1).upper() + m.group(2).zfill(2), w.strip())


def _stitched_image_exists(exp: str, well: str) -> bool:
    d = STITCHED_FF / exp
    if not d.is_dir():
        return False
    return any(d.glob(f"{well}_*stitch*"))


def read_imaging_qc(exp: str) -> dict[str, dict]:
    """Return {well: {imaging_qc_status, exclusion_flags, embryo_id}} for every sequenced well.

    Logic mirrors 3a audit() — build04 CSV is the source of truth.
    Status values: OK / EXCLUDED / ABSENT_IMAGED / ABSENT_NO_IMAGE / QC_NOT_RUN
    """
    grid = read_sequenced_grid(exp)
    if grid is None:
        return {}
    seq_wells = {w for w, v in grid.items() if v in (1, 2)}

    b04_path = BUILD04_DIR / f"qc_staged_{exp}.csv"
    b04 = pd.read_csv(b04_path) if b04_path.exists() else None
    b04_wells = set(b04["well"].astype(str).str.strip()) if b04 is not None else set()

    out: dict[str, dict] = {}
    for w in seq_wells:
        embryo_id, flags = "", ""
        if b04 is None:
            status = "QC_NOT_RUN"
        elif w not in b04_wells:
            status = "ABSENT_IMAGED" if _stitched_image_exists(exp, w) else "ABSENT_NO_IMAGE"
        else:
            row = b04[b04["well"].astype(str).str.strip() == w].iloc[0]
            embryo_id = str(row.get("embryo_id", "") or "")
            use_ok = bool(row.get("use_embryo_flag", False))
            fired = [f for f in REAL_QC_FLAGS if bool(row.get(f, False))]
            status = "OK" if use_ok else "EXCLUDED"
            flags = "|".join(fired)
        if not embryo_id:
            embryo_id = f"{exp}_{w}"
        out[w] = {"imaging_qc_status": status, "exclusion_flags": flags, "embryo_id": embryo_id}
    return out


def load_seq_qc_failures(query_all_rows: pd.DataFrame) -> set[tuple[str, str]]:
    """Return set of (experiment, well) pairs that failed sequencing QC.

    Reads missing_embryos_due_to_sequencing.csv and resolves to experiment_id
    via SEQ_QC_IMAGING_PLATE_MAP. Mirrors 3a load_seq_qc_failures() but returns
    a simpler (exp, well) set — sufficient for the final_usable gate in 0.
    """
    if not SEQ_QC_FAILURES_CSV.exists():
        print(f"  NOTE: seq-QC failures CSV not found at {SEQ_QC_FAILURES_CSV.relative_to(RUN_DIR)}; no seq failures applied")
        return set()

    src = pd.read_csv(SEQ_QC_FAILURES_CSV, dtype=str).fillna("")
    warned_keys: set[tuple[str, str]] = set()
    fail_pairs: set[tuple[str, str]] = set()

    for _, r in src.iterrows():
        plate = r["imaging_plate"].strip()
        pert  = r["perturbation"].strip()
        well  = _norm_well(r["imaging_well"].strip())
        key   = (plate, pert)

        if key in SEQ_QC_DATE_WARNINGS and key not in warned_keys:
            print(f"  WARNING seq-QC date mismatch: {plate} / {pert} — {SEQ_QC_DATE_WARNINGS[key]}")
            warned_keys.add(key)

        exp_ids = SEQ_QC_IMAGING_PLATE_MAP.get(key)
        if exp_ids is None:
            print(f"  WARNING seq-QC unmapped: ({plate}, {pert}) — skipping row {r['embryo_ID']}")
            continue
        for exp_id in exp_ids:
            fail_pairs.add((exp_id, well))

    print(f"  seq-QC failures resolved: {len(fail_pairs)} (experiment, well) pairs")
    return fail_pairs


def build_qc_registry(
    query_all_rows: pd.DataFrame,
    seq_fail_pairs: set[tuple[str, str]],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Build qc_registry.csv: one row per sequenced well with imaging + seq QC status.

    Spine: ALL (experiment, well) pairs with sequenced > 0 from the Excel sequenced sheet —
    including imaging-failed embryos that never reached build06. This is the correct universe
    for the Venn / loss accounting; build06 survivors are a strict subset.

    Joined to: build04 imaging QC, build06 latent existence, seq QC.
    final_usable = (imaging_qc_status == "OK") & (seq_qc_status == "seq_ok")
    """
    # One row per embryo_id from build06 survivors (for has_latents + metadata)
    b06_dedup = query_all_rows.drop_duplicates("embryo_id")[
        ["embryo_id", "experiment", "well", "physical_embryo_id",
         "collection_time_hpf", "data_source", "gene",
         "sequenced", "sequenced_stratum", "genotype_clean", "zygosity"]
    ]
    b06_lookup: dict[tuple[str, str], dict] = {
        (r["experiment"], r["well"]): r.to_dict()
        for _, r in b06_dedup.iterrows()
    }

    non_sci_exps = set(manifest.loc[~manifest["is_sci_timelapse"], "experiment"])
    gene_map = dict(zip(manifest["experiment"], manifest["gene"]))

    rows = []
    for _, mrow in manifest.iterrows():
        exp  = mrow["experiment"]
        gene = mrow["gene"]

        grid = read_sequenced_grid(exp)
        if grid is None:
            continue
        seq_wells = {w: v for w, v in grid.items() if v in (1, 2)}
        if not seq_wells:
            continue

        # Imaging QC map for this experiment (non-sci only; sci treated as OK)
        if exp in non_sci_exps:
            qc_map = read_imaging_qc(exp)
        else:
            qc_map = {}

        for well, seq_code in seq_wells.items():
            b06 = b06_lookup.get((exp, well))
            has_latents = b06 is not None

            if exp in non_sci_exps:
                qc_info = qc_map.get(well, {
                    "imaging_qc_status": "QC_NOT_RUN",
                    "exclusion_flags": "",
                    "embryo_id": f"{exp}_{well}",
                })
            else:
                qc_info = {
                    "imaging_qc_status": "OK",
                    "exclusion_flags": "",
                    "embryo_id": b06["embryo_id"] if b06 else f"{exp}_{well}",
                }

            embryo_id          = b06["embryo_id"]          if b06 else qc_info["embryo_id"]
            physical_embryo_id = b06["physical_embryo_id"] if b06 else ""
            collection_time_hpf = b06["collection_time_hpf"] if b06 else np.nan
            data_source        = b06["data_source"]        if b06 else ""
            genotype_clean     = b06["genotype_clean"]     if b06 else ""
            zygosity           = b06["zygosity"]           if b06 else ""
            sequenced_stratum  = b06["sequenced_stratum"]  if b06 else ""

            seq_qc_status = "seq_failed" if (exp, well) in seq_fail_pairs else "seq_ok"

            rows.append({
                "embryo_id":           embryo_id,
                "physical_embryo_id":  physical_embryo_id,
                "experiment":          exp,
                "gene":                gene,
                "well":                well,
                "sequenced":           int(seq_code),
                "sequenced_stratum":   sequenced_stratum,
                "genotype_clean":      genotype_clean,
                "zygosity":            zygosity,
                "collection_time_hpf": collection_time_hpf,
                "data_source":         data_source,
                "imaging_qc_status":   qc_info["imaging_qc_status"],
                "exclusion_flags":     qc_info["exclusion_flags"],
                "has_latents":         has_latents,
                "seq_qc_status":       seq_qc_status,
                "final_usable":        (qc_info["imaging_qc_status"] == "OK") and (seq_qc_status == "seq_ok"),
            })

    return pd.DataFrame(rows)


def main() -> None:
    print("0 - load and clean datasets")
    print("Reference: all valid labeled reference embryos")
    print("Query: sequenced embryos only for downstream analysis")

    experiment_manifest = read_experiment_list()
    experiment_manifest.to_csv(TABLE_DIR / "experiment_manifest.csv", index=False)

    print(f"\nQuery experiments in manifest: {len(experiment_manifest)}")
    print(experiment_manifest.groupby(["gene", "is_sci_timelapse", "build06_exists"]).size().to_string())

    if COPY_EXCEL_FILES:
        excel_manifest = copy_plate_metadata_excels(experiment_manifest)
        excel_manifest.to_csv(TABLE_DIR / "copied_plate_metadata_excels.csv", index=False)
        print(f"\nCopied plate metadata Excel files on {COPIED_ON}:")
        print(excel_manifest["status"].value_counts().to_string())
    else:
        print("\nCOPY_EXCEL_FILES is False; not copying or overwriting plate metadata Excels.")
        print(f"Existing copied files, if any, stay in: {EXCEL_COPY_DIR.relative_to(RUN_DIR)}/")

    query_tables = []
    for _, row in experiment_manifest[experiment_manifest["build06_exists"]].iterrows():
        print(f"  loading query {row['experiment']}")
        query_tables.append(load_query_table(row))
    query = pd.concat(query_tables, ignore_index=True)

    # All sequenced wells from build06 survivors (pre-QC-gate spine for registry).
    sequenced_embryos_all = (
        query[query["sequenced"] > 0]
        .drop_duplicates("embryo_id")
        [[
            "embryo_id",
            "experiment",
            "gene",
            "well",
            "sequenced",
            "sequenced_stratum",
            "genotype_clean",
            "zygosity",
        ]]
        .sort_values(["gene", "experiment", "well"])
    )

    query.to_csv(TABLE_DIR / "query_all_rows_clean.csv", index=False)

    # ------------------------------------------------------------------
    # QC registry: imaging QC + seq QC -> final_usable gate
    # ------------------------------------------------------------------
    print("\n=== QC REGISTRY ===")
    seq_fail_pairs = load_seq_qc_failures(query)
    registry = build_qc_registry(query, seq_fail_pairs, experiment_manifest)
    registry.to_csv(TABLE_DIR / "qc_registry.csv", index=False)
    print(f"  wrote tables/qc_registry.csv ({len(registry)} rows)")

    n_total     = len(registry)
    n_img_fail  = (registry["imaging_qc_status"] != "OK").sum()
    n_seq_fail  = (registry["seq_qc_status"] == "seq_failed").sum()
    n_usable    = registry["final_usable"].sum()
    print(f"  total sequenced wells : {n_total}")
    print(f"  imaging QC failures   : {n_img_fail}")
    print(f"  seq QC failures       : {n_seq_fail}")
    print(f"  final_usable          : {n_usable}")

    print("\nfinal_usable by gene x sequenced_stratum:")
    print(registry[registry["final_usable"]].groupby(["gene", "sequenced_stratum"]).size().to_string())

    # query_sequenced_embryos.csv: ONLY final_usable rows — this is what 1_fit and 2_predict read
    sequenced_embryos = (
        registry[registry["final_usable"]]
        [["embryo_id", "experiment", "gene", "well", "sequenced",
          "sequenced_stratum", "genotype_clean", "zygosity"]]
        .sort_values(["gene", "experiment", "well"])
    )
    sequenced_embryos.to_csv(TABLE_DIR / "query_sequenced_embryos.csv", index=False)
    print(f"\n  wrote tables/query_sequenced_embryos.csv ({len(sequenced_embryos)} final_usable rows)")

    reference_tables = []
    for gene, path in REFERENCE_FILES.items():
        print(f"  loading reference {gene}")
        ref = load_reference_table(gene, path)
        reference_tables.append(ref)
        ref.to_csv(TABLE_DIR / f"reference_{gene}_clean.csv", index=False)
    reference = pd.concat(reference_tables, ignore_index=True)
    reference.to_csv(TABLE_DIR / "reference_all_clean.csv", index=False)

    label_rows = []
    for gene in ["b9d2", "cep290"]:
        label_rows.append(
            {
                "gene": gene,
                "model": "genotype_qc",
                "reference_labels": "wildtype, heterozygous, homozygous",
                "query_scope": "sequenced embryos only",
            }
        )
        label_rows.append(
            {
                "gene": gene,
                "model": "homozygous_phenotype",
                "reference_labels": ", ".join(HOMO_PHENOTYPE_LABELS[gene]),
                "query_scope": "sequenced homozygous embryos for main interpretation",
            }
        )
    label_rows.append(
        {
            "gene": "crispant",
            "model": "genotype_qc",
            "reference_labels": "crispant/control labels",
            "query_scope": "sequenced embryos only if present",
        }
    )
    pd.DataFrame(label_rows).to_csv(TABLE_DIR / "label_plan.csv", index=False)

    print("\nLOUD plate01 rule: `_sci_` experiments are the TIMESERIES (label transfer);")
    print("`_t01`/`_t02` are redundant 48 hpf snapshots (portfolio/QC; prediction uses _t02).")
    print("\ncollection_time_hpf x data_source (query rows):")
    print(query.groupby(["collection_time_hpf", "data_source"]).size().to_string())
    missing_ct = query.loc[query["collection_time_hpf"].isna(), "experiment"].value_counts()
    if not missing_ct.empty:
        print("\nWARNING: query rows with NO collection_time_hpf, by experiment:")
        print(missing_ct.to_string())
    else:
        print("\nAll query rows have a collection_time_hpf.")

    print("\nphysical_embryo_id example (split-age plate, should show 18hpf AND 24hpf):")
    split_eg = query[query["experiment"].isin(SPLIT_AGE_EXPERIMENTS)]
    if not split_eg.empty:
        print(split_eg.drop_duplicates("embryo_id")["physical_embryo_id"].head(4).to_string(index=False))

    print("\nSequenced embryo counts:")
    print(sequenced_embryos.groupby(["gene", "sequenced_stratum"]).size().to_string())

    print("\nReference embryo counts by gene x zygosity:")
    ref_counts = reference.drop_duplicates("embryo_id").groupby(["gene", "zygosity"]).size()
    print(ref_counts.to_string())

    print(f"\nWrote cleaned tables to: {TABLE_DIR.relative_to(RUN_DIR)}/")
    if COPY_EXCEL_FILES:
        print(f"Copied source Excel files to: {EXCEL_COPY_DIR.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
