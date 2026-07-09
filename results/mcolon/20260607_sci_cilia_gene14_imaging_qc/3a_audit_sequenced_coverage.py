"""
3a - Sequenced-vs-pipeline coverage audit with real failure modes.

"The Excel `sequenced` sheet says these wells were sequenced — what actually happened to each one
in the pipeline?" For every non-sci_ b9d2/cep290 plate, read the `sequenced` sheet and the build04
qc_staged CSV, and classify each sequenced well into the failure-mode taxonomy:

  OK                 — in build04 AND use_embryo_flag truthy (passed QC)
  EXCLUDED           — in build04 AND use_embryo_flag falsy; records which REAL QC flags fired
  ABSENT_IMAGED      — not in build04 but a stitched image exists (GDino FN / empty-well candidate)
  ABSENT_NO_IMAGE    — not in build04 and no stitched image (truncated-acq candidate)
  QC_NOT_RUN         — no build04 CSV for this experiment; QC/latent pipeline never run
                       (images may already be stitched — recoverable by rerunning build04/06)

The auto status is then refined by a human-curated disposition sidecar
(tables/well_dispositions.csv: exp,well,disposition,note) carried over from the curated
MISSING_SEQUENCED_AUDIT_curated_20260608.md. Auto-detection fills what it can; the sidecar
supplies the final reason (truncated_acq / gdino_fn / empty_well / clipped_lost / needs_review)
where only a human could know. Re-running never destroys those notes.

Why this rewrite: the previous version tested a NON-EXISTENT `usable_embryo` column, so every
in-build04 well defaulted to OK (0 EXCLUDED — wrong; e.g. plate02_t01 A02 is clipped/excluded),
and ABSENT was a single flat bucket that hid the real failure modes.

Outputs:
    MISSING_SEQUENCED_AUDIT.md            (regenerated narrative)
    tables/sequenced_coverage_audit.csv   (one row per sequenced well)
    tables/embryo_loss_map.csv            (embryo_id -> status/disposition, machine-readable)
    plots/audit/sequenced_coverage_heatmap.png   (plate x status counts)
    plots/audit/status_stacked_bar.png           (per-plate stacked status bars)
    plots/audit/wellgrids/wellgrid_<exp>.png     (per-plate 8x12 well maps)

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3a_audit_sequenced_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REPO = PROJECT_ROOT
PLAY = REPO / "morphseq_playground"
PLATE_META = REPO / "metadata/plate_metadata"
BUILD04 = PLAY / "metadata/build04_output"
STITCHED_FF = PLAY / "built_image_data/stitched_FF_images"

TABLE_DIR = RUN_DIR / "tables"
AUDIT_PLOT_DIR = RUN_DIR / "plots" / "audit"
WELLGRID_DIR = AUDIT_PLOT_DIR / "wellgrids"
AUDIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
WELLGRID_DIR.mkdir(parents=True, exist_ok=True)

DISPOSITIONS_CSV = TABLE_DIR / "well_dispositions.csv"
CURATED_MD = "MISSING_SEQUENCED_AUDIT_curated_20260608.md"

ROWS = "ABCDEFGH"
COLS = list(range(1, 13))
WELLS = [f"{r}{c:02}" for r in ROWS for c in COLS]

# The REAL QC-exclusion flag columns in build04. Informational columns
# (well_qc_flag, control_flag, use_embryo_flag) are deliberately NOT here.
REAL_QC_FLAGS = [
    "frame_flag", "sam2_qc_flag", "no_yolk_flag", "focus_flag",
    "bubble_flag", "dead_flag", "dead_flag2", "sa_outlier_flag",
]

STATUS_ORDER = ["OK", "EXCLUDED", "ABSENT_IMAGED", "ABSENT_NO_IMAGE", "QC_NOT_RUN"]
SEQ_STATUS_ORDER = STATUS_ORDER + ["SEQ_FAILED"]

# ---------------------------------------------------------------------------- data dictionary
# SINGLE SOURCE OF TRUTH for the meaning of every value a downstream reader sees in
# embryo_loss_map.csv. write_schema_sidecar() emits these next to the CSV so the meaning
# travels WITH the data — a reader (human or AI) never has to reverse-engineer it from
# build03/build04 to learn why a well is ABSENT vs EXCLUDED. Keep this in sync with the
# logic in audit(): the values defined here must be exactly the ones audit() can emit.
LOSS_MAP_GRAIN = (
    "One row per SEQUENCED well (one imaged embryo). Key = `embryo_id` "
    "(experiment-scoped, e.g. 20260320_cilia_crispant_48hpf_H06_e01). Join prediction "
    "tables to this on `embryo_id`. NOTE: predictions ALSO carry `physical_embryo_id` "
    "(biology-scoped, bridges a plate01 timeseries to its _t02 snapshot); a _t02 "
    "snapshot's model call is filed under its timeseries sibling, so joining a _t02 "
    "row here on `embryo_id` finds no prediction — that is the timeseries-priority rule, "
    "NOT a dropped embryo."
)
STATUS_MEANINGS = {
    "OK": "In build04 AND use_embryo_flag truthy — detected, tracked, passed QC.",
    "EXCLUDED": "In build04 but use_embryo_flag falsy — detected and tracked, then a REAL "
                "QC flag fired (see exclusion_flags col). Contrast ABSENT_IMAGED: the "
                "embryo EXISTS, it was QC-failed.",
    "ABSENT_IMAGED": "Stitched image exists but NO build04 row — the detector (GroundingDINO "
                     "-> SAM2) produced no embryo mask, so there is no build03/04 record to "
                     "QC. Common cause: out-of-focus or empty well. NOT a QC exclusion; the "
                     "embryo was never detected. See disposition for the curated reason.",
    "ABSENT_NO_IMAGE": "No build04 row AND no stitched image — acquisition likely truncated "
                       "before this well (e.g. Keyence stopped early).",
    "QC_NOT_RUN": "No build04 CSV for this experiment — the QC/latent pipeline never ran. "
                  "Images may already be stitched (recoverable by rerunning build04/06).",
}
DISPOSITION_MEANINGS = {
    "ok": "Auto-default for OK wells; no human action needed.",
    "needs_review": "Auto-default for a non-OK well the curated sidecar has not yet explained.",
    "qc_not_run": "Auto-default for QC_NOT_RUN wells.",
    "truncated_acq": "Acquisition stopped before this well was imaged (no real data).",
    "gdino_fn": "Image exists but GroundingDINO detected 0 embryos (no build03 row). "
                "Covers true false-negatives AND out-of-focus wells the detector could not "
                "resolve. Note records the human call (e.g. 'out of focus', 'clearly an embryo').",
    "empty_well": "Image exists; GDino found 0 embryos; human confirmed the well is genuinely empty.",
    "clipped_lost": "Detected but QC_EXCLUDED (frame/sam2 flags); embryo clipped at frame edge; "
                    "human confirmed unrecoverable.",
}

# Plotting colors per status (well-grid + bars). non-sequenced wells render gray.
STATUS_COLORS = {
    "OK": "#2ca02c",                 # green
    "EXCLUDED": "#ff7f0e",           # orange
    "ABSENT_IMAGED": "#d62728",      # red
    "ABSENT_NO_IMAGE": "#7f1d1d",    # dark red
    "QC_NOT_RUN": "#9467bd",         # purple
    "SEQ_FAILED": "#00FFFF",         # bright cyan — imaging OK but sequencing failed
    "NOT_SEQUENCED": "#e0e0e0",      # light gray (background)
}

# ---------------------------------------------------------------------------- sequencing-QC layer

# (imaging_plate, perturbation) -> list[experiment_id]
# For 48hpf b9d2 and cep290 the same physical embryo exists in both the snapshot experiment
# AND the sci timeseries experiment, so the value is a two-element list.
SEQ_QC_IMAGING_PLATE_MAP: dict[tuple[str, str], list[str]] = {
    # crispant (plate index = collection order; dates confirmed)
    ("260319_plate_1", "crispants"):         ["20260319_cilia_crispant_18hpf"],
    ("260319_plate_2", "crispants"):         ["20260319_cilia_crispant_24hpf"],
    ("260319_plate_3", "crispants"):         ["20260319_cilia_crispant_30hpf"],
    ("260320_plate_4", "crispants"):         ["20260320_cilia_crispant_48hpf"],
    # cep290
    ("260414_plate_3", "18hpf_cep290_mut"):  ["20260415_cep290_18hpf_plate03"],   # date off-by-one
    ("260414_plate_3", "18hpf_cep290_wt"):   ["20260415_cep290_18hpf_plate03"],
    ("260401_plate_2", "24hpf_cep290_wt"):   ["20260324_cep290_18hpf_24hpf_plate02"],  # WARNING: 260401 has no manifest experiment
    ("260324_plate_1", "30hpf_cep290_mut"):  ["20260324_cep290_30hpf_plate01"],
    # 48hpf cep290: snapshot + sci timeseries (same physical embryo in both)
    ("260416_plate_1", "48hpf_cep290_mut"):  ["20260416_cep290_30to48hpf_plate01_t02", "20260415_sci_cep290_48hpf_plate01"],
    ("260416_plate_1", "48hpf_cep290_wt"):   ["20260416_cep290_30to48hpf_plate01_t02", "20260415_sci_cep290_48hpf_plate01"],
    ("260416_plate_1", "48hpf_cep290_AB"):   ["20260416_cep290_30to48hpf_plate01_t02", "20260415_sci_cep290_48hpf_plate01"],
    # b9d2 — 260414_plate_1 covers BOTH 14hpf and 48hpf; disambiguated by perturbation hpf prefix
    ("260414_plate_1", "14hpf_b9d2_mut"):    ["20260414_b9d2_14hpf_plate01"],
    ("260414_plate_1", "14hpf_b9d2_wt"):     ["20260414_b9d2_14hpf_plate01"],
    ("260414_plate_1", "14hpf_b9d2_AB"):     ["20260414_b9d2_14hpf_plate01"],
    ("260415_plate_1", "30hpf_b9d2_mut"):    ["20260414_b9d2_30hpf_plate01"],
    ("260415_plate_1", "30hpf_b9d2_wt"):     ["20260414_b9d2_30hpf_plate01"],
    # 48hpf b9d2: snapshot + sci timeseries (same physical embryo in both)
    ("260414_plate_1", "48hpf_b9d2_mut"):    ["20260415_b9d2_30to48hpf_plate01_t02", "20260414_sci_b9d2_48hpf_plate01"],
    ("260414_plate_1", "48hpf_b9d2_wt"):     ["20260415_b9d2_30to48hpf_plate01_t02", "20260414_sci_b9d2_48hpf_plate01"],
    ("260414_plate_1", "48hpf_b9d2_AB"):     ["20260415_b9d2_30to48hpf_plate01_t02", "20260414_sci_b9d2_48hpf_plate01"],
    ("260414_plate_2", "48hpf_b9d2_mut"):    ["20260415_b9d2_30to48hpf_plate02_t02"],  # date off-by-one; no sci for plate02
}

SEQ_QC_DATE_WARNINGS: dict[tuple[str, str], str] = {
    ("260401_plate_2", "24hpf_cep290_wt"):   "date 260401 has no manifest experiment; mapped to 20260324_cep290_18hpf_24hpf_plate02 via well D06",
    ("260414_plate_3", "18hpf_cep290_mut"):  "date 260414 vs experiment date 260415 for cep290_18hpf_plate03",
    ("260414_plate_3", "18hpf_cep290_wt"):   "date 260414 vs experiment date 260415 for cep290_18hpf_plate03",
    ("260414_plate_2", "48hpf_b9d2_mut"):    "date 260414 vs experiment date 260415 for b9d2_30to48hpf_plate02_t02",
}

SEQ_QC_FAILURES_CSV = TABLE_DIR / "failed_sequencing_qc" / "missing_embryos_due_to_sequencing.csv"
SEQ_QC_RESOLVED_CSV = TABLE_DIR / "failed_sequencing_qc" / "seq_qc_failures_resolved.csv"


def _norm_well(w: str) -> str:
    """Normalize imaging_well bare format (E3) to zero-padded build06 format (E03)."""
    import re
    return re.sub(r"([A-Ha-h])(\d)$", lambda m: m.group(1).upper() + m.group(2).zfill(2), w.strip())


def load_seq_qc_failures(query_all_rows: pd.DataFrame) -> pd.DataFrame:
    """Load missing_embryos_due_to_sequencing.csv, resolve to embryo_id via SEQ_QC_IMAGING_PLATE_MAP.

    Returns one row per (source CSV row × experiment_id) that could be matched to the query
    table. Unmatched rows are printed as warnings and omitted from the result.
    """
    if not SEQ_QC_FAILURES_CSV.exists():
        print(f"  WARNING: seq-QC failures CSV not found at {SEQ_QC_FAILURES_CSV.relative_to(RUN_DIR)}")
        return pd.DataFrame(columns=["source_embryo_ID", "imaging_plate", "imaging_well",
                                     "perturbation", "experiment", "well", "embryo_id",
                                     "physical_embryo_id", "gene", "seq_qc_status"])

    src = pd.read_csv(SEQ_QC_FAILURES_CSV, dtype=str).fillna("")
    # lookup dict: (experiment, well) -> first matching query row
    q_dedup = query_all_rows.drop_duplicates("embryo_id")[
        ["embryo_id", "experiment", "well", "physical_embryo_id", "gene"]
    ]
    q_lookup: dict[tuple[str, str], dict] = {
        (row["experiment"], row["well"]): row.to_dict()
        for _, row in q_dedup.iterrows()
    }

    warned_keys: set[tuple[str, str]] = set()
    rows = []
    for _, r in src.iterrows():
        plate = r["imaging_plate"].strip()
        pert = r["perturbation"].strip()
        raw_well = r["imaging_well"].strip()
        well = _norm_well(raw_well)
        key = (plate, pert)

        if key in SEQ_QC_DATE_WARNINGS and key not in warned_keys:
            print(f"  WARNING seq-QC date mismatch: {plate} / {pert} — {SEQ_QC_DATE_WARNINGS[key]}")
            warned_keys.add(key)

        exp_ids = SEQ_QC_IMAGING_PLATE_MAP.get(key)
        if exp_ids is None:
            print(f"  WARNING seq-QC unmapped: ({plate}, {pert}) — skipping row {r['embryo_ID']}")
            continue

        for exp_id in exp_ids:
            q = q_lookup.get((exp_id, well))
            if q is not None:
                rows.append({
                    "source_embryo_ID": r["embryo_ID"],
                    "imaging_plate": plate,
                    "imaging_well": raw_well,
                    "perturbation": pert,
                    "experiment": exp_id,
                    "well": well,
                    "embryo_id": q["embryo_id"],
                    "physical_embryo_id": q["physical_embryo_id"],
                    "gene": q["gene"],
                    "seq_qc_status": "seq_failed",
                })
            else:
                print(f"  WARNING seq-QC: ({exp_id}, {well}) not found in query table "
                      f"[source: {r['embryo_ID']}]")

    result = pd.DataFrame(rows)
    if not result.empty:
        SEQ_QC_RESOLVED_CSV.parent.mkdir(exist_ok=True)
        result.to_csv(SEQ_QC_RESOLVED_CSV, index=False)
        print(f"  wrote {SEQ_QC_RESOLVED_CSV.relative_to(RUN_DIR)} ({len(result)} rows)")
    return result


def build_embryo_registry(
    audit_df: pd.DataFrame,
    seq_failures: pd.DataFrame,
    query_all_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build tables/embryo_registry.csv: one row per physical embryo with imaging + seq QC status.

    Registry grain: physical_embryo_id (biology-scoped). Sequenced physical embryos are the
    universe; unsequenced embryos are excluded.
    Join chain:
      query_all_rows -> select_for_label_transfer (representative embryo_id per physical) ->
      embryo_loss_map (imaging QC, keyed on snapshot embryo_id) ->
      seq_failures (seq QC).
    """
    from cilia_qc_helpers import select_for_label_transfer  # noqa: F401

    # One row per embryo_id (deduplicated frames)
    emb = query_all_rows.drop_duplicates("embryo_id").copy()

    # Representative embryo per physical (timeseries wins over snapshot — same logic as script 2)
    rep = select_for_label_transfer(emb).drop_duplicates("physical_embryo_id")

    # Snapshot embryo_id per physical (for portfolio join and imaging-QC lookup)
    snap = (
        emb[emb["data_source"] == "snapshot"]
        .drop_duplicates("physical_embryo_id")
        [["physical_embryo_id", "embryo_id"]]
        .rename(columns={"embryo_id": "snapshot_embryo_id"})
    )

    registry = rep[[
        "physical_embryo_id", "gene", "collection_time_hpf", "data_source",
        "embryo_id", "well", "experiment", "sequenced", "sequenced_stratum",
        "genotype_clean",
    ]].rename(columns={"embryo_id": "representative_embryo_id", "data_source": "representative_source"})
    registry = registry.merge(snap, on="physical_embryo_id", how="left")

    # Imaging QC: join on snapshot_embryo_id if available, else representative (non-sci plates)
    loss = audit_df[["embryo_id", "status", "disposition"]].rename(
        columns={"embryo_id": "img_embryo_id", "status": "imaging_qc_status",
                 "disposition": "imaging_qc_disposition"}
    )
    registry["_join_id"] = registry["snapshot_embryo_id"].where(
        registry["snapshot_embryo_id"].notna(), registry["representative_embryo_id"]
    )
    registry = registry.merge(
        loss.rename(columns={"img_embryo_id": "_join_id"}), on="_join_id", how="left"
    ).drop(columns=["_join_id"])
    registry["imaging_qc_status"] = registry["imaging_qc_status"].fillna("NOT_IN_AUDIT")

    # Seq QC: flag any physical embryo that appears in failures
    failed_physicals = set(seq_failures["physical_embryo_id"].dropna()) if not seq_failures.empty else set()
    def _seq_status(row: pd.Series) -> str:
        if row["physical_embryo_id"] in failed_physicals:
            return "seq_failed"
        if row["sequenced"] > 0:
            return "seq_ok"
        return "not_submitted"
    registry["seq_qc_status"] = registry.apply(_seq_status, axis=1)
    registry["final_usable"] = (
        (registry["imaging_qc_status"] == "OK")
        & (registry["seq_qc_status"] == "seq_ok")
        & (registry["sequenced"] > 0)
    )

    # Only sequenced embryos in the registry
    registry = registry[registry["sequenced"] > 0].copy()
    registry.to_csv(TABLE_DIR / "embryo_registry.csv", index=False)
    print(f"  wrote {(TABLE_DIR / 'embryo_registry.csv').relative_to(RUN_DIR)} ({len(registry)} rows)")
    return registry


def cohort_experiments() -> list[str]:
    """Non-sci_ b9d2/cep290/crispant plates from this cohort's manifest (these carry a `sequenced` sheet)."""
    m = pd.read_csv(TABLE_DIR / "experiment_manifest.csv")
    sel = m[(~m["is_sci_timelapse"]) & (m["gene"].isin(["b9d2", "cep290", "crispant"]))]
    return sorted(sel["experiment"].astype(str).unique())


def sequenced_grid(exp: str) -> dict[str, int] | None:
    """Parse the 8x12 `sequenced` sheet -> {well: code}. Returns None if no Excel/sheet found."""
    for cand in (f"{exp}_well_metadata.xlsx", f"{exp}.xlsx"):
        p = PLATE_META / cand
        if not p.exists():
            continue
        with pd.ExcelFile(p) as xlf:
            if "sequenced" not in xlf.sheet_names:
                return None
            df = xlf.parse("sequenced", header=0)
            block = df.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
            arr = block.to_numpy(dtype=str).ravel()
        out: dict[str, int] = {}
        for w, v in zip(WELLS, arr):
            s = v.strip()
            try:
                out[w] = int(float(s)) if s not in ("", "nan") else 0
            except ValueError:
                out[w] = 0
        return out
    return None


def stitched_image_exists(exp: str, well: str) -> bool:
    """Was a stitched FF image produced for this well? (image-exists signal for ABSENT split)."""
    d = STITCHED_FF / exp
    if not d.is_dir():
        return False
    return any(d.glob(f"{well}_*stitch*"))


def load_dispositions() -> pd.DataFrame:
    """Human-curated (exp,well) -> disposition/note. Empty frame if the sidecar is missing."""
    if not DISPOSITIONS_CSV.exists():
        print(f"  WARNING: no dispositions sidecar at {DISPOSITIONS_CSV.relative_to(RUN_DIR)}")
        return pd.DataFrame(columns=["exp", "well", "disposition", "note"])
    d = pd.read_csv(DISPOSITIONS_CSV, dtype=str).fillna("")
    return d[["exp", "well", "disposition", "note"]]


def audit(exps: list[str]) -> pd.DataFrame:
    rows = []
    for exp in exps:
        grid = sequenced_grid(exp)
        if grid is None:
            print(f"  WARNING: no plate Excel / `sequenced` sheet found for {exp}")
            continue

        seq_wells = {w for w, v in grid.items() if v in (1, 2)}
        if not seq_wells:
            continue

        b04_path = BUILD04 / f"qc_staged_{exp}.csv"
        b04 = pd.read_csv(b04_path) if b04_path.exists() else None
        b04_wells = (
            set(b04["well"].astype(str).str.strip()) if b04 is not None else set()
        )

        for w in sorted(seq_wells):
            embryo_id, flags = "", ""
            if b04 is None:
                status = "QC_NOT_RUN"
            elif w not in b04_wells:
                status = "ABSENT_IMAGED" if stitched_image_exists(exp, w) else "ABSENT_NO_IMAGE"
            else:
                row = b04[b04["well"].astype(str).str.strip() == w].iloc[0]
                embryo_id = str(row.get("embryo_id", "") or "")
                use_ok = bool(row.get("use_embryo_flag", False))
                fired = [f for f in REAL_QC_FLAGS if bool(row.get(f, False))]
                status = "OK" if use_ok else "EXCLUDED"
                flags = "|".join(fired)

            if not embryo_id:
                embryo_id = f"{exp}_{w}"  # stable key for wells with no build04 embryo

            rows.append({
                "embryo_id": embryo_id, "exp": exp, "well": w,
                "seq_code": grid[w], "status": status, "exclusion_flags": flags,
            })

    df = pd.DataFrame(rows)

    # Join human dispositions. Fill auto-defaults where the sidecar is silent.
    disp = load_dispositions()
    df = df.merge(disp, on=["exp", "well"], how="left")
    df["disposition"] = df["disposition"].fillna("")
    df["note"] = df["note"].fillna("")
    df["disposition_conflict"] = ""

    # Self-heal stale dispositions: a sidecar "absence" reason that now contradicts a live
    # present-in-build04 status (OK/EXCLUDED) means the data was reprocessed since the curated
    # snapshot. Trust the data — drop the stale disposition and record the conflict so the
    # sidecar can be pruned. (e.g. cep290_18hpf G/H wells were "truncated_acq" on 2026-06-08
    # but build04 has since gained those rows and they are OK.)
    ABSENCE_DISPS = {"truncated_acq", "gdino_fn", "empty_well"}
    PRESENT = {"OK", "EXCLUDED"}
    stale = df["status"].isin(PRESENT) & df["disposition"].isin(ABSENCE_DISPS)
    df.loc[stale, "disposition_conflict"] = (
        "stale:" + df.loc[stale, "disposition"] + " (well now " + df.loc[stale, "status"] + ")"
    )
    df.loc[stale, ["disposition", "note"]] = ""

    # Auto-default dispositions for rows the sidecar didn't cover (or were just cleared).
    auto_disp = {
        "OK": "ok",
        "EXCLUDED": "needs_review",
        "ABSENT_IMAGED": "needs_review",
        "ABSENT_NO_IMAGE": "needs_review",
        "QC_NOT_RUN": "qc_not_run",
    }
    blank = df["disposition"] == ""
    df.loc[blank, "disposition"] = df.loc[blank, "status"].map(auto_disp)

    # For QC_NOT_RUN wells, note whether the stitched image already exists (recoverable)
    # so the label isn't misread as "never imaged".
    qnr = (df["status"] == "QC_NOT_RUN") & (df["note"] == "")
    df.loc[qnr, "note"] = df[qnr].apply(
        lambda r: ("image stitched — rerun build04/06 to recover"
                   if stitched_image_exists(r["exp"], r["well"])
                   else "no stitched image — QC pipeline not run"), axis=1)

    # Attach gene from the manifest (for the gene-split loss-reason bar plot).
    man = pd.read_csv(TABLE_DIR / "experiment_manifest.csv")[["experiment", "gene"]]
    df = df.merge(man.rename(columns={"experiment": "exp"}), on="exp", how="left")
    df["gene"] = df["gene"].fillna("unknown")
    return df


def coverage_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.groupby(["exp", "status"]).size().unstack(fill_value=0)
    for col in STATUS_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[STATUS_ORDER]
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_index()


# ---------------------------------------------------------------------------- writers

def write_markdown(df: pd.DataFrame, pivot: pd.DataFrame, path: Path) -> None:
    counts = df["status"].value_counts()
    lines = [
        "# Sequenced-vs-pipeline coverage audit", "",
        "Excel `sequenced` sheet vs build04 QC, for non-sci_ b9d2/cep290 plates. "
        f"Generated by `3a_audit_sequenced_coverage.py`. Human dispositions sourced from "
        f"[`{CURATED_MD}`](./{CURATED_MD}) via `tables/well_dispositions.csv`.", "",
        "Status vocabulary: `OK` (passed QC) · `EXCLUDED` (in build04, use_embryo_flag=0) · "
        "`ABSENT_IMAGED` (stitched image exists, no build04 row) · `ABSENT_NO_IMAGE` (no stitched "
        "image) · `QC_NOT_RUN` (no build04 CSV — QC pipeline never run; images may be stitched).", "",
        "## Totals", "",
    ]
    lines += [f"- **{s}**: {int(counts.get(s, 0))}" for s in STATUS_ORDER]
    lines += [f"- **TOTAL sequenced wells**: {len(df)}", "", "## By experiment", "",
              "| experiment | " + " | ".join(STATUS_ORDER) + " | total |",
              "|" + "---|" * (len(STATUS_ORDER) + 2)]
    for exp, r in pivot.iterrows():
        lines.append("| " + exp + " | "
                     + " | ".join(str(int(r[s])) for s in STATUS_ORDER)
                     + f" | {int(r['total'])} |")

    detail_specs = [
        ("EXCLUDED", "EXCLUDED", ["well", "seq_code", "exclusion_flags", "disposition", "note"]),
        ("ABSENT (imaged + no-image)", ("ABSENT_IMAGED", "ABSENT_NO_IMAGE"),
         ["well", "seq_code", "status", "disposition", "note"]),
        ("QC_NOT_RUN", "QC_NOT_RUN", ["well", "seq_code", "disposition", "note"]),
    ]
    for label, status, cols in detail_specs:
        sel = df["status"].isin(status if isinstance(status, tuple) else (status,))
        sub = df[sel].sort_values(["exp", "well"])
        lines += ["", f"## {label} detail", ""]
        if sub.empty:
            lines.append("_(none)_")
            continue
        head = ["exp"] + cols
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for _, rr in sub.iterrows():
            lines.append("| " + " | ".join(str(rr[c]) for c in head) + " |")

    conflicts = df[df["disposition_conflict"] != ""].sort_values(["exp", "well"])
    lines += ["", "## Stale dispositions (auto-status overrides curated note)", "",
              "Wells where the curated sidecar marked an absence reason but build04 has since "
              "been reprocessed and the well is now present. Data wins; prune these rows from "
              "`tables/well_dispositions.csv`.", ""]
    if conflicts.empty:
        lines.append("_(none)_")
    else:
        lines.append("| exp | well | status (now) | stale note |")
        lines.append("|---|---|---|---|")
        for _, rr in conflicts.iterrows():
            lines.append(f"| {rr['exp']} | {rr['well']} | {rr['status']} | "
                         f"{rr['disposition_conflict']} |")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(RUN_DIR)}")


def write_schema_sidecar(df: pd.DataFrame, path: Path) -> None:
    """Emit a data-dictionary next to embryo_loss_map.csv so the meaning of every column
    value travels WITH the data. Only documents values actually present in this run, plus
    the full vocabulary, so a reader never has to reverse-engineer status from build03/04."""
    present_status = [s for s in STATUS_ORDER if s in set(df["status"])]
    present_disp = sorted(set(df["disposition"]) - {""})
    lines = [
        "# embryo_loss_map.csv — data dictionary", "",
        "Auto-generated by `3a_audit_sequenced_coverage.py` (do not hand-edit; edit the "
        "STATUS_MEANINGS / DISPOSITION_MEANINGS dicts in that script). Curated reasons come "
        "from `tables/well_dispositions.csv`.", "",
        "## Grain & join key", "", LOSS_MAP_GRAIN, "",
        "## Columns", "",
        "| column | meaning |", "|---|---|",
        "| embryo_id | experiment-scoped embryo key; the join key for prediction tables |",
        "| exp / well | experiment id and 8x12 well |",
        "| seq_code | Excel `sequenced` sheet code: 1 or 2 = sequenced (2 = het+homo stratum) |",
        "| status | pipeline outcome — see Status values below |",
        "| exclusion_flags | for EXCLUDED rows, the REAL QC flags that fired (pipe-joined) |",
        "| disposition | curated/auto reason — see Disposition values below |",
        "| note | free-text human note (e.g. 'out of focus') |",
        "| disposition_conflict | set when a curated absence reason is now contradicted by live data |",
        "",
        "## Status values  (`ABSENT_IMAGED` ≠ `EXCLUDED` — the key distinction)", "",
    ]
    for s in STATUS_ORDER:
        mark = "" if s in present_status else "  _(not present this run)_"
        lines.append(f"- **{s}**{mark}: {STATUS_MEANINGS[s]}")
    lines += ["", "## Disposition values", ""]
    for d, meaning in DISPOSITION_MEANINGS.items():
        mark = "" if d in present_disp else "  _(not present this run)_"
        lines.append(f"- **{d}**{mark}: {meaning}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(RUN_DIR)}")


# ---------------------------------------------------------------------------- plots

def plot_heatmap(pivot: pd.DataFrame, path: Path) -> None:
    exps = list(pivot.index)
    M = pivot[STATUS_ORDER].to_numpy()
    fig, ax = plt.subplots(figsize=(1.8 + 1.0 * len(STATUS_ORDER), 1.2 + 0.34 * len(exps)))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(STATUS_ORDER)))
    ax.set_xticklabels(STATUS_ORDER, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps, fontsize=9)
    vmax = M.max() if M.max() > 0 else 1
    for i in range(len(exps)):
        for j in range(len(STATUS_ORDER)):
            v = int(M[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if v > vmax * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="sequenced wells")
    ax.set_title("Sequenced-well coverage by plate x status", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


def plot_stacked_bar(pivot: pd.DataFrame, path: Path,
                     seq_failed_by_exp: dict[str, int] | None = None) -> None:
    exps = list(pivot.index)
    y = np.arange(len(exps))
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.36 * len(exps)))
    left = np.zeros(len(exps))
    order = STATUS_ORDER + (["SEQ_FAILED"] if seq_failed_by_exp else [])
    for s in order:
        if s == "SEQ_FAILED":
            vals = np.array([seq_failed_by_exp.get(e, 0) for e in exps], dtype=float)
        else:
            vals = pivot[s].to_numpy() if s in pivot.columns else np.zeros(len(exps))
        ax.barh(y, vals, left=left, color=STATUS_COLORS[s], label=s,
                edgecolor="white", linewidth=1.4, height=0.7)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center", fontsize=9,
                        fontweight="bold", color="black" if s == "SEQ_FAILED" else "white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(exps, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("sequenced wells", fontsize=11)
    ax.set_title("Per-plate sequenced-well status", fontsize=12)
    ax.legend(fontsize=9, ncol=len(order), loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


def plot_loss_reasons_by_gene(df: pd.DataFrame, path: Path) -> None:
    """Grouped bars: count of lost (non-OK) sequenced wells per loss-reason, split by gene."""
    lost = df[df["status"] != "OK"]
    if lost.empty:
        print("  (no lost wells — skipping loss-reasons-by-gene plot)")
        return
    ct = lost.groupby(["disposition", "gene"]).size().unstack(fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]  # busiest reason first
    reasons = list(ct.index)
    genes = list(ct.columns)
    x = np.arange(len(reasons))
    w = 0.8 / max(len(genes), 1)
    gene_colors = {"b9d2": "#1f77b4", "cep290": "#ff7f0e", "crispant": "#2ca02c",
                   "unknown": "#888888"}
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(reasons), 4.2))
    for gi, g in enumerate(genes):
        vals = ct[g].to_numpy()
        bars = ax.bar(x + gi * w - 0.4 + w / 2, vals, w, label=g,
                      color=gene_colors.get(g, "#888888"),
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, str(int(v)),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("lost sequenced wells", fontsize=11)
    ax.set_title("Loss reasons by gene", fontsize=12)
    ax.legend(title="gene", fontsize=10, title_fontsize=10, frameon=False)
    ax.margins(y=0.12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


def plot_wellgrids(df: pd.DataFrame, out_dir: Path,
                   seq_failed_wells: dict[str, set[str]] | None = None) -> None:
    """One 8x12 plate map per experiment; sequenced wells colored by status, others gray.

    seq_failed_wells: {exp: {well, ...}} — wells that passed imaging QC but failed sequencing.
    These render cyan (SEQ_FAILED), overriding the OK green.
    """
    if seq_failed_wells is None:
        seq_failed_wells = {}
    for exp, sub in df.groupby("exp"):
        status_by_well = dict(zip(sub["well"], sub["status"]))
        seq_by_well = dict(zip(sub["well"], sub["seq_code"]))
        seq_fail_set = seq_failed_wells.get(str(exp), set())
        fig, (ax, ax_side) = plt.subplots(
            1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [2.0, 1.0]})
        for ri, r in enumerate(ROWS):
            for ci, c in enumerate(COLS):
                w = f"{r}{c:02}"
                st = status_by_well.get(w, "NOT_SEQUENCED")
                # Cyan overrides OK for seq-failed wells
                if w in seq_fail_set and st == "OK":
                    st = "SEQ_FAILED"
                ax.add_patch(mpatches.Rectangle(
                    (ci, ri), 0.92, 0.92, facecolor=STATUS_COLORS[st],
                    edgecolor="white", linewidth=0.8))
                if w in status_by_well:
                    text_color = "black" if st in ("NOT_SEQUENCED", "SEQ_FAILED") else "white"
                    ax.text(ci + 0.46, ri + 0.46, str(int(seq_by_well[w])),
                            ha="center", va="center", fontsize=8, fontweight="bold",
                            color=text_color)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([c - 0.54 for c in range(1, 13)])
        ax.set_xticklabels(COLS, fontsize=10)
        ax.set_yticks([r + 0.46 for r in range(8)])
        ax.set_yticklabels(list(ROWS), fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(exp, fontsize=11)
        legend_statuses = STATUS_ORDER + ["SEQ_FAILED"]
        handles = [mpatches.Patch(color=STATUS_COLORS[s], label=s) for s in legend_statuses]
        ax.legend(handles=handles, fontsize=8, ncol=len(legend_statuses), loc="upper center",
                  bbox_to_anchor=(0.5, -0.06), frameon=False)

        # Side panel: non-OK wells + seq-failed wells
        lost = sub[sub["status"] != "OK"].sort_values("well")
        ax_side.axis("off")
        ax_side.set_title("Lost / seq-failed wells", fontsize=10, loc="left")
        side_entries = []
        for _, rr in lost.iterrows():
            reason = rr["disposition"] if rr["disposition"] not in ("", "needs_review") \
                else (rr["exclusion_flags"] or rr["status"].lower())
            side_entries.append((rr["well"], rr["status"], reason))
        for w in sorted(seq_fail_set):
            # Only add if not already in lost list
            if w not in {e[0] for e in side_entries}:
                side_entries.append((w, "SEQ_FAILED", "seq_failed"))
        if not side_entries:
            ax_side.text(0.0, 0.97, "(all sequenced wells OK)", fontsize=9,
                         va="top", ha="left", color="gray")
        else:
            y = 0.97
            for well_label, st_label, reason in sorted(side_entries):
                ax_side.text(0.0, y, well_label, fontsize=9, va="top", ha="left",
                             color=STATUS_COLORS.get(st_label, "#333333"), fontweight="bold")
                ax_side.text(0.16, y, f"{st_label} — {reason}", fontsize=8,
                             va="top", ha="left", color="black")
                y -= 0.05
        fig.savefig(out_dir / f"wellgrid_{exp}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(df['exp'].unique())} well-grids to plots/audit/wellgrids/")


def plot_failure_venn(registry: pd.DataFrame, path: Path) -> None:
    """Two-set Venn per gene: imaging-failed vs seq-failed, with final_usable in center."""
    genes = [g for g in ["b9d2", "cep290", "crispant"] if g in registry["gene"].values]
    fig, axes = plt.subplots(1, len(genes), figsize=(4.5 * len(genes), 4.5))
    if len(genes) == 1:
        axes = [axes]

    try:
        from matplotlib_venn import venn2
        use_lib = True
    except ImportError:
        use_lib = False
        print("  NOTE: matplotlib_venn not installed; drawing manual Venn circles")

    for ax, gene in zip(axes, genes):
        sub = registry[registry["gene"] == gene]
        img_failed = set(sub.loc[sub["imaging_qc_status"] != "OK", "embryo_id"].dropna())
        seq_failed = set(sub.loc[sub["seq_qc_status"] == "seq_failed", "embryo_id"].dropna())
        both = img_failed & seq_failed
        img_only = img_failed - both
        seq_only = seq_failed - both
        final_ok = int((sub["final_usable"] == True).sum())  # noqa: E712
        total = len(sub)

        if use_lib:
            v = venn2(subsets=(len(img_only), len(seq_only), len(both)), ax=ax,
                      set_labels=("Imaging\nfailed", "Seq\nfailed"),
                      set_colors=("#ff7f0e", "#00FFFF"), alpha=0.55)
            if v.get_label_by_id("100"):
                v.get_label_by_id("100").set_text(str(len(img_only)))
            if v.get_label_by_id("010"):
                v.get_label_by_id("010").set_text(str(len(seq_only)))
            if v.get_label_by_id("110"):
                v.get_label_by_id("110").set_text(str(len(both)))
        else:
            import matplotlib.patches as mpatch
            ax.add_patch(mpatch.Circle((0.35, 0.5), 0.28, color="#ff7f0e", alpha=0.45, transform=ax.transAxes))
            ax.add_patch(mpatch.Circle((0.65, 0.5), 0.28, color="#00FFFF", alpha=0.45, transform=ax.transAxes))
            ax.text(0.20, 0.50, str(len(img_only)), ha="center", va="center",
                    fontsize=14, fontweight="bold", transform=ax.transAxes)
            ax.text(0.50, 0.50, str(len(both)), ha="center", va="center",
                    fontsize=14, fontweight="bold", transform=ax.transAxes)
            ax.text(0.80, 0.50, str(len(seq_only)), ha="center", va="center",
                    fontsize=14, fontweight="bold", transform=ax.transAxes)
            ax.text(0.15, 0.15, "Imaging\nfailed", ha="center", va="top",
                    fontsize=9, transform=ax.transAxes, color="#ff7f0e")
            ax.text(0.85, 0.15, "Seq\nfailed", ha="center", va="top",
                    fontsize=9, transform=ax.transAxes, color="#008888")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

        ax.set_title(f"{gene}\nfinal usable: {final_ok} / {total}", fontsize=11)

    fig.suptitle("Failure mode overlap: imaging QC vs sequencing QC", fontsize=12, y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


def plot_final_usable_by_gene(registry: pd.DataFrame, path: Path) -> None:
    """Horizontal stacked bar per gene: imaging_only / seq_only / both_failed / final_usable."""
    genes = [g for g in ["b9d2", "cep290", "crispant"] if g in registry["gene"].values]
    rows_data = []
    for gene in genes:
        sub = registry[registry["gene"] == gene]
        img_f = sub["imaging_qc_status"] != "OK"
        seq_f = sub["seq_qc_status"] == "seq_failed"
        rows_data.append({
            "gene": gene,
            "imaging_only": int((img_f & ~seq_f).sum()),
            "seq_only": int((~img_f & seq_f).sum()),
            "both_failed": int((img_f & seq_f).sum()),
            "final_usable": int(sub["final_usable"].sum()),
        })
    ct = pd.DataFrame(rows_data).set_index("gene")
    stacks = [
        ("imaging_only", "#ff7f0e", "Imaging failed only"),
        ("seq_only", "#FFAA00", "Seq failed only"),
        ("both_failed", "#d62728", "Both failed"),
        ("final_usable", "#2166AC", "Final usable"),
    ]
    y = np.arange(len(genes))
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.55 * len(genes)))
    left = np.zeros(len(genes))
    for col, color, label in stacks:
        vals = ct[col].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=color, label=label,
                edgecolor="white", linewidth=1.4, height=0.6)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("sequenced embryos", fontsize=11)
    ax.set_title("Final usable counts by gene", fontsize=12)
    ax.legend(fontsize=9, ncol=len(stacks), loc="upper center",
              bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


# ---------------------------------------------------------------------------- main

print("3a - sequenced-vs-pipeline coverage audit (failure-mode taxonomy)")
exps = cohort_experiments()
print(f"Auditing {len(exps)} non-sci b9d2/cep290/crispant plates")
df = audit(exps)
if df.empty:
    print("No sequenced wells found across the cohort — nothing to audit.")
    sys.exit(0)

print("\n=== SEQUENCED WELL COVERAGE ===")
print(df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_string())
print(f"Total sequenced wells in Excel: {len(df)}")

pivot = coverage_pivot(df)
print("\n=== BY EXPERIMENT ===")
print(pivot.to_string())

conflicts = df[df["disposition_conflict"] != ""]
if not conflicts.empty:
    print(f"\n=== STALE DISPOSITIONS ({len(conflicts)}) — prune these from well_dispositions.csv ===")
    print(conflicts[["exp", "well", "status", "disposition_conflict"]].to_string(index=False))

# tables
audit_csv = TABLE_DIR / "sequenced_coverage_audit.csv"
df.to_csv(audit_csv, index=False)
print(f"\n  wrote {audit_csv.relative_to(RUN_DIR)}")

loss_map = df[["embryo_id", "exp", "well", "seq_code", "status",
               "exclusion_flags", "disposition", "note", "disposition_conflict"]]
loss_csv = TABLE_DIR / "embryo_loss_map.csv"
loss_map.to_csv(loss_csv, index=False)
print(f"  wrote {loss_csv.relative_to(RUN_DIR)}")
write_schema_sidecar(df, TABLE_DIR / "embryo_loss_map.schema.md")

write_markdown(df, pivot, RUN_DIR / "MISSING_SEQUENCED_AUDIT.md")

# ---------------------------------------------------------------------------- read qc_registry from script 0

qc_registry_path = TABLE_DIR / "qc_registry.csv"
if qc_registry_path.exists():
    print("\n=== QC REGISTRY (from script 0) ===")
    registry = pd.read_csv(qc_registry_path, low_memory=False)

    # Build seq_failed_by_exp for well-grid + stacked-bar overlays.
    # Only snapshot experiments appear in the imaging audit; sci timeseries don't have well grids.
    audit_exps = set(df["exp"])
    seq_failed_by_exp: dict[str, set[str]] = {}
    for _, r in registry[registry["seq_qc_status"] == "seq_failed"].iterrows():
        if r["experiment"] in audit_exps:
            seq_failed_by_exp.setdefault(r["experiment"], set()).add(r["well"])
    seq_failed_count_by_exp = {exp: len(wells) for exp, wells in seq_failed_by_exp.items()}

    n_seq_fail = (registry["seq_qc_status"] == "seq_failed").sum()
    print(f"  {n_seq_fail} seq-failed wells across {len(seq_failed_by_exp)} experiments")
    print(f"  final_usable: {registry['final_usable'].sum()} / {len(registry)} total sequenced wells")

    # final-count summary (mirrors what was produced before)
    count_cols = ["gene", "sequenced_stratum", "collection_time_hpf"]
    summary_rows = []
    for keys, sub in registry.groupby(count_cols):
        img_f = (sub["imaging_qc_status"] != "OK").sum()
        seq_f = (sub["seq_qc_status"] == "seq_failed").sum()
        both_f = ((sub["imaging_qc_status"] != "OK") & (sub["seq_qc_status"] == "seq_failed")).sum()
        summary_rows.append({
            "gene": keys[0], "sequenced_stratum": keys[1],
            "collection_time_hpf": keys[2],
            "n_total_sequenced": len(sub),
            "n_imaging_failed": int(img_f),
            "n_seq_failed": int(seq_f),
            "n_both_failed": int(both_f),
            "n_final_usable": int(sub["final_usable"].sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "final_usable_counts.csv", index=False)
    print(f"\n=== FINAL USABLE COUNTS ===")
    print(summary.to_string(index=False))
    print(f"\n  wrote {(TABLE_DIR / 'final_usable_counts.csv').relative_to(RUN_DIR)}")
else:
    print("\n  NOTE: qc_registry.csv not found — run script 0 first to compute QC gates.")
    print("  Seq-failed overlays and Venn/usable plots will be skipped.")
    seq_failed_by_exp = {}
    seq_failed_count_by_exp = {}
    registry = pd.DataFrame()

# ---------------------------------------------------------------------------- plots
plot_heatmap(pivot, AUDIT_PLOT_DIR / "sequenced_coverage_heatmap.png")
plot_stacked_bar(pivot, AUDIT_PLOT_DIR / "status_stacked_bar.png",
                 seq_failed_by_exp=seq_failed_count_by_exp if seq_failed_count_by_exp else None)
plot_loss_reasons_by_gene(df, AUDIT_PLOT_DIR / "loss_reasons_by_gene.png")
plot_wellgrids(df, WELLGRID_DIR, seq_failed_wells=seq_failed_by_exp)

if not registry.empty:
    plot_failure_venn(registry, AUDIT_PLOT_DIR / "failure_mode_venn.png")
    plot_final_usable_by_gene(registry, AUDIT_PLOT_DIR / "final_usable_by_gene.png")

print("\nDone.")
