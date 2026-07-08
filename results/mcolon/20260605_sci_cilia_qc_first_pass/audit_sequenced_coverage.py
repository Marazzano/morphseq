"""
audit_sequenced_coverage.py — per-well coverage audit for sequenced b9d2 and cep290 embryos.

For each non-sci_ plate, reads the `sequenced` sheet from the plate Excel and checks the
build04 qc_staged CSV to classify every sequenced well as:
  OK          — in build04 and passes QC (usable_embryo=1)
  QC_EXCLUDED — in build04 but flagged out (usable_embryo=0); shows which flags
  ABSENT      — not in build04 at all (never stitched / GDino miss / never imaged)
  NO_BUILD04  — build04 CSV doesn't exist yet for this experiment

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/audit_sequenced_coverage.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PLAY = REPO / "morphseq_playground"
PLATE_META = REPO / "metadata/plate_metadata"
BUILD04 = PLAY / "metadata/build04_output"

WELLS = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]

EXPS = [
    "20260324_cep290_18hpf_24hpf_plate02",
    "20260324_cep290_18hpf_plate01",
    "20260324_cep290_24hpf_plate01",
    "20260324_cep290_24hpf_plate02",
    "20260324_cep290_30hpf_plate01",
    "20260324_cep290_30hpf_plate02",
    "20260331_b9d2_18hpf_plate01",
    "20260331_b9d2_18hpf_plate02",
    "20260414_b9d2_14hpf_plate01",
    "20260414_b9d2_14hpf_plate02",
    "20260414_b9d2_30hpf_plate01",
    "20260414_b9d2_30hpf_plate02",
    "20260415_b9d2_30to48hpf_plate01_t02",
    "20260415_b9d2_30to48hpf_plate02_t02",
    "20260415_cep290_18hpf_plate03",
    "20260415_cep290_30to48hpf_plate02_t01",
    "20260416_cep290_30to48hpf_plate01_t02",
    "20260416_cep290_30to48hpf_plate02_t02",
]


def sequenced_grid(exp: str) -> dict[str, int] | None:
    """Parse the 8×12 `sequenced` sheet → {well: code}. Returns None if no Excel found."""
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


def audit() -> pd.DataFrame:
    rows = []
    for exp in EXPS:
        grid = sequenced_grid(exp)
        if grid is None:
            print(f"  WARNING: no plate Excel found for {exp}")
            continue

        seq_wells = {w for w, v in grid.items() if v in (1, 2)}
        if not seq_wells:
            continue

        b04_path = BUILD04 / f"qc_staged_{exp}.csv"
        if not b04_path.exists():
            for w in sorted(seq_wells):
                rows.append({"exp": exp, "well": w, "seq_code": grid[w],
                             "status": "NO_BUILD04", "flags": ""})
            continue

        b04 = pd.read_csv(b04_path)
        b04_wells = set(b04["well"].astype(str).str.strip())

        for w in sorted(seq_wells):
            if w not in b04_wells:
                rows.append({"exp": exp, "well": w, "seq_code": grid[w],
                             "status": "ABSENT", "flags": ""})
            else:
                row = b04[b04["well"] == w].iloc[0]
                usable = bool(row.get("usable_embryo", 1))
                flags = [c for c in b04.columns if c.endswith("_flag") and row.get(c, 0)]
                status = "OK" if usable else "QC_EXCLUDED"
                rows.append({"exp": exp, "well": w, "seq_code": grid[w],
                             "status": status, "flags": "|".join(flags)})
    return pd.DataFrame(rows)


def main() -> None:
    df = audit()

    print("=== SEQUENCED WELL COVERAGE (b9d2 + cep290, excl. sci_) ===")
    print(df["status"].value_counts().to_string())
    print(f"\nTotal sequenced wells in Excel: {len(df)}")

    print("\n=== BY EXPERIMENT (OK / QC_EXCL / ABSENT / NO_BUILD04) ===")
    pivot = df.groupby(["exp", "status"]).size().unstack(fill_value=0)
    for col in ["OK", "QC_EXCLUDED", "ABSENT", "NO_BUILD04"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["OK", "QC_EXCLUDED", "ABSENT", "NO_BUILD04"]]
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.to_string())

    absent = df[df["status"] == "ABSENT"]
    print("\n=== ABSENT detail ===")
    print(absent[["exp", "well", "seq_code"]].to_string(index=False) if len(absent) else "(none)")

    excl = df[df["status"] == "QC_EXCLUDED"]
    print("\n=== QC_EXCLUDED detail ===")
    print(excl[["exp", "well", "seq_code", "flags"]].to_string(index=False) if len(excl) else "(none)")


if __name__ == "__main__":
    main()
