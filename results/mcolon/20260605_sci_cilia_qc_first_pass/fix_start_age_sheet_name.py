#!/usr/bin/env python
"""fix_start_age_sheet_name.py — audit + fix start_stage_hpf → start_age_hpf in plate Excel files.

Two jobs:
  1. For any file that still has 'start_stage_hpf', rename that sheet to 'start_age_hpf'
     IN PLACE, verifying that ONLY the sheet name changes (all cell values identical).
  2. Overwrite the stale backup files in _backup_before_sheet_rename_20260606/ with the
     current correct versions (so the backup dir reflects the current state, not the old broken state).

Usage:
    # dry run (default) — show what would change, write nothing
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/fix_start_age_sheet_name.py

    # apply
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/fix_start_age_sheet_name.py --apply
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PLATE_META = REPO / "metadata/plate_metadata"
WM = REPO / "morphseq_playground/metadata/well_metadata"
PM_BACKUP = PLATE_META / "_backup_before_sheet_rename_20260606"
WM_BACKUP = WM / "_backup_before_sheet_rename_20260606"

EXPS = """
20260319_cilia_crispant_18hpf
20260319_cilia_crispant_24hpf
20260319_cilia_crispant_30hpf
20260320_cilia_crispant_48hpf
20260324_cep290_18hpf_24hpf_plate02
20260324_cep290_18hpf_plate01
20260324_cep290_24hpf_plate01
20260324_cep290_24hpf_plate02
20260324_cep290_30hpf_plate01
20260324_cep290_30hpf_plate02
20260331_b9d2_18hpf_plate01
20260331_b9d2_18hpf_plate02
20260414_b9d2_14hpf_plate01
20260414_b9d2_14hpf_plate02
20260414_b9d2_30hpf_plate01
20260414_b9d2_30hpf_plate02
20260415_b9d2_30to48hpf_plate01_t02
20260415_b9d2_30to48hpf_plate02_t02
20260415_cep290_18hpf_plate03
20260416_cep290_30to48hpf_plate01_t02
20260415_cep290_30to48hpf_plate02_t01
20260416_cep290_30to48hpf_plate02_t02
20260414_sci_b9d2_48hpf_plate01
20260415_sci_cep290_48hpf_plate01
""".split()


def find_excel(base_dir: Path, exp: str) -> Path | None:
    for cand in (f"{exp}_well_metadata.xlsx", f"{exp}.xlsx"):
        p = base_dir / cand
        if p.exists():
            return p
    return None


def sheet_values(path: Path, sheet: str) -> np.ndarray:
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    return df.fillna("").to_numpy(dtype=str)


def rename_sheet_inplace(path: Path, old_name: str, new_name: str, dry: bool) -> str:
    """Rename a sheet in an xlsx file. Verifies cell values are unchanged after rename."""
    import openpyxl

    before = sheet_values(path, old_name)

    if not dry:
        wb = openpyxl.load_workbook(path)
        ws = wb[old_name]
        ws.title = new_name
        wb.save(path)
        wb.close()

        after = sheet_values(path, new_name)
        if not np.array_equal(before, after):
            return f"  ERROR: cell values changed after rename! {path.name}"

    return f"  {'[DRY]' if dry else 'RENAMED'}: {path.name}  '{old_name}' → '{new_name}'"


def main() -> None:
    dry = "--apply" not in sys.argv
    print(f"{'DRY RUN' if dry else 'APPLYING'} — {len(EXPS)} experiments\n")

    rename_needed = []
    backup_update_needed = []

    for exp in EXPS:
        for label, base_dir, backup_dir in [
            ("PLATE_META", PLATE_META, PM_BACKUP),
            ("WM", WM, WM_BACKUP),
        ]:
            p = find_excel(base_dir, exp)
            if p is None:
                print(f"  MISSING [{label}]: {exp}")
                continue

            with pd.ExcelFile(p) as xlf:
                sheets = xlf.sheet_names

            if "start_stage_hpf" in sheets:
                print(f"  NEEDS RENAME [{label}]: {p.name}")
                rename_needed.append((p, dry))
                msg = rename_sheet_inplace(p, "start_stage_hpf", "start_age_hpf", dry)
                print(msg)
            elif "start_age_hpf" in sheets:
                print(f"  OK [{label}]: {p.name}")
            else:
                print(f"  WARN — no start_age_hpf or start_stage_hpf [{label}]: {p.name}")
                continue

            # Check if the backup needs updating
            bak_name = p.name if (backup_dir / p.name).exists() else None
            if bak_name and backup_dir.exists():
                bak_path = backup_dir / bak_name
                with pd.ExcelFile(bak_path) as xlf:
                    bak_sheets = xlf.sheet_names
                if "start_stage_hpf" in bak_sheets:
                    backup_update_needed.append((p, bak_path))
                    print(f"    BACKUP STALE: {bak_path.name}  (has 'start_stage_hpf')")
                    if not dry:
                        shutil.copy2(p, bak_path)
                        print(f"    BACKUP UPDATED: {bak_path.name}")
                    else:
                        print(f"    [DRY] would overwrite backup: {bak_path.name}")

    print(f"\n--- Summary ---")
    print(f"Files needing sheet rename: {len(rename_needed)}")
    print(f"Stale backups to overwrite: {len(backup_update_needed)}")
    if dry:
        print("\nNo files written. Re-run with --apply to apply.")


if __name__ == "__main__":
    main()
