from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_DF01_COLS = [
    "snip_id", "embryo_id", "experiment_date", "well", "time_int",
    "Time Rel (s)", "predicted_stage_hpf", "surface_area_um",
    "short_pert_name", "phenotype", "control_flag", "temperature",
    "medium", "use_embryo_flag",
]


def check_df01_schema(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_DF01_COLS if c not in df.columns]
    return missing


def units_check(df: pd.DataFrame, n_rows: int = 3, tol: float = 1e-3) -> list[str]:
    """Rough units sanity: surface_area_um ≈ area_px × (um_per_px)^2.

    Returns list of row indices (as strings) that fail tolerance.
    """
    fails: list[str] = []
    sample = df.head(n_rows)
    for idx, row in sample.iterrows():
        try:
            if row.get("Height (px)", 0) and row.get("Height (um)", 0):
                um_per_px = float(row["Height (um)"]) / float(row["Height (px)"])
            else:
                continue
            expected = float(row.get("area_px", 0.0)) * (um_per_px ** 2)
            actual = float(row.get("surface_area_um", 0.0))
            if actual == 0:
                fails.append(str(idx))
                continue
            err = abs(actual - expected) / actual
            if err > tol:
                fails.append(str(idx))
        except Exception:
            fails.append(str(idx))
    return fails


def mask_paths_check(root: Path, df: pd.DataFrame) -> list[str]:
    """Verify exported mask files exist if column present.
    Looks under segmentation_sandbox/data/exported_masks/{exp}/masks.
    Returns list of missing relative paths.
    """
    if "exported_mask_path" not in df.columns:
        return []
    missing: list[str] = []
    for exp in df["experiment_date"].astype(str).unique():
        base = Path(root) / "segmentation_sandbox" / "data" / "exported_masks" / exp / "masks"
        sub = df[df["experiment_date"].astype(str) == exp]
        for rel in sub["exported_mask_path"].unique():
            if not (base / rel).exists():
                missing.append(str((base / rel)))
    return missing


def run_validation(root: str | Path, exp: str | None, df01: str, checks: str) -> None:
    root = Path(root)
    df01_path = Path(df01)
    if not df01_path.is_absolute():
        df01_path = root / df01_path
    if not df01_path.exists():
        raise FileNotFoundError(f"df01 not found: {df01_path}")
    df = pd.read_csv(df01_path)
    checks_list = [c.strip() for c in checks.split(",") if c.strip()]

    if "schema" in checks_list:
        missing = check_df01_schema(df)
        if missing:
            raise SystemExit(f"Schema check failed. Missing columns: {missing}")
        print("✅ Schema check passed.")

    if "units" in checks_list:
        failures = units_check(df)
        if failures:
            raise SystemExit(f"Units check failed for rows: {failures}")
        print("✅ Units check passed.")

    if "paths" in checks_list:
        miss = mask_paths_check(root, df)
        if miss:
            raise SystemExit(f"Mask path check failed. Missing: {miss[:5]}... (total {len(miss)})")
        print("✅ Mask path check passed.")

