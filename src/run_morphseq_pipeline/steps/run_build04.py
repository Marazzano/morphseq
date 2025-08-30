from __future__ import annotations
from pathlib import Path

from src.build.build04_perform_embryo_qc import perform_embryo_qc


def run_build04(root: str | Path, dead_lead_time: int = 2) -> None:
    perform_embryo_qc(root=str(Path(root)), dead_lead_time=dead_lead_time)
    print("✔️  Build04 complete.")
