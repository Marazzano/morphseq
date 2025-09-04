from __future__ import annotations
from pathlib import Path

from src.build.build04_perform_embryo_qc import perform_embryo_qc


def run_build04(
    root: str | Path,
    dead_lead_time: int = 2,
    pert_key_path: str | None = None,
    auto_augment_pert_key: bool = True,
    write_augmented_key: bool = False,
) -> None:
    perform_embryo_qc(
        root=str(Path(root)),
        dead_lead_time=dead_lead_time,
        pert_key_path=pert_key_path,
        auto_augment_pert_key=auto_augment_pert_key,
        write_augmented_key=write_augmented_key,
    )
    print("✔️  Build04 complete.")
