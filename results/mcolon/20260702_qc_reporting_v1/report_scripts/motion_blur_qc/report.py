"""motion_blur_qc report (STUB product) — bad-z-pair-fraction histogram + gallery (C+E)."""

from __future__ import annotations

from pathlib import Path

from .._scalar_qc import build_scalar_qc


def build(output_dir: Path) -> list[Path]:
    return build_scalar_qc("motion_blur_qc", output_dir)
