"""focus_qc report (STUB product) — interior-edge-fraction histogram + gallery (C+E)."""

from __future__ import annotations

from pathlib import Path

from .._scalar_qc import build_scalar_qc


def build(output_dir: Path) -> list[Path]:
    return build_scalar_qc("focus_qc", output_dir)
