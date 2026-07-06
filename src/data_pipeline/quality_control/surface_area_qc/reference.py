"""surface_area reference loader — the only door to the packaged percentile curve.

`entrypoint.py` calls `load_packaged_surface_area_reference(version=...)`, validates the
result, then passes the dataframe into `compute.py`. `compute.py` never reads the reference
file itself. The reference is a packaged source asset beside this code, resolved relative to
this module — never through `paths.py` and never as a raw caller-supplied string.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .reference_contract import validate_surface_area_reference

_REFERENCES_DIR = Path(__file__).resolve().parent / "references"


def packaged_reference_path(version: str = "v1") -> Path:
    """Return the on-disk path to the packaged reference for ``version`` (e.g. ``v1``)."""
    return _REFERENCES_DIR / f"surface_area_reference_{version}.csv"


def load_packaged_surface_area_reference(version: str = "v1") -> pd.DataFrame:
    """Load + validate the packaged surface-area reference curve for ``version``."""
    path = packaged_reference_path(version)
    if not path.exists():
        available = sorted(p.name for p in _REFERENCES_DIR.glob("surface_area_reference_*.csv"))
        raise FileNotFoundError(
            f"surface_area_qc: packaged reference {path.name!r} not found in {_REFERENCES_DIR}. "
            f"Available: {available}. Fix the configured reference_version."
        )
    df = pd.read_csv(path)
    validate_surface_area_reference(df, scope_label=f"surface_area_reference_{version}")
    return df


def interpolate_reference_band(
    stage_hpf: float, surface_area_reference_df: pd.DataFrame
) -> tuple[float, float]:
    """Return the ``(p5, p95)`` band interpolated at ``stage_hpf``.

    The reference curve is pre-filled/extrapolated, so a simple linear interp on the sorted
    `stage_hpf` axis is sufficient (matches the legacy ``np.interp`` behavior). Out-of-range
    stages clamp to the nearest endpoint (np.interp default).
    """
    stages = surface_area_reference_df["stage_hpf"].to_numpy(dtype=float)
    p5 = float(np.interp(stage_hpf, stages, surface_area_reference_df["p5"].to_numpy(dtype=float)))
    p95 = float(np.interp(stage_hpf, stages, surface_area_reference_df["p95"].to_numpy(dtype=float)))
    return p5, p95
