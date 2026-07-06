"""Shared loader for the legacy pilot ground-truth, reused by every product drift benchmark.

The legacy build pipeline ran the pilot experiment ``20250912`` to completion and wrote a wide
per-snip table at::

    morphseq_playground/metadata/build04_output/qc_staged_<exp>.csv

That file (``src/build/build04_perform_embryo_qc.py``) carries the row-for-row ground truth for
every value the rebuilt feature/QC products compute — geometry, stage, fraction_alive, and the QC
flags. The drift benchmarks join the *new* pipeline output against it on ``snip_id`` to detect a
silent change in numbers or flags between the old and new code paths.

This module owns the playground path, the skip-when-absent behaviour (a checkout without the
playground must not error), and the three known representational normalizations the harness must
apply rather than paper over:

1. ``well_id`` grain — legacy ``well_id`` is the LOCAL slug (``A01``); the new pipeline's is global
   (``20250912_A01``). Both ``snip_id`` are global, so we always join on ``snip_id`` and never on
   ``well_id``.
2. time axis anchor — the ``t####`` suffix in ``snip_id`` is the shared anchor. NOTE: the
   feature_world.md spec says legacy ``time_int`` is 1-based, but the pilot ``qc_staged_20250912``
   file is in fact 0-based (it contains a ``t0000`` row whose values match the new ``t0000``). So
   the legacy ``snip_id`` ``t####`` numbering is IDENTICAL to the new 0-based ``t####`` and we align
   on it directly (no +1 shift). Verified: new ``t0000`` area/centroid match legacy ``t0000``, not
   legacy ``t0001``. (Legacy ``time_int`` may still differ from ``snip_id`` ``t####`` on other
   experiments; always trust the ``snip_id`` anchor, never the ``time_int`` column.)
3. flag vocabulary — legacy ``sam2_qc_flags`` is a comma-joined token string; the new mask_quality
   flags are booleans. ``parse_sam2_qc_flag_tokens`` exposes the token set per snip so a flag
   product harness can map tokens → booleans.

The companion helper :func:`legacy_key_from_new_snip_id` normalizes a *new* ``snip_id``
(``20250912_B01_e01_BF_t0000``: global well, channel token, 0-based ``t####``) into the legacy
``snip_id`` grammar (``20250912_B01_e01_t0000``: no channel token, same 0-based ``t####``) so the
two sides can be inner-joined directly. The only transform is dropping the channel token; the
``t####`` number is kept as-is (see the time-axis note below).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

# Absolute playground path — the legacy build output lives outside this repo. The same root that
# tests/test_qc_restoration.py already uses.
LEGACY_PLAYGROUND_ROOT = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground"
)

# New-pipeline snip_id grammar: <experiment>_<well>_e##_<channel>_t#### (0-based t).
_NEW_SNIP_ID_RE = re.compile(r"^(?P<base>.+_e\d+)_(?P<channel>[A-Za-z0-9]+)_t(?P<t>\d+)$")


def legacy_qc_staged_path(experiment_id: str = "20250912") -> Path:
    """Absolute path to the legacy wide per-snip table for ``experiment_id``."""
    return (
        LEGACY_PLAYGROUND_ROOT
        / "metadata"
        / "build04_output"
        / f"qc_staged_{experiment_id}.csv"
    )


def legacy_key_from_new_snip_id(new_snip_id: str) -> str | None:
    """Normalize a *new* ``snip_id`` into the legacy ``snip_id`` grammar, or ``None``.

    Drops the channel token; keeps the ``t####`` number unchanged (legacy pilot is 0-based, same as
    new — see the module docstring's time-axis note)::

        20250912_B01_e01_BF_t0000  ->  20250912_B01_e01_t0000

    Returns ``None`` if the input does not match the expected new-pipeline grammar, so a caller can
    report unmatched rows as a coverage gap (itself a form of drift) rather than crash.
    """
    m = _NEW_SNIP_ID_RE.match(str(new_snip_id))
    if m is None:
        return None
    base = m.group("base")
    t = int(m.group("t"))
    return f"{base}_t{t:04d}"


def parse_sam2_qc_flag_tokens(value: object) -> frozenset[str]:
    """Split a legacy ``sam2_qc_flags`` cell into its token set (empty for NaN / empty string)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return frozenset()
    text = str(value).strip()
    if not text:
        return frozenset()
    return frozenset(tok.strip() for tok in text.split(",") if tok.strip())


def load_legacy_pilot(experiment_id: str = "20250912") -> pd.DataFrame:
    """Load the legacy ``qc_staged_<exp>.csv`` keyed by ``snip_id``.

    Skips the benchmark with a clear message (``pytest.skip``) if the playground file is absent, so
    a checkout without the playground deselects rather than errors. Returns the table indexed by
    ``snip_id`` with a parsed-token column (``sam2_qc_flag_tokens``) added when ``sam2_qc_flags`` is
    present; the 1-based legacy ``time_int`` is left as-is (callers align via the ``snip_id``
    ``t####`` anchor, see :func:`legacy_key_from_new_snip_id`).
    """
    path = legacy_qc_staged_path(experiment_id)
    if not path.exists():
        pytest.skip(
            f"legacy pilot artifact not found at {path} — the drift benchmark needs the "
            f"morphseq_playground build04 output. Skipping (this is expected on a checkout "
            f"without the playground)."
        )

    df = pd.read_csv(path)
    if "snip_id" not in df.columns:
        raise ValueError(
            f"legacy artifact {path} has no 'snip_id' column; cannot use it as ground truth. "
            f"Columns present: {list(df.columns)[:10]}..."
        )

    if "sam2_qc_flags" in df.columns:
        df = df.assign(
            sam2_qc_flag_tokens=df["sam2_qc_flags"].map(parse_sam2_qc_flag_tokens)
        )

    if df["snip_id"].duplicated().any():
        dupes = df.loc[df["snip_id"].duplicated(), "snip_id"].head(5).tolist()
        raise ValueError(
            f"legacy artifact {path} has duplicate snip_id rows (e.g. {dupes}); the ground "
            f"truth must be one row per snip_id."
        )

    return df.set_index("snip_id", drop=False)
