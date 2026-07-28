"""Atomic file-write helper for adapters.

A resident server writing outputs across many requests must never leave a partial
file at the final path — Snakemake (or any downstream consumer) treats file
existence as "done," so a partial write that happens to be visible mid-write is a
silent-corruption failure mode. The fix is the standard temp-file-then-rename
pattern: write to a sibling temp path on the SAME filesystem, then `os.replace`
(atomic on POSIX for same-filesystem renames) onto the final path.

Generic + tiny; no torch/pandas dependency. Adapters call this to finalize any
output file (CSV, image, etc.) they produce.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def atomic_write_bytes(final_path: Path, data: bytes) -> None:
    """Write `data` to final_path atomically (temp file + os.replace)."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_via(final_path: Path, writer: Callable[[Path], None]) -> None:
    """Call writer(tmp_path) to produce the file, then atomically rename onto final_path.

    Use this when the producer is a library call that writes directly to a path
    (e.g. `df.to_csv(tmp_path)`) rather than one that hands back an in-memory bytes
    object.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        writer(tmp_path)
        os.replace(tmp_path, final_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
