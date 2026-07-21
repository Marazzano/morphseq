#!/usr/bin/env python
"""Generate the back_half (analysis_ready) array manifest, ordered by science priority.

An experiment is ELIGIBLE only if its front_half finished cleanly, which is decided by
shard/validated PARITY: every per-well frame_inventory.csv has a matching .validated sibling, and
there is at least one shard. That parity check is the readiness signal -- a bare file count cannot
distinguish "96 wells validated" from "96 shards written, 14 validated", and the latter fails
mid-run once the back half starts consuming them.

Experiments with shards but incomplete validation (PARTIAL) are deliberately EXCLUDED and listed in
the header, so a stalled front_half never silently becomes a stalled back_half.

Regenerate after any front_half run:
    python make_back_half_manifest.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ACQ = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output/acquisition"
)
MANIFEST = Path(__file__).with_name("back_half_archive.txt")
# Scope is carried per-experiment, exactly as in the front_half manifest: the Snakefile resolves raw
# inputs as RAW_IMAGES_DIR/<microscope>/<experiment> from ONE global key, so leaving it out pins the
# whole array to config.yaml's default and every experiment of the other scope dies with a
# MissingInputException. Source of truth is the front_half manifest, which already has it right.
FRONT_MANIFEST = Path(__file__).with_name("front_half_archive.txt")

# Highest priority first; an experiment lands in the first group whose prefix it matches.
PRIORITY_PREFIXES = (
    "20240813",
    "20260702",
    "20250612",
    "20260320",
    "20260324",
    "20260331",
    "20260414",
    "20260415",
)


def _priority_rank(name: str) -> int:
    for i, pat in enumerate(PRIORITY_PREFIXES):
        if name.startswith(pat):
            return i
    return len(PRIORITY_PREFIXES)


def _readiness(exp_dir: Path) -> tuple[str, int, int]:
    """Return (status, n_shards, n_validated) for one experiment."""
    per_well = exp_dir / "frame_inventory" / "per_well"
    if not per_well.is_dir():
        return ("NOT_STARTED", 0, 0)
    shards = list(per_well.rglob("*frame_inventory.csv"))
    validated = list(per_well.rglob("*frame_inventory.csv.validated"))
    n_s, n_v = len(shards), len(validated)
    if n_s == 0:
        return ("NO_SHARDS", 0, 0)
    if n_s == n_v:
        return ("READY", n_s, n_v)
    return ("PARTIAL", n_s, n_v)


def _scope_map() -> dict[str, str]:
    """{experiment: scope} from the front_half manifest ("<experiment> <scope>" lines)."""
    if not FRONT_MANIFEST.is_file():
        sys.exit(f"front_half manifest not found (needed for scope): {FRONT_MANIFEST}")
    out: dict[str, str] = {}
    for line in FRONT_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def main() -> None:
    if not ACQ.is_dir():
        sys.exit(f"acquisition root not found: {ACQ}")

    scopes = _scope_map()

    ready: list[tuple[str, int, str]] = []
    skipped: list[tuple[str, str, int, int]] = []

    for exp_dir in sorted(p for p in ACQ.iterdir() if p.is_dir()):
        status, n_s, n_v = _readiness(exp_dir)
        if status != "READY":
            skipped.append((exp_dir.name, status, n_s, n_v))
            continue
        scope = scopes.get(exp_dir.name)
        if scope is None:
            # No scope == cannot resolve raw inputs. Exclude loudly rather than emit a line that
            # would fail at DAG-build time on the cluster.
            skipped.append((exp_dir.name, "NO_SCOPE", n_s, n_v))
            continue
        ready.append((exp_dir.name, n_s, scope))

    ready.sort(key=lambda t: (_priority_rank(t[0]), t[0]))

    n_priority = sum(1 for e, _, _ in ready if _priority_rank(e) < len(PRIORITY_PREFIXES))

    lines = [
        f"# back_half (analysis_ready) manifest — generated {date.today().isoformat()}",
        "#",
        "# FORMAT:  <experiment> <scope>   (SGE array task N == Nth non-comment line; scope is",
        "#          passed as --config microscope=<scope>)",
        "# ORDER:   priority groups first, in this order:",
        f"#          {', '.join(PRIORITY_PREFIXES)}",
        f"#          tasks 1-{n_priority} are priority; {n_priority + 1}-{len(ready)} are the remainder.",
        "# CONTENT: only experiments whose front_half shards are ALL validated (parity check).",
        "#",
        f"# ELIGIBLE: {len(ready)}    EXCLUDED: {len(skipped)}",
        "#",
        "# EXCLUDED (front_half incomplete — fix before adding these):",
    ]
    for name, status, n_s, n_v in skipped:
        lines.append(f"#   {name:<46} {status:<12} shards={n_s} validated={n_v}")
    lines += [
        "#",
        "# Regenerate with make_back_half_manifest.py — do not hand-edit.",
    ]
    lines += [f"{name} {scope}" for name, _, scope in ready]

    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {MANIFEST}")
    print(f"  eligible : {len(ready)}  (priority: {n_priority})")
    print(f"  excluded : {len(skipped)}")
    print("\n  first 12 tasks:")
    for i, (name, n_s, scope) in enumerate(ready[:12], start=1):
        print(f"    {i:>3}  {name:<46} {scope:<8} wells={n_s}")


if __name__ == "__main__":
    main()
