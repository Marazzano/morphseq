#!/usr/bin/env python
"""Regenerate manifests/front_half_archive.txt from what is actually on disk.

The manifest is the contract between `qsub -t` and the experiment list: SGE array task N runs the
Nth non-comment line. Keeping it a generated FILE (rather than globbing inside the .sge) means the
task->experiment mapping is auditable after the fact — you can read the manifest months later and
know exactly what task 43 was, even if the raw tree has changed since.

Each line is "<experiment> <scope>". The scope is REQUIRED, not decoration: the Snakefile resolves
raw inputs as RAW_IMAGES_DIR / MICROSCOPE / {experiment} from a single global `microscope` config
key, so a run is single-scope by construction. The .sge passes each task's scope through as
--config microscope=<scope>, which is what lets one array span both microscopes.

An experiment is included only if it has BOTH raw image data and a metadata workbook. Experiments
missing metadata are printed to stderr and excluded rather than silently dropped: staging fails loud
on missing metadata, so submitting them would just burn a task slot to reach the same conclusion.

Usage:
    python make_front_half_manifest.py [--input-root PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys

DEFAULT_INPUT_ROOT = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/input"
# Keyence first: ~1.4h and ~7-27GB each, versus YX1's 227-694GB nd2 files. With a concurrency cap
# the cheap half finishes early, so a systematic failure shows up in the first hour instead of the
# second day.
SCOPE_ORDER = ("Keyence", "YX1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "front_half_archive.txt"),
    )
    args = ap.parse_args()

    raw = os.path.join(args.input_root, "raw_image_data")
    meta = os.path.join(args.input_root, "plate_metadata")
    if not os.path.isdir(raw):
        print(f"ERROR: no raw_image_data under {args.input_root!r}", file=sys.stderr)
        return 2
    if not os.path.isdir(meta):
        print(f"ERROR: no plate_metadata under {args.input_root!r}", file=sys.stderr)
        return 2

    stems = {
        os.path.basename(p).replace("_well_metadata.xlsx", "").replace(".xlsx", "")
        for p in glob.glob(os.path.join(meta, "*.xlsx"))
    }

    included: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for scope in SCOPE_ORDER:
        d = os.path.join(raw, scope)
        if not os.path.isdir(d):
            continue
        for e in sorted(
            x for x in os.listdir(d)
            if os.path.isdir(os.path.join(d, x)) and not x.startswith(".") and x != "ignore"
        ):
            (included if e in stems else skipped).append((scope, e))

    with open(args.out, "w") as f:
        f.write(f"# front_half archive manifest — generated {datetime.date.today()}\n")
        f.write("#\n")
        f.write("# FORMAT:  <experiment> <scope>     (scope is passed as --config microscope=<scope>;\n")
        f.write("#          the Snakefile reads raw inputs from RAW_IMAGES_DIR/<scope>/<experiment>)\n")
        f.write("# ORDER:   SGE array task N == Nth non-comment line. Keyence first (cheap, ~1.4h\n")
        f.write("#          each) so failures surface early; YX1 after (large nd2s).\n")
        f.write("# CONTENT: only experiments with BOTH raw data and a metadata workbook.\n")
        f.write("#\n")
        f.write("# Regenerate with make_front_half_manifest.py — do not hand-edit.\n")
        for scope, e in included:
            f.write(f"{e} {scope}\n")

    n_key = sum(1 for s, _ in included if s == "Keyence")
    print(f"wrote {args.out}")
    print(f"  tasks: {len(included)}  (Keyence 1-{n_key}, YX1 {n_key + 1}-{len(included)})")
    if skipped:
        print(f"\nEXCLUDED — raw data present but no metadata workbook ({len(skipped)}):",
              file=sys.stderr)
        for s, e in skipped:
            print(f"  [{s}] {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
