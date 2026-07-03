"""Run every per-product report script and emit ONE self-contained review page: output/index.html.

Purpose: quality review BEFORE any DAG wiring. Each report lives in report_scripts/<product>/report.py
written exactly as its future src/data_pipeline/<product>/report.py — only the input-path seam differs.
This runner hand-drives them against real 20250912 output and lays every artifact out in a single HTML
grouped stage -> step (mirroring the output tree), PNGs embedded as base64 so the file is portable.

Usage:
    PYTHONPATH=src conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260702_qc_reporting_v1/build_review_index.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))  # so `report_scripts` is importable as a package

from data_pipeline.viz import HtmlReport  # noqa: E402

OUTPUT_ROOT = _HERE / "output"

# (stage, product) in pipeline order — drives both grouping and which report.py to run.
REPORTS: list[tuple[str, str]] = [
    ("object_extraction", "frame_detections"),
    ("object_extraction", "physical_embryo_registry"),
    ("feature_extraction", "mask_geometry"),
    ("feature_extraction", "curvature_metrics"),
    ("quality_control", "surface_area_qc"),
    ("quality_control", "focus_qc"),
    ("quality_control", "motion_blur_qc"),
    ("quality_control", "death_detection"),
]


def _run_one(product: str) -> tuple[list[Path], str | None]:
    out_dir = OUTPUT_ROOT / product
    try:
        mod = importlib.import_module(f"report_scripts.{product}.report")
        return mod.build(out_dir), None
    except Exception:
        return [], traceback.format_exc()


def main() -> None:
    report = HtmlReport("QC / feature report review", subtitle="20250912")
    # REPORTS is in pipeline order, so sections/subsections land in that order (first-seen).
    for stage, product in REPORTS:
        pngs, err = _run_one(product)
        if err:
            report.add_error(err, section=stage, subsection=product)
            print(f"[{product}] ERROR\n{err}")
        elif not pngs:
            report.add_note("stub — no report artifacts (see report.py)", section=stage, subsection=product)
            print(f"[{product}] stub (0 artifacts)")
        else:
            report.add_images(pngs, section=stage, subsection=product)
            print(f"[{product}] {len(pngs)} artifact(s)")

    for path in report.write(OUTPUT_ROOT / "index.html"):  # writes index.html + index.pdf
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
