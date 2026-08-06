#!/usr/bin/env python
"""Create the executable SeaHub surface-area calibration notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from jupyter_client import KernelManager


HERE = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = HERE / "02_seahub_surface_area_calibration.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (morphseq-env)",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    notebook["cells"] = [
        markdown(
            """
# SeaHub effective pixel-scale analysis from 2D mask area

**Question.** Can stage-matched control embryos constrain the unknown SeaHub
pixel calibration well enough to explain the surface-area QC failures?

**Executive answer.** Yes, as a rough *effective biological calibration*, but
not as physical metrology and not as one global number. The strict-WT
comparison implies approximately **4.25 µm/px at 12–24 hpf**, **6.90 µm/px at
36 hpf**, and **7.56 µm/px at 48–96 hpf** (stage-balanced regime medians).
The early/late split is about 1.8-fold. This is consistent with capture-scale
or magnification changes across source images. A single global SeaHub
calibration is therefore not defensible.

This notebook is analysis-only. It reads the quarantined July 30 products and
does not alter active pipeline outputs or pipeline code.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

NOTEBOOK_DIR = Path.cwd().resolve()
if NOTEBOOK_DIR.name != "20260723_seahub":
    NOTEBOOK_DIR = Path("results/nlammers/20260723_seahub").resolve()
sys.path.insert(0, str(NOTEBOOK_DIR))

from seahub_surface_area_calibration import run_analysis

results = run_analysis()
print(
    f"Loaded {len(results['rows']):,} mask/stage comparison rows from "
    f"{results['rows']['experiment_id'].nunique()} completed or partial shards."
)
print(
    f"Calibration cohort: {results['selected_controls']['well_id'].nunique():,} "
    f"reference-like controls, {results['selected_controls']['calibration_fov_id'].nunique()} "
    f"source FOVs, {results['selected_controls']['predicted_stage_hpf'].nunique()} stages."
)
"""
        ),
        markdown(
            """
## 1. Preferred strict-WT estimates

Each source crop is intended to contain one embryo. Where the downstream
segmenter produced multiple masks, the analysis uses the largest mask for that
intended well. It then takes a median across the eight embryos in each source
FOV and a median across FOVs, avoiding eightfold pseudoreplication.

The confidence intervals resample source FOV medians. At 12 and 15 hpf only
one control FOV is available, so those intervals resample embryos within that
single FOV and should be treated as especially weak.
"""
        ),
        code(
            """
stage = results["stage_summary"].copy()
display(
    stage[[
        "stage_hpf", "n_embryos", "n_fovs",
        "strict_wt_effective_um_per_px",
        "strict_wt_bootstrap_ci95_low", "strict_wt_bootstrap_ci95_high",
        "strict_wt_reference_n",
        "legacy_curve_effective_um_per_px",
    ]].round(3).style.format({
        "stage_hpf": "{:.0f}",
        "strict_wt_effective_um_per_px": "{:.3f}",
        "strict_wt_bootstrap_ci95_low": "{:.3f}",
        "strict_wt_bootstrap_ci95_high": "{:.3f}",
        "legacy_curve_effective_um_per_px": "{:.3f}",
    })
)

display(results["regime_summary"].round(3))
"""
        ),
        code(
            """
display(Image(filename=str(results["figure_paths"]["effective_scale"])))
"""
        ),
        markdown(
            """
The colored dots are source-FOV estimates against the strict-WT stage median;
black diamonds are the stage recommendations. Open gray circles show the
corresponding answer against the packaged legacy QC curve. The dashed line is
the current 7.8 µm/px placeholder.

Late-stage estimates are near the placeholder, but the early-stage images are
captured at much finer effective resolution. The integration code crops and
center-pads native pixels (`acquisition/seahub/integration.py`, crop → new
canvas → paste) without resizing, so the source-scale differences are carried
through materialization rather than introduced by it.
"""
        ),
        markdown(
            """
## 2. Calculation

Despite the historical “surface area” label, `area_um2` is a **2D projected
mask area**, not a 3D surface. Mask geometry obeys

\\[
A_{\\mu m^2} = A_{px}\\,s^2,
\\]

where \(s\) is µm/px. Because the quarantined SeaHub geometry was generated
with the known placeholder \(s_0=7.8\), pixel area is recoverable exactly:

\\[
A_{px}=A_{placeholder}/7.8^2.
\\]

For a SeaHub control and a stage-matched reference median:

\\[
\\hat{s}=\\sqrt{A_{reference,p50}/A_{SeaHub,px}}.
\\]

The square root is essential: area scales with the square of linear pixel
size.
"""
        ),
        markdown(
            """
## 3. Critical reference-cohort audit

The operational `surface_area_reference_v1.csv` is the exact curve used by the
pipeline, but its archived builder did **not** produce a clean WT cohort.
`control_flag` is true for every one of 724,816 archived source rows, and the
builder ORs that flag into its genotype filter. After the required-value
filter, only 22,958/285,844 rows (8.0%) match an explicit WT
genotype/phenotype criterion; 262,886 rows (92.0%) were admitted only through
the universal control flag. Those 285,844 frames represent only 6,369
embryos, adding substantial time-series pseudoreplication.

Accordingly this notebook reports two distinct quantities:

1. **Strict-WT effective scale** — preferred for biological interpretation.
   Strict filter: genotype in `wik`, `ab`, `wik-ab`, `wik/ab`, `AB`, `WIK`
   and/or phenotype `wt`; no chemical perturbation; `use_embryo_flag=True`.
   Raw median `area_um2` is taken within ±0.25 hpf of each SeaHub stage.
2. **Effective scale relative to the legacy SA curve** — useful only for
   predicting behavior of the current operational QC comparator.
"""
        ),
        code(
            """
display(results["cohort_audit"].T.rename(columns={0: "value"}))
display(results["strict_reference"].round(1))
display(results["provenance"])
"""
        ),
        markdown(
            """
## 4. Actual stage-matched image comparison

The right column below contains real raw frames from the strict-WT reference
cohort used for the audit; the left contains SeaHub control snips nearest each
stage estimate. Images use independent display contrast and do not share a
scale bar, so this is visual evidence rather than quantitative metrology.
The exact image paths and filter are saved in `reference_image_manifest.csv`.
"""
        ),
        code(
            """
display(Image(filename=str(results["figure_paths"]["image_comparison"])))
"""
        ),
        markdown(
            """
## 5. Source-experiment × stage resolution

The source FOV—or, where controls are sparse, source-experiment × stage—is the
most defensible provisional calibration unit. A stage-only lookup averages
over visible FOV-to-FOV variation and should be treated as a fallback.
"""
        ),
        code(
            """
display(results["experiment_stage_summary"].round(3))
"""
        ),
        markdown(
            """
## 6. What this means for the current SA QC

The plot below is deliberately an **operational** projection: it calibrates
against the same flawed legacy curve that the present QC rule uses. The active
band is `[0.9 × p5, 1.4 × p95]`. It is not a claim that the resulting pass
rate is biologically correct.
"""
        ),
        code(
            """
display(Image(filename=str(results["figure_paths"]["area_comparison"])))
display(Image(filename=str(results["figure_paths"]["qc_projection"])))
display(results["qc_projection"].assign(
    outlier_percent=lambda x: 100*x["outlier_fraction"]
).round(3))
"""
        ),
        markdown(
            """
## 7. Sensitivity and limitations

- **Control definition.** The primary SeaHub cohort includes genetic
  `ctrl-inj`, chemical vehicle/untreated controls, and nominal 28C controls.
  Nonstandard 24C/34C controls are excluded from the primary estimate and
  included only in sensitivity tables.
- **Mask multiplicity.** Each SeaHub crop is one intended embryo/well. The
  largest mask per well is primary; `e01` and all-mask alternatives are saved
  below. Largest-vs-e01 has little influence on the headline stage split.
- **QC exclusions.** Surface-area exclusions are not used because that would
  be circular. Death/focus/motion calls are not used because these are
  single-z SeaHub images and those products are known to be annotate-only or
  unreliable in this scope. FOV medians and largest-mask selection provide
  robustness without conditioning on the outcome being calibrated.
- **Biology is not a ruler.** Pose, strain, staging error, segmentation, and
  true biological size all contribute. This is an effective scale estimate,
  not microscope metrology.
- **Sparse late strict-WT reference.** The strict reference has only 21 rows
  at 72 hpf and 5 at 96 hpf. Late estimates remain provisional.
- **Recommended pipeline policy.** Keep physical-size QC annotate-only for
  SeaHub until independent calibration is available. If provisional physical
  features are needed, carry a calibration at source-FOV or
  source-experiment × stage level with explicit provenance; do not replace
  7.8 with one new global constant.
"""
        ),
        code(
            """
display(
    results["sensitivity"].round(3).style.format({
        "row_weighted_median_um_per_px": "{:.3f}",
        "fov_weighted_median_um_per_px": "{:.3f}",
        "stage_balanced_median_um_per_px": "{:.3f}",
    })
)
"""
        ),
        markdown(
            """
## 8. Output inventory

All derived tables and figures are under
`outputs/surface_area_calibration/`. Re-running this notebook regenerates them
from the stable quarantine roots.
"""
        ),
        code(
            """
for label, path in sorted(results["table_paths"].items()):
    print(f"TABLE  {label:28s} {path}")
for label, path in sorted(results["figure_paths"].items()):
    print(f"FIGURE {label:28s} {path}")
"""
        ),
    ]
    return notebook


def execute_notebook(notebook: nbf.NotebookNode) -> nbf.NotebookNode:
    """Execute code cells with the installed morphseq-env kernel."""
    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(HERE))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        execution_count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            message_id = client.execute(cell.source)
            while True:
                message = client.get_iopub_msg(timeout=300)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                message_type = message["msg_type"]
                content = message["content"]
                if message_type == "status" and content["execution_state"] == "idle":
                    break
                if message_type == "stream":
                    cell.outputs.append(
                        nbf.v4.new_output(
                            "stream",
                            name=content["name"],
                            text=content["text"],
                        )
                    )
                elif message_type in {"display_data", "execute_result"}:
                    output_kwargs = {
                        "data": content["data"],
                        "metadata": content.get("metadata", {}),
                    }
                    if message_type == "execute_result":
                        output_kwargs["execution_count"] = content.get(
                            "execution_count"
                        )
                    cell.outputs.append(
                        nbf.v4.new_output(message_type, **output_kwargs)
                    )
                elif message_type == "error":
                    cell.outputs.append(
                        nbf.v4.new_output(
                            "error",
                            ename=content["ename"],
                            evalue=content["evalue"],
                            traceback=content["traceback"],
                        )
                    )
                    raise RuntimeError(
                        f"Notebook cell {execution_count} failed: "
                        f"{content['ename']}: {content['evalue']}"
                    )
                elif message_type == "clear_output":
                    cell.outputs = []
        return notebook
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute all code cells with the morphseq-env python3 kernel.",
    )
    args = parser.parse_args()
    notebook = build_notebook()
    if args.execute:
        notebook = execute_notebook(notebook)
    nbf.write(notebook, OUTPUT_NOTEBOOK)
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
