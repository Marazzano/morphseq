"""Generate qc_flag_diagnosis.ipynb.

Follows the repo convention (cf. ../20260805/build_sheath_power_nb.py): the notebook is a thin
narrative wrapper and every non-trivial operation lives in qc_diagnostic_utils.py, so the analysis
is importable and testable outside Jupyter.

    python build_qc_diagnostic_nb.py        # writes the .ipynb (does not execute it)
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "qc_flag_diagnosis.ipynb"

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell


def cells() -> list:
    return [
        MD(
            "# Why are the 20250612 (GENE7) plates losing ~20% of embryos to QC?\n"
            "\n"
            "Manual inspection says these plates are high quality, but `analysis_ready` passes only\n"
            "74–96% of snips. Two candidate explanations:\n"
            "\n"
            "1. **segmentation is failing** — masks are wrong, so downstream features are garbage; or\n"
            "2. **QC flags are firing on good data** — masks are fine, thresholds are miscalibrated.\n"
            "\n"
            "This notebook distinguishes them. The headline work product is a montage of every failed\n"
            "embryo per plate, showing the extracted snip beside the raw focus-stacked frame with the\n"
            "segmentation outline drawn on both, so mask quality is directly inspectable.\n"
            "\n"
            "Everything is read from the pipeline's own artifacts. The surface-area band is re-derived\n"
            "using the packaged reference curve and the product's real `k_upper`/`k_lower` (imported,\n"
            "not copied), and `band_agrees` below asserts the reconstruction reproduces the flag the\n"
            "pipeline actually wrote."
        ),
        CODE(
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "REPO_ROOT = Path.cwd().parents[2] if Path.cwd().name.startswith('202') else Path.cwd()\n"
            "for p in (str(REPO_ROOT / 'src'), str(Path.cwd())):\n"
            "    if p not in sys.path:\n"
            "        sys.path.insert(0, p)\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "import qc_diagnostic_utils as Q\n"
            "\n"
            "pd.set_option('display.width', 200)\n"
            "FIGURES = Path.cwd() / 'figures'\n"
            "FIGURES.mkdir(exist_ok=True)\n"
            "\n"
            "def save(fig, name, dpi=110):\n"
            "    for ext in ('png', 'pdf'):\n"
            "        fig.savefig(FIGURES / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')\n"
            "    return fig"
        ),
        MD(
            "## 1. Load, and verify the reconstruction is faithful\n"
            "\n"
            "`band_agrees` must be `True` for every snip. If it were not, nothing downstream could be\n"
            "trusted, because the reconstructed band would not be the one that produced the flags."
        ),
        CODE(
            "df = Q.add_surface_area_band(Q.load_analysis_ready())\n"
            "print(f'snips: {len(df)}   plates: {df.experiment_id.nunique()}')\n"
            "print(f'band reconstruction matches the pipeline flag on all snips: {bool(df.band_agrees.all())}')\n"
            "print(f'  mismatches: {int((~df.band_agrees).sum())}')\n"
            "\n"
            "from data_pipeline.quality_control.surface_area_qc.config import band_statement, resolve_config\n"
            "print()\n"
            "print(band_statement(resolve_config()))"
        ),
        MD(
            "## 2. Which flags actually cause the failures?\n"
            "\n"
            "Flags are not mutually exclusive, so a snip can appear under several. `qc_fail_reasons`\n"
            "gives the exact combination per failed snip."
        ),
        CODE(
            "per_plate = (df.groupby('experiment_id')\n"
            "               .agg(snips=('snip_id', 'size'), passed=('use_snip', 'sum'))\n"
            "               .reindex(list(Q.EXPERIMENTS)))\n"
            "per_plate['failed'] = per_plate.snips - per_plate.passed\n"
            "per_plate['pass_%'] = (100 * per_plate.passed / per_plate.snips).round(1)\n"
            "for spec in Q.FLAG_SPECS:\n"
            "    counts = df.groupby('experiment_id')[spec.column].sum().reindex(list(Q.EXPERIMENTS))\n"
            "    if counts.sum():\n"
            "        per_plate[spec.column.replace('_flag', '')] = counts.astype(int)\n"
            "per_plate"
        ),
        CODE(
            "print('exact failure combinations (all six plates pooled):')\n"
            "print(df.loc[~df.use_snip, 'qc_fail_reasons'].value_counts().to_string())\n"
            "\n"
            "n_fail = int((~df.use_snip).sum())\n"
            "n_sa = int(df.loc[~df.use_snip, 'sa_outlier_flag'].sum())\n"
            "print(f'\\nsa_outlier_flag is implicated in {n_sa} of {n_fail} failures '\n"
            "      f'({100 * n_sa / n_fail:.0f}%).')"
        ),
        CODE("save(Q.plot_flag_counts(df), '01_flag_counts')\nplt.show()"),
        MD(
            "**Reading.** `sa_outlier_flag` dominates; every other flag is single digits. So the question\n"
            "reduces to: is the surface-area check right to fire?"
        ),
        MD(
            "## 3. The surface-area failures track temperature, not stage\n"
            "\n"
            "Each plate is a four-temperature block design (24 / 28.5 / 34 / 35 °C). If the check were\n"
            "responding to genuine developmental progression, the failure rate should vary with stage.\n"
            "It varies with **rearing temperature** instead."
        ),
        CODE(
            "grid = (df.groupby(['start_age_hpf', 'temperature'])\n"
            "          .agg(n=('snip_id', 'size'),\n"
            "               sa_flagged=('sa_outlier_flag', 'sum'),\n"
            "               too_small=('sa_too_small', 'sum'),\n"
            "               too_large=('sa_too_large', 'sum'),\n"
            "               median_area=('area_um2', 'median'),\n"
            "               lower_cut=('sa_lower', 'first')))\n"
            "grid['flag_%'] = (100 * grid.sa_flagged / grid.n).round(1)\n"
            "grid[['median_area', 'lower_cut']] = (grid[['median_area', 'lower_cut']] / 1000).round(0)\n"
            "grid"
        ),
        CODE("save(Q.plot_flag_rate_by_temperature(df), '02_flag_rate_by_temperature')\nplt.show()"),
        MD(
            "**Reading.** At 24 °C the flag rate reaches 65% (30 hpf) and 56% (36 hpf); at 34 °C it is\n"
            "~0–2%. Failures are almost entirely **too small**, and almost entirely in the cold cohort.\n"
            "\n"
            "The cold cohort really is smaller — that is the point of the experiment. The question is\n"
            "whether \"smaller than a 28.5 °C wildtype reference\" should be disqualifying."
        ),
        MD(
            "## 4. Are these broken masks or healthy embryos just under a tight cut?\n"
            "\n"
            "This is the discriminating measurement. A broken mask (e.g. yolk-only) lands *far* below the\n"
            "cut; a healthy-but-small embryo lands *just* below it."
        ),
        CODE("save(Q.plot_marginality(df), '03_marginality_and_threshold_sensitivity')\nplt.show()"),
        CODE(
            "small = df.loc[df.sa_too_small].copy()\n"
            "small['frac_of_cut'] = small.area_um2 / small.sa_lower\n"
            "print('area as a fraction of the lower cut, for \"too small\" failures:')\n"
            "print(small.frac_of_cut.describe().round(3).to_string())\n"
            "\n"
            "bands = {'<50% of cut': (0, .5), '50-80%': (.5, .8), '80-100%': (.8, 1.01)}\n"
            "print()\n"
            "for label, (lo, hi) in bands.items():\n"
            "    n = int(((small.frac_of_cut >= lo) & (small.frac_of_cut < hi)).sum())\n"
            "    print(f'  {label:12s} {n:3d}')"
        ),
        CODE(
            "cfg = resolve_config()\n"
            "other = [c for c in Q.FLAG_COLUMNS if c != 'sa_outlier_flag']\n"
            "other_fail = df[other].any(axis=1)\n"
            "rows = []\n"
            "for k in [0.90, 0.85, 0.80, 0.75, 0.70, 0.60]:\n"
            "    lower = (k / cfg.k_lower) * df.sa_lower\n"
            "    sa = (df.area_um2 < lower) | (df.area_um2 > df.sa_upper)\n"
            "    rows.append({'k_lower': k,\n"
            "                 'sa_flagged': int(sa.sum()),\n"
            "                 'pass_%': round(100 * (~(sa | other_fail)).sum() / len(df), 1)})\n"
            "pd.DataFrame(rows).set_index('k_lower')"
        ),
        MD(
            "**Reading.** The overwhelming majority of \"too small\" failures sit within 20% of the cut,\n"
            "median ~92% of it. That is the signature of a threshold that is too tight, not of broken\n"
            "segmentation.\n"
            "\n"
            "The sensitivity table shows the pass rate rising steeply as `k_lower` relaxes and then\n"
            "flattening around 0.70–0.75, leaving a residue of ~16–23 flags. That residue is the set of\n"
            "genuinely bad objects (see §6). The knee is the natural operating point.\n"
            "\n"
            "Context from [`surface_area_qc/config.py`](../../../src/data_pipeline/quality_control/surface_area_qc/config.py):\n"
            "`k_lower` was raised 0.7 → 0.8 → 0.9 on 2026-07-02 specifically to clear yolk-only SAM2\n"
            "masks out of the passing band, with the documented cost of \"also losing real thin/dorsal-pose\n"
            "embryos\". These plates are paying that cost at ~12 percentage points."
        ),
        MD(
            "## 5. Would indexing on a temperature-corrected stage fix it?\n"
            "\n"
            "`predicted_stage_hpf` is `start_age_hpf + elapsed_hours × rate(T)`. These are single-frame\n"
            "snapshots, so `elapsed_hours = 0` and the predicted stage collapses to nominal clock time —\n"
            "**temperature never enters**. The obvious fix is to index the band on the Arrhenius-corrected\n"
            "developmental stage instead. Tested below; it does not work."
        ),
        CODE(
            "print('Arrhenius-corrected stage, by nominal clock stage and rearing temperature:')\n"
            "for nominal in (24.0, 30.0, 36.0):\n"
            "    corrected = {t: round(Q.arrhenius_stage_hpf(nominal, t), 1) for t in Q.TEMPERATURES}\n"
            "    print(f'  nominal {nominal:.0f} hpf -> {corrected}')\n"
            "\n"
            "cf = Q.add_counterfactual_band(df)\n"
            "recovered = int((cf.sa_outlier_flag & ~cf.cf_sa_outlier_flag).sum())\n"
            "newly = int((~cf.sa_outlier_flag & cf.cf_sa_outlier_flag).sum())\n"
            "print(f'\\nrecovered by correcting the stage: {recovered}')\n"
            "print(f'newly flagged by correcting the stage: {newly}')\n"
            "print(f'net change in sa flags: {newly - recovered:+d}  '\n"
            "      f'({int(df.sa_outlier_flag.sum())} -> {int(cf.cf_sa_outlier_flag.sum())})')\n"
            "print()\n"
            "print('where they move:')\n"
            "print('  recovered:', (cf[cf.sa_outlier_flag & ~cf.cf_sa_outlier_flag]\n"
            "                        .groupby(['start_age_hpf', 'temperature']).size().to_dict()))\n"
            "print('  newly:    ', (cf[~cf.sa_outlier_flag & cf.cf_sa_outlier_flag]\n"
            "                        .groupby(['start_age_hpf', 'temperature']).size().to_dict()))"
        ),
        CODE(
            "fig, summary = Q.plot_counterfactual(df)\n"
            "save(fig, '04_counterfactual_arrhenius_stage')\n"
            "plt.show()\n"
            "summary[['total', 'as_run_%', 'corrected_%']].round(1)"
        ),
        CODE("save(Q.plot_area_against_band(df), '05_area_against_reference_band')\nplt.show()"),
        MD(
            "**Reading.** Correcting the stage recovers the cold cohort but breaks the hot cohort, for a\n"
            "net gain of ~10 snips. The reason is visible in the last figure: the X markers (median area\n"
            "at the corrected stage) do not track the reference curve. Raising temperature advances\n"
            "developmental stage without proportionally increasing projected area — total area is\n"
            "dominated by a roughly fixed yolk + body mass set at fertilization.\n"
            "\n"
            "So area is **not** a monotone proxy for stage across a temperature series, and no choice of\n"
            "stage index makes a tight two-sided area band behave. This rules out the fix I expected to\n"
            "work, and points at the width of the band rather than its indexing."
        ),
        MD(
            "## 6. The genuinely bad objects\n"
            "\n"
            "Not every failure is a false positive. The \"too large\" failures are real, and the flag is\n"
            "right to catch them."
        ),
        CODE(
            "big = df.loc[df.sa_too_large, ['experiment_id', 'well_index', 'temperature', 'area_um2',\n"
            "                               'sa_upper', 'length_um', 'width_um', 'focus_flag']].copy()\n"
            "big['x_over_cut'] = (big.area_um2 / big.sa_upper).round(2)\n"
            "big['area_k'] = (big.area_um2 / 1000).round(0)\n"
            "print(f'all {len(big)} \"too large\" failures also flagged out-of-focus: '\n"
            "      f'{bool(big.focus_flag.all())}')\n"
            "big.drop(columns=['area_um2', 'sa_upper']).reset_index(drop=True)"
        ),
        MD(
            "**Reading.** These have `length_um ≈ 4400` and `width_um ≈ 2000` — essentially the full field\n"
            "of view — at 5–7× the upper cut. They are whole-frame masks: wells where the embryo was not\n"
            "found and SAM2 segmented the background. Every one is independently flagged out of focus.\n"
            "This is the small, real segmentation-failure population, and it is exactly what the upper cut\n"
            "and the focus check exist to remove."
        ),
        MD(
            "## 7. Montages — every failed embryo, per plate\n"
            "\n"
            "The main work product. Per cell: **left** = extracted snip (what the model and QC see, at the\n"
            "fixed µm/px snip scale), **right** = raw focus-stacked frame cropped around the mask. The cyan\n"
            "contour is the segmentation mask, drawn on both so a bad mask is obvious against the raw\n"
            "image. Nested rectangles encode which QC flags fired — count the rings, read the colors off\n"
            "the legend; the outermost ring is the first flag in canonical order.\n"
            "\n"
            "Also written to `figures/montage_<plate>.png` at higher resolution for zooming."
        ),
        CODE(
            "montages = {}\n"
            "for experiment_id in Q.EXPERIMENTS:\n"
            "    fig = Q.build_failure_montage(experiment_id, df)\n"
            "    if fig is None:\n"
            "        print(f'{experiment_id}: no failures')\n"
            "        continue\n"
            "    save(fig, f'montage_{experiment_id}', dpi=150)\n"
            "    montages[experiment_id] = fig\n"
            "    n_fail = int((~df.loc[df.experiment_id == experiment_id, 'use_snip']).sum())\n"
            "    print(f'{experiment_id}: {n_fail} failed embryos rendered')"
        ),
        CODE("montages['20250612_30hpf_ctrl_atf6']"),
        CODE("montages['20250612_36hpf_wfs1_ctcf']"),
        CODE("montages['20250612_24hpf_ctrl_atf6']"),
        CODE(
            "for experiment_id in ('20250612_24hpf_wfs1_ctcf', '20250612_30hpf_wfs1_ctcf',\n"
            "                     '20250612_36hpf_ctrl_atf6'):\n"
            "    display(montages[experiment_id])"
        ),
        MD(
            "## 8. Preliminary conclusions\n"
            "\n"
            "**Segmentation is not the problem.** Across all six montages the cyan contour tracks the\n"
            "embryo boundary on the raw frame. The flagged embryos are, with few exceptions, well-segmented\n"
            "and normal-looking — consistent with your manual inspection.\n"
            "\n"
            "**The loss is one check, firing on good data.** `sa_outlier_flag` is implicated in 96 of 106\n"
            "failures. 87 of those are \"too small\", and 75 of the 87 sit within 20% of the cut (median\n"
            "~92% of it). That is a tight-threshold signature, not a broken-mask signature.\n"
            "\n"
            "**The mechanism is a reference-population mismatch, not a bug.** `k_lower = 0.9 × p5` is\n"
            "evaluated against a wildtype curve built at ~28.5 °C. Cold-reared embryos are genuinely and\n"
            "correctly smaller, so a healthy 24 °C cohort lands just under the 5th percentile of a\n"
            "reference it does not belong to. This is worst exactly where the biology is most interesting:\n"
            "65% of 30 hpf / 24 °C snips are discarded.\n"
            "\n"
            "**The obvious fix does not work.** Re-indexing the band on Arrhenius-corrected developmental\n"
            "stage recovers 55 cold-cohort snips but newly flags 45 hot-cohort ones — net 10. Temperature\n"
            "changes stage and projected area non-proportionally, so no stage index rescues a tight\n"
            "two-sided area band on a temperature series.\n"
            "\n"
            "**What does move the needle.** Relaxing `k_lower`: 0.90 → 0.80 takes the pass rate from 81.9%\n"
            "to 91.5%; → 0.75 gives 93.9%, after which the curve flattens with ~16–23 residual flags —\n"
            "essentially the 9 whole-frame blowouts plus a few genuinely tiny objects. The knee is around\n"
            "0.75, and `k_lower` was 0.7 before being raised on 2026-07-02.\n"
            "\n"
            "**Caveats.** The raised `k_lower` exists for a documented reason — separating yolk-only masks\n"
            "from real small embryos, which area alone cannot do\n"
            "(`surface_area_qc_pose_confound.md`). Relaxing it globally would re-admit whatever\n"
            "population motivated the change, which is not visible in these six plates. Two narrower\n"
            "options avoid that: make the reference temperature-aware (fit p5/p95 per rearing temperature\n"
            "rather than pooling), or make the tolerance per-experiment-class so temperature series get a\n"
            "wider lower band while standard plates keep 0.9. Both are product decisions, not something\n"
            "this notebook should settle.\n"
            "\n"
            "**Also worth noting:** `predicted_stage_hpf` silently ignores temperature for any\n"
            "single-frame experiment, because the rate multiplies an elapsed time of zero. That is correct\n"
            "arithmetic and misleading as a column name — anything downstream treating it as a\n"
            "developmental stage for a snapshot plate is really reading nominal clock time. It affects\n"
            "more than this QC check."
        ),
    ]


def main() -> None:
    notebook = nbf.v4.new_notebook(cells=cells())
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    NOTEBOOK_PATH.write_text(nbf.writes(notebook))
    print(f"wrote {NOTEBOOK_PATH}  ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
