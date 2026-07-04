# HANDOFF — WT-calibrated phenotype-geometry framework: build + first real results

**Status (2026-07-04):** Framework built, synthetic gate passed, real anchors rerun
against the QC reference tables from
`results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/`. The suspected
"wrong reference file" issue was checked directly and ruled out.

## What was built

Six-module decision-tree framework in this directory (see `README.md` for the map):

- `distribution_shift.py` (Stage 1), `support_geometry.py` (Stage 2),
  `density_geometry.py` (Stage 3), `component_geometry.py` (Stage 4),
  `confidence.py` (cross-cutting), `phenotype_geometry.py` (orchestrator).
- `synthetic_scenarios.py` + `validate_framework.py` — the validation gate.
- `run_real_anchors.py` — applies the full tree to cep290/b9d2 on two axes.
- `connectedness.py` — deprecated back-compat shim → `support_geometry.py`.

Everything runs in `segmentation_grounded_sam`. sklearn's native `HDBSCAN` is used
(no new dependency).

## Synthetic gate — PASSED

`validate_framework.py` (exit 0). Highlights:
- variance-only (WT σ=1 → σ=5) stays **continuous** with variance elevated — the
  key variance-vs-mode control.
- two-fate split and the rare-middle (50/5/50) read **discrete** (3 components,
  middle preserved).
- confidence rises with N (insufficient@8 → high@20/60).
- Three documented HARD cases are report-only (evenly-spaced trimodal, weak
  overlap, 2-D annulus) — the spec's "a 1-D axis can hide a mode" caveat.

## Real anchor results

Outputs: `tables/phenotype_geometry_summary.csv`,
`tables/phenotype_geometry_full_reference_summary.csv`,
`tables/bootstrap_nulls.npz` (70 nulls), `plots/phenotype_geometry_trajectory.png`.

The rerun now records `group_label_counts` per row. For b9d2, the full labeled
homozygous reference is still small and imbalanced: **8 CE vs 29 HTA**. The low
early-bin counts were therefore not caused by reading a subset file; they follow
from the actual homozygous-only CE/HTA support in the reference.

**cep290** (expected broad continuum): RECOVERED cleanly. Wildtype-like early
(14–30 hpf on hand; 14–24 hpf on embedding) → **continuous** at later stages with
elevated variance/entropy/tail metrics. Hand and embedding axes agree on no
discrete split.

**b9d2** (expected discrete CE/HTA split): still does **not** read discrete after
the reference audit. Per-bin labeled homozygous rows read wildtype-like or
continuous/connected; the full-reference pooled check is also not discrete:
hand = wildtype-like, embedding = connected continuum with elevated variance/IQR/tail.

## Reference Audit Resolution

Checked source:
- b9d2 reference: `results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv`
- cleaned QC reference used here: `results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv`

The cleaned table has the same 187 b9d2 embryos as the source. The apparent
"full reference should be much larger" expectation comes from counting CE labels
across all zygosities. Under the homozygous-only anchor rule used by the QC model,
there are only 8 CE and 29 HTA labeled embryos.

Do NOT tune the detector to force b9d2 discrete. If the expected split is still
biologically required, the next defensible checks are (a) revisit whether
heterozygous/unknown CE embryos should be included in this anchor, or (b) run Stage
2 in higher-dimensional embedding space rather than only the first two WT PCs.

## Note on transition detection

No variance→mode transition was detected for either gene (neither ever became
discrete). The change-point detector (`estimate_transition`) is wired and validated
in logic; it simply had no discrete call to trigger on. It will fire once a genuinely
fracturing group is run (or once the higher-dim b9d2 view, if #2, reveals the split).
