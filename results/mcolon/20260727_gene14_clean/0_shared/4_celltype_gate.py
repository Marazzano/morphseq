#!/usr/bin/env python
"""
4_celltype_gate.py  --  the ONE cell-type filter
=============================================================================
Answers exactly one question per (gene, timepoint, cell_type):

    is this contrast MEASURABLE?

    reliable = log_abund_ctrl >= -3  AND  nonzero_frac_ctrl >= 0.5

Both terms are about the CONTROL arm, because the control is the denominator --
if the control barely detects a cell type, the ratio is uninterpretable no matter
what the perturbed arm does. Nothing here looks at the mutant, which is why the
gate is identical across genes sharing a control (115/149/194/235 cell types at
18/24/30/48hpf for all three crispants, since they share ctrl-inj). That
invariance is the sign the gate describes the assay, not the biology.

WHY THERE IS NO `ablation_candidate` ANY MORE
    The old gate carried a second flag that conflated two OPPOSITE things:
      * a true ablation -- cell type genuinely gone in the mutant. The strongest
        result an experiment like this can produce.
      * a detection floor -- <1 cell/embryo, so zero is what noise looks like.
    The heatmap then masked both as artifacts, while the gate's own comment
    called them "the strongest possible result". Measured cost at 48hpf across
    the three crispants: 7 real ablations hidden, 4 of them significant --
    including `neural progenitor cell, midbrain` independently in foxj1a
    (q=0.006) and sspo (q=0.011).

    A real ablation now simply shows up as a large negative delta and is KEPT.
    Judging it is the reader's job, which is why every row carries
    mean_cells_ctrl / mean_cells_mut / nonzero_frac_* as INFORMATIONAL columns
    that drive no filtering.

NAMING
    The archive called the control fraction `nonzero_frac_sib`, which lies: for
    the crispants the control is ctrl-inj, not a sibling. Here it is
    `nonzero_frac_ctrl` throughout.

TWO MODES
    (default)  counts only  -- writes the counts-derived half of the gate, which
               is all that is knowable before the fits exist.
    --fits DIR joins `log_abund_x` from the per-timepoint contrast tables in DIR
               and writes the FULL gate. Run this after abundance_dact fits.

    conda run -n segmentation_grounded_sam --no-capture-output python \
        0_shared/4_celltype_gate.py [--fits ../abundance_dact/output]
=============================================================================
"""
import argparse
import glob
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COUNTS = HERE / "per_embryo_celltype_counts.tsv"
OUT = HERE / "celltype_gate.tsv"

LA_THRESH = -3.0      # control-arm log abundance floor
MIN_NONZERO = 0.5     # control-arm fraction of embryos with >=1 cell

# Which control each perturbation is contrasted against. This is the ONE place
# the control assignment is written down; abundance_dact reads it from here.
CONTROL_OF = {
    "foxj1a": "ctrl-inj",
    "ift88": "ctrl-inj",
    "sspo": "ctrl-inj",
    "cep290-mut": "cep290-negsib",
    "b9d2-mut": "b9d2-negsib",
}
# cep290 has NO negsib at 30hpf, so that one timepoint borrows b9d2-negsib.
# mcclintock dropped cep290@30hpf entirely rather than solve this, so the
# borrowed fit is the only cep290@30hpf estimate that exists anywhere.
BORROWED = {("cep290-mut", 30): "b9d2-negsib"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", default=None,
                    help="dir of per-timepoint contrast TSVs; adds log_abund_ctrl")
    a = ap.parse_args()

    counts = pd.read_csv(COUNTS, sep="\t")
    print(f"counts: {len(counts)} rows, {counts.embryo_ID.nunique()} embryos, "
          f"{counts.cell_type.nunique()} cell types")

    rows = []
    for gene, ctrl in CONTROL_OF.items():
        for tp in sorted(counts.timepoint.unique()):
            use_ctrl = BORROWED.get((gene, tp), ctrl)
            sub = counts[(counts.timepoint == tp)
                         & counts.perturbation.isin([gene, use_ctrl])]
            if sub.empty:
                continue
            n_mut = sub[sub.perturbation == gene].embryo_ID.nunique()
            n_ctl = sub[sub.perturbation == use_ctrl].embryo_ID.nunique()
            if n_mut == 0 or n_ctl == 0:
                continue

            sub = sub.assign(arm=lambda d: (d.perturbation == gene)
                             .map({True: "mut", False: "ctrl"}))
            g = (sub.groupby(["cell_type", "arm"])
                    .agg(n_emb=("embryo_ID", "nunique"),
                         n_zero=("n_cells", lambda s: int((s == 0).sum())),
                         mean_cells=("n_cells", "mean"))
                    .unstack("arm"))
            g.columns = [f"{a_}_{b}" for a_, b in g.columns]
            g = g.reset_index()
            g["nonzero_frac_ctrl"] = 1 - g.n_zero_ctrl / g.n_emb_ctrl
            g["nonzero_frac_mut"] = 1 - g.n_zero_mut / g.n_emb_mut
            g["gene"] = gene
            g["timepoint"] = tp
            g["control"] = use_ctrl
            g["xblock"] = (gene, tp) in BORROWED
            rows.append(g)

    gate = pd.concat(rows, ignore_index=True)
    gate = gate.rename(columns={"mean_cells_ctrl": "mean_cells_ctrl",
                                "mean_cells_mut": "mean_cells_mut"})

    if a.fits:
        # log_abund_x is the CONTROL arm's modelled abundance and only exists
        # once the fits have run. Without it the gate is incomplete, so say so
        # rather than silently emitting a half-gate that looks whole.
        fits = sorted(glob.glob(os.path.join(a.fits, "*contrast*tp*.tsv")))
        print(f"joining log_abund_ctrl from {len(fits)} contrast tables")
        fc = pd.concat([pd.read_csv(f, sep="\t") for f in fits], ignore_index=True)
        fc = (fc.rename(columns={"cell_group": "cell_type",
                                 "log_abund_x": "log_abund_ctrl"})
                [["cell_type", "timepoint", "log_abund_ctrl"]]
                .drop_duplicates(["cell_type", "timepoint"]))
        gate = gate.merge(fc, on=["cell_type", "timepoint"], how="left")
        gate["reliable"] = ((gate.log_abund_ctrl >= LA_THRESH)
                            & (gate.nonzero_frac_ctrl >= MIN_NONZERO))
        gate["gate_complete"] = True
    else:
        gate["log_abund_ctrl"] = pd.NA
        # counts-only: the nonzero half is decided, the abundance half is not
        gate["reliable"] = gate.nonzero_frac_ctrl >= MIN_NONZERO
        gate["gate_complete"] = False
        print("\nNOTE: no --fits given, so `reliable` reflects the nonzero-fraction "
              "term ONLY.\n      Re-run with --fits after abundance_dact to apply "
              "the log_abund_ctrl >= -3 term.")

    cols = ["gene", "timepoint", "cell_type", "control", "xblock", "reliable",
            "gate_complete", "log_abund_ctrl", "nonzero_frac_ctrl",
            "nonzero_frac_mut", "mean_cells_ctrl", "mean_cells_mut",
            "n_emb_ctrl", "n_emb_mut"]
    gate = gate[[c for c in cols if c in gate.columns]]
    gate.to_csv(OUT, sep="\t", index=False)
    print(f"\nwrote {OUT.name}: {len(gate)} rows")

    print("\n=== reliable cell types per gene x timepoint ===")
    print(gate[gate.reliable].pivot_table(index="gene", columns="timepoint",
                                          values="cell_type", aggfunc="count",
                                          fill_value=0).to_string())
    b = gate[gate.xblock]
    if len(b):
        print(f"\nborrowed-control rows (xblock=True): {len(b)} "
              f"-- {b.gene.iloc[0]} @ tp{b.timepoint.iloc[0]} "
              f"vs {b.control.iloc[0]}")


if __name__ == "__main__":
    main()
