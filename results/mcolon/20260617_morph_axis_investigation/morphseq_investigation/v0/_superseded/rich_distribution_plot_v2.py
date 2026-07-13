"""RETIRED (superseded) — the canonical driver is
``../b9d2_catalog_example.py`` (PATH A via ``build_1d_density_grid``). This
drove the removed ``DistributionGrouping`` / ``FacetCoordinate`` API and imports
the removed ``label_genotype``; it will NOT run on ``main``. Historical only.

b9d2 rich distribution grid — driven by the engine's DistributionGrouping API.

This is the refactor of ``rich_distribution_plot.py`` onto the new three-stage
engine pipeline (engine/plotting.py):

    DistributionGrouping                 (one distribution interpreted one way)
      -> materialize_distribution_marginals   (NUMERICAL: shared cell grids + KDEs)
      -> build_distribution_grid               (validate + renderer-neutral IR)
      -> render_distribution_grid              (appearance + output; never fits)

The caller's ONLY job is to produce a flat bag of ``DistributionGrouping``s —
each is just ``(distribution, label_group, sample_sets)``, all of which
``run_bin`` already computes. The three label views become three label groups
over the target/reference distributions:

  peak      : the 2-D unsupervised peak labeler, on target AND on reference
              (role tells them apart; reference dashed/gray downstream).
  phenotype : the provided CE/HTA column labeler on the target; WT reference as a
              single-set genotype-style group.
  genotype  : all-b9d2 vs all-WT, expressed as trivial one-SampleSet label groups
              (uniform: every curve still comes from a SampleSet in a LabelGroup).

Run:
  cd .../20260617_morph_axis_investigation
  PYTHONPATH=.:src:$PYTHONPATH conda run -n segmentation_grounded_sam \
      --no-capture-output python -m morphseq_investigation.v0.rich_distribution_plot_v2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..engine.labelers import label_genotype
from ..engine.plotting import (
    DistributionGrouping,
    FacetCoordinate,
    build_distribution_grid,
    materialize_distribution_marginals,
    plot_1d_density_grid,
)
from .b9d2_worked_example import (
    FEATURE_NAMES,
    TARGET_DESIGN_HPF,
    load_binned,
    run_bin,
)


def _constant_label_group(distribution, label: str, name: str):
    """A trivial LabelGroup: every sample -> one SampleSet named ``label``.

    This is how the 'genotype only' view stays uniform — no special
    whole-distribution code path; b9d2/WT are just one-set label groups.
    """
    return label_genotype(
        distribution,
        method="column",
        params={
            "labels": [label] * len(distribution.sample_ids),
            "column": name,
            "label_group_name": name,
        },
    )


def groupings_for_bin(res: dict[str, Any]) -> list[DistributionGrouping]:
    """Turn one run_bin result into the flat bag of DistributionGroupings."""
    target = res["target_distribution"]
    reference = res["reference_distribution"]
    out: list[DistributionGrouping] = []

    # --- peak view: the 2-D unsupervised labeler, both roles ------------------
    out.append(DistributionGrouping(target, res["target_peak_lg"], tuple(res["target_peak_sets"])))
    out.append(
        DistributionGrouping(reference, res["reference_peak_lg"], tuple(res["reference_peak_sets"]))
    )

    # --- phenotype view: provided CE/HTA on target; WT reference as one set ---
    # (run_bin names this LG "phenotype" explicitly so it does not collide with
    # the genotype row on the LABEL_GROUP facet coordinate — the name IS the row.)
    out.append(DistributionGrouping(target, res["phenotype_lg"], tuple(res["phenotype_sets"])))
    wt_pheno_lg, wt_pheno_sets = _constant_label_group(reference, "WT", "phenotype")
    out.append(DistributionGrouping(reference, wt_pheno_lg, tuple(wt_pheno_sets)))

    # --- genotype view: all-b9d2 vs all-WT, trivial one-set groups ------------
    b9d2_lg, b9d2_sets = _constant_label_group(target, "b9d2", "genotype")
    wt_lg, wt_sets = _constant_label_group(reference, "WT", "genotype")
    out.append(DistributionGrouping(target, b9d2_lg, tuple(b9d2_sets)))
    out.append(DistributionGrouping(reference, wt_lg, tuple(wt_sets)))
    return out


def main() -> None:
    binned = load_binned()
    groupings: list[DistributionGrouping] = []
    for design_hpf in TARGET_DESIGN_HPF:
        res = run_bin(binned, design_hpf)
        if res is None:
            print(f"[skip] {design_hpf} hpf — too few embryos")
            continue
        groupings.extend(groupings_for_bin(res))

    out_dir = Path(__file__).resolve().parent / "outputs"
    for feature_name in FEATURE_NAMES:
        # Stage 1 (numerical): shared cell grids + per-SampleSet marginals.
        materialized = materialize_distribution_marginals(
            groupings,
            feature_name,
            row=FacetCoordinate.LABEL_GROUP,
            col=FacetCoordinate.TIME_BIN,
        )
        # Stage 2 (IR): validate + flatten to curves.
        grid = build_distribution_grid(
            materialized,
            row=FacetCoordinate.LABEL_GROUP,
            col=FacetCoordinate.TIME_BIN,
        )
        # Stage 3 (appearance + output): overlaid panel, reference gray/dashed.
        out_path = out_dir / f"b9d2_grid_{feature_name}.png"
        plot_1d_density_grid(
            grid,
            title=f"b9d2 distribution grid — {feature_name} (rows = label group, cols = hpf)",
            output_path=out_path,
        )
        print(f"distribution grid written: {out_path}")


if __name__ == "__main__":
    main()
