"""Script 1 of 3 — export morphology covariates for the Hooke/PLN regressions.

Writes a flat CSV that the R script hands straight to ``new_cell_count_set(sample_metadata = ...)``.
This is the whole Python->R interface: R never computes morphology, never reads a parquet, and never
needs to know how the PCs were derived.

Hooke's contract for ``sample_metadata`` (verified against the package source,
``hooke/R/cell_count_model.R``):

- one row per distinct ``sample_group`` value in the CDS — here, per ``embryo_ID``
- a column literally named ``sample`` whose values match those ``embryo_ID``s
- ``assert_that(nrow(sample_metadata) == length(unique(colData(cds)[[sample_group]])))``

The CDS keys embryos as ``GENE7_P18_A1_Bl1`` (RT-block suffix); the crosswalk mints
``GENE7_P18_A1``. ``sample`` is emitted in the **RT-block form** so it joins the CDS directly, with
``seq_sample_id`` retained alongside for provenance.

Note the row-count assertion: if the CDS contains embryos this table lacks (non-imaged wells, or the
``pax1a``/``pax9`` perturbations that are in the sequencing run but not the imaging plates), the
assertion fails. ``--all-seq-embryos`` pads the table with the full sequencing roster, morphology
columns left empty, so the join is total. That is the flag to reach for when R complains about row
counts.

Curated image-QC exclusions are applied by default (``excluded_wells.csv``) so morphology and
sequencing analyze an identical well set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import cohort_axes as ca  # noqa: E402
import morph_pca_spline as mps  # noqa: E402
from morphseq_integration import build_master_table, load_seq_metadata  # noqa: E402
from morphseq_integration.identifiers import strip_rt_block  # noqa: E402

DEFAULT_N_COMPONENTS = 5   # shared denoised dims (first 5 of the 10-component fit)
DEFAULT_OUTPUT = HERE / "data" / "morph_covariates.csv"

# Identity + design columns carried through to R. Deliberately excludes the latent block: R needs
# covariates, not 200 columns of z_mu.
CARRY_COLUMNS: tuple[str, ...] = (
    "seq_sample_id",
    "well_id",
    "experiment_id",
    "well_index",
    "hash_plate",
    "hash_well",
    "target",
    "perturbation",
    "temperature",
    "timepoint_seq",
    "stage",
    "embryo_ID",
)


def build_covariates(
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    exclusions: str = "drop",
    all_seq_embryos: bool = False,
) -> pd.DataFrame:
    """Assemble the per-embryo covariate table.

    The PCA basis is GENE7-native and fit on ``z_mu_b_*`` (the 80 biological latents), unwhitened —
    the same basis the cohort-axis analysis uses, so PC coordinates are directly comparable between
    the morphology notebooks and the PLN regressions.
    """
    master = build_master_table(exclusions=exclusions)
    # gene7_latents_from_master renames temp->temperature / timepoint->timepoint_seq, which is the
    # naming cohort_axes keys cohorts on.
    with_morph = mps.gene7_latents_from_master(master.loc[master["has_morph"]].copy())

    # The 10-component GENE7-native basis; per-cohort PCs are then fit inside its first 5 dims.
    basis = ca.fit_global_basis(
        with_morph, n_components=ca.N_GLOBAL_COMPONENTS, whiten=False
    )
    scores, loadings = ca.cohort_local_pcs(
        basis, n_shared_dims=n_components, n_cohort_pcs=ca.N_COHORT_PCS
    )
    morph_pc_columns = [c for c in scores.columns if c.startswith("cohort_PC")]

    carried = [column for column in CARRY_COLUMNS if column in with_morph.columns]
    table = with_morph.loc[:, carried].merge(
        scores.loc[:, ["well_id", "cohort", "cohort_n_wells", *morph_pc_columns]],
        on="well_id",
        how="inner",
    )

    # Hooke keys on the CDS's embryo_ID, which carries the RT-block suffix.
    table["sample"] = table["embryo_ID"].astype(str)
    table["has_morph"] = True
    table["perturbation_group"] = [ca.base_target(value) for value in table["target"]]
    build_covariates.loadings = loadings

    if all_seq_embryos:
        table = _pad_to_full_roster(table, morph_pc_columns)

    leading = ["sample", "seq_sample_id", "well_id", "cohort", "perturbation_group", "has_morph"]
    ordered = leading + [c for c in table.columns if c not in leading]
    return table.loc[:, ordered].sort_values("sample").reset_index(drop=True)


def _pad_to_full_roster(table: pd.DataFrame, morph_pc_columns: "list[str]") -> pd.DataFrame:
    """Add rows for sequenced embryos with no morphology, so the CDS join is total.

    Morphology columns are NaN and ``has_morph`` is False for padded rows — visible, not silent. R
    must then either subset the CDS or use a formula that tolerates missing covariates.
    """
    roster = load_seq_metadata("GENE7", imaging_only=False)
    roster = roster.rename(columns={"embryo_ID": "_embryo_ID"})
    roster["sample"] = roster["_embryo_ID"].astype(str)
    missing = roster.loc[~roster["sample"].isin(set(table["sample"]))].copy()
    if missing.empty:
        return table

    padded = pd.DataFrame({"sample": missing["sample"].to_numpy()})
    padded["seq_sample_id"] = [strip_rt_block(value) for value in padded["sample"]]
    padded["has_morph"] = False
    for column in morph_pc_columns:
        padded[column] = np.nan
    for column in ("target", "temp", "timepoint", "hash_plate", "hash_well"):
        if column in missing.columns:
            padded[column] = missing[column].to_numpy()
    return pd.concat([table, padded], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument(
        "--exclusions",
        choices=("drop", "flag", "keep"),
        default="drop",
        help="curated image-QC exclusions (default: drop)",
    )
    parser.add_argument(
        "--all-seq-embryos",
        action="store_true",
        help="pad with sequenced embryos lacking morphology so the CDS row-count assertion passes",
    )
    args = parser.parse_args()

    table = build_covariates(
        n_components=args.n_components,
        exclusions=args.exclusions,
        all_seq_embryos=args.all_seq_embryos,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    loadings = getattr(build_covariates, "loadings", None)
    if loadings is not None:
        loadings_path = args.output.parent / "cohort_pc_loadings.csv"
        loadings.to_csv(loadings_path, index=False)
        print(f"wrote {len(loadings)} loading rows -> {loadings_path}")

    pc_columns = [c for c in table.columns if c.startswith("cohort_PC")]
    print(f"wrote {len(table)} rows x {table.shape[1]} cols -> {args.output}")
    print(f"  cohort PCs     : {pc_columns}")
    print(f"  with morphology: {int(table['has_morph'].sum())}")
    print(f"  unique sample  : {table['sample'].nunique()} (must equal row count: {len(table)})")
    print(f"  cohorts        : {table['cohort'].nunique() if 'cohort' in table else 0}")
    if table["sample"].duplicated().any():
        print("  !! duplicate sample values -- Hooke's sample_metadata assertion will fail")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
