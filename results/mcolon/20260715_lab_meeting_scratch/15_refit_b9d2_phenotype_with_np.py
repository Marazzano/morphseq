"""Refit the b9d2 homozygous phenotype model WITH a Not Penetrant third class.

Mirrors script 11 (cep290). The shipped ``b9d2_homozygous_phenotype.pkl`` is 2-class (CE / HTA);
b9d2 was never scored for penetrance, so it has no NP label at all. To give b9d2 a comparable
Not-Penetrant class -- so the resolve-the-mess / emergence figures can be symmetric with cep290 --
we define NP from WILDTYPE embryos: a non-penetrant homozygote develops like wildtype, so
"looks like wildtype" is the NP appearance.

CE / HTA come from homozygotes only. Wildtype rows (phenotype_clean=='wildtype') are RELABELED to
'Not Penetrant' and added as the NP class.

HONEST CAVEAT (weaker than cep290): the resulting classes are 8 CE / 29 HTA / 40 NP -- CE is
tiny and now the minority by a wide margin, and only 2 experiments carry wildtype (~20 each), so
leave-one-experiment-out CV is thin. class_weight='balanced' compensates, but the confusion
matrix is re-checked after fit: if CE recall collapses (NP/HTA cannibalizing it) this model
should not be used. Written to a NEW filename so the 2-class model is untouched.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/15_refit_b9d2_phenotype_with_np.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.classification.label_transfer import prepare_reference_perbin  # noqa: E402

SOURCE_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
TABLE_DIR = SOURCE_DIR / "tables"
OUT_MODEL = SOURCE_DIR / "models" / "b9d2_homozygous_phenotype_with_np.pkl"
OUT_CV = SOURCE_DIR / "models" / "b9d2_homozygous_phenotype_with_np_reference_cv.csv"

GENE = "b9d2"
LABEL_COL = "phenotype_clean"
HOMO_LABELS = ["CE", "HTA"]
NP_LABEL = "Not Penetrant"
GROUP_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
CV_GROUP_COL = "experiment_id"
BIN_WIDTH = 4.0
CV_MODE = "auto"


def shared_z_mu_b_features(ref: pd.DataFrame, query: pd.DataFrame) -> list[str]:
    return sorted(
        {c for c in ref.columns if c.startswith("z_mu_b_")}
        & {c for c in query.columns if c.startswith("z_mu_b_")},
        key=lambda c: int(c.split("_")[-1]),
    )


def main() -> None:
    reference_all = pd.read_csv(TABLE_DIR / "reference_all_clean.csv", low_memory=False)
    query_all = pd.read_csv(TABLE_DIR / "query_all_rows_clean.csv", low_memory=False)

    ref = reference_all[reference_all["gene"] == GENE].copy()
    query = query_all[query_all["gene"] == GENE].copy()
    ref = ref.dropna(subset=[TIME_COL, LABEL_COL])

    # CE/HTA from homozygotes; wildtype relabeled to Not Penetrant.
    homo_rows = (ref["zygosity"] == "homozygous") & ref[LABEL_COL].isin(HOMO_LABELS)
    wildtype_rows = ref["zygosity"] == "wildtype"
    ref = ref[homo_rows | wildtype_rows].copy()
    ref.loc[wildtype_rows.loc[ref.index], LABEL_COL] = NP_LABEL

    features = shared_z_mu_b_features(ref, query)
    print(f"Refitting b9d2 homozygous phenotype with NP: {ref[GROUP_COL].nunique()} embryos, "
          f"{len(features)} features")
    print(ref.groupby(GROUP_COL)[LABEL_COL].agg(lambda s: s.mode().iloc[0])
          .value_counts().to_string())

    model = prepare_reference_perbin(
        ref, features,
        label_col=LABEL_COL, group_col=GROUP_COL, time_col=TIME_COL,
        bin_width=BIN_WIDTH, cv_mode=CV_MODE, cv_group_col=CV_GROUP_COL,
        verbose=True,
    )

    with OUT_MODEL.open("wb") as fh:
        pickle.dump(model, fh)
    cv = model["per_bin"]["embryo_per_bin_prediction"]
    cv.to_csv(OUT_CV, index=False)

    perf = model["reference_performance"]
    print(f"\nClasses: {model['classes']}")
    print(f"Scored bins: {model['embryo_support']['n_bins_scored']}, "
          f"failed: {len(model['missing_bins'])}")
    print(f"Transferability: {perf['transferability']}")

    # Guard: did NP cannibalize CE/HTA? Reference-CV confusion + per-class recall.
    if {"true_label", "predicted_label"}.issubset(cv.columns):
        ct = pd.crosstab(cv["true_label"], cv["predicted_label"])
        print("\nReference-CV confusion (bin-level, true rows x predicted cols):")
        print(ct.to_string())
        print("\nPer-class recall:")
        for c in ct.index:
            recall = ct.loc[c, c] / ct.loc[c].sum() if c in ct.columns else 0.0
            print(f"  {c}: {recall:.2f}")

    print(f"\nSaved model: {OUT_MODEL}")
    print(f"Saved reference CV: {OUT_CV}")


if __name__ == "__main__":
    main()
