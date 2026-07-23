"""Refit the CEP290 homozygous phenotype model WITH a Not Penetrant third class.

The shipped ``cep290_homozygous_phenotype.pkl`` was trained on two classes
(``High_to_Low`` / ``Low_to_High``); all Not-Penetrant homozygotes were dropped before fitting,
so the model structurally cannot emit NP. That makes an OG-vs-F1 distribution comparison
asymmetric: OG figures can show a real NP category, but transferred F1 labels never can, and any
genuinely non-penetrant F1 embryo is silently forced into HtL/LtH -- which could by itself
manufacture the "distribution flattened" effect.

A homozygous-only NP class is starved: just 24 embryos vs ~119 HtL / ~104 LtH, and the pipeline
flagged NP transferability=skip (reference-CV recall ~0.28). To give the class enough data, this
version folds the 88 *wildtype* Not-Penetrant embryos into the NP class. Biologically this reads
cleanly: a non-penetrant homozygote develops like wildtype, so "looks like wildtype" is exactly
what NP should mean. Wildtype NP is well spread (3-18 per experiment across all 7 experiments),
so LOEO cross-validation is unaffected.

HtL and LtH remain homozygous-only (a homozygote is the only thing that can show a phenotype).
Written to a NEW filename so the original two-class model is untouched. IMPORTANT: after fitting,
the confusion matrix is re-checked to confirm the enlarged NP class does not cannibalize HtL/LtH.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260715_lab_meeting_scratch/11_refit_cep290_phenotype_with_np.py
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

# Reuse the exact training tables + hyperparameters from the original fit script
# (results/mcolon/20260607_sci_cilia_gene14_imaging_qc/1_fit_reference_models.py).
SOURCE_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
TABLE_DIR = SOURCE_DIR / "tables"
OUT_MODEL = SOURCE_DIR / "models" / "cep290_homozygous_phenotype_with_np.pkl"
OUT_CV = SOURCE_DIR / "models" / "cep290_homozygous_phenotype_with_np_reference_cv.csv"

GENE = "cep290"
LABEL_COL = "phenotype_clean"
KEEP_LABELS = ["High_to_Low", "Low_to_High", "Not Penetrant"]  # <- NP added here
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

    # HtL/LtH come from homozygotes only (only a homozygote can show a phenotype).
    # The NP class is augmented with wildtype NP embryos to escape data starvation:
    # a non-penetrant homozygote looks like wildtype, so "wildtype NP" is a legitimate
    # exemplar of the NP appearance. Het/unknown NP are deliberately NOT borrowed.
    homozygous_rows = (ref["zygosity"] == "homozygous") & ref[LABEL_COL].isin(KEEP_LABELS)
    wildtype_np_rows = (ref["zygosity"] == "wildtype") & (ref[LABEL_COL] == "Not Penetrant")
    ref = ref[homozygous_rows | wildtype_np_rows].copy()

    features = shared_z_mu_b_features(ref, query)
    print(f"Refitting cep290 homozygous phenotype with NP: {ref[GROUP_COL].nunique()} embryos, "
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

    # --- Guard: did the enlarged NP class cannibalize HtL/LtH? ---
    # Reference-CV bin-level confusion + per-class recall. Compare against the
    # homozygous-only-NP baseline (HtL 0.70 / LtH 0.62 / NP 0.28).
    if {"true_label", "predicted_label"}.issubset(cv.columns):
        ct = pd.crosstab(cv["true_label"], cv["predicted_label"])
        print("\nReference-CV confusion (bin-level, true rows x predicted cols):")
        print(ct.to_string())
        print("\nPer-class recall:")
        for c in ct.index:
            recall = ct.loc[c, c] / ct.loc[c].sum() if c in ct.columns else 0.0
            print(f"  {c}: {recall:.2f}")
        print("Baseline (homozygous-only NP): HtL 0.70 / LtH 0.62 / NP 0.28")

    print(f"\nSaved model: {OUT_MODEL}")
    print(f"Saved reference CV: {OUT_CV}")


if __name__ == "__main__":
    main()
