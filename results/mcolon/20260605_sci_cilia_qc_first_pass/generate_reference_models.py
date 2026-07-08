"""
Fit and save reusable reference models for sequenced-only SCI cilia QC.

Reference scope: all valid reference embryos.
Query scope is not applied here; query filtering happens in generate_sequenced_predictions.py.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/generate_reference_models.py
"""

from __future__ import annotations

import sys

import pandas as pd

from sci_cilia_qc_config import MODEL_SPECS, MODELS_DIR, RUN_DIR, SEQUENCED_RULE
from qc_artifacts import ensure_dirs, save_model_bundle

PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from src.analyze.classification.label_transfer import prepare_reference  # noqa: E402
import build_reference_and_transfer as T  # noqa: E402


def _existing_query_paths(gene: str) -> tuple[list[str], list[str]]:
    cfg = T.DATASETS[gene]
    queries = [e for e in cfg["queries"] if (T.B6 / f"df03_final_output_with_latents_{e}.csv").exists()]
    paths = [str(T.B6 / f"df03_final_output_with_latents_{e}.csv") for e in queries]
    return queries, paths


def _load_reference_for_spec(model_id: str, spec: dict) -> tuple[pd.DataFrame, list[str], dict]:
    gene = spec["gene"]
    cfg = T.DATASETS[gene]
    queries, qpaths = _existing_query_paths(gene)
    ref_path = cfg["ref"]
    feat = T.resolve_feature_cols([ref_path, *map(pd.io.common.stringify_path, qpaths)])
    gene_hint = gene if gene in ("b9d2", "cep290") else None
    ref = T._load(ref_path, feat, gene_hint=gene_hint)

    if spec["kind"] == "genotype":
        if spec["label"] == "zygosity":
            ref = ref.dropna(subset=[T.ZYG_COL, T.TIME_COL]).copy()
            ref = ref[ref[T.ZYG_COL] != "unknown"].copy()
            label_col = T.ZYG_COL
        else:
            ref = ref.dropna(subset=[T.GENO_COL, T.TIME_COL]).copy()
            ref = ref[~ref[T.GENO_COL].astype(str).str.endswith("_unknown")].copy()
            label_col = T.GENO_COL
    elif spec["kind"] == "homozygous_phenotype":
        ref = ref.dropna(subset=[T.PHENO_COL, T.TIME_COL]).copy()
        if gene == "b9d2":
            ref[T.PHENO_COL] = ref[T.PHENO_COL].replace("BA_rescue", "HTA")
        if gene == "cep290":
            ref[T.PHENO_COL] = ref[T.PHENO_COL].replace("Intermediate", "Low_to_High")
        classes = set(spec["classes"])
        ref = ref[(ref[T.ZYG_COL] == "homozygous") & ref[T.PHENO_COL].isin(classes)].copy()
        label_col = T.PHENO_COL
    else:
        raise ValueError(f"Unsupported model kind for {model_id}: {spec['kind']}")

    metadata = {
        "sequenced_rule": SEQUENCED_RULE,
        "gene": gene,
        "kind": spec["kind"],
        "label_col": label_col,
        "model_type": spec["model_type"],
        "reference_path": str(ref_path),
        "query_experiments_used_for_feature_intersection": queries,
        "query_paths_used_for_feature_intersection": qpaths,
        "feature_cols": feat,
        "n_reference_rows": int(len(ref)),
        "n_reference_embryos": int(ref[T.GROUP_COL].nunique()),
        "reference_label_counts": ref.groupby(T.GROUP_COL)[label_col]
        .agg(lambda s: s.mode().iloc[0])
        .value_counts()
        .to_dict(),
    }
    return ref, feat, metadata


def fit_one(model_id: str, spec: dict) -> None:
    ref, feat, metadata = _load_reference_for_spec(model_id, spec)
    cv_group_col = "experiment_id" if "experiment_id" in ref.columns and ref["experiment_id"].nunique() >= 2 else None
    model = prepare_reference(
        ref,
        feat,
        label_col=metadata["label_col"],
        group_col=T.GROUP_COL,
        time_col=T.TIME_COL,
        cv_group_col=cv_group_col,
        model_type=spec["model_type"],
    )
    metadata["cv_group_col"] = cv_group_col
    metadata["classes"] = model["classes"]
    metadata["quality_report"] = model["quality_report"]
    metadata["bin_model_count"] = len(model.get("bin_models", {}))
    save_model_bundle(model_id, model, metadata)
    if model_id == "b9d2_homo_ce_hta":
        import b9d2_homo_ce_hta as B  # noqa: WPS433

        cv = B._target_cv_predictions(ref, feat)
        cv.to_csv(RUN_DIR / "predictions" / "b9d2_homo_ce_hta_reference_cv_target_hpf_pm2.csv", index=False)
    elif model_id == "cep290_homo_low_to_high":
        import cep290_homo_low_to_high as C  # noqa: WPS433

        cv = C._target_cv_predictions(ref, feat)
        cv.to_csv(RUN_DIR / "predictions" / "cep290_homo_low_to_high_reference_cv_target_hpf_pm2.csv", index=False)
    print(
        f"[{model_id}] saved model | ref_embryos={metadata['n_reference_embryos']} "
        f"classes={model['classes']} bin_models={metadata['bin_model_count']}"
    )


def main() -> None:
    ensure_dirs()
    for model_id, spec in MODEL_SPECS.items():
        fit_one(model_id, spec)
    print(f"\nSaved model bundles under: {MODELS_DIR.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
