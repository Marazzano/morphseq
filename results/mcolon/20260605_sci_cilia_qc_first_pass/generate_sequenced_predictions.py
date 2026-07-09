"""
Generate sequenced-only query predictions from saved reference models.

Reference scope: all valid reference embryos from generate_reference_models.py.
Query scope: embryos with sequenced > 0 only.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/generate_sequenced_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sci_cilia_qc_config import (
    MODEL_SPECS,
    PREDICTIONS_DIR,
    RUN_DIR,
    SCI_TIMELAPSE_PLATES,
    SEQUENCED_RULE,
)
from qc_artifacts import compatibility_copy, ensure_dirs, load_model_bundle, prediction_path

PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from src.analyze.classification.label_transfer import transfer_labels  # noqa: E402
import build_reference_and_transfer as T  # noqa: E402
import label_transfer_snapshots as S  # noqa: E402


def _query_experiments(gene: str) -> list[str]:
    return [
        e for e in T.DATASETS[gene]["queries"]
        if (T.B6 / f"df03_final_output_with_latents_{e}.csv").exists()
    ]


def _load_query(gene: str, experiments: list[str], feat: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_hint = gene if gene in ("b9d2", "cep290") else None
    qparts = []
    for exp in experiments:
        path = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
        q = T._load(path, feat, gene_hint=gene_hint)
        q["query_experiment"] = exp
        qparts.append(q)
    qry = pd.concat(qparts, ignore_index=True) if qparts else pd.DataFrame()
    seqlk = S.build_sequenced_lookup(experiments) if experiments else pd.DataFrame()
    return qry, seqlk


def _sequenced_filter(qry: pd.DataFrame, seqlk: pd.DataFrame) -> pd.DataFrame:
    if qry.empty or seqlk.empty:
        return qry.iloc[0:0].copy()
    seq_ids = set(seqlk.loc[seqlk["sequenced"] > 0, T.GROUP_COL].astype(str))
    return qry[qry[T.GROUP_COL].astype(str).isin(seq_ids)].copy()


def _stage_lookup(experiments: list[str]) -> dict[str, float]:
    out = {}
    for exp in experiments:
        path = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
        try:
            df = pd.read_csv(path, usecols=[T.GROUP_COL, "start_age_hpf"], low_memory=False)
        except ValueError:
            continue
        out.update(
            df.dropna(subset=["start_age_hpf"])
            .groupby(T.GROUP_COL)["start_age_hpf"]
            .median()
            .to_dict()
        )
    return out


def _truth_group(gene: str, row: pd.Series) -> str:
    genotype = str(row.get("true_genotype", ""))
    stratum = str(row.get("stratum", ""))
    if stratum == "AB" or genotype == "ab_wildtype":
        return "AB -> wildtype"
    if stratum == "wildtype_sibling" or genotype.endswith("_wildtype"):
        return f"{gene}_wildtype -> wildtype"
    if stratum == "heterozygous" or genotype.endswith("_heterozygous"):
        return f"{gene}_heterozygous -> heterozygous"
    if stratum == "homozygous" or genotype.endswith("_homozygous"):
        return f"{gene}_homozygous -> homozygous"
    return "unknown"


def _attach_common_metadata(
    emb: pd.DataFrame,
    qry_seq: pd.DataFrame,
    seqlk: pd.DataFrame,
    gene: str,
    experiments: list[str],
) -> pd.DataFrame:
    meta = qry_seq.drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)
    out = emb.copy()
    out["dataset"] = gene
    out["query_experiment"] = out["query_embryo_id"].map(meta["query_experiment"])
    out["true_genotype"] = out["query_embryo_id"].map(meta[T.GENO_COL])
    out["true_zygosity"] = out["query_embryo_id"].map(meta[T.ZYG_COL]) if T.ZYG_COL in meta.columns else np.nan
    out["stage"] = out["query_embryo_id"].map(_stage_lookup(experiments))
    out = S._tag(out, seqlk)
    out["truth_group"] = out.apply(lambda r: _truth_group(gene, r), axis=1)
    return out


def _run_model_on_snapshot(model_id: str, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = load_model_bundle(model_id)
    model = bundle["model"]
    meta = bundle["metadata"]
    gene = spec["gene"]
    experiments = _query_experiments(gene)
    qry, seqlk = _load_query(gene, experiments, meta["feature_cols"])
    qry_seq = _sequenced_filter(qry, seqlk)
    if qry_seq.empty:
        print(f"[{model_id}] no sequenced query embryos; skipping")
        return pd.DataFrame(), pd.DataFrame()

    result = transfer_labels(model, qry_seq, skip_flagged=False)
    emb = _attach_common_metadata(result["embryo_predictions"], qry_seq, seqlk, gene, experiments)
    img = result["image_predictions"].copy()
    img["dataset"] = gene
    img["model_id"] = model_id
    emb["model_id"] = model_id
    return emb, img


def _save_prediction_pair(model_id: str, emb: pd.DataFrame, img: pd.DataFrame) -> None:
    if emb.empty:
        return
    emb_path = prediction_path(f"{model_id}_embryo_predictions.csv")
    img_path = prediction_path(f"{model_id}_image_predictions.csv")
    emb.to_csv(emb_path, index=False)
    img.to_csv(img_path, index=False)
    print(f"[{model_id}] wrote {len(emb)} sequenced embryos -> {emb_path.relative_to(RUN_DIR)}")


def _write_registry(geno_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not geno_frames:
        return pd.DataFrame()
    geno = pd.concat(geno_frames, ignore_index=True)
    cols = [
        "query_embryo_id", "dataset", "query_experiment", "sequenced", "stratum",
        "true_genotype", "true_zygosity", "predicted_label", "stage",
    ]
    reg = geno[[c for c in cols if c in geno.columns]].copy()
    reg = reg.rename(columns={"query_embryo_id": "embryo_id", "predicted_label": "predicted_zygosity"})
    reg.to_csv(prediction_path("sequenced_registry.csv"), index=False)
    return reg


def _write_compatibility_outputs(geno_frames: list[pd.DataFrame], pheno_frames: list[pd.DataFrame]) -> None:
    legacy = RUN_DIR / "transfer_results"
    legacy.mkdir(exist_ok=True)
    if geno_frames:
        geno_path = prediction_path("genotype_transfer_predictions.csv")
        pd.concat(geno_frames, ignore_index=True).to_csv(geno_path, index=False)
        compatibility_copy(geno_path, legacy / "genotype_transfer_predictions.csv")
    if pheno_frames:
        pheno_path = prediction_path("phenotype_transfer_predictions.csv")
        pd.concat(pheno_frames, ignore_index=True).to_csv(pheno_path, index=False)
        compatibility_copy(pheno_path, legacy / "phenotype_transfer_predictions.csv")
    reg_path = prediction_path("sequenced_registry.csv")
    if reg_path.exists():
        compatibility_copy(reg_path, legacy / "sequenced_registry.csv")


def _load_sci_query(gene: str, feat: list[str]) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    experiments = [exp for exp, g in SCI_TIMELAPSE_PLATES.items() if g == gene and (T.B6 / f"df03_final_output_with_latents_{exp}.csv").exists()]
    qry, seqlk = _load_query(gene, experiments, feat)
    return experiments, qry, seqlk


def _write_time_series_homo_predictions() -> None:
    out_dir = RUN_DIR / "time_series" / "sequenced_focus" / "homozygous_focus"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in ("b9d2_homo_ce_hta", "cep290_homo_low_to_high"):
        bundle = load_model_bundle(model_id)
        model = bundle["model"]
        meta = bundle["metadata"]
        gene = meta["gene"]
        experiments, qry, seqlk = _load_sci_query(gene, meta["feature_cols"])
        qry_seq = _sequenced_filter(qry, seqlk)
        if qry_seq.empty:
            print(f"[{model_id} time_series] no sequenced sci query embryos; skipping")
            continue
        result = transfer_labels(model, qry_seq, skip_flagged=False)
        emb = _attach_common_metadata(result["embryo_predictions"], qry_seq, seqlk, gene, experiments)
        hpf = (
            qry_seq.dropna(subset=[T.TIME_COL])
            .groupby(T.GROUP_COL)[T.TIME_COL]
            .agg(["min", "max", "median"])
            .rename(columns={"min": "hpf_min", "max": "hpf_max", "median": "hpf_median"})
        )
        emb = emb.join(hpf, on="query_embryo_id", how="left")
        emb["model"] = "homozygous_focus"
        emb["model_id"] = model_id
        pred_path = prediction_path(f"time_series_{gene}_homo_predictions.csv")
        emb.to_csv(pred_path, index=False)
        compat_path = out_dir / f"{gene}_homo_predictions.csv"
        compatibility_copy(pred_path, compat_path)
        print(f"[{model_id} time_series] wrote {len(emb)} sequenced embryos -> {pred_path.relative_to(RUN_DIR)}")


def main() -> None:
    ensure_dirs()
    print(SEQUENCED_RULE)
    geno_frames: list[pd.DataFrame] = []
    pheno_frames: list[pd.DataFrame] = []

    for model_id, spec in MODEL_SPECS.items():
        emb, img = _run_model_on_snapshot(model_id, spec)
        _save_prediction_pair(model_id, emb, img)
        if emb.empty:
            continue
        if spec["kind"] == "genotype":
            emb["benchmarkable"] = emb["true_zygosity"].notna() & (emb["true_zygosity"] != "unknown")
            emb["correct"] = emb["benchmarkable"] & (emb["predicted_label"] == emb["true_zygosity"])
            geno_frames.append(emb)
        elif spec["kind"] == "homozygous_phenotype":
            homo = emb[emb["stratum"] == "homozygous"].copy()
            homo_path = prediction_path(f"{model_id}_homozygous_only_embryo_predictions.csv")
            homo.to_csv(homo_path, index=False)
            pheno_frames.append(emb)

    _write_registry(geno_frames)
    _write_compatibility_outputs(geno_frames, pheno_frames)
    _write_time_series_homo_predictions()
    print(f"\nSaved sequenced-only predictions under: {PREDICTIONS_DIR.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
