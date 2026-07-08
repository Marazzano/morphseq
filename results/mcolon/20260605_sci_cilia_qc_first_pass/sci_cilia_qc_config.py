"""Shared paths and conventions for sequenced-only SCI cilia QC."""

from __future__ import annotations

from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]

MODELS_DIR = RUN_DIR / "models"
PREDICTIONS_DIR = RUN_DIR / "predictions"
PLOTS_DIR = RUN_DIR / "plots"

SEQUENCED_RULE = (
    "Reference models are trained on all valid reference embryos. Query outputs "
    "for this folder are restricted to embryos with sequenced > 0 unless a file "
    "is explicitly marked otherwise."
)

SNAPSHOT_GENES = ("b9d2", "cep290")
CRISPANT_GENE = "crispant"

SCI_TIMELAPSE_PLATES = {
    "20260414_sci_b9d2_48hpf_plate01": "b9d2",
    "20260415_sci_cep290_48hpf_plate01": "cep290",
}

MODEL_SPECS = {
    "b9d2_genotype": {
        "gene": "b9d2",
        "kind": "genotype",
        "label": "zygosity",
        "model_type": "global",
    },
    "cep290_genotype": {
        "gene": "cep290",
        "kind": "genotype",
        "label": "zygosity",
        "model_type": "global",
    },
    "crispant_genotype": {
        "gene": "crispant",
        "kind": "genotype",
        "label": "genotype",
        "model_type": "global",
    },
    "b9d2_homo_ce_hta": {
        "gene": "b9d2",
        "kind": "homozygous_phenotype",
        "label": "cluster_categories",
        "model_type": "global",
        "classes": ("CE", "HTA"),
    },
    "cep290_homo_low_to_high": {
        "gene": "cep290",
        "kind": "homozygous_phenotype",
        "label": "cluster_categories",
        "model_type": "per_bin",
        "classes": ("High_to_Low", "Low_to_High"),
    },
}
