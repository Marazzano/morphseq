#!/usr/bin/env python
"""Build a refreshable first-pass census of the MorphSeq imaging corpus.

The census deliberately separates:

* gross embryo-timepoints (physical embryo x time, collapsed over z/channels),
* gross embryo-plane observations (physical embryo x time x z, channels collapsed),
* observed valid snips,
* confirmed post-QC usable snips, and
* projected post-QC usable observations.

Keyence/YX1 counts come from canonical frame inventories and plate metadata. SeaHub gross
counts come from the signed-off reconciled FOV set (8 embryos/FOV); detected/materialized
counts come from the full-corpus detection manifest and bundle metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import nbformat as nbf
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
)
FRONT_MANIFEST = (
    REPO_ROOT
    / "src/data_pipeline/pipeline_orchestrator/manifests/front_half_archive.txt"
)
SEAHUB_ROOT = (
    REPO_ROOT / "results/nlammers/20260723_seahub/outputs/full_corpus_20260724"
)
SEAHUB_RECONCILED = SEAHUB_ROOT / "integration/reconciled_fovs.csv"
SEAHUB_DETECTIONS = SEAHUB_ROOT / "detection/embryo_manifest.csv"
SEAHUB_BUNDLE = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    "seahub/derived/20260724/bundle/experiments"
)

DATA_DIR = HERE / "corpus_census_data"
FIGURE_DIR = HERE / "figures/corpus_census"
NOTEBOOK_PATH = HERE / "training_corpus_census.ipynb"
WORKBOOK_PATH = DATA_DIR / "training_corpus_census.xlsx"
OVERRIDES_PATH = HERE / "sequencing_pairing_overrides.csv"

REFERENCE_TEMPERATURE_C = 28.5
ENVIRONMENTAL_TEMPERATURE_MIN_C = 27.0
ENVIRONMENTAL_TEMPERATURE_MAX_C = 30.0

NULL_STRINGS = {
    "",
    "nan",
    "none",
    "na",
    "n/a",
    "not recorded",
    "#no match",
    "unknown",
}
CONTROL_STRINGS = {
    "ab",
    "ctrl",
    "control",
    "wt",
    "wik",
    "wik-ab",
    "wildtype",
    "wild type",
    "uninjected",
    "un-injected",
    "ctrl-inj",
    "inj-ctrl",
    "crispr-inj-ctrl",
    "control-inj",
    "control injected",
    "dmso",
    "vehicle",
}
GENERIC_TREATMENT_STRINGS = {"treatment", "treated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-execute-notebook",
        action="store_true",
        help="Write census tables and notebook source without executing the notebook.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in NULL_STRINGS else text


def normalized_token(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[\s_]+", " ", text)
    return text.strip()


def is_control(value: Any) -> bool:
    token = normalized_token(value)
    if not token:
        return True
    if token in CONTROL_STRINGS:
        return True
    words = re.sub(r"[^a-z0-9]+", " ", token).split()
    if (
        any(word in {"ctrl", "control"} for word in words)
        and set(words).issubset(
            {"ab", "wik", "ctrl", "control", "inj", "injected", "crispr"}
        )
    ):
        return True
    return bool(
        re.fullmatch(
            r"(?:ctrl|control)(?:\s*(?:inj|injected))?"
            r"(?:\s*(?:at)?\s*\d+(?:\.\d+)?\s*(?:hpf|c))?",
            token,
        )
    )


def context_only_label(value: Any, heads: set[str]) -> bool:
    """True for labels such as control_at_24hpf_28C or treatment_34C."""
    words = re.sub(r"[^a-z0-9.]+", " ", normalized_token(value)).split()
    if not words or words[0] not in heads:
        return False
    return all(
        word in {"at", "wash"}
        or bool(re.fullmatch(r"\d+(?:\.\d+)?(?:hpf|c)", word))
        for word in words[1:]
    )


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def temperatures_from_text(*values: Any) -> list[float]:
    result = []
    for value in values:
        text = clean_text(value)
        for match in re.finditer(r"(?<!\d)(1[5-9]|2\d|3\d)(?:\.\d+)?\s*°?\s*[cC]\b", text):
            result.append(float(match.group(0).lower().replace("°", "").replace("c", "")))
    return result


def is_environmental_temperature(temperature_c: float) -> bool:
    """Return whether a temperature is outside the agreed non-perturbative range."""
    return (
        temperature_c < ENVIRONMENTAL_TEMPERATURE_MIN_C
        or temperature_c > ENVIRONMENTAL_TEMPERATURE_MAX_C
    )


def split_genetic_targets(value: Any, *, explicit_domain: bool = False) -> list[str]:
    text = clean_text(value)
    if not text or is_control(text):
        return []
    token = normalized_token(text)
    if token in {"uncertain", "unknown"}:
        return []
    heat_shock_construct = re.match(
        r"^hs(wnt|fgf)[_ -]+gfp\((pos|neg)\)", text, re.I
    )
    if explicit_domain and heat_shock_construct:
        pathway, status = heat_shock_construct.groups()
        return [f"heat-shock {pathway.upper()} activation"] if status.lower() == "pos" else []
    if explicit_domain and re.fullmatch(r"ab \d+(?:\.\d+)?hpf hs", token):
        return []
    if not explicit_domain:
        if re.search(r"(?:^|[_/-])(?:wildtype|wild type|wt)(?:$|[_/-])", text, re.I):
            return []
        text = re.sub(
            r"(?:[_/-](?:heterozygous|heterozygote|het|homozygous|homozyous|homo|"
            r"unknown|unkown|uncertain|crispant|crsipant|cispant|sg))+$",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"^crispr[-_]", "", text, flags=re.I)
    targets = []
    last_prefix = ""
    for item in re.split(r"[;,|+/_-]", text):
        token = item.strip()
        if not token or is_control(token):
            continue
        if normalized_token(token) in {
            "crispant",
            "crsipant",
            "cispant",
            "crispr",
            "heterozygous",
            "heterozygote",
            "het",
            "homozygous",
            "homozyous",
            "homo",
            "unknown",
            "unkown",
            "uncertain",
            "sg",
        }:
            continue
        if re.fullmatch(r"\d+[a-z]?", token, re.I) and last_prefix:
            token = last_prefix + token
        prefix_match = re.fullmatch(r"([a-z]+)\d+[a-z]?", token, re.I)
        if prefix_match:
            last_prefix = prefix_match.group(1)
        targets.append(token)
    return sorted(set(targets), key=str.lower)


def canonical_chemical_label(value: Any) -> str:
    text = clean_text(value)
    token = normalized_token(text)
    aliases = {
        "ruxolitibin": "ruxolitinib",
        "tgfb-i": "TGFβ inhibitor",
        "wnt-i": "Wnt inhibitor",
        "notch-i": "Notch inhibitor",
    }
    if token in aliases:
        return aliases[token]
    if re.fullmatch(r"shh i \d+(?:\.\d+)?", token):
        return "SHH inhibitor"
    if re.fullmatch(r"tri .+", token):
        return "tricaine"
    text = re.sub(
        r"(?:[_\s]+(?:at[_\s]+)?\d+(?:\.\d+)?hpf)"
        r"(?:[_\s]+wash)?(?:[_\s]+\d+(?:\.\d+)?c)?$",
        "",
        text,
        flags=re.I,
    )
    return text


def project_name(experiment_id: str) -> str:
    low = experiment_id.lower()
    rules = [
        ("hotchem", "Chemical perturbation"),
        ("chem", "Chemical perturbation"),
        ("cep290", "CEP290"),
        ("b9d2", "B9D2"),
        ("cilia", "Cilia crispant"),
        ("irx", "IRX"),
        ("zfpm", "ZFPM"),
        ("otx", "OTX"),
        ("pbx", "PBX"),
        ("wt_ref", "Wild-type reference"),
        ("atf6", "ER-stress regulators"),
        ("wfs1", "ER-stress regulators"),
        ("ctcf", "ER-stress regulators"),
    ]
    for token, label in rules:
        if token in low:
            return label
    return "Legacy / unspecified"


def classify_perturbation(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    getter = row.get
    domain = normalized_token(getter("perturbation_domain", ""))
    explicit = clean_text(getter("perturbation", ""))
    genotype = clean_text(getter("genotype", ""))
    chemical = clean_text(getter("chem_perturbation", ""))
    dose = clean_text(getter("dose", getter("metadata_dose", "")))
    plate_temperature = numeric(getter("temperature", None))
    text_temperatures = temperatures_from_text(
        explicit,
        getter("perturbation_key", ""),
        genotype,
        chemical,
    )
    effective_temperatures = sorted(
        set(text_temperatures or ([plate_temperature] if plate_temperature is not None else []))
    )
    heat_shock = bool(
        re.search(r"(?:^|_)hs(?:$|_)", explicit, re.I)
        or re.match(r"^hs(?:wnt|fgf)", explicit, re.I)
    )

    environmental = any(
        is_environmental_temperature(temp) for temp in effective_temperatures
    ) or heat_shock

    genetic_source = explicit if domain == "genetic" and explicit else genotype
    chemical_source = explicit if domain == "chemical" and explicit else chemical

    genetic_targets = split_genetic_targets(
        genetic_source, explicit_domain=domain == "genetic" and bool(explicit)
    )
    genetic = bool(genetic_targets)

    chemical_token = normalized_token(chemical_source)
    generic_treatment = context_only_label(
        chemical_source, {"treatment", "treated"}
    )
    generic_chemical_unknown = (
        domain == "chemical"
        and generic_treatment
        and (not environmental or "wash" in chemical_token)
    )
    chemical_active = (
        bool(chemical_token)
        and not is_control(chemical_source)
        and not context_only_label(
            chemical_source, {"control", "ctrl", "dmso", "vehicle", "treatment", "treated"}
        )
        and chemical_token != "tri 0"
    )
    if chemical_token in GENERIC_TREATMENT_STRINGS:
        chemical_active = False
    chemical_active = chemical_active or generic_chemical_unknown
    chemical_active = chemical_active or (
        domain == "chemical"
        and bool(explicit)
        and not is_control(explicit)
        and normalized_token(explicit) not in GENERIC_TREATMENT_STRINGS
        and not context_only_label(
            explicit, {"control", "ctrl", "dmso", "vehicle", "treatment", "treated"}
        )
        and normalized_token(explicit) != "tri 0"
    )

    domains = []
    if environmental:
        domains.append("environmental")
    if chemical_active:
        domains.append("chemical")
    if genetic:
        domains.append("genetic")
    primary = "+".join(domains) if domains else "control"

    pieces = []
    if genetic:
        pieces.append("G:" + ";".join(genetic_targets))
    if chemical_active:
        chem_label = chemical_source or explicit
        pieces.append("C:" + chem_label)
        if dose:
            pieces[-1] += f"@{dose}"
    if environmental:
        env_temps = [
            temp
            for temp in effective_temperatures
            if is_environmental_temperature(temp)
        ]
        env_labels = [f"{temp:g}C" for temp in env_temps]
        if heat_shock:
            env_labels.append("heat shock")
        pieces.append("T:" + "/".join(env_labels))
    condition_signature = " | ".join(pieces) if pieces else "control"

    review_reasons = []
    if domain not in {"", "chemical", "genetic", "environmental"}:
        review_reasons.append(f"unknown explicit domain={domain}")
    if "treatment" in normalized_token(explicit) and not pieces:
        review_reasons.append("generic treatment label")
    if generic_chemical_unknown:
        review_reasons.append("chemical treatment has no agent identity")
    if normalized_token(genotype) in {"uncertain", "unknown"}:
        review_reasons.append("uncertain genotype label")
    if re.search(r"(?:unknown|unkown|uncertain)", normalized_token(genotype)):
        review_reasons.append("unknown genetic status")
    if heat_shock:
        review_reasons.append("heat-shock construct classification should be curated")
    if not pieces and not any(
        is_control(value) for value in (explicit, genotype, chemical) if clean_text(value)
    ):
        if any(clean_text(value) for value in (explicit, genotype, chemical)):
            review_reasons.append("non-empty labels classified as control")

    atomic = []
    atomic.extend(("genetic", target) for target in genetic_targets)
    if chemical_active:
        atomic.append(
            (
                "chemical",
                (
                    "unspecified chemical treatment"
                    if generic_chemical_unknown
                    else canonical_chemical_label(chemical_source or explicit)
                ),
            )
        )
    if environmental:
        atomic.extend(
            ("environmental", f"{temp:g}C")
            for temp in effective_temperatures
            if is_environmental_temperature(temp)
        )
        if heat_shock:
            atomic.append(("environmental", "heat shock"))

    return {
        "has_environmental_perturbation": environmental,
        "has_chemical_perturbation": chemical_active,
        "has_genetic_perturbation": genetic,
        "perturbation_class": primary,
        "condition_signature": condition_signature,
        "atomic_perturbations": atomic,
        "classification_needs_review": bool(review_reasons),
        "classification_review_reason": "; ".join(review_reasons),
        "effective_temperature_c": (
            ";".join(f"{temp:g}" for temp in effective_temperatures)
            if effective_temperatures
            else ""
        ),
    }


def manifest_datasets() -> list[tuple[str, str]]:
    rows = []
    for raw in FRONT_MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        experiment_id, scope, *_ = line.split()
        if scope in {"Keyence", "YX1"}:
            rows.append((experiment_id, scope))
    return rows


def read_frame_inventory_stats(path: Path) -> dict[str, Any]:
    by_channel: dict[str, dict[str, set[Any]]] = defaultdict(
        lambda: {"times": set(), "zslots": set()}
    )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            channel = clean_text(row.get("channel_id")) or "unknown"
            time = clean_text(row.get("time_index"))
            if not time:
                continue
            try:
                time_key: Any = int(float(time))
            except ValueError:
                time_key = time
            by_channel[channel]["times"].add(time_key)
            product_type = normalized_token(row.get("image_product_type"))
            z_value = clean_text(row.get("z_index"))
            if product_type == "z stack" and z_value:
                try:
                    z_key: Any = int(float(z_value))
                except ValueError:
                    z_key = z_value
                by_channel[channel]["zslots"].add((time_key, z_key))

    if not by_channel:
        return {"n_timepoints": 0, "n_time_z_planes": 0, "primary_channel": ""}
    primary = next(
        (channel for channel in by_channel if channel.lower() == "bf"),
        sorted(by_channel)[0],
    )
    times = by_channel[primary]["times"]
    zslots = by_channel[primary]["zslots"]
    return {
        "n_timepoints": len(times),
        "n_time_z_planes": len(zslots) if zslots else len(times),
        "primary_channel": primary,
    }


def load_plate(experiment_id: str) -> pd.DataFrame:
    path = (
        PIPELINE_ROOT
        / "acquisition"
        / experiment_id
        / "ingest_metadata"
        / "plate_metadata.csv"
    )
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "well_id" not in frame and "well_index" in frame:
        frame["well_id"] = experiment_id + "_" + frame["well_index"].astype(str)
    return frame


def load_observed_snips(experiment_id: str) -> pd.DataFrame:
    path = (
        PIPELINE_ROOT
        / "object_extraction"
        / experiment_id
        / "snips"
        / f"{experiment_id}_snip_inventory.csv"
    )
    if not path.is_file():
        return pd.DataFrame()
    columns = ["well_id", "physical_embryo_id", "time_index", "channel_id", "is_valid_snip"]
    try:
        frame = pd.read_csv(path, usecols=lambda column: column in columns)
    except Exception:
        return pd.DataFrame()
    if "channel_id" in frame and frame["channel_id"].astype(str).str.lower().eq("bf").any():
        frame = frame.loc[frame["channel_id"].astype(str).str.lower().eq("bf")]
    frame["is_valid_snip"] = frame.get("is_valid_snip", True).fillna(False).astype(bool)
    return frame


def load_qc(experiment_id: str) -> pd.DataFrame:
    path = (
        PIPELINE_ROOT
        / "quality_control"
        / experiment_id
        / "snip_qc"
        / f"{experiment_id}_snip_qc.parquet"
    )
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path, columns=["well_id", "snip_id", "use_snip"])
    except Exception:
        return pd.DataFrame()
    if frame["snip_id"].astype(str).str.contains("_BF_").any():
        frame = frame.loc[frame["snip_id"].astype(str).str.contains("_BF_")]
    frame["use_snip"] = frame["use_snip"].fillna(False).astype(bool)
    return frame


def well_group_counts(frame: pd.DataFrame, kind: str) -> dict[str, tuple[int, int]]:
    if frame.empty or "well_id" not in frame:
        return {}
    result = {}
    for well_id, group in frame.groupby("well_id", dropna=False):
        if kind == "snip":
            if {"physical_embryo_id", "time_index"}.issubset(group.columns):
                group = group.drop_duplicates(["physical_embryo_id", "time_index"])
            total = len(group)
            usable = int(group["is_valid_snip"].sum())
        else:
            total = len(group)
            usable = int(group["use_snip"].sum())
        result[str(well_id)] = (int(total), int(usable))
    return result


def observed_embryo_stats(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Summarize embryo detections without interpreting invalid snips as absent embryos."""
    if frame.empty or "well_id" not in frame:
        return {}
    result = {}
    for well_id, group in frame.groupby("well_id", dropna=False):
        if {"physical_embryo_id", "time_index"}.issubset(group.columns):
            group = group.drop_duplicates(["physical_embryo_id", "time_index"])
            concurrent = group.groupby("time_index")["physical_embryo_id"].nunique()
            observed_unique = group["physical_embryo_id"].nunique()
        else:
            concurrent = pd.Series([1] * len(group))
            observed_unique = len(group)
        result[str(well_id)] = {
            "observed_snip_timepoints": int(len(group)),
            "observed_valid_snip_timepoints": int(group["is_valid_snip"].sum()),
            "observed_unique_physical_embryos": int(observed_unique),
            "peak_concurrent_embryos": int(concurrent.max()) if len(concurrent) else 0,
        }
    return result


def traditional_well_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = manifest_datasets()
    for dataset_index, (experiment_id, scope) in enumerate(datasets, start=1):
        plate = load_plate(experiment_id)
        plate_lookup = (
            {
                str(row["well_id"]): row
                for _, row in plate.iterrows()
                if clean_text(row.get("well_id"))
            }
            if not plate.empty
            else {}
        )
        frame_root = (
            PIPELINE_ROOT
            / "acquisition"
            / experiment_id
            / "frame_inventory"
            / "per_well"
        )
        frame_files = sorted(frame_root.glob("*/*_frame_inventory.csv"))
        frame_stats: dict[str, dict[str, Any]] = {}
        for path in frame_files:
            well_id = path.parent.name
            try:
                frame_stats[well_id] = read_frame_inventory_stats(path)
            except Exception as exc:
                frame_stats[well_id] = {
                    "n_timepoints": 0,
                    "n_time_z_planes": 0,
                    "primary_channel": "",
                    "frame_inventory_error": f"{type(exc).__name__}: {exc}",
                }

        positive_times = [
            item["n_timepoints"] for item in frame_stats.values() if item["n_timepoints"] > 0
        ]
        positive_planes = [
            item["n_time_z_planes"]
            for item in frame_stats.values()
            if item["n_time_z_planes"] > 0
        ]
        median_times = int(round(float(np.median(positive_times)))) if positive_times else 0
        median_planes = int(round(float(np.median(positive_planes)))) if positive_planes else 0

        snip_stats = observed_embryo_stats(load_observed_snips(experiment_id))
        qc_counts = well_group_counts(load_qc(experiment_id), "qc")
        wells = sorted(set(plate_lookup) | set(frame_stats) | set(snip_stats) | set(qc_counts))

        if not wells:
            classification = classify_perturbation({})
            rows.append(
                {
                    "scope": scope,
                    "corpus_dataset_id": experiment_id,
                    "pipeline_experiment_id": experiment_id,
                    "source_experiment_id": experiment_id,
                    "project_name": project_name(experiment_id),
                    "well_id": f"{experiment_id}__NO_ROW_LEVEL_PRODUCTS",
                    "well_index": "",
                    "source_fov_id": "",
                    "source_embryo_id": "",
                    "gross_embryos": 0,
                    "observed_unique_physical_embryos": 0,
                    "peak_concurrent_embryos": 0,
                    "n_timepoints": 0,
                    "n_time_z_planes": 0,
                    "gross_embryo_timepoints": 0,
                    "gross_embryo_z_observations": 0,
                    "observed_snip_timepoints": 0,
                    "observed_valid_snip_timepoints": 0,
                    "pipeline_snip_record_count": 0,
                    "qc_evaluated_timepoints": 0,
                    "confirmed_usable_timepoints": 0,
                    "pipeline_qc_record_count": 0,
                    "count_basis": "unavailable_no_row_level_products",
                    "embryo_count_basis": "unavailable",
                    "frame_inventory_present": False,
                    "plate_metadata_present": False,
                    "detected_or_registered": False,
                    "pipeline_materialized": False,
                    "genotype": "",
                    "chem_perturbation": "",
                    "dose": "",
                    "temperature": np.nan,
                    "stage_hpf": np.nan,
                    "perturbation": "",
                    "perturbation_key": "",
                    "perturbation_domain": "",
                    "has_seq_link": np.nan,
                    "sequencing_pairing_status": "unknown",
                    "collection_name": "",
                    **classification,
                }
            )

        for well_id in wells:
            metadata = plate_lookup.get(well_id, {})
            stats = frame_stats.get(well_id)
            if stats and stats["n_timepoints"] > 0:
                n_times = stats["n_timepoints"]
                n_planes = stats["n_time_z_planes"]
                count_basis = "exact_frame_inventory"
            elif median_times > 0:
                n_times = median_times
                n_planes = median_planes or median_times
                count_basis = "estimated_from_dataset_median"
            else:
                n_times = 0
                n_planes = 0
                count_basis = "unavailable"
            metadata_embryos = numeric(metadata.get("embryos_per_well", 1)) or 1
            observed = snip_stats.get(well_id, {})
            observed_total = observed.get("observed_snip_timepoints", 0)
            observed_valid = observed.get("observed_valid_snip_timepoints", 0)
            observed_unique = observed.get("observed_unique_physical_embryos", 0)
            peak_concurrent = observed.get("peak_concurrent_embryos", 0)
            embryos = max(metadata_embryos, peak_concurrent)
            embryo_count_basis = (
                "max(metadata_embryos_per_well, observed_peak_concurrent_embryos)"
                if peak_concurrent
                else "metadata_embryos_per_well"
            )
            qc_total, qc_usable = qc_counts.get(well_id, (0, 0))
            gross_timepoints = max(embryos * n_times, observed_total, qc_total)
            z_per_time = n_planes / n_times if n_times else 1.0
            gross_z_observations = max(
                embryos * n_planes,
                gross_timepoints * max(1.0, z_per_time),
            )
            if gross_timepoints > embryos * n_times:
                count_basis += "+bounded_below_by_observed_pipeline_products"
            metadata_row = dict(metadata) if hasattr(metadata, "items") else {}
            classification = classify_perturbation(metadata_row)
            rows.append(
                {
                    "scope": scope,
                    "corpus_dataset_id": experiment_id,
                    "pipeline_experiment_id": experiment_id,
                    "source_experiment_id": experiment_id,
                    "project_name": project_name(experiment_id),
                    "well_id": well_id,
                    "well_index": clean_text(metadata_row.get("well_index")),
                    "source_fov_id": "",
                    "source_embryo_id": "",
                    "gross_embryos": embryos,
                    "observed_unique_physical_embryos": observed_unique,
                    "peak_concurrent_embryos": peak_concurrent,
                    "n_timepoints": n_times,
                    "n_time_z_planes": n_planes,
                    "gross_embryo_timepoints": gross_timepoints,
                    "gross_embryo_z_observations": gross_z_observations,
                    "observed_snip_timepoints": observed_total,
                    "observed_valid_snip_timepoints": observed_valid,
                    "pipeline_snip_record_count": observed_total,
                    "qc_evaluated_timepoints": qc_total,
                    "confirmed_usable_timepoints": qc_usable,
                    "pipeline_qc_record_count": qc_total,
                    "count_basis": count_basis,
                    "embryo_count_basis": embryo_count_basis,
                    "frame_inventory_present": well_id in frame_stats,
                    "plate_metadata_present": well_id in plate_lookup,
                    "detected_or_registered": observed_total > 0,
                    "pipeline_materialized": well_id in frame_stats,
                    "genotype": clean_text(metadata_row.get("genotype")),
                    "chem_perturbation": clean_text(metadata_row.get("chem_perturbation")),
                    "dose": clean_text(metadata_row.get("dose")),
                    "temperature": numeric(metadata_row.get("temperature")),
                    "stage_hpf": numeric(
                        metadata_row.get("stage_hpf", metadata_row.get("start_age_hpf"))
                    ),
                    "perturbation": clean_text(metadata_row.get("perturbation")),
                    "perturbation_key": clean_text(metadata_row.get("perturbation_key")),
                    "perturbation_domain": clean_text(metadata_row.get("perturbation_domain")),
                    "has_seq_link": np.nan,
                    "sequencing_pairing_status": "unknown",
                    "collection_name": "",
                    **classification,
                }
            )
        if dataset_index % 25 == 0:
            print(f"traditional datasets scanned: {dataset_index}/{len(datasets)}")
    return rows


def seahub_well_rows() -> list[dict[str, Any]]:
    bundle_frames = []
    for plate_path in sorted(SEAHUB_BUNDLE.glob("20260724_seahub_*/plate_metadata.csv")):
        frame = pd.read_csv(plate_path)
        frame["bundle_plate_path"] = str(plate_path)
        bundle_frames.append(frame)
    bundle = pd.concat(bundle_frames, ignore_index=True) if bundle_frames else pd.DataFrame()

    observed_snips: dict[str, tuple[int, int]] = {}
    qc_counts: dict[str, tuple[int, int]] = {}
    for pipeline_experiment in sorted(bundle.get("experiment_id", pd.Series(dtype=str)).unique()):
        observed_snips.update(
            well_group_counts(load_observed_snips(str(pipeline_experiment)), "snip")
        )
        qc_counts.update(well_group_counts(load_qc(str(pipeline_experiment)), "qc"))

    rows = []
    for _, metadata in bundle.iterrows():
        pipeline_experiment = str(metadata["experiment_id"])
        well_id = str(metadata["well_id"])
        frame_path = (
            PIPELINE_ROOT
            / "acquisition"
            / pipeline_experiment
            / "frame_inventory"
            / "per_well"
            / well_id
            / f"{well_id}_frame_inventory.csv"
        )
        raw_observed_total, raw_observed_valid = observed_snips.get(well_id, (0, 0))
        raw_qc_total, raw_qc_usable = qc_counts.get(well_id, (0, 0))
        # A SeaHub bundle well is already an embryo crop. Downstream component/track
        # multiplicity must not manufacture additional physical embryos.
        observed_total = int(raw_observed_total > 0)
        observed_valid = int(raw_observed_valid > 0)
        qc_total = int(raw_qc_total > 0)
        qc_usable = int(raw_qc_usable > 0)
        classification = classify_perturbation(metadata)
        source_experiment = clean_text(metadata.get("source_experiment_id"))
        rows.append(
            {
                "scope": "SeaHub",
                "corpus_dataset_id": source_experiment,
                "pipeline_experiment_id": pipeline_experiment,
                "source_experiment_id": source_experiment,
                "project_name": (
                    "SeaHub chemical"
                    if normalized_token(metadata.get("perturbation_domain")) == "chemical"
                    else "SeaHub genetic"
                ),
                "well_id": well_id,
                "well_index": clean_text(metadata.get("well_index")),
                "source_fov_id": clean_text(metadata.get("source_fov_id")),
                "source_embryo_id": clean_text(metadata.get("source_embryo_id")),
                "gross_embryos": 1,
                "observed_unique_physical_embryos": 1,
                "peak_concurrent_embryos": 1,
                "n_timepoints": 1,
                "n_time_z_planes": 1,
                "gross_embryo_timepoints": 1,
                "gross_embryo_z_observations": 1,
                "observed_snip_timepoints": observed_total,
                "observed_valid_snip_timepoints": observed_valid,
                "pipeline_snip_record_count": raw_observed_total,
                "qc_evaluated_timepoints": qc_total,
                "confirmed_usable_timepoints": qc_usable,
                "pipeline_qc_record_count": raw_qc_total,
                "count_basis": "exact_detection_manifest",
                "embryo_count_basis": "one embryo per detected bundle well",
                "frame_inventory_present": frame_path.is_file(),
                "plate_metadata_present": True,
                "detected_or_registered": True,
                "pipeline_materialized": frame_path.is_file(),
                "genotype": clean_text(metadata.get("genotype")),
                "chem_perturbation": clean_text(metadata.get("chem_perturbation")),
                "dose": clean_text(metadata.get("dose")),
                "temperature": numeric(metadata.get("temperature")),
                "stage_hpf": numeric(metadata.get("stage_hpf")),
                "perturbation": clean_text(metadata.get("perturbation")),
                "perturbation_key": clean_text(metadata.get("perturbation_key")),
                "perturbation_domain": clean_text(metadata.get("perturbation_domain")),
                "has_seq_link": bool(metadata.get("has_seq_link", False)),
                "sequencing_pairing_status": (
                    "linked" if bool(metadata.get("has_seq_link", False)) else "not_linked"
                ),
                "collection_name": clean_text(metadata.get("collection_name")),
                **classification,
            }
        )

    # Included FOVs absent from the detection manifest: retain the gross denominator as
    # eight estimated embryo positions per FOV and explicitly mark them undetected.
    reconciled = pd.read_csv(SEAHUB_RECONCILED)
    detections = pd.read_csv(SEAHUB_DETECTIONS)
    included = reconciled.loc[reconciled["include_for_seahub"].astype(bool)].copy()
    detected_ids = set(detections["image_id"].astype(str))
    missing = included.loc[~included["image_id"].astype(str).isin(detected_ids)]
    for _, fov in missing.iterrows():
        source_experiment = clean_text(fov.get("experiment_id"))
        explicit = clean_text(fov.get("metadata_perturbation"))
        if not explicit:
            explicit = clean_text(fov.get("perturbation_parsed"))
        domain = clean_text(fov.get("perturbation_domain"))
        metadata = {
            "perturbation": explicit,
            "perturbation_key": clean_text(fov.get("perturbation_key")),
            "perturbation_domain": domain,
            "genotype": explicit if normalized_token(domain) == "genetic" else "ctrl",
            "chem_perturbation": (
                explicit if normalized_token(domain) == "chemical" else "ctrl"
            ),
            "temperature": (
                temperatures_from_text(explicit)[0]
                if temperatures_from_text(explicit)
                else REFERENCE_TEMPERATURE_C
            ),
        }
        classification = classify_perturbation(metadata)
        has_seq = bool(clean_text(fov.get("metadata_collection_name")))
        for position in range(1, 9):
            source_embryo = f"seahub_{fov['image_id']}_p{position:02d}"
            rows.append(
                {
                    "scope": "SeaHub",
                    "corpus_dataset_id": source_experiment,
                    "pipeline_experiment_id": "",
                    "source_experiment_id": source_experiment,
                    "project_name": (
                        "SeaHub chemical"
                        if normalized_token(domain) == "chemical"
                        else "SeaHub genetic"
                    ),
                    "well_id": source_embryo,
                    "well_index": "",
                    "source_fov_id": str(fov["image_id"]),
                    "source_embryo_id": source_embryo,
                    "gross_embryos": 1,
                    "observed_unique_physical_embryos": 0,
                    "peak_concurrent_embryos": 0,
                    "n_timepoints": 1,
                    "n_time_z_planes": 1,
                    "gross_embryo_timepoints": 1,
                    "gross_embryo_z_observations": 1,
                    "observed_snip_timepoints": 0,
                    "observed_valid_snip_timepoints": 0,
                    "pipeline_snip_record_count": 0,
                    "qc_evaluated_timepoints": 0,
                    "confirmed_usable_timepoints": 0,
                    "pipeline_qc_record_count": 0,
                    "count_basis": "estimated_8_embryos_in_included_undetected_fov",
                    "embryo_count_basis": "SeaHub design: eight embryo positions per FOV",
                    "frame_inventory_present": False,
                    "plate_metadata_present": False,
                    "detected_or_registered": False,
                    "pipeline_materialized": False,
                    "genotype": metadata["genotype"],
                    "chem_perturbation": metadata["chem_perturbation"],
                    "dose": clean_text(fov.get("metadata_dose")),
                    "temperature": metadata["temperature"],
                    "stage_hpf": numeric(
                        fov.get("metadata_stage_hpf", fov.get("stage_hpf"))
                    ),
                    "perturbation": explicit,
                    "perturbation_key": metadata["perturbation_key"],
                    "perturbation_domain": domain,
                    "has_seq_link": has_seq,
                    "sequencing_pairing_status": "linked" if has_seq else "not_linked",
                    "collection_name": clean_text(fov.get("metadata_collection_name")),
                    **classification,
                }
            )
    return rows


def wilson_interval(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def retention_summary(wells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in wells.groupby("scope"):
        total = float(group["qc_evaluated_timepoints"].sum())
        usable = float(group["confirmed_usable_timepoints"].sum())
        if total > 0:
            rate = usable / total
            low, high = wilson_interval(usable, total)
            basis = "completed snip_qc tables"
        elif scope == "SeaHub":
            reconciled = pd.read_csv(SEAHUB_RECONCILED)
            detections = pd.read_csv(SEAHUB_DETECTIONS)
            gross = int(reconciled["include_for_seahub"].astype(bool).sum()) * 8
            usable = len(detections)
            total = gross
            rate = usable / total
            low, high = wilson_interval(usable, total)
            basis = "detection completion only; downstream QC unavailable"
        else:
            rate = math.nan
            low = math.nan
            high = math.nan
            basis = "no completed QC"
        rows.append(
            {
                "scope": scope,
                "qc_evaluated_timepoints": int(total),
                "confirmed_usable_timepoints": int(usable),
                "retention_rate": rate,
                "retention_ci95_low": low,
                "retention_ci95_high": high,
                "projection_basis": basis,
            }
        )
    return pd.DataFrame(rows)


def apply_projections(wells: pd.DataFrame, retention: pd.DataFrame) -> pd.DataFrame:
    result = wells.copy()
    rates = retention.set_index("scope")["retention_rate"].to_dict()
    all_total = float(result["qc_evaluated_timepoints"].sum())
    all_usable = float(result["confirmed_usable_timepoints"].sum())
    fallback = all_usable / all_total if all_total else 1.0
    projected = []
    projected_z = []
    bases = []
    for _, row in result.iterrows():
        rate = rates.get(row["scope"], math.nan)
        if pd.isna(rate):
            rate = fallback
            basis = "all-scope completed-QC fallback"
        else:
            basis = "scope-specific retention"
        gross = float(row["gross_embryo_timepoints"])
        evaluated = min(float(row["qc_evaluated_timepoints"]), gross)
        confirmed = float(row["confirmed_usable_timepoints"])
        estimate = min(gross, confirmed + max(0.0, gross - evaluated) * rate)
        multiplier = (
            float(row["gross_embryo_z_observations"]) / gross if gross > 0 else 1.0
        )
        projected.append(estimate)
        projected_z.append(estimate * multiplier)
        bases.append(basis)
    result["projected_usable_timepoints"] = projected
    result["projected_usable_z_observations"] = projected_z
    result["projection_basis"] = bases
    return result


def aggregate_datasets(wells: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "gross_embryos",
        "observed_unique_physical_embryos",
        "gross_embryo_timepoints",
        "gross_embryo_z_observations",
        "observed_snip_timepoints",
        "observed_valid_snip_timepoints",
        "qc_evaluated_timepoints",
        "confirmed_usable_timepoints",
        "projected_usable_timepoints",
        "projected_usable_z_observations",
    ]
    grouped = wells.groupby(["scope", "corpus_dataset_id"], dropna=False)
    result = grouped[numeric_columns].sum().reset_index()
    extra = grouped.agg(
        pipeline_experiment_count=("pipeline_experiment_id", lambda x: x[x != ""].nunique()),
        well_count=("well_id", "nunique"),
        exact_well_count=(
            "count_basis",
            lambda x: int(
                x.isin(["exact_frame_inventory", "exact_detection_manifest"]).sum()
            ),
        ),
        perturbation_condition_count=("condition_signature", "nunique"),
        stage_min_hpf=("stage_hpf", "min"),
        stage_max_hpf=("stage_hpf", "max"),
        frame_inventory_fraction=("frame_inventory_present", "mean"),
        detected_fraction=("detected_or_registered", "mean"),
        sequencing_link_fraction=("has_seq_link", "mean"),
        project_name=("project_name", lambda x: "; ".join(sorted(set(x)))),
    ).reset_index()
    result = result.merge(extra, on=["scope", "corpus_dataset_id"], how="left")
    result["z_expansion_factor"] = (
        result["gross_embryo_z_observations"]
        / result["gross_embryo_timepoints"].replace(0, np.nan)
    )
    result["qc_coverage_fraction"] = (
        result["qc_evaluated_timepoints"]
        / result["gross_embryo_timepoints"].replace(0, np.nan)
    ).clip(upper=1)
    result["confirmed_retention_rate"] = (
        result["confirmed_usable_timepoints"]
        / result["qc_evaluated_timepoints"].replace(0, np.nan)
    )
    return result.sort_values(["scope", "corpus_dataset_id"]).reset_index(drop=True)


def aggregate_scope(datasets: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "gross_embryos",
        "observed_unique_physical_embryos",
        "gross_embryo_timepoints",
        "gross_embryo_z_observations",
        "observed_valid_snip_timepoints",
        "qc_evaluated_timepoints",
        "confirmed_usable_timepoints",
        "projected_usable_timepoints",
        "projected_usable_z_observations",
    ]
    result = datasets.groupby("scope")[numeric].sum()
    result["dataset_count"] = datasets.groupby("scope")["corpus_dataset_id"].nunique()
    result["z_expansion_factor"] = (
        result["gross_embryo_z_observations"]
        / result["gross_embryo_timepoints"].replace(0, np.nan)
    )
    result["projected_retention_rate"] = (
        result["projected_usable_timepoints"]
        / result["gross_embryo_timepoints"].replace(0, np.nan)
    )
    total = result[numeric + ["dataset_count"]].sum(numeric_only=True)
    total["z_expansion_factor"] = (
        total["gross_embryo_z_observations"] / total["gross_embryo_timepoints"]
    )
    total["projected_retention_rate"] = (
        total["projected_usable_timepoints"] / total["gross_embryo_timepoints"]
    )
    result.loc["All scopes"] = total
    return result.reset_index()


def perturbation_tables(wells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    measures = [
        "gross_embryos",
        "observed_unique_physical_embryos",
        "gross_embryo_timepoints",
        "gross_embryo_z_observations",
        "confirmed_usable_timepoints",
        "projected_usable_timepoints",
        "projected_usable_z_observations",
    ]
    condition = (
        wells.groupby(
            [
                "scope",
                "perturbation_class",
                "condition_signature",
                "has_environmental_perturbation",
                "has_chemical_perturbation",
                "has_genetic_perturbation",
            ],
            dropna=False,
        )[measures]
        .sum()
        .reset_index()
    )
    condition_counts = (
        wells.groupby(["scope", "perturbation_class", "condition_signature"])
        .agg(
            corpus_dataset_count=("corpus_dataset_id", "nunique"),
            well_count=("well_id", "nunique"),
            classification_needs_review=("classification_needs_review", "max"),
            classification_review_reason=(
                "classification_review_reason",
                lambda x: "; ".join(sorted(set(value for value in x if value))),
            ),
        )
        .reset_index()
    )
    condition = condition.merge(
        condition_counts,
        on=["scope", "perturbation_class", "condition_signature"],
        how="left",
    )

    atomic_rows = []
    for _, row in wells.iterrows():
        atoms = row["atomic_perturbations"]
        for domain, label in atoms:
            atomic_rows.append(
                {
                    "scope": row["scope"],
                    "corpus_dataset_id": row["corpus_dataset_id"],
                    "well_id": row["well_id"],
                    "perturbation_domain": domain,
                    "atomic_perturbation": label,
                    **{column: row[column] for column in measures},
                }
            )
    atomic_frame = pd.DataFrame(atomic_rows)
    if atomic_frame.empty:
        atomic = pd.DataFrame(
            columns=[
                "scope",
                "perturbation_domain",
                "atomic_perturbation",
                "corpus_dataset_count",
                "well_count",
                *measures,
            ]
        )
    else:
        atomic = (
            atomic_frame.groupby(
                ["scope", "perturbation_domain", "atomic_perturbation"],
                dropna=False,
            )
            .agg(
                corpus_dataset_count=("corpus_dataset_id", "nunique"),
                well_count=("well_id", "nunique"),
                **{column: (column, "sum") for column in measures},
            )
            .reset_index()
        )
    return condition, atomic


def sequencing_table(wells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overrides = (
        pd.read_csv(OVERRIDES_PATH)
        if OVERRIDES_PATH.is_file()
        else pd.DataFrame()
    )
    for (scope, dataset), group in wells.groupby(["scope", "corpus_dataset_id"]):
        known = group["has_seq_link"].dropna()
        if scope == "SeaHub":
            linked = int(known.astype(bool).sum())
            status = "linked" if linked else "not_linked"
        else:
            linked = 0
            status = "unknown"
        collections = "; ".join(
            sorted(set(value for value in group["collection_name"].astype(str) if value))
        )
        linked_mask = group["has_seq_link"].eq(True)
        rows.append(
            {
                "scope": scope,
                "corpus_dataset_id": dataset,
                "pairing_status": status,
                "linked_well_or_embryo_count": linked,
                "total_well_or_embryo_count": group["well_id"].nunique(),
                "linked_gross_embryo_timepoints": float(
                    group.loc[
                        linked_mask,
                        "gross_embryo_timepoints",
                    ].sum()
                ),
                "collection_name": collections,
                "sequencing_modality": "",
                "sequencing_dataset_id": "",
                "pairing_granularity": (
                    "collection/condition" if scope == "SeaHub" and linked else ""
                ),
                "evidence_source": (
                    "SeaHub collection metadata" if scope == "SeaHub" else ""
                ),
                "curator_notes": "",
            }
        )
    result = pd.DataFrame(rows)
    if not overrides.empty:
        keys = ["scope", "corpus_dataset_id"]
        override_columns = [
            column for column in overrides.columns if column not in keys
        ]
        merged = result.merge(overrides, on=keys, how="left", suffixes=("", "_override"))
        for column in override_columns:
            candidate = f"{column}_override"
            if candidate in merged:
                mask = merged[candidate].notna() & merged[candidate].astype(str).ne("")
                merged.loc[mask, column] = merged.loc[mask, candidate]
                merged = merged.drop(columns=candidate)
        result = merged
    return result.sort_values(["scope", "corpus_dataset_id"]).reset_index(drop=True)


def write_override_template() -> None:
    if OVERRIDES_PATH.exists():
        return
    columns = [
        "scope",
        "corpus_dataset_id",
        "pairing_status",
        "sequencing_modality",
        "sequencing_dataset_id",
        "pairing_granularity",
        "evidence_source",
        "curator_notes",
    ]
    pd.DataFrame(columns=columns).to_csv(OVERRIDES_PATH, index=False)


def validate_census(
    wells: pd.DataFrame,
    datasets: pd.DataFrame,
    atomic: pd.DataFrame,
) -> dict[str, Any]:
    expected_traditional = pd.Series(
        [scope for _, scope in manifest_datasets()]
    ).value_counts().to_dict()
    actual = datasets.groupby("scope")["corpus_dataset_id"].nunique().to_dict()
    reconciled = pd.read_csv(SEAHUB_RECONCILED)
    expected_seahub_gross = int(
        reconciled["include_for_seahub"].astype(bool).sum()
    ) * 8
    observed_seahub_gross = int(
        wells.loc[wells["scope"].eq("SeaHub"), "gross_embryo_timepoints"].sum()
    )

    inequalities = {}
    for numerator, denominator in [
        ("observed_snip_timepoints", "gross_embryo_timepoints"),
        ("qc_evaluated_timepoints", "gross_embryo_timepoints"),
        ("confirmed_usable_timepoints", "qc_evaluated_timepoints"),
        ("projected_usable_timepoints", "gross_embryo_timepoints"),
    ]:
        inequalities[f"{numerator}_gt_{denominator}"] = int(
            (wells[numerator] > wells[denominator] + 1e-6).sum()
        )

    suspicious_control_atoms = []
    if not atomic.empty:
        mask = atomic["atomic_perturbation"].astype(str).str.contains(
            r"ctrl|control|wildtype|inj.ctrl|^ab$|^wik$",
            case=False,
            regex=True,
        )
        suspicious_control_atoms = sorted(
            atomic.loc[mask, "atomic_perturbation"].astype(str).unique()
        )

    failures = []
    for scope, expected in expected_traditional.items():
        if actual.get(scope) != expected:
            failures.append(
                f"{scope} dataset count {actual.get(scope)} != manifest count {expected}"
            )
    if observed_seahub_gross != expected_seahub_gross:
        failures.append(
            f"SeaHub gross {observed_seahub_gross} != expected {expected_seahub_gross}"
        )
    failures.extend(
        f"{name}: {count} rows" for name, count in inequalities.items() if count
    )
    if suspicious_control_atoms:
        failures.append(
            "control-like labels remain in atomic perturbations: "
            + ", ".join(suspicious_control_atoms)
        )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "dataset_counts": actual,
        "expected_traditional_dataset_counts": expected_traditional,
        "seahub_gross_embryo_timepoints": observed_seahub_gross,
        "expected_seahub_gross_embryo_timepoints": expected_seahub_gross,
        "inequality_violations": inequalities,
        "suspicious_control_atoms": suspicious_control_atoms,
        "classification_review_row_count": int(
            wells["classification_needs_review"].sum()
        ),
        "estimated_or_unavailable_well_rows": int(
            wells["count_basis"].str.contains("estimated|unavailable", regex=True).sum()
        ),
        "failures": failures,
    }
    if failures:
        raise RuntimeError("Census validation failed: " + "; ".join(failures))
    return report


def write_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            """# MorphSeq training-corpus census — first pass

This notebook distinguishes **gross acquisition opportunity**, **currently observed pipeline
products**, **confirmed post-QC samples**, and **projected post-QC samples**.

Primary units:

1. **Embryo-timepoint**: physical embryo × time, collapsed over z and channels.
2. **Embryo-plane observation**: physical embryo × time × z, channels collapsed.

Keyence/YX1 gross counts come from canonical frame inventories and `embryos_per_well`.
SeaHub gross counts are 8 embryos per included FOV. Projections are preliminary and should
always be labeled as projections in talk graphics."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import display

ROOT = Path.cwd()
DATA = ROOT / "corpus_census_data"
FIGURES = ROOT / "figures" / "corpus_census"
FIGURES.mkdir(parents=True, exist_ok=True)

wells = pd.read_csv(DATA / "well_census.csv")
datasets = pd.read_csv(DATA / "dataset_census.csv")
scope = pd.read_csv(DATA / "scope_summary.csv")
retention = pd.read_csv(DATA / "qc_retention_summary.csv")
conditions = pd.read_csv(DATA / "perturbation_conditions.csv")
atomic = pd.read_csv(DATA / "atomic_perturbations.csv")
sequencing = pd.read_csv(DATA / "sequencing_pairing_table.csv")

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
scope_order = ["Keyence", "YX1", "SeaHub"]
colors = {"Keyence": "#3B82F6", "YX1": "#F59E0B", "SeaHub": "#10B981"}"""
        ),
        nbf.v4.new_markdown_cell("## Executive summary"),
        nbf.v4.new_code_cell(
            """summary_cols = [
    "scope", "dataset_count", "gross_embryos", "gross_embryo_timepoints",
    "gross_embryo_z_observations", "z_expansion_factor",
    "confirmed_usable_timepoints", "projected_usable_timepoints",
    "projected_retention_rate",
]
display(scope[summary_cols].style.format({
    "gross_embryos": "{:,.0f}",
    "gross_embryo_timepoints": "{:,.0f}",
    "gross_embryo_z_observations": "{:,.0f}",
    "z_expansion_factor": "{:.1f}×",
    "confirmed_usable_timepoints": "{:,.0f}",
    "projected_usable_timepoints": "{:,.0f}",
    "projected_retention_rate": "{:.1%}",
}))
display(retention.style.format({
    "retention_rate": "{:.1%}",
    "retention_ci95_low": "{:.1%}",
    "retention_ci95_high": "{:.1%}",
}))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Cumulative corpus growth by version

Each bar includes every acquisition dated on or before that version's cutoff. `v5` includes
the complete present-day **censusable** corpus.

- **Lower estimate:** strict current QC retention, extrapolated to every censusable dataset.
- **Upper estimate:** only death and structural mask-geometry exclusions.
- **Midpoint:** arithmetic midpoint of those bounds, used for the bar plots.

Both bounds assume that all censusable datasets eventually complete processing; they are not
counts of datasets that have already completed. SeaHub uses current detection completion as its
lower estimate and all reconciled embryos as its upper estimate because comparable downstream QC
is not yet available. The five-z comparison applies the fixed 5× expansion only to Keyence and
YX1 embryo-timepoints; SeaHub remains at one image per embryo. It is a projection, not measured
stack depth. Unique-perturbation trends count atomic perturbations from Keyence and YX1 only."""
        ),
        nbf.v4.new_code_cell(
            """from plot_training_corpus_versions import (
    REQUESTED_SLIDE_DIR,
    generate_version_figures,
)
from IPython.display import Image

# Plot controls.
COUNT_UNIT = "embryo_times"  # "embryos" or "embryo_times"
Z_SLICE_MULTIPLIER = 5

# The requested laptop path is used when the notebook runs there. Cluster execution falls
# back to a repository-local mirror because /Users/nick is not mounted on the cluster.
SLIDE_OUTPUT_DIR = (
    REQUESTED_SLIDE_DIR
    if REQUESTED_SLIDE_DIR.parent.is_dir()
    else ROOT / "data_census"
)

version_result = generate_version_figures(
    data_dir=DATA,
    output_dir=SLIDE_OUTPUT_DIR,
    count_unit=COUNT_UNIT,
    z_slice_multiplier=Z_SLICE_MULTIPLIER,
)
display(
    version_result["version_estimates"].style.format({
        "gross_count": "{:,.0f}",
        "lower_estimate": "{:,.0f}",
        "midpoint_estimate": "{:,.0f}",
        "upper_estimate": "{:,.0f}",
        "unique_perturbation_count": "{:,.0f}",
        "midpoint_5z_estimate": "{:,.0f}",
    })
)
display(
    version_result["projection_rates"].style.format({
        "lower_timepoint_rate": "{:.1%}",
        "upper_timepoint_rate": "{:.1%}",
        "lower_embryo_rate": "{:.1%}",
        "upper_embryo_rate": "{:.1%}",
    })
)
for figure_path in version_result["figure_paths"]:
    if figure_path.suffix.lower() == ".png":
        display(Image(filename=str(figure_path)))
print(f"Slide outputs: {version_result['output_dir']}")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Corpus scale by acquisition scope

Solid bars show gross acquisition opportunity; hatched overlays show projected post-QC
availability. Confirmed counts are intentionally shown separately because pipeline completion
is still changing."""
        ),
        nbf.v4.new_code_cell(
            """plot = scope[scope.scope.isin(scope_order)].set_index("scope").reindex(scope_order)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, gross_col, projected_col, title in [
    (axes[0], "gross_embryo_timepoints", "projected_usable_timepoints", "Embryo-timepoints"),
    (axes[1], "gross_embryo_z_observations", "projected_usable_z_observations", "Embryo × time × z"),
]:
    x = np.arange(len(plot))
    gross = plot[gross_col].to_numpy()
    projected = plot[projected_col].to_numpy()
    ax.bar(x, gross, color=[colors[s] for s in plot.index], alpha=.35, label="Gross")
    ax.bar(x, projected, color=[colors[s] for s in plot.index], label="Projected usable")
    ax.set_xticks(x, plot.index)
    ax.set_title(title)
    ax.set_ylabel("Observations")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[0].legend(frameon=False)
fig.suptitle("MorphSeq corpus size: gross versus projected post-QC")
fig.tight_layout()
fig.savefig(FIGURES / "corpus_size_by_scope.png", bbox_inches="tight")
fig.savefig(FIGURES / "corpus_size_by_scope.pdf", bbox_inches="tight")
plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## Processing and QC coverage"),
        nbf.v4.new_code_cell(
            """coverage = (
    datasets.groupby("scope")
    .agg(
        datasets=("corpus_dataset_id", "nunique"),
        gross=("gross_embryo_timepoints", "sum"),
        observed=("observed_valid_snip_timepoints", "sum"),
        qc_evaluated=("qc_evaluated_timepoints", "sum"),
        confirmed_usable=("confirmed_usable_timepoints", "sum"),
    )
    .reindex(scope_order)
)
for col in ["observed", "qc_evaluated", "confirmed_usable"]:
    coverage[col + "_fraction"] = coverage[col] / coverage["gross"].replace(0, np.nan)
display(coverage.style.format("{:,.0f}", subset=["gross", "observed", "qc_evaluated", "confirmed_usable"])
        .format("{:.1%}", subset=["observed_fraction", "qc_evaluated_fraction", "confirmed_usable_fraction"]))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Perturbation coverage

Conditions retain combinations (for example `environmental+chemical`). The atomic table
explodes combinations into unique temperature, chemical, and genetic targets."""
        ),
        nbf.v4.new_code_cell(
            """unique_atomic = (
    atomic.groupby(["scope", "perturbation_domain"])["atomic_perturbation"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(scope_order)
)
display(unique_atomic)
ax = unique_atomic.plot(kind="bar", stacked=True, figsize=(8, 4),
                        color={"environmental": "#8B5CF6", "chemical": "#EF4444", "genetic": "#06B6D4"})
ax.set_ylabel("Unique atomic perturbations")
ax.set_xlabel("")
ax.set_title("Perturbation diversity by acquisition scope")
ax.legend(title="", frameon=False)
plt.tight_layout()
plt.savefig(FIGURES / "unique_perturbations_by_scope.png", bbox_inches="tight")
plt.savefig(FIGURES / "unique_perturbations_by_scope.pdf", bbox_inches="tight")
plt.show()

condition_volume = (
    wells.groupby(["scope", "perturbation_class"])["gross_embryo_timepoints"]
    .sum().unstack(fill_value=0).reindex(scope_order)
)
display(condition_volume.style.format("{:,.0f}"))"""
        ),
        nbf.v4.new_markdown_cell("## Sequencing-linked imaging sets"),
        nbf.v4.new_code_cell(
            """seq_display = sequencing.loc[
    sequencing["pairing_status"].ne("not_linked"),
    [
        "scope", "corpus_dataset_id", "pairing_status",
        "linked_well_or_embryo_count", "total_well_or_embryo_count",
        "collection_name", "sequencing_modality", "sequencing_dataset_id",
        "pairing_granularity", "evidence_source", "curator_notes",
    ],
]
display(seq_display)
print("Edit sequencing_pairing_overrides.csv and refresh to curate Keyence/YX1 links.")"""
        ),
        nbf.v4.new_markdown_cell("## Classification review queue"),
        nbf.v4.new_code_cell(
            """review = (
    wells.loc[wells["classification_needs_review"].astype(bool),
              ["scope", "corpus_dataset_id", "perturbation", "genotype",
               "chem_perturbation", "temperature", "perturbation_class",
               "condition_signature", "classification_review_reason"]]
    .drop_duplicates()
    .sort_values(["scope", "corpus_dataset_id"])
)
print(f"{len(review):,} unique labels/conditions need review")
display(review.head(200))"""
        ),
        nbf.v4.new_markdown_cell("## Detailed dataset census"),
        nbf.v4.new_code_cell(
            """display(
    datasets.sort_values(["scope", "gross_embryo_timepoints"], ascending=[True, False])
    .style.format({
        "gross_embryo_timepoints": "{:,.0f}",
        "gross_embryo_z_observations": "{:,.0f}",
        "z_expansion_factor": "{:.1f}×",
        "qc_coverage_fraction": "{:.1%}",
        "confirmed_retention_rate": "{:.1%}",
        "projected_usable_timepoints": "{:,.0f}",
    })
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Important limitations

- Gross Keyence/YX1 counts use metadata `embryos_per_well`; where a frame inventory is
  missing, within-dataset median time/z depth is used and flagged.
- Where `embryos_per_well` is absent or too small, the peak number of concurrently detected
  embryos in that well supplies a conservative lower-bound embryo count. This avoids
  undercounting multi-embryo YX1 wells but can inherit detection errors.
- Projection rates describe the datasets that have completed QC; they may not represent
  the hardest unfinished datasets.
- SeaHub currently has detection-completion retention only, not completed downstream QC.
- “Sequencing linked” indicates a collection/condition link, not necessarily the identical
  individual embryo.
- Perturbation classes are rule-based. Review the generated classification queue before
  using exact diversity counts in a talk."""
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_override_template()

    print("Scanning Keyence/YX1 products...")
    rows = traditional_well_rows()
    print("Scanning SeaHub products...")
    rows.extend(seahub_well_rows())
    wells = pd.DataFrame(rows)

    retention = retention_summary(wells)
    wells = apply_projections(wells, retention)
    datasets = aggregate_datasets(wells)
    scope = aggregate_scope(datasets)
    conditions, atomic = perturbation_tables(wells)
    sequencing = sequencing_table(wells)
    validation = validate_census(wells, datasets, atomic)

    # Lists are convenient in memory but should be serialized explicitly in CSV/Excel.
    wells_out = wells.copy()
    wells_out["atomic_perturbations"] = wells_out["atomic_perturbations"].map(
        lambda value: json.dumps(value)
    )

    outputs = {
        "well_census.csv": wells_out,
        "dataset_census.csv": datasets,
        "scope_summary.csv": scope,
        "qc_retention_summary.csv": retention,
        "perturbation_conditions.csv": conditions,
        "atomic_perturbations.csv": atomic,
        "sequencing_pairing_table.csv": sequencing,
        "classification_review.csv": wells_out.loc[
            wells_out["classification_needs_review"].astype(bool)
        ].drop_duplicates(
            [
                "scope",
                "corpus_dataset_id",
                "perturbation",
                "genotype",
                "chem_perturbation",
                "temperature",
                "condition_signature",
            ]
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(DATA_DIR / filename, index=False)

    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl") as writer:
        scope.to_excel(writer, sheet_name="Scope Summary", index=False)
        datasets.to_excel(writer, sheet_name="Dataset Census", index=False)
        conditions.to_excel(writer, sheet_name="Conditions", index=False)
        atomic.to_excel(writer, sheet_name="Atomic Perturbations", index=False)
        retention.to_excel(writer, sheet_name="QC Retention", index=False)
        sequencing.to_excel(writer, sheet_name="Sequencing Pairing", index=False)
        outputs["classification_review.csv"].to_excel(
            writer, sheet_name="Classification Review", index=False
        )

    assumptions = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "reference_temperature_c": REFERENCE_TEMPERATURE_C,
        "environmental_temperature_rule": (
            f"temperature < {ENVIRONMENTAL_TEMPERATURE_MIN_C:g}C or "
            f"> {ENVIRONMENTAL_TEMPERATURE_MAX_C:g}C; 30C is not environmental"
        ),
        "channel_policy": "BF when present, otherwise first channel; channels collapsed",
        "seahub_gross_policy": "8 embryos per include_for_seahub FOV",
        "projection_policy": (
            "confirmed usable + scope retention x unevaluated gross remainder; "
            "SeaHub retention is detection completion only until downstream QC exists"
        ),
        "traditional_dataset_count": len(manifest_datasets()),
        "seahub_bundle_shard_count": len(
            list(SEAHUB_BUNDLE.glob("20260724_seahub_*"))
        ),
    }
    (DATA_DIR / "assumptions.json").write_text(
        json.dumps(assumptions, indent=2) + "\n"
    )
    (DATA_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )

    write_notebook()
    if not args.no_execute_notebook:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                NOTEBOOK_PATH.name,
            ],
            cwd=HERE,
            check=True,
        )

    print(f"well rows: {len(wells):,}")
    print(f"dataset rows: {len(datasets):,}")
    print(f"workbook: {WORKBOOK_PATH}")
    print(f"notebook: {NOTEBOOK_PATH}")
    print(scope.to_string(index=False))


if __name__ == "__main__":
    main()
