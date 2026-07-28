"""Minimal SeaHub image registry, metadata reconciliation, and embryo cropping.

All functions that write files restrict outputs to this module's directory.
The SeaHub image and metadata roots are treated as read-only inputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import types
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WORK_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/image_data"
)
DEFAULT_METADATA_PATH = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/metadata/"
    "collection_metadata.xlsx"
)
MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
SEGMENTATION_SANDBOX = MORPHSEQ_ROOT / "segmentation_sandbox"
DEFAULT_GDINO_SOURCE = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/GroundingDINO"
)
DEFAULT_GDINO_WEIGHTS = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/"
    "Open-GroundingDino/finetune_output/"
    "finetune_output_run_nick_masks_20250308/checkpoint_best_regular.pth"
)
DEFAULT_GDINO_CONFIG = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/"
    "Open-GroundingDino/finetune_output/"
    "finetune_output_run_nick_masks_20250308/config_cfg.py"
)
OPTIONAL_TRANSFORMERS_SITE = Path(
    "/net/trapnell/vol1/home/nlammers/micromamba/envs/"
    "points-ml/lib/python3.10/site-packages"
)
DEFAULT_THRESHOLD_SWEEP = (
    (0.15, 0.10),
    (0.10, 0.05),
    (0.05, 0.05),
)

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_STAGE_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)?hpf|\d+s|bud|shield)$", re.IGNORECASE
)
_EXPERIMENT_PATTERN = re.compile(r"\b((?:GENE|CHEM)\d+)\b", re.IGNORECASE)
_FOV_LABEL_PATTERN = re.compile(r"^[A-J](?:\d+)?(?:-\d+)?$", re.IGNORECASE)
_EXCLUDED_PATH_PATTERN = re.compile(
    r"(?:abandon(?:ed)?|not[\s_-]*used)", re.IGNORECASE
)


def _require_work_output(path: str | Path) -> Path:
    """Resolve an output path and reject writes outside WORK_DIR."""
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(WORK_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside designated work directory {WORK_DIR}: "
            f"{resolved}"
        ) from exc
    return resolved


def _column_number(cell_reference: str) -> int:
    letters_match = re.match(r"[A-Z]+", cell_reference.upper())
    if letters_match is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    number = 0
    for letter in letters_match.group(0):
        number = number * 26 + ord(letter) - ord("A") + 1
    return number - 1


def _unique_headers(values: Sequence[Any]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values):
        base = str(value).strip() if value not in (None, "") else f"column_{index}"
        counts[base] = counts.get(base, 0) + 1
        headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return headers


def read_xlsx_sheet(
    path: str | Path,
    sheet_name: str,
    *,
    header: bool = True,
) -> pd.DataFrame:
    """Read a simple XLSX worksheet using only the Python standard library.

    This handles shared strings, inline strings, booleans, errors, and cached
    numeric/formula values. It is intentionally small and is not a replacement
    for a full Excel engine.
    """
    path = Path(path)
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{_MAIN_NS}t"))
                for item in shared_root.findall(f"{_MAIN_NS}si")
            ]

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root.findall(f"{_PKG_REL_NS}Relationship")
        }
        sheet_target = None
        for sheet in workbook_root.findall(f".//{_MAIN_NS}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                sheet_target = targets[sheet.attrib[f"{_REL_NS}id"]]
                break
        if sheet_target is None:
            available = [
                sheet.attrib.get("name")
                for sheet in workbook_root.findall(f".//{_MAIN_NS}sheet")
            ]
            raise KeyError(f"Worksheet {sheet_name!r} not found; available={available}")

        xml_path = (
            sheet_target.lstrip("/")
            if sheet_target.startswith("/")
            else f"xl/{sheet_target}"
        )
        sheet_root = ET.fromstring(archive.read(xml_path))
        sparse_rows: list[dict[int, Any]] = []
        max_column = -1
        for row in sheet_root.iter(f"{_MAIN_NS}row"):
            values: dict[int, Any] = {}
            for cell in row.findall(f"{_MAIN_NS}c"):
                column = _column_number(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{_MAIN_NS}v")
                value: Any = None if value_node is None else value_node.text
                if cell_type == "s" and value is not None:
                    value = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{_MAIN_NS}t")
                    )
                elif cell_type == "b" and value is not None:
                    value = value == "1"
                elif cell_type in (None, "n") and value not in (None, ""):
                    try:
                        numeric = float(value)
                        value = int(numeric) if numeric.is_integer() else numeric
                    except ValueError:
                        pass
                values[column] = value
                max_column = max(max_column, column)
            if values and any(value not in (None, "") for value in values.values()):
                sparse_rows.append(values)

    if not sparse_rows:
        return pd.DataFrame()
    rows = [
        [sparse.get(column) for column in range(max_column + 1)]
        for sparse in sparse_rows
    ]
    if not header:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows[1:], columns=_unique_headers(rows[0]))


def _snake_case(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    return re.sub(r"_+", "_", value).strip("_").lower()


def normalize_stage(value: Any) -> str | None:
    """Normalize collection stages while retaining meaningful decimal values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    normalized = re.sub(r"\s+", "", str(value)).casefold()
    normalized = normalized.replace("hrs", "hpf").replace("hr", "hpf")
    if normalized == "48hpd":
        normalized = "48hpf"
    return normalized or None


def stage_to_hpf(value: Any) -> float | None:
    """Return a numeric hour-post-fertilization value when explicitly encoded.

    Somite and named stages are left null rather than converted using an
    unstated biological approximation.
    """
    normalized = normalize_stage(value)
    if not normalized or not normalized.endswith("hpf"):
        return None
    try:
        return float(normalized.removesuffix("hpf"))
    except ValueError:
        return None


def _stage_token(value: Any) -> str | None:
    normalized = normalize_stage(value)
    return normalized if normalized and _STAGE_PATTERN.fullmatch(normalized) else None


def canonical_condition(value: Any) -> str | None:
    """Build a separator- and order-tolerant condition key."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).casefold().replace("\n", " ")
    tokens = re.findall(r"[a-z]+[a-z0-9]*|\d+", text)
    ignored = {
        "cropped",
        "uncropped",
        "closeup",
        "image",
        "images",
        "hpf",
        "pos",
        "neg",
    }
    tokens = [token for token in tokens if token not in ignored]
    return "|".join(sorted(tokens)) or None


def extract_experiment_id(parts: Iterable[str]) -> str | None:
    for part in parts:
        match = _EXPERIMENT_PATTERN.search(str(part))
        if match:
            return match.group(1).upper()
    return None


def _strip_crop_suffix(stem: str) -> tuple[str, str | None]:
    match = re.search(r"_(un)?cropped$", stem, flags=re.IGNORECASE)
    if not match:
        return stem, None
    return stem[: match.start()], match.group(0).lstrip("_").casefold()


def _directory_condition_stage(relative_path: Path) -> tuple[str | None, str | None]:
    for part in reversed(relative_path.parts[:-1]):
        match = re.match(
            r"^(?P<condition>.+)_(?P<stage>\d+(?:\.\d+)?hpf|\d+s|bud|shield)$",
            part.strip(),
            flags=re.IGNORECASE,
        )
        if match:
            return match.group("condition").strip(), normalize_stage(match.group("stage"))
    return None, None


def parse_image_name(relative_path: str | Path) -> dict[str, Any]:
    """Parse the main observed SeaHub filename and directory conventions."""
    relative_path = Path(relative_path)
    stem, crop_variant = _strip_crop_suffix(relative_path.stem)
    tokens = [token.strip() for token in stem.split("_") if token.strip()]
    condition_from_dir, stage_from_dir = _directory_condition_stage(relative_path)
    stage_collected: str | None = None
    stage_addition: str | None = None
    condition: str | None = None
    fov_label: str | None = None
    source_embryo_label: str | None = None
    parse_method = "unparsed"

    # Legacy pre-existing single-embryo snips use condition_stage directories
    # and names such as tbx6_20240717_A05_uncropped.jpg.
    if crop_variant and stage_from_dir:
        stage_collected = stage_from_dir
        condition = condition_from_dir
        if len(tokens) >= 3 and re.fullmatch(r"[A-H]\d+", tokens[-1], re.IGNORECASE):
            source_embryo_label = tokens[-1].upper()
        parse_method = "condition_stage_directory"
    elif tokens and _stage_token(tokens[0]):
        stage_collected = _stage_token(tokens[0])
        remainder = tokens[1:]
        if remainder and _FOV_LABEL_PATTERN.fullmatch(remainder[0]):
            fov_label = remainder.pop(0).upper()
        elif remainder and remainder[-1].isdigit():
            fov_label = remainder.pop()
        if remainder and _stage_token(remainder[-1]) in {"shield", "bud"}:
            stage_addition = _stage_token(remainder.pop())
        condition = "_".join(remainder).strip() or condition_from_dir
        parse_method = "stage_first_filename"
    elif (
        len(tokens) >= 4
        and tokens[0].isdigit()
        and _stage_token(tokens[1])
        and _stage_token(tokens[-1])
    ):
        stage_addition = _stage_token(tokens[1])
        stage_collected = _stage_token(tokens[-1])
        condition = "_".join(tokens[2:-1]).strip() or None
        fov_label = tokens[0]
        parse_method = "numbered_chemical_filename"
    elif stage_from_dir:
        stage_collected = stage_from_dir
        condition = condition_from_dir
        parse_method = "condition_stage_directory"

    return {
        "stage_hpf": stage_to_hpf(stage_collected),
        "stage_source_label": stage_collected,
        "stage_addition_hpf": stage_to_hpf(stage_addition),
        "stage_addition_source_label": stage_addition,
        "perturbation_parsed": condition,
        "perturbation_key": canonical_condition(condition),
        "fov_label": fov_label,
        "source_embryo_label": source_embryo_label,
        "crop_variant": crop_variant,
        "filename_parse_method": parse_method,
    }


def build_image_registry(
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    *,
    read_dimensions: bool = True,
) -> pd.DataFrame:
    """Recursively register every JPG/JPEG without changing the source tree."""
    image_root = Path(image_root).resolve()
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
    )
    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(image_root)
        stat = image_path.stat()
        experiment_id = extract_experiment_id(relative_path.parts)
        parsed = parse_image_name(relative_path)
        width: int | None = None
        height: int | None = None
        image_mode: str | None = None
        read_error: str | None = None
        if read_dimensions:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_mode = image.mode
            except Exception as exc:  # pragma: no cover - only corrupt inputs
                read_error = f"{type(exc).__name__}: {exc}"

        explicit_crop = parsed["crop_variant"] is not None
        if (
            (width, height) == (1280, 960)
            and not explicit_crop
            and "closeup" not in image_path.stem.casefold()
            and "pheno_grid" not in image_path.stem.casefold()
        ):
            image_role = "eight_embryo_fov"
        elif explicit_crop or (width, height) in {(256, 576), (576, 256)}:
            image_role = "existing_single_embryo"
        elif read_error:
            image_role = "unreadable"
        else:
            image_role = "review"

        relative_text = relative_path.as_posix()
        batch_folder = next(
            (
                part
                for part in relative_path.parts
                if re.search(r"\bbatch\b|^B\d+_", part, re.IGNORECASE)
            ),
            None,
        )
        image_id = hashlib.sha1(relative_text.encode("utf-8")).hexdigest()[:16]
        record = {
            "image_id": image_id,
            "image_path": str(image_path),
            "relative_path": relative_text,
            "filename": image_path.name,
            "stem": image_path.stem,
            "extension": image_path.suffix.casefold(),
            "size_bytes": stat.st_size,
            "modified_time": pd.Timestamp(stat.st_mtime, unit="s"),
            "perturbation_domain": (
                relative_path.parts[0].removesuffix("_perturbations")
                if relative_path.parts
                else None
            ),
            "experiment_id": experiment_id,
            "batch_folder": batch_folder,
            "width_px": width,
            "height_px": height,
            "image_mode": image_mode,
            "image_role": image_role,
            "excluded_path": bool(_EXCLUDED_PATH_PATTERN.search(relative_text)),
            "read_error": read_error,
        }
        record.update(parsed)
        records.append(record)
    registry = pd.DataFrame.from_records(records)
    if not registry.empty and registry["image_id"].duplicated().any():
        raise RuntimeError("A truncated SHA-1 image_id collision occurred")
    return registry


def load_collection_metadata(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    sheet_name: str = "collection_metadata",
) -> pd.DataFrame:
    """Read and lightly standardize the SeaHub collection metadata workbook."""
    metadata = read_xlsx_sheet(metadata_path, sheet_name=sheet_name)
    metadata.columns = [_snake_case(column) for column in metadata.columns]
    metadata = metadata.dropna(how="all").reset_index(drop=True)
    metadata.insert(0, "metadata_row_number", metadata.index + 2)
    if "collection_date" in metadata:
        numeric_dates = pd.to_numeric(metadata["collection_date"], errors="coerce")
        converted_dates = pd.to_datetime(
            numeric_dates, unit="D", origin="1899-12-30", errors="coerce"
        )
        metadata["collection_date_excel_raw"] = metadata["collection_date"]
        metadata["collection_date"] = converted_dates.dt.date
    if "stage_collected" in metadata:
        metadata["stage_hpf"] = metadata["stage_collected"].map(stage_to_hpf)
    return metadata


def _parse_collection_name(
    collection_name: Any,
    experiment_id: Any,
    fallback_stage: Any,
) -> tuple[str | None, str | None, str | None]:
    if collection_name is None or pd.isna(collection_name):
        return None, normalize_stage(fallback_stage), None
    tokens = [
        token.strip()
        for token in re.sub(r"[\r\n]+", "", str(collection_name)).split("_")
        if token.strip()
    ]
    if tokens and experiment_id and tokens[0].casefold() == str(experiment_id).casefold():
        tokens = tokens[1:]
    stage_indices = [
        index for index, token in enumerate(tokens) if _stage_token(token) is not None
    ]
    if not stage_indices:
        return "_".join(tokens) or None, normalize_stage(fallback_stage), None
    collected_index = stage_indices[-1]
    collected_stage = _stage_token(tokens[collected_index])
    addition_stage = (
        _stage_token(tokens[stage_indices[-2]]) if len(stage_indices) > 1 else None
    )
    condition_tokens = [
        token
        for index, token in enumerate(tokens[:collected_index])
        if index not in stage_indices
    ]
    return "_".join(condition_tokens) or None, collected_stage, addition_stage


def _prepare_metadata_for_matching(metadata: pd.DataFrame) -> pd.DataFrame:
    prepared = metadata.copy()
    if "expt" not in prepared:
        raise KeyError("Collection metadata must contain an 'expt' column")
    parsed = prepared.apply(
        lambda row: _parse_collection_name(
            row.get("collection_name"),
            row.get("expt"),
            row.get("stage_collected"),
        ),
        axis=1,
        result_type="expand",
    )
    parsed.columns = [
        "_collection_condition",
        "_match_stage",
        "_match_addition_stage",
    ]
    prepared = pd.concat([prepared, parsed], axis=1)
    prepared["_match_experiment"] = prepared["expt"].map(
        lambda value: str(value).strip().upper() if pd.notna(value) else None
    )
    prepared["_condition_keys"] = prepared.apply(
        lambda row: (
            (canonical_condition(row.get("_collection_condition")),)
            if canonical_condition(row.get("_collection_condition"))
            else tuple(
                key
                for key in (canonical_condition(row.get("perturbation")),)
                if key
            )
        ),
        axis=1,
    )
    return prepared


def _condition_similarity(image_key: str | None, candidate_keys: Sequence[str]) -> float:
    if not image_key or not candidate_keys:
        return 0.0
    return max(SequenceMatcher(None, image_key, key).ratio() for key in candidate_keys)


def reconcile_with_collection_metadata(
    registry: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    fuzzy_threshold: float = 0.90,
) -> pd.DataFrame:
    """Match images to metadata on experiment, stage, and perturbation.

    Exact matches are order/separator-tolerant. Conservative fuzzy matching is
    attempted only within the same experiment and collection stage.
    """
    prepared = _prepare_metadata_for_matching(metadata)
    by_experiment = {
        key: group
        for key, group in prepared.groupby("_match_experiment", dropna=True)
    }
    metadata_columns = [
        column for column in metadata.columns if column != "metadata_row_number"
    ]
    outputs: list[dict[str, Any]] = []

    for _, image_row in registry.iterrows():
        image_record = image_row.to_dict()
        experiment_id = image_record.get("experiment_id")
        stage = normalize_stage(image_record.get("stage_source_label"))
        addition_stage = normalize_stage(
            image_record.get("stage_addition_source_label")
        )
        condition_key = image_record.get("perturbation_key")
        candidates = by_experiment.get(experiment_id, prepared.iloc[0:0])
        same_stage = candidates[
            candidates["_match_stage"].map(normalize_stage).eq(stage)
        ]
        if addition_stage:
            same_stage = same_stage[
                same_stage["_match_addition_stage"]
                .map(normalize_stage)
                .eq(addition_stage)
            ]
        exact = same_stage[
            same_stage["_condition_keys"].map(
                lambda keys: bool(condition_key and condition_key in keys)
            )
        ]

        selected: pd.Series | None = None
        status = "unmatched"
        score = np.nan
        candidate_count = 0
        if not exact.empty:
            selected = exact.iloc[0]
            candidate_count = len(exact)
            status = "exact" if len(exact) == 1 else "exact_duplicate"
            score = 1.0
        elif condition_key and not same_stage.empty:
            scored = same_stage.copy()
            scored["_similarity"] = scored["_condition_keys"].map(
                lambda keys: _condition_similarity(condition_key, keys)
            )
            scored = scored.sort_values("_similarity", ascending=False)
            best_score = float(scored.iloc[0]["_similarity"])
            close = scored[scored["_similarity"] >= best_score - 0.02]
            candidate_count = len(close)
            if best_score >= fuzzy_threshold:
                selected = scored.iloc[0]
                score = best_score
                status = "fuzzy" if len(close) == 1 else "fuzzy_ambiguous"
            else:
                status = "unmatched_condition"
                score = best_score
        elif experiment_id is None:
            status = "unmatched_experiment"
        elif stage is None:
            status = "unmatched_stage"
        elif candidates.empty:
            status = "unmatched_experiment"
        elif same_stage.empty:
            status = "unmatched_stage"
        else:
            status = "unmatched_condition"

        image_record["metadata_match_status"] = status
        image_record["metadata_match_score"] = score
        image_record["metadata_candidate_count"] = candidate_count
        image_record["metadata_row_number"] = (
            selected.get("metadata_row_number") if selected is not None else np.nan
        )
        for column in metadata_columns:
            image_record[f"metadata_{column}"] = (
                selected.get(column) if selected is not None else None
            )
        outputs.append(image_record)
    return pd.DataFrame.from_records(outputs)


def export_registry_tables(
    registry: pd.DataFrame,
    reconciled: pd.DataFrame,
    output_dir: str | Path = WORK_DIR / "outputs",
) -> tuple[Path, Path]:
    """Write the two exploratory tables inside the designated work directory."""
    output_dir = _require_work_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "image_registry.csv"
    reconciled_path = output_dir / "image_metadata_reconciliation.csv"
    registry.to_csv(registry_path, index=False)
    reconciled.to_csv(reconciled_path, index=False)
    return registry_path, reconciled_path


def load_grounding_dino_detector(
    *,
    weights_path: str | Path = DEFAULT_GDINO_WEIGHTS,
    config_path: str | Path = DEFAULT_GDINO_CONFIG,
    source_path: str | Path = DEFAULT_GDINO_SOURCE,
    device: str = "cpu",
    cache_dir: str | Path = WORK_DIR / ".model_cache",
):
    """Load the same fine-tuned GroundingDINO model used by the beta script."""
    weights_path = Path(weights_path)
    config_path = Path(config_path)
    source_path = Path(source_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"GroundingDINO weights not found: {weights_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
    if not (source_path / "groundingdino").is_dir():
        raise FileNotFoundError(f"GroundingDINO source not found: {source_path}")
    cache_dir = _require_work_output(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    sys.dont_write_bytecode = True

    # morphseq-env has torch/timm but lacks the optional text-model stack.
    # Reuse the already-installed, read-only packages from points-ml.
    if importlib.util.find_spec("transformers") is None:
        if not OPTIONAL_TRANSFORMERS_SITE.is_dir():
            raise ImportError(
                "transformers is absent and the existing points-ml fallback "
                f"was not found at {OPTIONAL_TRANSFORMERS_SITE}"
            )
        sys.path.append(str(OPTIONAL_TRANSFORMERS_SITE))

    # GroundingDINO uses these two tiny packages only for configuration/log
    # conveniences. Local no-op-compatible shims avoid installing packages.
    if importlib.util.find_spec("addict") is None:
        addict_module = types.ModuleType("addict")

        class AttributeDict(dict):
            def __getattr__(self, key):
                try:
                    value = self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc
                if isinstance(value, dict) and not isinstance(value, AttributeDict):
                    value = AttributeDict(value)
                    self[key] = value
                return value

            def __setattr__(self, key, value):
                self[key] = value

        addict_module.Dict = AttributeDict
        sys.modules["addict"] = addict_module
    if importlib.util.find_spec("termcolor") is None:
        termcolor_module = types.ModuleType("termcolor")
        termcolor_module.colored = lambda text, *args, **kwargs: text
        sys.modules["termcolor"] = termcolor_module
    if importlib.util.find_spec("pycocotools") is None:
        pycocotools_module = types.ModuleType("pycocotools")
        mask_module = types.ModuleType("pycocotools.mask")
        pycocotools_module.mask = mask_module
        sys.modules["pycocotools"] = pycocotools_module
        sys.modules["pycocotools.mask"] = mask_module

    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    import argparse
    import torch
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
    arguments = SLConfig.fromfile(str(config_path))
    arguments.device = device
    model = build_model(arguments)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model._annotation_metadata = {
        "model_config_path": str(config_path),
        "model_weights_path": str(weights_path),
        "model_architecture": "GroundingDINO",
    }
    return model


def _boxes_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=float)
    cx, cy, width, height = boxes.T
    return np.column_stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    )


def _intersection_over_union(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_a + area_b - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def non_max_suppression(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float = 0.50,
) -> np.ndarray:
    """Return retained indices after confidence-ordered NMS."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        order = remaining[
            _intersection_over_union(boxes[current], boxes[remaining])
            <= iou_threshold
        ]
    return np.asarray(keep, dtype=int)


def assign_embryo_positions(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Assign 1..8 as top-row L->R then bottom-row L->R.

    For exactly eight detections the four smallest y centers form the top row.
    Other counts are sorted by y then x and are intended only for QC previews.
    """
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    if len(boxes) == 8:
        y_order = np.argsort(centers_y)
        top = y_order[:4][np.argsort(centers_x[y_order[:4]])]
        bottom = y_order[4:][np.argsort(centers_x[y_order[4:]])]
        order = np.concatenate([top, bottom])
    else:
        order = np.lexsort((centers_x, centers_y))
    positions = np.empty(len(boxes), dtype=int)
    positions[order] = np.arange(1, len(boxes) + 1)
    return positions


def _predict_threshold_sweep(
    model,
    image_path: str | Path,
    threshold_sweep: Sequence[tuple[float, float]],
    *,
    device: str,
    expected_count: int,
    nms_iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    import torch

    import groundingdino.datasets.transforms as transforms
    from groundingdino.util.utils import get_phrases_from_posmap

    transform = transforms.Compose(
        [
            transforms.RandomResize([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    with Image.open(image_path) as source_image:
        image_tensor, _ = transform(source_image.convert("RGB"), None)
    caption = "individual embryo."
    tokenized = model.tokenizer(caption)
    candidates: list[
        tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]
    ] = []
    for box_threshold, text_threshold in threshold_sweep:
        model = model.to(device)
        with torch.no_grad():
            outputs = model(image_tensor[None].to(device), captions=[caption])
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
        prediction_boxes = outputs["pred_boxes"].cpu()[0]
        detection_mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits_tensor = prediction_logits[detection_mask]
        boxes_tensor = prediction_boxes[detection_mask]
        phrases = [
            get_phrases_from_posmap(
                logit > text_threshold, tokenized, model.tokenizer
            ).replace(".", "")
            for logit in logits_tensor
        ]
        boxes_cxcywh = boxes_tensor.detach().cpu().numpy()
        scores = logits_tensor.max(dim=1)[0].detach().cpu().numpy().reshape(-1)
        boxes = np.clip(_boxes_cxcywh_to_xyxy(boxes_cxcywh), 0, 1)
        keep = non_max_suppression(
            boxes, scores, iou_threshold=nms_iou_threshold
        )
        boxes, scores = boxes[keep], scores[keep]
        phrases = [phrases[index] for index in keep]
        if len(boxes) > expected_count:
            best = np.argsort(scores)[::-1][:expected_count]
            boxes, scores = boxes[best], scores[best]
            phrases = [phrases[index] for index in best]
        details = {
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "raw_detection_count": len(boxes_cxcywh),
            "nms_detection_count": len(keep),
            "selected_detection_count": len(boxes),
        }
        candidate = (boxes, scores, phrases, details)
        candidates.append(candidate)
        if len(boxes) == expected_count:
            return candidate

    return min(
        candidates,
        key=lambda item: (
            abs(len(item[0]) - expected_count),
            -float(np.mean(item[1])) if len(item[1]) else np.inf,
        ),
    )


def _expanded_pixel_box(
    normalized_box: Sequence[float],
    image_size: tuple[int, int],
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = np.asarray(normalized_box, dtype=float)
    pad_x = (x2 - x1) * padding_fraction
    pad_y = (y2 - y1) * padding_fraction
    return (
        max(0, int(np.floor((x1 - pad_x) * width))),
        max(0, int(np.floor((y1 - pad_y) * height))),
        min(width, int(np.ceil((x2 + pad_x) * width))),
        min(height, int(np.ceil((y2 + pad_y) * height))),
    )


def save_embryo_contact_sheet(
    manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    variant: str = "color",
    columns: int = 4,
    gap_px: int = 8,
) -> Path:
    """Save position-sorted snips with per-embryo confidence labels.

    Each panel is annotated in its upper-left corner as
    ``Pos <position>  conf=<score>``. Input snips retain their aspect ratio and
    native resolution; smaller snips are centered within a common panel size.
    """
    output_path = _require_work_output(output_path)
    if variant not in {"color", "grayscale"}:
        raise ValueError("variant must be 'color' or 'grayscale'")
    if columns < 1:
        raise ValueError("columns must be at least 1")
    if gap_px < 0:
        raise ValueError("gap_px cannot be negative")
    if manifest.empty:
        raise ValueError("Cannot make a contact sheet from an empty manifest")

    path_column = (
        "snip_color_path" if variant == "color" else "snip_grayscale_path"
    )
    required = {"embryo_position", "detection_confidence", path_column}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing contact-sheet columns: {sorted(missing)}"
        )

    ordered = manifest.sort_values("embryo_position")
    mode = "RGB" if variant == "color" else "L"
    background = (255, 255, 255) if mode == "RGB" else 255
    label_background = (0, 0, 0) if mode == "RGB" else 0
    label_foreground = (255, 255, 255) if mode == "RGB" else 255
    panels: list[tuple[Image.Image, int, float]] = []
    for _, row in ordered.iterrows():
        with Image.open(row[path_column]) as source:
            panel = source.convert(mode).copy()
        panels.append(
            (
                panel,
                int(row["embryo_position"]),
                float(row["detection_confidence"]),
            )
        )

    panel_width = max(panel.width for panel, _, _ in panels)
    panel_height = max(panel.height for panel, _, _ in panels)
    rows = (len(panels) + columns - 1) // columns
    sheet_width = columns * panel_width + (columns + 1) * gap_px
    sheet_height = rows * panel_height + (rows + 1) * gap_px
    sheet = Image.new(mode, (sheet_width, sheet_height), background)
    draw = ImageDraw.Draw(sheet)
    font_size = max(12, min(20, panel_width // 9))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()

    for index, (panel, position, confidence) in enumerate(panels):
        row_index, column_index = divmod(index, columns)
        cell_x = gap_px + column_index * (panel_width + gap_px)
        cell_y = gap_px + row_index * (panel_height + gap_px)
        image_x = cell_x + (panel_width - panel.width) // 2
        image_y = cell_y + (panel_height - panel.height) // 2
        sheet.paste(panel, (image_x, image_y))

        label = f"P{position}  {confidence:.3f}"
        label_x, label_y = image_x + 4, image_y + 4
        text_box = draw.textbbox((label_x, label_y), label, font=font)
        draw.rounded_rectangle(
            (
                text_box[0] - 3,
                text_box[1] - 2,
                text_box[2] + 3,
                text_box[3] + 2,
            ),
            radius=3,
            fill=label_background,
        )
        draw.text(
            (label_x, label_y),
            label,
            fill=label_foreground,
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def run_grounding_dino_segmentation(
    images: pd.DataFrame,
    output_dir: str | Path,
    *,
    model=None,
    weights_path: str | Path = DEFAULT_GDINO_WEIGHTS,
    device: str = "cpu",
    threshold_sweep: Sequence[tuple[float, float]] = DEFAULT_THRESHOLD_SWEEP,
    expected_count: int = 8,
    nms_iou_threshold: float = 0.50,
    padding_fraction: float = 0.08,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect embryos, assign FOV positions, and save individual JPG snips.

    Crops are written only when exactly ``expected_count`` detections survive
    filtering. Mismatches receive a QC preview and row but no partial snip set.
    """
    output_dir = _require_work_output(output_dir)
    color_snip_dir = output_dir / "snips"
    grayscale_snip_dir = output_dir / "snips_grayscale"
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    color_snip_dir.mkdir(parents=True, exist_ok=True)
    grayscale_snip_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        images = images.head(limit)
    if model is None:
        model = load_grounding_dino_detector(
            weights_path=weights_path,
            device=device,
            cache_dir=output_dir / ".model_cache",
        )

    manifest_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for _, image_record in images.iterrows():
        image_path = Path(image_record["image_path"])
        boxes, scores, phrases, details = _predict_threshold_sweep(
            model,
            image_path,
            threshold_sweep,
            device=device,
            expected_count=expected_count,
            nms_iou_threshold=nms_iou_threshold,
        )
        detected_count = len(boxes)
        qc_status = "pass" if detected_count == expected_count else "count_mismatch"
        positions = assign_embryo_positions(boxes)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        preview = image.copy()
        draw = ImageDraw.Draw(preview)

        for box, score, phrase, position in zip(
            boxes, scores, phrases, positions
        ):
            pixel_box = _expanded_pixel_box(box, image.size, padding_fraction)
            color = "#00ff66" if qc_status == "pass" else "#ff9f1c"
            draw.rectangle(pixel_box, outline=color, width=3)
            draw.text(
                (pixel_box[0] + 4, pixel_box[1] + 4),
                f"{position}: {float(score):.2f}",
                fill=color,
                stroke_width=2,
                stroke_fill="black",
            )
            if qc_status != "pass":
                continue
            snip_name = (
                f"{image_record['image_id']}__embryo_{int(position):02d}.jpg"
            )
            snip_color_path = color_snip_dir / snip_name
            snip_grayscale_path = grayscale_snip_dir / snip_name
            color_crop = image.crop(pixel_box)
            color_crop.save(snip_color_path, quality=95)
            color_crop.convert("L").save(snip_grayscale_path, quality=95)
            row = image_record.to_dict()
            row.update(
                {
                    "embryo_position": int(position),
                    "snip_color_path": str(snip_color_path),
                    "snip_grayscale_path": str(snip_grayscale_path),
                    "detection_confidence": float(score),
                    "detection_phrase": phrase,
                    "box_x1_norm": float(box[0]),
                    "box_y1_norm": float(box[1]),
                    "box_x2_norm": float(box[2]),
                    "box_y2_norm": float(box[3]),
                    "crop_x1_px": pixel_box[0],
                    "crop_y1_px": pixel_box[1],
                    "crop_x2_px": pixel_box[2],
                    "crop_y2_px": pixel_box[3],
                    "segmentation_qc_status": qc_status,
                    **details,
                }
            )
            manifest_rows.append(row)

        preview_path = preview_dir / f"{image_record['image_id']}__detections.jpg"
        preview.save(preview_path, quality=92)
        qc_rows.append(
            {
                "image_id": image_record["image_id"],
                "image_path": str(image_path),
                "preview_path": str(preview_path),
                "segmentation_qc_status": qc_status,
                "expected_detection_count": expected_count,
                **details,
            }
        )

    manifest = pd.DataFrame.from_records(manifest_rows)
    qc = pd.DataFrame.from_records(qc_rows)
    manifest.to_csv(output_dir / "embryo_manifest.csv", index=False)
    qc.to_csv(output_dir / "segmentation_qc.csv", index=False)
    return manifest, qc


def segmentation_candidates(reconciled: pd.DataFrame) -> pd.DataFrame:
    """Return active FOVs with a unique, high-confidence metadata match."""
    return reconciled[
        reconciled["image_role"].eq("eight_embryo_fov")
        & ~reconciled["excluded_path"]
        & reconciled["metadata_match_status"].isin(["exact", "fuzzy"])
    ].copy()
