"""Build, validate, atomically write, and merge snip-rendering sidecars."""

from __future__ import annotations

import fcntl
import inspect
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage
import skimage.exposure

from data_pipeline.object_extraction.snip_processing.inventory_contract import (
    SNIP_RENDERING_CONTRACT_VERSION,
)


# "git" is DELIBERATELY NOT a merge gate (dropped 2026-08-28). It is still captured in every
# sidecar for provenance -- it just no longer has to AGREE across the wells being merged.
#
# _git_identity() records `git rev-parse HEAD` for the WHOLE REPOSITORY, which has nothing to do
# with how any pixel was produced. A README edit, a notebook, a docs commit -- HEAD moves, and every
# snip rendered after it carries a different hash. A 96-well render takes long enough to straddle a
# commit, so an unrelated commit mid-run made merge_snip_inventory refuse the experiment outright.
# That happened to the SeaHub corpus on 2026-08-28: one shard held three commits across nine
# minutes of rendering, and merging died on a difference no pixel could observe.
#
# What the gate is FOR -- refusing to stack snips made under different rendering contracts -- is
# preserved by the two keys that remain. `contract_version` pins the sidecar schema and `software`
# pins the libraries that actually touch pixels; per-product rendering parameters are compared
# separately below via _fixed_product. HEAD was only ever a proxy for those, and a bad one.
_GLOBAL_CONTRACT_KEYS = ("contract_version", "software")


def _git_identity() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Cannot capture the git commit and branch required by the snip rendering contract."
        ) from exc
    return {
        "commit": commit,
        "branch": branch or "DETACHED",
        "detached_head": not bool(branch),
    }


def _json_default(value: Any) -> Any:
    if value is inspect.Parameter.empty:
        return "required"
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return repr(value)


def _call_defaults(callable_object: Any) -> dict[str, Any]:
    return {
        name: _json_default(parameter.default)
        for name, parameter in inspect.signature(callable_object).parameters.items()
        if name != "image"
    }


def build_rendering_contract(
    *,
    snip_product_key: str,
    source_image_product_key: str,
    snip_recipe: str,
    target_micrometers_per_pixel: float,
    blend_radius_micrometers: float,
    blend_radius_applied: bool,
    frame_shape: tuple[int, int],
    mask_contract: dict[str, Any],
    orientation_contract: dict[str, Any],
    clahe_enabled: bool,
    background_contract: dict[str, Any],
    resampling_contract: dict[str, Any],
    file_encoding: dict[str, Any],
) -> dict[str, Any]:
    """Return a one-product contract ready to merge into a run sidecar.

    The explicit product map is required: one experiment run now emits several
    products whose recipe, output dtype, source calibration observations, and
    CLAHE/background applicability can differ.
    """

    height_px, width_px = (int(frame_shape[0]), int(frame_shape[1]))
    return {
        "contract_version": SNIP_RENDERING_CONTRACT_VERSION,
        "git": _git_identity(),
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
            "opencv": cv2.__version__,
        },
        "products": {
            str(snip_product_key): {
                "source_image_product_key": str(source_image_product_key),
                "snip_recipe": str(snip_recipe),
                "rendering": {
                    "target_micrometers_per_pixel": float(
                        target_micrometers_per_pixel
                    ),
                    "blend_radius_micrometers": {
                        "value": float(blend_radius_micrometers),
                        "applied_to_pixels": bool(blend_radius_applied),
                    },
                    "mask": deepcopy(mask_contract),
                    "orientation": deepcopy(orientation_contract),
                    "clahe": {
                        "enabled": bool(clahe_enabled),
                        "implementation": (
                            "skimage.exposure.equalize_adapthist"
                            if clahe_enabled
                            else None
                        ),
                        "call_overrides": {},
                        "resolved_library_defaults": (
                            _call_defaults(skimage.exposure.equalize_adapthist)
                            if clahe_enabled
                            else None
                        ),
                    },
                    "background": deepcopy(background_contract),
                    "frame_shape": {
                        "height_px": height_px,
                        "width_px": width_px,
                    },
                    "resampling": deepcopy(resampling_contract),
                    "file_encoding": deepcopy(file_encoding),
                },
                "observations": {"wells": {}},
            }
        },
    }


def observed_mask_sources(
    rows: Any,
    *,
    source_column: str,
    version_column: str,
) -> list[dict[str, str]]:
    """Return stable unique mask source/version pairs used by rendered rows."""

    if len(rows) == 0:
        return []
    missing = [
        column for column in (source_column, version_column) if column not in rows
    ]
    if missing:
        raise ValueError(
            "Cannot capture mask source/version for snip provenance; "
            f"missing columns: {missing}"
        )
    records = {
        (str(source), str(version))
        for source, version in zip(rows[source_column], rows[version_column])
        if not (pd.isna(source) or pd.isna(version))
    }
    return [
        {"source": source, "version": version} for source, version in sorted(records)
    ]


def rendering_sidecar_path(inventory_path: Path) -> Path:
    inventory_path = Path(inventory_path)
    return inventory_path.with_name(inventory_path.name + ".provenance.json")


# Compatibility name used by the pre-rewrite commit. The current layout puts a
# product-keyed inventory under each well, so no experiment-path reconstruction
# is correct here; the sidecar is always adjacent to the inventory it describes.
def inventory_run_sidecar_path(
    output_csv: Path, *, experiment_id: str | None = None
) -> Path:
    del experiment_id
    return rendering_sidecar_path(output_csv)


def _unique_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded = {
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False): record
        for record in records
    }
    return [encoded[key] for key in sorted(encoded)]


def _with_product_aggregates(document: dict[str, Any]) -> dict[str, Any]:
    for product in document.get("products", {}).values():
        observations = product.setdefault("observations", {})
        wells = observations.setdefault("wells", {})
        observations["mask_sources_and_versions"] = _unique_records(
            source
            for observation in wells.values()
            for source in observation.get("mask_sources_and_versions", [])
        )
        observations["source_micrometers_per_pixel_values"] = sorted(
            {
                float(value)
                for observation in wells.values()
                for value in observation.get(
                    "source_micrometers_per_pixel_values", []
                )
            }
        )
        observations["pixel_dtypes"] = sorted(
            {
                str(value)
                for observation in wells.values()
                for value in observation.get("pixel_dtypes", [])
            }
        )
    return document


def _read_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable snip rendering sidecar: {path}") from exc
    if document.get("contract_version") != SNIP_RENDERING_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported snip rendering contract version "
            f"{document.get('contract_version')!r} in {path}."
        )
    if not isinstance(document.get("products"), dict):
        raise ValueError(f"{path}: rendering sidecar has no product-keyed 'products' object.")
    return document


def _fixed_product(product: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in product.items() if key != "observations"}


def _merge_contract_document(
    document: dict[str, Any],
    incoming: dict[str, Any],
    *,
    source_label: str,
) -> dict[str, Any]:
    for key in _GLOBAL_CONTRACT_KEYS:
        if document.get(key) != incoming.get(key):
            raise ValueError(
                f"Snip rendering sidecars disagree on {key!r}; offending source: "
                f"{source_label}"
            )

    products = document.setdefault("products", {})
    for product_key, incoming_product in incoming["products"].items():
        existing = products.get(product_key)
        if existing is None:
            products[product_key] = deepcopy(incoming_product)
            continue
        if _fixed_product(existing) != _fixed_product(incoming_product):
            raise ValueError(
                "Refusing to mix rendering contracts for "
                f"snip_product_key={product_key!r}; offending source: {source_label}"
            )
        wells = existing.setdefault("observations", {}).setdefault("wells", {})
        incoming_wells = (
            incoming_product.get("observations", {}).get("wells", {})
        )
        for well_id, observation in incoming_wells.items():
            present = wells.get(str(well_id))
            if present is not None and present != observation:
                raise ValueError(
                    "Conflicting rendering observations for "
                    f"snip_product_key={product_key!r}, well_id={well_id!r}; "
                    f"offending source: {source_label}"
                )
            wells[str(well_id)] = deepcopy(observation)
    return _with_product_aggregates(document)


def update_rendering_sidecar(
    path: Path,
    *,
    fixed_contract: dict[str, Any],
    snip_product_key: str,
    well_observations: dict[str, dict[str, Any]],
) -> Path:
    """Create/update one sidecar safely across product and per-well writers."""

    path = Path(path)
    products = fixed_contract.get("products", {})
    if set(products) != {str(snip_product_key)}:
        raise ValueError(
            "update_rendering_sidecar requires a one-product fixed contract matching "
            f"snip_product_key={snip_product_key!r}; found {sorted(products)}."
        )
    incoming = deepcopy(fixed_contract)
    incoming["products"][str(snip_product_key)]["observations"] = {
        "wells": {str(key): deepcopy(value) for key, value in well_observations.items()}
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                document = _read_document(path)
                document = _merge_contract_document(
                    document, incoming, source_label=str(path)
                )
            else:
                document = _with_product_aggregates(incoming)
            payload = json.dumps(
                document, indent=2, sort_keys=True, allow_nan=False
            ) + "\n"
            temp_path.write_text(payload, encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return path


def merge_rendering_sidecars(
    input_paths: Iterable[Path], output_path: Path
) -> Path:
    """Merge product-aware per-well sidecars beside a merged inventory."""

    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("No snip rendering sidecars were provided for merge.")

    document: dict[str, Any] | None = None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing per-well snip rendering sidecar required for merge: {path}"
            )
        incoming = _read_document(path)
        if document is None:
            document = deepcopy(incoming)
        else:
            document = _merge_contract_document(
                document, incoming, source_label=str(path)
            )

    assert document is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(
                _with_product_aggregates(document),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, output_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    return output_path
