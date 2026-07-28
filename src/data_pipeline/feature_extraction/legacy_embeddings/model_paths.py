"""Legacy VAE model-directory resolution — the model-domain path helper.

This is the model-world sibling of orchestration ``paths.py``, NOT a second copy of
it. Orchestration ``paths.py`` knows where *pipeline artifacts* live (it owns the
``PIPELINE_STEPS`` registry); this file knows the one odd thing it must not:
**where the legacy VAE weights sit on disk and the quirks of that layout** (the
``legacy/<model_name>`` convention and the tolerated nested ``final_model/``).
Keeping that here means orchestration never learns legacy-model folder oddities.

Path-pure: it constructs/inspects the model directory and fails loud; it imports
no orchestration code. ``models_root`` is passed in explicitly (it comes from
``env.yaml.paths.models_root`` at the call site) — there is no ``PROJECT_ROOT``
fallback.

Layout (what ``AutoModel.load_from_folder`` expects)::

    <models_root>/legacy/<model_name>/
        model_config.json        # (or *config*.json)
        model.pt
        encoder.pkl  decoder.pkl # optional, custom modules

    # tolerated variant — the config + weights live one level down:
    <models_root>/legacy/<model_name>/final_model/
        *config*.json ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Controlled vocabulary — the legacy on-disk layout, defined once
# ─────────────────────────────────────────────────────────────────────────────────────────────

LEGACY_SUBDIR = "legacy"          # <models_root>/legacy/<model_name>/
FINAL_MODEL_SUBDIR = "final_model"  # tolerated nested home for config + weights
CONFIG_GLOB = "*config*.json"     # model_config.json (AutoModel.load_from_folder reads this)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Resolver — construct + validate the legacy model directory
# ─────────────────────────────────────────────────────────────────────────────────────────────

def legacy_models_dir(models_root: PathLike) -> Path:
    """The directory that holds all legacy models: ``<models_root>/legacy``.

    Pure path construction — does not check that it exists.
    """
    return Path(models_root) / LEGACY_SUBDIR


def resolve_legacy_model_dir(models_root: PathLike, model_name: str) -> Path:
    """Resolve the on-disk directory for a named legacy VAE model.

    Returns the directory ``AutoModel.load_from_folder`` should be pointed at:
    ``<models_root>/legacy/<model_name>`` — or its ``final_model/`` child when the
    config + weights live one level down.

    Fail-loud: if the model directory does not exist, raise ``FileNotFoundError``
    naming the exact path that is missing AND what to do about it (the honest
    current state is that no ``legacy/`` weights are staged under ``models_root``
    yet). A missing *config file* (when the directory itself exists) is only a
    warning — some exported models carry no JSON config.

    Path-pure aside from existence checks at this consume boundary; imports no
    orchestration code.

    Args:
        models_root: the models root (``env.yaml.paths.models_root``). The legacy
            tree lives at ``<models_root>/legacy/``.
        model_name: e.g. ``"20241107_ds_sweep01_optimum"``.

    Returns:
        The validated model directory to hand to the loader.

    Raises:
        FileNotFoundError: if ``<models_root>/legacy/<model_name>`` does not exist.
    """
    model_dir = legacy_models_dir(models_root) / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Legacy model directory not found: {model_dir}. "
            f"Stage the model weights at <models_root>/{LEGACY_SUBDIR}/{model_name}/ "
            f"(model_config.json + model.pt, the layout AutoModel.load_from_folder "
            f"expects), or pass a model_name that is already staged under "
            f"{legacy_models_dir(models_root)}."
        )

    # Prefer the directory that actually carries the config. If the top level has
    # none but a nested final_model/ does, point the loader at the nested dir.
    if not list(model_dir.glob(CONFIG_GLOB)):
        nested = model_dir / FINAL_MODEL_SUBDIR
        if nested.exists() and list(nested.glob(CONFIG_GLOB)):
            return nested
        logger.warning(
            "No %s found in %s (or its %s/ child); returning the directory anyway "
            "— some exported legacy models carry no JSON config.",
            CONFIG_GLOB, model_dir, FINAL_MODEL_SUBDIR,
        )

    return model_dir
