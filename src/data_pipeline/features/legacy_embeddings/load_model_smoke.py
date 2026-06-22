"""Load-smoke for the legacy VAE — proves the model world's engine before any rule exists.

WHY this exists (read first):
    The legacy VAE encoder is Python-3.9-pinned. The embeddings stage runs its whole
    command body in Python 3.9; the main pipeline (3.10) only schedules the job and
    validates the output file. **Only files cross the env boundary — no model object
    ever does.** (Doctrine + full design: specs/model_input_handoff_contract.md §9.)

    This script is the smallest possible proof of that engine: load the model
    in-process in 3.9 and report its metadata. It is NOT the encode step — it reads
    no snips and writes no latents. It exists so we can confirm "the 3.9 env + the
    staged weights + the loader actually work" before wiring the real embeddings rule.

    >>> Spec names the border. Smoke proves the engine. Rules come when the product exists.

DO NOT create ``load_model_subprocess.py``. That file (referenced but never written
at services/legacy_model_utils.py:201) belongs to an abandoned design that moved a
*model object* across the 3.9/3.10 boundary. Under the file-boundary doctrine it is
the wrong shape. This in-process 3.9 load is its replacement.

HOW to run (prefer the direct executable from env.yaml.runtime.model_python_executable):

    <model_python_executable> -m data_pipeline.features.legacy_embeddings.load_model_smoke \
        --models-root <models_root> --model-name 20241107_ds_sweep01_optimum

    # fallback, if no direct executable is configured (env.yaml.runtime.model_python_env):
    conda run -n mseq_pipeline_py3.9 --no-capture-output \
        python -m data_pipeline.features.legacy_embeddings.load_model_smoke \
        --models-root <models_root> --model-name 20241107_ds_sweep01_optimum

Exit codes: 0 = model loaded + metadata printed; non-zero (with a loud message
naming the missing path) otherwise.
"""

from __future__ import annotations

import argparse
import sys

from data_pipeline.features.legacy_embeddings.model_paths import resolve_legacy_model_dir


def _model_metadata(lit_model) -> dict:
    """Best-effort read of the three identity fields, tolerant of model variants.

    ``model_name`` / ``latent_dim`` / ``nuisance_indices`` are present on the
    ScriptedLegacyModelAdapter and on most AutoModel variants, but not guaranteed —
    fall back to ``getattr`` with a sentinel so the smoke reports what it can rather
    than crashing on a missing attribute.
    """
    nuisance = getattr(lit_model, "nuisance_indices", None)
    if hasattr(nuisance, "tolist"):
        nuisance = nuisance.tolist()
    return {
        "model_name": getattr(lit_model, "model_name", None),
        "latent_dim": getattr(lit_model, "latent_dim", None),
        "nuisance_indices": nuisance,
    }


def load_smoke(models_root: str, model_name: str, device: str = "cpu") -> int:
    """Resolve + load the legacy model in-process and print its metadata.

    Runs wholly in Python 3.9. Returns a process exit code.
    """
    # Loud guard: this MUST be the 3.9 interpreter. If a 3.10 caller runs it directly,
    # the legacy pickles will fail cryptically — name the real problem instead.
    if sys.version_info[:2] != (3, 9):
        print(
            f"ERROR: load_model_smoke must run under Python 3.9 (the legacy VAE is "
            f"3.9-pinned); got {sys.version_info[0]}.{sys.version_info[1]}. Invoke via "
            f"env.yaml.runtime.model_python_executable, or "
            f"`conda run -n mseq_pipeline_py3.9 python -m "
            f"data_pipeline.features.legacy_embeddings.load_model_smoke ...`.",
            file=sys.stderr,
        )
        return 2

    # resolve_legacy_model_dir fails loud naming the missing path — let it propagate.
    model_dir = resolve_legacy_model_dir(models_root, model_name)
    print(f"Resolved legacy model dir: {model_dir}")

    # Import the loader only now (inside 3.9) — keeps the module importable from 3.10
    # for the unit tests that exercise resolve_legacy_model_dir without torch present.
    from src.legacy.vae import AutoModel  # noqa: E402  (3.9-only, lazy on purpose)

    lit_model = AutoModel.load_from_folder(str(model_dir))
    lit_model.to(device)
    lit_model.eval()

    meta = _model_metadata(lit_model)
    print("✅ Legacy model loaded in-process (Python 3.9). Metadata:")
    print(f"    model_name       = {meta['model_name']}")
    print(f"    latent_dim       = {meta['latent_dim']}")
    print(f"    nuisance_indices = {meta['nuisance_indices']}")
    print("(No model object crosses to Python 3.10 — only files do.)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load-smoke the Python-3.9 legacy VAE and report its metadata."
    )
    parser.add_argument(
        "--models-root", required=True,
        help="env.yaml.paths.models_root — the legacy tree lives at <models_root>/legacy/.",
    )
    parser.add_argument(
        "--model-name", required=True,
        help="Legacy model name, e.g. 20241107_ds_sweep01_optimum.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to place the model on for the smoke (default: cpu).",
    )
    args = parser.parse_args(argv)
    return load_smoke(args.models_root, args.model_name, args.device)


if __name__ == "__main__":
    sys.exit(main())
