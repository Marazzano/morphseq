"""Legacy-VAE encode entrypoint — the Python-3.9 batch body for `latent_embeddings`.

This is the ONE process that crosses into the 3.9 model env. The orchestrating Snakemake
process (3.10) only schedules this job and validates the parquet it writes; no model object
ever crosses the version line (model_input_handoff_contract.md §9).

Batch shape (`execution=RUN_BATCH_WRITES_PER_WELL_SHARDS`): load the encoder ONCE, then iterate
the run wells, writing each well's ``<well_id>_latents.parquet`` before exiting. Model load
(process spawn + checkpoint deserialize) dominates the cost of encoding a few hundred snips, so it
must not be paid per well.

Glue only — every piece of real work lives in a tested sibling module:
  resolve_legacy_model_dir (model_paths) → load_legacy_vae_encoder (legacy_vae_inference_loader)
  → collect_snip_inputs (snip_source) → encode_snips (encode) → validate_latent_embeddings (contract).

Inputs/outputs are paired positionally: ``--snip-inventory-csv`` i pairs with ``--output-parquet`` i,
one pair per run well. The Snakemake rule lists the run set; this file does not discover wells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.feature_extraction.legacy_embeddings.contract import (
    EMBEDDING_MODEL_NAME_COL,
    validate_latent_embeddings,
)
from data_pipeline.feature_extraction.legacy_embeddings.encode import encode_snips
from data_pipeline.feature_extraction.legacy_embeddings.legacy_vae_inference_loader import (
    load_legacy_vae_encoder,
)
from data_pipeline.feature_extraction.legacy_embeddings.model_paths import resolve_legacy_model_dir
from data_pipeline.feature_extraction.legacy_embeddings.snip_source import collect_snip_inputs
from data_pipeline.model_servers.atomic_write import atomic_write_bytes, atomic_write_via
from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PATH_MODE_PER_WELL,
    artifact_path,
)


def _derive_output_parquets(
    *,
    snip_inventory_csvs: list[Path],
    output_root: Path,
    experiment_id: str,
) -> list[Path]:
    """Resolve canonical per-well latent paths from checkpoint-resolved inventory paths."""
    return [
        artifact_path(
            output_root,
            "latent_embeddings",
            "latents",
            experiment_id,
            path_mode=PATH_MODE_PER_WELL,
            well_id=inventory_path.parent.name,
        )
        for inventory_path in snip_inventory_csvs
    ]


def run_legacy_embeddings(
    *,
    snip_inventory_csvs: list[Path],
    output_parquets: list[Path],
    output_root: Path,
    models_root: Path,
    model_name: str,
    model_input_shape: tuple[int, int],
    model_input_channels: int = 1,
    batch_size: int = 64,
    device: str = "cpu",
    completion_flag: Path | None = None,
) -> None:
    """Load the legacy VAE once, encode each well's snips, write one latents parquet per well.

    Args:
        snip_inventory_csvs: per-well snip_inventory CSVs (the run set), in well order.
        output_parquets: per-well output paths, paired positionally with ``snip_inventory_csvs``.
        output_root: the pipeline output root (``data_pipeline_output``) that each
            manifest's ``processed_snip_path`` is stored relative to.
        models_root: machine path holding ``legacy/<model_name>/`` (env.yaml.paths.models_root,
            or the config ``models_root_override`` if set — resolved by the caller).
        model_name: which trained model to load (the science knob; names the weights dir).
        model_input_shape: ``(height, width)`` the model expects.
        model_input_channels: channel count the model expects (1 = grayscale).
        batch_size: snips per encoder forward pass.
        device: torch device string.
        completion_flag: optional experiment-level marker written only after every
            per-well parquet and validation sentinel has been finalized.
    """
    if len(snip_inventory_csvs) != len(output_parquets):
        raise ValueError(
            "run_legacy_embeddings: --snip-inventory-csv and --output-parquet counts must match "
            f"(got {len(snip_inventory_csvs)} inventories vs {len(output_parquets)} outputs); "
            "the rule pairs them positionally per well."
        )

    # Invalidate the experiment gate and every per-well eligibility marker before doing work.
    # If this attempt fails, neither an old completion flag nor an untouched later-well sentinel
    # may make stale data look current. Existing parquet payloads remain recoverable but ineligible.
    if completion_flag is not None:
        completion_flag.unlink(missing_ok=True)
    for output_parquet in output_parquets:
        output_parquet.with_name(output_parquet.name + ".validated").unlink(missing_ok=True)

    # Load once — the whole reason this is a batch step.
    model_dir = resolve_legacy_model_dir(models_root, model_name)
    encoder = load_legacy_vae_encoder(model_dir, device=device)

    for snip_inventory_csv, output_parquet in zip(snip_inventory_csvs, output_parquets):
        validated_flag = output_parquet.with_name(output_parquet.name + ".validated")
        snip_inputs = collect_snip_inputs(snip_inventory_csv, output_root=output_root)
        latents = encode_snips(
            snip_inputs,
            encoder=encoder,
            model_input_shape=model_input_shape,
            model_input_channels=model_input_channels,
            batch_size=batch_size,
            device=device,
        )
        # Stamp provenance: latents without a model name are unlabeled vials (contract.py).
        latents[EMBEDDING_MODEL_NAME_COL] = model_name
        validate_latent_embeddings(latents, source=str(output_parquet))
        atomic_write_via(
            output_parquet,
            lambda tmp_path: latents.to_parquet(tmp_path, index=False),
        )
        atomic_write_bytes(validated_flag, b"ok\n")

    if completion_flag is not None:
        atomic_write_bytes(completion_flag, b"ok\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snip-inventory-csv", type=Path, nargs="+", required=True)
    p.add_argument(
        "--output-parquet",
        type=Path,
        nargs="+",
        help="Explicit per-well outputs. When omitted, --experiment-id is required and "
             "canonical paths are derived from each inventory's parent well_id.",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment-id")
    p.add_argument("--models-root", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    # model_input_shape is (height, width) — two ints, in that order.
    p.add_argument("--model-input-height", type=int, required=True)
    p.add_argument("--model-input-width", type=int, required=True)
    p.add_argument("--model-input-channels", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--completion-flag", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.output_parquet and not args.experiment_id:
        raise ValueError(
            "--experiment-id is required when --output-parquet is omitted"
        )
    output_parquets = (
        list(args.output_parquet)
        if args.output_parquet
        else _derive_output_parquets(
            snip_inventory_csvs=list(args.snip_inventory_csv),
            output_root=args.output_root,
            experiment_id=str(args.experiment_id or ""),
        )
    )
    run_legacy_embeddings(
        snip_inventory_csvs=list(args.snip_inventory_csv),
        output_parquets=output_parquets,
        output_root=args.output_root,
        models_root=args.models_root,
        model_name=args.model_name,
        model_input_shape=(args.model_input_height, args.model_input_width),
        model_input_channels=args.model_input_channels,
        batch_size=args.batch_size,
        device=args.device,
        completion_flag=args.completion_flag,
    )


if __name__ == "__main__":
    main()
