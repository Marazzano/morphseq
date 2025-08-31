from __future__ import annotations

from pathlib import Path

from src.vae.auxiliary_scripts.embed_training_snips import embed_snips


def run_embed(
    root: str | Path,
    train_name: str,
    model_dir: str | Path | None = None,
    out_csv: str | Path | None = None,
    batch_size: int = 64,
    simulate: bool = False,
    latent_dim: int = 16,
    seed: int = 0,
) -> None:
    embed_snips(
        root=root,
        train_name=train_name,
        model_dir=model_dir,
        out_csv=out_csv,
        batch_size=batch_size,
        simulate=simulate,
        latent_dim=latent_dim,
        seed=seed,
    )
    print("✔️  Embedding step complete.")

