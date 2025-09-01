from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from src.vae.auxiliary_scripts.embed_training_snips import embed_snips


def run_build06(
    root: str | Path,
    train_name: str,
    simulate: bool = False,
    latent_dim: int = 16,
    seed: int = 0,
    model_dir: Optional[str | Path] = None,
    batch_size: int = 64,
    join_df02: bool = True,
) -> None:
    """Standardized embeddings generation and optional merge with df02 (df03 output).

    - Generates `training_data/<train_name>/embeddings.csv` using either a pretrained
      VAE (`model_dir`) or simulate mode for CI/wiring.
    - Writes `training_data/<train_name>/embryo_metadata_with_embeddings.csv` by joining
      the train metadata with embeddings on `snip_id`.
    - If `join_df02=True`, merges embeddings into df02 across the entire root and writes
      `metadata/combined_metadata_files/embryo_metadata_df03.csv` (df02 + z_mu_* columns
      where available via snip_id match).
    """
    root = Path(root)
    train_root = root / "training_data" / train_name

    # 1) Generate embeddings
    embed_csv = embed_snips(
        root=root,
        train_name=train_name,
        model_dir=model_dir,
        out_csv=None,
        batch_size=batch_size,
        simulate=simulate,
        latent_dim=latent_dim,
        seed=seed,
    )

    # 2) Join with train metadata if available
    train_meta = train_root / "embryo_metadata_df_train.csv"
    if train_meta.exists():
        df_train = pd.read_csv(train_meta)
        df_emb = pd.read_csv(embed_csv)
        df_join = df_train.merge(df_emb, how="left", on="snip_id")
        out_train_join = train_root / "embryo_metadata_with_embeddings.csv"
        df_join.to_csv(out_train_join, index=False)
        print(f"✔️  Wrote {out_train_join}")

    # 3) Optionally merge into df02 for full root
    if join_df02:
        df02_path = root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
        if df02_path.exists():
            df02 = pd.read_csv(df02_path)
            df_emb = pd.read_csv(embed_csv)
            df03 = df02.merge(df_emb, how="left", on="snip_id")
            df03_path = df02_path.with_name("embryo_metadata_df03.csv")
            df03.to_csv(df03_path, index=False)
            print(f"✔️  Wrote {df03_path}")
        else:
            print(f"ℹ️  Skipping df03 merge: {df02_path} not found")

    print("✔️  Build06 complete.")

