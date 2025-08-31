from __future__ import annotations

from pathlib import Path
import numpy as np
import skimage.io as io
import pandas as pd

from src.vae.auxiliary_scripts.embed_training_snips import embed_snips


def _write_jpg(p: Path, h: int = 12, w: int = 16):
    p.parent.mkdir(parents=True, exist_ok=True)
    img = (np.random.rand(h, w) * 255).astype(np.uint8)
    io.imsave(str(p), img, check_contrast=False)


def test_embed_training_snips_simulate(tmp_path: Path):
    root = tmp_path / "proj"
    train_name = "unit_train"
    images_dir = root / "training_data" / train_name / "images" / "0"

    # Two snips
    snips = ["20250612_30hpf_ctrl_atf6_C12_e00_t0000.jpg", "20250612_30hpf_ctrl_atf6_E06_e01_t0001.jpg"]
    for s in snips:
        _write_jpg(images_dir / s)

    out_csv = embed_snips(root=str(root), train_name=train_name, simulate=True, latent_dim=4, seed=42)

    # Assert outputs
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["snip_id", "z_mu_00", "z_mu_01", "z_mu_02", "z_mu_03"]
    assert set(df["snip_id"]) == {p[:-4] for p in snips}
    # No NaNs
    assert not df.isna().any().any()

