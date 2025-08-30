from __future__ import annotations
from pathlib import Path

from src.build.build02B_segment_bf_main import apply_unet


def run_build02(
    root: str | Path,
    mode: str = "skip",
    model_name: str = "mask_v1_0050",
    n_classes: int = 2,
    overwrite: bool = False,
) -> None:
    """Run legacy segmentation, or skip if using SAM2.

    Writes legacy masks under `segmentation/{model}_predictions/{exp}/`.
    """
    if mode == "skip":
        print("ℹ️  build02: skipping legacy segmentation (SAM2 path expected)")
        return
    root = Path(root)
    apply_unet(root=str(root), model_name=model_name, n_classes=n_classes, overwrite_flag=overwrite)
    print("✔️  Build02 (legacy segmentation) complete.")
