"""Snip image → model input tensor transform.

Single public function: ``snip_to_model_input_tensor()``. Returns an unbatched
``[3, H, W]`` float32 tensor with values in ``[0, 1]``.

``encode_snips()`` handles batching — stacking individual tensors into ``[B, 3, H, W]``.
This function is the explicit, config-driven replacement for
``basic_transform(target_size=(288, 128))`` from the legacy pipeline.

Path-pure side-effect of the module: imports PIL and torchvision only. No model loading,
no orchestration imports. Runnable from the 3.9 encode env or the 3.10 main env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image
import torch
import torchvision.transforms.functional as TF

PathLike = Union[str, Path]


def snip_to_model_input_tensor(
    image_path: PathLike,
    model_input_shape: tuple[int, int],
) -> torch.Tensor:
    """Load a snip PNG and return a [3, H, W] float32 tensor in [0, 1].

    ``model_input_shape`` is ``(height, width)`` — the NumPy/torchvision (H, W) convention.
    A shape of ``(288, 128)`` produces a tensor of shape ``[3, 288, 128]``.

    Grayscale and palette PNGs are converted to 3-channel RGB before the resize so the
    model always receives a 3-channel input regardless of source mode.

    Args:
        image_path: Absolute path to the processed snip PNG.
        model_input_shape: ``(height, width)`` — NOT ``(width, height)``.

    Returns:
        ``torch.Tensor`` of shape ``[3, H, W]``, dtype ``float32``, values in ``[0.0, 1.0]``.
    """
    height, width = model_input_shape

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = TF.resize(img, [height, width])
    tensor = TF.to_tensor(img)  # [3, H, W], float32, [0, 1]
    return tensor
