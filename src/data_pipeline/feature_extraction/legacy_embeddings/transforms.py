"""Snip image → model input tensor transform.

Single public function: ``snip_to_model_input_tensor()``. Returns an unbatched
``[C, H, W]`` float32 tensor with values in ``[0, 1]``, where C is ``model_input_channels``.

The production model (``20241107_ds_sweep01_optimum``) is a SeqVAE trained on
**grayscale** input: ``model_config.json`` records ``"input_dim": [1, 288, 128]``.
``model_input_channels`` must be ``1`` — passing ``3`` would silently feed
wrong-channel data to an encoder whose first conv expects one channel.

``encode_snips()`` handles batching — stacking individual tensors into ``[B, C, H, W]``.
This function is the explicit, config-driven replacement for
``basic_transform(target_size=(288, 128))`` from the legacy pipeline, which used
``transforms.Grayscale(num_output_channels=1)``.

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
    model_input_channels: int = 1,
) -> torch.Tensor:
    """Load a snip PNG and return a [C, H, W] float32 tensor in [0, 1].

    ``model_input_shape`` is ``(height, width)`` — the NumPy/torchvision (H, W) convention.
    ``model_input_channels`` must match the channel count the model was trained with.
    The production legacy VAE uses ``model_input_channels=1`` (grayscale).

    Args:
        image_path: Absolute path to the processed snip PNG.
        model_input_shape: ``(height, width)`` — NOT ``(width, height)``.
        model_input_channels: Number of output channels. ``1`` = grayscale (default,
            matching the legacy VAE's ``input_dim=(1, 288, 128)``). ``3`` = RGB.

    Returns:
        ``torch.Tensor`` of shape ``[C, H, W]``, dtype ``float32``, values in ``[0.0, 1.0]``.
    """
    if model_input_channels not in (1, 3):
        raise ValueError(
            f"model_input_channels must be 1 (grayscale) or 3 (RGB), got {model_input_channels}."
        )

    height, width = model_input_shape

    img = Image.open(image_path)

    if model_input_channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img = TF.resize(img, [height, width])
    tensor = TF.to_tensor(img)  # [C, H, W], float32, [0, 1]
    return tensor
