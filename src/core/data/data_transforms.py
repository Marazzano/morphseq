"""Pinned image transforms for the manifest-backed model boundary.

The deterministic boundary is deliberately explicit:

``PIL open -> PIL grayscale -> PIL bilinear resize (antialias=True) -> float32 tensor``.

Contrastive augmentation starts only after the same deterministic PIL resize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional


DEFAULT_TARGET_SIZE = (288, 128)
RESIZE_INTERPOLATION = InterpolationMode.BILINEAR
RESIZE_ANTIALIAS = True


def _validated_target_size(target_size: Sequence[int] | None) -> tuple[int, int]:
    size = DEFAULT_TARGET_SIZE if target_size is None else tuple(target_size)
    if len(size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in size):
        raise ValueError(f"target_size must be integer (height, width), got {target_size!r}.")
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size dimensions must be positive, got {size!r}.")
    return height, width


def _prepare_resized_grayscale(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Manifest image transforms require PIL.Image input, got {type(image)!r}.")
    grayscale = image.convert("L")
    return tv_functional.resize(
        grayscale,
        list(target_size),
        interpolation=RESIZE_INTERPOLATION,
        antialias=RESIZE_ANTIALIAS,
    )


@dataclass(frozen=True)
class DeterministicImageTransform:
    """Convert a PIL image to a pinned single-channel model tensor."""

    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = _prepare_resized_grayscale(image, self.target_size)
        tensor = tv_functional.to_tensor(resized)
        return tensor.to(dtype=torch.float32)


class ContrastiveImageTransform:
    """Apply one independently sampled augmentation after deterministic resize."""

    def __init__(
        self,
        target_size: tuple[int, int],
        *,
        brightness: float = 0.3,
    ) -> None:
        self.target_size = target_size
        color_jitter = transforms.ColorJitter(brightness=brightness)
        self.augmentation = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=15,
                    scale=(0.7, 1.3),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = _prepare_resized_grayscale(image, self.target_size)
        tensor = self.augmentation(resized)
        return tensor.to(dtype=torch.float32)


def basic_transform(target_size: Sequence[int] | None = None) -> Callable[[Image.Image], torch.Tensor]:
    """Build the deterministic grayscale/resize/tensor transform."""

    return DeterministicImageTransform(_validated_target_size(target_size))


def contrastive_transform(
    target_size: Sequence[int] | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """Build the current compatibility augmentation at a configured output size."""

    return ContrastiveImageTransform(_validated_target_size(target_size))
