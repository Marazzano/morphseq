"""Pinned image transforms for the manifest-backed model boundary.

The deterministic boundary is deliberately singular:

``PIL image -> grayscale L -> PIL resize (H, W) -> float32 CHW tensor [0, 1]``

Resize uses torchvision 0.20's ``InterpolationMode.BILINEAR`` with
``antialias=True``. Augmentations, when requested, run on the already resized
float tensor. Calling :class:`ContrastiveTransform` twice therefore produces two
independent views without changing the deterministic size boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DEFAULT_TARGET_SIZE = (288, 128)
RESIZE_INTERPOLATION = InterpolationMode.BILINEAR
RESIZE_ANTIALIAS = True


def _validate_target_size(target_size: Sequence[int] | None) -> tuple[int, int]:
    if target_size is None:
        return DEFAULT_TARGET_SIZE
    if len(target_size) != 2:
        raise ValueError(f"target_size must be (height, width), got {target_size!r}.")
    height, width = (int(target_size[0]), int(target_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(
            f"target_size values must be positive, got {(height, width)!r}."
        )
    return height, width


@dataclass(frozen=True)
class DeterministicImageTransform:
    """Convert one decoded PIL image to the configured model-input tensor."""

    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_size", _validate_target_size(self.target_size))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"DeterministicImageTransform expects PIL.Image.Image, got {type(image)!r}."
            )
        grayscale = image.convert("L")
        resized = TF.resize(
            grayscale,
            list(self.target_size),
            interpolation=RESIZE_INTERPOLATION,
            antialias=RESIZE_ANTIALIAS,
        )
        tensor = TF.pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
        return tensor


class ContrastiveTransform:
    """Apply stochastic tensor augmentation after the deterministic boundary."""

    def __init__(
        self,
        target_size: Sequence[int] | None = None,
        *,
        augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.boundary = DeterministicImageTransform(_validate_target_size(target_size))
        self.augmentation = augmentation or transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=15,
                    scale=(0.7, 1.3),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.3)], p=0.8),
            ]
        )

    @property
    def target_size(self) -> tuple[int, int]:
        return self.boundary.target_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = self.boundary(image)
        augmented = self.augmentation(tensor)
        if not isinstance(augmented, torch.Tensor):
            raise TypeError(
                f"Contrastive augmentation must return torch.Tensor, got {type(augmented)!r}."
            )
        return augmented.to(dtype=torch.float32)


def basic_transform(
    target_size: Sequence[int] | None = None,
) -> DeterministicImageTransform:
    """Build the deterministic grayscale transform (default ``[1, 288, 128]``)."""

    return DeterministicImageTransform(_validate_target_size(target_size))


def contrastive_transform(
    target_size: Sequence[int] | None = None,
) -> ContrastiveTransform:
    """Build a size-aware stochastic view transform.

    ``target_size`` is intentionally forwarded to the deterministic boundary; this
    fixes the former implementation that accepted and ignored the argument.
    """

    return ContrastiveTransform(target_size=target_size)
