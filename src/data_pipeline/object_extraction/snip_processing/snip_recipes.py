"""Snip recipes: what happens to pixels AFTER geometry.

A recipe is the PHOTOMETRIC half of a snip product. Geometry (crop, rotation, placement) is shared
by every product of an embryo-time and lives in ``snip_transform``; a recipe decides only what the
values become.

    no_change    geometry only; source dtype and scale preserved
    clahe_blend  CLAHE + synthetic-noise blend + uint8   (the historical BF path, named honestly)

Named for WHAT THEY DO TO PIXELS, not what they are for. "clahe_blend" rather than "bf" is what lets
a third recipe be added without renegotiating vocabulary — and it makes it obvious at every call site
that the historical BF snip is a display product, not a measurement.

WHY `no_change` IS THE ONLY RECIPE THAT CAN BE QUANTITATIVE. A recipe runs per snip inside a per-well
job, so it sees exactly one embryo's pixels. Every quantitative normalization worth having — log1p,
asinh with a fixed scale, MAD scaling, background-mode anchoring — is FITTED FROM A POPULATION: a
pooled negative mode, a reference set, an experiment's background distribution. None of that is
visible at snip grain. Normalization is therefore architecturally impossible as a recipe, not merely
undesirable, and everything quantitative downstream is downstream of a fit.

`no_change` IS NOT `raw`. In this codebase ``raw_*`` always means *from acquisition*
(``raw_image_source_path``, ``raw_channel_name``, ``raw_time_index``). This is a read of the
materialized product, so calling it "raw" would imply a second read of the ND2.

AND `no_change` DOES NOT MEAN "PIXELS ARE IDENTICAL". It means no PHOTOMETRIC change: no
normalization, no contrast transform, no intensity scaling, no dtype conversion, no synthetic
background. Geometric resampling MAY still change individual values — a rotated or rescaled output
pixel is an interpolated blend of source pixels. That is a real measurement effect, not a loophole,
which is why quantitative work measures on the native raster (see
``object_extraction/channel_intensity``) rather than on a rendered snip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_CHANGE = "no_change"
CLAHE_BLEND = "clahe_blend"


class SnipRecipeError(ValueError):
    """A recipe was requested that cannot run, or was run against an input it cannot accept."""


@dataclass(frozen=True)
class SnipRecipeContract:
    """What a recipe requires of its input and promises about its output.

    THE DTYPE REQUIREMENT IS THE POINT. The read boundary used to treat dtype as a proxy for recipe —
    uint8 was assumed to be a display frame and everything else was silently autoscaled into one,
    which erases relative intensity. Making each recipe state its own precondition is what replaces
    that inference with a declaration.
    """

    #: dtypes this recipe accepts, or None for "any". A recipe that rejects an input must say so
    #: rather than coercing it: coercion is exactly the destructive behavior being removed.
    required_dtypes: tuple[str, ...] | None
    #: True when the recipe alters pixel VALUES beyond geometric resampling. A recipe that does is
    #: unusable for cross-embryo quantification, and this flag is what lets that be asserted rather
    #: than remembered.
    is_photometric: bool
    summary: str


SNIP_RECIPE_CONTRACTS: dict[str, SnipRecipeContract] = {
    NO_CHANGE: SnipRecipeContract(
        required_dtypes=None,
        is_photometric=False,
        summary="Geometry only. Source dtype and intensity scale preserved.",
    ),
    CLAHE_BLEND: SnipRecipeContract(
        # apply_clahe documents uint8 in / uint8 out, and the noise blend is tuned for 8-bit BF.
        required_dtypes=("uint8",),
        is_photometric=True,
        summary="CLAHE + synthetic background noise blend, uint8 out. Display/model input only.",
    ),
}

#: The closed vocabulary, DERIVED from the contract table rather than restated beside it.
#:
#: No SUPPORTED/IMPLEMENTED pair here, deliberately. `materialization_plan.py` has that split for a
#: real reason — a config may legally NAME `projection_method: mean` while no primitive implements
#: it, so grammar is genuinely broader than capability and plan validation must fail with a good
#: message rather than a layer deeper. Here a recipe named in config that has no contract entry is
#: simply unknown, and there is no state where a recipe is legal-to-name but unimplemented. Two sets
#: plus an assert forcing them equal would be one set written twice, and that duplication is itself
#: the drift mechanism the assert exists to prevent.
SUPPORTED_SNIP_RECIPES = frozenset(SNIP_RECIPE_CONTRACTS)


def recipe_contract(snip_recipe: str) -> SnipRecipeContract:
    """The contract for one recipe, or a loud failure naming what is available."""
    try:
        return SNIP_RECIPE_CONTRACTS[str(snip_recipe)]
    except KeyError:
        raise SnipRecipeError(
            f"unknown snip_recipe {snip_recipe!r}. Supported: "
            f"{sorted(SUPPORTED_SNIP_RECIPES)}."
        ) from None


def assert_source_dtype_is_acceptable(image: np.ndarray, *, snip_recipe: str, source: str) -> None:
    """Fail loudly when a source frame's dtype is not what the recipe requires.

    REFUSING IS THE WHOLE POINT. The removed behavior coerced any non-uint8 frame with
    ``rescale_intensity(in_range="image")`` — per-frame min/max autoscaling, so each frame got its
    own affine map and a 2-copy embryo in one frame became indistinguishable from a 0-copy embryo in
    another. A loud failure is recoverable; a silently rescaled uint16 frame is not.
    """
    contract = recipe_contract(snip_recipe)
    if contract.required_dtypes is None:
        return
    actual = str(np.asarray(image).dtype)
    if actual not in contract.required_dtypes:
        raise SnipRecipeError(
            f"snip_recipe {snip_recipe!r} requires dtype in {list(contract.required_dtypes)}; got "
            f"{actual} from {source}. Render this source through a dtype-preserving recipe "
            f"({NO_CHANGE!r}) instead — coercing it here would destroy absolute intensity and with "
            "it any cross-embryo comparison."
        )


def assert_recipe_is_quantitative(snip_recipe: str) -> None:
    """Guard for callers that intend to MEASURE the result.

    Intensity numbers off a ``clahe_blend`` snip are plausible and meaningless: CLAHE is *local
    adaptive* equalization, so it destroys both absolute intensity and cross-embryo comparability
    while leaving an image that looks fine.
    """
    contract = recipe_contract(snip_recipe)
    if contract.is_photometric:
        raise SnipRecipeError(
            f"snip_recipe {snip_recipe!r} is photometric ({contract.summary}), so its pixels do not "
            "carry comparable intensity. Measure a non-photometric product instead."
        )


def render_no_change(image: np.ndarray, **_ignored) -> np.ndarray:
    """Identity. The geometry already happened; this recipe adds nothing.

    Returned as-is rather than copied or cast: a cast here would be exactly the silent dtype
    conversion the recipe promises not to do, and it would be invisible in a diff.
    """
    return image


def render_clahe_blend(
    image: np.ndarray,
    *,
    mask: np.ndarray,
    background_mean: float,
    background_std: float,
    blend_radius_um: float,
    pixel_size_um: float,
    **_ignored,
) -> np.ndarray:
    """The historical BF path: CLAHE, then blend against synthetic background noise.

    Photometric and proud of it -- this product is a model input, not a measurement. Imported
    lazily so the recipe registry does not drag skimage's CLAHE into every module that merely wants
    to KNOW what recipes exist.
    """
    from data_pipeline.object_extraction.snip_processing.augmentation import augment_snip

    augmented, _clahe_only = augment_snip(
        image,
        mask,
        background_mean,
        background_std,
        blend_radius_um=float(blend_radius_um),
        pixel_size_um=float(pixel_size_um),
    )
    return augmented


#: THE SINGLE SOURCE OF TRUTH for what a recipe DOES, keyed identically to SNIP_RECIPE_CONTRACTS
#: (what a recipe REQUIRES). Two tables rather than one because they answer different questions and
#: are consulted at different times -- the contract gates the source read, the renderer runs after
#: geometry -- but they must cover the same recipes, which the assert below enforces at import.
_SNIP_RECIPE_RENDERERS = {
    NO_CHANGE: render_no_change,
    CLAHE_BLEND: render_clahe_blend,
}

# Bidirectional, and at IMPORT time. A recipe with a contract but no renderer would pass plan
# validation and fail a layer deeper while rendering; a renderer with no contract would run without
# its dtype precondition ever being checked -- which is precisely the hazard-zero failure mode.
_declared, _wired = set(SNIP_RECIPE_CONTRACTS), set(_SNIP_RECIPE_RENDERERS)
if _declared != _wired:
    raise RuntimeError(
        "snip_recipes: contract/renderer tables disagree. "
        f"declared-not-wired={sorted(_declared - _wired)}, "
        f"wired-not-declared={sorted(_wired - _declared)}. Adding a recipe means writing the "
        "renderer, registering it, and declaring its contract -- all three."
    )


def render_snip(image: np.ndarray, *, snip_recipe: str, **kwargs) -> np.ndarray:
    """Apply one recipe's photometric transform to an already-cropped snip.

    Geometry has happened by the time this runs; every recipe sees the same pixels and differs only
    in what it does to their values.
    """
    try:
        renderer = _SNIP_RECIPE_RENDERERS[str(snip_recipe)]
    except KeyError:
        raise SnipRecipeError(
            f"unknown snip_recipe {snip_recipe!r}. Supported: {sorted(SUPPORTED_SNIP_RECIPES)}."
        ) from None
    return renderer(image, **kwargs)
