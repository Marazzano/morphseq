"""Read per-channel exposure and illumination settings out of an ND2's text metadata.

WHY THIS EXISTS. Fluorescence intensity is only comparable across frames that were acquired the same
way, and exposure is the setting most likely to differ without anyone noticing -- it is adjusted to
get a good-looking image, not recorded as an experimental variable.

MEASURED, and the reason this module was written: in ``20260624_2x_td_bf_pbx_coll`` the three
"timepoints" are three ND2 files from three consecutive days, and the fluorescence exposure was
**600 ms on day 1 and 300 ms on days 2 and 3**. Laser power and iris were identical. Every embryo
looked ~2x brighter at t0, which reads exactly like a 1-vs-2-copy dosage difference and is entirely
the camera. Without exposure on the row, that confound is invisible to every downstream analysis.

WHY THE TEXT BLOB AND NOT THE STRUCTURED METADATA. ``nd2``'s typed ``metadata.channels[i].channel``
does not expose exposure for these files (``exposure_ms`` is absent), and the microscope sub-object
carries only optics. The value is present only in the free-text acquisition dump under
``ND2File.text_info``. That makes this a PARSER, with all the fragility that implies -- so it fails
soft (returns ``None`` per field) rather than taking down an ingest, and every field it returns is
optional by contract.

ORDER IS THE JOIN. The text dump lists one "Camera Settings" block per channel, in channel order, so
the Nth exposure belongs to the Nth channel. That is an assumption about Nikon's formatting, and it
is asserted against the channel count rather than trusted silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "Camera Settings:   Exposure: 300 ms" -- the whitespace varies, the unit does not.
_EXPOSURE_MS = re.compile(r"Exposure:\s*([\d.]+)\s*ms", re.IGNORECASE)
# "Nikon Ti2, Illuminator(DIA) Iris intensity: 18.1"
_IRIS = re.compile(r"Iris intensity:\s*([\d.]+)", re.IGNORECASE)
# The Celesta/SpectraIII block lists one Power line per laser line; the ACTIVE line is the one that
# is not at its 1.0 idle floor, so the max is the illumination that actually mattered.
_LIGHT_SOURCE_BLOCK = re.compile(r"(?:SpectraIII|Celesta|LUN-F)[^\n]*MultiLaser[^\n]*(.*?)(?=Camera|\Z)", re.S)
_POWER = re.compile(r"Power:\s*([\d.]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ChannelIllumination:
    """What one channel was acquired with. Every field optional -- this is parsed, not guaranteed."""

    exposure_ms: float | None = None
    illumination_power: float | None = None
    dia_iris_intensity: float | None = None


def read_channel_illumination(nd2_file, n_channels: int) -> list[ChannelIllumination]:
    """Per-channel illumination settings, in channel order; one entry per channel always.

    Returns all-``None`` entries rather than raising when the text dump is missing or does not
    parse. A missing exposure must degrade an analysis, not break an ingest -- but a WRONG exposure
    would be worse than none, so a count mismatch yields empties rather than a guessed alignment.
    """
    empty = [ChannelIllumination() for _ in range(n_channels)]
    try:
        text_info = nd2_file.text_info or {}
    except Exception:  # noqa: BLE001 - a metadata read must never take down an ingest
        return empty
    blob = " | ".join(str(value) for value in text_info.values())
    if not blob:
        return empty

    exposures = [float(v) for v in _EXPOSURE_MS.findall(blob)]
    irises = [float(v) for v in _IRIS.findall(blob)]
    powers = [
        max((float(p) for p in _POWER.findall(block)), default=None)
        for block in _LIGHT_SOURCE_BLOCK.findall(blob)
    ]

    # THE ALIGNMENT CHECK. Nikon repeats the whole settings dump once per channel, so a well-formed
    # file yields a multiple of n_channels. Anything else means the format is not what this parser
    # assumes, and a mis-aligned exposure is far more dangerous than a missing one: it would silently
    # attribute one channel's settings to another.
    if not exposures or len(exposures) % n_channels != 0:
        return empty

    # THE VALUES REPEAT AS BLOCKS OF n_channels, NOT AS RUNS PER CHANNEL. Measured on the pbx files:
    # two channels yield [11, 600, 11, 600] -- BF then tdtomato, the pair repeated -- NOT
    # [11, 11, 600, 600]. So channel i is at index i, and striding by len/n_channels would read
    # exposures[0] and exposures[2], reporting 11 ms for BOTH channels.
    #
    # That is the failure this module exists to prevent, produced by the module itself: it would have
    # reported the two channels as identically exposed and hidden the exact 600-vs-300 confound that
    # motivated writing it. A parser that fabricates agreement is worse than no parser.
    def nth(values: list, index: int):
        return values[index] if len(values) > index and len(values) % n_channels == 0 else None

    return [
        ChannelIllumination(
            exposure_ms=exposures[index],
            illumination_power=nth(powers, index),
            dia_iris_intensity=nth(irises, index),
        )
        for index in range(n_channels)
    ]
