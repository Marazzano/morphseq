"""death_detection compute tests — the two independent flags.

Covers spec "Done When": alive, clear death, transient dip (viability true / persistence false),
too-few-timepoints, single-channel fail-loud. Lead-time (hours) is exercised in test_lead_time.py
and the two-grain wiring in test_integration.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.quality_control.death_detection.compute import (
    compute_death_detection_flags,
    compute_viability_dead_flag,
)
from data_pipeline.quality_control.death_detection.config import resolve_config
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_physical_embryo_id,
    build_snip_id,
    build_well_id,
)

EXP = "20250912"
WELL = build_well_id(EXP, "B01")
CHANNEL = "BF"


def _trace(phys_index, fractions, *, dt_s=3600.0):
    """One animal's fraction_alive trace + matching uniform frame timing. Returns (fa_df, timing_df)."""
    phys = build_physical_embryo_id(WELL, phys_index)
    fa_rows, timing_rows = [], []
    for t, frac in enumerate(fractions):
        image_id = build_image_id(WELL, CHANNEL, t)
        embryo_id = build_embryo_id(phys, image_id)
        snip_id = build_snip_id(embryo_id, image_id)
        fa_rows.append(
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": phys,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "image_id": image_id,
                "time_index": t,
                "channel_id": CHANNEL,
                "fraction_alive": frac,
            }
        )
        timing_rows.append(
            {"experiment_id": EXP, "well_id": WELL, "time_index": t, "elapsed_time_s": t * dt_s}
        )
    return pd.DataFrame(fa_rows), pd.DataFrame(timing_rows)


def test_viability_flag_per_frame_and_null_is_not_dead():
    fa, _ = _trace(1, [1.0, 0.5, np.nan])
    flag = compute_viability_dead_flag(fa, dead_fraction_threshold=0.90)
    # 1.0 alive (False), 0.5 <= 0.9 dead (True), null -> not a death call (False)
    assert flag.tolist() == [False, True, False]


def test_alive_animal_no_persistence():
    fa, timing = _trace(1, [1.0, 0.98, 0.97, 0.99, 0.98, 0.97])
    flags, inflections = compute_death_detection_flags(fa, timing, config=resolve_config())
    assert not flags["persistence_dead_flag"].any()
    assert len(inflections) == 0


def test_clear_death_persistence_from_D_onward():
    # Healthy then collapses and stays dead; lead_time_hr=0 so D == inflection frame.
    fa, timing = _trace(1, [1.0, 1.0, 1.0, 0.2, 0.1, 0.05, 0.05])
    cfg = resolve_config({"lead_time_hr": 0.0})
    flags, inflections = compute_death_detection_flags(fa, timing, config=cfg)
    assert len(inflections) == 1
    # persistence true for the late (dead) frames, false for the early healthy ones
    persistence = flags["persistence_dead_flag"].tolist()
    assert persistence[0] is False and persistence[-1] is True
    assert any(p is False for p in persistence) and any(p is True for p in persistence)


def test_transient_dip_viability_true_persistence_false():
    # A one-frame dip that recovers: viability fires on the dip, persistence never does.
    fa, timing = _trace(1, [1.0, 1.0, 0.1, 1.0, 1.0, 1.0])
    flags, inflections = compute_death_detection_flags(fa, timing, config=resolve_config())
    assert flags["viability_dead_flag"].any()          # the dip frame trips viability
    assert not flags["persistence_dead_flag"].any()    # but it is not a persistent death
    assert len(inflections) == 0


def test_too_few_timepoints_no_death():
    fa, timing = _trace(1, [1.0, 0.1])  # < min_timepoints (3)
    flags, inflections = compute_death_detection_flags(fa, timing, config=resolve_config())
    assert not flags["persistence_dead_flag"].any()
    assert len(inflections) == 0


def test_multi_channel_trace_fails_loud():
    fa, timing = _trace(1, [1.0, 0.9, 0.8])
    # duplicate a time_index for the same animal -> two traces
    fa = pd.concat([fa, fa.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly ONE viability trace"):
        compute_death_detection_flags(fa, timing, config=resolve_config())
