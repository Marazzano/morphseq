"""death_detection lead-time tests — hours (not frames) are subtracted, via frame timing.

The key proof: with a NON-UNIFORM frame interval, subtracting lead_time_hr in hours lands on a
different frame D than naively subtracting a frame count would.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.quality_control.death_detection.death_event import called_death_time_index


def _timing(elapsed_hours_by_time_index: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_index": list(elapsed_hours_by_time_index),
            "elapsed_time_s": [h * 3600.0 for h in elapsed_hours_by_time_index.values()],
        }
    )


def test_uniform_interval_lead_time_maps_back_by_hours():
    # 1 hour per frame; inflection at t=6 (6h). lead 4h -> 2h -> frame t=2.
    timing = _timing({i: float(i) for i in range(8)})
    assert called_death_time_index(timing, 6, lead_time_hr=4.0) == 2


def test_non_uniform_interval_hours_not_frames():
    # Non-uniform: frames are 0h,1h,2h,3h,4h, then SPARSE 8h,12h,16h.
    # Inflection at t=7 (16h). lead 4h -> 12h -> that is exactly frame t=6 (12h),
    # NOT t=6-by-count semantics on a uniform grid. Proves hours drive the mapping.
    timing = _timing({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8, 6: 12, 7: 16})
    assert called_death_time_index(timing, 7, lead_time_hr=4.0) == 6


def test_lead_time_before_first_frame_clamps():
    timing = _timing({0: 0.0, 1: 1.0, 2: 2.0})
    # inflection at t=1 (1h), lead 4h -> -3h -> before first frame -> clamp to t=0.
    assert called_death_time_index(timing, 1, lead_time_hr=4.0) == 0


def test_zero_lead_time_is_inflection_frame():
    timing = _timing({i: float(i) for i in range(5)})
    assert called_death_time_index(timing, 3, lead_time_hr=0.0) == 3
