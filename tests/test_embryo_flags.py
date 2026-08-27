from __future__ import annotations

import pandas as pd

from src.build.qc import determine_use_embryo_flag


def test_no_yolk_flag_is_informational_only():
    """no_yolk_flag never excludes — it is informational only.

    ``frame_flag`` is varied here to prove it does not exclude EITHER: it was deliberately demoted
    to informational (too many false positives on snapshot plates). See
    ``src/build/qc/embryo_flags.py`` — the exclusion set is exactly dead_flag / dead_flag2 /
    sa_outlier_flag / sam2_qc_flag.
    """
    df = pd.DataFrame(
        {
            "dead_flag": [False, False],
            "dead_flag2": [False, False],
            "sa_outlier_flag": [False, False],
            "sam2_qc_flag": [False, False],
            "frame_flag": [False, True],
            "no_yolk_flag": [True, True],
        }
    )

    got = determine_use_embryo_flag(df).tolist()

    assert got == [True, True]


def test_only_the_four_hard_flags_exclude():
    """Each hard flag excludes on its own; each informational flag never does."""
    excluding = ["dead_flag", "dead_flag2", "sa_outlier_flag", "sam2_qc_flag"]
    informational = ["frame_flag", "no_yolk_flag", "focus_flag", "bubble_flag"]

    for flag in excluding:
        assert determine_use_embryo_flag(pd.DataFrame({flag: [True]})).tolist() == [False], flag
    for flag in informational:
        assert determine_use_embryo_flag(pd.DataFrame({flag: [True]})).tolist() == [True], flag
