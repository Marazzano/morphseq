"""Frame-metadata addressing must not stride by the channel count.

REWRITTEN AT THE MERGE, not retuned. main and this branch fixed the SAME bug independently — ND2
frame-metadata indexing treating C as an addressable axis — and the merge kept this branch's
``Nd2Axes.frame_index``. main's ``_metadata_sequence_index`` helper is therefore gone, and the tests
it came with are re-expressed against the surviving API rather than deleted: the BEHAVIOUR they pin
is the whole point of the fix and is worth keeping under either implementation.

Both assertions below are byte-for-byte the values main asserted (15, 20, and the [0, 20, 40]
request sequence), so this is the same guarantee, not a weakened one.

Why frame_index survived: it derives strides from the file's declared ``sequence_order`` instead of
hardcoding ``(T x n_positions + P) x n_z``, so it is also correct for a file whose loops are ordered
differently, and for the real pbx pilot that has NO time axis at all — where reading ``shape[:3]``
as (T, W, Z) turned 96 positions into 96 timepoints.
"""

from types import SimpleNamespace

import numpy as np

from data_pipeline.acquisition.metadata_ingest.scope.yx1.extract_yx1_scope_metadata import (
    _extract_timestamps,
)
from data_pipeline.acquisition.metadata_ingest.scope.yx1.nd2_axes import Nd2Axes


def _axes(*, n_t: int = 3, n_p: int = 4, n_z: int = 5, n_c: int = 2) -> Nd2Axes:
    """Axes for a file whose SHAPE includes C but whose frame metadata does not."""
    return Nd2Axes(
        n_t=n_t,
        n_p=n_p,
        n_z=n_z,
        n_c=n_c,
        height_px=10,
        width_px=10,
        sequence_order=("T", "P", "Z"),
    )


class _FakeNd2:
    def __init__(self) -> None:
        self.requested_indices: list[int] = []

    def frame_metadata(self, sequence_index: int) -> SimpleNamespace:
        self.requested_indices.append(sequence_index)
        return SimpleNamespace(
            channels=[
                SimpleNamespace(
                    time=SimpleNamespace(relativeTimeMs=float(sequence_index * 1000))
                ),
                SimpleNamespace(
                    time=SimpleNamespace(relativeTimeMs=float(sequence_index * 1000))
                ),
            ]
        )


def test_frame_index_excludes_bundled_channel_axis() -> None:
    # ND2File.shape may be T=3, P=4, Z=5, C=2, but frame_metadata records are flattened over
    # T/P/Z only; BOTH channels live inside each record. Including n_c in the stride would
    # overshoot the frame count by a factor of 2.
    axes = _axes()
    assert axes.frame_index(time=0, position=3, z=0) == 15
    assert axes.frame_index(time=1, position=0, z=0) == 20


def test_extract_timestamps_does_not_stride_by_channel_count() -> None:
    nd = _FakeNd2()

    timestamps = _extract_timestamps(nd, _axes())

    assert nd.requested_indices == [0, 20, 40]
    np.testing.assert_array_equal(timestamps, np.array([0.0, 20.0, 40.0]))
