from types import SimpleNamespace

import numpy as np

from data_pipeline.acquisition.metadata_ingest.scope.yx1.extract_yx1_scope_metadata import (
    _extract_timestamps,
    _metadata_sequence_index,
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


def test_metadata_sequence_index_excludes_bundled_channel_axis() -> None:
    # ND2File.shape may be T=3, P=4, Z=5, C=2, but frame_metadata records
    # are flattened over T/P/Z; both channels live in each record.
    assert _metadata_sequence_index(
        time_index=0, position_index=3, n_positions=4, n_z=5
    ) == 15
    assert _metadata_sequence_index(
        time_index=1, position_index=0, n_positions=4, n_z=5
    ) == 20


def test_extract_timestamps_does_not_stride_by_channel_count() -> None:
    nd = _FakeNd2()

    timestamps = _extract_timestamps(nd, n_t=3, n_w=4, n_z=5)

    assert nd.requested_indices == [0, 20, 40]
    np.testing.assert_array_equal(timestamps, np.array([0.0, 20.0, 40.0]))
