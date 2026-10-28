import numpy as np
import pytest

from stage2.realtime_force_field import _build_frame_to_row


def test_frame_lookup_accepts_nonzero_contiguous_clip():
    frame_min, frame_to_row = _build_frame_to_row(np.arange(30, 230), 30, 200)

    assert frame_min == 30
    np.testing.assert_array_equal(frame_to_row, np.arange(200))


def test_frame_lookup_rejects_gaps():
    with pytest.raises(ValueError, match="unique and contiguous"):
        _build_frame_to_row(np.array([30, 31, 33]), 30, 3)


@pytest.mark.parametrize(
    ("frame_index", "frame_offset", "clip_length"),
    [
        (np.arange(31, 230), 30, 200),
        (np.arange(30, 230), 30, 201),
    ],
)
def test_frame_lookup_rejects_incomplete_clip_coverage(frame_index, frame_offset, clip_length):
    with pytest.raises(ValueError, match="does not cover the requested demo clip"):
        _build_frame_to_row(frame_index, frame_offset, clip_length)
