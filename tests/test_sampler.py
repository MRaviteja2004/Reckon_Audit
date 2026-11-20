from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reckon.sampler import (
    MAX_WINDOW_MS,
    derive_action_windows,
    extract_observed_trace,
    select_window_frames,
)
from reckon.schema import ActionType, ObservedAction, ObservedTrace

_VIDEO = Path("recording.mp4")
_SOURCE = "recording.mp4"


def _observed_action(index: int = 1, timestamp_ms: int = 0) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=ActionType.click,
        target="Submit",
        confidence=0.9,
    )


def _chunk(clip_offset_ms: int, clip_end_ms: int, actions: list[ObservedAction]) -> ObservedTrace:
    return ObservedTrace(
        source_file=_SOURCE,
        clip_offset_ms=clip_offset_ms,
        clip_end_ms=clip_end_ms,
        actions=actions,
    )


# ---------------------------------------------------------------------------
# derive_action_windows
# ---------------------------------------------------------------------------

def test_derive_windows_empty_returns_empty():
    assert derive_action_windows([]) == []


def test_derive_windows_single_event_pads_both_sides():
    assert derive_action_windows([10_000], pre_pad_ms=2000, post_pad_ms=3000) == [(8000, 13_000)]


def test_derive_windows_clamps_start_at_zero():
    assert derive_action_windows([500], pre_pad_ms=2000, post_pad_ms=3000) == [(0, 3500)]


def test_derive_windows_merges_overlapping():
    windows = derive_action_windows([10_000, 11_000], pre_pad_ms=2000, post_pad_ms=3000)
    assert windows == [(8000, 14_000)]


def test_derive_windows_keeps_distant_events_separate():
    windows = derive_action_windows([5000, 60_000], pre_pad_ms=1000, post_pad_ms=1000)
    assert windows == [(4000, 6000), (59_000, 61_000)]


def test_derive_windows_splits_window_exceeding_max_length():
    windows = derive_action_windows([0], pre_pad_ms=0, post_pad_ms=50_000, max_window_ms=20_000)
    assert len(windows) > 1
    assert all(end - start <= 20_000 for start, end in windows)


def test_derive_windows_split_produces_disjoint_windows():
    """Split windows must not share an endpoint.

    A shared endpoint puts one frame in two windows; each window is extracted
    independently, so that frame's action is reported twice and reaches the aligner as
    a phantom extra_action.
    """
    windows = derive_action_windows([0], pre_pad_ms=0, post_pad_ms=90_000, max_window_ms=20_000)
    assert len(windows) > 1
    for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
        assert later_start > earlier_end


def test_derive_windows_returns_ascending_order():
    windows = derive_action_windows([30_000, 5000, 60_000], pre_pad_ms=500, post_pad_ms=500)
    assert windows == sorted(windows)


def test_derive_windows_rejects_zero_max_window():
    with pytest.raises(ValueError, match="max_window_ms must be positive"):
        derive_action_windows([1000], max_window_ms=0)


def test_derive_windows_default_max_window_is_respected():
    windows = derive_action_windows([0], pre_pad_ms=0, post_pad_ms=MAX_WINDOW_MS * 3)
    assert all(end - start <= MAX_WINDOW_MS for start, end in windows)


# ---------------------------------------------------------------------------
# select_window_frames
# ---------------------------------------------------------------------------

def test_select_window_frames_includes_window_edges():
    frames = select_window_frames(1000, 5000, [3000])
    assert frames[0] == 1000
    assert frames[-1] == 5000


def test_select_window_frames_includes_change_events_inside_window():
    frames = select_window_frames(0, 10_000, [2000, 6000])
    assert 2000 in frames and 6000 in frames


def test_select_window_frames_excludes_events_outside_window():
    frames = select_window_frames(1000, 5000, [500, 3000, 9000])
    assert 500 not in frames and 9000 not in frames


def test_select_window_frames_returns_ascending_unique_timestamps():
    frames = select_window_frames(0, 8000, [4000, 2000, 4000])
    assert frames == sorted(set(frames))


def test_select_window_frames_thins_to_max_frames():
    events = list(range(1000, 9000, 250))
    frames = select_window_frames(0, 10_000, events, max_frames=5)
    assert len(frames) == 5


def test_select_window_frames_thinning_keeps_first_and_last():
    events = list(range(1000, 9000, 250))
    frames = select_window_frames(0, 10_000, events, max_frames=4)
    assert frames[0] == 0
    assert frames[-1] == 10_000


def test_select_window_frames_without_events_still_returns_edges():
    assert select_window_frames(2000, 6000, []) == [2000, 6000]


def test_select_window_frames_rejects_inverted_window():
    with pytest.raises(ValueError, match="must be greater than"):
        select_window_frames(5000, 5000, [])


def test_select_window_frames_rejects_max_frames_below_two():
    with pytest.raises(ValueError, match="max_frames must be at least 2"):
        select_window_frames(0, 1000, [], max_frames=1)


# ---------------------------------------------------------------------------
# extract_observed_trace
# ---------------------------------------------------------------------------

def test_extract_observed_trace_no_change_events_still_samples_the_recording():
    """A recording with no detected change must not silently produce an empty trace.

    Changes smaller than the moving cursor are undetectable by frame differencing, and
    an empty trace reads downstream as "nothing happened" — turning every expected step
    into a missing_action against a possibly faithful recording.
    """
    with (
        patch("reckon.sampler.detect_change_events", return_value=[]),
        patch("reckon.sampler.video_duration_ms", return_value=14_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = _chunk(0, 14_000, [_observed_action()])
        trace = extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    mock_extract.assert_called()
    assert len(trace.actions) == 1


def test_extract_observed_trace_fallback_frames_span_the_recording():
    with (
        patch("reckon.sampler.detect_change_events", return_value=[]),
        patch("reckon.sampler.video_duration_ms", return_value=14_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = _chunk(0, 14_000, [])
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    frame_timestamps = mock_extract.call_args.args[3]
    assert len(frame_timestamps) >= 4
    assert max(frame_timestamps) > 10_000


def test_extract_observed_trace_zero_duration_returns_empty_trace():
    with (
        patch("reckon.sampler.detect_change_events", return_value=[]),
        patch("reckon.sampler.video_duration_ms", return_value=0),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        trace = extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    mock_extract.assert_not_called()
    assert trace.actions == []


def test_extract_observed_trace_runs_one_extraction_per_window():
    with (
        patch("reckon.sampler.detect_change_events", return_value=[5000, 60_000]),
        patch("reckon.sampler.video_duration_ms", return_value=90_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.side_effect = [
            _chunk(3000, 8000, [_observed_action()]),
            _chunk(58_000, 63_000, [_observed_action()]),
        ]
        trace = extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    assert mock_extract.call_count == 2
    assert len(trace.actions) == 2


def test_extract_observed_trace_merges_chunks_to_absolute_timestamps():
    with (
        patch("reckon.sampler.detect_change_events", return_value=[5000, 60_000]),
        patch("reckon.sampler.video_duration_ms", return_value=90_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.side_effect = [
            _chunk(3000, 8000, [_observed_action(timestamp_ms=2000)]),
            _chunk(58_000, 63_000, [_observed_action(timestamp_ms=1000)]),
        ]
        trace = extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    assert [a.timestamp_ms for a in trace.actions] == [5000, 59_000]


def test_extract_observed_trace_reports_split_seam_action_once():
    def edge_click(*args: object, clip_offset_ms: int, clip_end_ms: int, **kwargs: object) -> ObservedTrace:
        edge_ms = 0 if clip_offset_ms else clip_end_ms - clip_offset_ms
        return _chunk(clip_offset_ms, clip_end_ms, [_observed_action(timestamp_ms=edge_ms)])

    events = list(range(1000, 2 * MAX_WINDOW_MS, 1000))
    with (
        patch("reckon.sampler.detect_change_events", return_value=events),
        patch("reckon.sampler.video_duration_ms", return_value=3 * MAX_WINDOW_MS),
        patch("reckon.sampler.extract_window", side_effect=edge_click) as mock_extract,
    ):
        trace = extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    assert mock_extract.call_count == 3
    assert len(trace.actions) == 2


def test_extract_observed_trace_clamps_windows_to_video_duration():
    with (
        patch("reckon.sampler.detect_change_events", return_value=[9500]),
        patch("reckon.sampler.video_duration_ms", return_value=10_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = _chunk(7500, 10_000, [])
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    assert mock_extract.call_args.kwargs["clip_end_ms"] == 10_000


def test_extract_observed_trace_passes_selected_frames_within_window():
    with (
        patch("reckon.sampler.detect_change_events", return_value=[5000]),
        patch("reckon.sampler.video_duration_ms", return_value=20_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = _chunk(3000, 8000, [])
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    frame_timestamps = mock_extract.call_args.args[3]
    assert all(3000 <= ts <= 8000 for ts in frame_timestamps)
    assert 5000 in frame_timestamps


def test_extract_observed_trace_forwards_retry_callback():
    def on_retry(attempt: int, code: int, delay_s: float) -> None:
        return None

    with (
        patch("reckon.sampler.detect_change_events", return_value=[5000]),
        patch("reckon.sampler.video_duration_ms", return_value=20_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = _chunk(3000, 8000, [])
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE, on_retry=on_retry)
    assert mock_extract.call_args.kwargs["on_retry"] is on_retry


def test_extract_observed_trace_events_past_duration_yield_no_windows():
    with (
        patch("reckon.sampler.detect_change_events", return_value=[500_000]),
        patch("reckon.sampler.video_duration_ms", return_value=10_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        trace = extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)
    mock_extract.assert_not_called()
    assert trace.actions == []
