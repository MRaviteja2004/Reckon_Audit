from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from reckon.keyframes import (
    SAMPLE_FPS,
    _sampled_thumbnails,
    detect_change_events,
    video_duration_ms,
)

_FPS = 10
_WIDTH = 160
_HEIGHT = 120
_BACKGROUND = 90

# A region covering a quarter of the frame, standing in for a page-level change.
_LARGE_REGION = (40, 30, 120, 90)

# A region covering ~1% of the frame, standing in for a localised change such as text
# appearing in one field. Real screen captures are dominated by unchanged pixels, so
# this is the case detection must not average away.
_SMALL_REGION = (72, 54, 88, 66)

# Three scenes of two seconds each: the screen changes exactly twice, at 2000 ms and
# 4000 ms, which is what the detector must recover.
_SCENE_LEVELS = [0, 200, 40]
_FRAMES_PER_SCENE = 20
_SCENE_CHANGE_MS = [2000, 4000]
_DURATION_MS = 6000


def _write_scene_video(
    path: Path,
    scene_levels: list[int],
    region: tuple[int, int, int, int],
) -> None:
    """Encode a video locally (no network) where only `region` changes between scenes.

    The background is held constant so the frame-to-frame difference is confined to
    the region — the shape of a real UI change, where most of the screen is static.
    """
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=_FPS)
    stream.width = _WIDTH
    stream.height = _HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, _FPS)

    left, top, right, bottom = region
    for level in scene_levels:
        for _ in range(_FRAMES_PER_SCENE):
            array = np.full((_HEIGHT, _WIDTH, 3), _BACKGROUND, dtype=np.uint8)
            array[top:bottom, left:right] = level
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def scene_video(tmp_path: Path) -> Path:
    path = tmp_path / "scenes.mp4"
    _write_scene_video(path, _SCENE_LEVELS, _LARGE_REGION)
    return path


@pytest.fixture
def localised_change_video(tmp_path: Path) -> Path:
    path = tmp_path / "localised.mp4"
    _write_scene_video(path, _SCENE_LEVELS, _SMALL_REGION)
    return path


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    path = tmp_path / "static.mp4"
    _write_scene_video(path, [120], _LARGE_REGION)
    return path


# ---------------------------------------------------------------------------
# video_duration_ms
# ---------------------------------------------------------------------------

def test_video_duration_ms_matches_encoded_length(scene_video: Path):
    assert video_duration_ms(scene_video) == pytest.approx(_DURATION_MS, abs=200)


def test_video_duration_ms_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Recording not found"):
        video_duration_ms(tmp_path / "nope.mp4")


# ---------------------------------------------------------------------------
# detect_change_events
# ---------------------------------------------------------------------------

def test_detect_change_events_finds_each_scene_boundary(scene_video: Path):
    events = detect_change_events(scene_video)
    assert len(events) == len(_SCENE_CHANGE_MS)


def test_detect_change_events_timestamps_land_on_the_boundaries(scene_video: Path):
    events = detect_change_events(scene_video)
    for actual, expected in zip(events, _SCENE_CHANGE_MS):
        assert actual == pytest.approx(expected, abs=300)


def test_detect_change_events_returns_ascending_timestamps(scene_video: Path):
    events = detect_change_events(scene_video)
    assert events == sorted(events)


def test_detect_change_events_static_video_returns_empty(static_video: Path):
    assert detect_change_events(static_video) == []


def test_detect_change_events_high_threshold_suppresses_everything(scene_video: Path):
    # The changed region covers a quarter of the frame, so demanding half suppresses it.
    assert detect_change_events(scene_video, threshold=0.5) == []


def test_detect_change_events_finds_localised_change(localised_change_video: Path):
    # Regression guard: a change confined to ~1% of the frame is exactly what a mean
    # inter-frame difference averages into invisibility, and what typing looks like.
    assert len(detect_change_events(localised_change_video)) == len(_SCENE_CHANGE_MS)


def test_detect_change_events_localised_change_lands_on_boundaries(
    localised_change_video: Path,
):
    events = detect_change_events(localised_change_video)
    for actual, expected in zip(events, _SCENE_CHANGE_MS):
        assert actual == pytest.approx(expected, abs=300)


def test_detect_change_events_respects_max_events(scene_video: Path):
    events = detect_change_events(scene_video, max_events=1)
    assert len(events) == 1


def test_detect_change_events_min_gap_collapses_nearby_events(scene_video: Path):
    # A gap wider than the whole clip can only admit one event.
    events = detect_change_events(scene_video, min_gap_ms=_DURATION_MS * 2)
    assert len(events) == 1


def test_sampled_thumbnails_grid_does_not_drift(scene_video: Path):
    """Sample points must track a fixed grid, not compound each frame's overshoot.

    Deriving the next sample point from the frame that satisfied the previous one adds
    that frame's overshoot to every later sample. Measured on a real 42 FPS recording
    that reached ~500 ms of lag by the 13-second mark, which lands the reported change
    on the wrong frame.
    """
    interval_ms = 1000 / SAMPLE_FPS
    frame_period_ms = 1000 / _FPS

    with av.open(str(scene_video)) as container:
        stream = container.streams.video[0]
        timestamps = [ts for ts, _ in _sampled_thumbnails(container, stream, SAMPLE_FPS)]

    for index, timestamp_ms in enumerate(timestamps):
        assert timestamp_ms <= index * interval_ms + frame_period_ms


def test_detect_change_events_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Recording not found"):
        detect_change_events(tmp_path / "nope.mp4")


def test_detect_change_events_rejects_zero_sample_fps(scene_video: Path):
    with pytest.raises(ValueError, match="sample_fps must be positive"):
        detect_change_events(scene_video, sample_fps=0)


def test_detect_change_events_rejects_threshold_above_one(scene_video: Path):
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        detect_change_events(scene_video, threshold=1.5)


def test_detect_change_events_rejects_negative_min_gap(scene_video: Path):
    with pytest.raises(ValueError, match="min_gap_ms must be non-negative"):
        detect_change_events(scene_video, min_gap_ms=-1)


def test_detect_change_events_rejects_zero_max_events(scene_video: Path):
    with pytest.raises(ValueError, match="max_events must be at least 1"):
        detect_change_events(scene_video, max_events=0)
