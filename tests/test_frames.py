import io
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
from PIL import Image

from reckon.frames import (
    extract_evidence_frames_in_window,
    extract_frame,
    extract_frames_at,
    extract_frames_in_window,
    frame_to_jpeg_bytes,
)

_FPS = 10
_WIDTH = 64
_HEIGHT = 48


def _write_synthetic_video(path: Path, frame_count: int) -> None:
    """Encode a short solid-colour video locally (no network) for frame-extraction tests.

    Each frame is tinted by its index so an extracted frame can be checked against the
    timestamp it was requested at.
    """
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=_FPS)
    stream.width = _WIDTH
    stream.height = _HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, _FPS)

    for i in range(frame_count):
        array = np.full((_HEIGHT, _WIDTH, 3), i * 10 % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    _write_synthetic_video(path, frame_count=_FPS * 2)  # 2 seconds
    return path


# ---------------------------------------------------------------------------
# extract_frame
# ---------------------------------------------------------------------------

def test_extract_frame_returns_image(synthetic_video: Path):
    image = extract_frame(synthetic_video, 1000)
    assert isinstance(image, Image.Image)
    assert image.size == (_WIDTH, _HEIGHT)


def test_extract_frame_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Recording not found"):
        extract_frame(tmp_path / "nope.mp4", 0)


def test_extract_frame_negative_timestamp_raises(synthetic_video: Path):
    with pytest.raises(ValueError, match="non-negative"):
        extract_frame(synthetic_video, -1)


def test_extract_frame_past_end_returns_last_frame(synthetic_video: Path):
    # 2s clip; requesting 99s should yield the final frame, not raise.
    image = extract_frame(synthetic_video, 99_000)
    assert isinstance(image, Image.Image)


def test_extract_frame_no_video_stream_raises(tmp_path: Path):
    # An audio-only file has no video stream; the error must be explicit, not an IndexError.
    audio_path = tmp_path / "audio_only.m4a"
    container = av.open(str(audio_path), mode="w")
    stream = container.add_stream("aac", rate=16_000)
    stream.layout = "mono"
    samples = np.zeros((1, 16_000), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.rate = 16_000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    with pytest.raises(ValueError, match="No video stream"):
        extract_frame(audio_path, 0)


# ---------------------------------------------------------------------------
# extract_frames_in_window
# ---------------------------------------------------------------------------

def test_extract_frames_in_window_returns_requested_count(synthetic_video: Path):
    frames = extract_frames_in_window(synthetic_video, 0, 2000, count=3)
    assert len(frames) == 3


def test_extract_frames_in_window_timestamps_ascending_and_in_range(synthetic_video: Path):
    frames = extract_frames_in_window(synthetic_video, 200, 1800, count=4)
    timestamps = [ts for ts, _ in frames]
    assert timestamps == sorted(timestamps)
    assert all(200 <= ts <= 1800 for ts in timestamps)


def test_extract_frames_in_window_single_count_uses_midpoint(synthetic_video: Path):
    frames = extract_frames_in_window(synthetic_video, 400, 1400, count=1)
    assert len(frames) == 1
    assert frames[0][0] == 900


def test_extract_frames_in_window_rejects_bad_range(synthetic_video: Path):
    with pytest.raises(ValueError, match="greater than"):
        extract_frames_in_window(synthetic_video, 1000, 1000, count=2)


def test_extract_frames_in_window_rejects_zero_count(synthetic_video: Path):
    with pytest.raises(ValueError, match="at least 1"):
        extract_frames_in_window(synthetic_video, 0, 1000, count=0)


# ---------------------------------------------------------------------------
# extract_evidence_frames_in_window
# ---------------------------------------------------------------------------

def test_extract_evidence_frames_in_window_uses_actual_pts_in_range(
    synthetic_video: Path,
):
    frames = extract_evidence_frames_in_window(
        synthetic_video,
        200,
        1800,
        count=3,
    )

    assert [timestamp for timestamp, _ in frames] == [200, 1000, 1800]


def test_extract_evidence_frames_in_window_replaces_out_of_range_timestamp(
    synthetic_video: Path,
):
    frames = extract_evidence_frames_in_window(
        synthetic_video,
        1000,
        3000,
        count=3,
    )
    timestamps = [timestamp for timestamp, _ in frames]

    assert timestamps == [1000, 1900]
    assert 3000 not in timestamps


def test_extract_evidence_frames_in_window_deduplicates_final_frame(
    synthetic_video: Path,
):
    frames = extract_evidence_frames_in_window(
        synthetic_video,
        2500,
        3500,
        count=3,
    )

    assert len(frames) == 1
    assert frames[0][0] == 1900


def test_extract_evidence_frames_in_window_rejects_missing_pts(
    synthetic_video: Path,
):
    frame = MagicMock()
    frame.pts = None

    with (
        patch("reckon.frames._decode_video_frame_at", return_value=frame),
        pytest.raises(ValueError, match="no presentation timestamp"),
    ):
        extract_evidence_frames_in_window(
            synthetic_video,
            0,
            1000,
            count=1,
        )


def test_extract_evidence_frames_in_window_rejects_invalid_duration(
    synthetic_video: Path,
):
    with (
        patch("reckon.frames.video_duration_ms", return_value=0),
        pytest.raises(ValueError, match="duration must be positive"),
    ):
        extract_evidence_frames_in_window(
            synthetic_video,
            0,
            1000,
            count=1,
        )


# ---------------------------------------------------------------------------
# frame_to_jpeg_bytes
# ---------------------------------------------------------------------------

def test_frame_to_jpeg_bytes_produces_jpeg_header(synthetic_video: Path):
    image = extract_frame(synthetic_video, 500)
    data = frame_to_jpeg_bytes(image)
    # JPEG files start with the SOI marker 0xFFD8.
    assert data[:2] == b"\xff\xd8"
    assert len(data) > 0


def test_frame_to_jpeg_bytes_downscales_above_max_dimension(synthetic_video: Path):
    image = extract_frame(synthetic_video, 500)
    data = frame_to_jpeg_bytes(image, max_dimension=16)
    assert max(Image.open(io.BytesIO(data)).size) <= 16


def test_frame_to_jpeg_bytes_leaves_small_frame_unscaled(synthetic_video: Path):
    image = extract_frame(synthetic_video, 500)
    data = frame_to_jpeg_bytes(image, max_dimension=4096)
    assert Image.open(io.BytesIO(data)).size == (_WIDTH, _HEIGHT)


def test_frame_to_jpeg_bytes_rejects_zero_max_dimension(synthetic_video: Path):
    image = extract_frame(synthetic_video, 500)
    with pytest.raises(ValueError, match="at least 1"):
        frame_to_jpeg_bytes(image, max_dimension=0)


# ---------------------------------------------------------------------------
# extract_frames_at
# ---------------------------------------------------------------------------

def test_extract_frames_at_returns_one_frame_per_timestamp(synthetic_video: Path):
    frames = extract_frames_at(synthetic_video, [100, 900, 1500])
    assert len(frames) == 3


def test_extract_frames_at_preserves_requested_timestamps(synthetic_video: Path):
    requested = [100, 900, 1500]
    frames = extract_frames_at(synthetic_video, requested)
    assert [ts for ts, _ in frames] == requested


def test_extract_frames_at_returns_images(synthetic_video: Path):
    frames = extract_frames_at(synthetic_video, [500])
    assert isinstance(frames[0][1], Image.Image)


def test_extract_frames_at_rejects_empty_list(synthetic_video: Path):
    with pytest.raises(ValueError, match="must not be empty"):
        extract_frames_at(synthetic_video, [])


def test_extract_frames_at_rejects_negative_timestamp(synthetic_video: Path):
    with pytest.raises(ValueError, match="non-negative"):
        extract_frames_at(synthetic_video, [100, -5])


def test_extract_frames_at_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Recording not found"):
        extract_frames_at(tmp_path / "nope.mp4", [0])
