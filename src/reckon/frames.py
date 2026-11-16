"""Targeted frame extraction with PyAV (Phase 6).

The cross-check model is shown specific frames, not the whole video — it is asked a
narrow, grounded question ("does text matching X appear here?") about the exact moment
an assertion should hold. Pulling those frames locally with PyAV is cheaper and sharper
than another video pass, and the frames are full-resolution stills, which is what an
OCR-grade assertion check needs.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image

from reckon.keyframes import video_duration_ms

_MS_PER_S = 1000

# JPEG keeps frame payloads small for the vision API while preserving enough detail to
# read on-screen text; quality is high because the check is OCR-grade.
_JPEG_QUALITY = 90


@dataclass(frozen=True)
class _DecodedEvidenceFrame:
    pts: int
    timestamp_ms: int
    image: Image.Image


def _decode_video_frame_at(
    container: "av.container.InputContainer",
    stream: "av.video.stream.VideoStream",
    timestamp_ms: int,
) -> "av.VideoFrame":
    """Seek to timestamp_ms and return the selected decoded video frame.

    Seeks to the keyframe at or before the target, then decodes forward to the first
    frame whose presentation time reaches the target. Falls back to the last decoded
    frame when the target is past the final frame.
    """
    target_s = timestamp_ms / _MS_PER_S
    seek_offset = int(target_s / stream.time_base) if stream.time_base else 0
    container.seek(seek_offset, stream=stream, backward=True, any_frame=False)

    last_frame = None
    for frame in container.decode(stream):
        last_frame = frame
        if frame.time is not None and frame.time >= target_s:
            return frame
    if last_frame is None:
        raise ValueError(f"No decodable frame found at {timestamp_ms}ms")
    return last_frame


def _decode_frame_at(
    container: "av.container.InputContainer",
    stream: "av.video.stream.VideoStream",
    timestamp_ms: int,
) -> Image.Image:
    """Preserve the existing image-only decoder API."""
    return _decode_video_frame_at(container, stream, timestamp_ms).to_image()


def _decode_evidence_frame_at(
    container: "av.container.InputContainer",
    stream: "av.video.stream.VideoStream",
    timestamp_ms: int,
) -> _DecodedEvidenceFrame:
    """Decode evidence with its actual presentation timestamp."""
    frame = _decode_video_frame_at(container, stream, timestamp_ms)
    if frame.pts is None or stream.time_base is None:
        raise ValueError(
            f"Decoded frame at requested {timestamp_ms}ms has no presentation timestamp"
        )
    actual_timestamp_ms = int(frame.pts * stream.time_base * _MS_PER_S)
    if actual_timestamp_ms < 0:
        raise ValueError(
            f"Decoded frame has negative presentation timestamp {actual_timestamp_ms}ms"
        )
    return _DecodedEvidenceFrame(
        pts=frame.pts,
        timestamp_ms=actual_timestamp_ms,
        image=frame.to_image(),
    )


def extract_frame(video_path: Path, timestamp_ms: int) -> Image.Image:
    """Return the video frame at (or immediately after) the given timestamp."""
    if not video_path.exists():
        raise FileNotFoundError(f"Recording not found: {video_path}")
    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms must be non-negative, got {timestamp_ms}")

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")
        return _decode_frame_at(container, container.streams.video[0], timestamp_ms)


def extract_frames_in_window(
    video_path: Path,
    start_ms: int,
    end_ms: int,
    count: int,
) -> list[tuple[int, Image.Image]]:
    """Sample `count` evenly spaced frames across [start_ms, end_ms].

    Returns (timestamp_ms, image) pairs in ascending time order. The timestamp is the
    requested sample point, computed in code, so it is a trustworthy evidence pointer
    regardless of where the decoder landed. The container is opened once for the whole
    window rather than once per frame.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Recording not found: {video_path}")
    if end_ms <= start_ms:
        raise ValueError(f"end_ms ({end_ms}) must be greater than start_ms ({start_ms})")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    if count == 1:
        sample_points = [(start_ms + end_ms) // 2]
    else:
        step = (end_ms - start_ms) / (count - 1)
        sample_points = [int(start_ms + step * i) for i in range(count)]

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")
        stream = container.streams.video[0]
        return [(ts, _decode_frame_at(container, stream, ts)) for ts in sample_points]


def extract_evidence_frames_in_window(
    video_path: Path,
    start_ms: int,
    end_ms: int,
    count: int,
) -> list[tuple[int, Image.Image]]:
    """Sample Stage 4 evidence using actual decoder PTS and unique decoded frames."""
    if not video_path.exists():
        raise FileNotFoundError(f"Recording not found: {video_path}")
    if end_ms <= start_ms:
        raise ValueError(f"end_ms ({end_ms}) must be greater than start_ms ({start_ms})")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    duration_ms = video_duration_ms(video_path)
    if duration_ms <= 0:
        raise ValueError(f"Video duration must be positive, got {duration_ms}ms")

    if count == 1:
        sample_points = [(start_ms + end_ms) // 2]
    else:
        step = (end_ms - start_ms) / (count - 1)
        sample_points = [int(start_ms + step * i) for i in range(count)]
    bounded_points = list(dict.fromkeys(min(timestamp, duration_ms) for timestamp in sample_points))

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")
        stream = container.streams.video[0]
        seen_pts: set[int] = set()
        evidence: list[tuple[int, Image.Image]] = []
        for timestamp_ms in bounded_points:
            decoded = _decode_evidence_frame_at(
                container,
                stream,
                timestamp_ms,
            )
            if decoded.pts in seen_pts:
                continue
            seen_pts.add(decoded.pts)
            evidence.append((decoded.timestamp_ms, decoded.image))
        return evidence


def extract_frames_at(video_path: Path, timestamps_ms: list[int]) -> list[tuple[int, Image.Image]]:
    """Decode the frames at the given timestamps, opening the container once.

    Unlike extract_frames_in_window, which samples a window evenly, this takes the
    exact moments worth looking at — the change events found in keyframes.py. Returns
    (timestamp_ms, image) pairs in the order requested, with the requested timestamp
    kept as the evidence pointer rather than wherever the decoder happened to land.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Recording not found: {video_path}")
    if not timestamps_ms:
        raise ValueError("timestamps_ms must not be empty")
    if any(ts < 0 for ts in timestamps_ms):
        raise ValueError(f"timestamps_ms must all be non-negative, got {timestamps_ms}")

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")
        stream = container.streams.video[0]
        return [(ts, _decode_frame_at(container, stream, ts)) for ts in timestamps_ms]


def frame_to_jpeg_bytes(image: Image.Image, *, max_dimension: int | None = None) -> bytes:
    """Encode a frame as JPEG bytes for transmission to a vision model.

    max_dimension bounds the longest side before encoding. Vision APIs downscale
    large images server-side anyway, so sending a raw 4K screen frame costs upload
    time without buying legibility; bounding it locally keeps the text sharp at a
    fraction of the payload.
    """
    if max_dimension is not None and max_dimension < 1:
        raise ValueError(f"max_dimension must be at least 1, got {max_dimension}")

    prepared = image.convert("RGB")
    if max_dimension is not None and max(prepared.size) > max_dimension:
        prepared.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    buffer = io.BytesIO()
    prepared.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return buffer.getvalue()
