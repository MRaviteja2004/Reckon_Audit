"""Stage 2a — deterministic change detection over a screen recording.

A UI recording is static most of the time, punctuated by discrete state changes: a
click navigates, a dialog opens, a toast appears. Those moments are found here by
comparing decoded frames, not by asking a model — which matters for two reasons.

The first is accuracy. Model-reported times are a known weak point: they drift, and an
audit whose whole value is an evidence pointer cannot afford a timestamp that is
approximately right. Every timestamp produced here is a decoder presentation time, so
it always names a frame that genuinely exists. The reported moment lags the change
itself by at most one sampling interval (250 ms at the default SAMPLE_FPS; measured
mean 124 ms on a real recording), since a change is noticed at the first sample taken
after it — raise sample_fps to narrow that at proportional decode cost.

The second is cost. This pass replaces what used to be a whole-recording model call
just to find *when* things happened. It needs no API key and no network.
"""

from pathlib import Path
from typing import NamedTuple

import av
from PIL import Image, ImageChops, ImageStat

from reckon.stage2.selector import select_events

_MS_PER_S = 1000

# Frames are compared as small greyscale thumbnails. Changes that matter survive
# downscaling; cursor movement, caret blink, and compression noise do not. The size is
# not smaller because a localised change (one field gaining text) must still cover
# enough thumbnail pixels to clear the threshold.
_COMPARE_SIZE = (160, 90)

# Per-pixel difference (0–255) above which a pixel counts as changed. Set above JPEG
# ringing and subpixel antialiasing, below any genuine repaint.
_PIXEL_DELTA = 24

# Rate at which frames are pulled for comparison. Higher rates resolve briefer events
# at proportionally more decode cost; 4 FPS reliably catches a 250 ms toast.
SAMPLE_FPS = 4.0

# Fraction of pixels (0–1) that must change before the screen counts as changed.
#
# The metric is a changed-pixel *count*, not a mean difference, because a mean is
# diluted by frame size: on a 3456×2234 capture, typing into one field moves the whole
# frame's average by almost nothing while changing its own pixels completely.
#
# Calibrated against every inter-frame difference in a real 21 s login recording:
# page-level changes score ~0.05, typing ~0.004, clearing a filled form ~0.0017, and
# the least conspicuous real change ~0.0012, while cursor movement and compression
# artefacts top out at ~0.0005. This sits in that gap.
#
# It is deliberately nearer the noise floor than the signal, because the errors are not
# symmetric: a false positive costs a few extra frames in a window that probably merges
# anyway, while a false negative drops a real action and surfaces as a phantom
# missing_action finding against a faithful recording.
CHANGE_THRESHOLD = 0.0008

# Minimum spacing between reported events. Typing produces a burst of small changes;
# without debouncing, one typed field becomes dozens of near-identical events.
MIN_EVENT_GAP_MS = 700

# Upper bound on reported events. Continuously changing content — an embedded video,
# a spinner, an animation — would otherwise produce an unbounded list. The strongest
# changes survive the cap.
MAX_CHANGE_EVENTS = 60


class _ScoredChangeEvent(NamedTuple):
    """Adapter so debounced (timestamp, score) pairs satisfy select_events."""

    timestamp_ms: int
    confidence: float


def video_duration_ms(video_path: Path) -> int:
    """Return the recording's duration in milliseconds.

    Prefers the container duration and falls back to the video stream's own duration,
    which some screen-capture encoders set when the container header does not.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Recording not found: {video_path}")

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")
        if container.duration is not None:
            return int(container.duration / av.time_base * _MS_PER_S)
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return int(stream.duration * stream.time_base * _MS_PER_S)
        raise ValueError(f"Could not determine duration of {video_path}")


def _sampled_thumbnails(
    container: "av.container.InputContainer",
    stream: "av.video.stream.VideoStream",
    sample_fps: float,
):
    """Yield (timestamp_ms, greyscale thumbnail) pairs at approximately sample_fps.

    Decoding is sequential rather than seek-based: a screen recording is scanned end
    to end here, and seeking per sample would be slower than decoding straight through.
    """
    interval_s = 1.0 / sample_fps
    next_sample_s = 0.0
    for frame in container.decode(stream):
        if frame.time is None or frame.time < next_sample_s:
            continue
        # Advance along a fixed grid anchored at zero. Deriving the next sample point
        # from this frame's own time instead would add that frame's overshoot to every
        # later sample, compounding into whole seconds of lag on a long recording —
        # and these timestamps are the audit's evidence pointers.
        while next_sample_s <= frame.time:
            next_sample_s += interval_s
        thumbnail = frame.to_image().convert("L").resize(_COMPARE_SIZE, Image.BILINEAR)
        yield int(frame.time * _MS_PER_S), thumbnail


def _difference_score(previous: Image.Image, current: Image.Image) -> float:
    """Fraction of thumbnail pixels that changed materially between two frames (0–1).

    Thresholding each pixel before counting is what keeps a small, high-contrast change
    (a validation message appearing) from being averaged into invisibility by the
    unchanged screen around it.
    """
    difference = ImageChops.difference(previous, current)
    changed_mask = difference.point(lambda value: 255 if value > _PIXEL_DELTA else 0)
    return ImageStat.Stat(changed_mask).mean[0] / 255.0


def _debounce(
    scored_events: list[tuple[int, float]],
    min_gap_ms: int,
    max_events: int,
) -> list[int]:
    """Collapse bursts into single events and cap the total, keeping strongest changes.

    Events arrive in time order. Within min_gap_ms only the highest-scoring event
    survives, so a burst of keystrokes collapses to the moment the field changed most.
    """
    kept: list[tuple[int, float]] = []
    for timestamp_ms, score in scored_events:
        if kept and timestamp_ms - kept[-1][0] < min_gap_ms:
            if score > kept[-1][1]:
                kept[-1] = (timestamp_ms, score)
            continue
        kept.append((timestamp_ms, score))

    if len(kept) > max_events:
        # Budgeted selection is policy-driven (default: legacy top-K by score).
        # STAGE2_EVENT_SELECTION_POLICY is read inside select_events via config.
        candidates = [
            _ScoredChangeEvent(timestamp_ms=timestamp_ms, confidence=score)
            for timestamp_ms, score in kept
        ]
        selected = select_events(candidates, max_events=max_events)
        return [event.timestamp_ms for event in selected]

    return [timestamp_ms for timestamp_ms, _ in kept]


def detect_change_events(
    video_path: Path,
    *,
    sample_fps: float = SAMPLE_FPS,
    threshold: float = CHANGE_THRESHOLD,
    min_gap_ms: int = MIN_EVENT_GAP_MS,
    max_events: int = MAX_CHANGE_EVENTS,
) -> list[int]:
    """Return ascending millisecond offsets where the screen changed materially.

    Each returned timestamp is a decoder presentation time, so it points at a frame
    that genuinely exists in the recording. An empty list means nothing on screen
    changed by more than `threshold` — a static or near-static recording.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Recording not found: {video_path}")
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if min_gap_ms < 0:
        raise ValueError(f"min_gap_ms must be non-negative, got {min_gap_ms}")
    if max_events < 1:
        raise ValueError(f"max_events must be at least 1, got {max_events}")

    scored_events: list[tuple[int, float]] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")
        stream = container.streams.video[0]

        previous: Image.Image | None = None
        for timestamp_ms, thumbnail in _sampled_thumbnails(container, stream, sample_fps):
            if previous is not None:
                score = _difference_score(previous, thumbnail)
                if score >= threshold:
                    scored_events.append((timestamp_ms, score))
            previous = thumbnail

    return _debounce(scored_events, min_gap_ms, max_events)
