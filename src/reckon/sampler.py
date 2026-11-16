"""Stage 2 orchestration — change-detected windows over a full recording.

Recordings run longer than a minute, so the recording is not analysed in one gulp.
The pass is organised around what actually happened in it:

  1. Change detection — keyframes.py scans the recording locally and returns the exact
     moments the screen changed. No model, no API key, no network.
  2. Action windows — each change event becomes a padded window; overlapping windows
     merge, and an over-long window splits so its frame budget stays meaningful.
  3. Frame selection — within each window, frames are taken *at* the change events
     rather than on a fixed cadence, so the budget is spent on moments that carry
     information instead of on a static screen.
  4. Extraction — one vision call per window.
  5. Merge — merge_observed_chunks() reassembles the chunks, converting each
     clip-relative timestamp back to a recording-absolute one.

Evidence timestamps are anchored to frames chosen in code, never to a time the model
reported; see extractor.py for why that distinction carries the audit.
"""

import logging
from pathlib import Path

import instructor

from reckon.extractor import EXTRACTION_MODEL, extract_window
from reckon.keyframes import detect_change_events, video_duration_ms
from reckon.latency import (
    LatencyMetrics,
    bind_retry_callback,
    elapsed_ms,
    monotonic_ms,
    resolve_latency,
)
from reckon.retry import RetryCallback
from reckon.schema import ObservedTrace, merge_observed_chunks

logger = logging.getLogger(__name__)

# Padding around each change event when forming a window. The post-pad is larger
# because an assertion-bearing event (an error banner, a success toast) usually
# renders shortly after the action that triggers it.
WINDOW_PRE_PAD_MS = 2000
WINDOW_POST_PAD_MS = 3000

# Longest window handed to a single request. Cost here scales with frame count, not
# duration, so this is not a billing guard: it stops a long merged window from being
# described by a frame budget too sparse to cover it. Splitting is a last resort — each
# seam risks describing the same moment twice — so this is set well above the length of
# a typical POC recording.
MAX_WINDOW_MS = 30_000

# Frames sent per window. Enough to show an action and its consequence; beyond this the
# marginal frame is usually another still of an unchanged screen. Set above the event
# count of a typical recording so a window is rarely thinned or split.
MAX_FRAMES_PER_WINDOW = 12

# Frames sampled uniformly when detection finds nothing at all.
#
# Whole-frame differencing cannot separate a change smaller than the moving cursor from
# the cursor itself — a toggled checkbox measures the same as pointer drift. Returning
# an empty trace in that case is the dangerous outcome, because downstream it reads as
# "nothing happened" rather than "nothing was detected", and every expected step becomes
# a missing_action against a recording that may be perfectly faithful. Sampling the
# recording evenly instead keeps one model call as a floor under that failure.
FALLBACK_FRAME_COUNT = 6


def derive_action_windows(
    change_timestamps_ms: list[int],
    *,
    pre_pad_ms: int = WINDOW_PRE_PAD_MS,
    post_pad_ms: int = WINDOW_POST_PAD_MS,
    max_window_ms: int = MAX_WINDOW_MS,
) -> list[tuple[int, int]]:
    """Derive merged, length-bounded windows around a list of change timestamps.

    Each timestamp becomes a window [ts - pre_pad, ts + post_pad] (clamped at 0).
    Overlapping or touching windows are merged. Any merged window longer than
    max_window_ms is split into equal sub-windows. Returns ascending start order.
    """
    if not change_timestamps_ms:
        return []
    if max_window_ms <= 0:
        raise ValueError(f"max_window_ms must be positive, got {max_window_ms}")

    raw = sorted(
        (max(0, ts - pre_pad_ms), ts + post_pad_ms)
        for ts in change_timestamps_ms
    )

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    bounded: list[tuple[int, int]] = []
    for start, end in merged:
        span = end - start
        if span <= max_window_ms:
            bounded.append((start, end))
            continue
        sub_count = -(-span // max_window_ms)  # ceiling division
        sub_span = -(-span // sub_count)
        cursor = start
        while cursor < end:
            sub_end = min(cursor + sub_span, end)
            bounded.append((cursor, sub_end))
            # Resume past the seam rather than on it. Sharing an endpoint would put the
            # same frame in two windows, and two independent extractions of one frame
            # report the same action twice — surfacing later as a phantom extra_action.
            cursor = sub_end + 1
    return bounded


def _thin_evenly(values: list[int], max_count: int) -> list[int]:
    """Reduce values to at most max_count entries, always keeping the first and last."""
    if len(values) <= max_count:
        return values
    if max_count == 1:
        return [values[0]]
    step = (len(values) - 1) / (max_count - 1)
    return [values[round(i * step)] for i in range(max_count)]


def select_window_frames(
    start_ms: int,
    end_ms: int,
    change_timestamps_ms: list[int],
    *,
    max_frames: int = MAX_FRAMES_PER_WINDOW,
) -> list[int]:
    """Choose which frames to send for one window: its change events plus its edges.

    The window bounds are always included so the model sees the screen before the
    first change and after the last one — an action is only legible against the state
    it changed.
    """
    if end_ms <= start_ms:
        raise ValueError(f"end_ms ({end_ms}) must be greater than start_ms ({start_ms})")
    if max_frames < 2:
        raise ValueError(f"max_frames must be at least 2, got {max_frames}")

    in_window = [ts for ts in change_timestamps_ms if start_ms <= ts <= end_ms]
    candidates = sorted({start_ms, *in_window, end_ms})
    return _thin_evenly(candidates, max_frames)


def extract_observed_trace(
    openai_client: instructor.Instructor,
    video_path: Path,
    source_file: str,
    *,
    model: str = EXTRACTION_MODEL,
    max_frames_per_window: int = MAX_FRAMES_PER_WINDOW,
    on_retry: RetryCallback | None = None,
    latency: LatencyMetrics | None = None,
) -> ObservedTrace:
    """Run the full extraction over a recording and return one merged ObservedTrace.

    When detection finds nothing, the recording is sampled uniformly instead of
    returning an empty trace — see FALLBACK_FRAME_COUNT for why silence is the more
    dangerous answer.

    When latency is provided (argument or active latency_scope), records Stage 2a
    (change detection) and Stage 2b (per-window extraction) durations, window/frame
    counts, and retry backoff totals via a wrapped on_retry callback. Without a
    collector, on_retry is forwarded unchanged.
    """
    latency = resolve_latency(latency)
    retry_callback = bind_retry_callback(latency, on_retry)

    stage2a_started = monotonic_ms()
    change_events = detect_change_events(video_path)
    duration_ms = video_duration_ms(video_path)
    if latency is not None:
        latency.stage2a_ms = elapsed_ms(stage2a_started)
        logger.info(
            "stage2a duration_ms=%.1f change_events=%d",
            latency.stage2a_ms,
            len(change_events),
        )

    if not change_events:
        # Stand-in events on a uniform grid, so the fallback flows through the same
        # windowing and frame-selection path as a normally detected recording.
        stride_ms = duration_ms // (FALLBACK_FRAME_COUNT + 1)
        if stride_ms <= 0:
            if latency is not None:
                latency.stage2b_ms = 0.0
                latency.window_count = 0
            return ObservedTrace(source_file=source_file, actions=[])
        change_events = [stride_ms * i for i in range(1, FALLBACK_FRAME_COUNT + 1)]
    windows = [
        (start, min(end, duration_ms))
        for start, end in derive_action_windows(change_events)
        if start < duration_ms
    ]
    windows = [(start, end) for start, end in windows if end > start]
    if not windows:
        if latency is not None:
            latency.stage2b_ms = 0.0
            latency.window_count = 0
        return ObservedTrace(source_file=source_file, actions=[])

    if latency is not None:
        latency.window_count = len(windows)

    stage2b_started = monotonic_ms()
    chunks = [
        extract_window(
            openai_client,
            video_path,
            source_file,
            select_window_frames(start, end, change_events, max_frames=max_frames_per_window),
            clip_offset_ms=start,
            clip_end_ms=end,
            model=model,
            on_retry=retry_callback,
            latency=latency,
            window_index=index,
        )
        for index, (start, end) in enumerate(windows)
    ]
    if latency is not None:
        latency.stage2b_ms = elapsed_ms(stage2b_started)
        logger.info(
            "stage2b duration_ms=%.1f window_count=%d frame_count=%d "
            "retry_count=%d retry_sleep_s=%.3f",
            latency.stage2b_ms,
            latency.window_count,
            latency.frame_count,
            latency.retry_count,
            latency.retry_sleep_s,
        )
    return merge_observed_chunks(chunks)
