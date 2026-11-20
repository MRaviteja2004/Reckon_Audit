"""Opt-in Stage 2A coverage of long quiet stretches.

Change detection opens a window only where the screen moved. A static screen or a
small interaction, such as ticking a checkbox, can leave a long stretch of the
recording unsampled, and Stage 2C can only search frames that were sampled. Each long
gap between windows gets one representative frame near its middle.

A frame in a gap is more visual evidence for later validation. It does not show that
anything happened there, and nothing here treats it as an observation.
"""

# Selected only when STAGE2_GAP_SAMPLING_POLICY=gap_coverage; "legacy" changes nothing.
GAP_COVERAGE_POLICY = "gap_coverage"


def find_coverage_gaps(
    windows: list[tuple[int, int]],
    duration_ms: int,
    *,
    min_gap_ms: int,
) -> list[tuple[int, int]]:
    """Stretches longer than ``min_gap_ms`` that no window covers, in time order.

    The stretch before the first window and after the last count, so a static
    opening screen is a gap like any other.
    """
    if duration_ms <= 0:
        return []
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(windows):
        if start - cursor > min_gap_ms:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration_ms - cursor > min_gap_ms:
        gaps.append((cursor, duration_ms))
    return gaps


def samples_for_policy(
    policy: str,
    windows: list[tuple[int, int]],
    duration_ms: int,
    existing_timestamps_ms: list[int],
    *,
    min_gap_ms: int,
    max_extra_frames: int,
) -> list[int]:
    """Gap samples under ``policy``. Any policy other than gap coverage samples nothing."""
    if policy != GAP_COVERAGE_POLICY:
        return []
    return gap_sample_timestamps(
        windows,
        duration_ms,
        existing_timestamps_ms,
        min_gap_ms=min_gap_ms,
        max_extra_frames=max_extra_frames,
    )


def gap_sample_timestamps(
    windows: list[tuple[int, int]],
    duration_ms: int,
    existing_timestamps_ms: list[int],
    *,
    min_gap_ms: int,
    max_extra_frames: int,
) -> list[int]:
    """One midpoint timestamp per long gap, longest gaps first when the cap binds.

    Every result lies inside the recording, is distinct from ``existing_timestamps_ms``,
    and the list is in time order. A gap too short to need a frame yields none.
    """
    if max_extra_frames <= 0:
        return []
    existing = set(existing_timestamps_ms)
    gaps = find_coverage_gaps(windows, duration_ms, min_gap_ms=min_gap_ms)
    longest_first = sorted(gaps, key=lambda gap: (gap[0] - gap[1], gap[0]))
    chosen: list[int] = []
    for start, end in longest_first:
        midpoint = min(start + (end - start) // 2, duration_ms - 1)
        if midpoint in existing or midpoint in chosen:
            continue
        chosen.append(midpoint)
        if len(chosen) == max_extra_frames:
            break
    return sorted(chosen)
