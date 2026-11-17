"""Stage 2 event selection — choose a budgeted subset of change-event candidates.

The default ``legacy`` policy preserves the historical top-K-by-confidence then
chronological-sort behaviour. The optional ``coverage_aware`` policy spreads the
same budget across the timeline so a dense high-confidence burst cannot starve
later (or earlier) regions.
"""

from typing import Protocol, Sequence, TypeVar

from reckon import config

# Kept equal to keyframes.MAX_CHANGE_EVENTS; imported from keyframes would cycle
# because keyframes calls select_events at the event-budget cap.
_DEFAULT_MAX_EVENTS = 60
_MIN_BUCKET_SIZE_MS = 1000


class EventCandidate(Protocol):
    """Structural type for selectable Stage 2 change events."""

    @property
    def confidence(self) -> float: ...

    @property
    def timestamp_ms(self) -> int: ...


T = TypeVar("T", bound=EventCandidate)


def _legacy_select(candidates: Sequence[T], max_events: int) -> list[T]:
    selected = sorted(
        candidates,
        key=lambda x: (x.confidence, -x.timestamp_ms),
        reverse=True,
    )[:max_events]
    return sorted(selected, key=lambda x: x.timestamp_ms)


def _coverage_aware_select(candidates: Sequence[T], max_events: int) -> list[T]:
    if len(candidates) <= max_events:
        return sorted(candidates, key=lambda x: x.timestamp_ms)

    timestamps = [c.timestamp_ms for c in candidates]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    if max_ts == min_ts:
        return _legacy_select(candidates, max_events)

    duration = max_ts - min_ts
    bucket_size_ms = max(_MIN_BUCKET_SIZE_MS, duration / max_events)

    buckets: dict[int, list[T]] = {}
    for candidate in candidates:
        # Clamp so a timestamp exactly at max_ts lands in the final bucket
        # rather than creating a MAX_CHANGE_EVENTS+1st bucket.
        bucket_index = min(
            int((candidate.timestamp_ms - min_ts) / bucket_size_ms),
            max_events - 1,
        )
        buckets.setdefault(bucket_index, []).append(candidate)

    selected: list[T] = []
    selected_ids: set[int] = set()
    for bucket_index in sorted(buckets):
        local_best = max(
            buckets[bucket_index],
            key=lambda x: (x.confidence, -x.timestamp_ms),
        )
        selected.append(local_best)
        selected_ids.add(id(local_best))

    if len(selected) > max_events:
        selected = sorted(
            selected,
            key=lambda x: (x.confidence, -x.timestamp_ms),
            reverse=True,
        )[:max_events]
        selected_ids = {id(c) for c in selected}

    if len(selected) < max_events:
        unselected = [c for c in candidates if id(c) not in selected_ids]
        fillers = sorted(
            unselected,
            key=lambda x: (x.confidence, -x.timestamp_ms),
            reverse=True,
        )
        for candidate in fillers:
            if len(selected) >= max_events:
                break
            selected.append(candidate)

    return sorted(selected, key=lambda x: x.timestamp_ms)[:max_events]


def select_events(
    candidates: Sequence[T],
    *,
    policy: str | None = None,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> list[T]:
    """Select up to ``max_events`` candidates under the configured policy.

    Does not mutate ``candidates``. Returned objects are the original candidate
    instances (same identity and fields). Invalid or missing policies fall back
    to ``legacy``.
    """
    resolved = (
        policy if policy is not None else config.STAGE2_EVENT_SELECTION_POLICY
    )
    if resolved == "coverage_aware":
        return _coverage_aware_select(candidates, max_events)
    return _legacy_select(candidates, max_events)
