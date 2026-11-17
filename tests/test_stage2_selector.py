"""Unit tests for Stage 2 event selection policies."""

from dataclasses import dataclass

from reckon.keyframes import MAX_CHANGE_EVENTS
from reckon.stage2.selector import select_events


@dataclass(frozen=True)
class Candidate:
    timestamp_ms: int
    confidence: float
    event_id: str = ""


def _legacy_reference(candidates: list[Candidate]) -> list[Candidate]:
    selected = sorted(
        candidates,
        key=lambda x: (x.confidence, -x.timestamp_ms),
        reverse=True,
    )[:MAX_CHANGE_EVENTS]
    return sorted(selected, key=lambda x: x.timestamp_ms)


def test_exact_legacy_behavior():
    candidates = [
        Candidate(timestamp_ms=i * 1000, confidence=(i % 7) / 10, event_id=str(i))
        for i in range(100)
    ]
    original = list(candidates)

    result_default = select_events(candidates)
    result_explicit = select_events(candidates, policy="legacy")
    result_invalid = select_events(candidates, policy="not-a-policy")
    expected = _legacy_reference(candidates)

    assert result_default == expected
    assert result_explicit == expected
    assert result_invalid == expected
    assert candidates == original
    assert [c.event_id for c in result_default] == [c.event_id for c in expected]
    assert [c.timestamp_ms for c in result_default] == [c.timestamp_ms for c in expected]


def test_temporal_coverage_distribution():
    # Dense high-confidence spike early (> budget), plus one clear local max late.
    # Legacy top-K keeps only the spike; coverage_aware must still keep the late peak.
    candidates = [
        Candidate(timestamp_ms=i * 100, confidence=0.99, event_id=f"spike-{i}")
        for i in range(MAX_CHANGE_EVENTS + 10)
    ] + [
        Candidate(timestamp_ms=120_000, confidence=0.55, event_id="late-peak"),
    ]

    legacy = select_events(candidates, policy="legacy")
    covered = select_events(candidates, policy="coverage_aware")

    legacy_ids = {c.event_id for c in legacy}
    covered_ids = {c.event_id for c in covered}

    assert "late-peak" not in legacy_ids
    assert "late-peak" in covered_ids
    assert any(c.timestamp_ms >= 100_000 for c in covered)
    assert min(c.timestamp_ms for c in covered) < 5_000


def test_hard_event_budget_enforcement():
    candidates = [
        Candidate(timestamp_ms=i * 500, confidence=0.5 + (i % 50) / 100, event_id=str(i))
        for i in range(MAX_CHANGE_EVENTS * 3)
    ]

    legacy = select_events(candidates, policy="legacy")
    covered = select_events(candidates, policy="coverage_aware")

    assert len(legacy) == MAX_CHANGE_EVENTS
    assert len(covered) == MAX_CHANGE_EVENTS
    assert len(legacy) <= MAX_CHANGE_EVENTS
    assert len(covered) <= MAX_CHANGE_EVENTS


def test_zero_duration_fallback():
    candidates = [
        Candidate(timestamp_ms=5_000, confidence=0.9, event_id="a"),
        Candidate(timestamp_ms=5_000, confidence=0.8, event_id="b"),
        Candidate(timestamp_ms=5_000, confidence=0.7, event_id="c"),
    ]
    # Pad past the budget so coverage_aware would otherwise bucket.
    candidates.extend(
        Candidate(timestamp_ms=5_000, confidence=0.1, event_id=f"pad-{i}")
        for i in range(MAX_CHANGE_EVENTS)
    )

    result = select_events(candidates, policy="coverage_aware")
    expected = _legacy_reference(candidates)

    assert result == expected
    assert all(c.timestamp_ms == 5_000 for c in result)


def test_chronological_output_ordering():
    candidates = [
        Candidate(timestamp_ms=t, confidence=c, event_id=str(i))
        for i, (t, c) in enumerate(
            [
                (90_000, 0.2),
                (1_000, 0.99),
                (50_000, 0.5),
                (10_000, 0.8),
                (70_000, 0.4),
            ]
        )
    ]
    # Add enough extras that both policies must trim.
    candidates.extend(
        Candidate(timestamp_ms=2_000 + i * 1_500, confidence=0.3, event_id=f"x-{i}")
        for i in range(MAX_CHANGE_EVENTS)
    )

    for policy in ("legacy", "coverage_aware", None, "bogus"):
        result = select_events(candidates, policy=policy)
        timestamps = [c.timestamp_ms for c in result]
        assert timestamps == sorted(timestamps)
        assert len(result) <= MAX_CHANGE_EVENTS
