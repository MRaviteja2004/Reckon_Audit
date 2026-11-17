"""Integration tests wiring Stage 2 event selection into the keyframe pipeline."""

import json
from pathlib import Path

from reckon import config
from reckon.keyframes import MAX_CHANGE_EVENTS, _ScoredChangeEvent, _debounce
from reckon.sampler import derive_action_windows
from reckon.stage2.selector import select_events

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mentorship_change_event_candidates.json"
)


def _load_mentorship_candidates() -> tuple[list[_ScoredChangeEvent], int, int]:
    payload = json.loads(_FIXTURE.read_text())
    candidates = [
        _ScoredChangeEvent(
            timestamp_ms=row["timestamp_ms"],
            confidence=row["confidence"],
        )
        for row in payload["candidates"]
    ]
    return (
        candidates,
        payload["english_checkbox_timestamp_ms"],
        payload["terms_checkbox_timestamp_ms"],
    )


def _legacy_reference_timestamps(
    candidates: list[_ScoredChangeEvent],
) -> list[int]:
    """Exact pre-wiring top-K algorithm used by keyframes._debounce."""
    scored = [(c.timestamp_ms, c.confidence) for c in candidates]
    strongest = sorted(scored, key=lambda event: event[1], reverse=True)[
        :MAX_CHANGE_EVENTS
    ]
    return [timestamp_ms for timestamp_ms, _ in sorted(strongest)]


def test_mentorship_pipeline_legacy_matches_prewiring_selection(monkeypatch):
    candidates, english_ts, terms_ts = _load_mentorship_candidates()
    assert len(candidates) > MAX_CHANGE_EVENTS

    monkeypatch.setattr(config, "STAGE2_EVENT_SELECTION_POLICY", "legacy")

    # Pipeline path: already-debounced Mentorship pool enters the cap branch.
    scored = [(c.timestamp_ms, c.confidence) for c in candidates]
    selected = _debounce(scored, min_gap_ms=0, max_events=MAX_CHANGE_EVENTS)
    expected = _legacy_reference_timestamps(candidates)

    assert selected == expected
    assert english_ts not in selected
    assert terms_ts not in selected
    assert len(selected) == MAX_CHANGE_EVENTS
    assert selected == sorted(selected)

    windows = derive_action_windows(selected)
    assert windows == sorted(windows, key=lambda window: window[0])


def test_mentorship_pipeline_coverage_aware_retains_checkbox_events(monkeypatch):
    candidates, english_ts, terms_ts = _load_mentorship_candidates()

    monkeypatch.setattr(config, "STAGE2_EVENT_SELECTION_POLICY", "coverage_aware")

    scored = [(c.timestamp_ms, c.confidence) for c in candidates]
    selected = _debounce(scored, min_gap_ms=0, max_events=MAX_CHANGE_EVENTS)

    assert len(selected) <= MAX_CHANGE_EVENTS
    assert len(selected) == MAX_CHANGE_EVENTS
    assert selected == sorted(selected)
    assert english_ts in selected
    assert terms_ts in selected

    # Chronological selection is what window construction consumes.
    windows = derive_action_windows(selected)
    assert windows == sorted(windows, key=lambda window: window[0])
    assert any(start <= english_ts <= end for start, end in windows)
    assert any(start <= terms_ts <= end for start, end in windows)


def test_mentorship_missing_policy_defaults_to_legacy(monkeypatch):
    candidates, english_ts, terms_ts = _load_mentorship_candidates()
    monkeypatch.setattr(config, "STAGE2_EVENT_SELECTION_POLICY", "legacy")

    via_config = select_events(candidates, max_events=MAX_CHANGE_EVENTS)
    via_explicit = select_events(
        candidates, policy="legacy", max_events=MAX_CHANGE_EVENTS
    )
    expected = _legacy_reference_timestamps(candidates)

    assert [c.timestamp_ms for c in via_config] == expected
    assert [c.timestamp_ms for c in via_explicit] == expected
    assert english_ts not in expected
    assert terms_ts not in expected
