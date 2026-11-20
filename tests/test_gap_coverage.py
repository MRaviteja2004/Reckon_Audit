"""Quiet stretches get one midpoint frame, and only when that policy is selected."""

from reckon.persistence.extraction import frame_manifest_for
from reckon.schema import ActionType, ObservedAction, ObservedTrace
from reckon.stage2.gap_coverage import GAP_COVERAGE_POLICY, samples_for_policy

WINDOWS = [(20_000, 25_000), (40_000, 45_000)]
DURATION_MS = 60_000
EXISTING = [20_000, 22_000, 40_000, 44_000]
MIN_GAP_MS = 8_000


def _samples(
    *,
    policy: str = GAP_COVERAGE_POLICY,
    duration_ms: int = DURATION_MS,
    existing: list[int] | None = None,
    min_gap_ms: int = MIN_GAP_MS,
    max_extra_frames: int = 8,
) -> list[int]:
    return samples_for_policy(
        policy,
        WINDOWS,
        duration_ms,
        EXISTING if existing is None else existing,
        min_gap_ms=min_gap_ms,
        max_extra_frames=max_extra_frames,
    )


def test_long_gaps_receive_one_midpoint_each():
    assert _samples() == [10_000, 32_500, 52_500]


def test_short_gaps_receive_no_extra_sample():
    assert _samples(min_gap_ms=30_000) == []


def test_extra_frame_cap_keeps_the_longest_gaps():
    assert _samples(max_extra_frames=2) == [10_000, 32_500]


def test_a_midpoint_already_selected_is_not_added_again():
    assert 10_000 not in _samples(existing=[*EXISTING, 10_000])


def test_samples_stay_inside_the_recording_and_in_order():
    chosen = _samples()
    assert chosen == sorted(chosen)
    assert chosen == sorted(set(chosen))
    assert all(0 <= timestamp < DURATION_MS for timestamp in chosen)


def test_legacy_policy_adds_no_samples():
    assert _samples(policy="legacy") == []
    assert _samples(policy="") == []


def test_gap_samples_reach_stage2c_through_the_existing_manifest():
    observed = ObservedTrace(
        source_file="quiet.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=20_000,
                action=ActionType.click,
                target="Continue",
                confidence=0.9,
            )
        ],
        gap_sample_timestamps_ms=[10_000],
    )
    manifest = frame_manifest_for(observed)
    assert manifest is not None
    assert [frame.timestamp_ms for frame in manifest.frames] == [10_000, 20_000]


def test_a_trace_without_gap_samples_still_loads():
    loaded = ObservedTrace.model_validate({
        "source_file": "saved.mp4",
        "actions": [],
    })
    assert loaded.gap_sample_timestamps_ms == []
    assert frame_manifest_for(loaded) is None
