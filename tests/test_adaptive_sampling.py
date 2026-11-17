"""Focused tests for Stage 2 adaptive choice-control frame sampling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from reckon import config
from reckon.schema import ActionType, AssertionEvidence, ObservedAction, ObservedTrace
from reckon.sampler import MAX_FRAMES_PER_WINDOW, extract_observed_trace, select_window_frames
from reckon.stage2.adaptive_sampling import (
    AdaptiveSamplingState,
    densify_window_frames,
    detect_choice_control_screen,
    dominant_screen_key,
    frames_were_densified,
    update_adaptive_state,
)

_VIDEO = Path("recording.mp4")
_SOURCE = "recording.mp4"


def _action(
    *,
    index: int = 1,
    timestamp_ms: int = 0,
    action: ActionType = ActionType.assert_visible,
    target: str | None = None,
    screen_description: str | None = None,
    visible_text: str | None = None,
    confidence: float = 0.9,
) -> ObservedAction:
    evidence = None
    if visible_text is not None:
        evidence = AssertionEvidence(
            description="on-screen text",
            visible_text=visible_text,
        )
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=action,
        target=target,
        assertion_evidence=evidence,
        confidence=confidence,
        screen_description=screen_description,
    )


def test_choice_control_screen_receives_denser_sampling():
    base = select_window_frames(153_000, 174_000, [153_261])
    denser = densify_window_frames(
        base,
        153_000,
        174_000,
        interval_ms=1000,
        max_extra_frames=8,
        max_frames=16,
    )
    assert len(denser) > len(base)
    assert 158_000 in denser or any(157_000 <= ts <= 159_000 for ts in denser)
    assert denser[0] == min(denser)
    assert denser == sorted(denser)


def test_normal_static_screen_retains_legacy_sampling():
    actions = [
        _action(
            target="Home map",
            screen_description="Driver home map is visible",
        )
    ]
    assert detect_choice_control_screen(actions) is False

    base = select_window_frames(0, 10_000, [5000])
    # Adaptive inactive → callers keep legacy frames unchanged.
    state = AdaptiveSamplingState()
    assert state.active is False
    assert base == select_window_frames(0, 10_000, [5000], max_frames=MAX_FRAMES_PER_WINDOW)


def test_screen_transition_ends_adaptive_sampling():
    state = AdaptiveSamplingState(
        active=True,
        activated_at_ms=150_000,
        screen_key="language preferences form",
    )
    next_actions = [
        _action(
            target="Join Mentorship button",
            screen_description="Mentorship confirmation success",
        )
    ]
    updated = update_adaptive_state(
        state,
        next_actions,
        window_start_ms=194_000,
        interaction_window_ms=45_000,
    )
    assert updated.active is False


def test_interaction_window_timeout_ends_adaptive_sampling():
    state = AdaptiveSamplingState(
        active=True,
        activated_at_ms=100_000,
        screen_key="language preferences form",
    )
    still_form = [
        _action(
            target="English checkbox",
            screen_description="language preferences form",
        )
    ]
    updated = update_adaptive_state(
        state,
        still_form,
        window_start_ms=100_000 + 45_000,
        interaction_window_ms=45_000,
    )
    assert updated.active is False


def test_frame_budget_is_respected():
    base = select_window_frames(0, 40_000, [1000, 20_000])
    denser = densify_window_frames(
        base,
        0,
        40_000,
        interval_ms=500,
        max_extra_frames=8,
        max_frames=12,
    )
    assert len(denser) <= 12
    assert denser == sorted(set(denser))
    assert denser[0] == min(denser)
    assert denser[-1] == max(denser)


def test_duplicate_frames_are_removed():
    base = [1000, 2000, 5000]
    denser = densify_window_frames(
        base,
        1000,
        5000,
        interval_ms=1000,
        max_extra_frames=8,
        max_frames=16,
    )
    assert denser == sorted(denser)
    assert len(denser) == len(set(denser))
    assert denser.count(1000) == 1
    assert denser.count(2000) == 1


def test_timestamps_remain_chronological():
    base = [9000, 1000, 5000]
    denser = densify_window_frames(
        base,
        1000,
        9000,
        interval_ms=1000,
        max_extra_frames=8,
        max_frames=16,
    )
    assert denser == sorted(denser)


def test_disabled_policy_path_matches_legacy_select_window_frames(monkeypatch):
    events = [5000]
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_SAMPLING_POLICY", "legacy")

    with (
        patch("reckon.sampler.detect_change_events", return_value=events),
        patch("reckon.sampler.video_duration_ms", return_value=20_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = ObservedTrace(
            source_file=_SOURCE,
            clip_offset_ms=3000,
            clip_end_ms=8000,
            actions=[],
        )
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)

    legacy_frames = select_window_frames(3000, 8000, events)
    assert mock_extract.call_count == 1
    assert mock_extract.call_args.args[3] == legacy_frames


def test_uncertain_detection_falls_back_safely():
    assert detect_choice_control_screen([]) is None

    state = AdaptiveSamplingState(
        active=True,
        activated_at_ms=150_000,
        screen_key="language preferences form",
    )
    # Empty / uncertain observations must not flip the state off or on by themselves
    # before timeout or a clear screen change.
    updated = update_adaptive_state(
        state,
        [],
        window_start_ms=160_000,
        interaction_window_ms=45_000,
    )
    assert updated.active is True
    assert updated.screen_key == "language preferences form"

    inactive = AdaptiveSamplingState()
    still_inactive = update_adaptive_state(
        inactive,
        [],
        window_start_ms=160_000,
        interaction_window_ms=45_000,
    )
    assert still_inactive.active is False


def test_choice_control_detection_from_target_and_screen_text():
    assert detect_choice_control_screen(
        [_action(target="English checkbox", screen_description="Language form")]
    ) is True
    assert detect_choice_control_screen(
        [
            _action(
                target="Form",
                screen_description="Terms and Conditions acceptance with I agree",
            )
        ]
    ) is True
    assert dominant_screen_key(
        [
            _action(screen_description="Language form"),
            _action(screen_description="Language form"),
            _action(screen_description="Other"),
        ]
    ) == "language form"


def test_frames_were_densified_helper():
    base = [0, 10_000]
    denser = densify_window_frames(
        base, 0, 10_000, interval_ms=1000, max_extra_frames=3, max_frames=16
    )
    assert frames_were_densified(base, denser)
    assert not frames_were_densified(base, base)


def test_update_state_starts_on_choice_control_evidence():
    state = AdaptiveSamplingState()
    updated = update_adaptive_state(
        state,
        [
            _action(
                target="Terms and Conditions checkbox",
                screen_description="Mentorship join form",
            )
        ],
        window_start_ms=153_000,
        interaction_window_ms=45_000,
    )
    assert updated.active is True
    assert updated.activated_at_ms == 153_000
    assert updated.screen_key == "mentorship join form"


def test_extract_trace_resamples_when_choice_controls_detected(monkeypatch):
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_SAMPLING_POLICY", "choice_control")
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_SAMPLE_INTERVAL_MS", 1000)
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_MAX_EXTRA_FRAMES", 8)
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_MAX_FRAMES_PER_WINDOW", 16)

    choice_chunk = ObservedTrace(
        source_file=_SOURCE,
        clip_offset_ms=151_000,
        clip_end_ms=174_000,
        actions=[
            _action(
                target="English checkbox",
                screen_description="Language preferences form",
                timestamp_ms=2000,
            )
        ],
    )

    with (
        patch("reckon.sampler.detect_change_events", return_value=[153_261]),
        patch("reckon.sampler.video_duration_ms", return_value=200_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = choice_chunk
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)

    assert mock_extract.call_count == 2
    first_frames = mock_extract.call_args_list[0].args[3]
    second_frames = mock_extract.call_args_list[1].args[3]
    assert len(second_frames) > len(first_frames)
    assert second_frames == sorted(second_frames)
    assert len(second_frames) == len(set(second_frames))
    assert len(second_frames) <= 16


def test_extract_trace_densifies_subsequent_window_while_active(monkeypatch):
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_SAMPLING_POLICY", "choice_control")
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_SAMPLE_INTERVAL_MS", 1000)
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_MAX_EXTRA_FRAMES", 8)
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_MAX_FRAMES_PER_WINDOW", 16)
    monkeypatch.setattr(config, "STAGE2_ADAPTIVE_INTERACTION_WINDOW_MS", 45_000)

    form_chunk = ObservedTrace(
        source_file=_SOURCE,
        clip_offset_ms=0,
        clip_end_ms=5_000,
        actions=[
            _action(
                index=1,
                target="Terms checkbox",
                screen_description="Mentorship join form",
            )
        ],
    )
    follow_chunk = ObservedTrace(
        source_file=_SOURCE,
        clip_offset_ms=55_000,
        clip_end_ms=65_000,
        actions=[
            _action(
                index=1,
                target="Join Mentorship",
                screen_description="Mentorship join form",
            )
        ],
    )

    with (
        patch("reckon.sampler.detect_change_events", return_value=[3000, 60_000]),
        patch("reckon.sampler.video_duration_ms", return_value=90_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        # First window: legacy then resample. Later window: densified up front.
        mock_extract.side_effect = [form_chunk, form_chunk, follow_chunk]
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)

    assert mock_extract.call_count == 3
    follow_frames = mock_extract.call_args_list[2].args[3]
    legacy_follow = select_window_frames(
        mock_extract.call_args_list[2].kwargs["clip_offset_ms"],
        mock_extract.call_args_list[2].kwargs["clip_end_ms"],
        [3000, 60_000],
    )
    assert len(follow_frames) > len(legacy_follow)
    assert follow_frames == sorted(set(follow_frames))
