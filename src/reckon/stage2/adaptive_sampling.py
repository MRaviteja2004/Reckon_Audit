"""Opt-in denser Stage 2 frame sampling on static choice-control screens.

Detection reuses extraction observations (targets / screen text) and the shared
``is_choice_control_target`` heuristic — no extra classification LLM call.
Adaptive density is inactive unless ``STAGE2_ADAPTIVE_SAMPLING_POLICY=choice_control``.
"""

from dataclasses import dataclass
from typing import Sequence

from reckon.schema import ObservedAction
from reckon.tolerance import is_choice_control_target

# Reuse the choice-control noun matcher indirectly via is_choice_control_target;
# also accept the same nouns when they appear only in screen/evidence text.
_CHOICE_TEXT_HINTS = (
    "checkbox",
    "check box",
    "radio button",
    "radio",
    "toggle",
    "switch",
    "dropdown",
    "drop down",
    "terms and conditions",
    "i agree",
)


@dataclass(frozen=True)
class AdaptiveSamplingState:
    """Tracks whether denser sampling is active for the current form screen."""

    active: bool = False
    activated_at_ms: int | None = None
    screen_key: str | None = None


def _thin_evenly(values: list[int], max_count: int) -> list[int]:
    if len(values) <= max_count:
        return values
    if max_count <= 1:
        return values[:1]
    step = (len(values) - 1) / (max_count - 1)
    return [values[round(i * step)] for i in range(max_count)]


def _observation_text(action: ObservedAction) -> str:
    parts: list[str] = []
    for value in (action.target, action.screen_description, action.input_value):
        if value:
            parts.append(value)
    evidence = action.assertion_evidence
    if evidence is not None:
        for value in (evidence.visible_text, evidence.description):
            if value:
                parts.append(value)
    return " ".join(parts)


def _text_has_choice_hint(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _CHOICE_TEXT_HINTS)


def detect_choice_control_screen(
    actions: Sequence[ObservedAction],
) -> bool | None:
    """Return True/False when evidence is clear, or None when detection is uncertain.

    Empty observations are uncertain — callers must keep the legacy frame set.
    """
    if not actions:
        return None

    for action in actions:
        if is_choice_control_target(action.target):
            return True
        text = _observation_text(action)
        if text and (
            is_choice_control_target(text) or _text_has_choice_hint(text)
        ):
            return True
    return False


def dominant_screen_key(actions: Sequence[ObservedAction]) -> str | None:
    """Most common non-empty screen_description; None when unavailable."""
    counts: dict[str, int] = {}
    for action in actions:
        if not action.screen_description:
            continue
        key = action.screen_description.strip().lower()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda item: (counts[item], item))


def densify_window_frames(
    base_frames: Sequence[int],
    start_ms: int,
    end_ms: int,
    *,
    interval_ms: int,
    max_extra_frames: int,
    max_frames: int,
) -> list[int]:
    """Add evenly spaced timestamps inside the window without duplicating bases.

    Result is chronological, unique, and capped at ``max_frames``. When the cap
    binds, edges are preserved via even thinning.
    """
    if end_ms <= start_ms:
        return sorted(set(base_frames))
    if interval_ms <= 0 or max_extra_frames <= 0 or max_frames < 2:
        return sorted(set(base_frames))

    existing = set(base_frames)
    extras: list[int] = []
    cursor = start_ms
    while cursor <= end_ms and len(extras) < max_extra_frames:
        if cursor not in existing:
            extras.append(cursor)
            existing.add(cursor)
        cursor += interval_ms

    merged = sorted(existing)
    if len(merged) > max_frames:
        return _thin_evenly(merged, max_frames)
    return merged


def update_adaptive_state(
    state: AdaptiveSamplingState,
    actions: Sequence[ObservedAction],
    *,
    window_start_ms: int,
    interaction_window_ms: int,
) -> AdaptiveSamplingState:
    """Advance adaptive state from one window's observations.

    Starts on clear choice-control evidence. Stops on screen change, clear
    non-choice evidence, or when the interaction window expires. Uncertain
    detection leaves the prior state unchanged (aside from timeout).
    """
    if (
        state.active
        and state.activated_at_ms is not None
        and window_start_ms >= state.activated_at_ms + interaction_window_ms
    ):
        return AdaptiveSamplingState()

    detection = detect_choice_control_screen(actions)
    screen_key = dominant_screen_key(actions)

    if state.active:
        if (
            state.screen_key
            and screen_key
            and screen_key != state.screen_key
        ):
            return AdaptiveSamplingState()
        # Same screen (or uncertain identity): keep densifying until timeout.
        # Clear non-choice evidence without a screen key ends the session so a
        # navigated-away window that omitted screen_description still falls back.
        if detection is False and screen_key is None and state.screen_key is None:
            return AdaptiveSamplingState()
        return AdaptiveSamplingState(
            active=True,
            activated_at_ms=state.activated_at_ms,
            screen_key=screen_key or state.screen_key,
        )

    if detection is True:
        return AdaptiveSamplingState(
            active=True,
            activated_at_ms=window_start_ms,
            screen_key=screen_key,
        )

    return state


def frames_were_densified(
    base_frames: Sequence[int],
    densified_frames: Sequence[int],
) -> bool:
    """True when densification added at least one new timestamp."""
    return sorted(set(densified_frames)) != sorted(set(base_frames))
