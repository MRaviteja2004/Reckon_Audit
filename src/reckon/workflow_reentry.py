"""Detect workflow re-entry observations (URL / deeplink launches).

Evidence recovery may re-anchor search windows after early recording noise when
a navigate/open-URL signal appears. This never equates a feed-card (or other
in-app control) click with a deeplink — those remain distinct scenario paths.
Needleman–Wunsch pairing is unchanged.
"""

import re

from reckon.schema import (
    ActionType,
    ExpectedStep,
    ExpectedTrace,
    ObservedAction,
    ObservedTrace,
)
from reckon.tolerance import assertion_satisfied

_URL_RE = re.compile(
    r"(?:https?://|[a-z][a-z0-9+.-]*://)",
    re.IGNORECASE,
)
_DEEPLINK_HINT_RE = re.compile(
    r"\b(?:deep\s*-?\s*link|universal\s+link|app\s+link|"
    r"launch\s+deeplink|"
    r"open\s+(?:the\s+)?(?:url|link)|paste(?:d)?\s+(?:the\s+)?(?:url|link))\b",
    re.IGNORECASE,
)

# Expected-step cues that mark an alternate in-app entry control (not a deeplink).
_ENTRY_CONTROL_RE = re.compile(
    r"\b(?:feed\s*card|home\s*card|banner\s*card|mentor\s+card)\b",
    re.IGNORECASE,
)

_REENTRY_ACTIONS = frozenset({
    ActionType.navigate,
    ActionType.other,
    ActionType.click,
})


def _action_text_blob(action: ObservedAction) -> str:
    parts = [action.target or "", action.screen_description or ""]
    if action.assertion_evidence is not None:
        parts.append(action.assertion_evidence.description or "")
        parts.append(action.assertion_evidence.visible_text or "")
    return " ".join(part for part in parts if part)


def is_workflow_reentry_action(action: ObservedAction) -> bool:
    """True when the observation is a URL / deeplink style workflow re-entry."""
    if action.action not in _REENTRY_ACTIONS:
        return False
    text = _action_text_blob(action)
    if not text.strip():
        return False
    return bool(_URL_RE.search(text) or _DEEPLINK_HINT_RE.search(text))


def workflow_reentry_indices(observed: ObservedTrace) -> list[int]:
    """Observed indices that look like URL / deeplink workflow re-entries."""
    return [
        action.index
        for action in observed.actions
        if is_workflow_reentry_action(action)
    ]


def primary_workflow_reentry(observed: ObservedTrace) -> ObservedAction | None:
    """Best deeplink / URL re-entry — prefer one immediately followed by landing proof.

    Notes-app copy/paste and URL-field focus are weaker than the launch that
    actually opens the destination workflow.
    """
    candidates = [
        action for action in observed.actions if is_workflow_reentry_action(action)
    ]
    if not candidates:
        return None

    def _score(action: ObservedAction) -> tuple[int, int]:
        # Higher is better. Prefer launch-button style, then post-entry landing text.
        blob = _action_text_blob(action).lower()
        launch = 2 if "launch" in blob and "deeplink" in blob.replace(" ", "") else (
            1 if "launch" in blob or "deeplink" in blob.replace("-", " ") else 0
        )
        followed = 0
        for later in observed.actions:
            if later.index <= action.index:
                continue
            if later.timestamp_ms > action.timestamp_ms + 15_000:
                break
            text = _action_text_blob(later).lower()
            if any(
                cue in text
                for cue in (
                    "mentor",
                    "mentorship",
                    "earn when you",
                    "i'm interested",
                    "im interested",
                )
            ):
                followed = 1
                break
        return (followed, launch, action.index)

    return max(candidates, key=_score)


def _step_blob(step: ExpectedStep) -> str:
    return " ".join(
        part
        for part in (step.target, step.raw_text, step.assertion)
        if part
    )


def _is_alternate_entry_step(step: ExpectedStep) -> bool:
    """True for in-app entry controls (feed card, …) that deeplink may replace."""
    if step.action not in {ActionType.click, ActionType.other, ActionType.navigate}:
        return False
    return bool(_ENTRY_CONTROL_RE.search(_step_blob(step)))


def reentry_lower_timestamp_ms(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    *,
    step_index: int,
) -> int | None:
    """Timestamp floor for post-entry steps after a deeplink re-entry.

    Applied only when a prior expected step is an alternate in-app entry control
    (e.g. feed card). The deeplink does not satisfy that entry step — it only
    re-anchors later evidence search / visual targeting.
    """
    prior_entry = any(
        step.index < step_index and _is_alternate_entry_step(step)
        for step in expected.steps
    )
    if not prior_entry:
        return None
    primary = primary_workflow_reentry(observed)
    if primary is None:
        return None
    return primary.timestamp_ms


def reentry_floor_for_step(
    step: ExpectedStep,
    observed: ObservedTrace,
    *,
    min_index: int,
    max_index: int,
) -> int:
    """Raise the evidence lower bound to a prior deeplink when proof follows it.

    Only applies when the step has an assertion and satisfying evidence exists at
    or after a re-entry inside ``[min_index, max_index]``. Does not rewrite DP
    matches and does not clear click findings for alternate entry controls.
    """
    if step.assertion is None:
        return min_index
    reentries = [
        index
        for index in workflow_reentry_indices(observed)
        if min_index <= index <= max_index
    ]
    if not reentries:
        return min_index

    best_floor = min_index
    for reentry in reentries:
        has_post_proof = any(
            assertion_satisfied(
                step.assertion, action, expected_target=step.target
            )
            for action in observed.actions
            if reentry <= action.index <= max_index
        )
        if has_post_proof and reentry > best_floor:
            best_floor = reentry
    return best_floor
