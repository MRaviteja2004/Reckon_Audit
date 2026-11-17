"""Select Stage 3 assist candidates after deterministic alignment.

First slice:
  - semantic wrong_target on already-matched pairs
  - delayed assertion evidence from later ObservedTrace text

Composite many-to-one selection is intentionally not implemented yet; keep
AssistCandidateKind and this module's return shape stable so it can be added later.
"""

from dataclasses import dataclass, field

from reckon.schema import (
    AssistCandidateKind,
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedAction,
    ObservedTrace,
)
from reckon.tolerance import (
    _DIGIT_RUN,
    _introduces_negation,
    _normalise_assertion_text,
    actions_compatible,
    assertion_satisfied,
)

# Later observations considered for delayed assertion text (ms after match).
DELAYED_ASSERTION_WINDOW_MS = 10_000
# Hard cap on later events offered per assertion candidate.
MAX_DELAYED_EVIDENCE_EVENTS = 8

_OPPOSITE_STATE_PHRASES = (
    (("enabled",), ("disabled",)),
    (("checked",), ("unchecked",)),
    (("success",), ("failure",)),
    (("completed",), ("pending",)),
    (("logged", "in"), ("logged", "out")),
    (("approved",), ("declined",)),
)


@dataclass(frozen=True)
class AssistEvidenceEvent:
    """One later observed action offered as delayed assertion evidence."""

    observed_index: int
    timestamp_ms: int
    action: str
    target: str | None
    visible_text: str | None
    evidence_description: str | None
    screen_description: str | None


@dataclass(frozen=True)
class AssistCandidate:
    """One eligible finding for Stage 3 LLM (or deterministic delayed) assist."""

    candidate_id: str
    kind: AssistCandidateKind
    expected_step_index: int
    matched_observed_index: int
    finding_types: tuple[FindingType, ...]
    expected_action: str
    observed_action: str
    expected_text: str
    observed_text: str
    observed_visible_text: str | None = None
    observed_evidence_description: str | None = None
    observed_screen_description: str | None = None
    delayed_events: tuple[AssistEvidenceEvent, ...] = field(default_factory=tuple)
    # When set, Python already found supporting delayed evidence — no LLM needed.
    deterministic_support_index: int | None = None


def _contains_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(
        tuple(tokens[start:start + width]) == phrase
        for start in range(len(tokens) - width + 1)
    )


def texts_have_protected_contradiction(expected_text: str, observed_text: str) -> bool:
    """Return True when observed text contradicts expected on digits, negation, or state."""
    expected_digits = _DIGIT_RUN.findall(expected_text)
    observed_digits = _DIGIT_RUN.findall(observed_text)
    if expected_digits and observed_digits and expected_digits != observed_digits:
        return True
    if (
        _introduces_negation(expected_text, observed_text)
        or _introduces_negation(observed_text, expected_text)
    ):
        return True

    expected_tokens = _normalise_assertion_text(expected_text)
    observed_tokens = _normalise_assertion_text(observed_text)
    return any(
        (
            _contains_phrase(expected_tokens, left)
            and _contains_phrase(observed_tokens, right)
        )
        or (
            _contains_phrase(expected_tokens, right)
            and _contains_phrase(observed_tokens, left)
        )
        for left, right in _OPPOSITE_STATE_PHRASES
    )


def _evidence_text(action: ObservedAction) -> str | None:
    if action.assertion_evidence is None:
        return None
    evidence = action.assertion_evidence
    text = evidence.visible_text or evidence.description
    if text is None or not text.strip():
        return None
    return text


def _findings_by_matched_pair(
    audit_result: AuditResult,
) -> dict[tuple[int, int], set[FindingType]]:
    grouped: dict[tuple[int, int], set[FindingType]] = {}
    for finding in audit_result.findings:
        if finding.expected_step_index is None or finding.observed_action_index is None:
            continue
        key = (finding.expected_step_index, finding.observed_action_index)
        grouped.setdefault(key, set()).add(finding.type)
    return grouped


def _later_evidence_events(
    matched: ObservedAction,
    observed_trace: ObservedTrace,
    assertion: str,
) -> list[AssistEvidenceEvent]:
    events: list[AssistEvidenceEvent] = []
    for action in observed_trace.actions:
        if action.index <= matched.index:
            continue
        if action.timestamp_ms < matched.timestamp_ms:
            continue
        if action.timestamp_ms - matched.timestamp_ms > DELAYED_ASSERTION_WINDOW_MS:
            continue
        text = _evidence_text(action)
        screen = action.screen_description
        if text is None and (screen is None or not screen.strip()):
            continue
        probe = text or screen or ""
        if texts_have_protected_contradiction(assertion, probe):
            continue
        events.append(
            AssistEvidenceEvent(
                observed_index=action.index,
                timestamp_ms=action.timestamp_ms,
                action=action.action.value,
                target=action.target,
                visible_text=(
                    action.assertion_evidence.visible_text
                    if action.assertion_evidence
                    else None
                ),
                evidence_description=(
                    action.assertion_evidence.description
                    if action.assertion_evidence
                    else None
                ),
                screen_description=action.screen_description,
            )
        )
        if len(events) >= MAX_DELAYED_EVIDENCE_EVENTS:
            break
    return events


def _deterministic_delayed_support(
    step: ExpectedStep,
    events: list[AssistEvidenceEvent],
    action_by_index: dict[int, ObservedAction],
) -> int | None:
    if step.assertion is None:
        return None
    for event in events:
        action = action_by_index[event.observed_index]
        if assertion_satisfied(
            step.assertion, action, expected_target=step.target
        ):
            return action.index
    return None


def select_assist_candidates(
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
    audit_result: AuditResult,
) -> list[AssistCandidate]:
    """Return only findings where Stage 3 assist can add value without rematching DP."""
    step_by_index = {step.index: step for step in expected_trace.steps}
    action_by_index = {action.index: action for action in observed_trace.actions}
    findings_by_pair = _findings_by_matched_pair(audit_result)

    candidates: list[AssistCandidate] = []
    next_id = 1

    for step_verdict in audit_result.step_verdicts:
        matched_index = step_verdict.matched_observed_index
        if matched_index is None:
            continue
        step = step_by_index.get(step_verdict.expected_step_index)
        action = action_by_index.get(matched_index)
        if step is None or action is None:
            continue
        if not actions_compatible(step.action, action.action):
            continue

        pair_findings = findings_by_pair.get((step.index, action.index), set())
        if FindingType.wrong_input_value in pair_findings:
            continue

        # Delayed assertion: prefer later text evidence over the matched action alone.
        if (
            FindingType.assertion_not_observed in pair_findings
            and step.assertion is not None
        ):
            later = _later_evidence_events(action, observed_trace, step.assertion)
            if later:
                support = _deterministic_delayed_support(step, later, action_by_index)
                candidates.append(
                    AssistCandidate(
                        candidate_id=f"C{next_id}",
                        kind=AssistCandidateKind.delayed_assertion,
                        expected_step_index=step.index,
                        matched_observed_index=action.index,
                        finding_types=(FindingType.assertion_not_observed,),
                        expected_action=step.action.value,
                        observed_action=action.action.value,
                        expected_text=step.assertion,
                        observed_text=_evidence_text(action) or action.target or "",
                        observed_visible_text=(
                            action.assertion_evidence.visible_text
                            if action.assertion_evidence
                            else None
                        ),
                        observed_evidence_description=(
                            action.assertion_evidence.description
                            if action.assertion_evidence
                            else None
                        ),
                        observed_screen_description=action.screen_description,
                        delayed_events=tuple(later),
                        deterministic_support_index=support,
                    )
                )
                next_id += 1
                continue

        if (
            FindingType.wrong_target in pair_findings
            and step.target is not None
            and action.target is not None
            and not texts_have_protected_contradiction(step.target, action.target)
        ):
            candidates.append(
                AssistCandidate(
                    candidate_id=f"C{next_id}",
                    kind=AssistCandidateKind.wrong_target,
                    expected_step_index=step.index,
                    matched_observed_index=action.index,
                    finding_types=(FindingType.wrong_target,),
                    expected_action=step.action.value,
                    observed_action=action.action.value,
                    expected_text=step.target,
                    observed_text=action.target,
                )
            )
            next_id += 1

    return candidates
