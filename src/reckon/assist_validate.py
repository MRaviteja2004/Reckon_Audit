"""Validate Stage 3 assist proposals against original traces. Claude never owns the verdict."""

from reckon.assist_propose import ASSIST_CONFIDENCE_FLOOR, _AssistDecision
from reckon.assist_select import AssistCandidate, texts_have_protected_contradiction
from reckon.schema import (
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    ExpectedTrace,
    ObservedTrace,
)
from reckon.tolerance import actions_compatible, assertion_satisfied


def _action_by_index(observed: ObservedTrace) -> dict[int, object]:
    return {action.index: action for action in observed.actions}


def _step_by_index(expected: ExpectedTrace) -> dict[int, object]:
    return {step.index: step for step in expected.steps}


def validate_deterministic_delayed(
    candidate: AssistCandidate,
    expected: ExpectedTrace,
    observed: ObservedTrace,
) -> AssistProposalRecord:
    """Accept delayed assertion support found by Python without an LLM call."""
    support_index = candidate.deterministic_support_index
    if support_index is None:
        return AssistProposalRecord(
            kind=candidate.kind,
            status=AssistProposalStatus.rejected,
            expected_step_index=candidate.expected_step_index,
            matched_observed_index=candidate.matched_observed_index,
            finding_types=list(candidate.finding_types),
            reason="no_deterministic_support",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    step = _step_by_index(expected).get(candidate.expected_step_index)
    action = _action_by_index(observed).get(support_index)
    if step is None or action is None or getattr(step, "assertion", None) is None:
        return AssistProposalRecord(
            kind=candidate.kind,
            status=AssistProposalStatus.rejected,
            expected_step_index=candidate.expected_step_index,
            matched_observed_index=candidate.matched_observed_index,
            finding_types=list(candidate.finding_types),
            reason="missing_trace_refs",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if not assertion_satisfied(
        step.assertion, action, expected_target=step.target
    ):
        return AssistProposalRecord(
            kind=candidate.kind,
            status=AssistProposalStatus.rejected,
            expected_step_index=candidate.expected_step_index,
            matched_observed_index=candidate.matched_observed_index,
            supporting_observed_index=support_index,
            finding_types=list(candidate.finding_types),
            reason="deterministic_support_unsatisfied",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    return AssistProposalRecord(
        kind=candidate.kind,
        status=AssistProposalStatus.accepted,
        expected_step_index=candidate.expected_step_index,
        matched_observed_index=candidate.matched_observed_index,
        supporting_observed_index=support_index,
        finding_types=list(candidate.finding_types),
        rationale="deterministic delayed assertion evidence",
        matching_method=AssistMatchMethod.deterministic,
    )


def validate_assist_decision(
    candidate: AssistCandidate,
    decision: _AssistDecision | None,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    *,
    parse_error: str | None = None,
) -> AssistProposalRecord:
    """Validate one model decision; reject anything that lacks grounded evidence."""
    base = dict(
        kind=candidate.kind,
        expected_step_index=candidate.expected_step_index,
        matched_observed_index=candidate.matched_observed_index,
        finding_types=list(candidate.finding_types),
    )

    if parse_error is not None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            reason=parse_error,
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if decision is None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            reason="missing_decision",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if decision.candidate_id != candidate.candidate_id:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=decision.answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="candidate_id_mismatch",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    answer = decision.answer.strip().lower()
    if answer in {"unresolved", "different", "unsupported"}:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason=f"model_{answer}",
            matching_method=AssistMatchMethod.unresolved
            if answer == "unresolved"
            else AssistMatchMethod.rejected_proposal,
        )
    if answer != "equivalent":
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="invalid_answer",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if decision.confidence < ASSIST_CONFIDENCE_FLOOR:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="low_confidence",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    step = _step_by_index(expected).get(candidate.expected_step_index)
    matched = _action_by_index(observed).get(candidate.matched_observed_index)
    if step is None or matched is None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="missing_trace_refs",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if not actions_compatible(step.action, matched.action):
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="incompatible_actions",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    if candidate.kind == AssistCandidateKind.wrong_target:
        if step.target is None or matched.target is None:
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer=answer,
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="missing_targets",
                matching_method=AssistMatchMethod.rejected_proposal,
            )
        if texts_have_protected_contradiction(step.target, matched.target):
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer=answer,
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="protected_contradiction",
                matching_method=AssistMatchMethod.rejected_proposal,
            )
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.accepted,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            matching_method=AssistMatchMethod.llm_assisted,
        )

    # delayed_assertion
    if step.assertion is None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="missing_assertion",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if decision.supporting_event_id is None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="missing_supporting_event_id",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    raw_id = decision.supporting_event_id.strip().upper()
    if not raw_id.startswith("O") or not raw_id[1:].isdigit():
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="invalid_supporting_event_id",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    support_index = int(raw_id[1:])
    allowed = {event.observed_index for event in candidate.delayed_events}
    if support_index not in allowed:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="supporting_event_not_offered",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    support = _action_by_index(observed).get(support_index)
    if support is None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="supporting_event_missing",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if support.timestamp_ms < matched.timestamp_ms or support.index <= matched.index:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            supporting_observed_index=support_index,
            reason="supporting_event_order_invalid",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    # Delayed asserts require structured assertion_evidence — never screen_description alone.
    evidence = support.assertion_evidence
    evidence_bits: list[str] = []
    if evidence is not None:
        if evidence.visible_text and evidence.visible_text.strip():
            evidence_bits.append(evidence.visible_text)
        if evidence.description and evidence.description.strip():
            evidence_bits.append(evidence.description)
    if not evidence_bits:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            supporting_observed_index=support_index,
            reason="llm_accepted_without_assertion_evidence_support",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    if any(
        texts_have_protected_contradiction(step.assertion, text)
        for text in evidence_bits
    ):
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            supporting_observed_index=support_index,
            reason="negative_or_contradictory_evidence",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    # LLM "equivalent" is never enough — Python must independently confirm the text.
    if not assertion_satisfied(
        step.assertion, support, expected_target=step.target
    ):
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer=answer,
            confidence=decision.confidence,
            rationale=decision.rationale,
            supporting_observed_index=support_index,
            reason="llm_accepted_without_assertion_evidence_support",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    return AssistProposalRecord(
        **base,
        status=AssistProposalStatus.accepted,
        model_answer=answer,
        confidence=decision.confidence,
        rationale=decision.rationale,
        supporting_observed_index=support_index,
        matching_method=AssistMatchMethod.llm_assisted,
    )
