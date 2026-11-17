"""Validate evidence-aware Claude decisions. Claude never owns the verdict."""

from reckon.assist_evidence_propose import (
    EVIDENCE_CONFIDENCE_FLOOR,
    _EvidenceDecision,
)
from reckon.assist_evidence_select import EvidenceBundleCandidate
from reckon.assist_select import texts_have_protected_contradiction
from reckon.schema import (
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    EvidenceRole,
    ExpectedTrace,
    ObservedTrace,
)
from reckon.tolerance import assertion_satisfied


def _parse_event_id(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip().upper()
    if not text.startswith("O") or not text[1:].isdigit():
        return None
    return int(text[1:])


def _action_by_index(observed: ObservedTrace) -> dict[int, object]:
    return {action.index: action for action in observed.actions}


def _step_by_index(expected: ExpectedTrace) -> dict[int, object]:
    return {step.index: step for step in expected.steps}


def _challenge_eligible(
    *,
    candidate: EvidenceBundleCandidate,
    matched_is_distractor: bool,
    matched_assertion_ok: bool,
    validated_support: int | None,
) -> bool:
    """False-pass challenge is separate from pass acceptance.

    Eligible when the DP match is an explicit distractor, the matched observation
    does not satisfy the assertion, the baseline pass was already suspicious, and
    no independently validated assertion support was accepted. Confidence floor
    does not gate this path.
    """
    return (
        candidate.suspicious_pass
        and matched_is_distractor
        and not matched_assertion_ok
        and validated_support is None
    )


def validate_evidence_decision(
    candidate: EvidenceBundleCandidate,
    decision: _EvidenceDecision | None,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    *,
    parse_error: str | None = None,
) -> AssistProposalRecord:
    """Validate one evidence-bundle decision; reject anything ungrounded."""
    base = dict(
        kind=AssistCandidateKind.evidence_bundle,
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
            model_answer="labels",
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="candidate_id_mismatch",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    # Parse/ground labels before applying the pass-acceptance confidence floor so
    # false-pass challenges can still evaluate below that floor.
    allowed = {event.observed_index for event in candidate.evidence_events}
    role_by_index: dict[int, EvidenceRole] = {}
    for label in decision.labels:
        idx = _parse_event_id(label.observed_event_id)
        if idx is None or idx not in allowed:
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer="labels",
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="label_event_not_offered",
                matching_method=AssistMatchMethod.rejected_proposal,
            )
        role_by_index[idx] = label.role

    initiation_idx = _parse_event_id(decision.best_initiation_id)
    completion_idx = _parse_event_id(decision.best_completion_id)
    support_idx = _parse_event_id(decision.best_assertion_support_id)

    for idx, name in (
        (initiation_idx, "initiation"),
        (completion_idx, "completion"),
        (support_idx, "assertion_support"),
    ):
        if idx is None:
            continue
        if idx not in allowed:
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer="labels",
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason=f"{name}_not_offered",
                matching_method=AssistMatchMethod.rejected_proposal,
            )

    if initiation_idx is not None:
        role = role_by_index.get(initiation_idx)
        if role not in {EvidenceRole.initiation, None}:
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer="labels",
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="initiation_role_mismatch",
                matching_method=AssistMatchMethod.rejected_proposal,
                initiation_observed_index=initiation_idx,
            )

    if completion_idx is not None and initiation_idx is not None:
        if completion_idx < initiation_idx:
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer="labels",
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="completion_before_initiation",
                matching_method=AssistMatchMethod.rejected_proposal,
                initiation_observed_index=initiation_idx,
                completion_observed_index=completion_idx,
            )

    if support_idx is not None and initiation_idx is not None:
        if support_idx < initiation_idx:
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer="labels",
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="assertion_before_initiation",
                matching_method=AssistMatchMethod.rejected_proposal,
                initiation_observed_index=initiation_idx,
                supporting_observed_index=support_idx,
            )

    step = _step_by_index(expected).get(candidate.expected_step_index)
    actions = _action_by_index(observed)
    if step is None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer="labels",
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="missing_trace_refs",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    matched_idx = candidate.matched_observed_index
    matched_action = actions.get(matched_idx) if matched_idx is not None else None
    matched_is_distractor = (
        matched_idx is not None
        and role_by_index.get(matched_idx) == EvidenceRole.distractor
    )
    matched_assertion_ok = bool(
        getattr(step, "assertion", None)
        and matched_action is not None
        and assertion_satisfied(
            step.assertion, matched_action, expected_target=step.target
        )
    )

    # Validate optional assertion support independently. Never treat completion
    # alone as a pass without assertion_satisfied.
    validated_support: int | None = None
    if support_idx is not None and getattr(step, "assertion", None):
        support_action = actions.get(support_idx)
        if support_action is None:
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer="labels",
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="missing_support_action",
                matching_method=AssistMatchMethod.rejected_proposal,
                supporting_observed_index=support_idx,
            )
        support_text = " ".join(
            part
            for part in (
                getattr(
                    getattr(support_action, "assertion_evidence", None),
                    "visible_text",
                    None,
                ),
                getattr(
                    getattr(support_action, "assertion_evidence", None),
                    "description",
                    None,
                ),
                getattr(support_action, "screen_description", None),
                getattr(support_action, "target", None),
            )
            if part
        )
        if texts_have_protected_contradiction(step.assertion, support_text):
            return AssistProposalRecord(
                **base,
                status=AssistProposalStatus.rejected,
                model_answer="labels",
                confidence=decision.confidence,
                rationale=decision.rationale,
                reason="protected_contradiction",
                matching_method=AssistMatchMethod.rejected_proposal,
                supporting_observed_index=support_idx,
                initiation_observed_index=initiation_idx,
                completion_observed_index=completion_idx,
            )
        if assertion_satisfied(
            step.assertion, support_action, expected_target=step.target
        ):
            role = role_by_index.get(support_idx)
            if role not in {
                EvidenceRole.assertion_evidence,
                EvidenceRole.completion,
                None,
            }:
                return AssistProposalRecord(
                    **base,
                    status=AssistProposalStatus.rejected,
                    model_answer="labels",
                    confidence=decision.confidence,
                    rationale=decision.rationale,
                    reason="assertion_role_mismatch",
                    matching_method=AssistMatchMethod.rejected_proposal,
                    supporting_observed_index=support_idx,
                )
            validated_support = support_idx

    if _challenge_eligible(
        candidate=candidate,
        matched_is_distractor=matched_is_distractor,
        matched_assertion_ok=matched_assertion_ok,
        validated_support=validated_support,
    ):
        # Challenge removes an unsupported pass only. It never creates a pass
        # and does not require the pass-acceptance confidence floor.
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.accepted,
            model_answer="challenge",
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="challenge_false_pass",
            matching_method=AssistMatchMethod.llm_assisted,
            initiation_observed_index=initiation_idx,
            completion_observed_index=completion_idx,
            supporting_observed_index=None,
            challenged_false_pass=True,
        )

    # Pass acceptance (clearing assertion_not_observed) still requires the floor.
    if decision.confidence < EVIDENCE_CONFIDENCE_FLOOR:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer="labels",
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason="low_confidence",
            matching_method=AssistMatchMethod.rejected_proposal,
            initiation_observed_index=initiation_idx,
            completion_observed_index=completion_idx,
            supporting_observed_index=support_idx,
        )

    if validated_support is None:
        return AssistProposalRecord(
            **base,
            status=AssistProposalStatus.rejected,
            model_answer="labels",
            confidence=decision.confidence,
            rationale=decision.rationale,
            reason=(
                "insufficient_evidence"
                if not matched_is_distractor
                else "no_assertion_support"
            ),
            matching_method=AssistMatchMethod.rejected_proposal,
            initiation_observed_index=initiation_idx,
            completion_observed_index=completion_idx,
        )

    return AssistProposalRecord(
        **base,
        status=AssistProposalStatus.accepted,
        model_answer="support",
        confidence=decision.confidence,
        rationale=decision.rationale,
        reason="assertion_supported",
        matching_method=AssistMatchMethod.llm_assisted,
        initiation_observed_index=initiation_idx,
        completion_observed_index=completion_idx,
        supporting_observed_index=validated_support,
        challenged_false_pass=False,
    )
