"""Convert validated Stage 2C candidates into Stage 3 AssistProposalRecords."""

from reckon.schema import (
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    CandidateEvidenceItem,
    CandidateEvidenceTrace,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
)
from reckon.tolerance import _contains_assertion, _introduces_negation


def _finding_types_for(item: CandidateEvidenceItem) -> list[FindingType]:
    if item.evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME:
        # Unmatched assertion steps are often reported as missing_action.
        return [FindingType.assertion_not_observed, FindingType.missing_action]
    return [FindingType.missing_action]


def candidate_to_proposal(item: CandidateEvidenceItem) -> AssistProposalRecord:
    """Map one Stage 2C item to an assist proposal (accepted or rejected)."""
    evidence_ts = item.timestamp_start_ms
    if item.timestamp_end_ms is not None:
        evidence_ts = item.timestamp_end_ms

    if item.validation_status == CandidateValidationStatus.accepted:
        return AssistProposalRecord(
            kind=AssistCandidateKind.frame_evidence,
            status=AssistProposalStatus.accepted,
            expected_step_index=item.expected_step_index,
            finding_types=_finding_types_for(item),
            model_answer="frame_evidence",
            confidence=item.confidence,
            rationale=item.observed_description or None,
            reason=(
                "assertion_frame_supported"
                if item.evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME
                else "action_frame_supported"
            ),
            matching_method=AssistMatchMethod.llm_assisted,
            evidence_timestamp_ms=evidence_ts,
            frame_ids=list(item.frame_ids),
            candidate_evidence_type=item.evidence_type,
        )

    return AssistProposalRecord(
        kind=AssistCandidateKind.frame_evidence,
        status=AssistProposalStatus.rejected,
        expected_step_index=item.expected_step_index,
        finding_types=_finding_types_for(item),
        model_answer="frame_evidence",
        confidence=item.confidence,
        rationale=item.observed_description or None,
        reason=item.rejection_reason or "rejected",
        matching_method=AssistMatchMethod.rejected_proposal,
        evidence_timestamp_ms=evidence_ts,
        frame_ids=list(item.frame_ids),
        candidate_evidence_type=item.evidence_type,
    )


def visible_text_supports_step(step: ExpectedStep, item: CandidateEvidenceItem) -> bool:
    """Frame pixels support an assertion only when the visible text contains it.

    The model description is not evidence. A visible label does not prove a click,
    a navigation, or any other interaction.
    """
    if item.evidence_type != CandidateEvidenceType.ASSERTION_OUTCOME:
        return False
    if item.search_status != CandidateSearchStatus.candidate_found:
        return False
    if step.assertion is None:
        return False
    visible = " ".join(text.strip() for text in item.visible_text if text and text.strip())
    if not visible or _introduces_negation(step.assertion, visible):
        return False
    return _contains_assertion(step.assertion, visible)


def proposals_from_candidate_trace(
    trace: CandidateEvidenceTrace,
    expected: ExpectedTrace | None = None,
) -> list[AssistProposalRecord]:
    """Emit assist proposals for every Stage 2C candidate (accepted and rejected).

    When ``expected`` is provided, a validator accept is not a pass. The visible
    text has to support the step, and the description is ignored.
    """
    steps = {step.index: step for step in expected.steps} if expected is not None else {}
    proposals: list[AssistProposalRecord] = []
    for item in trace.candidates:
        proposal = candidate_to_proposal(item)
        step = steps.get(item.expected_step_index)
        if (
            proposal.status == AssistProposalStatus.accepted
            and step is not None
            and not visible_text_supports_step(step, item)
        ):
            proposal = proposal.model_copy(update={
                "status": AssistProposalStatus.rejected,
                "reason": (
                    "visible_text_does_not_support_assertion"
                    if item.evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME
                    else "visible_text_does_not_prove_action"
                ),
                "matching_method": AssistMatchMethod.rejected_proposal,
            })
        proposals.append(proposal)
    return proposals


def high_confidence_visual_skip_steps(
    trace: CandidateEvidenceTrace,
    *,
    confidence_floor: float = 0.85,
) -> set[int]:
    """Steps whose ASSERTION_OUTCOME evidence Stage 4 should not re-check visually."""
    skip: set[int] = set()
    for item in trace.candidates:
        if item.validation_status != CandidateValidationStatus.accepted:
            continue
        if item.evidence_type != CandidateEvidenceType.ASSERTION_OUTCOME:
            continue
        if item.confidence < confidence_floor:
            continue
        skip.add(item.expected_step_index)
    return skip
