"""Re-check an already accepted Stage 2C candidate. No model call.

The candidate was validated once, against the deterministic audit. Ownership can
change which timestamps are anchors, so the same evidence, target, assertion,
and timestamp rules have to be applied again before the candidate may pass a step.
"""

from reckon.ownership import _proposal_claim as proposal_claim
from reckon.schema import (
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    CandidateEvidenceItem,
    CandidateEvidenceType,
    CandidateSearchStatus,
    ExpectedStep,
    ExpectedTrace,
    ObservedTrace,
    StepVerdict,
)
from reckon.stage2c.roles import (
    evidence_semantic_role,
    expectation_semantic_role,
    role_mismatch_reason,
)
from reckon.stage2c.to_assist import candidate_to_proposal, visible_text_supports_step
from reckon.stage2c.validate import STAGE2C_CONFIDENCE_FLOOR
from reckon.stage2c.validate import _anchor_bounds as anchor_bounds


def rejection_reason(
    item: CandidateEvidenceItem,
    step: ExpectedStep | None,
    expected: ExpectedTrace,
    audit: AuditResult,
    observed: ObservedTrace,
) -> str | None:
    """A rejection reason, or None when the accepted candidate still holds."""
    if item.confidence < STAGE2C_CONFIDENCE_FLOOR:
        return "low_confidence"
    if item.timestamp_start_ms is None or item.timestamp_end_ms is None:
        return "unknown_frame_id"
    if item.search_status != CandidateSearchStatus.candidate_found:
        return f"search_{item.search_status.value}"
    if item.evidence_type == CandidateEvidenceType.ACTION and step is not None:
        mismatch = role_mismatch_reason(
            expectation_semantic_role(step, expected),
            evidence_semantic_role(
                item.observed_description, list(item.visible_text)
            ),
        )
        if mismatch is not None:
            return mismatch
    if step is not None and not visible_text_supports_step(step, item):
        if item.evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME:
            return "visible_text_does_not_support_assertion"
        return "visible_text_does_not_prove_action"
    lower, upper = anchor_bounds(
        audit,
        item.expected_step_index,
        observed=observed,
        expected=expected,
    )
    if lower is not None and item.timestamp_end_ms < lower:
        return "before_previous_anchor"
    if upper is not None and item.timestamp_start_ms > upper:
        return "after_next_anchor"
    return None


def accepted_frame_proposal(item: CandidateEvidenceItem) -> AssistProposalRecord | None:
    """The proposal Stage 2C would emit, when it is still an accepted frame clear."""
    proposal = candidate_to_proposal(item)
    if proposal.status != AssistProposalStatus.accepted:
        return None
    if proposal.evidence_timestamp_ms is None:
        return None
    return proposal


def rejected_frame_proposal(
    item: CandidateEvidenceItem,
    reason: str,
) -> AssistProposalRecord:
    return candidate_to_proposal(item).model_copy(update={
        "status": AssistProposalStatus.rejected,
        "reason": reason,
        "matching_method": AssistMatchMethod.rejected_proposal,
    })


def _evidence_timestamp(
    record: AssistProposalRecord,
    observed: ObservedTrace,
    step: ExpectedStep | None,
    verdict: StepVerdict | None,
) -> int | None:
    if record.evidence_timestamp_ms is not None:
        return record.evidence_timestamp_ms
    claim = proposal_claim(record, step, verdict)
    if claim is None:
        return None
    for action in observed.actions:
        if action.index == claim:
            return action.timestamp_ms
    return None


def audit_with_standing(
    deterministic: AuditResult,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    standing: list[AssistProposalRecord],
) -> AuditResult:
    """Deterministic anchors plus timestamps of proposals that still stand."""
    steps = {step.index: step for step in expected.steps}
    by_step = {proposal.expected_step_index: proposal for proposal in standing}
    updated: list[StepVerdict] = []
    for verdict in deterministic.step_verdicts:
        record = by_step.get(verdict.expected_step_index)
        if record is None:
            updated.append(verdict)
            continue
        step = steps.get(verdict.expected_step_index)
        timestamp = _evidence_timestamp(record, observed, step, verdict)
        if timestamp is None:
            updated.append(verdict)
            continue
        updated.append(verdict.model_copy(update={
            "passed": True,
            "matched_observed_index": proposal_claim(record, step, verdict),
            "evidence_timestamp_ms": timestamp,
            "finding_types": [],
        }))
    return deterministic.model_copy(update={"step_verdicts": updated})


def first_usable_frame(
    items: list[CandidateEvidenceItem],
    *,
    step: ExpectedStep | None,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
) -> tuple[AssistProposalRecord | None, AssistProposalRecord | None]:
    """The first candidate that still passes, or a rejected proposal for the first failure."""
    failure: str | None = None
    failed: CandidateEvidenceItem | None = None
    for item in items:
        reason = rejection_reason(item, step, expected, audit, observed)
        if reason is None:
            proposal = accepted_frame_proposal(item)
            if proposal is not None:
                return proposal, None
            reason = "rejected"
        if failure is None:
            failure = reason
            failed = item
    if failure is None or failed is None:
        return None, None
    return None, rejected_frame_proposal(failed, failure)
