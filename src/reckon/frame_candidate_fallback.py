"""Use an already validated Stage 2C candidate when ownership blocks a pass.

A proposal can take an observation before another step is considered, or lose
that observation to a step that was applied first. The blocked step, or the
step holding the observation, may already have a Stage 2C candidate from this
run. That candidate is reconsidered with the checks Stage 2C and proposal
conversion already apply. Nothing here calls a model, and a candidate that
fails those checks stays rejected.

Two passed steps never share an observation. Frame evidence that cites no
observation does not take one. A step that already passes is left on its
evidence unless releasing that observation lets a blocked step pass on evidence
that already survived its own checks.
"""

from reckon.ownership import (
    OWNED_BY_ANOTHER_STEP,
    as_owned_conflict,
    split_conflicting_claims,
)
from reckon.ownership import _proposal_claim as proposal_claim
from reckon.schema import (
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    CandidateEvidenceItem,
    CandidateEvidenceTrace,
    CandidateValidationStatus,
    ExpectedStep,
    ExpectedTrace,
    ObservedTrace,
    StepVerdict,
)
from reckon.stage2c.revalidate import audit_with_standing, first_usable_frame

# The holding proposal stopped claiming the observation because frame evidence
# already covers that step, so the blocked step can use the observation.
FRAME_EVIDENCE_RELEASED_OBSERVATION = "frame_evidence_released_observation"


def _proposal_identity(record: AssistProposalRecord) -> tuple:
    return (
        record.kind,
        record.expected_step_index,
        record.matched_observed_index,
        record.supporting_observed_index,
        record.initiation_observed_index,
        record.completion_observed_index,
        record.evidence_timestamp_ms,
        tuple(record.frame_ids),
        record.candidate_evidence_type,
    )


def _has_frame_proposal(proposals: list[AssistProposalRecord], step_index: int) -> bool:
    return any(
        proposal.kind == AssistCandidateKind.frame_evidence
        and proposal.expected_step_index == step_index
        for proposal in proposals
    )


def _claim_taken(
    proposal: AssistProposalRecord,
    step: ExpectedStep | None,
    verdict: StepVerdict | None,
    owners: dict[int, int],
) -> bool:
    claim = proposal_claim(proposal, step, verdict)
    if claim is None:
        return False
    return owners.get(claim, proposal.expected_step_index) != proposal.expected_step_index


def _owners(
    deterministic: AuditResult,
    expected: ExpectedTrace,
    standing: list[AssistProposalRecord],
) -> dict[int, int]:
    steps = {step.index: step for step in expected.steps}
    verdicts = {
        verdict.expected_step_index: verdict for verdict in deterministic.step_verdicts
    }
    owners = {
        verdict.matched_observed_index: verdict.expected_step_index
        for verdict in deterministic.step_verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    }
    for proposal in standing:
        claim = proposal_claim(
            proposal,
            steps.get(proposal.expected_step_index),
            verdicts.get(proposal.expected_step_index),
        )
        if claim is not None:
            owners[claim] = proposal.expected_step_index
    return owners


def _holder_for(
    claim: int,
    standing: list[AssistProposalRecord],
    steps: dict[int, ExpectedStep],
    verdicts: dict[int, StepVerdict],
) -> AssistProposalRecord | None:
    for proposal in standing:
        owned = proposal_claim(
            proposal,
            steps.get(proposal.expected_step_index),
            verdicts.get(proposal.expected_step_index),
        )
        if owned == claim:
            return proposal
    return None


def _sync_ownership(
    deterministic: AuditResult,
    expected: ExpectedTrace,
    result: list[AssistProposalRecord],
    standing: list[AssistProposalRecord],
) -> tuple[list[AssistProposalRecord], list[AssistProposalRecord]]:
    kept, conflicting = split_conflicting_claims(deterministic, standing, expected)
    if not conflicting:
        return result, kept
    conflict_ids = {id(proposal) for proposal in conflicting}
    synced = [
        as_owned_conflict(proposal) if id(proposal) in conflict_ids else proposal
        for proposal in result
    ]
    return synced, kept


def _recover_on_own_candidate(
    lost: AssistProposalRecord,
    items: list[CandidateEvidenceItem],
    *,
    deterministic: AuditResult,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    steps: dict[int, ExpectedStep],
    verdicts: dict[int, StepVerdict],
    result: list[AssistProposalRecord],
    standing_now: list[AssistProposalRecord],
) -> bool:
    """Pass ``lost`` on its own frame candidate when that candidate still holds."""
    step = steps.get(lost.expected_step_index)
    verdict = verdicts.get(lost.expected_step_index)
    audit = audit_with_standing(deterministic, expected, observed, standing_now)
    frame, rejected = first_usable_frame(
        items, step=step, expected=expected, observed=observed, audit=audit
    )
    if frame is not None and _claim_taken(
        frame, step, verdict, _owners(deterministic, expected, standing_now)
    ):
        rejected = frame.model_copy(update={
            "status": AssistProposalStatus.rejected,
            "reason": OWNED_BY_ANOTHER_STEP,
            "matching_method": AssistMatchMethod.rejected_proposal,
        })
        frame = None
    if frame is not None:
        result.append(frame)
        standing_now.append(frame)
        return True
    if rejected is not None and not _has_frame_proposal(result, lost.expected_step_index):
        result.append(rejected)
        return True
    return False


def _release_holder(
    lost: AssistProposalRecord,
    holder: AssistProposalRecord,
    frame: AssistProposalRecord,
    *,
    result: list[AssistProposalRecord],
    standing_now: list[AssistProposalRecord],
) -> tuple[list[AssistProposalRecord], list[AssistProposalRecord]]:
    demoted = holder.model_copy(update={
        "status": AssistProposalStatus.rejected,
        "reason": FRAME_EVIDENCE_RELEASED_OBSERVATION,
        "matching_method": AssistMatchMethod.rejected_proposal,
    })
    identity = _proposal_identity(lost)
    replaced = [
        demoted if proposal is holder else proposal for proposal in result
    ]
    restored: list[AssistProposalRecord] = []
    for proposal in replaced:
        if (
            proposal.status == AssistProposalStatus.rejected
            and proposal.reason == OWNED_BY_ANOTHER_STEP
            and _proposal_identity(proposal) == identity
        ):
            restored.append(lost)
            continue
        restored.append(proposal)
    if not any(proposal is frame for proposal in restored):
        restored.append(frame)
    standing = [proposal for proposal in standing_now if proposal is not holder]
    standing.extend([frame, lost])
    return restored, standing


def fallback_to_frame_candidates(
    deterministic: AuditResult,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    original_proposals: list[AssistProposalRecord],
    settled: list[AssistProposalRecord],
    standing: list[AssistProposalRecord],
    candidate_evidence: CandidateEvidenceTrace | None,
) -> tuple[list[AssistProposalRecord], list[AssistProposalRecord]]:
    """Reconsider validated Stage 2C candidates blocked by observation ownership.

    Returns the settled proposals and the ones that still stand. A missing trace
    changes nothing.
    """
    if candidate_evidence is None:
        return settled, standing
    accepted_items: dict[int, list[CandidateEvidenceItem]] = {}
    for item in candidate_evidence.candidates:
        if item.validation_status != CandidateValidationStatus.accepted:
            continue
        accepted_items.setdefault(item.expected_step_index, []).append(item)
    if not accepted_items:
        return settled, standing

    steps = {step.index: step for step in expected.steps}
    verdicts = {
        verdict.expected_step_index: verdict for verdict in deterministic.step_verdicts
    }
    originals = {
        _proposal_identity(proposal): proposal
        for proposal in original_proposals
        if proposal.status == AssistProposalStatus.accepted
    }
    displaced = [
        originals[_proposal_identity(proposal)]
        for proposal in settled
        if proposal.status == AssistProposalStatus.rejected
        and proposal.reason == OWNED_BY_ANOTHER_STEP
        and _proposal_identity(proposal) in originals
    ]
    if not displaced:
        return settled, standing

    result = list(settled)
    standing_now = list(standing)
    changed = False
    for lost in displaced:
        if lost.expected_step_index in {item.expected_step_index for item in standing_now}:
            continue
        items = accepted_items.get(lost.expected_step_index) or []
        if not items:
            continue
        changed = _recover_on_own_candidate(
            lost,
            items,
            deterministic=deterministic,
            expected=expected,
            observed=observed,
            steps=steps,
            verdicts=verdicts,
            result=result,
            standing_now=standing_now,
        ) or changed

    released: set[int] = set()
    for lost in displaced:
        if lost.expected_step_index in {item.expected_step_index for item in standing_now}:
            continue
        claim = proposal_claim(
            lost, steps.get(lost.expected_step_index), verdicts.get(lost.expected_step_index)
        )
        if claim is None or claim in released:
            continue
        holder = _holder_for(claim, standing_now, steps, verdicts)
        if holder is None or holder.expected_step_index == lost.expected_step_index:
            continue
        items = accepted_items.get(holder.expected_step_index) or []
        if not items:
            continue
        hypothetical = [
            proposal for proposal in standing_now if proposal is not holder
        ]
        hypothetical.append(lost)
        audit = audit_with_standing(deterministic, expected, observed, hypothetical)
        frame, _rejected = first_usable_frame(
            items,
            step=steps.get(holder.expected_step_index),
            expected=expected,
            observed=observed,
            audit=audit,
        )
        if frame is None or proposal_claim(
            frame,
            steps.get(holder.expected_step_index),
            verdicts.get(holder.expected_step_index),
        ) == claim:
            continue
        if _claim_taken(
            frame,
            steps.get(holder.expected_step_index),
            verdicts.get(holder.expected_step_index),
            _owners(deterministic, expected, hypothetical),
        ):
            continue
        trial = [proposal for proposal in hypothetical if proposal is not lost]
        trial.extend([frame, lost])
        kept, conflicting = split_conflicting_claims(deterministic, trial, expected)
        kept_ids = {id(proposal) for proposal in kept}
        if id(frame) not in kept_ids or id(lost) not in kept_ids or conflicting:
            continue
        result, standing_now = _release_holder(
            lost, holder, frame, result=result, standing_now=standing_now
        )
        released.add(claim)
        changed = True

    if not changed:
        return settled, standing
    return _sync_ownership(deterministic, expected, result, standing_now)
