"""Ownership is applied before a later proposal is treated as already covered."""

from reckon.assist_apply import apply_assist_proposals
from reckon.ownership import OWNED_BY_ANOTHER_STEP
from reckon.proposal_selection import settle_proposals, steps_with_standing_proposal
from reckon.schema import (
    ActionType,
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    AuditVerdict,
    CandidateEvidenceType,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)


def _trace() -> tuple[ExpectedTrace, ObservedTrace, AuditResult]:
    expected = ExpectedTrace(
        scenario_title="Two steps",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Save"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="receipt",
                assertion="Saved",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="two.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Save",
                confidence=0.9,
            ),
            ObservedAction(
                index=2,
                timestamp_ms=2000,
                action=ActionType.assert_visible,
                target="receipt",
                confidence=0.9,
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.5,
        findings=[
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.critical,
                expected_step_index=2,
                description="Expected step 2 was not observed.",
            )
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                finding_types=[FindingType.missing_action],
            ),
        ],
    )
    return expected, observed, audit


def _episode(observation: int) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=AssistCandidateKind.episode_verification,
        status=AssistProposalStatus.accepted,
        expected_step_index=2,
        matched_observed_index=observation,
        supporting_observed_index=observation,
        completion_observed_index=observation,
        matching_method=AssistMatchMethod.deterministic,
        reason="episode_verified",
    )


def _frame(*, observation: int | None) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=AssistCandidateKind.frame_evidence,
        status=AssistProposalStatus.accepted,
        expected_step_index=2,
        supporting_observed_index=observation,
        finding_types=[FindingType.missing_action],
        confidence=0.9,
        matching_method=AssistMatchMethod.llm_assisted,
        reason="assertion_frame_supported",
        evidence_timestamp_ms=4000,
        frame_ids=["F9"],
        candidate_evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
    )


def _passed_observations(audit: AuditResult) -> list[int]:
    return [
        verdict.matched_observed_index
        for verdict in audit.step_verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    ]


def test_demoted_episode_proposal_yields_to_valid_frame_evidence():
    expected, observed, audit = _trace()
    episode = _episode(1)
    frame = _frame(observation=None)

    assert 2 not in steps_with_standing_proposal(audit, expected, [episode])

    settled, standing = settle_proposals(audit, expected, [episode, frame])
    by_kind = {proposal.kind: proposal for proposal in settled}
    assert by_kind[AssistCandidateKind.episode_verification].status == (
        AssistProposalStatus.rejected
    )
    assert by_kind[AssistCandidateKind.episode_verification].reason == OWNED_BY_ANOTHER_STEP
    assert standing == [frame]

    refined = apply_assist_proposals(audit, standing, expected, observed)
    recovered = refined.step_verdicts[1]
    assert recovered.passed is True
    assert recovered.matched_observed_index is None
    assert recovered.recovery_reason == "frame_evidence"
    assert _passed_observations(refined) == [1]


def test_standing_episode_proposal_keeps_the_frame_proposal_out():
    expected, _, audit = _trace()
    episode = _episode(2)
    assert steps_with_standing_proposal(audit, expected, [episode]) == {2}

    settled, standing = settle_proposals(audit, expected, [episode])
    assert standing == [episode]
    assert settled[0].status == AssistProposalStatus.accepted


def test_both_proposals_lose_when_each_claims_an_owned_observation():
    expected, _, audit = _trace()
    settled, standing = settle_proposals(
        audit, expected, [_episode(1), _frame(observation=1)]
    )
    assert standing == []
    assert [proposal.status for proposal in settled] == [
        AssistProposalStatus.rejected,
        AssistProposalStatus.rejected,
    ]
    assert {proposal.reason for proposal in settled} == {OWNED_BY_ANOTHER_STEP}


def test_two_standing_proposals_cannot_pass_on_the_same_observation():
    expected, observed, audit = _trace()
    first = _episode(2)
    second = AssistProposalRecord(
        kind=AssistCandidateKind.episode_verification,
        status=AssistProposalStatus.accepted,
        expected_step_index=2,
        matched_observed_index=2,
        supporting_observed_index=2,
        completion_observed_index=2,
        matching_method=AssistMatchMethod.deterministic,
    )
    _, standing = settle_proposals(audit, expected, [first, second])
    refined = apply_assist_proposals(audit, standing, expected, observed)
    claimed = _passed_observations(refined)
    assert claimed == [1, 2]
    assert len(claimed) == len(set(claimed))
