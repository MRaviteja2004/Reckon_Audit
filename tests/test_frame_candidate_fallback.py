"""Stage 2C candidates are reused when ownership blocks a step, and only then."""

from reckon.assist_apply import apply_assist_proposals
from reckon.frame_candidate_fallback import (
    FRAME_EVIDENCE_RELEASED_OBSERVATION,
    fallback_to_frame_candidates,
)
from reckon.ownership import OWNED_BY_ANOTHER_STEP
from reckon.proposal_selection import settle_proposals
from reckon.schema import (
    ActionType,
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    AuditVerdict,
    CandidateEvidenceItem,
    CandidateEvidenceTrace,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)


def _click(index: int, target: str, *, raw_text: str | None = None) -> ExpectedStep:
    return ExpectedStep(index=index, action=ActionType.click, target=target, raw_text=raw_text)


def _assertion(index: int, target: str, assertion: str) -> ExpectedStep:
    return ExpectedStep(
        index=index,
        action=ActionType.assert_visible,
        target=target,
        assertion=assertion,
    )


def _observation(index: int, timestamp_ms: int, target: str) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=ActionType.click,
        target=target,
        confidence=0.9,
    )


def _missing(step_index: int) -> Finding:
    return Finding(
        type=FindingType.missing_action,
        severity=FindingSeverity.critical,
        expected_step_index=step_index,
        description=f"Expected step {step_index} was not observed.",
    )


def _audit(verdicts: list[StepVerdict], findings: list[Finding]) -> AuditResult:
    return AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=findings,
        step_verdicts=verdicts,
    )


def _episode(step_index: int, observation: int) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=AssistCandidateKind.episode_verification,
        status=AssistProposalStatus.accepted,
        expected_step_index=step_index,
        matched_observed_index=observation,
        supporting_observed_index=observation,
        completion_observed_index=observation,
        matching_method=AssistMatchMethod.deterministic,
        reason="episode_verified",
    )


def _wrong_target(step_index: int, observation: int) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=AssistCandidateKind.wrong_target,
        status=AssistProposalStatus.accepted,
        expected_step_index=step_index,
        matched_observed_index=observation,
        finding_types=[FindingType.wrong_target],
        matching_method=AssistMatchMethod.llm_assisted,
    )


def _candidate(
    step_index: int,
    *,
    visible_text: list[str],
    timestamp_ms: int,
    evidence_type: CandidateEvidenceType = CandidateEvidenceType.ASSERTION_OUTCOME,
    description: str = "",
) -> CandidateEvidenceItem:
    return CandidateEvidenceItem(
        expected_step_id=f"E{step_index}",
        expected_step_index=step_index,
        frame_ids=[f"F{step_index}"],
        timestamp_start_ms=timestamp_ms,
        timestamp_end_ms=timestamp_ms,
        evidence_type=evidence_type,
        observed_description=description,
        visible_text=visible_text,
        confidence=0.9,
        search_status=CandidateSearchStatus.candidate_found,
        validation_status=CandidateValidationStatus.accepted,
    )


def _trace(*items: CandidateEvidenceItem) -> CandidateEvidenceTrace:
    return CandidateEvidenceTrace(source_file="fallback.mp4", candidates=list(items))


def _settle(
    audit: AuditResult,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    proposals: list[AssistProposalRecord],
    trace: CandidateEvidenceTrace,
) -> tuple[list[AssistProposalRecord], list[AssistProposalRecord]]:
    settled, standing = settle_proposals(audit, expected, proposals)
    return fallback_to_frame_candidates(
        audit, expected, observed, proposals, settled, standing, trace
    )


def _passed_observations(audit: AuditResult) -> list[int | None]:
    return [
        verdict.matched_observed_index
        for verdict in audit.step_verdicts
        if verdict.passed
    ]


def test_competing_action_and_assertion_release_the_observation_to_frame_evidence():
    expected = ExpectedTrace(
        scenario_title="Save",
        steps=[_click(1, "Save"), _assertion(2, "receipt", "Saved")],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Save")],
    )
    audit = _audit(
        [
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                finding_types=[FindingType.missing_action],
            ),
        ],
        [
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=1000,
                description="Save was paired with the wrong control.",
            ),
            _missing(2),
        ],
    )
    # The assertion proposal is first, so it takes the observation the action needs.
    proposals = [_episode(2, 1), _wrong_target(1, 1)]
    settled, standing = _settle(
        audit, expected, observed, proposals, _trace(_candidate(2, visible_text=["Saved"], timestamp_ms=1500))
    )

    by_step = {proposal.expected_step_index: proposal for proposal in standing}
    assert by_step[1].kind == AssistCandidateKind.wrong_target
    assert by_step[2].kind == AssistCandidateKind.frame_evidence
    assert by_step[2].matched_observed_index is None
    released = next(
        proposal
        for proposal in settled
        if proposal.kind == AssistCandidateKind.episode_verification
    )
    assert released.status == AssistProposalStatus.rejected
    assert released.reason == FRAME_EVIDENCE_RELEASED_OBSERVATION

    refined = apply_assist_proposals(audit, standing, expected, observed)
    assert [verdict.passed for verdict in refined.step_verdicts] == [True, True]
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[1].matched_observed_index is None
    assert refined.step_verdicts[1].recovery_reason == "frame_evidence"
    claimed = [index for index in _passed_observations(refined) if index is not None]
    assert claimed == [1]
    assert len(claimed) == len(set(claimed))


def test_owned_assertion_falls_back_to_its_validated_frame_candidate():
    expected = ExpectedTrace(
        scenario_title="Save",
        steps=[_click(1, "Save"), _assertion(2, "receipt", "Saved")],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Save")],
    )
    audit = _audit(
        [
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
        [_missing(2)],
    )
    settled, standing = _settle(
        audit,
        expected,
        observed,
        [_episode(2, 1)],
        _trace(_candidate(2, visible_text=["Saved"], timestamp_ms=2000)),
    )

    assert [proposal.kind for proposal in standing] == [AssistCandidateKind.frame_evidence]
    assert standing[0].expected_step_index == 2
    assert standing[0].status == AssistProposalStatus.accepted
    lost = next(
        proposal
        for proposal in settled
        if proposal.kind == AssistCandidateKind.episode_verification
    )
    assert lost.reason == OWNED_BY_ANOTHER_STEP

    refined = apply_assist_proposals(audit, standing, expected, observed)
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[1].passed is True
    assert refined.step_verdicts[1].matched_observed_index is None
    assert refined.step_verdicts[1].evidence_timestamp_ms == 2000
    claimed = [index for index in _passed_observations(refined) if index is not None]
    assert claimed == [1]


def test_frame_fallback_rejects_candidate_when_visible_text_misses_assertion():
    expected = ExpectedTrace(
        scenario_title="Save",
        steps=[_click(1, "Save"), _assertion(2, "receipt", "Saved")],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Save")],
    )
    audit = _audit(
        [
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
        [_missing(2)],
    )
    settled, standing = _settle(
        audit,
        expected,
        observed,
        [_episode(2, 1)],
        _trace(_candidate(2, visible_text=["Unrelated"], timestamp_ms=2000)),
    )

    assert standing == []
    frame = next(
        proposal for proposal in settled if proposal.kind == AssistCandidateKind.frame_evidence
    )
    assert frame.status == AssistProposalStatus.rejected
    assert frame.reason == "visible_text_does_not_support_assertion"
    refined = apply_assist_proposals(audit, standing, expected, observed)
    assert refined.step_verdicts[1].passed is False


def test_frame_fallback_rejects_candidate_before_previous_anchor():
    expected = ExpectedTrace(
        scenario_title="Save",
        steps=[_click(1, "Save"), _assertion(2, "receipt", "Saved")],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Save")],
    )
    audit = _audit(
        [
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
        [_missing(2)],
    )
    _settled, standing = _settle(
        audit,
        expected,
        observed,
        [_episode(2, 1)],
        _trace(_candidate(2, visible_text=["Saved"], timestamp_ms=500)),
    )

    assert standing == []
    frame = next(
        proposal for proposal in _settled if proposal.kind == AssistCandidateKind.frame_evidence
    )
    assert frame.reason == "before_previous_anchor"


def test_frame_fallback_rejects_action_candidate_with_conflicting_target_role():
    expected = ExpectedTrace(
        scenario_title="Launch",
        steps=[
            _click(1, "Continue"),
            _click(2, "Driver app", raw_text="Open the Driver app"),
        ],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Continue")],
    )
    audit = _audit(
        [
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
        [_missing(2)],
    )
    settled, standing = _settle(
        audit,
        expected,
        observed,
        [_episode(2, 1)],
        _trace(
            _candidate(
                2,
                visible_text=["Enter your password"],
                timestamp_ms=2000,
                evidence_type=CandidateEvidenceType.ACTION,
                description="Enter your password",
            )
        ),
    )

    assert standing == []
    frame = next(
        proposal for proposal in settled if proposal.kind == AssistCandidateKind.frame_evidence
    )
    assert frame.status == AssistProposalStatus.rejected
    assert frame.reason == "login_evidence_for_app_launch"


def test_missing_candidate_trace_leaves_ownership_unchanged():
    expected = ExpectedTrace(
        scenario_title="Save",
        steps=[_click(1, "Save"), _assertion(2, "receipt", "Saved")],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Save")],
    )
    audit = _audit(
        [
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
        [_missing(2)],
    )
    proposals = [_episode(2, 1)]
    settled, standing = settle_proposals(audit, expected, proposals)
    again, still = fallback_to_frame_candidates(
        audit, expected, observed, proposals, settled, standing, None
    )
    assert again == settled
    assert still == standing
    assert standing == []


def test_rejected_stage2c_candidate_is_not_reconsidered():
    expected = ExpectedTrace(
        scenario_title="Save",
        steps=[_click(1, "Save"), _assertion(2, "receipt", "Saved")],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Save")],
    )
    audit = _audit(
        [
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
        [_missing(2)],
    )
    rejected = _candidate(2, visible_text=["Saved"], timestamp_ms=2000).model_copy(
        update={
            "validation_status": CandidateValidationStatus.rejected,
            "rejection_reason": "low_confidence",
        }
    )
    settled, standing = _settle(
        audit, expected, observed, [_episode(2, 1)], _trace(rejected)
    )
    assert standing == []
    assert not any(
        proposal.kind == AssistCandidateKind.frame_evidence for proposal in settled
    )


def test_fallback_does_not_let_a_second_step_take_an_owned_observation():
    expected = ExpectedTrace(
        scenario_title="Save",
        steps=[
            _click(1, "Save"),
            _assertion(2, "receipt", "Saved"),
            _click(3, "Done"),
        ],
    )
    observed = ObservedTrace(
        source_file="fallback.mp4",
        actions=[_observation(1, 1000, "Save")],
    )
    audit = _audit(
        [
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
            StepVerdict(
                expected_step_index=3,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
                finding_types=[FindingType.wrong_target],
            ),
        ],
        [_missing(2)],
    )
    settled, standing = _settle(
        audit,
        expected,
        observed,
        [_episode(2, 1), _wrong_target(3, 1)],
        _trace(_candidate(2, visible_text=["Saved"], timestamp_ms=2000)),
    )

    refined = apply_assist_proposals(audit, standing, expected, observed)
    assert refined.step_verdicts[0].passed is True
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[1].passed is True
    assert refined.step_verdicts[1].matched_observed_index is None
    assert refined.step_verdicts[2].passed is False
    owned = next(
        proposal
        for proposal in settled
        if proposal.kind == AssistCandidateKind.wrong_target
    )
    assert owned.status == AssistProposalStatus.rejected
    assert owned.reason == OWNED_BY_ANOTHER_STEP
    claimed = [index for index in _passed_observations(refined) if index is not None]
    assert claimed == [1]
    assert len(claimed) == len(set(claimed))
