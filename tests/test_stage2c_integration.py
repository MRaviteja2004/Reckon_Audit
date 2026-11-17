"""Integration tests for Stage 2C recovery via align_with_assist."""

from unittest.mock import MagicMock

from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.assist_apply import apply_assist_proposals
from reckon.assist_propose import _AssistBatchResponse
from reckon.crosscheck import select_crosscheck_targets
from reckon.schema import (
    ActionType,
    AssistCandidateKind,
    AssistProposalStatus,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedAction,
    ObservedTrace,
)
from reckon.stage2c.manifest import build_frame_manifest
from reckon.stage2c.propose import RawFrameEvidenceBatch, RawFrameEvidenceCandidate
from reckon.stage2c.select import Stage2CSearchTarget
from reckon.stage2c.to_assist import (
    high_confidence_visual_skip_steps,
    proposals_from_candidate_trace,
)
from reckon.stage2c.validate import validate_stage2c_batch


def _quiet_client() -> MagicMock:
    """Mock client so classic/episode assist never hits the network."""
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(decisions=[])
    return client


def _missing_login_traces():
    """ObservedTrace lacks the login click and home assertion Stage 2 missed."""
    expected = ExpectedTrace(
        scenario_title="Login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Login",
                raw_text="Click the Login button",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_state,
                target="home",
                assertion="You're offline",
                raw_text="Driver home screen is displayed",
            ),
        ],
    )
    # Only an unrelated early splash — DP will miss both expected steps.
    observed = ObservedTrace(
        source_file="login.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=500,
                action=ActionType.other,
                target="Splash",
                confidence=0.8,
                screen_description="Splash screen",
            ),
        ],
    )
    return expected, observed


def test_successful_recovery_clears_missing_and_assertion():
    expected, observed = _missing_login_traces()
    deterministic = align(expected, observed)
    assert {f.type for f in deterministic.findings} >= {FindingType.missing_action}
    # Unmatched assertion steps are reported as missing_action by align().
    assert all(
        FindingType.missing_action in v.finding_types
        for v in deterministic.step_verdicts
    )

    manifest = build_frame_manifest([500, 2000, 3500, 5000])
    raw = RawFrameEvidenceBatch(
        candidates=[
            RawFrameEvidenceCandidate(
                expected_step_id="E1",
                frame_ids=["F2"],
                evidence_type=CandidateEvidenceType.ACTION,
                observed_description="Login button pressed",
                visible_text=["Login"],
                confidence=0.92,
                search_status=CandidateSearchStatus.candidate_found,
            ),
            RawFrameEvidenceCandidate(
                expected_step_id="E2",
                frame_ids=["F3"],
                evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
                observed_description="Driver home",
                visible_text=["You're offline"],
                confidence=0.91,
                search_status=CandidateSearchStatus.candidate_found,
            ),
        ]
    )
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=_quiet_client(),
        frame_evidence=True,
        frame_manifest=manifest,
        stage2c_raw_batch=raw,
    )
    by = {v.expected_step_index: v for v in assisted.result.step_verdicts}
    # The word Login on a frame does not prove the click. The assertion text does.
    assert by[1].passed is False
    assert by[2].passed is True
    assert by[2].evidence_id == "F3"
    assert by[2].evidence_timestamp_ms == 3500
    assert by[2].frame_ids == ["F3"]
    assert by[2].recovery_reason == "frame_evidence"
    assert by[2].matched_observed_index is None
    assert by[2].explanation == "Frames F3 at 3500 ms show the asserted text."
    assert any(
        finding.type == FindingType.missing_action and finding.expected_step_index == 1
        for finding in assisted.result.findings
    )
    assert FindingType.assertion_not_observed not in {
        f.type for f in assisted.result.findings
    }
    assert assisted.assist.candidate_evidence is not None
    assert all(
        c.validation_status == CandidateValidationStatus.accepted
        for c in assisted.assist.candidate_evidence.candidates
    )
    assert 2 in assisted.assist.stage2c_skip_visual_steps
    # Baseline preserved
    assert assisted.deterministic_result.findings


def test_rejected_candidates_remain_auditable_and_do_not_clear():
    expected, observed = _missing_login_traces()
    manifest = build_frame_manifest([500, 2000, 3500])
    raw = RawFrameEvidenceBatch(
        candidates=[
            RawFrameEvidenceCandidate(
                expected_step_id="E1",
                frame_ids=["F99"],
                evidence_type=CandidateEvidenceType.ACTION,
                observed_description="bogus",
                visible_text=[],
                confidence=0.95,
                search_status=CandidateSearchStatus.candidate_found,
            ),
            RawFrameEvidenceCandidate(
                expected_step_id="E2",
                frame_ids=["F2"],
                evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
                observed_description="Authentication failed",
                visible_text=["Login failed"],
                confidence=0.95,
                search_status=CandidateSearchStatus.candidate_found,
            ),
        ]
    )
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=_quiet_client(),
        frame_evidence=True,
        frame_manifest=manifest,
        stage2c_raw_batch=raw,
    )
    assert assisted.result.findings
    assert any(f.type == FindingType.missing_action for f in assisted.result.findings)
    # Rejected Stage 2C must not clear either unmatched step.
    assert all(not v.passed for v in assisted.result.step_verdicts)
    proposals = [
        p
        for p in assisted.assist.proposals
        if p.kind == AssistCandidateKind.frame_evidence
    ]
    assert proposals
    assert all(p.status == AssistProposalStatus.rejected for p in proposals)
    reasons = {p.reason for p in proposals}
    assert "unknown_frame_id" in reasons
    assert "blocking_error_collision" in reasons
    assert assisted.assist.candidate_evidence is not None
    assert all(
        c.rejection_reason for c in assisted.assist.candidate_evidence.candidates
    )


def test_frame_evidence_preserves_wrong_input_value():
    expected = ExpectedTrace(
        scenario_title="Type email",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.type_text,
                target="email",
                input_value="alpha@example.invalid",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_text,
                assertion="Welcome",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.type_text,
                target="email",
                input_value="beta@example.invalid",
                confidence=0.9,
            ),
        ],
    )
    deterministic = align(expected, observed)
    # If tolerance treats emails as equivalent, force the finding shape we need.
    if not any(f.type == FindingType.wrong_input_value for f in deterministic.findings):
        from reckon.schema import (
            AuditResult,
            Finding,
            FindingSeverity,
            StepVerdict,
        )

        deterministic = AuditResult(
            findings=[
                Finding(
                    type=FindingType.wrong_input_value,
                    severity=FindingSeverity.critical,
                    expected_step_index=1,
                    observed_action_index=1,
                    evidence_timestamp_ms=1000,
                    description="wrong email",
                ),
                Finding(
                    type=FindingType.missing_action,
                    severity=FindingSeverity.critical,
                    expected_step_index=2,
                    description="missing welcome",
                ),
            ],
            step_verdicts=[
                StepVerdict(
                    expected_step_index=1,
                    passed=False,
                    matched_observed_index=1,
                    evidence_timestamp_ms=1000,
                    finding_types=[FindingType.wrong_input_value],
                ),
                StepVerdict(
                    expected_step_index=2,
                    passed=False,
                    finding_types=[FindingType.missing_action],
                ),
            ],
            fidelity_score=0.0,
        )

    manifest = build_frame_manifest([1000, 2000, 3000])
    # Apply Stage 2C proposals directly so we isolate apply() preservation.
    from reckon.stage2c.select import select_stage2c_targets
    from reckon.stage2c.validate import validate_stage2c_batch

    targets = select_stage2c_targets(expected, deterministic)
    raw_list = [
        RawFrameEvidenceCandidate(
            expected_step_id="E2",
            frame_ids=["F3"],
            evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
            observed_description="Welcome banner",
            visible_text=["Welcome"],
            confidence=0.95,
            search_status=CandidateSearchStatus.candidate_found,
        ),
    ]
    trace = validate_stage2c_batch(
        raw_list, targets, manifest, deterministic, source_file="x.mp4"
    )
    accepted = [
        p
        for p in proposals_from_candidate_trace(trace)
        if p.status == AssistProposalStatus.accepted
    ]
    refined = apply_assist_proposals(deterministic, accepted, expected, observed)
    assert any(f.type == FindingType.wrong_input_value for f in refined.findings)
    # E2 may recover; E1 wrong_input_value must remain.
    assert any(
        f.type == FindingType.wrong_input_value and f.expected_step_index == 1
        for f in refined.findings
    )


def test_raw_llm_candidates_cannot_clear_without_validation():
    """Proposals built only after validate; unknown frames never accepted."""
    targets = [
        Stage2CSearchTarget(
            expected_step_id="E1",
            expected_step_index=1,
            evidence_type=CandidateEvidenceType.ACTION,
            expected_action="click",
            expected_text="Login",
            expected_assertion=None,
            finding_types=(FindingType.missing_action,),
        )
    ]
    expected, observed = _missing_login_traces()
    deterministic = align(expected, observed)
    manifest = build_frame_manifest([1000, 2000])
    raw = [
        RawFrameEvidenceCandidate(
            expected_step_id="E1",
            frame_ids=["F9"],
            evidence_type=CandidateEvidenceType.ACTION,
            observed_description="claimed click",
            visible_text=["Login"],
            confidence=0.99,
            search_status=CandidateSearchStatus.candidate_found,
        )
    ]
    trace = validate_stage2c_batch(
        raw, targets, manifest, deterministic, source_file="x.mp4"
    )
    proposals = proposals_from_candidate_trace(trace)
    accepted = [p for p in proposals if p.status == AssistProposalStatus.accepted]
    refined = apply_assist_proposals(deterministic, accepted, expected, observed)
    assert any(f.type == FindingType.missing_action for f in refined.findings)
    assert trace.candidates[0].rejection_reason == "unknown_frame_id"


def test_stage4_skips_visual_recheck_for_stage2c_high_confidence():
    expected, observed = _missing_login_traces()
    manifest = build_frame_manifest([500, 2000, 3500])
    raw = RawFrameEvidenceBatch(
        candidates=[
            RawFrameEvidenceCandidate(
                expected_step_id="E1",
                frame_ids=["F2"],
                evidence_type=CandidateEvidenceType.ACTION,
                observed_description="Login",
                visible_text=["Login"],
                confidence=0.9,
                search_status=CandidateSearchStatus.candidate_found,
            ),
            RawFrameEvidenceCandidate(
                expected_step_id="E2",
                frame_ids=["F3"],
                evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
                observed_description="Home",
                visible_text=["You're offline"],
                confidence=0.9,
                search_status=CandidateSearchStatus.candidate_found,
            ),
        ]
    )
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=_quiet_client(),
        frame_evidence=True,
        frame_manifest=manifest,
        stage2c_raw_batch=raw,
    )
    skip = set(assisted.assist.stage2c_skip_visual_steps)
    assert skip == high_confidence_visual_skip_steps(
        assisted.assist.candidate_evidence
    )
    targets = select_crosscheck_targets(
        expected,
        observed,
        assisted.result,
        skip_visual_steps=skip,
    )
    assert not any(
        t.expected_step_index == 2 and t.kind.value == "assertion" for t in targets
    )


def test_frame_evidence_off_is_backward_compatible():
    expected, observed = _missing_login_traces()
    assisted = align_with_assist(expected, observed)
    assert assisted.assist.candidate_evidence is None
    assert assisted.assist.stage2c_skip_visual_steps == []
    assert not any(
        p.kind == AssistCandidateKind.frame_evidence for p in assisted.assist.proposals
    )
