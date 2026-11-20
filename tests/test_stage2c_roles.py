"""Regression tests for Stage 2C semantic role exclusivity."""

from reckon.schema import (
    ActionType,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    StepVerdict,
    AuditResult,
    AuditVerdict,
)
from reckon.stage2c.manifest import build_frame_manifest
from reckon.stage2c.propose import RawFrameEvidenceCandidate
from reckon.stage2c.roles import (
    SemanticRole,
    evidence_semantic_role,
    expectation_semantic_role,
    role_mismatch_reason,
)
from reckon.stage2c.select import Stage2CSearchTarget
from reckon.stage2c.validate import validate_raw_candidate


def _manifest():
    return build_frame_manifest([1000, 5000, 10000])


def _audit_missing(step_index: int = 1) -> AuditResult:
    return AuditResult(
        verdict=AuditVerdict.unfaithful,
        findings=[
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.advisory,
                expected_step_index=step_index,
                description="missing",
            )
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=step_index,
                passed=False,
                finding_types=[FindingType.missing_action],
            )
        ],
        fidelity_score=0.0,
    )


def _action_target(step: ExpectedStep) -> Stage2CSearchTarget:
    return Stage2CSearchTarget(
        expected_step_id=f"E{step.index}",
        expected_step_index=step.index,
        evidence_type=CandidateEvidenceType.ACTION,
        expected_action=step.action.value,
        expected_text=f"{step.target or ''} {step.raw_text or ''}".strip(),
        expected_assertion=None,
        finding_types=(FindingType.missing_action,),
    )


def test_login_evidence_does_not_prove_app_launch_requirement():
    expected = ExpectedTrace(
        scenario_title="Launch only",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="driver app",
                raw_text="Open the driver app",
            )
        ],
    )
    step = expected.steps[0]
    assert expectation_semantic_role(step, expected) == SemanticRole.APP_LAUNCH
    raw = RawFrameEvidenceCandidate(
        expected_step_id="E1",
        frame_ids=["F3"],
        evidence_type=CandidateEvidenceType.ACTION,
        observed_description=(
            "The user taps Login with password and enters a masked password."
        ),
        visible_text=["Welcome back", "Login with password", "Next"],
        confidence=0.9,
        search_status=CandidateSearchStatus.candidate_found,
    )
    item = validate_raw_candidate(
        raw,
        _action_target(step),
        _manifest(),
        _audit_missing(),
        expected=expected,
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "login_evidence_for_app_launch"


def test_login_evidence_proves_explicit_login_requirement():
    expected = ExpectedTrace(
        scenario_title="Login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="Driver app login",
                raw_text="Login to Driver app with valid credential.",
            )
        ],
    )
    step = expected.steps[0]
    assert expectation_semantic_role(step, expected) == SemanticRole.AUTHENTICATION
    raw = RawFrameEvidenceCandidate(
        expected_step_id="E1",
        frame_ids=["F3"],
        evidence_type=CandidateEvidenceType.ACTION,
        observed_description=(
            "The user taps Login with password, enters a masked password and taps Next."
        ),
        visible_text=["Welcome back", "Login with password", "Next"],
        confidence=0.9,
        search_status=CandidateSearchStatus.candidate_found,
    )
    item = validate_raw_candidate(
        raw,
        _action_target(step),
        _manifest(),
        _audit_missing(),
        expected=expected,
    )
    assert item.validation_status == CandidateValidationStatus.accepted
    assert item.rejection_reason is None


def test_app_launch_evidence_proves_app_launch_requirement():
    expected = ExpectedTrace(
        scenario_title="Launch",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="driver app",
                raw_text="Open the driver app",
            )
        ],
    )
    step = expected.steps[0]
    raw = RawFrameEvidenceCandidate(
        expected_step_id="E1",
        frame_ids=["F1"],
        evidence_type=CandidateEvidenceType.ACTION,
        observed_description="User taps the Uber Driver app icon; splash launching.",
        visible_text=["Uber Driver"],
        confidence=0.9,
        search_status=CandidateSearchStatus.candidate_found,
    )
    item = validate_raw_candidate(
        raw,
        _action_target(step),
        _manifest(),
        _audit_missing(),
        expected=expected,
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_pure_driver_app_target_with_login_raw_text_is_app_launch():
    """R2-style coarse step: target is app, raw mentions login → still app_launch."""
    step = ExpectedStep(
        index=1,
        action=ActionType.other,
        target="Driver app",
        raw_text="Login to Driver app with valid credentail.",
    )
    expected = ExpectedTrace(scenario_title="R2", steps=[step])
    assert expectation_semantic_role(step, expected) == SemanticRole.APP_LAUNCH
    assert (
        role_mismatch_reason(
            SemanticRole.APP_LAUNCH,
            evidence_semantic_role(
                "Login with password on Welcome back screen",
                ["Login with password"],
            ),
        )
        == "login_evidence_for_app_launch"
    )


def test_unknown_role_evidence_does_not_reject_unrelated_action():
    expected = ExpectedTrace(
        scenario_title="Search",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Search"),
        ],
    )
    step = expected.steps[0]
    assert expectation_semantic_role(step, expected) == SemanticRole.OTHER
    raw = RawFrameEvidenceCandidate(
        expected_step_id="E1",
        frame_ids=["F2"],
        evidence_type=CandidateEvidenceType.ACTION,
        observed_description="Tap indicator on Search button",
        visible_text=["Search"],
        confidence=0.85,
        search_status=CandidateSearchStatus.candidate_found,
    )
    item = validate_raw_candidate(
        raw,
        _action_target(step),
        _manifest(),
        _audit_missing(),
        expected=expected,
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_role_mismatch_does_not_apply_to_assertion_outcome():
    expected = ExpectedTrace(
        scenario_title="Home",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_state,
                target="driver app",
                assertion="User logs in successfully.",
            )
        ],
    )
    raw = RawFrameEvidenceCandidate(
        expected_step_id="E1",
        frame_ids=["F3"],
        evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
        observed_description="Login with password succeeded; home visible",
        visible_text=["You're offline"],
        confidence=0.9,
        search_status=CandidateSearchStatus.candidate_found,
    )
    target = Stage2CSearchTarget(
        expected_step_id="E1",
        expected_step_index=1,
        evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
        expected_action="assert_state",
        expected_text="driver app User logs in successfully.",
        expected_assertion="User logs in successfully.",
        finding_types=(FindingType.assertion_not_observed,),
    )
    item = validate_raw_candidate(
        raw,
        target,
        _manifest(),
        AuditResult(
            verdict=AuditVerdict.unfaithful,
            findings=[
                Finding(
                    type=FindingType.assertion_not_observed,
                    severity=FindingSeverity.critical,
                    expected_step_index=1,
                    description="missing",
                )
            ],
            step_verdicts=[
                StepVerdict(
                    expected_step_index=1,
                    passed=False,
                    finding_types=[FindingType.assertion_not_observed],
                )
            ],
            fidelity_score=0.0,
        ),
        expected=expected,
    )
    assert item.validation_status == CandidateValidationStatus.accepted
