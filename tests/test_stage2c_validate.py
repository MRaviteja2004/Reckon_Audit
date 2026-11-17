"""Unit tests for Stage 2C deterministic validation rules."""

from reckon.aligner import align
from reckon.schema import (
    ActionType,
    AuditResult,
    AuditVerdict,
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
from reckon.stage2c.manifest import build_frame_manifest
from reckon.stage2c.propose import RawFrameEvidenceCandidate
from reckon.stage2c.select import Stage2CSearchTarget, select_stage2c_targets
from reckon.stage2c.validate import validate_raw_candidate, validate_stage2c_batch


def _manifest():
    return build_frame_manifest([1000, 2000, 3000, 4000, 5000])


def _target(
    index: int = 2,
    *,
    evidence_type: CandidateEvidenceType = CandidateEvidenceType.ASSERTION_OUTCOME,
    assertion: str | None = "Driver home screen",
) -> Stage2CSearchTarget:
    return Stage2CSearchTarget(
        expected_step_id=f"E{index}",
        expected_step_index=index,
        evidence_type=evidence_type,
        expected_action="assert_state",
        expected_text=assertion or "click login",
        expected_assertion=assertion,
        finding_types=(FindingType.assertion_not_observed,),
    )


def _raw(
    *,
    step_id: str = "E2",
    frame_ids: list[str] | None = None,
    evidence_type: CandidateEvidenceType = CandidateEvidenceType.ASSERTION_OUTCOME,
    confidence: float = 0.9,
    search_status: CandidateSearchStatus = CandidateSearchStatus.candidate_found,
    description: str = "Home visible",
    visible_text: list[str] | None = None,
) -> RawFrameEvidenceCandidate:
    return RawFrameEvidenceCandidate(
        expected_step_id=step_id,
        frame_ids=frame_ids if frame_ids is not None else ["F3"],
        evidence_type=evidence_type,
        observed_description=description,
        visible_text=visible_text or ["You're offline"],
        confidence=confidence,
        search_status=search_status,
    )


def _audit_with_anchors() -> AuditResult:
    """Step 1 matched at 1000ms; step 3 matched at 5000ms; step 2 unmatched."""
    return AuditResult(
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=2,
                description="missing assertion",
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
                matched_observed_index=None,
                finding_types=[FindingType.assertion_not_observed],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=True,
                matched_observed_index=3,
                evidence_timestamp_ms=5000,
            ),
        ],
        fidelity_score=0.66,
    )


def test_rule1_unknown_frame_id_rejected():
    item = validate_raw_candidate(
        _raw(frame_ids=["F99"]),
        _target(),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "unknown_frame_id"
    assert item.timestamp_start_ms is None


def test_rule1_materializes_timestamp_range_from_manifest():
    item = validate_raw_candidate(
        _raw(frame_ids=["F2", "F4"]),
        _target(),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.validation_status == CandidateValidationStatus.accepted
    assert item.timestamp_start_ms == 2000
    assert item.timestamp_end_ms == 4000
    assert item.frame_ids == ["F2", "F4"]


def test_rule1_empty_frame_ids_rejected():
    item = validate_raw_candidate(
        _raw(frame_ids=[]),
        _target(),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "unknown_frame_id"


def test_rule2_before_previous_anchor_rejected():
    # Anchor at step 1 is 1000ms; candidate at F1 (1000) with end < lower is tricky.
    # Use a candidate entirely before a later lower bound by moving lower up via audit.
    audit = _audit_with_anchors()
    # Raise previous anchor to 3500 so F2 (2000) is before it.
    audit = audit.model_copy(
        update={
            "step_verdicts": [
                StepVerdict(
                    expected_step_index=1,
                    passed=True,
                    matched_observed_index=1,
                    evidence_timestamp_ms=3500,
                ),
                audit.step_verdicts[1],
                audit.step_verdicts[2],
            ]
        }
    )
    item = validate_raw_candidate(
        _raw(frame_ids=["F2"]),
        _target(),
        _manifest(),
        audit,
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "before_previous_anchor"


def test_rule2_unreliable_dp_anchor_does_not_block_earlier_evidence():
    """A passed-but-wrong_target prior step must not create before_previous_anchor."""
    audit = AuditResult(
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=3500,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                matched_observed_index=None,
                evidence_timestamp_ms=None,
                finding_types=[FindingType.missing_action],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=True,
                matched_observed_index=3,
                evidence_timestamp_ms=5000,
            ),
        ],
        fidelity_score=0.5,
        verdict=AuditVerdict.unfaithful,
    )
    item = validate_raw_candidate(
        _raw(frame_ids=["F2"]),
        _target(),
        _manifest(),
        audit,
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_rule2_after_next_anchor_rejected():
    # start_ms == upper is allowed; force strictly after via an earlier next anchor.
    audit = _audit_with_anchors().model_copy(
        update={
            "step_verdicts": [
                _audit_with_anchors().step_verdicts[0],
                _audit_with_anchors().step_verdicts[1],
                StepVerdict(
                    expected_step_index=3,
                    passed=True,
                    matched_observed_index=3,
                    evidence_timestamp_ms=2500,
                ),
            ]
        }
    )
    item = validate_raw_candidate(
        _raw(frame_ids=["F4"]),
        _target(),
        _manifest(),
        audit,
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "after_next_anchor"


def test_rule2_within_anchor_window_accepted():
    item = validate_raw_candidate(
        _raw(frame_ids=["F3"]),
        _target(),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.validation_status == CandidateValidationStatus.accepted
    assert item.rejection_reason is None


def test_rule3_outcome_before_action_rejected():
    item = validate_raw_candidate(
        _raw(
            frame_ids=["F2"],
            evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
        ),
        _target(),
        _manifest(),
        _audit_with_anchors(),
        peer_action_end_ms=3000,
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "outcome_before_action"


def test_rule3_outcome_after_action_accepted():
    item = validate_raw_candidate(
        _raw(
            frame_ids=["F4"],
            evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
        ),
        _target(),
        _manifest(),
        _audit_with_anchors(),
        peer_action_end_ms=2000,
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_rule4_action_frame_reuse_across_steps_rejected():
    claimed = {"F3": (1, CandidateEvidenceType.ACTION)}
    item = validate_raw_candidate(
        _raw(
            frame_ids=["F3"],
            evidence_type=CandidateEvidenceType.ACTION,
            description="Second click",
            visible_text=["Submit"],
        ),
        _target(
            2,
            evidence_type=CandidateEvidenceType.ACTION,
            assertion=None,
        ),
        _manifest(),
        _audit_with_anchors(),
        claimed_frames=claimed,
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "frame_reuse_conflict"


def test_rule4_assertion_may_share_persistent_assertion_frame():
    claimed = {"F3": (1, CandidateEvidenceType.ASSERTION_OUTCOME)}
    item = validate_raw_candidate(
        _raw(
            frame_ids=["F3"],
            description="Description text on the same landing page",
            visible_text=["Earn when you mentor"],
        ),
        _target(2, assertion="Description is visible"),
        _manifest(),
        _audit_with_anchors(),
        claimed_frames=claimed,
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_rule4_assertion_may_share_terminal_action_frame():
    claimed = {"F3": (1, CandidateEvidenceType.ACTION)}
    item = validate_raw_candidate(
        _raw(
            frame_ids=["F3"],
            description="Confirmation appears after join",
            visible_text=["You've joined"],
        ),
        _target(2, assertion="confirmation is shown"),
        _manifest(),
        _audit_with_anchors(),
        claimed_frames=claimed,
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_rule4_same_step_frame_reuse_allowed():
    claimed = {"F3": (2, CandidateEvidenceType.ASSERTION_OUTCOME)}
    item = validate_raw_candidate(
        _raw(frame_ids=["F3"]),
        _target(2),
        _manifest(),
        _audit_with_anchors(),
        claimed_frames=claimed,
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_rule5_contradictory_description_rejected_for_settings_claim():
    item = validate_raw_candidate(
        _raw(
            description=(
                "The page after the settings tap is titled Opt out of mentorship "
                "programme. No page explicitly labelled Settings is seen."
            ),
            visible_text=["Opt out of mentorship programme"],
        ),
        _target(assertion="The Program / Mentorship Settings page is displayed"),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "contradictory_description"


def test_rule5_blocking_error_collision_rejected():
    item = validate_raw_candidate(
        _raw(
            description="Login failed banner",
            visible_text=["Authentication failed"],
        ),
        _target(assertion="User logged in successfully"),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.validation_status == CandidateValidationStatus.rejected
    assert item.rejection_reason == "blocking_error_collision"


def test_rule5_error_expected_allows_blocking_text():
    item = validate_raw_candidate(
        _raw(
            description="Error banner",
            visible_text=["Invalid password"],
        ),
        _target(assertion="Invalid password error is shown"),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.validation_status == CandidateValidationStatus.accepted


def test_low_confidence_rejected():
    item = validate_raw_candidate(
        _raw(confidence=0.4),
        _target(),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.rejection_reason == "low_confidence"


def test_search_not_found_rejected_with_reason():
    item = validate_raw_candidate(
        _raw(search_status=CandidateSearchStatus.not_found, frame_ids=[]),
        _target(),
        _manifest(),
        _audit_with_anchors(),
    )
    assert item.rejection_reason == "search_not_found"


def test_batch_enforces_action_before_outcome_across_candidates():
    targets = [
        Stage2CSearchTarget(
            expected_step_id="E1",
            expected_step_index=1,
            evidence_type=CandidateEvidenceType.ACTION,
            expected_action="click",
            expected_text="Click Login",
            expected_assertion=None,
            finding_types=(FindingType.missing_action,),
        ),
        _target(2),
    ]
    audit = AuditResult(
        findings=[
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                description="missing click",
            ),
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=2,
                description="missing home",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                finding_types=[FindingType.missing_action],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
        fidelity_score=0.0,
    )
    # Outcome on F2 before action on F4 → outcome rejected.
    raw = [
        RawFrameEvidenceCandidate(
            expected_step_id="E1",
            frame_ids=["F4"],
            evidence_type=CandidateEvidenceType.ACTION,
            observed_description="Login click",
            visible_text=["Login"],
            confidence=0.9,
            search_status=CandidateSearchStatus.candidate_found,
        ),
        RawFrameEvidenceCandidate(
            expected_step_id="E2",
            frame_ids=["F2"],
            evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
            observed_description="Home",
            visible_text=["offline"],
            confidence=0.9,
            search_status=CandidateSearchStatus.candidate_found,
        ),
    ]
    trace = validate_stage2c_batch(
        raw, targets, _manifest(), audit, source_file="x.mp4"
    )
    by = {c.expected_step_id: c for c in trace.candidates}
    assert by["E1"].validation_status == CandidateValidationStatus.accepted
    assert by["E2"].validation_status == CandidateValidationStatus.rejected
    assert by["E2"].rejection_reason == "outcome_before_action"


def test_select_stage2c_targets_only_recoverable_findings():
    expected = ExpectedTrace(
        scenario_title="t",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Login"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_state,
                assertion="home",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Cancel",
                confidence=0.9,
            )
        ],
    )
    audit = align(expected, observed)
    targets = select_stage2c_targets(expected, audit)
    assert all(
        ft in {FindingType.missing_action, FindingType.assertion_not_observed}
        for t in targets
        for ft in t.finding_types
    )
    assert any(t.expected_step_index == 2 for t in targets)
