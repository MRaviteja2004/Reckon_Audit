"""Regression tests for Stage 3 evidence recovery and alignment-drift fixes."""

from reckon.aligner import align
from reckon.assist_apply import apply_assist_proposals
from reckon.assist_evidence_select import retrieve_evidence_window
from reckon.crosscheck import _preferred_assertion_anchor, select_crosscheck_targets
from reckon.episode_verify_fast import proof_is_premature
from reckon.schema import (
    ActionType,
    AssertionEvidence,
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
from reckon.stage2c.select import select_stage2c_targets
from reckon.tolerance import (
    assertion_satisfied,
    assertion_target_relevant,
    empty_state_evidence_supports_assertion,
    _introduces_negation,
)
from reckon.workflow_reentry import (
    is_workflow_reentry_action,
    primary_workflow_reentry,
    reentry_floor_for_step,
    reentry_lower_timestamp_ms,
)


def _github_login_page_traces():
    expected = ExpectedTrace(
        scenario_title="GitHub sign in",
        steps=[
            ExpectedStep(index=1, action=ActionType.navigate, target="GitHub homepage"),
            ExpectedStep(index=2, action=ActionType.click, target="Sign in"),
            ExpectedStep(
                index=3,
                action=ActionType.assert_visible,
                target="GitHub login page",
                assertion="the GitHub login page is displayed",
            ),
            ExpectedStep(
                index=4,
                action=ActionType.type_text,
                target="username field",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="github.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=3000,
                action=ActionType.navigate,
                target="github.com",
                confidence=0.9,
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5000,
                action=ActionType.click,
                target="Sign in button",
                confidence=0.9,
            ),
            ObservedAction(
                index=3,
                timestamp_ms=6500,
                action=ActionType.navigate,
                target="github.com/login",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="GitHub login page displayed",
                    visible_text="Sign in to GitHub",
                ),
                screen_description="Sign in to GitHub page",
            ),
            ObservedAction(
                index=4,
                timestamp_ms=12000,
                action=ActionType.type_text,
                target="Username or email address field",
                input_value="user",
                confidence=0.9,
            ),
            ObservedAction(
                index=5,
                timestamp_ms=28000,
                action=ActionType.click,
                target="Sign in button",
                confidence=0.9,
            ),
        ],
    )
    return expected, observed


def test_screen_landing_click_without_evidence_does_not_satisfy_page_assertion():
    obs = ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=ActionType.click,
        target="Terms and Conditions link",
        confidence=0.9,
        screen_description="Terms text screen opened",
    )
    assert (
        assertion_satisfied(
            "The Terms and Conditions page opens",
            obs,
            expected_target="Terms and Conditions page",
        )
        is False
    )


def test_screen_landing_navigate_without_visible_text_does_not_satisfy_page_assertion():
    obs = ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=ActionType.navigate,
        target="github.com/login",
        confidence=0.9,
        screen_description="GitHub login form",
    )
    assert (
        assertion_satisfied(
            "the GitHub login page is displayed",
            obs,
            expected_target="GitHub login page",
        )
        is False
    )


def test_episode_verify_prefers_passive_terms_page_over_terms_click():
    from reckon.episode_verifier import verify_step_against_episodes
    from reckon.episodes import cluster_observed_episodes

    expected = ExpectedTrace(
        scenario_title="Terms",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Terms and Conditions"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="Terms and Conditions page",
                assertion="The Terms and Conditions page opens",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Terms and Conditions link",
                confidence=0.9,
                screen_description="Terms text screen opened",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=1100,
                action=ActionType.assert_visible,
                target="Terms and Conditions content",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="Mentorship terms text displayed",
                    visible_text='By clicking the "Join mentorship" button, you are agreeing',
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    # Force assertion failure on step 2 so episode verify runs.
    audit = audit.model_copy(
        update={
            "step_verdicts": [
                audit.step_verdicts[0],
                audit.step_verdicts[1].model_copy(
                    update={
                        "passed": False,
                        "finding_types": [FindingType.assertion_not_observed],
                    }
                ),
            ],
            "findings": [
                Finding(
                    type=FindingType.assertion_not_observed,
                    severity=FindingSeverity.critical,
                    description="missing terms page",
                    expected_step_index=2,
                )
            ],
        }
    )
    record = verify_step_against_episodes(
        expected.steps[1],
        expected,
        observed,
        audit,
        cluster_observed_episodes(observed),
        allow_llm=False,
    )
    assert record.status.value == "accepted"
    assert record.supporting_observed_index == 2


def test_app_launch_other_recovers_driver_app_icon_click_when_account_split():
    from reckon.episode_interaction import remap_interactions_against_episodes
    from reckon.episodes import cluster_observed_episodes

    expected = ExpectedTrace(
        scenario_title="Launch",
        steps=[
            ExpectedStep(index=1, action=ActionType.other, target="test account"),
            ExpectedStep(index=2, action=ActionType.other, target="driver app"),
            ExpectedStep(
                index=3,
                action=ActionType.assert_visible,
                target="welcome",
                assertion="Welcome to the Driver app",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Uber Driver app icon",
                confidence=0.9,
                screen_description="Uber app launching",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=2000,
                action=ActionType.assert_visible,
                target="Uber Driver welcome screen",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="Driver app welcome screen",
                    visible_text="Welcome to the Driver app",
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    records = remap_interactions_against_episodes(
        expected,
        observed,
        audit,
        cluster_observed_episodes(observed),
    )
    assert any(
        r.expected_step_index == 2
        and r.status.value == "accepted"
        and r.supporting_observed_index == 1
        for r in records
    )


def test_app_launch_without_split_account_does_not_recover():
    from reckon.episode_interaction import remap_interactions_against_episodes
    from reckon.episodes import cluster_observed_episodes

    expected = ExpectedTrace(
        scenario_title="Open app only",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="driver app",
                raw_text="Open the driver app",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_state,
                target="home",
                assertion="You're offline",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Uber Driver app icon",
                confidence=0.9,
                screen_description="Uber app launching with splash screen",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=3000,
                action=ActionType.assert_visible,
                target="Driver home screen",
                confidence=0.85,
                assertion_evidence=AssertionEvidence(
                    description="Driver home shows offline status",
                    visible_text="You're offline",
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    records = remap_interactions_against_episodes(
        expected,
        observed,
        audit,
        cluster_observed_episodes(observed),
    )
    assert not any(
        r.expected_step_index == 1 and r.status.value == "accepted"
        for r in records
    )


def test_app_launch_does_not_recover_login_worded_step_when_account_step_split():
    from reckon.episode_interaction import remap_interactions_against_episodes
    from reckon.episodes import cluster_observed_episodes

    expected = ExpectedTrace(
        scenario_title="Split account and launch",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="test account",
                raw_text="Create a test account and log in to the driver app.",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.other,
                target="driver app",
                raw_text="Create a test account and log in to the driver app.",
            ),
            ExpectedStep(
                index=3,
                action=ActionType.assert_state,
                target="driver app",
                assertion="User logs in successfully.",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Uber Driver app icon",
                confidence=0.9,
                screen_description="Uber app launching",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5000,
                action=ActionType.assert_visible,
                target="Driver home screen",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="home",
                    visible_text="You're offline",
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    records = remap_interactions_against_episodes(
        expected,
        observed,
        audit,
        cluster_observed_episodes(observed),
    )
    assert not any(r.status.value == "accepted" for r in records)


def test_mentor_feed_card_not_satisfied_by_deeplink_launch():
    expected = ExpectedTrace(
        scenario_title="Feed vs deeplink",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Mentor feed card"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="Mentor landing page",
                assertion="Earn when you mentor new drivers",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=5000,
                action=ActionType.click,
                target="Launch Deeplink button",
                confidence=0.95,
                screen_description="Mentorship screen opened",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5100,
                action=ActionType.assert_visible,
                target="Mentor landing",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="landing",
                    visible_text="Earn when you mentor new drivers",
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    assert any(
        f.expected_step_index == 1
        and f.type in {FindingType.missing_action, FindingType.wrong_target}
        for f in audit.findings
    )


def test_p0_evidence_bundle_clears_assertion_only_missing_action():
    expected, observed = _github_login_page_traces()
    steps = list(expected.steps)
    steps[2] = steps[2].model_copy(update={"assertion": "two-factor code"})
    expected = expected.model_copy(update={"steps": steps})
    deterministic = align(expected, observed)
    assert any(
        f.type == FindingType.missing_action and f.expected_step_index == 3
        for f in deterministic.findings
    )
    accepted = [
        AssistProposalRecord(
            kind=AssistCandidateKind.evidence_bundle,
            status=AssistProposalStatus.accepted,
            expected_step_index=3,
            supporting_observed_index=3,
            finding_types=[FindingType.missing_action],
            confidence=0.85,
            reason="assertion_supported",
            matching_method=AssistMatchMethod.llm_assisted,
        )
    ]
    refined = apply_assist_proposals(deterministic, accepted, expected, observed)
    by = {v.expected_step_index: v for v in refined.step_verdicts}
    assert by[3].passed is True
    assert FindingType.missing_action not in by[3].finding_types
    assert by[3].cleared_by_assist is True
    assert by[3].assist_confidence == 0.85
    assert by[3].matched_observed_index == 3
    assert not any(
        f.type == FindingType.missing_action and f.expected_step_index == 3
        for f in refined.findings
    )


def test_p0_assertion_evidence_does_not_clear_genuine_user_action():
    expected = ExpectedTrace(
        scenario_title="Need click",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Submit"),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.assert_visible,
                target="Form",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Form shown",
                    visible_text="Submit",
                ),
            ),
        ],
    )
    deterministic = align(expected, observed)
    accepted = [
        AssistProposalRecord(
            kind=AssistCandidateKind.evidence_bundle,
            status=AssistProposalStatus.accepted,
            expected_step_index=1,
            supporting_observed_index=1,
            finding_types=[FindingType.missing_action],
            confidence=0.9,
            reason="assertion_supported",
            matching_method=AssistMatchMethod.llm_assisted,
        )
    ]
    refined = apply_assist_proposals(deterministic, accepted, expected, observed)
    # Click requirement must remain — assertion evidence cannot stand in for it.
    assert any(
        f.type == FindingType.missing_action and f.expected_step_index == 1
        for f in refined.findings
    )


def test_p0_stage2c_timestamp_recovery_clears_wrong_target_on_assertion_step():
    expected = ExpectedTrace(
        scenario_title="Landing",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_visible,
                target="Mentor landing page",
                assertion="Earn when you mentor new drivers",
            )
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.assert_visible,
                target="splash",
                confidence=0.7,
            )
        ],
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
                finding_types=[
                    FindingType.wrong_target,
                    FindingType.missing_action,
                ],
            )
        ],
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.critical,
                description="bad pair",
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=1000,
            ),
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.critical,
                description="missing",
                expected_step_index=1,
            ),
        ],
    )
    accepted = [
        AssistProposalRecord(
            kind=AssistCandidateKind.frame_evidence,
            status=AssistProposalStatus.accepted,
            expected_step_index=1,
            finding_types=[FindingType.missing_action],
            matching_method=AssistMatchMethod.llm_assisted,
            confidence=0.9,
            evidence_timestamp_ms=8000,
            candidate_evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
        )
    ]
    refined = apply_assist_proposals(deterministic, accepted, expected, observed)
    assert refined.step_verdicts[0].passed is True
    assert refined.step_verdicts[0].cleared_by_assist is True
    assert [(f.type, f.observed_action_index) for f in refined.findings] == [
        (FindingType.extra_action, 1)
    ]


def test_p0_wrong_input_value_remains_protected():
    expected = ExpectedTrace(
        scenario_title="Type",
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
            ObservedAction(
                index=2,
                timestamp_ms=2000,
                action=ActionType.assert_text,
                target="banner",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Welcome",
                    visible_text="Welcome",
                ),
            ),
        ],
    )
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
    accepted = [
        AssistProposalRecord(
            kind=AssistCandidateKind.frame_evidence,
            status=AssistProposalStatus.accepted,
            expected_step_index=2,
            confidence=0.9,
            reason="assertion_frame_supported",
            matching_method=AssistMatchMethod.llm_assisted,
            evidence_timestamp_ms=2000,
            candidate_evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
        )
    ]
    refined = apply_assist_proposals(deterministic, accepted, expected, observed)
    assert any(f.type == FindingType.wrong_input_value for f in refined.findings)
    by = {v.expected_step_index: v for v in refined.step_verdicts}
    assert by[2].passed is True


def test_p1_sign_in_then_login_page_not_premature():
    expected, observed = _github_login_page_traces()
    proof = observed.actions[2]  # login page at 6500
    assert proof_is_premature(expected.steps[2], expected, proof, observed) is False


def test_p1_genuine_premature_proof_still_rejected():
    expected = ExpectedTrace(
        scenario_title="Open then assert",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Driver app"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_state,
                target="home",
                assertion="You're offline",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.assert_visible,
                target="home",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="offline",
                    visible_text="You're offline",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5000,
                action=ActionType.click,
                target="Uber Driver app icon",
                confidence=0.9,
            ),
        ],
    )
    assert proof_is_premature(
        expected.steps[1], expected, observed.actions[0], observed
    )


def test_p2_empty_dashboard_supported_by_no_learners_yet():
    assertion = "The empty Mentor Dashboard is displayed"
    evidence = "No learners yet. You'll be alerted soon."
    assert empty_state_evidence_supports_assertion(assertion, evidence)
    assert _introduces_negation(assertion, evidence) is False


def test_p2_genuine_negative_login_failure_still_rejected():
    assertion = "User logs in successfully"
    evidence = "Login failed"
    assert _introduces_negation(assertion, evidence) is True


def test_p2_past_learners_mentees_compatible():
    obs = ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=ActionType.assert_visible,
        target="Past learners page",
        confidence=0.9,
        assertion_evidence=AssertionEvidence(
            description="Past learners stats",
            visible_text="Past learners Matched learners 0",
        ),
        screen_description="Past learners",
    )
    assert assertion_target_relevant("Past Mentees Dashboard", obs) is True


def test_p2_mentorship_settings_program_settings_compatible():
    obs = ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=ActionType.assert_visible,
        target="Mentorship settings",
        confidence=0.9,
        assertion_evidence=AssertionEvidence(
            description="Settings entry",
            visible_text="Mentorship settings",
        ),
        screen_description="Mentorship settings screen",
    )
    assert assertion_target_relevant("Program Settings page", obs) is True


def test_p2_genuine_target_role_mismatch_preserved():
    obs = ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=ActionType.click,
        target="Join Mentorship button",
        confidence=0.9,
        screen_description="Language screen",
    )
    assert assertion_target_relevant("Mentorship title", obs) is False


def test_p4_evidence_window_finds_later_proof_despite_bad_dp():
    expected = ExpectedTrace(
        scenario_title="Mentor landing",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Continue"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="Mentor landing page",
                assertion="Earn when you mentor new drivers",
            ),
            ExpectedStep(index=3, action=ActionType.click, target="I'm interested"),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Continue button",
                confidence=0.8,
            ),
            ObservedAction(
                index=2,
                timestamp_ms=2000,
                action=ActionType.assert_visible,
                target="Uber splash screen",
                confidence=0.8,
                assertion_evidence=AssertionEvidence(
                    description="splash",
                    visible_text="Uber",
                ),
            ),
            ObservedAction(
                index=3,
                timestamp_ms=5000,
                action=ActionType.assert_visible,
                target="Mentorship program screen",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="landing",
                    visible_text="Earn when you mentor new drivers",
                ),
            ),
            ObservedAction(
                index=4,
                timestamp_ms=6000,
                action=ActionType.click,
                target="I'm interested button",
                confidence=0.9,
            ),
        ],
    )
    audit = align(expected, observed)
    # Force drifted pairing for step 2 onto splash if needed.
    window = retrieve_evidence_window(
        expected.steps[1],
        expected,
        observed,
        audit,
        matched_observed_index=2,
    )
    indices = {event.observed_index for event in window}
    assert 3 in indices


def test_p3_stage2c_selects_despite_unreliable_dp_anchors():
    expected = ExpectedTrace(
        scenario_title="Landing after noise",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Continue"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="Mentor landing",
                assertion="Earn when you mentor new drivers",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Splash dismiss",
                confidence=0.7,
            ),
            ObservedAction(
                index=2,
                timestamp_ms=8000,
                action=ActionType.assert_visible,
                target="Mentor landing",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="landing",
                    visible_text="Earn when you mentor new drivers",
                ),
            ),
        ],
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        step_verdicts=[
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
                matched_observed_index=None,
                evidence_timestamp_ms=None,
                finding_types=[FindingType.missing_action],
            ),
        ],
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.critical,
                description="bad early pair",
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=1000,
            ),
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.critical,
                description="unmatched landing",
                expected_step_index=2,
            ),
        ],
    )
    targets = select_stage2c_targets(expected, deterministic, observed=observed)
    assert any(t.expected_step_index == 2 for t in targets)
    assert all(t.expected_step_index != 1 for t in targets)


def test_p3_stage2c_includes_suspicious_assertion_pass():
    expected = ExpectedTrace(
        scenario_title="Suspicious pass",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_visible,
                target="Mentor landing",
                assertion="Earn when you mentor new drivers",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=2000,
                action=ActionType.assert_visible,
                target="splash",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="splash",
                    visible_text="Uber",
                ),
            ),
        ],
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=2000,
                finding_types=[],
            ),
        ],
        findings=[],
    )
    targets = select_stage2c_targets(expected, deterministic, observed=observed)
    assert len(targets) == 1
    assert targets[0].evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME


def test_p4_primary_reentry_prefers_launch_over_notes_copy():
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.navigate,
                target="Notes app",
                confidence=0.9,
                screen_description="Note containing deeplink URL",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5000,
                action=ActionType.click,
                target="Launch Deeplink button",
                confidence=0.95,
                screen_description="Mentorship screen opened",
            ),
            ObservedAction(
                index=3,
                timestamp_ms=5100,
                action=ActionType.assert_visible,
                target="Mentor landing",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="landing",
                    visible_text="Earn when you mentor new drivers",
                ),
            ),
        ],
    )
    primary = primary_workflow_reentry(observed)
    assert primary is not None
    assert primary.index == 2


def test_p4_reentry_floor_applies_after_feed_card_not_as_feed_card_proof():
    expected = ExpectedTrace(
        scenario_title="Mentorship entry",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Mentor feed card"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="Mentor landing page",
                assertion="Earn when you mentor new drivers",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Continue",
                confidence=0.8,
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5000,
                action=ActionType.click,
                target="Launch Deeplink button",
                confidence=0.95,
                screen_description="Mentorship screen opened via deeplink",
            ),
            ObservedAction(
                index=3,
                timestamp_ms=6000,
                action=ActionType.assert_visible,
                target="Mentor landing",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="landing",
                    visible_text="Earn when you mentor new drivers",
                ),
            ),
        ],
    )
    assert reentry_lower_timestamp_ms(expected, observed, step_index=2) == 5000
    audit = align(expected, observed)
    assert any(
        f.expected_step_index == 1
        and f.type in {FindingType.missing_action, FindingType.wrong_target}
        for f in audit.findings
    )


def test_p4_deeplink_reanchors_evidence_window_without_passing_feed_card():
    expected = ExpectedTrace(
        scenario_title="Mentorship entry",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Mentor feed card"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="Mentor landing page",
                assertion="Earn when you mentor new drivers",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.assert_visible,
                target="home splash",
                confidence=0.8,
                assertion_evidence=AssertionEvidence(
                    description="splash",
                    visible_text="Uber",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5000,
                action=ActionType.navigate,
                target="uber://mentorship/signup",
                confidence=0.9,
                screen_description="Opened mentorship deeplink URL",
            ),
            ObservedAction(
                index=3,
                timestamp_ms=6000,
                action=ActionType.assert_visible,
                target="Mentor landing",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="landing",
                    visible_text="Earn when you mentor new drivers",
                ),
            ),
        ],
    )
    assert is_workflow_reentry_action(observed.actions[1]) is True
    floor = reentry_floor_for_step(
        expected.steps[1], observed, min_index=1, max_index=3
    )
    assert floor == 2
    audit = align(expected, observed)
    # Feed-card click must remain unmet — deeplink is an alternate path, not a substitute.
    assert any(
        f.expected_step_index == 1 and f.type == FindingType.missing_action
        for f in audit.findings
    ) or any(
        f.expected_step_index == 1 and f.type == FindingType.wrong_target
        for f in audit.findings
    )
    window = retrieve_evidence_window(
        expected.steps[1],
        expected,
        observed,
        audit,
        matched_observed_index=1,
    )
    indices = {event.observed_index for event in window}
    assert 3 in indices
    assert 1 not in indices or indices == {1, 2, 3}


def test_p5_stage4_prefers_satisfying_observation_over_dp():
    step = ExpectedStep(
        index=1,
        action=ActionType.assert_visible,
        target="Mentor landing page",
        assertion="Earn when you mentor new drivers",
    )
    observed = ObservedTrace(
        source_file="m.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=2000,
                action=ActionType.assert_visible,
                target="splash",
                confidence=0.8,
                assertion_evidence=AssertionEvidence(
                    description="splash",
                    visible_text="Uber",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=5000,
                action=ActionType.assert_visible,
                target="Mentorship program screen",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="landing",
                    visible_text="Earn when you mentor new drivers",
                ),
            ),
        ],
    )
    center, obs_index = _preferred_assertion_anchor(step, observed, 2000, 1)
    assert center == 5000
    assert obs_index == 2


def test_p5_stage4_includes_assertion_missing_action():
    expected = ExpectedTrace(
        scenario_title="Login page",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_visible,
                target="login",
                assertion="Sign in to GitHub",
            )
        ],
    )
    observed = ObservedTrace(
        source_file="g.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=6500,
                action=ActionType.navigate,
                target="github.com/login",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="login",
                    visible_text="Welcome",
                ),
            )
        ],
    )
    audit = align(expected, observed)
    targets = select_crosscheck_targets(expected, observed, audit)
    assert any(
        t.expected_step_index == 1 and t.kind.value == "assertion" for t in targets
    )
    assert any(t.anchor_timestamp_ms == 6500 for t in targets)
