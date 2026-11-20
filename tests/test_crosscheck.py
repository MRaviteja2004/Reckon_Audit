from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from PIL import Image

from reckon.aligner import align
from reckon.crosscheck import (
    _AssertionCheck,
    _EvidenceSupport,
    _SemanticAnswer,
    _SemanticCandidate,
    _SemanticCandidateKind,
    _SemanticDecision,
    _apply_semantic_decisions,
    _apply_verdicts,
    run_cross_check,
    select_crosscheck_targets,
    select_semantic_candidates,
    verify_semantic_candidate,
    verify_target,
)
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditResult,
    AuditVerdict,
    CrossCheckAnswer,
    CrossCheckKind,
    CrossCheckTarget,
    CrossCheckVerdict,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected(steps: list[ExpectedStep]) -> ExpectedTrace:
    return ExpectedTrace(scenario_title="login", steps=steps)


def _observed(actions: list[ObservedAction]) -> ObservedTrace:
    return ObservedTrace(source_file="rec.mp4", actions=actions)


def _api_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://genai-api.uberinternal.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


def _assertion_finding(step_index: int, obs_index: int, ts: int) -> Finding:
    return Finding(
        type=FindingType.assertion_not_observed,
        expected_step_index=step_index,
        observed_action_index=obs_index,
        evidence_timestamp_ms=ts,
        description="assertion not confirmed",
    )


def _ambiguous_assertion_traces() -> tuple[ExpectedTrace, ObservedTrace]:
    return (
        _expected([
            ExpectedStep(
                index=1,
                action=ActionType.assert_text,
                assertion="Invalid credentials",
            ),
        ]),
        _observed([
            ObservedAction(
                index=1,
                timestamp_ms=8000,
                action=ActionType.assert_text,
                confidence=0.95,
            ),
        ]),
    )


# ---------------------------------------------------------------------------
# text-only semantic verification
# ---------------------------------------------------------------------------

def _semantic_assertion_case(
    expected_text: str,
    observed_text: str,
) -> tuple[ExpectedTrace, ObservedTrace, AuditResult]:
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            target="Status",
            assertion=expected_text,
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=8000,
            action=ActionType.assert_text,
            target="Status",
            assertion_evidence=AssertionEvidence(
                description=observed_text,
                visible_text=observed_text,
            ),
            confidence=0.95,
        ),
    ])
    return expected, observed, align(expected, observed)


def _equivalent_decision(confidence: float = 0.95) -> _SemanticDecision:
    return _SemanticDecision(
        answer=_SemanticAnswer.equivalent,
        confidence=confidence,
        rationale="same UI meaning",
    )


@pytest.mark.parametrize(
    ("expected_target", "observed_target"),
    [
        ("username", "Username input field"),
        ("I'm interested", "I'm interested button"),
    ],
)
def test_semantic_candidates_skip_deterministic_target_matches(
    expected_target: str,
    observed_target: str,
):
    expected = _expected([
        ExpectedStep(index=1, action=ActionType.click, target=expected_target),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=1000,
            action=ActionType.click,
            target=observed_target,
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)

    assert select_semantic_candidates(expected, observed, audit) == []


@pytest.mark.parametrize(
    ("expected_text", "observed_text"),
    [
        (
            "Language selection page is displayed",
            "Language selection screen is visible",
        ),
        (
            "Account created successfully",
            "Your account has been created",
        ),
    ],
)
def test_semantic_candidates_select_ambiguous_assertions(
    expected_text: str,
    observed_text: str,
):
    expected, observed, audit = _semantic_assertion_case(
        expected_text, observed_text
    )

    candidates = select_semantic_candidates(expected, observed, audit)

    assert len(candidates) == 1
    assert candidates[0].kind == _SemanticCandidateKind.assertion
    assert candidates[0].expected_text == expected_text
    assert candidates[0].observed_text == observed_text


def test_apply_semantic_equivalence_keeps_critical_assertion_for_visual():
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    candidate = select_semantic_candidates(expected, observed, audit)[0]

    refined = _apply_semantic_decisions(
        audit,
        [(candidate, _equivalent_decision())],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined is audit
    assert FindingType.assertion_not_observed in {
        finding.type for finding in refined.findings
    }
    assert refined.step_verdicts[0].passed is False


def test_assertion_equivalence_does_not_clear_same_pair_findings():
    expected, observed, _ = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    expected.steps[0].target = "Success confirmation"
    observed.actions[0].target = "Account panel"
    observed.actions[0].screen_description = "Success state of the account panel"
    audit = align(expected, observed)
    assert {finding.type for finding in audit.findings} == {
        FindingType.wrong_target,
        FindingType.assertion_not_observed,
    }
    candidate = select_semantic_candidates(expected, observed, audit)[0]

    refined = _apply_semantic_decisions(
        audit,
        [(candidate, _equivalent_decision())],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert {finding.type for finding in refined.findings} == {
        FindingType.wrong_target,
        FindingType.assertion_not_observed,
    }
    assert refined.verdict == AuditVerdict.unfaithful


@pytest.mark.parametrize(
    ("answer", "confidence"),
    [
        (_SemanticAnswer.different, 0.99),
        (_SemanticAnswer.unclear, 0.99),
        (_SemanticAnswer.equivalent, 0.5),
    ],
)
def test_apply_semantic_nonaccepted_decision_keeps_finding(
    answer: _SemanticAnswer,
    confidence: float,
):
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    candidate = select_semantic_candidates(expected, observed, audit)[0]
    decision = _SemanticDecision(
        answer=answer,
        confidence=confidence,
        rationale="not confidently equivalent",
    )

    refined = _apply_semantic_decisions(
        audit,
        [(candidate, decision)],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined is audit


def test_semantic_candidates_skip_assertion_without_evidence():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            assertion="Account created successfully",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=8000,
            action=ActionType.assert_text,
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)

    assert select_semantic_candidates(expected, observed, audit) == []


def test_semantic_candidates_do_not_use_unmatched_adjacent_action():
    expected = _expected([
        ExpectedStep(index=1, action=ActionType.click, target="Primary action"),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=1000,
            action=ActionType.navigate,
            target="Primary action button",
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)

    assert audit.step_verdicts[0].matched_observed_index is None
    assert select_semantic_candidates(expected, observed, audit) == []


@pytest.mark.parametrize(
    "observed_text",
    ["Login failed", "Login not successful"],
)
def test_semantic_verification_does_not_call_model_for_negation_conflict(
    observed_text: str,
):
    expected, observed, audit = _semantic_assertion_case(
        "Login successful", observed_text
    )
    client = MagicMock()

    report = run_cross_check(
        expected, observed, audit, Path("rec.mp4"), openai_client=client
    )

    client.chat.completions.create.assert_not_called()
    assert report.refined_result == audit


def test_semantic_verification_does_not_call_model_for_numeric_conflict():
    expected, observed, audit = _semantic_assertion_case(
        "Order 4211 refunded", "Order 4212 refunded"
    )
    client = MagicMock()

    report = run_cross_check(
        expected, observed, audit, Path("rec.mp4"), openai_client=client
    )

    client.chat.completions.create.assert_not_called()
    assert report.refined_result == audit


def test_semantic_verification_does_not_call_model_for_wrong_input_value():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.type_text,
            target="Order",
            input_value="4211",
            assertion="Order accepted",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=8000,
            action=ActionType.type_text,
            target="Order",
            input_value="4212",
            assertion_evidence=AssertionEvidence(
                description="Your order has been accepted",
                visible_text="Your order has been accepted",
            ),
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)
    client = MagicMock()

    report = run_cross_check(
        expected, observed, audit, Path("rec.mp4"), openai_client=client
    )

    client.chat.completions.create.assert_not_called()
    assert report.refined_result == audit
    assert FindingType.wrong_input_value in {
        finding.type for finding in report.refined_result.findings
    }


def test_forged_semantic_equivalence_cannot_clear_protected_contradiction():
    expected, observed, audit = _semantic_assertion_case(
        "Login successful", "Login failed"
    )
    forged = _SemanticCandidate(
        kind=_SemanticCandidateKind.assertion,
        expected_step_index=1,
        observed_action_index=1,
        expected_action=ActionType.assert_text,
        observed_action=ActionType.assert_text,
        expected_text="Login successful",
        observed_text="Login failed",
    )

    refined = _apply_semantic_decisions(
        audit,
        [(forged, _equivalent_decision(0.99))],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined is audit


@pytest.mark.parametrize(
    ("expected_text", "observed_text"),
    [
        ("Control enabled", "Control disabled"),
        ("Option checked", "Option unchecked"),
        ("Operation success", "Operation failure"),
        ("Process completed", "Process pending"),
        ("User logged-in", "User logged-out"),
        ("Request approved", "Request declined"),
    ],
)
def test_semantic_equivalence_cannot_override_opposite_state(
    expected_text: str,
    observed_text: str,
):
    expected, observed, audit = _semantic_assertion_case(
        expected_text, observed_text
    )
    assert select_semantic_candidates(expected, observed, audit) == []
    forged = _SemanticCandidate(
        kind=_SemanticCandidateKind.assertion,
        expected_step_index=1,
        observed_action_index=1,
        expected_action=ActionType.assert_text,
        observed_action=ActionType.assert_text,
        expected_text=expected_text,
        observed_text=observed_text,
    )

    refined = _apply_semantic_decisions(
        audit,
        [(forged, _equivalent_decision(0.99))],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined is audit
    assert FindingType.assertion_not_observed in {
        finding.type for finding in refined.findings
    }


def test_semantic_equivalence_preserves_observed_index_and_timestamp():
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    candidate = select_semantic_candidates(expected, observed, audit)[0]

    refined = _apply_semantic_decisions(
        audit,
        [(candidate, _equivalent_decision())],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[0].evidence_timestamp_ms == 8000


def test_target_equivalence_preserves_unrelated_assertion_finding():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="Primary action",
            assertion="Dialog closes",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=2000,
            action=ActionType.click,
            target="Continue control",
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)
    candidate = select_semantic_candidates(expected, observed, audit)[0]
    assert candidate.kind == _SemanticCandidateKind.target

    refined = _apply_semantic_decisions(
        audit,
        [(candidate, _equivalent_decision())],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert [finding.type for finding in refined.findings] == [
        FindingType.assertion_not_observed
    ]
    assert refined.step_verdicts[0].finding_types == [
        FindingType.assertion_not_observed
    ]


def test_semantic_assertion_equivalence_preserves_score_and_verdict():
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    candidate = select_semantic_candidates(expected, observed, audit)[0]

    refined = _apply_semantic_decisions(
        audit,
        [(candidate, _equivalent_decision())],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert audit.verdict == AuditVerdict.unfaithful
    assert refined is audit
    assert refined.verdict == AuditVerdict.unfaithful
    assert refined.fidelity_score == audit.fidelity_score


def test_verify_semantic_candidate_does_not_send_code_owned_metadata():
    candidate = _SemanticCandidate(
        kind=_SemanticCandidateKind.assertion,
        expected_step_index=2,
        observed_action_index=3,
        expected_action=ActionType.assert_text,
        observed_action=ActionType.assert_visible,
        expected_text="Account created successfully",
        observed_text="Your account has been created",
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _equivalent_decision()

    decision = verify_semantic_candidate(client, candidate)

    assert decision.answer == _SemanticAnswer.equivalent
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_model"] is _SemanticDecision
    assert not hasattr(candidate, "evidence_timestamp_ms")
    user_content = kwargs["messages"][-1]["content"]
    assert "expected_step_index" not in user_content
    assert "observed_action_index" not in user_content


@pytest.mark.parametrize(
    "side_effect",
    [
        None,
        _api_error(400),
    ],
)
def test_semantic_model_failure_does_not_modify_audit(side_effect: Exception | None):
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    client = MagicMock()
    if side_effect is None:
        client.chat.completions.create.return_value = {
            "answer": "equivalent",
            "confidence": 0.99,
            "rationale": "same",
            "observed_action_index": 1,
        }
    else:
        client.chat.completions.create.side_effect = side_effect

    with patch("reckon.crosscheck.select_crosscheck_targets", return_value=[]):
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )

    assert report.refined_result is audit
    diagnostic = report.semantic_diagnostics[0]
    assert diagnostic.answer is None
    assert diagnostic.confidence is None
    assert diagnostic.applied is False
    assert diagnostic.error_type in {"APIStatusError", "ValidationError"}


def test_semantic_programming_error_is_not_silently_swallowed():
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )


def test_semantic_assertion_equivalence_followed_by_visual_unclear_keeps_finding():
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    assert FindingType.assertion_not_observed in {
        finding.type for finding in audit.findings
    }
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _equivalent_decision(0.95),
        _AssertionCheck(
            answer=CrossCheckAnswer.unclear,
            confidence=0.95,
            rationale="The relevant frame is ambiguous.",
            supporting_frame_index=None,
            evidence_support=_EvidenceSupport.ambiguous,
        ),
    ]
    fake_samples = [(8000, Image.new("RGB", (8, 8)))]

    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )

    refined = report.refined_result
    assert FindingType.assertion_not_observed in {
        finding.type for finding in refined.findings
    }
    assert refined.step_verdicts[0].passed is False
    assert refined.verdict == AuditVerdict.unfaithful
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[0].evidence_timestamp_ms == 8000
    assert client.chat.completions.create.call_count == 2
    assert report.verdicts[0].answer == CrossCheckAnswer.unclear
    assert len(report.semantic_diagnostics) == 1
    diagnostic = report.semantic_diagnostics[0]
    assert diagnostic.expected_step_index == 1
    assert diagnostic.observed_action_index == 1
    assert diagnostic.comparison_kind == "assertion"
    assert diagnostic.answer == "equivalent"
    assert diagnostic.confidence == 0.95
    assert diagnostic.applied is False
    assert diagnostic.error_type is None


def test_semantic_assertion_equivalence_requires_direct_visual_present():
    expected, observed, audit = _semantic_assertion_case(
        "Account created successfully",
        "Your account has been created",
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _equivalent_decision(0.95),
        _AssertionCheck(
            answer=CrossCheckAnswer.present,
            confidence=0.95,
            rationale="The relevant frame directly shows the confirmation.",
            supporting_frame_index=1,
            evidence_support=_EvidenceSupport.direct,
        ),
    ]
    fake_samples = [(8000, Image.new("RGB", (8, 8)))]

    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )

    assert report.semantic_diagnostics[0].applied is False
    assert report.verdicts[0].answer == CrossCheckAnswer.present
    assert report.refined_result.findings == []
    assert report.refined_result.step_verdicts[0].passed is True
    assert report.refined_result.fidelity_score == 1.0
    assert report.refined_result.verdict == AuditVerdict.faithful


def test_run_cross_check_records_each_semantic_candidate():
    # Sign in ↔ GitHub header Sign in is recovered in Stage 3 baseline;
    # Stage 4 semantic candidates cover remaining non-auth label drift.
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="Sign in button",
        ),
        ExpectedStep(
            index=2,
            action=ActionType.click,
            target="username field",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=2000,
            action=ActionType.click,
            target="Sign in button in the GitHub homepage header",
            confidence=0.95,
        ),
        ObservedAction(
            index=2,
            timestamp_ms=4000,
            action=ActionType.click,
            target="Username or email address field",
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)
    assert audit.step_verdicts[0].passed is True
    assert audit.step_verdicts[0].recovery_reason == "auth_cta_synonym_recovery"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _equivalent_decision(0.95),
    ]

    with patch("reckon.crosscheck.select_crosscheck_targets", return_value=[]):
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )

    assert client.chat.completions.create.call_count == 1
    assert len(report.semantic_diagnostics) == 1
    assert [
        (
            diagnostic.expected_step_index,
            diagnostic.observed_action_index,
            diagnostic.comparison_kind,
            diagnostic.applied,
        )
        for diagnostic in report.semantic_diagnostics
    ] == [
        (2, 2, "target", True),
    ]


def test_run_cross_check_cannot_semantically_override_login_failure():
    expected, observed, audit = _semantic_assertion_case(
        "Login successful",
        "Login failed",
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _equivalent_decision(0.99)

    report = run_cross_check(
        expected,
        observed,
        audit,
        Path("rec.mp4"),
        openai_client=client,
    )

    refined = report.refined_result
    assert FindingType.assertion_not_observed in {
        finding.type for finding in refined.findings
    }
    assert refined.step_verdicts[0].passed is False
    assert refined.verdict == AuditVerdict.unfaithful
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[0].evidence_timestamp_ms == 8000
    client.chat.completions.create.assert_not_called()


def test_github_auth_cta_header_recovered_before_stage4_semantic():
    """Sign in ↔ GitHub header Sign in is owned by Stage 3 auth CTA recovery."""
    expected = _expected([
        ExpectedStep(index=1, action=ActionType.click, target="Sign in button"),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=4000,
            action=ActionType.click,
            target="Sign in button in the GitHub homepage header",
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)
    assert audit.step_verdicts[0].passed is True
    assert audit.step_verdicts[0].recovery_reason == "auth_cta_synonym_recovery"
    assert select_semantic_candidates(expected, observed, audit) == []


@pytest.mark.parametrize(
    (
        "kind",
        "expected_text",
        "observed_text",
        "expected_action",
    ),
    [
        (
            _SemanticCandidateKind.target,
            "username field",
            "Username or email address field",
            ActionType.click,
        ),
    ],
)
def test_run_cross_check_applies_github_semantic_candidates(
    kind: _SemanticCandidateKind,
    expected_text: str,
    observed_text: str,
    expected_action: ActionType,
):
    expected = _expected([
        ExpectedStep(index=1, action=expected_action, target=expected_text),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=4000,
            action=expected_action,
            target=observed_text,
            confidence=0.95,
        ),
    ])
    audit = align(expected, observed)
    finding_type = FindingType.wrong_target

    candidates = select_semantic_candidates(expected, observed, audit)
    assert len(candidates) == 1
    assert candidates[0].kind == kind
    client = MagicMock()
    client.chat.completions.create.return_value = _equivalent_decision(0.95)

    with patch(
        "reckon.crosscheck._apply_semantic_decisions",
        wraps=_apply_semantic_decisions,
    ) as apply_decisions:
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )

    client.chat.completions.create.assert_called_once()
    user_content = client.chat.completions.create.call_args.kwargs[
        "messages"
    ][-1]["content"]
    expected_user_content = (
        f"Comparison kind: {kind.value}\n"
        f"Expected action: {expected_action.value}\n"
        f"Observed action: {expected_action.value}\n"
        f"Expected text: {expected_text!r}\n"
        f"Observed text: {observed_text!r}"
    )
    assert user_content == expected_user_content
    decisions = apply_decisions.call_args.args[1]
    assert len(decisions) == 1
    assert decisions[0][1].answer == _SemanticAnswer.equivalent
    assert decisions[0][1].confidence == 0.95
    assert finding_type not in {
        finding.type for finding in report.refined_result.findings
    }
    assert report.refined_result.step_verdicts[0].passed is True


def test_semantic_assertion_uses_matching_screen_context():
    expected, observed, audit = _semantic_assertion_case(
        "GitHub dashboard",
        "Dashboard",
    )
    observed.actions[0].screen_description = (
        "GitHub dashboard home page with repository list and feed visible."
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _equivalent_decision(0.95)

    with patch("reckon.crosscheck.select_crosscheck_targets", return_value=[]):
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )

    user_content = client.chat.completions.create.call_args.kwargs[
        "messages"
    ][-1]["content"]
    assert (
        "Observed screen context: 'GitHub dashboard home page with repository "
        "list and feed visible.'"
    ) in user_content
    assert report.semantic_diagnostics[0].answer == "equivalent"
    assert report.semantic_diagnostics[0].applied is False
    assert FindingType.assertion_not_observed in {
        finding.type for finding in report.refined_result.findings
    }


def test_semantic_assertion_rejects_conflicting_screen_context():
    expected, observed, audit = _semantic_assertion_case(
        "GitHub dashboard",
        "Dashboard",
    )
    observed.actions[0].screen_description = (
        "GitHub Settings page with profile settings visible."
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _SemanticDecision(
        answer=_SemanticAnswer.different,
        confidence=0.95,
        rationale="The context identifies the Settings page.",
    )

    with patch("reckon.crosscheck.select_crosscheck_targets", return_value=[]):
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("rec.mp4"),
            openai_client=client,
        )

    user_content = client.chat.completions.create.call_args.kwargs[
        "messages"
    ][-1]["content"]
    assert (
        "Observed screen context: 'GitHub Settings page with profile settings "
        "visible.'"
    ) in user_content
    assert report.semantic_diagnostics[0].answer == "different"
    assert report.semantic_diagnostics[0].applied is False
    assert FindingType.assertion_not_observed in {
        finding.type for finding in report.refined_result.findings
    }


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Current contradiction protection does not classify a sign-in page and "
        "dashboard as deterministic opposites"
    ),
)
def test_run_cross_check_keeps_github_page_state_contradiction():
    expected, observed, audit = _semantic_assertion_case(
        "GitHub sign-in page",
        "GitHub dashboard",
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _equivalent_decision(0.99)

    report = run_cross_check(
        expected,
        observed,
        audit,
        Path("rec.mp4"),
        openai_client=client,
    )

    assert FindingType.assertion_not_observed in {
        finding.type for finding in report.refined_result.findings
    }
    assert report.refined_result.verdict == AuditVerdict.unfaithful


# ---------------------------------------------------------------------------
# select_crosscheck_targets
# ---------------------------------------------------------------------------

def test_select_targets_routes_assertion_finding():
    expected = _expected([
        ExpectedStep(index=1, action=ActionType.assert_text, assertion="Invalid credentials"),
    ])
    observed = _observed([
        ObservedAction(index=1, timestamp_ms=8000, action=ActionType.assert_text, confidence=0.95),
    ])
    audit = AuditResult(
        fidelity_score=0.0,
        findings=[_assertion_finding(1, 1, 8000)],
        step_verdicts=[StepVerdict(
            expected_step_index=1, passed=False, matched_observed_index=1,
            evidence_timestamp_ms=8000, finding_types=[FindingType.assertion_not_observed],
        )],
    )

    targets = select_crosscheck_targets(expected, observed, audit)

    assertion_targets = [t for t in targets if t.kind == CrossCheckKind.assertion]
    assert len(assertion_targets) == 1
    assert "Invalid credentials" in assertion_targets[0].question
    assert assertion_targets[0].expected_step_index == 1


def test_select_targets_window_centered_on_evidence_timestamp():
    expected = _expected([
        ExpectedStep(index=1, action=ActionType.assert_text, assertion="Error"),
    ])
    observed = _observed([
        ObservedAction(index=1, timestamp_ms=10_000, action=ActionType.assert_text, confidence=0.95),
    ])
    audit = AuditResult(
        fidelity_score=0.0,
        findings=[_assertion_finding(1, 1, 10_000)],
        step_verdicts=[],
    )

    targets = select_crosscheck_targets(expected, observed, audit, window_pad_ms=2000)

    target = targets[0]
    assert target.window_start_ms == 8000
    assert target.window_end_ms == 12_000
    assert target.anchor_timestamp_ms == 10_000


def test_select_targets_routes_low_confidence_action():
    expected = _expected([ExpectedStep(index=1, action=ActionType.click, target="button")])
    observed = _observed([
        ObservedAction(index=1, timestamp_ms=3000, action=ActionType.click,
                       target="button", confidence=0.4),
    ])
    audit = AuditResult(fidelity_score=1.0, findings=[], step_verdicts=[])

    targets = select_crosscheck_targets(expected, observed, audit, confidence_threshold=0.7)

    low_conf = [t for t in targets if t.kind == CrossCheckKind.low_confidence]
    assert len(low_conf) == 1
    assert low_conf[0].observed_action_index == 1


def test_select_targets_skips_confident_actions():
    expected = _expected([ExpectedStep(index=1, action=ActionType.click, target="button")])
    observed = _observed([
        ObservedAction(index=1, timestamp_ms=3000, action=ActionType.click, confidence=0.95),
    ])
    audit = AuditResult(fidelity_score=1.0, findings=[], step_verdicts=[])

    targets = select_crosscheck_targets(expected, observed, audit)

    assert targets == []


def test_select_targets_does_not_double_target_same_action():
    expected = _expected([
        ExpectedStep(index=1, action=ActionType.assert_text, assertion="Error"),
    ])
    # The same action is both the assertion's match AND low-confidence.
    observed = _observed([
        ObservedAction(index=1, timestamp_ms=5000, action=ActionType.assert_text, confidence=0.3),
    ])
    audit = AuditResult(
        fidelity_score=0.0,
        findings=[_assertion_finding(1, 1, 5000)],
        step_verdicts=[],
    )

    targets = select_crosscheck_targets(expected, observed, audit)

    # Only the assertion target — the low-confidence path must skip the already-targeted action.
    assert len(targets) == 1
    assert targets[0].kind == CrossCheckKind.assertion


# ---------------------------------------------------------------------------
# verify_target
# ---------------------------------------------------------------------------

def test_verify_target_builds_verdict_from_model_answer():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion,
        question="Does 'Error' appear?",
        window_start_ms=4000,
        window_end_ms=8000,
        expected_step_index=2,
        observed_action_index=3,
    )
    client = MagicMock()
    check = _AssertionCheck(
        answer=CrossCheckAnswer.present,
        confidence=0.88,
        rationale="banner visible",
        supporting_frame_index=1,
        evidence_support=_EvidenceSupport.direct,
    )
    client.chat.completions.create.return_value = check

    fake_samples = [(5000, Image.new("RGB", (8, 8))), (7000, Image.new("RGB", (8, 8)))]
    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        verdict = verify_target(client, Path("rec.mp4"), target, frames_per_target=2)

    assert verdict.answer == CrossCheckAnswer.present
    assert verdict.confidence == 0.88
    assert verdict.frame_timestamps_ms == [5000, 7000]
    assert verdict.expected_step_index == 2
    assert verdict.observed_action_index == 3


def test_verify_target_sends_images_to_model():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion, question="q",
        window_start_ms=0, window_end_ms=2000,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssertionCheck(
        answer=CrossCheckAnswer.absent,
        confidence=0.9,
        rationale="not there",
        supporting_frame_index=None,
        evidence_support=_EvidenceSupport.none,
    )
    fake_samples = [(1000, Image.new("RGB", (8, 8)))]
    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        verify_target(client, Path("rec.mp4"), target, frames_per_target=1)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = messages[-1]["content"]
    image_parts = [c for c in user_content if c.get("type") == "image_url"]
    frame_labels = [
        c["text"]
        for c in user_content
        if c.get("type") == "text" and c["text"].startswith("Frame ")
    ]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert frame_labels == ["Frame 1"]


def test_verify_target_sends_each_unique_evidence_frame_once():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion,
        question="q",
        window_start_ms=1000,
        window_end_ms=7000,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssertionCheck(
        answer=CrossCheckAnswer.present,
        confidence=0.9,
        rationale="visible",
        supporting_frame_index=2,
        evidence_support=_EvidenceSupport.direct,
    )
    fake_samples = [
        (1000, Image.new("RGB", (8, 8))),
        (5900, Image.new("RGB", (8, 8))),
    ]

    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        verdict = verify_target(
            client,
            Path("rec.mp4"),
            target,
            frames_per_target=3,
        )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    image_parts = [
        part
        for part in messages[-1]["content"]
        if part.get("type") == "image_url"
    ]
    assert len(image_parts) == 2
    assert verdict.frame_timestamps_ms == [1000, 5900]
    assert 7000 not in verdict.frame_timestamps_ms


def test_verify_target_rejects_matching_phrase_on_pre_anchor_screen():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion,
        question="Is the Mentorship Program menu item visible in the drawer?",
        window_start_ms=1766,
        window_end_ms=6766,
        anchor_timestamp_ms=4266,
        expected_step_index=1,
        observed_action_index=1,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssertionCheck(
        answer=CrossCheckAnswer.present,
        confidence=0.86,
        rationale="The earlier confirmation mentions the Mentorship Program.",
        supporting_frame_index=1,
        evidence_support=_EvidenceSupport.direct,
    )
    fake_samples = [
        (1766, Image.new("RGB", (8, 8))),
        (4266, Image.new("RGB", (8, 8))),
    ]

    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        verdict = verify_target(client, Path("TC 29.mp4"), target)

    assert verdict.answer == CrossCheckAnswer.unclear


def test_verify_target_rejects_ambiguous_menu_item():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion,
        question="Is the Mentorship Program menu item visible in the drawer?",
        window_start_ms=1766,
        window_end_ms=6766,
        anchor_timestamp_ms=4266,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssertionCheck(
        answer=CrossCheckAnswer.present,
        confidence=0.86,
        rationale="An unlabeled menu item may represent the program.",
        supporting_frame_index=2,
        evidence_support=_EvidenceSupport.ambiguous,
    )
    fake_samples = [
        (1766, Image.new("RGB", (8, 8))),
        (4266, Image.new("RGB", (8, 8))),
    ]

    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        verdict = verify_target(client, Path("TC 29.mp4"), target)

    assert verdict.answer == CrossCheckAnswer.unclear


def test_verify_target_accepts_direct_support_on_relevant_screen():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion,
        question="Is the Mentorship Program menu item visible in the drawer?",
        window_start_ms=1766,
        window_end_ms=6766,
        anchor_timestamp_ms=4266,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssertionCheck(
        answer=CrossCheckAnswer.present,
        confidence=0.86,
        rationale="The named menu item is legible in the open drawer.",
        supporting_frame_index=2,
        evidence_support=_EvidenceSupport.direct,
    )
    fake_samples = [
        (1766, Image.new("RGB", (8, 8))),
        (4266, Image.new("RGB", (8, 8))),
    ]

    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        verdict = verify_target(client, Path("TC 29.mp4"), target)

    assert verdict.answer == CrossCheckAnswer.present


def test_verify_target_does_not_call_model_without_timestamped_evidence():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion,
        question="q",
        window_start_ms=1000,
        window_end_ms=7000,
    )
    client = MagicMock()

    with (
        patch(
            "reckon.crosscheck.extract_evidence_frames_in_window",
            return_value=[],
        ),
        pytest.raises(ValueError, match="No timestamped evidence frames"),
    ):
        verify_target(client, Path("rec.mp4"), target)

    client.chat.completions.create.assert_not_called()


def test_verify_target_retries_rate_limit():
    target = CrossCheckTarget(
        kind=CrossCheckKind.assertion,
        question="q",
        window_start_ms=0,
        window_end_ms=2000,
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _api_error(429),
        _AssertionCheck(
            answer=CrossCheckAnswer.present,
            confidence=0.9,
            rationale="visible",
            supporting_frame_index=1,
            evidence_support=_EvidenceSupport.direct,
        ),
    ]
    fake_samples = [(1000, Image.new("RGB", (8, 8)))]

    with (
        patch(
            "reckon.crosscheck.extract_evidence_frames_in_window",
            return_value=fake_samples,
        ),
        patch("reckon.retry.time.sleep"),
    ):
        verdict = verify_target(client, Path("rec.mp4"), target, frames_per_target=1)

    assert verdict.answer == CrossCheckAnswer.present
    assert client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# _apply_verdicts — overturning false assertion findings
# ---------------------------------------------------------------------------

def _audit_with_failed_assertion() -> AuditResult:
    return AuditResult(
        fidelity_score=0.0,
        findings=[_assertion_finding(1, 1, 8000)],
        step_verdicts=[StepVerdict(
            expected_step_index=1, passed=False, matched_observed_index=1,
            evidence_timestamp_ms=8000, finding_types=[FindingType.assertion_not_observed],
        )],
    )


def _critical_assertion_audit() -> AuditResult:
    return AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=8000,
                description="deterministic contradiction",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=8000,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
    )


def _wrong_target_visual_case() -> tuple[ExpectedTrace, ObservedTrace, AuditResult]:
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_visible,
            target="Uber Pro menu item",
            assertion="Uber Pro is visible in the navigation drawer",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=4266,
            action=ActionType.assert_visible,
            target="Side navigation drawer",
            confidence=0.95,
        ),
    ])
    audit = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=4266,
                description="Expected Uber Pro item, observed navigation drawer",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=4266,
                finding_types=[FindingType.wrong_target],
            ),
        ],
    )
    return expected, observed, audit


def test_apply_verdicts_resolves_matching_wrong_target_pair():
    expected, observed, audit = _wrong_target_visual_case()
    audit.findings.append(Finding(
        type=FindingType.extra_action,
        severity=FindingSeverity.advisory,
        observed_action_index=2,
        evidence_timestamp_ms=5000,
        description="Unrelated extra action",
    ))
    verdict = CrossCheckVerdict(
        question="Is Uber Pro visible?",
        answer=CrossCheckAnswer.present,
        confidence=0.98,
        rationale="Uber Pro is directly visible in the drawer.",
        expected_step_index=1,
        observed_action_index=1,
    )

    refined = _apply_verdicts(
        audit,
        [verdict],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert [finding.type for finding in refined.findings] == [
        FindingType.extra_action
    ]
    assert refined.step_verdicts[0].passed is True
    assert refined.step_verdicts[0].finding_types == []
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[0].evidence_timestamp_ms == 4266
    assert refined.fidelity_score == 1.0


def test_apply_verdicts_keeps_findings_for_incompatible_imperative_pair():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="account panel",
            assertion="Account panel closed",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=3000,
            action=ActionType.scroll,
            target="account panel",
            confidence=0.9,
        ),
    ])
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=3000,
                description="Incompatible action pair",
            ),
            _assertion_finding(1, 1, 3000),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=3000,
                finding_types=[
                    FindingType.wrong_target,
                    FindingType.assertion_not_observed,
                ],
            ),
        ],
    )
    verdict = CrossCheckVerdict(
        question="Is the account panel closed?",
        answer=CrossCheckAnswer.present,
        confidence=0.98,
        rationale="The account panel is not visible.",
        expected_step_index=1,
        observed_action_index=1,
    )

    refined = _apply_verdicts(
        audit,
        [verdict],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined is audit
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[0].evidence_timestamp_ms == 3000
    assert expected.steps[0].action == ActionType.click
    assert observed.actions[0].action == ActionType.scroll


def test_apply_verdicts_allows_direct_state_evidence_for_assertion_step():
    expected, observed, audit = _wrong_target_visual_case()
    observed.actions[0].action = ActionType.navigate
    verdict = CrossCheckVerdict(
        question="Is the expected state visible?",
        answer=CrossCheckAnswer.present,
        confidence=0.98,
        rationale="The expected state is directly visible.",
        expected_step_index=1,
        observed_action_index=1,
    )

    refined = _apply_verdicts(
        audit,
        [verdict],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined.findings == []
    assert refined.step_verdicts[0].passed is True
    assert refined.step_verdicts[0].matched_observed_index == 1
    assert refined.step_verdicts[0].evidence_timestamp_ms == 4266


def test_apply_verdicts_keeps_unrelated_wrong_target_pair():
    expected, observed, audit = _wrong_target_visual_case()
    audit.findings.append(Finding(
        type=FindingType.wrong_target,
        severity=FindingSeverity.advisory,
        expected_step_index=1,
        observed_action_index=2,
        evidence_timestamp_ms=5000,
        description="Different observed action",
    ))
    verdict = CrossCheckVerdict(
        question="Is Uber Pro visible?",
        answer=CrossCheckAnswer.present,
        confidence=0.98,
        rationale="Uber Pro is directly visible in the drawer.",
        expected_step_index=1,
        observed_action_index=1,
    )

    refined = _apply_verdicts(
        audit,
        [verdict],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert len(refined.findings) == 1
    assert refined.findings[0].type == FindingType.wrong_target
    assert refined.findings[0].observed_action_index == 2
    assert refined.step_verdicts[0].passed is False


@pytest.mark.parametrize(
    "answer",
    [CrossCheckAnswer.absent, CrossCheckAnswer.unclear],
)
def test_apply_verdicts_non_present_keeps_wrong_target(
    answer: CrossCheckAnswer,
):
    expected, observed, audit = _wrong_target_visual_case()
    verdict = CrossCheckVerdict(
        question="Is Uber Pro visible?",
        answer=answer,
        confidence=0.98,
        rationale="The item is not directly established.",
        expected_step_index=1,
        observed_action_index=1,
    )

    refined = _apply_verdicts(
        audit,
        [verdict],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined is audit


def test_apply_verdicts_low_confidence_present_keeps_wrong_target():
    expected, observed, audit = _wrong_target_visual_case()
    verdict = CrossCheckVerdict(
        question="Is Uber Pro visible?",
        answer=CrossCheckAnswer.present,
        confidence=0.69,
        rationale="The item may be visible.",
        expected_step_index=1,
        observed_action_index=1,
    )

    refined = _apply_verdicts(
        audit,
        [verdict],
        expected_trace=expected,
        observed_trace=observed,
    )

    assert refined is audit


def test_apply_verdicts_overturns_confident_present():
    expected, observed = _ambiguous_assertion_traces()
    audit = _audit_with_failed_assertion()
    verdicts = [CrossCheckVerdict(
        question="q", answer=CrossCheckAnswer.present, confidence=0.85,
        rationale="banner clearly visible", expected_step_index=1, observed_action_index=1,
    )]

    targets = select_crosscheck_targets(expected, observed, audit)
    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert len(targets) == 1
    assert targets[0].kind == CrossCheckKind.assertion
    assert refined.findings == []
    assert refined.step_verdicts[0].passed is True
    assert refined.fidelity_score == 1.0


def test_apply_verdicts_keeps_finding_on_absent():
    expected, observed = _ambiguous_assertion_traces()
    audit = _audit_with_failed_assertion()
    verdicts = [CrossCheckVerdict(
        question="q", answer=CrossCheckAnswer.absent, confidence=0.95,
        rationale="no banner", expected_step_index=1, observed_action_index=1,
    )]

    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert len(refined.findings) == 1
    assert refined.step_verdicts[0].passed is False


def test_apply_verdicts_absent_reinstates_low_confidence_assist_clear():
    expected, observed = _ambiguous_assertion_traces()
    audit = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
                finding_types=[],
                cleared_by_assist=True,
                assist_confidence=0.7,
            )
        ],
    )
    verdicts = [
        CrossCheckVerdict(
            question="q",
            answer=CrossCheckAnswer.absent,
            confidence=0.9,
            rationale="opt-out screen is not settings",
            expected_step_index=1,
            observed_action_index=1,
        )
    ]

    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert refined.step_verdicts[0].passed is False
    assert refined.step_verdicts[0].cleared_by_assist is False
    assert any(
        f.type == FindingType.assertion_not_observed and f.expected_step_index == 1
        for f in refined.findings
    )


def test_apply_verdicts_absent_keeps_high_confidence_assist_clear():
    expected, observed = _ambiguous_assertion_traces()
    audit = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
                finding_types=[],
                cleared_by_assist=True,
                assist_confidence=0.95,
            )
        ],
    )
    verdicts = [
        CrossCheckVerdict(
            question="q",
            answer=CrossCheckAnswer.absent,
            confidence=0.9,
            rationale="model disagrees",
            expected_step_index=1,
            observed_action_index=1,
        )
    ]

    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert refined.step_verdicts[0].passed is True
    assert refined.findings == []


def test_apply_verdicts_does_not_overturn_low_confidence_present():
    expected, observed = _ambiguous_assertion_traces()
    audit = _audit_with_failed_assertion()
    verdicts = [CrossCheckVerdict(
        question="q", answer=CrossCheckAnswer.present, confidence=0.5,
        rationale="maybe", expected_step_index=1, observed_action_index=1,
    )]

    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert len(refined.findings) == 1


def test_apply_verdicts_unclear_does_not_overturn():
    expected, observed = _ambiguous_assertion_traces()
    audit = _audit_with_failed_assertion()
    verdicts = [CrossCheckVerdict(
        question="q", answer=CrossCheckAnswer.unclear, confidence=0.9,
        rationale="ambiguous", expected_step_index=1, observed_action_index=1,
    )]

    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert len(refined.findings) == 1


@pytest.mark.parametrize("observed_text", ["Login failed", "Login not successful"])
def test_crosscheck_cannot_override_explicit_negation(
    observed_text: str,
):
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            assertion="Login successful",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=8000,
            action=ActionType.assert_text,
            assertion_evidence=AssertionEvidence(
                description="Login status",
                visible_text=observed_text,
            ),
            confidence=0.95,
        ),
    ])
    audit = _critical_assertion_audit()
    verdicts = [
        CrossCheckVerdict(
            question="q",
            answer=CrossCheckAnswer.present,
            confidence=0.99,
            rationale="matching words appear",
            expected_step_index=1,
            observed_action_index=1,
        ),
    ]

    assert select_crosscheck_targets(expected, observed, audit) == []
    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert refined.findings == audit.findings
    assert refined.verdict == AuditVerdict.unfaithful
    assert refined.step_verdicts[0].passed is False


def test_crosscheck_cannot_override_numeric_assertion_conflict():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            assertion="Order 4211 refunded",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=8000,
            action=ActionType.assert_text,
            assertion_evidence=AssertionEvidence(
                description="Refund confirmation",
                visible_text="Order 4212 refunded",
            ),
            confidence=0.95,
        ),
    ])
    audit = _critical_assertion_audit()
    verdicts = [
        CrossCheckVerdict(
            question="q",
            answer=CrossCheckAnswer.present,
            confidence=0.99,
            rationale="refund text appears",
            expected_step_index=1,
            observed_action_index=1,
        ),
    ]

    assert select_crosscheck_targets(expected, observed, audit) == []
    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert refined.findings == audit.findings
    assert refined.verdict == AuditVerdict.unfaithful


def test_apply_verdicts_preserves_other_findings_on_same_step():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.type_text,
            input_value="4211",
            assertion="Order accepted",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=8000,
            action=ActionType.type_text,
            input_value="4212",
            confidence=0.95,
        ),
    ])
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_input_value,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=8000,
                description="expected 4211 but observed 4212",
            ),
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=8000,
                description="target wording differs",
            ),
            _assertion_finding(1, 1, 8000),
        ],
        step_verdicts=[StepVerdict(
            expected_step_index=1, passed=False, matched_observed_index=1,
            evidence_timestamp_ms=8000,
            finding_types=[
                FindingType.wrong_input_value,
                FindingType.wrong_target,
                FindingType.assertion_not_observed,
            ],
        )],
    )
    verdicts = [CrossCheckVerdict(
        question="q", answer=CrossCheckAnswer.present, confidence=0.9,
        rationale="visible", expected_step_index=1, observed_action_index=1,
    )]

    assert select_crosscheck_targets(expected, observed, audit) == []
    refined = _apply_verdicts(
        audit, verdicts, expected_trace=expected, observed_trace=observed
    )

    assert {finding.type for finding in refined.findings} == {
        FindingType.wrong_input_value,
        FindingType.wrong_target,
        FindingType.assertion_not_observed,
    }
    assert refined.verdict == AuditVerdict.unfaithful
    assert refined.step_verdicts[0].passed is False


def test_apply_verdicts_retargets_mentor_dashboard_evidence_away_from_join_confirmation():
    """E27: Stage 4 clear must cite dashboard evidence, not join confirmation."""
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_visible,
            target="Mentor Dashboard",
            assertion="The empty Mentor Dashboard is displayed",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=61,
            timestamp_ms=209765,
            action=ActionType.assert_visible,
            target="Join confirmation screen",
            confidence=0.9,
            assertion_evidence=AssertionEvidence(
                description="Success message with checkmark confirming joining mentorship programme",
                visible_text="You've now joined the Uber mentorship programme",
            ),
            screen_description="Confirmation screen with Continue button",
        ),
        ObservedAction(
            index=70,
            timestamp_ms=240750,
            action=ActionType.assert_visible,
            target="Mentorship screen",
            confidence=0.9,
            assertion_evidence=AssertionEvidence(
                description="Empty state on the Mentorship page",
                visible_text="No learners yet. You'll be alerted soon.",
            ),
            screen_description=(
                "Mentorship page showing 0 active learners, $0 you can earn, "
                "and a How it works button"
            ),
        ),
    ])
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=61,
                evidence_timestamp_ms=209765,
                description="empty dashboard not confirmed",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=61,
                evidence_timestamp_ms=209765,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
    )
    verdict = CrossCheckVerdict(
        question="Is the empty Mentor Dashboard visible?",
        answer=CrossCheckAnswer.present,
        confidence=0.92,
        rationale="Empty dashboard visible in later frame",
        expected_step_index=1,
        observed_action_index=61,
        supporting_frame_timestamp_ms=240750,
    )

    refined = _apply_verdicts(
        audit, [verdict], expected_trace=expected, observed_trace=observed
    )

    assert [(f.type, f.observed_action_index) for f in refined.findings] == [
        (FindingType.extra_action, 61)
    ]
    assert refined.step_verdicts[0].passed is True
    assert refined.step_verdicts[0].matched_observed_index == 70
    assert refined.step_verdicts[0].evidence_timestamp_ms == 240750


def test_apply_verdicts_does_not_retarget_onto_another_passed_steps_observation():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_visible,
            target="Mentorship screen",
            assertion="No learners yet",
        ),
        ExpectedStep(
            index=2,
            action=ActionType.assert_visible,
            target="Mentor Dashboard",
            assertion="The empty Mentor Dashboard is displayed",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=61,
            timestamp_ms=209765,
            action=ActionType.assert_visible,
            target="Join confirmation screen",
            confidence=0.9,
            assertion_evidence=AssertionEvidence(
                description="Success message",
                visible_text="You've now joined the mentorship programme",
            ),
        ),
        ObservedAction(
            index=70,
            timestamp_ms=240750,
            action=ActionType.assert_visible,
            target="Mentorship screen",
            confidence=0.9,
            assertion_evidence=AssertionEvidence(
                description="Empty state",
                visible_text="No learners yet. You'll be alerted soon.",
            ),
        ),
    ])
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.5,
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=2,
                observed_action_index=61,
                evidence_timestamp_ms=209765,
                description="empty dashboard not confirmed",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=70,
                evidence_timestamp_ms=240750,
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                matched_observed_index=61,
                evidence_timestamp_ms=209765,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
    )
    verdict = CrossCheckVerdict(
        question="Is the empty Mentor Dashboard visible?",
        answer=CrossCheckAnswer.present,
        confidence=0.92,
        rationale="Empty dashboard visible in later frame",
        expected_step_index=2,
        observed_action_index=61,
        supporting_frame_timestamp_ms=240750,
    )

    refined = _apply_verdicts(
        audit, [verdict], expected_trace=expected, observed_trace=observed
    )

    assert (
        refined.step_verdicts[1].passed,
        refined.step_verdicts[1].matched_observed_index,
    ) == (True, 61)


# ---------------------------------------------------------------------------
# run_cross_check
# ---------------------------------------------------------------------------

def test_run_cross_check_no_targets_returns_original_unchanged():
    expected = _expected([ExpectedStep(index=1, action=ActionType.click, target="b")])
    observed = _observed([
        ObservedAction(index=1, timestamp_ms=1000, action=ActionType.click, confidence=0.95),
    ])
    audit = AuditResult(fidelity_score=1.0, findings=[], step_verdicts=[])

    report = run_cross_check(expected, observed, audit, Path("rec.mp4"))

    assert report.verdicts == []
    assert report.refined_result is audit


def test_run_cross_check_preserves_finding_for_unrelated_pre_anchor_phrase():
    expected = _expected([
        ExpectedStep(
            index=1,
            action=ActionType.assert_visible,
            target="Mentorship Program menu item",
            assertion="Mentorship Program menu item is visible",
        ),
    ])
    observed = _observed([
        ObservedAction(
            index=1,
            timestamp_ms=4266,
            action=ActionType.assert_visible,
            target="navigation drawer",
            confidence=0.95,
            screen_description="Open navigation drawer with Uber Pro item",
        ),
    ])
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=4266,
                description="Expected menu item, observed navigation drawer",
            ),
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=4266,
                description="Mentorship Program item was not identified",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=4266,
                finding_types=[
                    FindingType.wrong_target,
                    FindingType.assertion_not_observed,
                ],
            ),
        ],
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssertionCheck(
        answer=CrossCheckAnswer.present,
        confidence=0.86,
        rationale="An earlier confirmation mentions Mentorship Program.",
        supporting_frame_index=1,
        evidence_support=_EvidenceSupport.direct,
    )
    fake_samples = [
        (1766, Image.new("RGB", (8, 8))),
        (4266, Image.new("RGB", (8, 8))),
    ]

    with (
        patch(
            "reckon.crosscheck.extract_evidence_frames_in_window",
            return_value=fake_samples,
        ),
        patch("reckon.crosscheck.select_semantic_candidates", return_value=[]),
    ):
        report = run_cross_check(
            expected,
            observed,
            audit,
            Path("TC 29.mp4"),
            openai_client=client,
        )

    assert report.verdicts[0].answer == CrossCheckAnswer.unclear
    assert report.refined_result.findings == audit.findings
    assert report.refined_result.verdict == AuditVerdict.unfaithful
    assert report.refined_result.step_verdicts[0].passed is False


def test_run_cross_check_overturns_via_injected_client():
    expected = _expected([
        ExpectedStep(index=1, action=ActionType.assert_text, assertion="Invalid credentials"),
    ])
    observed = _observed([
        ObservedAction(index=1, timestamp_ms=8000, action=ActionType.assert_text, confidence=0.95),
    ])
    audit = _audit_with_failed_assertion()

    client = MagicMock()
    client.chat.completions.create.return_value = _AssertionCheck(
        answer=CrossCheckAnswer.present,
        confidence=0.9,
        rationale="banner visible",
        supporting_frame_index=1,
        evidence_support=_EvidenceSupport.direct,
    )

    fake_samples = [(8000, Image.new("RGB", (8, 8)))]
    with patch(
        "reckon.crosscheck.extract_evidence_frames_in_window",
        return_value=fake_samples,
    ):
        report = run_cross_check(
            expected, observed, audit, Path("rec.mp4"),
            openai_client=client, frames_per_target=1,
        )

    assert len(report.verdicts) == 1
    assert report.semantic_diagnostics == []
    assert report.refined_result.fidelity_score == 1.0
    assert report.refined_result.findings == []
