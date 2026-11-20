"""An assertion step pairs only with evidence for that assertion."""

from reckon.aligner import align
from reckon.ownership import drop_shared_finding_citations
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditResult,
    AuditVerdict,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)


def _expected(*steps: ExpectedStep) -> ExpectedTrace:
    return ExpectedTrace(scenario_title="Assertions", steps=list(steps))


def _observed(*actions: ObservedAction) -> ObservedTrace:
    return ObservedTrace(source_file="assertions.mp4", actions=list(actions))


def _assert(
    index: int,
    target: str,
    assertion: str,
    *,
    visible: str | None = None,
    description: str | None = None,
    screen: str | None = None,
    timestamp_ms: int = 1000,
) -> ObservedAction:
    evidence = None
    if visible is not None or description is not None:
        evidence = AssertionEvidence(
            description=description or "",
            visible_text=visible or "",
        )
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=ActionType.assert_visible,
        target=target,
        confidence=0.9,
        assertion_evidence=evidence,
        screen_description=screen,
    )


def test_unrelated_visible_text_does_not_consume_the_observation():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Continue"),
        ExpectedStep(
            index=2,
            action=ActionType.assert_visible,
            target="receipt",
            assertion="Payment received",
        ),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=1000,
            action=ActionType.click,
            target="Continue",
            confidence=0.9,
        ),
        _assert(2, "inbox", "Payment received", visible="Unread messages", timestamp_ms=2000),
    )
    result = align(expected, observed)
    receipt = result.step_verdicts[1]
    assert receipt.passed is False
    assert receipt.matched_observed_index is None
    assert receipt.finding_types == [FindingType.missing_action]
    assert any(
        finding.type == FindingType.extra_action and finding.observed_action_index == 2
        for finding in result.findings
    )


def test_grounded_assertion_evidence_still_pairs():
    expected = _expected(ExpectedStep(
        index=1,
        action=ActionType.assert_visible,
        target="receipt",
        assertion="Payment received",
    ))
    observed = _observed(_assert(
        1, "receipt", "Payment received", visible="Payment received"
    ))
    verdict = align(expected, observed).step_verdicts[0]
    assert verdict.passed is True
    assert verdict.matched_observed_index == 1


def test_screen_description_alone_does_not_prove_a_visible_text_assertion():
    expected = _expected(ExpectedStep(
        index=1,
        action=ActionType.assert_visible,
        target="receipt",
        assertion="Payment received",
    ))
    observed = _observed(_assert(
        1,
        "settings",
        "Payment received",
        visible="Settings",
        description="Settings",
        screen="Payment received",
    ))
    verdict = align(expected, observed).step_verdicts[0]
    assert verdict.passed is False
    assert verdict.matched_observed_index is None


def test_navigation_url_still_supports_a_page_assertion():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Continue"),
        ExpectedStep(
            index=2,
            action=ActionType.assert_visible,
            target="account page",
            assertion="the account page is displayed",
        ),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=1000,
            action=ActionType.click,
            target="Continue",
            confidence=0.9,
        ),
        ObservedAction(
            index=2,
            timestamp_ms=2000,
            action=ActionType.navigate,
            target="example.com/account",
            confidence=0.9,
        ),
        _assert(
            3,
            "account page",
            "the account page is displayed",
            visible="account",
            timestamp_ms=3000,
        ),
    )
    verdict = align(expected, observed).step_verdicts[1]
    assert verdict.passed is True
    assert verdict.matched_observed_index == 3


def test_failed_finding_does_not_cite_an_observation_another_step_passed_on():
    verdicts = [
        StepVerdict(
            expected_step_index=1,
            passed=True,
            matched_observed_index=1,
            evidence_timestamp_ms=1000,
        ),
        StepVerdict(
            expected_step_index=2,
            passed=False,
            finding_types=[FindingType.wrong_target],
        ),
    ]
    findings = drop_shared_finding_citations(
        [
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=2,
                observed_action_index=1,
                evidence_timestamp_ms=1000,
                description="Step 2 was paired with observation 1.",
            )
        ],
        verdicts,
    )
    assert findings[0].type == FindingType.wrong_target
    assert findings[0].expected_step_index == 2
    assert findings[0].description == "Step 2 was paired with observation 1."
    assert findings[0].observed_action_index is None
    assert AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.5,
        findings=findings,
        step_verdicts=verdicts,
    ).findings[0].observed_action_index is None
