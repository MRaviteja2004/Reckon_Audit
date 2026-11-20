import pytest

from reckon.assist_apply import apply_assist_proposals
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
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


def _expected() -> ExpectedTrace:
    return ExpectedTrace(
        scenario_title="Sign in",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Sign in button"),
            ExpectedStep(index=2, action=ActionType.type_text, target="username field"),
            ExpectedStep(
                index=3,
                action=ActionType.assert_visible,
                target="Dashboard",
                assertion="Dashboard",
            ),
        ],
    )


def _observed() -> ObservedTrace:
    def action(index: int, kind: ActionType, target: str, **extra: object) -> ObservedAction:
        return ObservedAction(
            index=index,
            timestamp_ms=index * 1000,
            action=kind,
            target=target,
            confidence=0.9,
            **extra,
        )

    return ObservedTrace(
        source_file="x.mp4",
        actions=[
            action(1, ActionType.click, "Sign in"),
            action(2, ActionType.type_text, "Username or email address", input_value="ann"),
            action(3, ActionType.type_text, "Username or email address", input_value="ann@"),
            action(4, ActionType.hover, "Stop recording"),
            action(
                5,
                ActionType.assert_visible,
                "Dashboard",
                assertion_evidence=AssertionEvidence(
                    description="Heading", visible_text="Dashboard"
                ),
            ),
        ],
    )


def _extra(index: int) -> Finding:
    return Finding(
        type=FindingType.extra_action,
        severity=FindingSeverity.advisory,
        observed_action_index=index,
        evidence_timestamp_ms=index * 1000,
        description=f"Observed action {index} was not in the expected scenario.",
    )


def _deterministic() -> AuditResult:
    return AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=2 / 3,
        findings=[
            _extra(2),
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=2,
                observed_action_index=3,
                evidence_timestamp_ms=3000,
                description="Step 2: wrong target.",
            ),
            _extra(4),
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
                matched_observed_index=3,
                evidence_timestamp_ms=3000,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=True,
                matched_observed_index=5,
                evidence_timestamp_ms=5000,
            ),
        ],
    )


def _remap(
    terminal: int,
    *,
    initiation: int | None = None,
    completion: int | None = None,
) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=AssistCandidateKind.episode_verification,
        status=AssistProposalStatus.accepted,
        expected_step_index=2,
        matched_observed_index=terminal,
        supporting_observed_index=terminal,
        initiation_observed_index=initiation,
        completion_observed_index=completion if completion is not None else terminal,
        finding_types=[FindingType.wrong_target],
        matching_method=AssistMatchMethod.deterministic,
    )


def _refined(record: AssistProposalRecord) -> AuditResult:
    return apply_assist_proposals(_deterministic(), [record], _expected(), _observed())


def _extras(audit: AuditResult) -> list[int | None]:
    return sorted(
        finding.observed_action_index
        for finding in audit.findings
        if finding.type == FindingType.extra_action
    )


def test_apply_drops_extra_for_the_observation_a_remapped_step_now_cites():
    refined = _refined(_remap(2))
    assert refined.step_verdicts[1].passed is True
    assert 2 not in _extras(refined)


def test_apply_keeps_a_genuine_extra_action():
    assert 4 in _extras(_refined(_remap(2)))


def test_apply_reports_the_released_pair_outside_any_span_as_extra():
    assert _extras(_refined(_remap(2))) == [3, 4]


def test_apply_does_not_report_a_released_pair_inside_the_accepted_span():
    assert _extras(_refined(_remap(2, initiation=2, completion=3))) == [4]


def test_apply_ignores_a_remap_onto_another_passed_steps_observation():
    refined = _refined(_remap(5))
    assert refined.step_verdicts[1].passed is False
    assert refined.step_verdicts[2].matched_observed_index == 5


@pytest.mark.parametrize(
    "record",
    [_remap(2), _remap(2, initiation=2, completion=3), _remap(5)],
    ids=["remap", "remap_with_span", "conflicting_remap"],
)
def test_apply_never_lists_a_cited_passed_observation_as_extra(
    record: AssistProposalRecord,
):
    refined = _refined(record)
    cited = {
        verdict.matched_observed_index
        for verdict in refined.step_verdicts
        if verdict.passed
    }
    assert cited.isdisjoint(_extras(refined))
