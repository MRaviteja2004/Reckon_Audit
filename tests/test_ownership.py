from reckon.ownership import (
    OWNED_BY_ANOTHER_STEP,
    as_owned_conflict,
    reconcile_ownership,
    split_conflicting_claims,
)
from reckon.schema import (
    ActionType,
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


def _observed(count: int) -> ObservedTrace:
    return ObservedTrace(
        source_file="x.mp4",
        actions=[
            ObservedAction(
                index=index,
                timestamp_ms=index * 1000,
                action=ActionType.click,
                target=f"Control {index}",
                confidence=0.9,
            )
            for index in range(1, count + 1)
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


def _verdict(step: int, passed: bool, matched: int | None) -> StepVerdict:
    return StepVerdict(
        expected_step_index=step,
        passed=passed,
        matched_observed_index=matched,
        finding_types=[] if passed else [FindingType.wrong_target],
    )


def _audit(verdicts: list[StepVerdict], findings: list[Finding]) -> AuditResult:
    return AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=0.5,
        findings=findings,
        step_verdicts=verdicts,
    )


def _extras(audit: AuditResult) -> list[int | None]:
    return sorted(
        finding.observed_action_index
        for finding in audit.findings
        if finding.type == FindingType.extra_action
    )


def test_reconcile_ownership_drops_extra_for_observation_a_passed_step_cites():
    audit = _audit([_verdict(1, True, 2)], [_extra(2), _extra(3)])
    assert _extras(reconcile_ownership(audit, _observed(3))) == [3]


def test_reconcile_ownership_drops_extra_inside_an_accepted_span():
    audit = _audit([_verdict(1, True, 2)], [_extra(3)])
    assert _extras(reconcile_ownership(audit, _observed(3), covered={3})) == []


def test_reconcile_ownership_reports_a_released_unaccounted_observation():
    audit = _audit([_verdict(1, True, 2)], [])
    assert _extras(reconcile_ownership(audit, _observed(3), released={3})) == [3]


def test_reconcile_ownership_does_not_duplicate_an_already_reported_extra():
    audit = _audit([_verdict(1, True, 2)], [_extra(3)])
    assert _extras(reconcile_ownership(audit, _observed(3), released={3})) == [3]


def test_reconcile_ownership_does_not_report_a_released_observation_a_failed_step_cites():
    audit = _audit([_verdict(1, True, 2), _verdict(2, False, 3)], [])
    assert _extras(reconcile_ownership(audit, _observed(3), released={3})) == []


def test_reconcile_ownership_clears_a_failed_citation_of_an_owned_observation():
    audit = _audit([_verdict(1, True, 2), _verdict(2, False, 2)], [])
    reconciled = reconcile_ownership(audit, _observed(2))
    assert reconciled.step_verdicts[1].matched_observed_index is None


def test_reconcile_ownership_keeps_verdict_and_score():
    audit = _audit([_verdict(1, True, 2)], [_extra(2)])
    reconciled = reconcile_ownership(audit, _observed(3), released={3})
    assert (reconciled.verdict, reconciled.fidelity_score) == (
        audit.verdict,
        audit.fidelity_score,
    )


def _remap(step: int, terminal: int) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=AssistCandidateKind.episode_verification,
        status=AssistProposalStatus.accepted,
        expected_step_index=step,
        matched_observed_index=terminal,
        supporting_observed_index=terminal,
        completion_observed_index=terminal,
        matching_method=AssistMatchMethod.deterministic,
    )


def _expected(count: int) -> ExpectedTrace:
    return ExpectedTrace(
        scenario_title="x",
        steps=[
            ExpectedStep(index=index, action=ActionType.click, target=f"Control {index}")
            for index in range(1, count + 1)
        ],
    )


def test_split_conflicting_claims_rejects_a_claim_on_a_passed_steps_observation():
    audit = _audit([_verdict(1, True, 2), _verdict(2, False, 3)], [])
    kept, conflicting = split_conflicting_claims(audit, [_remap(2, 2)], _expected(2))
    assert (kept, [record.expected_step_index for record in conflicting]) == ([], [2])


def test_split_conflicting_claims_rejects_the_second_claim_on_one_observation():
    audit = _audit(
        [_verdict(1, False, None), _verdict(2, False, None)],
        [],
    )
    kept, conflicting = split_conflicting_claims(
        audit, [_remap(1, 4), _remap(2, 4)], _expected(2)
    )
    assert [record.expected_step_index for record in kept] == [1]
    assert [record.expected_step_index for record in conflicting] == [2]


def test_split_conflicting_claims_keeps_a_claim_on_a_free_observation():
    audit = _audit([_verdict(1, True, 2), _verdict(2, False, 3)], [])
    kept, conflicting = split_conflicting_claims(audit, [_remap(2, 4)], _expected(2))
    assert (len(kept), conflicting) == (1, [])


def test_as_owned_conflict_marks_the_proposal_rejected():
    record = as_owned_conflict(_remap(2, 2))
    assert (record.status, record.reason, record.matching_method) == (
        AssistProposalStatus.rejected,
        OWNED_BY_ANOTHER_STEP,
        AssistMatchMethod.rejected_proposal,
    )
