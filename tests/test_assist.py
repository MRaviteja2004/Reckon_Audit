"""Unit and mocked integration tests for Stage 3 align_with_assist (first slice)."""

from unittest.mock import MagicMock

import openai
import pytest

from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.assist_apply import apply_assist_proposals
from reckon.assist_propose import _AssistBatchResponse, _AssistDecision
from reckon.assist_select import select_assist_candidates
from reckon.assist_validate import validate_assist_decision
from reckon.crosscheck import select_semantic_candidates
from reckon.ownership import OWNED_BY_ANOTHER_STEP
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


def _expected(*steps: ExpectedStep) -> ExpectedTrace:
    return ExpectedTrace(scenario_title="t", steps=list(steps))


def _observed(*actions: ObservedAction) -> ObservedTrace:
    return ObservedTrace(source_file="rec.mp4", actions=list(actions))


def test_align_unchanged_on_login_faithful_fixture():
    from pathlib import Path

    case = Path(__file__).resolve().parents[1] / "eval" / "aligner" / "login_faithful"
    expected = ExpectedTrace.model_validate_json((case / "expected.json").read_text())
    observed = ObservedTrace.model_validate_json((case / "observed.json").read_text())

    first = align(expected, observed)
    second = align(expected, observed)

    assert first.model_dump() == second.model_dump()


def test_select_skips_when_no_eligible_findings():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Submit"),
    )
    observed = _observed(
        ObservedAction(
            index=1, timestamp_ms=10, action=ActionType.click, target="Submit", confidence=1.0
        ),
    )
    audit = align(expected, observed)

    assert select_assist_candidates(expected, observed, audit) == []


def test_select_wrong_target_candidate():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save draft button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.click,
            target="Keep editing CTA",
            confidence=1.0,
        ),
    )
    audit = align(expected, observed)
    assert any(f.type == FindingType.wrong_target for f in audit.findings)

    candidates = select_assist_candidates(expected, observed, audit)

    assert len(candidates) == 1
    assert candidates[0].kind == AssistCandidateKind.wrong_target


def test_select_delayed_assertion_with_deterministic_support():
    expected = _expected(
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            target="banner",
            assertion="Invalid username",
        ),
    )
    # Obs2 is a click so DP keeps the expected assertion paired with obs1.
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.assert_text,
            target="banner",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Please wait", visible_text="Please wait"
            ),
        ),
        ObservedAction(
            index=2,
            timestamp_ms=500,
            action=ActionType.click,
            target="dismiss",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Invalid username", visible_text="Invalid username"
            ),
        ),
    )
    audit = align(expected, observed)
    assert audit.step_verdicts[0].passed is True
    assert audit.step_verdicts[0].matched_observed_index == 2
    assert audit.step_verdicts[0].evidence_timestamp_ms == 500
    assert audit.step_verdicts[0].evidence_id == "O2"

    candidates = select_assist_candidates(expected, observed, audit)
    assert candidates == []


def test_select_skips_wrong_input_value_pairs():
    expected = _expected(
        ExpectedStep(
            index=1,
            action=ActionType.type_text,
            target="amount",
            input_value="89.00",
        ),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.type_text,
            target="amount field",
            input_value="8.90",
            confidence=1.0,
        ),
    )
    audit = align(expected, observed)
    assert any(f.type == FindingType.wrong_input_value for f in audit.findings)

    assert select_assist_candidates(expected, observed, audit) == []


def test_validate_rejects_hallucinated_supporting_event():
    expected = _expected(
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            assertion="credentials were rejected",
            target="banner",
        ),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.assert_text,
            target="banner",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Loading", visible_text="Loading"
            ),
        ),
        ObservedAction(
            index=2,
            timestamp_ms=200,
            action=ActionType.click,
            target="ok",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Invalid username or password",
                visible_text="Invalid username or password",
            ),
        ),
    )
    audit = align(expected, observed)
    candidate = select_assist_candidates(expected, observed, audit)[0]
    assert candidate.deterministic_support_index is None
    decision = _AssistDecision(
        candidate_id=candidate.candidate_id,
        answer="equivalent",
        confidence=0.95,
        rationale="later banner",
        supporting_event_id="O99",
    )

    record = validate_assist_decision(candidate, decision, expected, observed)

    assert record.status == AssistProposalStatus.rejected
    assert record.reason == "supporting_event_not_offered"


def test_select_excludes_negative_delayed_evidence():
    expected = _expected(
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            assertion="Login successful",
            target="status",
        ),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.assert_text,
            target="status",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Checking", visible_text="Checking"
            ),
        ),
        ObservedAction(
            index=2,
            timestamp_ms=300,
            action=ActionType.click,
            target="ok",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Login not successful",
                visible_text="Login not successful",
            ),
        ),
    )
    audit = align(expected, observed)
    assert any(f.type == FindingType.assertion_not_observed for f in audit.findings)

    # Negated later evidence is not offered as a delayed-assertion candidate.
    assert select_assist_candidates(expected, observed, audit) == []


def test_apply_clears_wrong_target_only():
    deterministic = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=10,
                description="wrong target",
            )
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=10,
                finding_types=[FindingType.wrong_target],
            )
        ],
    )
    expected = _expected(ExpectedStep(index=1, action=ActionType.click, target="A"))
    observed = _observed(
        ObservedAction(
            index=1, timestamp_ms=10, action=ActionType.click, target="B", confidence=1.0
        )
    )
    accepted = [
        AssistProposalRecord(
            kind=AssistCandidateKind.wrong_target,
            status=AssistProposalStatus.accepted,
            expected_step_index=1,
            matched_observed_index=1,
            finding_types=[FindingType.wrong_target],
            matching_method=AssistMatchMethod.llm_assisted,
        )
    ]

    refined = apply_assist_proposals(deterministic, accepted, expected, observed)

    assert refined.findings == []
    assert refined.step_verdicts[0].passed is True
    assert deterministic.findings  # baseline untouched


def test_align_with_assist_llm_failure_keeps_deterministic():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save draft button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.click,
            target="Keep editing CTA",
            confidence=1.0,
        ),
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())

    assisted = align_with_assist(expected, observed, openai_client=client)
    baseline = align(expected, observed)

    assert assisted.result.model_dump() == baseline.model_dump()
    assert assisted.deterministic_result.model_dump() == baseline.model_dump()
    assert assisted.assist.error is not None
    assert assisted.assist.metrics.llm_request_count == 1


def test_align_with_assist_accepts_wrong_target_via_mock():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save draft button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.click,
            target="Keep editing CTA",
            confidence=1.0,
        ),
    )
    baseline = align(expected, observed)
    assert any(f.type == FindingType.wrong_target for f in baseline.findings)

    candidate = select_assist_candidates(expected, observed, baseline)[0]
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(
        decisions=[
            _AssistDecision(
                candidate_id=candidate.candidate_id,
                answer="equivalent",
                confidence=0.92,
                rationale="same control",
            )
        ]
    )

    assisted = align_with_assist(expected, observed, openai_client=client)

    assert assisted.deterministic_result.model_dump() == baseline.model_dump()
    assert not any(f.type == FindingType.wrong_target for f in assisted.result.findings)
    assert assisted.result.step_verdicts[0].passed is True
    assert assisted.assist.metrics.proposals_accepted == 1
    assert assisted.assist.metrics.llm_request_count == 1


def test_align_with_assist_deterministic_delayed_without_llm():
    expected = _expected(
        ExpectedStep(
            index=1,
            action=ActionType.assert_text,
            target="banner",
            assertion="Invalid username",
        ),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.assert_text,
            target="banner",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Please wait", visible_text="Please wait"
            ),
        ),
        ObservedAction(
            index=2,
            timestamp_ms=500,
            action=ActionType.click,
            target="dismiss",
            confidence=1.0,
            assertion_evidence=AssertionEvidence(
                description="Invalid username", visible_text="Invalid username"
            ),
        ),
    )
    client = MagicMock()
    assisted = align_with_assist(expected, observed, openai_client=client)

    client.chat.completions.create.assert_not_called()
    assert assisted.assist.metrics.llm_request_count == 0
    assert assisted.assist.metrics.proposals_accepted == 0
    assert assisted.result.step_verdicts[0].passed is True
    assert assisted.result.step_verdicts[0].evidence_timestamp_ms == 500
    assert assisted.result.step_verdicts[0].evidence_id == "O2"
    assert assisted.deterministic_result.step_verdicts[0].evidence_id == "O2"


def test_stage4_skip_pairs_avoids_duplicate_semantic_review():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save draft button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.click,
            target="Keep editing CTA",
            confidence=1.0,
        ),
    )
    audit = align(expected, observed)
    without_skip = select_semantic_candidates(expected, observed, audit)
    assert without_skip

    with_skip = select_semantic_candidates(
        expected, observed, audit, skip_pairs={(1, 1)}
    )
    assert with_skip == []


def test_align_with_assist_no_candidates_makes_no_model_call():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Submit"),
    )
    observed = _observed(
        ObservedAction(
            index=1, timestamp_ms=10, action=ActionType.click, target="Submit", confidence=1.0
        ),
    )
    client = MagicMock()
    assisted = align_with_assist(expected, observed, openai_client=client)

    client.chat.completions.create.assert_not_called()
    assert assisted.result == assisted.deterministic_result
    assert assisted.assist.metrics.candidate_count == 0


def test_rejected_proposal_does_not_suppress_stage4_semantic_review():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save draft button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.click,
            target="Keep editing CTA",
            confidence=1.0,
        ),
    )
    baseline = align(expected, observed)
    candidate = select_assist_candidates(expected, observed, baseline)[0]
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(
        decisions=[
            _AssistDecision(
                candidate_id=candidate.candidate_id,
                answer="different",
                confidence=0.99,
                rationale="distinct controls",
            )
        ]
    )

    assisted = align_with_assist(expected, observed, openai_client=client)

    assert assisted.assist.reviewed_semantic_pairs == []
    assert any(p.status == AssistProposalStatus.rejected for p in assisted.assist.proposals)
    # Stage 4 must still be able to select this unresolved wrong_target pair.
    remaining = select_semantic_candidates(
        expected,
        observed,
        assisted.result,
        skip_pairs={
            (p.expected_step_index, p.observed_action_index)
            for p in assisted.assist.reviewed_semantic_pairs
        },
    )
    assert remaining


def test_failed_proposal_does_not_suppress_stage4_semantic_review():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save draft button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.click,
            target="Keep editing CTA",
            confidence=1.0,
        ),
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock()
    )

    assisted = align_with_assist(expected, observed, openai_client=client)

    assert assisted.assist.reviewed_semantic_pairs == []
    assert any(p.status == AssistProposalStatus.failed for p in assisted.assist.proposals)
    remaining = select_semantic_candidates(
        expected,
        observed,
        assisted.result,
        skip_pairs={
            (p.expected_step_index, p.observed_action_index)
            for p in assisted.assist.reviewed_semantic_pairs
        },
    )
    assert remaining


def test_accepted_proposal_is_listed_for_stage4_skip():
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save draft button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=10,
            action=ActionType.click,
            target="Keep editing CTA",
            confidence=1.0,
        ),
    )
    baseline = align(expected, observed)
    candidate = select_assist_candidates(expected, observed, baseline)[0]
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(
        decisions=[
            _AssistDecision(
                candidate_id=candidate.candidate_id,
                answer="equivalent",
                confidence=0.92,
                rationale="same control",
            )
        ]
    )

    assisted = align_with_assist(expected, observed, openai_client=client)

    assert len(assisted.assist.reviewed_semantic_pairs) == 1
    pair = assisted.assist.reviewed_semantic_pairs[0]
    assert (pair.expected_step_index, pair.observed_action_index) == (1, 1)
    remaining = select_semantic_candidates(
        expected,
        observed,
        assisted.result,
        skip_pairs={(pair.expected_step_index, pair.observed_action_index)},
    )
    assert remaining == []


def test_align_with_assist_reports_a_claim_on_an_owned_observation_as_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = _expected(
        ExpectedStep(index=1, action=ActionType.click, target="Save button"),
        ExpectedStep(index=2, action=ActionType.click, target="Refund button"),
    )
    observed = _observed(
        ObservedAction(
            index=1,
            timestamp_ms=1000,
            action=ActionType.click,
            target="Save button",
            confidence=0.9,
        ),
    )
    claim = AssistProposalRecord(
        kind=AssistCandidateKind.episode_verification,
        status=AssistProposalStatus.accepted,
        expected_step_index=2,
        matched_observed_index=1,
        supporting_observed_index=1,
        completion_observed_index=1,
        matching_method=AssistMatchMethod.deterministic,
    )
    monkeypatch.setattr(
        "reckon.assist.verify_trace_against_episodes",
        lambda *args, **kwargs: ([claim], 0),
    )
    monkeypatch.setattr(
        "reckon.assist.select_assist_candidates", lambda *args, **kwargs: []
    )

    assisted = align_with_assist(expected, observed, openai_client=MagicMock())

    assert assisted.result.step_verdicts[1].passed is False
    assert [
        (record.status, record.reason) for record in assisted.assist.proposals
    ] == [(AssistProposalStatus.rejected, OWNED_BY_ANOTHER_STEP)]
