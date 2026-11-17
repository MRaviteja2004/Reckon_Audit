"""Unit tests for experimental Stage 3 evidence-aware matching."""

from pathlib import Path
from unittest.mock import MagicMock

from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.assist_evidence_propose import (
    _EvidenceBatchResponse,
    _EvidenceDecision,
    _EvidenceRoleAssignment,
)
from reckon.assist_evidence_select import select_evidence_bundle_candidates
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AssistCandidateKind,
    AssistProposalStatus,
    EvidenceRole,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedAction,
    ObservedTrace,
)


def _coarse_login_traces():
    expected = ExpectedTrace(
        scenario_title="Driver app login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="driver app",
                raw_text="Login into the driver app using the test account",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_state,
                target="driver app",
                assertion="User should able to log in succesfully",
                raw_text="User should able to log in succesfully",
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
                target="Uber Driver app icon",
                confidence=0.9,
                screen_description="Uber app launching",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=2000,
                action=ActionType.other,
                target="Loading spinner",
                confidence=0.8,
                screen_description="Loading spinner",
            ),
            ObservedAction(
                index=3,
                timestamp_ms=3000,
                action=ActionType.assert_visible,
                target="Driver home screen",
                confidence=0.85,
                assertion_evidence=AssertionEvidence(
                    description="Offline status",
                    visible_text="You're offline",
                ),
                screen_description="Map You're offline",
            ),
            ObservedAction(
                index=4,
                timestamp_ms=4000,
                action=ActionType.navigate,
                target="Uber driver app Studio Dev Settings",
                confidence=0.8,
                screen_description="Studio Dev Settings with Launch Deeplink",
            ),
        ],
    )
    return expected, observed


def test_retrieve_selects_coarse_login_candidates():
    expected, observed = _coarse_login_traces()
    audit = align(expected, observed)
    candidates = select_evidence_bundle_candidates(expected, observed, audit)
    assert len(candidates) == 2
    by_step = {c.expected_step_index: c for c in candidates}
    assert by_step[1].suspicious_pass is False
    # Phase 1: screen-token heuristics no longer false-pass E2, so the baseline
    # already carries assertion_not_observed rather than a suspicious pass.
    assert by_step[2].suspicious_pass is False
    assert FindingType.assertion_not_observed in by_step[2].finding_types
    assert any(e.observed_index == 1 for e in by_step[1].evidence_events)
    assert any(e.observed_index == 3 for e in by_step[2].evidence_events)


def test_evidence_aware_challenges_false_pass_without_clearing_initiation():
    expected, observed = _coarse_login_traces()
    client = MagicMock()

    def _create(**kwargs):
        response_model = kwargs.get("response_model")
        if response_model is _EvidenceBatchResponse:
            # Per-candidate decisions with labels limited to each window.
            return _EvidenceBatchResponse(
                decisions=[
                    _EvidenceDecision(
                        candidate_id="E1",
                        labels=[
                            _EvidenceRoleAssignment(
                                observed_event_id="O1",
                                role=EvidenceRole.initiation,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O2",
                                role=EvidenceRole.distractor,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O3",
                                role=EvidenceRole.completion,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O4",
                                role=EvidenceRole.distractor,
                            ),
                        ],
                        best_initiation_id="O1",
                        best_completion_id="O3",
                        best_assertion_support_id=None,
                        confidence=0.92,
                        rationale="initiation only; no assertion clear",
                    ),
                    _EvidenceDecision(
                        candidate_id="E2",
                        labels=[
                            _EvidenceRoleAssignment(
                                observed_event_id="O3",
                                role=EvidenceRole.completion,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O4",
                                role=EvidenceRole.distractor,
                            ),
                        ],
                        best_initiation_id=None,
                        best_completion_id="O3",
                        best_assertion_support_id=None,
                        confidence=0.92,
                        rationale="studio settings distractor; offline text not login claim",
                    ),
                ]
            )
        from reckon.assist_propose import _AssistBatchResponse, _AssistDecision

        return _AssistBatchResponse(
            decisions=[
                _AssistDecision(
                    candidate_id="C1",
                    answer="different",
                    confidence=0.9,
                    rationale="not same",
                )
            ]
        )

    client.chat.completions.create.side_effect = _create
    assisted = align_with_assist(
        expected, observed, openai_client=client, evidence_aware=True
    )
    by = {v.expected_step_index: v for v in assisted.result.step_verdicts}
    assert by[1].passed is False
    assert FindingType.wrong_target in by[1].finding_types
    assert by[2].passed is False
    assert FindingType.assertion_not_observed in by[2].finding_types
    # Phase 1 removed the false pass, so challenge is no longer required; the
    # deterministic baseline already fails the login assertion.
    assert assisted.deterministic_result.step_verdicts[1].passed is False
    assert FindingType.assertion_not_observed in {
        ft for ft in assisted.deterministic_result.step_verdicts[1].finding_types
    }


def test_evidence_aware_off_does_not_challenge():
    expected, observed = _coarse_login_traces()
    client = MagicMock()
    from reckon.assist_propose import _AssistBatchResponse, _AssistDecision

    client.chat.completions.create.return_value = _AssistBatchResponse(
        decisions=[
            _AssistDecision(
                candidate_id="C1",
                answer="different",
                confidence=0.9,
                rationale="different",
            )
        ]
    )
    assisted = align_with_assist(
        expected, observed, openai_client=client, evidence_aware=False
    )
    by = {v.expected_step_index: v for v in assisted.result.step_verdicts}
    # Phase 1: deterministic path already fails the login assertion.
    assert by[2].passed is False
    assert FindingType.assertion_not_observed in by[2].finding_types
    assert not any(
        p.kind == AssistCandidateKind.evidence_bundle
        for p in assisted.assist.proposals
    )


def test_initiation_alone_never_clears_assertion():
    expected = ExpectedTrace(
        scenario_title="Login success required",
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
                raw_text="User is logged in and offline",
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
                target="Uber Driver app icon",
                confidence=0.9,
                screen_description="Splash",
            ),
            ObservedAction(
                index=2,
                timestamp_ms=2000,
                action=ActionType.other,
                target="Loading spinner",
                confidence=0.8,
                screen_description="Still loading",
            ),
        ],
    )
    labels = [
        _EvidenceRoleAssignment(observed_event_id="O1", role=EvidenceRole.initiation),
        _EvidenceRoleAssignment(observed_event_id="O2", role=EvidenceRole.progress),
    ]
    client = MagicMock()

    def _create(**kwargs):
        response_model = kwargs.get("response_model")
        if getattr(response_model, "__name__", "") == "_EvidenceBatchResponse":
            user = kwargs["messages"][1]["content"]
            ids = [
                line.split(":", 1)[1].strip()
                for line in user.splitlines()
                if line.startswith("candidate_id:")
            ]
            return _EvidenceBatchResponse(
                decisions=[
                    _EvidenceDecision(
                        candidate_id=cid,
                        labels=labels,
                        best_initiation_id="O1",
                        best_completion_id=None,
                        best_assertion_support_id=None,
                        confidence=0.9,
                        rationale="icon only",
                    )
                    for cid in ids
                ]
            )
        from reckon.assist_propose import _AssistBatchResponse, _AssistDecision

        return _AssistBatchResponse(decisions=[])

    client.chat.completions.create.side_effect = _create
    assisted = align_with_assist(
        expected, observed, openai_client=client, evidence_aware=True
    )
    by = {v.expected_step_index: v for v in assisted.result.step_verdicts}
    assert by[2].passed is False
    assert not any(
        p.status == AssistProposalStatus.accepted
        and p.kind == AssistCandidateKind.evidence_bundle
        for p in assisted.assist.proposals
    )


def test_rejected_evidence_not_in_reviewed_semantic_pairs():
    expected, observed = _coarse_login_traces()
    client = MagicMock()

    def _create(**kwargs):
        if getattr(kwargs.get("response_model"), "__name__", "") == "_EvidenceBatchResponse":
            user = kwargs["messages"][1]["content"]
            ids = [
                line.split(":", 1)[1].strip()
                for line in user.splitlines()
                if line.startswith("candidate_id:")
            ]
            return _EvidenceBatchResponse(
                decisions=[
                    _EvidenceDecision(
                        candidate_id=cid,
                        labels=[
                            _EvidenceRoleAssignment(
                                observed_event_id="O1",
                                role=EvidenceRole.initiation,
                            )
                        ],
                        best_initiation_id="O1",
                        confidence=0.2,
                        rationale="low confidence",
                    )
                    for cid in ids
                ]
            )
        from reckon.assist_propose import _AssistBatchResponse, _AssistDecision

        return _AssistBatchResponse(
            decisions=[
                _AssistDecision(
                    candidate_id="C1",
                    answer="different",
                    confidence=0.9,
                    rationale="different",
                )
            ]
        )

    client.chat.completions.create.side_effect = _create
    assisted = align_with_assist(
        expected, observed, openai_client=client, evidence_aware=True
    )
    assert assisted.assist.reviewed_semantic_pairs == []
    assert not any(
        p.status == AssistProposalStatus.accepted for p in assisted.assist.proposals
    )


def test_challenge_at_confidence_065_for_coarse_login_case():
    """Phase 1: baseline already fails; low-confidence evidence must not invent a pass."""
    expected, observed = _coarse_login_traces()
    baseline = align(expected, observed)
    assert baseline.step_verdicts[1].passed is False
    assert FindingType.assertion_not_observed in baseline.step_verdicts[1].finding_types
    client = MagicMock()

    def _create(**kwargs):
        if kwargs.get("response_model") is _EvidenceBatchResponse:
            return _EvidenceBatchResponse(
                decisions=[
                    _EvidenceDecision(
                        candidate_id="E1",
                        labels=[
                            _EvidenceRoleAssignment(
                                observed_event_id="O1",
                                role=EvidenceRole.initiation,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O2",
                                role=EvidenceRole.distractor,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O3",
                                role=EvidenceRole.completion,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O4",
                                role=EvidenceRole.distractor,
                            ),
                        ],
                        best_initiation_id="O1",
                        best_completion_id="O3",
                        best_assertion_support_id=None,
                        confidence=0.85,
                        rationale="initiation only",
                    ),
                    _EvidenceDecision(
                        candidate_id="E2",
                        labels=[
                            _EvidenceRoleAssignment(
                                observed_event_id="O3",
                                role=EvidenceRole.completion,
                            ),
                            _EvidenceRoleAssignment(
                                observed_event_id="O4",
                                role=EvidenceRole.distractor,
                            ),
                        ],
                        best_initiation_id=None,
                        best_completion_id="O3",
                        best_assertion_support_id=None,
                        confidence=0.65,
                        rationale=(
                            "O4 Studio Settings is a distractor; O3 shows offline "
                            "home but never says logged in"
                        ),
                    ),
                ]
            )
        from reckon.assist_propose import _AssistBatchResponse, _AssistDecision

        return _AssistBatchResponse(
            decisions=[
                _AssistDecision(
                    candidate_id="C1",
                    answer="different",
                    confidence=0.9,
                    rationale="not same",
                )
            ]
        )

    client.chat.completions.create.side_effect = _create
    assisted = align_with_assist(
        expected, observed, openai_client=client, evidence_aware=True
    )

    # Deterministic baseline preserved for comparison.
    assert assisted.deterministic_result.model_dump() == baseline.model_dump()
    by = {v.expected_step_index: v for v in assisted.result.step_verdicts}
    assert by[2].passed is False
    assert FindingType.assertion_not_observed in by[2].finding_types
    # Offline home is the DP match; Studio Settings remains an extra action.
    assert by[2].matched_observed_index == 3
    assert not any(
        p.challenged_false_pass and p.status == AssistProposalStatus.accepted
        for p in assisted.assist.proposals
    )
    # No accepted evidence pass/challenge → Stage 4 skip list stays empty.
    assert assisted.assist.reviewed_semantic_pairs == []


def test_distractor_unsupported_assertion_triggers_challenge():
    from dataclasses import replace

    from reckon.assist_evidence_validate import validate_evidence_decision

    expected, observed = _coarse_login_traces()
    audit = align(expected, observed)
    candidate = next(
        c
        for c in select_evidence_bundle_candidates(expected, observed, audit)
        if c.expected_step_index == 2
    )
    # Challenge path still requires a suspicious baseline pass whose DP match is
    # labeled distractor. Synthesize that state — Phase 1 no longer produces it
    # for this fixture, but the validator must keep challenging when it appears.
    candidate = replace(
        candidate,
        matched_observed_index=4,
        suspicious_pass=True,
    )
    decision = _EvidenceDecision(
        candidate_id=candidate.candidate_id,
        labels=[
            _EvidenceRoleAssignment(
                observed_event_id="O3",
                role=EvidenceRole.completion,
            ),
            _EvidenceRoleAssignment(
                observed_event_id="O4",
                role=EvidenceRole.distractor,
            ),
        ],
        best_completion_id="O3",
        confidence=0.65,
        rationale="distractor match",
    )
    record = validate_evidence_decision(candidate, decision, expected, observed)
    assert record.status == AssistProposalStatus.accepted
    assert record.challenged_false_pass is True
    assert record.reason == "challenge_false_pass"


def test_low_confidence_support_cannot_create_a_pass():
    from reckon.assist_evidence_validate import validate_evidence_decision
    from reckon.assist_evidence_propose import EVIDENCE_CONFIDENCE_FLOOR

    expected, observed = _coarse_login_traces()
    # Use completion fixture-style assertion that O3 can satisfy? O3 has
    # "You're offline" which does NOT satisfy login claim — craft a candidate
    # where support would satisfy if confidence were high enough.
    expected = ExpectedTrace(
        scenario_title="offline",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_state,
                target="driver home",
                assertion="You're offline",
                raw_text="Assert You're offline",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=500,
                action=ActionType.assert_visible,
                target="driver home",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Loading",
                    visible_text="Please wait",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=12000,
                action=ActionType.click,
                target="home chrome",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Offline",
                    visible_text="You're offline",
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    candidate = select_evidence_bundle_candidates(expected, observed, audit)[0]
    decision = _EvidenceDecision(
        candidate_id=candidate.candidate_id,
        labels=[
            _EvidenceRoleAssignment(
                observed_event_id=f"O{candidate.matched_observed_index}",
                role=EvidenceRole.progress,
            ),
            _EvidenceRoleAssignment(
                observed_event_id="O2",
                role=EvidenceRole.assertion_evidence,
            ),
        ],
        best_assertion_support_id="O2",
        confidence=EVIDENCE_CONFIDENCE_FLOOR - 0.05,
        rationale="support present but low confidence",
    )
    record = validate_evidence_decision(candidate, decision, expected, observed)
    assert record.status == AssistProposalStatus.rejected
    assert record.reason == "low_confidence"
    assert record.challenged_false_pass is False


def test_distractor_without_suspicious_pass_does_not_force_failure():
    """Labeling a failing matched obs as distractor must not invent a challenge."""
    from reckon.assist_evidence_validate import validate_evidence_decision

    expected, observed = _coarse_login_traces()
    audit = align(expected, observed)
    candidate = next(
        c
        for c in select_evidence_bundle_candidates(expected, observed, audit)
        if c.expected_step_index == 1
    )
    assert candidate.suspicious_pass is False
    decision = _EvidenceDecision(
        candidate_id=candidate.candidate_id,
        labels=[
            _EvidenceRoleAssignment(
                observed_event_id="O1",
                role=EvidenceRole.initiation,
            ),
            _EvidenceRoleAssignment(
                observed_event_id="O2",
                role=EvidenceRole.distractor,
            ),
        ],
        best_initiation_id="O1",
        confidence=0.95,
        rationale="spinner distractor on already-failing step",
    )
    record = validate_evidence_decision(candidate, decision, expected, observed)
    assert record.status == AssistProposalStatus.rejected
    assert record.challenged_false_pass is False


def test_unresolved_evidence_does_not_enter_reviewed_semantic_pairs():
    expected, observed = _coarse_login_traces()
    client = MagicMock()

    def _create(**kwargs):
        if kwargs.get("response_model") is _EvidenceBatchResponse:
            return _EvidenceBatchResponse(
                decisions=[
                    _EvidenceDecision(
                        candidate_id="E1",
                        labels=[],
                        confidence=0.0,
                        rationale="unresolved",
                    ),
                    _EvidenceDecision(
                        candidate_id="E2",
                        labels=[],
                        confidence=0.0,
                        rationale="unresolved",
                    ),
                ]
            )
        from reckon.assist_propose import _AssistBatchResponse, _AssistDecision

        return _AssistBatchResponse(
            decisions=[
                _AssistDecision(
                    candidate_id="C1",
                    answer="unresolved",
                    confidence=0.0,
                    rationale="unresolved",
                )
            ]
        )

    client.chat.completions.create.side_effect = _create
    assisted = align_with_assist(
        expected, observed, openai_client=client, evidence_aware=True
    )
    assert assisted.assist.reviewed_semantic_pairs == []
    # Unresolved evidence must not invent a pass over a failing baseline.
    assert assisted.result.step_verdicts[1].passed is False
    assert FindingType.assertion_not_observed in (
        assisted.result.step_verdicts[1].finding_types
    )


def test_aligner_module_present():
    aligner = Path(__file__).resolve().parents[1] / "src" / "reckon" / "aligner.py"
    assert aligner.is_file()
