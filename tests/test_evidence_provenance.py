"""Frame text must support a pass, and that pass keeps one evidence record."""

from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditVerdict,
    CandidateEvidenceType,
    CandidateSearchStatus,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedAction,
    ObservedTrace,
)
from reckon.stage2c.manifest import build_frame_manifest
from reckon.stage2c.propose import RawFrameEvidenceBatch, RawFrameEvidenceCandidate
from tests.test_stage2c_integration import _quiet_client


def test_frame_description_cannot_pass_without_the_assertion_text():
    expected = ExpectedTrace(
        scenario_title="Confirm",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_state,
                target="confirm",
                assertion="Confirm is disabled",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="offer.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.assert_visible,
                target="other screen",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Loading", visible_text="Loading",
                ),
            ),
        ],
    )
    manifest = build_frame_manifest([1000, 4000])
    raw = RawFrameEvidenceBatch(candidates=[
        RawFrameEvidenceCandidate(
            expected_step_id="E1",
            frame_ids=["F2"],
            evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
            observed_description="The Confirm button looks greyed out and disabled.",
            visible_text=["Confirm"],
            confidence=0.95,
            search_status=CandidateSearchStatus.candidate_found,
        ),
    ])
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=_quiet_client(),
        frame_evidence=True,
        frame_manifest=manifest,
        stage2c_raw_batch=raw,
    )
    verdict = assisted.result.step_verdicts[0]
    assert verdict.passed is False
    assert FindingType.assertion_not_observed in verdict.finding_types
    assert assisted.result.verdict == AuditVerdict.unfaithful
    proposal = next(
        item for item in assisted.assist.proposals
        if item.kind.value == "frame_evidence"
    )
    assert proposal.status.value == "rejected"
    assert proposal.reason == "visible_text_does_not_support_assertion"


def test_supported_frame_evidence_keeps_one_record():
    expected = ExpectedTrace(
        scenario_title="Offer",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_text,
                target="heading",
                assertion="Choose your guarantee",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="offer.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.assert_text,
                target="home map",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Map", visible_text="Home map",
                ),
            ),
        ],
    )
    manifest = build_frame_manifest([1000, 8000])
    raw = RawFrameEvidenceBatch(candidates=[
        RawFrameEvidenceCandidate(
            expected_step_id="E1",
            frame_ids=["F2"],
            evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
            observed_description="model prose is not the evidence",
            visible_text=["Choose your guarantee"],
            confidence=0.95,
            search_status=CandidateSearchStatus.candidate_found,
        ),
    ])
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=_quiet_client(),
        frame_evidence=True,
        frame_manifest=manifest,
        stage2c_raw_batch=raw,
    )
    verdict = assisted.result.step_verdicts[0]
    assert verdict.passed is True
    assert verdict.matched_observed_index is None
    assert verdict.evidence_id == "F2"
    assert verdict.frame_ids == ["F2"]
    assert verdict.evidence_timestamp_ms == 8000
    assert verdict.recovery_reason == "frame_evidence"
    assert verdict.explanation == "Frames F2 at 8000 ms show the asserted text."


def test_later_clean_observation_replaces_the_earlier_mismatch():
    expected = ExpectedTrace(
        scenario_title="Banner",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_text,
                target="banner",
                assertion="Invalid username",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="banner.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=100,
                action=ActionType.assert_text,
                target="banner",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Please wait", visible_text="Please wait",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=800,
                action=ActionType.click,
                target="dismiss",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Invalid username", visible_text="Invalid username",
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    verdict = audit.step_verdicts[0]
    assert verdict.passed is True
    assert verdict.matched_observed_index == 2
    assert verdict.evidence_timestamp_ms == 800
    assert verdict.evidence_id == "O2"
    assert verdict.recovery_reason == "later_observed_action"
    assert verdict.explanation == "Observed action O2 at 800 ms supports this step."
    assert any(
        finding.type == FindingType.extra_action and finding.observed_action_index == 1
        for finding in audit.findings
    )


def test_two_later_supporters_do_not_become_a_pass():
    expected = ExpectedTrace(
        scenario_title="Banner",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_text,
                target="banner",
                assertion="Invalid username",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="banner.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=100,
                action=ActionType.assert_text,
                target="banner",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Please wait", visible_text="Please wait",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=800,
                action=ActionType.click,
                target="dismiss",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Invalid username", visible_text="Invalid username",
                ),
            ),
            ObservedAction(
                index=3,
                timestamp_ms=900,
                action=ActionType.click,
                target="ok",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Invalid username", visible_text="Invalid username",
                ),
            ),
        ],
    )
    audit = align(expected, observed)
    assert audit.step_verdicts[0].passed is False
    assert audit.step_verdicts[0].matched_observed_index == 1
    assert audit.verdict == AuditVerdict.unfaithful
