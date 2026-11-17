"""Golden evaluation for Stage 3 evidence-aware cases under eval/aligner_evidence/."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval.schema import EvidenceCaseTruth
from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.assist_evidence_propose import (
    _EvidenceBatchResponse,
    _EvidenceDecision,
    _EvidenceRoleAssignment,
)
from reckon.assist_evidence_select import select_evidence_bundle_candidates
from reckon.assist_propose import _AssistBatchResponse
from reckon.schema import EvidenceRole, ExpectedTrace, ObservedTrace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval" / "aligner_evidence"
CASE_DIRS = sorted(path for path in EVAL_DIR.iterdir() if path.is_dir())

_ROLE_MAP = {
    "initiation": EvidenceRole.initiation,
    "progress": EvidenceRole.progress,
    "completion": EvidenceRole.completion,
    "assertion_evidence": EvidenceRole.assertion_evidence,
    "distractor": EvidenceRole.distractor,
}


def _load(case_dir: Path):
    expected = ExpectedTrace.model_validate_json(
        (case_dir / "expected.json").read_text(encoding="utf-8")
    )
    observed = ObservedTrace.model_validate_json(
        (case_dir / "observed.json").read_text(encoding="utf-8")
    )
    truth = EvidenceCaseTruth.model_validate_json(
        (case_dir / "truth.json").read_text(encoding="utf-8")
    )
    return expected, observed, truth


def _assert_steps(audit, steps) -> None:
    by_index = {v.expected_step_index: v for v in audit.step_verdicts}
    for step in steps:
        actual = by_index[step.index]
        assert actual.passed == step.should_pass
        assert sorted(ft.value for ft in actual.finding_types) == sorted(
            ft.value for ft in step.expected_finding_types
        )


def _assert_case_findings(audit, expected_findings) -> None:
    got = sorted(f.type.value for f in audit.findings if f.expected_step_index is None)
    want = sorted(f.value for f in expected_findings)
    assert got == want


def _id_if_allowed(raw: str | None, allowed: set[int]) -> str | None:
    if raw is None:
        return None
    text = raw.strip().upper()
    if not text.startswith("O") or not text[1:].isdigit():
        return None
    index = int(text[1:])
    return raw if index in allowed else None


def _scripted_evidence_response(candidates, truth: EvidenceCaseTruth) -> _EvidenceBatchResponse:
    labels = [
        _EvidenceRoleAssignment(
            observed_event_id=f"O{item.observed_index}",
            role=_ROLE_MAP[item.role],
        )
        for item in truth.scripted_labels
    ]
    decisions = []
    for candidate in candidates:
        allowed = {event.observed_index for event in candidate.evidence_events}
        filtered = [
            label
            for label in labels
            if int(label.observed_event_id[1:]) in allowed
        ]
        decisions.append(
            _EvidenceDecision(
                candidate_id=candidate.candidate_id,
                labels=filtered,
                best_initiation_id=_id_if_allowed(
                    truth.scripted_best_initiation_id, allowed
                ),
                best_completion_id=_id_if_allowed(
                    truth.scripted_best_completion_id, allowed
                ),
                best_assertion_support_id=_id_if_allowed(
                    truth.scripted_best_assertion_support_id, allowed
                ),
                confidence=truth.scripted_confidence,
                rationale="scripted evidence eval",
            )
        )
    return _EvidenceBatchResponse(decisions=decisions)


def _mock_client(candidates, truth: EvidenceCaseTruth) -> MagicMock:
    client = MagicMock()

    def _create(**kwargs):
        response_model = kwargs.get("response_model")
        if response_model is _EvidenceBatchResponse:
            if not truth.scripted_labels and truth.scripted_confidence < 0.7:
                return _EvidenceBatchResponse(
                    decisions=[
                        _EvidenceDecision(
                            candidate_id=c.candidate_id,
                            labels=[],
                            confidence=truth.scripted_confidence,
                            rationale="stub unresolved",
                        )
                        for c in candidates
                    ]
                )
            return _scripted_evidence_response(candidates, truth)
        return _AssistBatchResponse(decisions=[])

    client.chat.completions.create.side_effect = _create
    return client


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
def test_evidence_golden_baseline(case_dir: Path):
    expected, observed, truth = _load(case_dir)
    audit = align(expected, observed)
    _assert_steps(audit, truth.baseline_steps)
    _assert_case_findings(audit, truth.baseline_case_findings)
    candidates = select_evidence_bundle_candidates(expected, observed, audit)
    assert bool(candidates) == truth.candidate_expected


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
def test_evidence_golden_scripted(case_dir: Path):
    expected, observed, truth = _load(case_dir)
    baseline = align(expected, observed)
    candidates = select_evidence_bundle_candidates(expected, observed, baseline)
    client = _mock_client(candidates, truth)
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=client,
        evidence_aware=True,
    )
    _assert_steps(assisted.result, truth.evidence_steps)
    _assert_case_findings(assisted.result, truth.evidence_case_findings)


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
def test_evidence_golden_stub_no_false_pass(case_dir: Path):
    expected, observed, truth = _load(case_dir)
    baseline = align(expected, observed)
    candidates = select_evidence_bundle_candidates(expected, observed, baseline)
    stub_truth = truth.model_copy(
        update={"scripted_labels": [], "scripted_confidence": 0.0}
    )
    client = _mock_client(candidates, stub_truth)
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=client,
        evidence_aware=True,
    )
    if truth.llm_required:
        # Unresolved/empty evidence LLM must not invent an evidence_bundle accept.
        # Deterministic episode verification may still clear findings — that is
        # intentional Phase 3 behavior, not an LLM false pass.
        assert not any(
            proposal.kind.value == "evidence_bundle"
            and proposal.status.value == "accepted"
            for proposal in assisted.assist.proposals
        )
    else:
        _assert_steps(assisted.result, truth.evidence_steps)
