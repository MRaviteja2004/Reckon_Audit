"""Golden evaluation for Stage 3 assist cases under eval/aligner_assist/."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval.schema import AssistCaseTruth
from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.assist_propose import _AssistBatchResponse, _AssistDecision
from reckon.assist_select import select_assist_candidates
from reckon.schema import ExpectedTrace, ObservedTrace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval" / "aligner_assist"

CASE_DIRS = sorted(path for path in EVAL_DIR.iterdir() if path.is_dir())


def _load(case_dir: Path):
    expected = ExpectedTrace.model_validate_json(
        (case_dir / "expected.json").read_text(encoding="utf-8")
    )
    observed = ObservedTrace.model_validate_json(
        (case_dir / "observed.json").read_text(encoding="utf-8")
    )
    truth = AssistCaseTruth.model_validate_json(
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


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
def test_assist_golden_baseline(case_dir: Path):
    expected, observed, truth = _load(case_dir)
    audit = align(expected, observed)
    _assert_steps(audit, truth.baseline_steps)
    _assert_case_findings(audit, truth.baseline_case_findings)
    candidates = select_assist_candidates(expected, observed, audit)
    assert bool(candidates) == truth.candidate_expected


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
def test_assist_golden_scripted(case_dir: Path):
    expected, observed, truth = _load(case_dir)
    baseline = align(expected, observed)
    candidates = select_assist_candidates(expected, observed, baseline)
    decisions = []
    for candidate in candidates:
        if candidate.deterministic_support_index is not None:
            continue
        decisions.append(
            _AssistDecision(
                candidate_id=candidate.candidate_id,
                answer=truth.scripted_answer or "unresolved",
                confidence=truth.scripted_confidence,
                rationale="test scripted",
                supporting_event_id=truth.scripted_supporting_event_id,
            )
        )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(decisions=decisions)
    assisted = align_with_assist(expected, observed, openai_client=client)
    _assert_steps(assisted.result, truth.assisted_steps)
    _assert_case_findings(assisted.result, truth.assisted_case_findings)


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
def test_assist_golden_stub_no_false_pass(case_dir: Path):
    expected, observed, truth = _load(case_dir)
    baseline = align(expected, observed)
    candidates = select_assist_candidates(expected, observed, baseline)
    decisions = [
        _AssistDecision(
            candidate_id=c.candidate_id,
            answer="unresolved",
            confidence=0.0,
            rationale="stub",
        )
        for c in candidates
        if c.deterministic_support_index is None
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(decisions=decisions)
    assisted = align_with_assist(expected, observed, openai_client=client)

    if truth.llm_required:
        by_base = {s.index: s for s in truth.baseline_steps}
        by_res = {v.expected_step_index: v for v in assisted.result.step_verdicts}
        for idx, step in by_base.items():
            actual = by_res[idx]
            if not step.should_pass:
                assert actual.passed is False
    else:
        _assert_steps(assisted.result, truth.assisted_steps)
