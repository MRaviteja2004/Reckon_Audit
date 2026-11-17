"""Offline evaluation for Stage 3 assist against eval/aligner_assist/.

Modes (no live API unless --live is explicitly passed later):
  baseline   — score deterministic align() against truth.baseline_*
  scripted   — align_with_assist with mock Claude answers from truth
  stub       — align_with_assist with unresolved answers; must not invent passes
               over baseline for llm_required cases. Deterministic delayed may still improve.

Usage:
    PYTHONPATH=src:. python scripts/eval_align_assist.py --mode baseline
    PYTHONPATH=src:. python scripts/eval_align_assist.py --mode scripted
    PYTHONPATH=src:. python scripts/eval_align_assist.py --mode stub
    PYTHONPATH=src:. python scripts/eval_align_assist.py --mode all --out /tmp/assist_eval.json

Do not pass --live without explicit approval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import MagicMock

from dotenv import load_dotenv

load_dotenv()

from eval.schema import AssistCaseTruth, StepTruth  # noqa: E402
from reckon.aligner import align  # noqa: E402
from reckon.assist import align_with_assist  # noqa: E402
from reckon.assist_propose import _AssistBatchResponse, _AssistDecision  # noqa: E402
from reckon.assist_select import select_assist_candidates  # noqa: E402
from reckon.schema import ExpectedTrace, FindingType, ObservedTrace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval" / "aligner_assist"


@dataclass
class CaseScore:
    case_id: str
    mode: str
    improvement_class: str
    llm_required: bool
    candidate_expected: bool
    candidates: int
    llm_requests: int
    proposals_accepted: int
    baseline_step_correct: int
    assisted_step_correct: int
    step_total: int
    baseline_false_pass_steps: int
    assisted_false_pass_steps: int
    baseline_false_fail_steps: int
    assisted_false_fail_steps: int
    baseline_matches_truth: bool
    assisted_matches_truth: bool
    stub_no_false_improvement: bool | None
    deterministic_ms: float
    llm_ms: float
    total_ms: float
    errors: list[str]


def _load(case_dir: Path) -> tuple[ExpectedTrace, ObservedTrace, AssistCaseTruth]:
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


def _step_score(audit, steps: list[StepTruth]) -> tuple[int, int, int, bool]:
    correct = false_pass = false_fail = 0
    by_index = {v.expected_step_index: v for v in audit.step_verdicts}
    matches = True
    for step in steps:
        actual = by_index.get(step.index)
        if actual is None:
            false_fail += 1
            matches = False
            continue
        if actual.passed == step.should_pass:
            correct += 1
        elif actual.passed and not step.should_pass:
            false_pass += 1
            matches = False
        else:
            false_fail += 1
            matches = False
        got = sorted(ft.value for ft in actual.finding_types)
        want = sorted(ft.value for ft in step.expected_finding_types)
        if got != want:
            matches = False
    return correct, false_pass, false_fail, matches


def _case_findings_match(audit, expected: list[FindingType]) -> bool:
    got = sorted(
        f.type.value for f in audit.findings if f.expected_step_index is None
    )
    want = sorted(f.value for f in expected)
    return got == want


def _scripted_client(expected, observed, audit, truth: AssistCaseTruth):
    candidates = select_assist_candidates(expected, observed, audit)
    decisions: list[_AssistDecision] = []
    for candidate in candidates:
        if candidate.deterministic_support_index is not None:
            continue
        answer = truth.scripted_answer or "unresolved"
        decisions.append(
            _AssistDecision(
                candidate_id=candidate.candidate_id,
                answer=answer,
                confidence=truth.scripted_confidence,
                rationale=f"scripted:{truth.case_id}",
                supporting_event_id=truth.scripted_supporting_event_id,
            )
        )
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(decisions=decisions)
    return client


def _stub_client(expected, observed, audit):
    candidates = select_assist_candidates(expected, observed, audit)
    decisions = [
        _AssistDecision(
            candidate_id=c.candidate_id,
            answer="unresolved",
            confidence=0.0,
            rationale="stub unresolved",
        )
        for c in candidates
        if c.deterministic_support_index is None
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(decisions=decisions)
    return client


def _score_case(case_dir: Path, mode: str) -> CaseScore:
    expected, observed, truth = _load(case_dir)
    baseline = align(expected, observed)
    errors: list[str] = []

    b_correct, b_fp, b_ff, b_match_steps = _step_score(baseline, truth.baseline_steps)
    b_match = b_match_steps and _case_findings_match(
        baseline, truth.baseline_case_findings
    )
    if not b_match:
        errors.append("baseline_mismatch")

    candidates = select_assist_candidates(expected, observed, baseline)
    if truth.candidate_expected and not candidates:
        errors.append("missing_expected_candidate")
    if not truth.candidate_expected and candidates:
        errors.append("unexpected_candidate")

    if mode == "baseline":
        return CaseScore(
            case_id=truth.case_id,
            mode=mode,
            improvement_class=truth.improvement_class,
            llm_required=truth.llm_required,
            candidate_expected=truth.candidate_expected,
            candidates=len(candidates),
            llm_requests=0,
            proposals_accepted=0,
            baseline_step_correct=b_correct,
            assisted_step_correct=b_correct,
            step_total=len(truth.baseline_steps),
            baseline_false_pass_steps=b_fp,
            assisted_false_pass_steps=b_fp,
            baseline_false_fail_steps=b_ff,
            assisted_false_fail_steps=b_ff,
            baseline_matches_truth=b_match,
            assisted_matches_truth=b_match,
            stub_no_false_improvement=None,
            deterministic_ms=0.0,
            llm_ms=0.0,
            total_ms=0.0,
            errors=errors,
        )

    if mode == "scripted":
        client = _scripted_client(expected, observed, baseline, truth)
    elif mode == "stub":
        client = _stub_client(expected, observed, baseline)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    assisted = align_with_assist(expected, observed, openai_client=client)
    compare_steps = (
        truth.assisted_steps if mode == "scripted" else truth.baseline_steps
    )
    # Stub mode: llm_required cases must not improve over baseline truth.
    # Deterministic delayed may still clear findings without Claude.
    if mode == "stub" and not truth.llm_required and truth.improvement_class == "deterministic_delayed":
        compare_steps = truth.assisted_steps
        compare_case = truth.assisted_case_findings
    elif mode == "stub":
        compare_steps = truth.baseline_steps
        compare_case = truth.baseline_case_findings
    else:
        compare_case = truth.assisted_case_findings

    a_correct, a_fp, a_ff, a_match_steps = _step_score(assisted.result, compare_steps)
    a_match = a_match_steps and _case_findings_match(assisted.result, compare_case)
    if mode == "scripted" and not a_match:
        errors.append("assisted_mismatch")

    stub_ok: bool | None = None
    if mode == "stub":
        if truth.llm_required:
            # No improvement over baseline for LLM-dependent cases.
            stub_ok = assisted.result.model_dump() == baseline.model_dump() or (
                a_fp == 0 and a_match_steps and _case_findings_match(
                    assisted.result, truth.baseline_case_findings
                )
            )
            # Stronger: step pass flags must not flip false→true vs baseline truth.
            by_base = {s.index: s for s in truth.baseline_steps}
            by_res = {v.expected_step_index: v for v in assisted.result.step_verdicts}
            for idx, step in by_base.items():
                actual = by_res.get(idx)
                if actual is not None and (not step.should_pass) and actual.passed:
                    stub_ok = False
                    errors.append("stub_false_pass")
                    break
            if stub_ok is None:
                stub_ok = True
        else:
            stub_ok = a_match
        if stub_ok is False and "stub_false_pass" not in errors and not a_match:
            errors.append("stub_mismatch")

    return CaseScore(
        case_id=truth.case_id,
        mode=mode,
        improvement_class=truth.improvement_class,
        llm_required=truth.llm_required,
        candidate_expected=truth.candidate_expected,
        candidates=assisted.assist.metrics.candidate_count,
        llm_requests=assisted.assist.metrics.llm_request_count,
        proposals_accepted=assisted.assist.metrics.proposals_accepted,
        baseline_step_correct=b_correct,
        assisted_step_correct=a_correct,
        step_total=len(truth.baseline_steps),
        baseline_false_pass_steps=b_fp,
        assisted_false_pass_steps=a_fp,
        baseline_false_fail_steps=b_ff,
        assisted_false_fail_steps=a_ff,
        baseline_matches_truth=b_match,
        assisted_matches_truth=a_match if mode == "scripted" else bool(stub_ok),
        stub_no_false_improvement=stub_ok,
        deterministic_ms=assisted.assist.metrics.deterministic_ms,
        llm_ms=assisted.assist.metrics.llm_ms,
        total_ms=assisted.assist.metrics.total_ms,
        errors=errors,
    )


def _aggregate(scores: list[CaseScore]) -> dict:
    step_total = sum(s.step_total for s in scores) or 1
    det = [s for s in scores if s.improvement_class == "deterministic_delayed"]
    llm = [s for s in scores if s.improvement_class == "llm_semantic"]
    return {
        "cases": len(scores),
        "baseline_truth_match_rate": sum(s.baseline_matches_truth for s in scores) / len(scores),
        "assisted_truth_match_rate": sum(s.assisted_matches_truth for s in scores) / len(scores),
        "baseline_step_accuracy": sum(s.baseline_step_correct for s in scores) / step_total,
        "assisted_step_accuracy": sum(s.assisted_step_correct for s in scores) / step_total,
        "baseline_false_pass_steps": sum(s.baseline_false_pass_steps for s in scores),
        "assisted_false_pass_steps": sum(s.assisted_false_pass_steps for s in scores),
        "baseline_false_fail_steps": sum(s.baseline_false_fail_steps for s in scores),
        "assisted_false_fail_steps": sum(s.assisted_false_fail_steps for s in scores),
        "candidates_total": sum(s.candidates for s in scores),
        "llm_requests_total": sum(s.llm_requests for s in scores),
        "proposals_accepted_total": sum(s.proposals_accepted for s in scores),
        "deterministic_delayed_cases": len(det),
        "deterministic_delayed_assisted_match_rate": (
            sum(s.assisted_matches_truth for s in det) / len(det) if det else None
        ),
        "llm_semantic_cases": len(llm),
        "llm_semantic_assisted_match_rate": (
            sum(s.assisted_matches_truth for s in llm) / len(llm) if llm else None
        ),
        "cases_with_errors": sum(1 for s in scores if s.errors),
        "latency_ms": {
            "deterministic_sum": sum(s.deterministic_ms for s in scores),
            "llm_sum": sum(s.llm_ms for s in scores),
            "total_sum": sum(s.total_ms for s in scores),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "scripted", "stub", "all"),
        default="all",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Disabled in this preparation pass; requires separate approval",
    )
    args = parser.parse_args()
    if args.live:
        raise SystemExit(
            "Live Claude evaluation is gated. Re-run only after explicit approval."
        )

    case_dirs = sorted(path for path in EVAL_DIR.iterdir() if path.is_dir())
    modes = ["baseline", "scripted", "stub"] if args.mode == "all" else [args.mode]
    report: dict = {"suite": "aligner_assist", "modes": {}}
    for mode in modes:
        scores = [_score_case(case_dir, mode) for case_dir in case_dirs]
        report["modes"][mode] = {
            "aggregate": _aggregate(scores),
            "cases": [asdict(score) for score in scores],
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
