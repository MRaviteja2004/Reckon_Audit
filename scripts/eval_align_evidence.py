"""Offline evaluation for Stage 3 evidence-aware assist (eval/aligner_evidence/).

Modes (no live API):
  baseline  — score deterministic align() against truth.baseline_*
  scripted  — align_with_assist(evidence_aware=True) with mock Claude roles from truth
  stub      — unresolved/low-confidence answers must not invent passes
  all       — run baseline + scripted + stub

Usage:
    PYTHONPATH=src:. python scripts/eval_align_evidence.py --mode baseline
    PYTHONPATH=src:. python scripts/eval_align_evidence.py --mode scripted
    PYTHONPATH=src:. python scripts/eval_align_evidence.py --mode stub
    PYTHONPATH=src:. python scripts/eval_align_evidence.py --mode all --out /tmp/evidence_eval.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import MagicMock

from dotenv import load_dotenv

load_dotenv()

from eval.schema import EvidenceCaseTruth, StepTruth  # noqa: E402
from reckon.aligner import align  # noqa: E402
from reckon.assist import align_with_assist  # noqa: E402
from reckon.assist_evidence_propose import (  # noqa: E402
    _EvidenceBatchResponse,
    _EvidenceDecision,
    _EvidenceRoleAssignment,
)
from reckon.assist_evidence_select import select_evidence_bundle_candidates  # noqa: E402
from reckon.assist_propose import _AssistBatchResponse  # noqa: E402
from reckon.schema import EvidenceRole, ExpectedTrace, ObservedTrace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval" / "aligner_evidence"

_ROLE_MAP = {
    "initiation": EvidenceRole.initiation,
    "progress": EvidenceRole.progress,
    "completion": EvidenceRole.completion,
    "assertion_evidence": EvidenceRole.assertion_evidence,
    "distractor": EvidenceRole.distractor,
}


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
    evidence_step_correct: int
    step_total: int
    baseline_false_pass_steps: int
    evidence_false_pass_steps: int
    baseline_matches_truth: bool
    evidence_matches_truth: bool
    stub_no_false_improvement: bool | None
    errors: list[str]


def _load(case_dir: Path) -> tuple[ExpectedTrace, ObservedTrace, EvidenceCaseTruth]:
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


def _step_score(audit, steps: list[StepTruth]) -> tuple[int, int, bool]:
    correct = false_pass = 0
    by_index = {v.expected_step_index: v for v in audit.step_verdicts}
    matches = True
    for step in steps:
        actual = by_index.get(step.index)
        if actual is None:
            matches = False
            continue
        if actual.passed == step.should_pass:
            correct += 1
        elif actual.passed and not step.should_pass:
            false_pass += 1
            matches = False
        else:
            matches = False
        got = sorted(ft.value for ft in actual.finding_types)
        want = sorted(ft.value for ft in step.expected_finding_types)
        if got != want:
            matches = False
    return correct, false_pass, matches


def _id_if_allowed(raw: str | None, allowed: set[int]) -> str | None:
    if raw is None:
        return None
    text = raw.strip().upper()
    if not text.startswith("O") or not text[1:].isdigit():
        return None
    return raw if int(text[1:]) in allowed else None


def _mock_client(candidates, truth: EvidenceCaseTruth, *, stub: bool) -> MagicMock:
    client = MagicMock()

    def _create(**kwargs):
        response_model = kwargs.get("response_model")
        if response_model is _EvidenceBatchResponse:
            if stub or (not truth.scripted_labels and truth.scripted_confidence < 0.7):
                return _EvidenceBatchResponse(
                    decisions=[
                        _EvidenceDecision(
                            candidate_id=c.candidate_id,
                            labels=[],
                            confidence=0.0,
                            rationale="stub unresolved",
                        )
                        for c in candidates
                    ]
                )
            labels = [
                _EvidenceRoleAssignment(
                    observed_event_id=f"O{item.observed_index}",
                    role=_ROLE_MAP[item.role],
                )
                for item in truth.scripted_labels
            ]
            decisions = []
            for candidate in candidates:
                allowed = {e.observed_index for e in candidate.evidence_events}
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
                        rationale="scripted",
                    )
                )
            return _EvidenceBatchResponse(decisions=decisions)
        return _AssistBatchResponse(decisions=[])

    client.chat.completions.create.side_effect = _create
    return client


def score_case(case_dir: Path, mode: str) -> CaseScore:
    expected, observed, truth = _load(case_dir)
    errors: list[str] = []
    baseline = align(expected, observed)
    candidates = select_evidence_bundle_candidates(expected, observed, baseline)
    base_correct, base_fp, base_ok = _step_score(baseline, truth.baseline_steps)

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
            baseline_step_correct=base_correct,
            evidence_step_correct=base_correct,
            step_total=len(truth.baseline_steps),
            baseline_false_pass_steps=base_fp,
            evidence_false_pass_steps=base_fp,
            baseline_matches_truth=base_ok,
            evidence_matches_truth=base_ok,
            stub_no_false_improvement=None,
            errors=errors,
        )

    stub = mode == "stub"
    client = _mock_client(candidates, truth, stub=stub)
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=client,
        evidence_aware=True,
    )
    ev_correct, ev_fp, ev_ok = _step_score(assisted.result, truth.evidence_steps)
    stub_ok = None
    if stub and truth.llm_required:
        by_base = {s.index: s for s in truth.baseline_steps}
        by_res = {v.expected_step_index: v for v in assisted.result.step_verdicts}
        stub_ok = all(
            (not step.should_pass and by_res[idx].passed is False)
            or step.should_pass
            for idx, step in by_base.items()
            if idx in by_res
        )
        # Unresolved must not turn a failing baseline step into a pass.
        stub_ok = all(
            not (
                (not step.should_pass)
                and by_res[idx].passed
            )
            for idx, step in by_base.items()
            if idx in by_res
        )

    return CaseScore(
        case_id=truth.case_id,
        mode=mode,
        improvement_class=truth.improvement_class,
        llm_required=truth.llm_required,
        candidate_expected=truth.candidate_expected,
        candidates=len(candidates),
        llm_requests=assisted.assist.metrics.llm_request_count,
        proposals_accepted=assisted.assist.metrics.proposals_accepted,
        baseline_step_correct=base_correct,
        evidence_step_correct=ev_correct,
        step_total=len(truth.evidence_steps),
        baseline_false_pass_steps=base_fp,
        evidence_false_pass_steps=ev_fp,
        baseline_matches_truth=base_ok,
        evidence_matches_truth=ev_ok if mode == "scripted" else base_ok,
        stub_no_false_improvement=stub_ok,
        errors=errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["baseline", "scripted", "stub", "all"],
        default="all",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    modes = (
        ["baseline", "scripted", "stub"]
        if args.mode == "all"
        else [args.mode]
    )
    case_dirs = sorted(path for path in EVAL_DIR.iterdir() if path.is_dir())
    scores = [
        score_case(case_dir, mode)
        for mode in modes
        for case_dir in case_dirs
    ]
    payload = {"scores": [asdict(score) for score in scores]}
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for score in scores:
        flag = "OK" if (
            (score.mode == "baseline" and score.baseline_matches_truth)
            or (score.mode == "scripted" and score.evidence_matches_truth)
            or (score.mode == "stub" and score.stub_no_false_improvement is not False)
        ) else "FAIL"
        print(
            f"{flag} {score.mode:8} {score.case_id} "
            f"cands={score.candidates} accepted={score.proposals_accepted}"
        )


if __name__ == "__main__":
    main()
