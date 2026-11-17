"""Controlled live Stage 3 assist evaluation on approved LLM-required cases only.

Uses ALIGN_ASSIST_MODEL (default claude-opus-5-5) through the existing Uber gateway
and USSO_TOKEN. Does not modify goldens, aligner.py, or production callers.

Usage:
    PYTHONPATH=src:. python scripts/run_assist_live_eval.py \\
        --out /tmp/assist_live_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from eval.schema import AssistCaseTruth, StepTruth  # noqa: E402
from reckon.aligner import align  # noqa: E402
from reckon.assist import align_with_assist  # noqa: E402
from reckon.assist_propose import ALIGN_ASSIST_MODEL  # noqa: E402
from reckon.assist_select import select_assist_candidates  # noqa: E402
from reckon.parser import make_openai_client  # noqa: E402
from reckon.schema import ExpectedTrace, ObservedTrace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval" / "aligner_assist"

APPROVED_LIVE_CASES = (
    "wt_synonym_sign_in",
    "wt_different_controls",
    "da_synonym_later_banner",
    "da_unrelated_later_toast",
    "da_ambiguous_partial_text",
)


class _UsageTrackingClient:
    """Thin wrapper that records token usage from instructor/OpenAI responses."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.requests: list[dict[str, Any]] = []

    @property
    def chat(self) -> Any:
        return self

    @property
    def completions(self) -> Any:
        return self

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._client.chat.completions.create(*args, **kwargs)
        usage_info: dict[str, Any] = {"available": False}
        raw = getattr(response, "_raw_response", None) or getattr(response, "completion", None)
        usage = None
        if raw is not None:
            usage = getattr(raw, "usage", None)
            if usage is None and hasattr(raw, "model_extra"):
                usage = (raw.model_extra or {}).get("usage")
        if usage is None:
            usage = getattr(response, "usage", None)
        if usage is not None:
            usage_info = {
                "available": True,
                "prompt_tokens": getattr(usage, "prompt_tokens", None)
                or (usage.get("prompt_tokens") if isinstance(usage, dict) else None),
                "completion_tokens": getattr(usage, "completion_tokens", None)
                or (usage.get("completion_tokens") if isinstance(usage, dict) else None),
                "total_tokens": getattr(usage, "total_tokens", None)
                or (usage.get("total_tokens") if isinstance(usage, dict) else None),
            }
        self.requests.append(
            {
                "model": kwargs.get("model"),
                "usage": usage_info,
            }
        )
        return response


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


def _step_compare(audit, steps: list[StepTruth]) -> dict[str, Any]:
    by_index = {v.expected_step_index: v for v in audit.step_verdicts}
    rows = []
    correct = false_pass = false_fail = 0
    for step in steps:
        actual = by_index.get(step.index)
        if actual is None:
            false_fail += 1
            rows.append(
                {
                    "index": step.index,
                    "expected_pass": step.should_pass,
                    "actual_pass": None,
                    "expected_findings": [ft.value for ft in step.expected_finding_types],
                    "actual_findings": None,
                    "match": False,
                }
            )
            continue
        match = actual.passed == step.should_pass and sorted(
            ft.value for ft in actual.finding_types
        ) == sorted(ft.value for ft in step.expected_finding_types)
        if actual.passed == step.should_pass:
            correct += 1
        elif actual.passed and not step.should_pass:
            false_pass += 1
        else:
            false_fail += 1
        rows.append(
            {
                "index": step.index,
                "expected_pass": step.should_pass,
                "actual_pass": actual.passed,
                "expected_findings": [ft.value for ft in step.expected_finding_types],
                "actual_findings": [ft.value for ft in actual.finding_types],
                "evidence_timestamp_ms": actual.evidence_timestamp_ms,
                "matched_observed_index": actual.matched_observed_index,
                "match": match,
            }
        )
    return {
        "steps": rows,
        "correct": correct,
        "false_pass": false_pass,
        "false_fail": false_fail,
        "all_match": all(row["match"] for row in rows),
    }


def _support_timestamp(observed: ObservedTrace, support_index: int | None) -> int | None:
    if support_index is None:
        return None
    for action in observed.actions:
        if action.index == support_index:
            return action.timestamp_ms
    return None


def _investigate(
    truth: AssistCaseTruth,
    baseline_cmp: dict[str, Any],
    assisted_cmp: dict[str, Any],
    assisted_result_dump: dict[str, Any],
    baseline_dump: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    if assisted_cmp["false_pass"]:
        notes.append(
            "FALSE_PASS: assisted passed a step that assisted_truth requires to fail"
        )
    if assisted_cmp["false_fail"]:
        notes.append(
            "FALSE_FAIL: assisted failed a step that assisted_truth requires to pass"
        )
    if baseline_dump == assisted_result_dump and truth.improvement_class == "llm_semantic":
        notes.append(
            "NO_CHANGE: assisted AuditResult identical to deterministic baseline "
            "(Claude rejected/unresolved/failed or validation blocked apply)"
        )
    if not assisted_cmp["all_match"] and not notes:
        notes.append("MISMATCH: assisted outcome differs from assisted truth labels")
    return notes


def run_case(case_id: str, tracker: _UsageTrackingClient) -> dict[str, Any]:
    case_dir = EVAL_DIR / case_id
    expected, observed, truth = _load(case_dir)
    baseline = align(expected, observed)
    candidates = select_assist_candidates(expected, observed, baseline)

    before_requests = len(tracker.requests)
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=tracker,  # type: ignore[arg-type]
        model=os.environ.get("ALIGN_ASSIST_MODEL", ALIGN_ASSIST_MODEL),
    )
    new_requests = tracker.requests[before_requests:]

    baseline_cmp = _step_compare(baseline, truth.baseline_steps)
    assisted_vs_assisted_truth = _step_compare(assisted.result, truth.assisted_steps)
    assisted_vs_baseline_truth = _step_compare(assisted.result, truth.baseline_steps)

    proposals = []
    for record in assisted.assist.proposals:
        proposals.append(
            {
                "kind": record.kind.value,
                "status": record.status.value,
                "matching_method": record.matching_method.value,
                "expected_step_index": record.expected_step_index,
                "matched_observed_index": record.matched_observed_index,
                "supporting_observed_index": record.supporting_observed_index,
                "supporting_timestamp_ms": _support_timestamp(
                    observed, record.supporting_observed_index
                ),
                "finding_types": [ft.value for ft in record.finding_types],
                "model_answer": record.model_answer,
                "confidence": record.confidence,
                "rationale": record.rationale,
                "reason": record.reason,
            }
        )

    accepted = [p for p in proposals if p["status"] == "accepted"]
    rejected = [p for p in proposals if p["status"] == "rejected"]
    unresolved = [p for p in proposals if p.get("reason") == "model_unresolved"]
    failed = [p for p in proposals if p["status"] == "failed"]

    baseline_dump = baseline.model_dump(mode="json")
    assisted_dump = assisted.result.model_dump(mode="json")
    investigation = _investigate(
        truth,
        baseline_cmp,
        assisted_vs_assisted_truth,
        assisted_dump,
        baseline_dump,
    )

    return {
        "case_id": case_id,
        "model": assisted.assist.model,
        "truth": {
            "assist_kind": truth.assist_kind,
            "improvement_class": truth.improvement_class,
            "expected_resolution": truth.expected_resolution,
            "assisted_polarity": truth.assisted_polarity,
            "baseline_polarity": truth.baseline_polarity,
            "notes": truth.notes,
        },
        "baseline": {
            "verdict": baseline.verdict.value,
            "fidelity_score": baseline.fidelity_score,
            "findings": [f.model_dump(mode="json") for f in baseline.findings],
            "step_verdicts": [v.model_dump(mode="json") for v in baseline.step_verdicts],
            "vs_baseline_truth": baseline_cmp,
        },
        "assisted": {
            "verdict": assisted.result.verdict.value,
            "fidelity_score": assisted.result.fidelity_score,
            "findings": [f.model_dump(mode="json") for f in assisted.result.findings],
            "step_verdicts": [
                v.model_dump(mode="json") for v in assisted.result.step_verdicts
            ],
            "vs_assisted_truth": assisted_vs_assisted_truth,
            "vs_baseline_truth": assisted_vs_baseline_truth,
            "identical_to_deterministic": baseline_dump == assisted_dump,
            "assist_error": assisted.assist.error,
        },
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "kind": c.kind.value,
                "expected_step_index": c.expected_step_index,
                "matched_observed_index": c.matched_observed_index,
                "deterministic_support_index": c.deterministic_support_index,
            }
            for c in candidates
        ],
        "proposals": proposals,
        "proposal_counts": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "unresolved": len(unresolved),
            "failed": len(failed),
            "total": len(proposals),
        },
        "accepted_evidence": [
            {
                "expected_step_index": p["expected_step_index"],
                "matched_observed_index": p["matched_observed_index"],
                "supporting_observed_index": p["supporting_observed_index"],
                "supporting_timestamp_ms": p["supporting_timestamp_ms"],
                "model_answer": p["model_answer"],
                "confidence": p["confidence"],
            }
            for p in accepted
        ],
        "metrics": assisted.assist.metrics.model_dump(mode="json"),
        "gateway_requests": new_requests,
        "reviewed_semantic_pairs": [
            p.model_dump(mode="json") for p in assisted.assist.reviewed_semantic_pairs
        ],
        "investigation": investigation,
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases) or 1
    return {
        "cases": len(cases),
        "baseline_truth_match_rate": sum(
            c["baseline"]["vs_baseline_truth"]["all_match"] for c in cases
        )
        / n,
        "assisted_truth_match_rate": sum(
            c["assisted"]["vs_assisted_truth"]["all_match"] for c in cases
        )
        / n,
        "false_pass_cases": sum(
            1 for c in cases if c["assisted"]["vs_assisted_truth"]["false_pass"]
        ),
        "false_fail_cases": sum(
            1 for c in cases if c["assisted"]["vs_assisted_truth"]["false_fail"]
        ),
        "identical_to_baseline_cases": sum(
            1 for c in cases if c["assisted"]["identical_to_deterministic"]
        ),
        "proposals_accepted_total": sum(c["proposal_counts"]["accepted"] for c in cases),
        "proposals_rejected_total": sum(c["proposal_counts"]["rejected"] for c in cases),
        "proposals_unresolved_total": sum(
            c["proposal_counts"]["unresolved"] for c in cases
        ),
        "proposals_failed_total": sum(c["proposal_counts"]["failed"] for c in cases),
        "llm_requests_total": sum(c["metrics"]["llm_request_count"] for c in cases),
        "latency_ms": {
            "deterministic_sum": sum(c["metrics"]["deterministic_ms"] for c in cases),
            "llm_sum": sum(c["metrics"]["llm_ms"] for c in cases),
            "total_sum": sum(c["metrics"]["total_ms"] for c in cases),
            "deterministic_mean": sum(c["metrics"]["deterministic_ms"] for c in cases) / n,
            "llm_mean": sum(c["metrics"]["llm_ms"] for c in cases) / n,
            "total_mean": sum(c["metrics"]["total_ms"] for c in cases) / n,
        },
        "token_usage": {
            "recorded_requests": sum(len(c["gateway_requests"]) for c in cases),
            "usage_available_count": sum(
                1
                for c in cases
                for req in c["gateway_requests"]
                if req.get("usage", {}).get("available")
            ),
            "prompt_tokens_sum": sum(
                (req.get("usage") or {}).get("prompt_tokens") or 0
                for c in cases
                for req in c["gateway_requests"]
            ),
            "completion_tokens_sum": sum(
                (req.get("usage") or {}).get("completion_tokens") or 0
                for c in cases
                for req in c["gateway_requests"]
            ),
            "total_tokens_sum": sum(
                (req.get("usage") or {}).get("total_tokens") or 0
                for c in cases
                for req in c["gateway_requests"]
            ),
        },
        "cases_with_investigation_notes": sum(1 for c in cases if c["investigation"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/assist_live_eval.json"),
        help="JSON report output path",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(APPROVED_LIVE_CASES),
        help="Case ids to run (default: the five approved live cases)",
    )
    args = parser.parse_args()

    if not os.environ.get("USSO_TOKEN"):
        raise SystemExit("USSO_TOKEN is not set")

    model = os.environ.get("ALIGN_ASSIST_MODEL", ALIGN_ASSIST_MODEL)
    unknown = [c for c in args.cases if c not in APPROVED_LIVE_CASES]
    if unknown:
        raise SystemExit(
            f"Refusing non-approved live cases: {unknown}. "
            f"Approved: {list(APPROVED_LIVE_CASES)}"
        )

    tracker = _UsageTrackingClient(make_openai_client())
    results = [run_case(case_id, tracker) for case_id in args.cases]
    report = {
        "suite": "aligner_assist_live",
        "model": model,
        "approved_cases": list(APPROVED_LIVE_CASES),
        "ran_cases": list(args.cases),
        "aggregate": _aggregate(results),
        "cases": results,
    }
    text = json.dumps(report, indent=2)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
