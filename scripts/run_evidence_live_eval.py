"""Controlled live Stage 3 evidence-aware evaluation on eval/aligner_evidence/.

Compares deterministic align() with evidence-aware assist (evidence_aware=True)
using claude-opus-5-5 via the Uber gateway.

Stops immediately if any false pass is observed against evidence_steps truth.

Usage:
    PYTHONPATH=src:. python scripts/run_evidence_live_eval.py \\
        --out /tmp/evidence_live_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from eval.schema import EvidenceCaseTruth, StepTruth  # noqa: E402
from reckon.aligner import align  # noqa: E402
from reckon.assist import align_with_assist  # noqa: E402
from reckon.assist_evidence_propose import ALIGN_EVIDENCE_MODEL  # noqa: E402
from reckon.assist_evidence_select import select_evidence_bundle_candidates  # noqa: E402
from reckon.assist_propose import ALIGN_ASSIST_MODEL  # noqa: E402
from reckon.parser import make_openai_client  # noqa: E402
from reckon.schema import ExpectedTrace, ObservedTrace  # noqa: E402
from reckon.tolerance import assertion_satisfied  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval" / "aligner_evidence"


class _UsageTrackingClient:
    """Record token usage and wall latency per gateway create() call."""

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
        started = time.perf_counter()
        response = self._client.chat.completions.create(*args, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage_info: dict[str, Any] = {"available": False}
        raw = getattr(response, "_raw_response", None) or getattr(
            response, "completion", None
        )
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
                or (
                    usage.get("completion_tokens") if isinstance(usage, dict) else None
                ),
                "total_tokens": getattr(usage, "total_tokens", None)
                or (usage.get("total_tokens") if isinstance(usage, dict) else None),
            }
        self.requests.append(
            {
                "model": kwargs.get("model"),
                "latency_ms": latency_ms,
                "usage": usage_info,
            }
        )
        return response


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


def _obs_lookup(observed: ObservedTrace) -> dict[int, Any]:
    return {action.index: action for action in observed.actions}


def _support_meta(
    observed: ObservedTrace, support_index: int | None
) -> dict[str, Any]:
    if support_index is None:
        return {
            "supporting_observed_index": None,
            "supporting_timestamp_ms": None,
            "timestamp_matches_observed_trace": None,
        }
    action = _obs_lookup(observed).get(support_index)
    if action is None:
        return {
            "supporting_observed_index": support_index,
            "supporting_timestamp_ms": None,
            "timestamp_matches_observed_trace": False,
            "error": "supporting index not in ObservedTrace",
        }
    return {
        "supporting_observed_index": support_index,
        "supporting_timestamp_ms": action.timestamp_ms,
        "timestamp_matches_observed_trace": True,
        "supporting_target": action.target,
        "supporting_action": action.action.value,
    }


def _assertion_checks(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    step_by = {s.index: s for s in expected.steps}
    action_by = _obs_lookup(observed)
    checks = []
    for proposal in proposals:
        if proposal["kind"] != "evidence_bundle":
            continue
        step = step_by.get(proposal["expected_step_index"])
        support_idx = proposal.get("supporting_observed_index")
        matched_idx = proposal.get("matched_observed_index")
        row: dict[str, Any] = {
            "expected_step_index": proposal["expected_step_index"],
            "status": proposal["status"],
            "challenged_false_pass": proposal.get("challenged_false_pass"),
            "reason": proposal.get("reason"),
            "assertion": getattr(step, "assertion", None) if step else None,
        }
        if support_idx is not None and step is not None and step.assertion:
            action = action_by.get(support_idx)
            row["support_assertion_satisfied"] = (
                assertion_satisfied(step.assertion, action) if action else False
            )
        if matched_idx is not None and step is not None and step.assertion:
            action = action_by.get(matched_idx)
            row["matched_assertion_satisfied"] = (
                assertion_satisfied(step.assertion, action) if action else False
            )
        checks.append(row)
    return checks


def _workflow_isolation_notes(
    observed: ObservedTrace,
    proposals: list[dict[str, Any]],
) -> list[str]:
    """Flag if initiation/support ids look like they span unrelated later workflows."""
    notes: list[str] = []
    action_by = _obs_lookup(observed)
    for proposal in proposals:
        if proposal["kind"] != "evidence_bundle":
            continue
        init = proposal.get("initiation_observed_index")
        support = proposal.get("supporting_observed_index")
        completion = proposal.get("completion_observed_index")
        ids = [i for i in (init, support, completion) if i is not None]
        if len(ids) < 2:
            continue
        times = []
        for idx in ids:
            action = action_by.get(idx)
            if action is None:
                notes.append(
                    f"step {proposal['expected_step_index']}: cited O{idx} missing from trace"
                )
                continue
            times.append((idx, action.timestamp_ms))
        if len(times) >= 2:
            span = max(t for _, t in times) - min(t for _, t in times)
            # Heuristic only: large gaps may indicate cross-attempt mixing.
            if span > 60_000:
                notes.append(
                    f"step {proposal['expected_step_index']}: cited evidence span "
                    f"{span}ms across {times} — inspect for repeated-workflow mixing"
                )
    return notes


def _proposal_rows(assisted, observed: ObservedTrace) -> list[dict[str, Any]]:
    rows = []
    for record in assisted.assist.proposals:
        support_meta = _support_meta(observed, record.supporting_observed_index)
        initiation_meta = _support_meta(observed, record.initiation_observed_index)
        completion_meta = _support_meta(observed, record.completion_observed_index)
        rows.append(
            {
                "kind": record.kind.value,
                "status": record.status.value,
                "matching_method": record.matching_method.value,
                "expected_step_index": record.expected_step_index,
                "matched_observed_index": record.matched_observed_index,
                "supporting_observed_index": record.supporting_observed_index,
                "initiation_observed_index": record.initiation_observed_index,
                "completion_observed_index": record.completion_observed_index,
                "supporting_timestamp_ms": support_meta["supporting_timestamp_ms"],
                "initiation_timestamp_ms": initiation_meta["supporting_timestamp_ms"],
                "completion_timestamp_ms": completion_meta["supporting_timestamp_ms"],
                "timestamps_match_observed_trace": all(
                    meta["timestamp_matches_observed_trace"] in {True, None}
                    for meta in (support_meta, initiation_meta, completion_meta)
                ),
                "finding_types": [ft.value for ft in record.finding_types],
                "model_answer": record.model_answer,
                "confidence": record.confidence,
                "rationale": record.rationale,
                "reason": record.reason,
                "challenged_false_pass": record.challenged_false_pass,
            }
        )
    return rows


def _path_result(
    label: str,
    assisted,
    observed: ObservedTrace,
    truth: EvidenceCaseTruth,
    baseline,
    gateway_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    proposals = _proposal_rows(assisted, observed)
    vs_evidence = _step_compare(assisted.result, truth.evidence_steps)
    vs_baseline = _step_compare(assisted.result, truth.baseline_steps)
    baseline_dump = baseline.model_dump(mode="json")
    result_dump = assisted.result.model_dump(mode="json")
    return {
        "label": label,
        "verdict": assisted.result.verdict.value,
        "fidelity_score": assisted.result.fidelity_score,
        "findings": [f.model_dump(mode="json") for f in assisted.result.findings],
        "step_verdicts": [
            v.model_dump(mode="json") for v in assisted.result.step_verdicts
        ],
        "vs_evidence_truth": vs_evidence,
        "vs_baseline_truth": vs_baseline,
        "identical_to_deterministic": baseline_dump == result_dump,
        "deterministic_preserved": (
            assisted.deterministic_result.model_dump(mode="json") == baseline_dump
        ),
        "assist_error": assisted.assist.error,
        "proposals": proposals,
        "proposal_counts": {
            "accepted": sum(1 for p in proposals if p["status"] == "accepted"),
            "rejected": sum(1 for p in proposals if p["status"] == "rejected"),
            "failed": sum(1 for p in proposals if p["status"] == "failed"),
            "unresolved": sum(
                1 for p in proposals if p.get("reason") == "model_unresolved"
            ),
            "challenged_false_pass": sum(
                1 for p in proposals if p.get("challenged_false_pass")
            ),
            "total": len(proposals),
        },
        "assertion_satisfied_checks": [],
        "workflow_isolation_notes": _workflow_isolation_notes(observed, proposals),
        "metrics": assisted.assist.metrics.model_dump(mode="json"),
        "gateway_requests": gateway_requests,
        "reviewed_semantic_pairs": [
            p.model_dump(mode="json") for p in assisted.assist.reviewed_semantic_pairs
        ],
        "reviewed_pairs_accepted_only": all(
            any(
                p["status"] == "accepted"
                and p["expected_step_index"] == pair.expected_step_index
                and p["matched_observed_index"] == pair.observed_action_index
                for p in proposals
            )
            for pair in assisted.assist.reviewed_semantic_pairs
        )
        if assisted.assist.reviewed_semantic_pairs
        else True,
    }


def run_case(
    case_id: str,
    tracker: _UsageTrackingClient,
    model: str,
) -> dict[str, Any]:
    case_dir = EVAL_DIR / case_id
    expected, observed, truth = _load(case_dir)
    baseline = align(expected, observed)
    evidence_candidates = select_evidence_bundle_candidates(
        expected, observed, baseline
    )

    before = len(tracker.requests)
    assisted_evidence = align_with_assist(
        expected,
        observed,
        openai_client=tracker,  # type: ignore[arg-type]
        model=model,
        evidence_aware=True,
    )
    evidence_requests = tracker.requests[before:]

    evidence = _path_result(
        "evidence_aware",
        assisted_evidence,
        observed,
        truth,
        baseline,
        evidence_requests,
    )
    evidence["assertion_satisfied_checks"] = _assertion_checks(
        expected, observed, evidence["proposals"]
    )

    baseline_cmp = _step_compare(baseline, truth.baseline_steps)
    investigation: list[str] = []
    if evidence["vs_evidence_truth"]["false_pass"]:
        investigation.append(
            "FALSE_PASS: evidence-aware passed a step that evidence_truth requires to fail"
        )
    if not evidence["deterministic_preserved"]:
        investigation.append("SAFETY: deterministic_result mutated vs align() baseline")
    if not evidence["reviewed_pairs_accepted_only"]:
        investigation.append(
            "SAFETY: reviewed_semantic_pairs includes non-accepted proposals"
        )
    for note in evidence["workflow_isolation_notes"]:
        investigation.append(f"WORKFLOW: {note}")
    if evidence["vs_evidence_truth"]["false_fail"]:
        investigation.append(
            "FALSE_FAIL: evidence-aware failed a step that evidence_truth requires to pass"
        )
    if not evidence["vs_evidence_truth"]["all_match"] and not any(
        n.startswith("FALSE_") for n in investigation
    ):
        investigation.append("MISMATCH: evidence-aware outcome differs from evidence truth")

    return {
        "case_id": case_id,
        "model": model,
        "truth": {
            "notes": truth.notes,
            "baseline_polarity": truth.baseline_polarity,
            "evidence_polarity": truth.evidence_polarity,
            "expected_resolution": truth.expected_resolution,
            "improvement_class": truth.improvement_class,
            "candidate_expected": truth.candidate_expected,
            "llm_required": truth.llm_required,
        },
        "baseline": {
            "verdict": baseline.verdict.value,
            "fidelity_score": baseline.fidelity_score,
            "findings": [f.model_dump(mode="json") for f in baseline.findings],
            "step_verdicts": [v.model_dump(mode="json") for v in baseline.step_verdicts],
            "vs_baseline_truth": baseline_cmp,
        },
        "evidence_candidates": [
            {
                "candidate_id": c.candidate_id,
                "expected_step_index": c.expected_step_index,
                "matched_observed_index": c.matched_observed_index,
                "suspicious_pass": c.suspicious_pass,
                "window_ids": [e.observed_index for e in c.evidence_events],
            }
            for c in evidence_candidates
        ],
        "evidence_aware": evidence,
        "investigation": investigation,
        "stop_recommended": bool(
            evidence["vs_evidence_truth"]["false_pass"]
            or any(n.startswith("FALSE_PASS") for n in investigation)
        ),
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases) or 1

    def _path_agg(key: str) -> dict[str, Any]:
        return {
            "truth_match_rate": sum(
                c[key]["vs_evidence_truth"]["all_match"] for c in cases
            )
            / n,
            "false_pass_cases": sum(
                1 for c in cases if c[key]["vs_evidence_truth"]["false_pass"]
            ),
            "false_fail_cases": sum(
                1 for c in cases if c[key]["vs_evidence_truth"]["false_fail"]
            ),
            "identical_to_baseline_cases": sum(
                1 for c in cases if c[key]["identical_to_deterministic"]
            ),
            "proposals_accepted_total": sum(
                c[key]["proposal_counts"]["accepted"] for c in cases
            ),
            "proposals_rejected_total": sum(
                c[key]["proposal_counts"]["rejected"] for c in cases
            ),
            "proposals_failed_total": sum(
                c[key]["proposal_counts"]["failed"] for c in cases
            ),
            "challenged_false_pass_total": sum(
                c[key]["proposal_counts"]["challenged_false_pass"] for c in cases
            ),
            "llm_requests_total": sum(
                c[key]["metrics"]["llm_request_count"] for c in cases
            ),
            "latency_ms": {
                "deterministic_sum": sum(
                    c[key]["metrics"]["deterministic_ms"] for c in cases
                ),
                "llm_sum": sum(c[key]["metrics"]["llm_ms"] for c in cases),
                "total_sum": sum(c[key]["metrics"]["total_ms"] for c in cases),
                "deterministic_mean": sum(
                    c[key]["metrics"]["deterministic_ms"] for c in cases
                )
                / n,
                "llm_mean": sum(c[key]["metrics"]["llm_ms"] for c in cases) / n,
                "total_mean": sum(c[key]["metrics"]["total_ms"] for c in cases) / n,
            },
            "token_usage": {
                "recorded_requests": sum(len(c[key]["gateway_requests"]) for c in cases),
                "usage_available_count": sum(
                    1
                    for c in cases
                    for req in c[key]["gateway_requests"]
                    if req.get("usage", {}).get("available")
                ),
                "prompt_tokens_sum": sum(
                    (req.get("usage") or {}).get("prompt_tokens") or 0
                    for c in cases
                    for req in c[key]["gateway_requests"]
                ),
                "completion_tokens_sum": sum(
                    (req.get("usage") or {}).get("completion_tokens") or 0
                    for c in cases
                    for req in c[key]["gateway_requests"]
                ),
                "total_tokens_sum": sum(
                    (req.get("usage") or {}).get("total_tokens") or 0
                    for c in cases
                    for req in c[key]["gateway_requests"]
                ),
                "per_request_latency_ms": [
                    req.get("latency_ms")
                    for c in cases
                    for req in c[key]["gateway_requests"]
                ],
            },
            "reviewed_pairs_accepted_only_ok": all(
                c[key]["reviewed_pairs_accepted_only"] for c in cases
            ),
            "deterministic_preserved_ok": all(
                c[key]["deterministic_preserved"] for c in cases
            ),
        }

    return {
        "cases": len(cases),
        "baseline_truth_match_rate": sum(
            c["baseline"]["vs_baseline_truth"]["all_match"] for c in cases
        )
        / n,
        "evidence_aware": _path_agg("evidence_aware"),
        "stop_recommended_cases": [
            c["case_id"] for c in cases if c.get("stop_recommended")
        ],
        "cases_with_investigation_notes": sum(1 for c in cases if c["investigation"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/evidence_live_eval.json"),
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Subset of case ids (default: all under eval/aligner_evidence/)",
    )
    args = parser.parse_args()

    if not os.environ.get("USSO_TOKEN"):
        raise SystemExit("USSO_TOKEN is not set")

    model = os.environ.get(
        "ALIGN_EVIDENCE_MODEL",
        os.environ.get("ALIGN_ASSIST_MODEL", ALIGN_EVIDENCE_MODEL or ALIGN_ASSIST_MODEL),
    )
    available = sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())
    case_ids = list(args.cases) if args.cases else available
    unknown = [c for c in case_ids if c not in available]
    if unknown:
        raise SystemExit(f"Unknown cases: {unknown}. Available: {available}")

    tracker = _UsageTrackingClient(make_openai_client())
    results: list[dict[str, Any]] = []
    stopped_early = False
    for case_id in case_ids:
        print(f"Running {case_id} ...", flush=True)
        result = run_case(case_id, tracker, model)
        results.append(result)
        print(
            f"  evidence match={result['evidence_aware']['vs_evidence_truth']['all_match']} "
            f"fp={result['evidence_aware']['vs_evidence_truth']['false_pass']} "
            f"notes={result['investigation']}",
            flush=True,
        )
        if result["stop_recommended"]:
            stopped_early = True
            print(
                f"STOP: false pass observed on {case_id}. Halting remaining cases.",
                flush=True,
            )
            break

    report = {
        "suite": "aligner_evidence_live",
        "model": model,
        "ran_cases": [c["case_id"] for c in results],
        "available_cases": available,
        "stopped_early": stopped_early,
        "aligner_hash": "623c1d84fbac1003fa2759cea214a91f5363ff16",
        "aggregate": _aggregate(results),
        "cases": results,
    }
    text = json.dumps(report, indent=2)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if stopped_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
