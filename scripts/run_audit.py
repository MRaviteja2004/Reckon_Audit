import argparse
import json
from pathlib import Path

from reckon.aligner import align
from reckon.schema import AuditResult, ExpectedTrace, ObservedTrace

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EXPECTED = _REPO_ROOT / "tests" / "fixtures" / "login_expected.json"
_DEFAULT_OBSERVED = _REPO_ROOT / "tests" / "fixtures" / "login_observed_broken.json"


def _load_expected(path: Path) -> ExpectedTrace:
    return ExpectedTrace.model_validate(json.loads(path.read_text()))


def _load_observed(path: Path) -> ObservedTrace:
    return ObservedTrace.model_validate(json.loads(path.read_text()))


def _format_report(scenario_title: str, result: AuditResult) -> str:
    lines = [
        f"Scenario:       {scenario_title}",
        f"Fidelity score: {result.fidelity_score:.0%}",
        "",
        "Per-step verdicts:",
    ]
    for verdict in result.step_verdicts:
        mark = "PASS" if verdict.passed else "FAIL"
        evidence = (
            f"@{verdict.evidence_timestamp_ms}ms"
            if verdict.evidence_timestamp_ms is not None
            else "(no evidence)"
        )
        lines.append(f"  [{mark}] step {verdict.expected_step_index} {evidence}")

    lines.append("")
    lines.append("Findings:")
    if not result.findings:
        lines.append("  (none)")
    for finding in result.findings:
        evidence = (
            f"@{finding.evidence_timestamp_ms}ms"
            if finding.evidence_timestamp_ms is not None
            else ""
        )
        lines.append(f"  - {finding.type.value} {evidence}: {finding.description}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage 3 deterministic aligner on two trace JSON files.",
    )
    parser.add_argument(
        "expected",
        nargs="?",
        type=Path,
        default=_DEFAULT_EXPECTED,
        help="Path to an expected-trace JSON file",
    )
    parser.add_argument(
        "observed",
        nargs="?",
        type=Path,
        default=_DEFAULT_OBSERVED,
        help="Path to an observed-trace JSON file",
    )
    args = parser.parse_args()

    for path in (args.expected, args.observed):
        if not path.exists():
            raise SystemExit(f"Trace file not found: {path}")

    expected = _load_expected(args.expected)
    observed = _load_observed(args.observed)
    result = align(expected, observed)
    print(_format_report(expected.scenario_title, result))


if __name__ == "__main__":
    main()
