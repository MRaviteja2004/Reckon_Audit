from pathlib import Path
import sys
import re

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

# Allow root-level eval package imports when running this file directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from eval.schema import CaseTruth
from reckon.aligner import align
from reckon.parser import parse_scenario
from reckon.schema import ExpectedTrace, ObservedTrace


ALIGNER_EVAL_DIR = PROJECT_ROOT / "eval" / "aligner"
PARSER_EVAL_DIR = PROJECT_ROOT / "eval" / "parser"

DEFAULT_PARSER_RUNS = 1
DEFAULT_PARSER_RETRIES = 2


def load_aligner_case(case_dir: Path):
    """Load and validate one aligner golden-set case."""

    expected = ExpectedTrace.model_validate_json(
        (case_dir / "expected.json").read_text(encoding="utf-8")
    )

    observed = ObservedTrace.model_validate_json(
        (case_dir / "observed.json").read_text(encoding="utf-8")
    )

    truth = CaseTruth.model_validate_json(
        (case_dir / "truth.json").read_text(encoding="utf-8")
    )

    return expected, observed, truth


def evaluate_aligner_case(case_dir: Path) -> dict:
    """Run one golden case and compare AuditResult against human truth."""

    expected, observed, truth = load_aligner_case(case_dir)

    audit = align(expected, observed)

    all_correct = True

    # ------------------------------------------------------------
    # Compare expected-step verdicts
    # ------------------------------------------------------------

    for step_truth in truth.steps:
        actual = next(
            (
                verdict
                for verdict in audit.step_verdicts
                if verdict.expected_step_index == step_truth.index
            ),
            None,
        )

        if actual is None:
            all_correct = False
            continue

        pass_matches = actual.passed == step_truth.should_pass

        findings_match = (
            sorted(f.value for f in actual.finding_types)
            == sorted(
                f.value
                for f in step_truth.expected_finding_types
            )
        )

        if not (pass_matches and findings_match):
            all_correct = False

    # ------------------------------------------------------------
    # Compare case-level findings
    # ------------------------------------------------------------

    actual_case_findings = [
        finding.type
        for finding in audit.findings
        if finding.expected_step_index is None
    ]

    expected_case_findings = truth.expected_case_findings

    case_findings_match = (
        sorted(f.value for f in actual_case_findings)
        == sorted(f.value for f in expected_case_findings)
    )

    if not case_findings_match:
        all_correct = False

    # ------------------------------------------------------------
    # Determine human and Reckon overall verdicts
    # ------------------------------------------------------------

    human_pass = truth.polarity == "faithful"

    # A Reckon case is considered passing only when all expected
    # steps passed and there are no findings.
    reckon_pass = (
        all(verdict.passed for verdict in audit.step_verdicts)
        and len(audit.findings) == 0
    )

    false_pass = (not human_pass) and reckon_pass
    false_fail = human_pass and (not reckon_pass)

    return {
        "case_id": truth.case_id,
        "polarity": truth.polarity,
        "correct": all_correct,
        "human_pass": human_pass,
        "reckon_pass": reckon_pass,
        "false_pass": false_pass,
        "false_fail": false_fail,
        "fidelity_score": audit.fidelity_score,
        "findings": audit.findings,
    }

def load_parser_case(case_dir: Path):
    """Load one parser golden-set case."""

    scenario_text = (case_dir / "scenario.txt").read_text(
        encoding="utf-8"
    )

    expected = ExpectedTrace.model_validate_json(
        (case_dir / "expected.json").read_text(
            encoding="utf-8"
        )
    )

    return scenario_text, expected

def _normalize_text(value: str | None) -> str | None:
    """Normalize semantic text for parser evaluation."""

    if value is None:
        return None

    value = value.strip().lower()

    # Ignore punctuation differences.
    value = re.sub(r"[^\w\s]", "", value)

    return " ".join(value.split())

def compare_parser_trace(
    expected: ExpectedTrace,
    actual: ExpectedTrace,
) -> list[dict]:
    """Compare parser output against the golden ExpectedTrace.

    Returns one mismatch record per semantic difference.
    An empty list means the parser case passed.
    """

    mismatches: list[dict] = []

    if len(actual.steps) != len(expected.steps):
        mismatches.append(
            {
                "type": "step_count",
                "expected": len(expected.steps),
                "actual": len(actual.steps),
            }
        )

        return mismatches
    # Only compare positions that exist in both traces.
    for expected_step, actual_step in zip(
        expected.steps,
        actual.steps,
    ):
        step_index = expected_step.index

        if actual_step.index != expected_step.index:
            mismatches.append(
                {
                    "type": "index",
                    "step": step_index,
                    "expected": expected_step.index,
                    "actual": actual_step.index,
                }
            )

        if actual_step.action != expected_step.action:
            mismatches.append(
                {
                    "type": "action",
                    "step": step_index,
                    "expected": expected_step.action.value,
                    "actual": actual_step.action.value,
                }
            )

        # A missing golden target means the target is not constrained.
        # However, when the golden trace specifies a target, it must match.
        if (
            expected_step.target is not None
            and _normalize_text(actual_step.target)
            != _normalize_text(expected_step.target)
        ):
            mismatches.append(
                {
                    "type": "target",
                    "step": step_index,
                    "expected": expected_step.target,
                    "actual": actual_step.target,
                }
            )

        # Compare input values using the same normalization as production.
        if (
            _normalize_text(actual_step.input_value)
            != _normalize_text(expected_step.input_value)
        ):
            mismatches.append(
                {
                    "type": "input_value",
                    "step": step_index,
                    "expected": expected_step.input_value,
                    "actual": actual_step.input_value,
                }
            )

        # Compare assertions semantically rather than by exact formatting.
        if (
            _normalize_text(actual_step.assertion)
            != _normalize_text(expected_step.assertion)
        ):
            mismatches.append(
                {
                    "type": "assertion",
                    "step": step_index,
                    "expected": expected_step.assertion,
                    "actual": actual_step.assertion,
                }
            )

        # Preconditions must match exactly.
        if actual_step.preconditions != expected_step.preconditions:
            mismatches.append(
                {
                    "type": "preconditions",
                    "step": step_index,
                    "expected": expected_step.preconditions,
                    "actual": actual_step.preconditions,
                }
            )

        # Raw text should be preserved.
        if actual_step.raw_text != expected_step.raw_text:
            mismatches.append(
                {
                    "type": "raw_text",
                    "step": step_index,
                    "expected": expected_step.raw_text,
                    "actual": actual_step.raw_text,
                }
            )

    return mismatches

def evaluate_parser_case(case_dir: Path) -> dict:
    """Run Stage 1 and compare it against parser golden truth."""

    scenario_text, expected = load_parser_case(case_dir)

    actual = parse_scenario(scenario_text)

    mismatches = compare_parser_trace(
        expected,
        actual,
    )

    return {
        "case_id": case_dir.name,
        "correct": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def run_aligner_eval() -> bool:
    """Run Layer A — aligner golden-set evaluation."""

    case_dirs = sorted(
        path
        for path in ALIGNER_EVAL_DIR.iterdir()
        if path.is_dir()
    )

    if not case_dirs:
        print("No aligner evaluation cases found.")
        return False

    results = []
    errors = 0

    print("=" * 70)
    print("LAYER A - ALIGNER GOLDEN SET")
    print("=" * 70)
    print()

    for case_dir in case_dirs:
        try:
            result = evaluate_aligner_case(case_dir)
            results.append(result)

            status = "PASS" if result["correct"] else "FAIL"

            print(
                f"{result['case_id']:<30} "
                f"{status:<6} "
                f"human={'PASS' if result['human_pass'] else 'FAIL':<4} "
                f"reckon={'PASS' if result['reckon_pass'] else 'FAIL':<4} "
                f"score={result['fidelity_score']:.0%}"
            )

            if result["false_pass"]:
                print("  -> FALSE PASS")

            if result["false_fail"]:
                print("  -> FALSE FAIL")

            if not result["correct"]:
                finding_names = [
                    finding.type.value
                    for finding in result["findings"]
                ]
                print(f"  -> findings: {finding_names}")
        except Exception as exc:
            errors += 1
            print(f"{case_dir.name:<30} ERROR")
            print(f"  -> {type(exc).__name__}: {exc}")

    total = len(case_dirs)
    correct = sum(result["correct"] for result in results)
    incorrect = total - correct
    false_passes = sum(result["false_pass"] for result in results)
    false_fails = sum(result["false_fail"] for result in results)

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Cases:          {total}")
    print(f"Correct:        {correct}")
    print(f"Incorrect:      {incorrect}")
    print(f"Errors:         {errors}")
    print(f"False passes:   {false_passes}")
    print(f"False fails:    {false_fails}")
    print()

    if false_passes == 0:
        print("False-pass target: PASS (0 false passes)")
    else:
        print(
            f"False-pass target: FAIL "
            f"({false_passes} false pass(es))"
        )
    return false_passes == 0 and errors == 0


def run_parser_eval(
    runs: int = DEFAULT_PARSER_RUNS,
    retries: int = DEFAULT_PARSER_RETRIES,
) -> bool:
    """Run Layer B — parser golden-set evaluation."""

    case_dirs = sorted(
        path
        for path in PARSER_EVAL_DIR.iterdir()
        if path.is_dir()
    )

    if not case_dirs:
        print("No parser evaluation cases found.")
        return False

    case_results = []

    print("=" * 70)
    print("LAYER B - PARSER GOLDEN SET")
    print("=" * 70)
    print()

    errors = 0

    for case_dir in case_dirs:
        case_runs = []

        # --------------------------------------------------------
        # Initial run(s)
        # --------------------------------------------------------

        for run_number in range(1, runs + 1):
            try:
                result = evaluate_parser_case(case_dir)
                case_runs.append(result)

            except Exception as exc:
                errors += 1
                print(
                    f"{case_dir.name:<30} "
                    f"ERROR on run {run_number}"
                )
                print(
                    f"  -> {type(exc).__name__}: {exc}"
                )

        # --------------------------------------------------------
        # Cheap mode: retry only failed cases
        # --------------------------------------------------------

        if runs == 1 and case_runs and not case_runs[0]["correct"]:
            for retry_number in range(1, retries + 1):
                run_number = len(case_runs) + 1

                try:
                    result = evaluate_parser_case(case_dir)
                    case_runs.append(result)

                except Exception as exc:
                    errors += 1
                    print(
                        f"{case_dir.name:<30} "
                        f"ERROR on retry {retry_number}"
                    )
                    print(
                        f"  -> {type(exc).__name__}: {exc}"
                    )

        if not case_runs:
            continue

        passed_runs = sum(
            result["correct"]
            for result in case_runs
        )

        total_runs = len(case_runs)

        # A case is stable only when every run passed.
        stable = passed_runs == total_runs

        # A case is flaky when at least one run passed and
        # at least one run failed.
        flaky = (
            passed_runs > 0
            and passed_runs < total_runs
        )

        case_results.append(
            {
                "case_id": case_dir.name,
                "passed_runs": passed_runs,
                "total_runs": total_runs,
                "stable": stable,
                "flaky": flaky,
                "runs": case_runs,
            }
        )

        if stable:
            status = "PASS"
        elif flaky:
            status = "FLAKY"
        else:
            status = "FAIL"

        print(
            f"{case_dir.name:<30} "
            f"{status:<6} "
            f"{passed_runs}/{total_runs} "
            f"({passed_runs / total_runs:.0%})"
        )
        if total_runs > 1:
            for run_number, result in enumerate(
                case_runs,
                start=1,
            ):
                print(
                    f"  -> run {run_number}: "
                    f"{'PASS' if result['correct'] else 'FAIL'}"
                )

        # --------------------------------------------------------
        # Show mismatches from failed runs
        # --------------------------------------------------------

        for run_number, result in enumerate(
            case_runs,
            start=1,
        ):
            if not result["correct"]:
                for mismatch in result["mismatches"]:
                    mismatch_type = mismatch["type"]

                    if mismatch_type == "step_count":
                        print(
                            f"  -> run {run_number} "
                            f"step_count: "
                            f"expected={mismatch['expected']!r}, "
                            f"actual={mismatch['actual']!r}"
                        )
                    else:
                        print(
                            f"  -> run {run_number} "
                            f"step {mismatch['step']} "
                            f"{mismatch_type}: "
                            f"expected={mismatch['expected']!r}, "
                            f"actual={mismatch['actual']!r}"
                        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    total_cases = len(case_results)

    stable_cases = sum(
        result["stable"]
        for result in case_results
    )

    flaky_cases = sum(
        result["flaky"]
        for result in case_results
    )

    failed_cases = sum(
        result["passed_runs"] == 0
        for result in case_results
    )

    total_runs = sum(
        result["total_runs"]
        for result in case_results
    )

    passed_runs = sum(
        result["passed_runs"]
        for result in case_results
    )

    run_pass_rate = (
        passed_runs / total_runs
        if total_runs
        else 0
    )

    # Count mismatches across all runs.
    mismatch_counts: dict[str, int] = {}

    for case_result in case_results:
        for result in case_result["runs"]:
            for mismatch in result["mismatches"]:
                mismatch_type = mismatch["type"]

                mismatch_counts[mismatch_type] = (
                    mismatch_counts.get(mismatch_type, 0) + 1
                )

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(f"Cases:              {total_cases}")
    print(f"Stable cases:       {stable_cases}")
    print(f"Flaky cases:        {flaky_cases}")
    print(f"Failed cases:       {failed_cases}")
    print(f"Errors:             {errors}")
    print(f"Runs:               {total_runs}")
    print(f"Passed runs:        {passed_runs}")

    if total_runs:
        print(f"Run pass rate:      {run_pass_rate:.0%}")

    print()

    if mismatch_counts:
        print("Mismatch counts:")

        for mismatch_type, count in sorted(
            mismatch_counts.items()
        ):
            print(
                f"  {mismatch_type:<15} {count}"
            )
    else:
        print("No semantic mismatches found.")

    # Evaluation succeeds only when every case is stable
    # and there were no execution errors.
    return (
        total_cases == len(case_dirs)
        and flaky_cases == 0
        and failed_cases == 0
        and errors == 0
    )


def main():
    args = sys.argv[1:]

    if not args:
        layer = "all"
        parser_runs = DEFAULT_PARSER_RUNS
    elif args[0].lower() in {"aligner", "parser", "all"}:
        layer = args[0].lower()

        parser_runs = DEFAULT_PARSER_RUNS

        if len(args) == 3 and args[1] == "--runs":
            try:
                parser_runs = int(args[2])
            except ValueError:
                raise SystemExit(
                    "--runs must be a positive integer"
                )

            if parser_runs < 1:
                raise SystemExit(
                    "--runs must be a positive integer"
                )

        elif len(args) != 1:
            raise SystemExit(
                "Usage: python scripts/run_eval.py "
                "[aligner|parser|all] [--runs N]"
            )
    else:
        raise SystemExit(
            f"Unknown evaluation layer: {args[0]!r}\n"
            "Expected one of: aligner, parser, all"
        )

    success = True

    if layer == "aligner":
        success = run_aligner_eval()

    elif layer == "parser":
        if parser_runs == 1:
            success = run_parser_eval(
                runs=1,
                retries=DEFAULT_PARSER_RETRIES,
            )
        else:
            success = run_parser_eval(
                runs=parser_runs,
                retries=0,
            )

    elif layer == "all":
        aligner_ok = run_aligner_eval()
        print()

        parser_ok = run_parser_eval(
            runs=parser_runs,
            retries=(
                DEFAULT_PARSER_RETRIES
                if parser_runs == 1
                else 0
            ),
        )

        success = aligner_ok and parser_ok

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
