from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from eval.schema import CaseTruth
from reckon.aligner import align
from reckon.schema import ExpectedTrace, ObservedTrace


EVAL_DIR = PROJECT_ROOT / "eval" / "aligner"


def _load_case(case_dir: Path):
    """Load one Layer-A golden-set case."""
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


CASE_DIRS = sorted(
    path
    for path in EVAL_DIR.iterdir()
    if path.is_dir()
)


@pytest.mark.parametrize(
    "case_dir",
    CASE_DIRS,
    ids=lambda path: path.name,
)
def test_aligner_golden_case(case_dir: Path):
    """Aligner output must exactly match human-authored ground truth."""

    expected, observed, truth = _load_case(case_dir)

    audit = align(expected, observed)

    # ---------------------------------------------------------
    # Check every expected step
    # ---------------------------------------------------------

    for step_truth in truth.steps:
        actual = next(
            (
                verdict
                for verdict in audit.step_verdicts
                if verdict.expected_step_index == step_truth.index
            ),
            None,
        )

        assert actual is not None, (
            f"{truth.case_id}: no verdict produced "
            f"for expected step {step_truth.index}"
        )

        assert actual.passed == step_truth.should_pass, (
            f"{truth.case_id}, step {step_truth.index}: "
            f"expected passed={step_truth.should_pass}, "
            f"got passed={actual.passed}"
        )

        expected_findings = sorted(
            finding.value
            for finding in step_truth.expected_finding_types
        )

        actual_findings = sorted(
            finding.value
            for finding in actual.finding_types
        )

        assert actual_findings == expected_findings, (
            f"{truth.case_id}, step {step_truth.index}: "
            f"expected findings={expected_findings}, "
            f"got findings={actual_findings}"
        )

    # ---------------------------------------------------------
    # Check case-level findings
    # ---------------------------------------------------------

    actual_case_findings = sorted(
        finding.type.value
        for finding in audit.findings
        if finding.expected_step_index is None
    )

    expected_case_findings = sorted(
        finding.value
        for finding in truth.expected_case_findings
    )

    assert actual_case_findings == expected_case_findings, (
        f"{truth.case_id}: "
        f"expected case findings={expected_case_findings}, "
        f"got findings={actual_case_findings}"
    )

    # ---------------------------------------------------------
    # Check overall PASS / FAIL
    # ---------------------------------------------------------

    human_pass = truth.polarity == "faithful"

    reckon_pass = (
        all(verdict.passed for verdict in audit.step_verdicts)
        and len(audit.findings) == 0
    )

    assert reckon_pass == human_pass, (
        f"{truth.case_id}: "
        f"human={'PASS' if human_pass else 'FAIL'}, "
        f"reckon={'PASS' if reckon_pass else 'FAIL'}"
    )