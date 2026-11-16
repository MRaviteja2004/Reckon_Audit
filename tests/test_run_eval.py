from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reckon.schema import (
    ActionType,
    ExpectedStep,
    ExpectedTrace,
)
from scripts.run_eval import (
    compare_parser_trace,
    evaluate_parser_case,
)

def _make_trace(
    action=ActionType.click,
    target=None,
    assertion=None,
    input_value=None,
    raw_text=None,
    preconditions=None,
):
    return ExpectedTrace(
        scenario_title="Test Scenario",
        steps=[
            ExpectedStep(
                index=1,
                action=action,
                target=target,
                input_value=input_value,
                assertion=assertion,
                preconditions=preconditions,
                raw_text=raw_text,
            )
        ],
    )


def test_compare_parser_trace_returns_no_mismatches_for_identical_traces():
    expected = _make_trace(
        target="Login button",
        assertion="Login successful",
    )

    actual = _make_trace(
        target="Login button",
        assertion="Login successful",
    )

    assert compare_parser_trace(expected, actual) == []


def test_compare_parser_trace_reports_step_count_mismatch():
    expected = ExpectedTrace(
        scenario_title="Test",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Login",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_text,
                assertion="Success",
            ),
        ],
    )

    actual = ExpectedTrace(
        scenario_title="Test",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Login",
            )
        ],
    )

    mismatches = compare_parser_trace(expected, actual)

    assert mismatches == [
        {
            "type": "step_count",
            "expected": 2,
            "actual": 1,
        }
    ]
    assert mismatches[0]["expected"] == 2
    assert mismatches[0]["actual"] == 1


def test_compare_parser_trace_reports_action_mismatch():
    expected = _make_trace(action=ActionType.assert_text)
    actual = _make_trace(action=ActionType.assert_visible)

    mismatches = compare_parser_trace(expected, actual)

    assert len(mismatches) == 1
    assert mismatches[0]["type"] == "action"


def test_compare_parser_trace_reports_target_mismatch():
    expected = _make_trace(target="Login button")
    actual = _make_trace(target="Submit button")

    mismatches = compare_parser_trace(expected, actual)

    assert len(mismatches) == 1
    assert mismatches[0]["type"] == "target"
    assert mismatches[0]["expected"] == "Login button"
    assert mismatches[0]["actual"] == "Submit button"

def test_compare_parser_trace_target_match_is_case_insensitive():
    expected = _make_trace(target="Login Button")

    actual = _make_trace(target="login button")

    mismatches = compare_parser_trace(expected, actual)

    assert mismatches == []


def test_compare_parser_trace_reports_input_value_mismatch():
    expected = _make_trace(input_value="tomsmith")
    actual = _make_trace(input_value="admin")

    mismatches = compare_parser_trace(expected, actual)

    assert len(mismatches) == 1
    assert mismatches[0]["type"] == "input_value"
    assert mismatches[0]["expected"] == "tomsmith"
    assert mismatches[0]["actual"] == "admin"

def test_compare_parser_trace_input_value_match_is_case_insensitive():
    expected = _make_trace(input_value="TomSmith")

    actual = _make_trace(input_value="tomsmith")

    assert compare_parser_trace(expected, actual) == []


def test_compare_parser_trace_reports_assertion_mismatch():
    expected = _make_trace(
        assertion="Invalid username is displayed",
    )

    actual = _make_trace(
        assertion="Invalid username",
    )

    mismatches = compare_parser_trace(expected, actual)

    assert len(mismatches) == 1
    assert mismatches[0]["type"] == "assertion"
    assert (
        mismatches[0]["expected"]
        == "Invalid username is displayed"
    )
    assert mismatches[0]["actual"] == "Invalid username"

def test_compare_parser_trace_assertion_match_is_case_insensitive():
    expected = _make_trace(
        assertion="Invalid Username Is Displayed"
    )

    actual = _make_trace(
        assertion="invalid username is displayed"
    )

    assert compare_parser_trace(expected, actual) == []

def test_compare_parser_trace_raw_text_mismatch():
    expected = _make_trace(raw_text="Click the Login button")
    actual = _make_trace(raw_text="Click Login")

    mismatches = compare_parser_trace(expected, actual)

    assert mismatches == [
        {
            "type": "raw_text",
            "step": 1,
            "expected": "Click the Login button",
            "actual": "Click Login",
        }
    ]

def test_compare_parser_trace_preconditions_mismatch():
    expected = _make_trace(preconditions="User is logged out")
    actual = _make_trace(preconditions="User is logged in")

    mismatches = compare_parser_trace(expected, actual)

    assert mismatches == [
        {
            "type": "preconditions",
            "step": 1,
            "expected": "User is logged out",
            "actual": "User is logged in",
        }
    ]

def test_evaluate_parser_case_detects_step_count_mismatch(tmp_path):
    case_dir = tmp_path / "step_count_mismatch"
    case_dir.mkdir()

    scenario_text = (
        "Click the Login button and verify that Success is displayed."
    )

    (case_dir / "scenario.txt").write_text(
        scenario_text,
        encoding="utf-8",
    )

    expected = ExpectedTrace(
        scenario_title="Login success display",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Login button",
                raw_text=scenario_text,
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_text,
                assertion="Success is displayed.",
                raw_text=scenario_text,
            ),
        ],
    )

    (case_dir / "expected.json").write_text(
        expected.model_dump_json(),
        encoding="utf-8",
    )

    actual = ExpectedTrace(
        scenario_title="Login success display",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Login button",
                raw_text=scenario_text,
            ),
        ],
    )

    with patch(
        "scripts.run_eval.parse_scenario",
        return_value=actual,
    ):
        result = evaluate_parser_case(case_dir)

    assert result["correct"] is False
    assert result["mismatches"] == [
        {
            "type": "step_count",
            "expected": 2,
            "actual": 1,
        }
    ]
