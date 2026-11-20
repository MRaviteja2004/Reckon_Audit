import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from reckon.schema import (
    ActionType,
    AssertionEvidence,
    CrossCheckAnswer,
    CrossCheckKind,
    CrossCheckTarget,
    CrossCheckVerdict,
    ExpectedStep,
    ExpectedTrace,
    IntentDiagnostic,
    IntentParseReport,
    IntentParseStatus,
    IntentRejectionReason,
    InteractionParseStatus,
    InteractionRejectionReason,
    ObservedAction,
    ObservedInteractionDiagnostic,
    ObservedTrace,
    merge_observed_chunks,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# ExpectedTrace
# ---------------------------------------------------------------------------

def test_expected_trace_loads_from_fixture():
    data = json.loads((FIXTURES / "login_expected.json").read_text())
    trace = ExpectedTrace.model_validate(data)
    assert trace.scenario_title == "Login with invalid password shows error message"
    assert len(trace.steps) == 5
    assert trace.steps[0].action == ActionType.navigate
    assert trace.steps[4].assertion == "Invalid email or password"


def test_expected_step_requires_index():
    with pytest.raises(ValidationError):
        ExpectedStep.model_validate({"action": "click"})


def test_expected_step_requires_action():
    with pytest.raises(ValidationError):
        ExpectedStep.model_validate({"index": 1})


def test_expected_trace_rejects_out_of_order_steps():
    with pytest.raises(ValidationError, match="in order"):
        ExpectedTrace.model_validate({
            "scenario_title": "test",
            "steps": [
                {"index": 2, "action": "click"},
                {"index": 1, "action": "navigate"},
            ],
        })


def test_expected_trace_rejects_duplicate_indices():
    with pytest.raises(ValidationError, match="in order"):
        ExpectedTrace.model_validate({
            "scenario_title": "test",
            "steps": [
                {"index": 1, "action": "click"},
                {"index": 1, "action": "navigate"},
            ],
        })


def test_index_must_be_positive():
    with pytest.raises(ValidationError):
        ExpectedStep.model_validate({"index": 0, "action": "click"})


# ---------------------------------------------------------------------------
# ObservedTrace
# ---------------------------------------------------------------------------

def test_observed_broken_fixture_loads():
    data = json.loads((FIXTURES / "login_observed_broken.json").read_text())
    trace = ObservedTrace.model_validate(data)
    assert len(trace.actions) == 5
    assert all(a.timestamp_ms >= 0 for a in trace.actions)
    assert all(0.0 <= a.confidence <= 1.0 for a in trace.actions)
    assert trace.actions[-1].assertion_evidence is not None


def test_observed_passing_fixture_loads():
    data = json.loads((FIXTURES / "login_observed_passing.json").read_text())
    trace = ObservedTrace.model_validate(data)
    assert len(trace.actions) == 5
    last = trace.actions[-1]
    assert last.action == ActionType.assert_text
    assert last.assertion_evidence is not None
    assert last.assertion_evidence.visible_text == "Invalid email or password"


def test_clip_offset_defaults_to_zero():
    trace = ObservedTrace(source_file="test.mp4", actions=[])
    assert trace.clip_offset_ms == 0


def test_clip_end_ms_defaults_to_none():
    trace = ObservedTrace(source_file="test.mp4", actions=[])
    assert trace.clip_end_ms is None


def test_clip_end_must_be_after_offset():
    with pytest.raises(ValidationError, match="clip_end_ms"):
        ObservedTrace(source_file="test.mp4", clip_offset_ms=5000, clip_end_ms=5000, actions=[])


def test_clip_offset_must_be_non_negative():
    with pytest.raises(ValidationError):
        ObservedTrace(source_file="test.mp4", clip_offset_ms=-1, actions=[])


def test_timestamp_must_be_non_negative():
    with pytest.raises(ValidationError):
        ObservedAction.model_validate({
            "index": 1,
            "timestamp_ms": -1,
            "action": "click",
            "confidence": 0.9,
        })


# ---------------------------------------------------------------------------
# Blank-string normalisation (LLM emits "" for absent optional fields)
# ---------------------------------------------------------------------------

def test_expected_step_blank_target_becomes_none():
    step = ExpectedStep.model_validate({"index": 1, "action": "click", "target": ""})
    assert step.target is None


def test_expected_step_whitespace_assertion_becomes_none():
    step = ExpectedStep.model_validate({"index": 1, "action": "assert_text", "assertion": "   "})
    assert step.assertion is None


def test_expected_step_non_blank_target_is_preserved():
    step = ExpectedStep.model_validate({"index": 1, "action": "click", "target": "Submit"})
    assert step.target == "Submit"


def test_observed_action_blank_input_value_becomes_none():
    action = ObservedAction.model_validate({
        "index": 1, "timestamp_ms": 1000, "action": "type_text",
        "input_value": "", "confidence": 0.9,
    })
    assert action.input_value is None


def test_assertion_evidence_blank_visible_text_becomes_none():
    ev = AssertionEvidence.model_validate({"description": "Error shown", "visible_text": "  "})
    assert ev.visible_text is None


# ---------------------------------------------------------------------------
# ObservedAction — confidence validation
# ---------------------------------------------------------------------------

def test_observed_action_confidence_rejects_above_one():
    with pytest.raises(ValidationError):
        ObservedAction.model_validate({
            "index": 1, "timestamp_ms": 1000, "action": "click", "confidence": 1.5,
        })


def test_observed_action_confidence_rejects_below_zero():
    with pytest.raises(ValidationError):
        ObservedAction.model_validate({
            "index": 1, "timestamp_ms": 1000, "action": "click", "confidence": -0.1,
        })


def test_confidence_rejects_nan():
    with pytest.raises(ValidationError):
        ObservedAction.model_validate({
            "index": 1, "timestamp_ms": 1000, "action": "click", "confidence": math.nan,
        })


def test_confidence_rejects_inf():
    with pytest.raises(ValidationError):
        ObservedAction.model_validate({
            "index": 1, "timestamp_ms": 1000, "action": "click", "confidence": math.inf,
        })


# ---------------------------------------------------------------------------
# merge_observed_chunks
# ---------------------------------------------------------------------------

def _make_chunk(
    source: str,
    offset_ms: int,
    end_ms: int,
    action_timestamps: list[int],
) -> ObservedTrace:
    actions = [
        ObservedAction(
            index=i,
            timestamp_ms=ts,
            action=ActionType.click,
            confidence=0.9,
        )
        for i, ts in enumerate(action_timestamps, start=1)
    ]
    return ObservedTrace(
        source_file=source,
        clip_offset_ms=offset_ms,
        clip_end_ms=end_ms,
        actions=actions,
    )


def test_merge_chunks_converts_timestamps_to_absolute():
    chunk1 = _make_chunk("rec.mp4", offset_ms=0, end_ms=30_000, action_timestamps=[1000, 5000])
    chunk2 = _make_chunk("rec.mp4", offset_ms=30_000, end_ms=60_000, action_timestamps=[2000, 8000])
    merged = merge_observed_chunks([chunk1, chunk2])
    timestamps = [a.timestamp_ms for a in merged.actions]
    assert timestamps == [1000, 5000, 32_000, 38_000]


def test_merge_chunks_reindexes_sequentially():
    chunk1 = _make_chunk("rec.mp4", offset_ms=0, end_ms=30_000, action_timestamps=[1000, 2000])
    chunk2 = _make_chunk("rec.mp4", offset_ms=30_000, end_ms=60_000, action_timestamps=[1000])
    merged = merge_observed_chunks([chunk1, chunk2])
    assert [a.index for a in merged.actions] == [1, 2, 3]


def test_merge_chunks_sorts_by_offset():
    chunk1 = _make_chunk("rec.mp4", offset_ms=0, end_ms=30_000, action_timestamps=[1000])
    chunk2 = _make_chunk("rec.mp4", offset_ms=30_000, end_ms=60_000, action_timestamps=[500])
    merged = merge_observed_chunks([chunk2, chunk1])  # deliberately out of order
    assert merged.actions[0].timestamp_ms == 1000
    assert merged.actions[1].timestamp_ms == 30_500


def test_merge_chunks_rejects_mixed_source_files():
    chunk1 = _make_chunk("rec_a.mp4", offset_ms=0, end_ms=30_000, action_timestamps=[1000])
    chunk2 = _make_chunk("rec_b.mp4", offset_ms=30_000, end_ms=60_000, action_timestamps=[1000])
    with pytest.raises(ValueError, match="source_file"):
        merge_observed_chunks([chunk1, chunk2])


def test_merge_chunks_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one chunk"):
        merge_observed_chunks([])


def test_merge_single_chunk_is_identity():
    chunk = _make_chunk("rec.mp4", offset_ms=0, end_ms=30_000, action_timestamps=[1000, 5000])
    merged = merge_observed_chunks([chunk])
    assert [a.timestamp_ms for a in merged.actions] == [1000, 5000]


# ---------------------------------------------------------------------------
# CrossCheck models
# ---------------------------------------------------------------------------

def test_crosscheck_verdict_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        CrossCheckVerdict(
            question="q", answer=CrossCheckAnswer.present, confidence=1.5,
            rationale="r",
        )


def test_crosscheck_verdict_defaults_frame_timestamps_to_empty():
    verdict = CrossCheckVerdict(
        question="q", answer=CrossCheckAnswer.absent, confidence=0.9, rationale="r",
    )
    assert verdict.frame_timestamps_ms == []


def test_crosscheck_target_rejects_unordered_window():
    with pytest.raises(ValidationError, match="window_end_ms"):
        CrossCheckTarget(
            kind=CrossCheckKind.assertion, question="q",
            window_start_ms=5000, window_end_ms=5000,
        )


def test_crosscheck_target_accepts_ordered_window():
    target = CrossCheckTarget(
        kind=CrossCheckKind.low_confidence, question="q",
        window_start_ms=1000, window_end_ms=3000, anchor_timestamp_ms=2000,
    )
    assert target.window_end_ms == 3000
    assert target.anchor_timestamp_ms == 2000


def test_crosscheck_target_rejects_anchor_outside_window():
    with pytest.raises(ValidationError, match="anchor_timestamp_ms"):
        CrossCheckTarget(
            kind=CrossCheckKind.assertion,
            question="q",
            window_start_ms=1000,
            window_end_ms=3000,
            anchor_timestamp_ms=4000,
        )


# ---------------------------------------------------------------------------
# Stage 1 intent report
# ---------------------------------------------------------------------------

def test_expected_trace_from_legacy_json_has_no_intent_report():
    data = json.loads((FIXTURES / "login_expected.json").read_text())
    assert "intent_report" not in data

    trace = ExpectedTrace.model_validate(data)

    assert trace.intent_report is None


def test_expected_trace_from_legacy_json_has_no_step_intents():
    data = json.loads((FIXTURES / "login_expected.json").read_text())

    trace = ExpectedTrace.model_validate(data)

    assert all(step.action_intent is None for step in trace.steps)


def test_expected_trace_round_trips_its_intent_report():
    data = json.loads((FIXTURES / "login_expected.json").read_text())
    trace = ExpectedTrace.model_validate(data)
    report = IntentParseReport(
        contract_version="2",
        diagnostics=[
            IntentDiagnostic(
                step_index=1,
                status=IntentParseStatus.rejected,
                reason=IntentRejectionReason.unsupported_kind,
                slot="kind",
            )
        ],
    )

    reloaded = ExpectedTrace.model_validate_json(
        trace.model_copy(update={"intent_report": report}).model_dump_json()
    )

    assert reloaded.intent_report == report


def test_intent_diagnostic_requires_a_reason_when_rejected():
    with pytest.raises(ValidationError):
        IntentDiagnostic(step_index=1, status=IntentParseStatus.rejected)


@pytest.mark.parametrize("status", [IntentParseStatus.bound, IntentParseStatus.omitted])
def test_intent_diagnostic_forbids_a_reason_unless_rejected(status: IntentParseStatus):
    with pytest.raises(ValidationError):
        IntentDiagnostic(
            step_index=1, status=status, reason=IntentRejectionReason.missing_field
        )


def test_intent_rejection_reasons_are_the_documented_set():
    assert {reason.value for reason in IntentRejectionReason} == {
        "missing_field", "unsupported_operation", "unsupported_kind",
        "step_span_unavailable", "invalid_citation", "ambiguous_citation",
        "binding_failed",
    }


# ---------------------------------------------------------------------------
# Stage 2 interaction diagnostics
# ---------------------------------------------------------------------------

def test_observed_trace_from_legacy_json_has_no_interaction_data():
    data = json.loads((FIXTURES / "login_observed_passing.json").read_text())

    trace = ObservedTrace.model_validate(data)

    assert all(
        (a.interaction, a.interaction_diagnostic) == (None, None) for a in trace.actions
    )


def test_observed_action_round_trips_its_interaction_diagnostic():
    diagnostic = ObservedInteractionDiagnostic(
        contract_version="1",
        status=InteractionParseStatus.rejected,
        reason=InteractionRejectionReason.unsupported_kind,
        slot="kind",
        offered_kind="textbox",
    )
    action = ObservedAction(
        index=1, timestamp_ms=0, action=ActionType.click, target="Sign in",
        confidence=0.9, interaction_diagnostic=diagnostic,
    )

    assert ObservedAction.model_validate_json(action.model_dump_json()) == action


def test_interaction_diagnostic_requires_a_reason_when_rejected():
    with pytest.raises(ValidationError):
        ObservedInteractionDiagnostic(contract_version="1", status=InteractionParseStatus.rejected)


@pytest.mark.parametrize("status", [InteractionParseStatus.bound, InteractionParseStatus.omitted])
def test_interaction_diagnostic_forbids_a_reason_unless_rejected(status: InteractionParseStatus):
    with pytest.raises(ValidationError):
        ObservedInteractionDiagnostic(
            contract_version="1", status=status, reason=InteractionRejectionReason.missing_field
        )


def test_interaction_rejection_reasons_are_the_documented_set():
    assert {reason.value for reason in InteractionRejectionReason} == {
        "missing_field", "unsupported_operation", "unsupported_kind",
        "unsupported_surface", "surface_unavailable", "operation_contradicts_action",
        "invalid_citation", "ambiguous_citation", "binding_failed",
    }
