import pytest

from reckon.equivalence.grounding import citation_matches
from reckon.intent_binding import bind_reported_intent, diagnostics_for_final_steps
from reckon.intent_contract import ReportedIntent
from reckon.schema import (
    ActionType,
    ExpectedStep,
    IntentDiagnostic,
    IntentOperation,
    IntentParseStatus,
    IntentRejectionReason,
    ObjectKind,
    StepSource,
)

SIGN_IN = "Click the **Sign in** button."


def _bind(scenario: str, span: tuple[int, int] | None = (-1, -1), **fields: str | None):
    resolved = (0, len(scenario)) if span == (-1, -1) else span
    return bind_reported_intent(scenario, resolved, ReportedIntent(**fields), step_index=1)


def _sign_in(**overrides: str | None):
    fields = {
        "operation": "activate",
        "operation_text": "Click",
        "kind": "button",
        "kind_text": "button",
        "name": "Sign in",
    }
    fields.update(overrides)
    return _bind(SIGN_IN, **fields)


def test_bind_reported_intent_binds_a_simple_click_intent():
    outcome = _sign_in()

    assert outcome.intent is not None
    assert (outcome.intent.operation, outcome.intent.object.kind, outcome.intent.object.name) == (
        IntentOperation.activate, ObjectKind.button, "Sign in",
    )
    assert outcome.diagnostic.status == IntentParseStatus.bound
    assert outcome.diagnostic.reason is None


@pytest.mark.parametrize("operation", ["Activate", "ACTIVATE", "  activate  "])
def test_bind_reported_intent_normalises_operation_casing(operation: str):
    outcome = _sign_in(operation=operation)

    assert outcome.intent is not None
    assert outcome.intent.operation == IntentOperation.activate


@pytest.mark.parametrize("kind", ["Button", "BUTTON"])
def test_bind_reported_intent_normalises_kind_casing(kind: str):
    outcome = _sign_in(kind=kind)

    assert outcome.intent is not None
    assert outcome.intent.object.kind == ObjectKind.button


@pytest.mark.parametrize("operation", [m for m in IntentOperation if m != IntentOperation.unspecified])
def test_bind_reported_intent_accepts_every_supported_operation(operation: IntentOperation):
    scenario = f"Do the thing, {operation.value} the Save control."

    outcome = _bind(
        scenario,
        operation=operation.value,
        operation_text=operation.value,
        kind="control",
        kind_text="control",
        name="Save",
    )

    assert outcome.intent is not None
    assert outcome.intent.operation == operation


@pytest.mark.parametrize("kind", [m for m in ObjectKind if m != ObjectKind.unspecified])
def test_bind_reported_intent_accepts_every_supported_kind(kind: ObjectKind):
    scenario = f"Click the Save {kind.value}."

    outcome = _bind(
        scenario,
        operation="activate",
        operation_text="Click",
        kind=kind.value,
        kind_text=kind.value,
        name="Save",
    )

    assert outcome.intent is not None
    assert outcome.intent.object.kind == kind


@pytest.mark.parametrize("operation", ["click", "press", "unspecified", "activate it", ""])
def test_bind_reported_intent_rejects_unsupported_operation(operation: str):
    outcome = _sign_in(operation=operation)

    assert outcome.intent is None
    assert outcome.diagnostic.status == IntentParseStatus.rejected
    assert outcome.diagnostic.reason in (
        IntentRejectionReason.unsupported_operation,
        IntentRejectionReason.missing_field,
    )


def test_bind_reported_intent_reports_unsupported_operation_with_the_offered_token():
    outcome = _sign_in(operation="Click")

    assert outcome.diagnostic.reason == IntentRejectionReason.unsupported_operation
    assert (outcome.diagnostic.slot, outcome.diagnostic.offered_operation) == ("operation", "click")


@pytest.mark.parametrize("kind", ["website", "textbox", "input", "unspecified", "menu item"])
def test_bind_reported_intent_rejects_unsupported_kind(kind: str):
    outcome = _sign_in(kind=kind)

    assert outcome.intent is None
    assert outcome.diagnostic.reason == IntentRejectionReason.unsupported_kind
    assert outcome.diagnostic.slot == "kind"


@pytest.mark.parametrize("slot", ["operation", "operation_text", "kind", "kind_text"])
def test_bind_reported_intent_rejects_a_missing_required_field(slot: str):
    outcome = _sign_in(**{slot: None})

    assert outcome.diagnostic.reason == IntentRejectionReason.missing_field
    assert outcome.diagnostic.slot == slot


def test_bind_reported_intent_treats_a_blank_required_field_as_missing():
    outcome = _sign_in(kind_text="   ")

    assert outcome.diagnostic.reason == IntentRejectionReason.missing_field


def test_bind_reported_intent_treats_a_missing_intent_as_omitted():
    outcome = bind_reported_intent(SIGN_IN, (0, len(SIGN_IN)), None, step_index=3)

    assert (outcome.intent, outcome.diagnostic.status, outcome.diagnostic.reason) == (
        None, IntentParseStatus.omitted, None,
    )


def test_bind_reported_intent_treats_an_empty_intent_as_omitted():
    outcome = _bind(SIGN_IN)

    assert outcome.diagnostic.status == IntentParseStatus.omitted


def test_bind_reported_intent_keeps_omission_distinct_from_rejection():
    omitted = _bind(SIGN_IN)
    rejected = _sign_in(kind="website")

    assert omitted.diagnostic.status != rejected.diagnostic.status


def test_bind_reported_intent_binds_citation_across_extra_whitespace():
    scenario = "Click   the\n  Sign  in   button."

    outcome = _bind(
        scenario, operation="activate", operation_text="Click",
        kind="button", kind_text="button", name="Sign in",
    )

    assert outcome.intent is not None
    assert outcome.intent.object.name_citation.text == "Sign  in"


def test_bind_reported_intent_binds_citation_that_omits_markdown_emphasis():
    outcome = _sign_in(name="Sign in button")

    assert outcome.intent is not None
    assert outcome.intent.object.name_citation.text == "Sign in** button"


@pytest.mark.parametrize("cited", ["See prices", '"See prices"', "\u201cSee prices\u201d", "See prices."])
def test_bind_reported_intent_binds_citation_despite_quotes_and_edge_punctuation(cited: str):
    scenario = 'Click "See prices".'

    outcome = _bind(
        scenario, operation="activate", operation_text="Click",
        kind="control", kind_text="Click", name=cited,
    )

    assert outcome.intent is not None
    assert outcome.intent.object.name_citation.text == "See prices"


def test_bind_reported_intent_binds_citations_that_are_exact_source_slices():
    outcome = _sign_in(name="Sign in button")

    assert outcome.intent is not None
    citations = [
        outcome.intent.operation_citation,
        outcome.intent.object.kind_citation,
        outcome.intent.object.name_citation,
    ]
    assert all(citation_matches(SIGN_IN, citation) for citation in citations)


def test_bind_reported_intent_rejects_a_citation_that_is_not_in_the_sentence():
    outcome = _sign_in(name="Log out")

    assert outcome.diagnostic.reason == IntentRejectionReason.invalid_citation
    assert outcome.diagnostic.slot == "name"


def test_bind_reported_intent_rejects_a_citation_that_differs_in_case():
    outcome = _sign_in(name="sign in")

    assert outcome.diagnostic.reason == IntentRejectionReason.invalid_citation


def test_bind_reported_intent_rejects_a_citation_that_differs_in_inner_punctuation():
    outcome = _sign_in(name="Sign, in")

    assert outcome.diagnostic.reason == IntentRejectionReason.invalid_citation


def test_bind_reported_intent_rejects_a_citation_that_is_only_punctuation():
    outcome = _sign_in(name="**")

    assert outcome.diagnostic.reason == IntentRejectionReason.invalid_citation


def test_bind_reported_intent_rejects_an_ambiguous_citation():
    scenario = "Click Sign in, then wait for Sign in."

    outcome = _bind(
        scenario, operation="activate", operation_text="Click",
        kind="control", kind_text="Click", name="Sign in",
    )

    assert outcome.diagnostic.reason == IntentRejectionReason.ambiguous_citation
    assert outcome.diagnostic.slot == "name"


def test_bind_reported_intent_rejects_a_citation_found_only_in_another_sentence():
    scenario = "1. Click Save.\n2. Open Reports.\n"
    first_line = (3, len("1. Click Save."))

    outcome = _bind(
        scenario, first_line, operation="activate", operation_text="Click",
        kind="control", kind_text="Save", name="Reports",
    )

    assert outcome.diagnostic.reason == IntentRejectionReason.invalid_citation


def test_bind_reported_intent_rejects_when_the_step_sentence_is_unknown():
    outcome = _sign_in_with_span(None)

    assert outcome.diagnostic.reason == IntentRejectionReason.step_span_unavailable


def test_bind_reported_intent_reports_a_binding_failure_for_an_out_of_range_span():
    outcome = _sign_in_with_span((0, len(SIGN_IN) + 50))

    assert outcome.diagnostic.reason == IntentRejectionReason.binding_failed


def _sign_in_with_span(span: tuple[int, int] | None):
    return bind_reported_intent(
        SIGN_IN,
        span,
        ReportedIntent(
            operation="activate", operation_text="Click", kind="button", kind_text="button"
        ),
        step_index=1,
    )


def test_bind_reported_intent_records_no_scenario_text_in_a_rejection():
    outcome = _sign_in(name="hunter2-secret")

    assert "hunter2-secret" not in outcome.diagnostic.model_dump_json()


def test_bind_reported_intent_records_no_free_text_offered_as_a_kind():
    outcome = _sign_in(kind="password=hunter2!")

    assert outcome.diagnostic.reason == IntentRejectionReason.unsupported_kind
    assert "hunter2" not in outcome.diagnostic.model_dump_json()


def _step(index: int, raw_text: str, source: StepSource = StepSource.scenario) -> ExpectedStep:
    return ExpectedStep(
        index=index, action=ActionType.click, raw_text=raw_text, source=source
    )


def _diagnostic(index: int, status: IntentParseStatus = IntentParseStatus.omitted):
    return IntentDiagnostic(step_index=index, status=status)


def test_diagnostics_for_final_steps_rekeys_after_macro_expansion():
    pending = [("Open app", _diagnostic(1)), ("Sign in", _diagnostic(2, IntentParseStatus.bound))]
    steps = [
        _step(1, "Expanded", StepSource.macro),
        _step(2, "Open app"),
        _step(3, "Sign in"),
    ]

    result = diagnostics_for_final_steps(pending, steps)

    assert [(d.step_index, d.status) for d in result] == [
        (2, IntentParseStatus.omitted), (3, IntentParseStatus.bound),
    ]


def test_diagnostics_for_final_steps_pairs_repeated_sentences_in_order():
    pending = [
        ("Click Next", _diagnostic(1, IntentParseStatus.bound)),
        ("Click Next", _diagnostic(2)),
    ]
    steps = [_step(1, "Click Next"), _step(2, "Click Next")]

    result = diagnostics_for_final_steps(pending, steps)

    assert [(d.step_index, d.status) for d in result] == [
        (1, IntentParseStatus.bound), (2, IntentParseStatus.omitted),
    ]
