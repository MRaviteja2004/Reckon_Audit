import pytest

from reckon.equivalence.grounding import citation_matches
from reckon.schema import (
    ActionType,
    CitationSurface,
    ElementKind,
    IntentOperation,
    InteractionParseStatus,
    InteractionRejectionReason,
)
from reckon.stage2.interaction_binding import bind_reported_interaction
from reckon.stage2.interaction_contract import (
    INTERACTION_CONTRACT_VERSION,
    OPERATIONS_FOR_ACTION,
    ReportedInteraction,
)

SIGN_IN = "Sign in button"


def _bind(
    action: ActionType = ActionType.click,
    *,
    target: str | None = SIGN_IN,
    input_value: str | None = None,
    visible_text: str | None = None,
    frame_index: int = 2,
    **fields: str | None,
):
    return bind_reported_interaction(
        ReportedInteraction(**{f"interaction_{k}": v for k, v in fields.items()}),
        action=action,
        surfaces={"target": target, "input_value": input_value, "visible_text": visible_text},
        frame_index=frame_index,
    )


def _sign_in(**overrides: str | None):
    fields = {
        "operation": "activate",
        "kind": "button",
        "kind_text": "button",
        "name": "Sign in",
        "surface": "target",
    }
    fields.update(overrides)
    return _bind(**fields)


def test_bind_reported_interaction_binds_a_click_on_a_named_button():
    outcome = _sign_in()

    element = outcome.interaction.element
    assert (outcome.interaction.operation, element.kind, element.name) == (
        IntentOperation.activate, ElementKind.button, "Sign in",
    )
    assert outcome.diagnostic.status == InteractionParseStatus.bound


def test_bind_reported_interaction_binds_to_the_cited_surface_and_frame():
    outcome = _sign_in()

    assert (outcome.interaction.citation_surface, outcome.interaction.cited_frame_index) == (
        CitationSurface.target, 2,
    )


def test_bind_reported_interaction_binds_citations_that_are_exact_surface_slices():
    outcome = _sign_in()

    element = outcome.interaction.element
    assert citation_matches(SIGN_IN, element.kind_citation)
    assert citation_matches(SIGN_IN, element.name_citation)


def test_bind_reported_interaction_stamps_the_contract_version():
    assert _sign_in().diagnostic.contract_version == INTERACTION_CONTRACT_VERSION


@pytest.mark.parametrize(
    ("action", "operation"),
    [(action, op) for action, ops in OPERATIONS_FOR_ACTION.items() for op in ops],
)
def test_bind_reported_interaction_accepts_every_operation_its_action_denotes(
    action: ActionType, operation: IntentOperation
):
    outcome = _bind(
        action, operation=operation.value, kind="control", kind_text="button",
        name="Sign in", surface="target",
    )

    assert outcome.interaction is not None
    assert outcome.interaction.operation == operation


@pytest.mark.parametrize("kind", [m for m in ElementKind if m != ElementKind.unspecified])
def test_bind_reported_interaction_accepts_every_supported_kind(kind: ElementKind):
    target = f"Save {kind.value}"

    outcome = _bind(
        target=target, operation="activate", kind=kind.value, kind_text=kind.value,
        name="Save", surface="target",
    )

    assert outcome.interaction is not None
    assert outcome.interaction.element.kind == kind


def test_bind_reported_interaction_normalises_operation_kind_and_surface_casing():
    outcome = _sign_in(operation=" Activate ", kind="BUTTON", surface="Target")

    assert outcome.interaction is not None
    assert outcome.interaction.element.kind == ElementKind.button


def test_bind_reported_interaction_binds_citation_across_extra_whitespace():
    outcome = _bind(
        target="Sign  in\n button", operation="activate", kind="button",
        kind_text="button", name="Sign in", surface="target",
    )

    assert outcome.interaction.element.name_citation.text == "Sign  in"


def test_bind_reported_interaction_binds_a_typed_value_from_input_value():
    outcome = _bind(
        ActionType.type_text, target="Email field", input_value="user@example.invalid",
        operation="enter", kind="field", kind_text="field", name="Email", surface="target",
    )

    assert outcome.interaction.element.name == "Email"


def test_bind_reported_interaction_binds_an_assertion_from_visible_text():
    outcome = _bind(
        ActionType.assert_visible, target=None, visible_text="Welcome dialog",
        operation="assert", kind="dialog", kind_text="dialog", name="Welcome",
        surface="visible_text",
    )

    assert outcome.interaction.citation_surface == CitationSurface.visible_text


@pytest.mark.parametrize("operation", ["click", "press", "unspecified", "activate it"])
def test_bind_reported_interaction_rejects_unsupported_operation(operation: str):
    outcome = _sign_in(operation=operation)

    assert (outcome.interaction, outcome.diagnostic.reason) == (
        None, InteractionRejectionReason.unsupported_operation,
    )


@pytest.mark.parametrize("kind", ["textbox", "input", "unspecified", "menu item"])
def test_bind_reported_interaction_rejects_unsupported_kind(kind: str):
    outcome = _sign_in(kind=kind)

    assert (outcome.interaction, outcome.diagnostic.reason) == (
        None, InteractionRejectionReason.unsupported_kind,
    )


@pytest.mark.parametrize("surface", ["screen_description", "description", "frame"])
def test_bind_reported_interaction_rejects_unsupported_surface(surface: str):
    outcome = _sign_in(surface=surface)

    assert outcome.diagnostic.reason == InteractionRejectionReason.unsupported_surface


@pytest.mark.parametrize("slot", ["operation", "kind", "kind_text", "surface"])
def test_bind_reported_interaction_rejects_a_missing_required_field(slot: str):
    outcome = _sign_in(**{slot: None})

    assert (outcome.diagnostic.reason, outcome.diagnostic.slot) == (
        InteractionRejectionReason.missing_field, slot,
    )


def test_bind_reported_interaction_treats_a_blank_field_as_missing():
    outcome = _sign_in(kind_text="  ")

    assert outcome.diagnostic.reason == InteractionRejectionReason.missing_field


def test_bind_reported_interaction_treats_no_interaction_as_omitted():
    outcome = _bind()

    assert (outcome.interaction, outcome.diagnostic.status, outcome.diagnostic.reason) == (
        None, InteractionParseStatus.omitted, None,
    )


def test_bind_reported_interaction_keeps_omission_distinct_from_rejection():
    assert _bind().diagnostic.status != _sign_in(kind="textbox").diagnostic.status


@pytest.mark.parametrize(
    ("action", "operation"),
    [
        (ActionType.type_text, "activate"),
        (ActionType.click, "enter"),
        (ActionType.navigate, "activate"),
        (ActionType.select, "enter"),
        (ActionType.assert_visible, "activate"),
        (ActionType.click, "assert"),
        (ActionType.scroll, "activate"),
        (ActionType.hover, "activate"),
        (ActionType.other, "dismiss"),
    ],
)
def test_bind_reported_interaction_rejects_an_operation_the_action_does_not_denote(
    action: ActionType, operation: str
):
    outcome = _sign_in(operation=operation) if action == ActionType.click else _bind(
        action, operation=operation, kind="button", kind_text="button", surface="target",
    )

    assert outcome.diagnostic.reason == InteractionRejectionReason.operation_contradicts_action


def test_bind_reported_interaction_accepts_a_dismissal_click():
    outcome = _bind(
        target="Close icon", operation="dismiss", kind="icon", kind_text="icon",
        name="Close", surface="target",
    )

    assert outcome.interaction.operation == IntentOperation.dismiss


def test_bind_reported_interaction_rejects_a_surface_the_observation_lacks():
    outcome = _sign_in(surface="visible_text")

    assert outcome.diagnostic.reason == InteractionRejectionReason.surface_unavailable


def test_bind_reported_interaction_rejects_a_kind_word_absent_from_the_surface():
    outcome = _bind(
        target="Sign in", operation="activate", kind="button", kind_text="button",
        name="Sign in", surface="target",
    )

    assert (outcome.diagnostic.reason, outcome.diagnostic.slot) == (
        InteractionRejectionReason.invalid_citation, "kind_text",
    )


def test_bind_reported_interaction_rejects_a_name_absent_from_the_surface():
    outcome = _sign_in(name="Log out")

    assert (outcome.diagnostic.reason, outcome.diagnostic.slot) == (
        InteractionRejectionReason.invalid_citation, "name",
    )


def test_bind_reported_interaction_rejects_a_citation_that_differs_in_case():
    outcome = _sign_in(name="sign in")

    assert outcome.diagnostic.reason == InteractionRejectionReason.invalid_citation


def test_bind_reported_interaction_rejects_text_cited_from_a_different_surface():
    outcome = _bind(
        ActionType.type_text, target="Email field", input_value="Sign in",
        operation="enter", kind="field", kind_text="field", name="Sign in", surface="target",
    )

    assert outcome.diagnostic.reason == InteractionRejectionReason.invalid_citation


def test_bind_reported_interaction_rejects_an_ambiguous_citation():
    outcome = _bind(
        target="Save button and Save link", operation="activate", kind="button",
        kind_text="button", name="Save", surface="target",
    )

    assert (outcome.diagnostic.reason, outcome.diagnostic.slot) == (
        InteractionRejectionReason.ambiguous_citation, "name",
    )


def test_bind_reported_interaction_reports_a_binding_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "reckon.stage2.interaction_binding.bind_observed_interaction",
        lambda **kwargs: None,
    )

    assert _sign_in().diagnostic.reason == InteractionRejectionReason.binding_failed


def test_bind_reported_interaction_records_no_screen_text_in_a_rejection():
    outcome = _bind(
        ActionType.type_text, target="Password field", input_value="hunter2-secret",
        operation="enter", kind="field", kind_text="field", value="hunter2-secretX",
        surface="input_value",
    )

    assert outcome.diagnostic.status == InteractionParseStatus.rejected
    assert "hunter2" not in outcome.diagnostic.model_dump_json()


def test_bind_reported_interaction_records_no_free_text_offered_as_a_kind():
    outcome = _sign_in(kind="token=abc123!")

    assert outcome.diagnostic.offered_kind is None
    assert "abc123" not in outcome.diagnostic.model_dump_json()


def test_bind_reported_interaction_records_the_offered_vocabulary_on_rejection():
    outcome = _sign_in(kind="Textbox")

    assert (outcome.diagnostic.offered_operation, outcome.diagnostic.offered_kind) == (
        "activate", "textbox",
    )
