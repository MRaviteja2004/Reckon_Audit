import pytest

from reckon.equivalence.engine import _ACTION_OPERATION
from reckon.extractor import EXTRACTION_SYSTEM, _ExtractedActions
from reckon.schema import ActionType, CitationSurface, ElementKind, IntentOperation
from reckon.stage2.interaction_contract import (
    CITATION_SURFACES,
    INTERACTION_CONTRACT_VERSION,
    MAX_VOCABULARY_TOKEN_LENGTH,
    OBSERVED_KINDS,
    OBSERVED_OPERATIONS,
    OPERATIONS_FOR_ACTION,
    ReportedInteraction,
    interaction_prompt_section,
    loggable_token,
    normalise_token,
)


def _slot_schema(slot: str) -> dict:
    return _ExtractedActions.model_json_schema()["$defs"]["_ExtractedAction"]["properties"][slot]


def _enum(slot: str) -> list[str]:
    return next(b for b in _slot_schema(slot)["anyOf"] if b.get("type") == "string")["enum"]


def test_observed_operations_are_every_operation_but_unspecified():
    assert set(OBSERVED_OPERATIONS) == {
        m.value for m in IntentOperation if m != IntentOperation.unspecified
    }


def test_observed_kinds_are_every_element_kind_but_unspecified():
    assert set(OBSERVED_KINDS) == {m.value for m in ElementKind if m != ElementKind.unspecified}


def test_citation_surfaces_are_the_citation_surface_enum():
    assert set(CITATION_SURFACES) == {m.value for m in CitationSurface} == {
        "target", "input_value", "visible_text",
    }


def test_operations_for_action_agree_with_the_equivalence_engine():
    assert set(OPERATIONS_FOR_ACTION) == set(_ACTION_OPERATION)
    assert all(_ACTION_OPERATION[a] in ops for a, ops in OPERATIONS_FOR_ACTION.items())


def test_only_a_click_may_be_a_dismissal():
    assert [a for a, ops in OPERATIONS_FOR_ACTION.items() if IntentOperation.dismiss in ops] == [
        ActionType.click
    ]


@pytest.mark.parametrize("action", [ActionType.scroll, ActionType.hover, ActionType.other])
def test_actions_without_an_operation_carry_no_interaction(action: ActionType):
    assert action not in OPERATIONS_FOR_ACTION


def test_response_schema_lists_operations_as_an_enum():
    assert _enum("interaction_operation") == list(OBSERVED_OPERATIONS)


def test_response_schema_lists_kinds_as_an_enum():
    assert _enum("interaction_kind") == list(OBSERVED_KINDS)


def test_response_schema_lists_surfaces_as_an_enum():
    assert _enum("interaction_surface") == list(CITATION_SURFACES)


def test_response_schema_keeps_interaction_fields_nullable():
    assert {"type": "null"} in _slot_schema("interaction_kind")["anyOf"]


def test_reported_interaction_accepts_an_unsupported_value_without_failing_the_window():
    assert ReportedInteraction(interaction_kind="textbox").interaction_kind == "textbox"


def test_extraction_prompt_includes_the_interaction_section():
    assert interaction_prompt_section() in EXTRACTION_SYSTEM


def test_interaction_section_lists_every_allowed_value():
    section = interaction_prompt_section()
    values = (*OBSERVED_OPERATIONS, *OBSERVED_KINDS, *CITATION_SURFACES)
    assert all(value in section for value in values)


def test_interaction_section_does_not_mention_a_scenario():
    lowered = interaction_prompt_section().lower()
    assert "scenario" not in lowered and "expected" not in lowered


def test_interaction_contract_version_is_one():
    assert INTERACTION_CONTRACT_VERSION == "1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Activate", "activate"), (" BUTTON  ", "button"), ("", None), (None, None)],
)
def test_normalise_token_changes_only_case_and_edge_whitespace(raw: str | None, expected: str | None):
    assert normalise_token(raw) == expected


def test_loggable_token_drops_text_that_could_carry_a_secret():
    assert loggable_token("token=abc123!") is None


def test_loggable_token_drops_overlong_text():
    assert loggable_token("a" * (MAX_VOCABULARY_TOKEN_LENGTH + 1)) is None
