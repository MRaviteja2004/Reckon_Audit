import pytest

from reckon.intent_contract import (
    EXPECTED_INTENT_KINDS,
    EXPECTED_INTENT_OPERATIONS,
    INTENT_CONTRACT_VERSION,
    MAX_VOCABULARY_TOKEN_LENGTH,
    ReportedIntent,
    intent_prompt_section,
    loggable_token,
    normalise_token,
)
from reckon.parser import _FREE_TEXT_SYSTEM, _GHERKIN_SYSTEM, _LLMTrace
from reckon.schema import IntentOperation, ObjectKind


def _slot_schema(slot: str) -> dict:
    return _LLMTrace.model_json_schema()["$defs"]["ReportedIntent"]["properties"][slot]


def _string_branch(slot: str) -> dict:
    return next(b for b in _slot_schema(slot)["anyOf"] if b.get("type") == "string")


def test_expected_operations_are_every_operation_but_unspecified():
    assert set(EXPECTED_INTENT_OPERATIONS) == {
        member.value for member in IntentOperation if member != IntentOperation.unspecified
    }


def test_expected_kinds_are_every_kind_but_unspecified():
    assert set(EXPECTED_INTENT_KINDS) == {
        member.value for member in ObjectKind if member != ObjectKind.unspecified
    }


def test_expected_operations_have_the_documented_values():
    assert EXPECTED_INTENT_OPERATIONS == (
        "activate", "enter", "choose", "traverse", "dismiss", "assert",
    )


def test_response_schema_lists_operations_as_an_enum():
    assert _string_branch("operation")["enum"] == list(EXPECTED_INTENT_OPERATIONS)


def test_response_schema_lists_kinds_as_an_enum():
    assert _string_branch("kind")["enum"] == list(EXPECTED_INTENT_KINDS)


def test_response_schema_keeps_operation_nullable():
    assert {"type": "null"} in _slot_schema("operation")["anyOf"]


def test_response_schema_describes_citation_slots_as_verbatim():
    assert "verbatim" in _slot_schema("operation_text")["description"]


def test_reported_intent_accepts_an_unsupported_operation_without_failing_the_response():
    assert ReportedIntent(operation="click").operation == "click"


@pytest.mark.parametrize("system", [_FREE_TEXT_SYSTEM, _GHERKIN_SYSTEM])
def test_prompts_list_the_same_values_as_the_schema(system: str):
    assert intent_prompt_section() in system


def test_prompt_section_lists_every_allowed_value():
    section = intent_prompt_section()
    assert all(value in section for value in (*EXPECTED_INTENT_OPERATIONS, *EXPECTED_INTENT_KINDS))


def test_contract_version_is_two():
    assert INTENT_CONTRACT_VERSION == "2"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Activate", "activate"), ("  BUTTON ", "button"), ("menu_item", "menu_item"),
     ("", None), ("   ", None), (None, None)],
)
def test_normalise_token_changes_only_case_and_edge_whitespace(raw: str | None, expected: str | None):
    assert normalise_token(raw) == expected


def test_normalise_token_does_not_map_a_different_spelling_onto_a_value():
    assert normalise_token("menu item") == "menu item"


def test_loggable_token_keeps_plain_vocabulary():
    assert loggable_token("Menu_Item") == "menu_item"


def test_loggable_token_drops_text_that_could_carry_a_secret():
    assert loggable_token("password=hunter2!") is None


def test_loggable_token_drops_overlong_text():
    assert loggable_token("a" * (MAX_VOCABULARY_TOKEN_LENGTH + 1)) is None
