"""Runtime model defaults for the normal LLM pipeline stages."""

from unittest.mock import MagicMock

from reckon.crosscheck import CROSSCHECK_MODEL
from reckon.extractor import EXTRACTION_MODEL
from reckon.parser import OPENAI_MODEL, _LLMStep, _LLMTrace, parse_scenario
from reckon.schema import ActionType

DEFAULT_RUNTIME_MODEL = "claude-opus-5-5"


def test_runtime_model_defaults_are_claude_opus_5_5():
    assert OPENAI_MODEL == DEFAULT_RUNTIME_MODEL
    assert EXTRACTION_MODEL == DEFAULT_RUNTIME_MODEL
    assert CROSSCHECK_MODEL == DEFAULT_RUNTIME_MODEL


def test_parse_scenario_uses_default_model_when_omitted():
    client = MagicMock()
    client.chat.completions.create.return_value = _LLMTrace(
        scenario_title="t",
        steps=[_LLMStep(action=ActionType.click, target="Go")],
    )

    parse_scenario("Click Go", openai_client=client)

    assert client.chat.completions.create.call_args.kwargs["model"] == DEFAULT_RUNTIME_MODEL


def test_no_gpt_model_ids_remain_as_runtime_defaults():
    for value in (OPENAI_MODEL, EXTRACTION_MODEL, CROSSCHECK_MODEL):
        assert value == DEFAULT_RUNTIME_MODEL
        assert not value.startswith("gpt-")
