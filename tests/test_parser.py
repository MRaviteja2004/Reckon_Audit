from unittest.mock import MagicMock, patch

import pytest

from reckon.parser import (
    DEFAULT_TEMPERATURE,
    OPENAI_MODEL,
    _ASSERTION_RULES,
    _FREE_TEXT_SYSTEM,
    _GHERKIN_SYSTEM,
    _LLMStep,
    _LLMTrace,
    _build_expected_trace,
    _extract_gherkin_scenario_title,
    _extract_gherkin_steps,
    _is_lossless_gherkin,
    _looks_like_gherkin,
    make_openai_client,
    parse_scenario,
)
from reckon.schema import ActionType, ExpectedTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_trace(
    title: str = "Test scenario",
    steps: list[_LLMStep] | None = None,
) -> _LLMTrace:
    if steps is None:
        steps = [
            _LLMStep(action=ActionType.navigate, target="/login", raw_text="navigate to login"),
            _LLMStep(
                action=ActionType.type_text,
                target="Email field",
                input_value="user@example.invalid",
                raw_text="enter email",
            ),
            _LLMStep(
                action=ActionType.assert_text,
                target="Error banner",
                assertion="Invalid credentials",
                raw_text="error message is shown",
            ),
        ]
    return _LLMTrace(scenario_title=title, steps=steps)


def _make_instructor_client(llm_trace: _LLMTrace) -> MagicMock:
    """Return a mock instructor client that returns llm_trace from create()."""
    client = MagicMock()
    client.chat.completions.create.return_value = llm_trace
    return client


# ---------------------------------------------------------------------------
# Prompt contract — free text
# ---------------------------------------------------------------------------

def test_free_text_system_instructs_no_invented_steps():
    assert "do not invent" in _FREE_TEXT_SYSTEM.lower()


def test_free_text_system_instructs_preserve_order():
    assert "order" in _FREE_TEXT_SYSTEM.lower()


def test_free_text_system_lists_valid_action_types():
    for action in ("click", "type_text", "navigate", "assert_text"):
        assert action in _FREE_TEXT_SYSTEM


def test_free_text_system_requires_raw_text():
    assert "raw_text" in _FREE_TEXT_SYSTEM


# ---------------------------------------------------------------------------
# Prompt contract — Gherkin
# ---------------------------------------------------------------------------

def test_gherkin_system_forbids_splitting_or_merging_steps():
    lowered = _GHERKIN_SYSTEM.lower()
    assert "split" in lowered and "merge" in lowered


def test_free_text_system_keeps_asserted_text_out_of_input_value():
    # The aligner keys assertion_not_observed off `assertion` and wrong_input_value
    # off `input_value`; conflating them would both fabricate and mask findings.
    lowered = _FREE_TEXT_SYSTEM.lower()
    assert "never put asserted" in lowered


def test_gherkin_system_keeps_asserted_text_out_of_input_value():
    lowered = _GHERKIN_SYSTEM.lower()
    assert "never put asserted" in lowered


def test_free_text_system_skips_generic_value_placeholders():
    # A generic description like "a password" is not a typed value; extracting it would
    # produce a spurious wrong_input_value against the real (masked / different) value.
    lowered = _FREE_TEXT_SYSTEM.lower()
    assert "generic" in lowered
    assert "a password" in lowered


def test_gherkin_system_skips_generic_value_placeholders():
    lowered = _GHERKIN_SYSTEM.lower()
    assert "generic" in lowered
    assert "a password" in lowered


def test_gherkin_system_requires_one_output_step_per_input_line():
    assert "one" in _GHERKIN_SYSTEM.lower() and "line" in _GHERKIN_SYSTEM.lower()


def test_gherkin_system_lists_valid_action_types():
    for action in ("click", "type_text", "navigate", "assert_text"):
        assert action in _GHERKIN_SYSTEM


def test_gherkin_system_requires_raw_text():
    assert "raw_text" in _GHERKIN_SYSTEM


# ---------------------------------------------------------------------------
# make_openai_client
# ---------------------------------------------------------------------------

def test_make_openai_client_patches_gateway_client():
    with patch("reckon.parser.make_gateway_client") as mock_gateway_client, \
         patch("reckon.parser.instructor.from_openai") as mock_from_openai:
        make_openai_client()
        mock_gateway_client.assert_called_once_with()
        mock_from_openai.assert_called_once_with(mock_gateway_client.return_value)


# ---------------------------------------------------------------------------
# _looks_like_gherkin
# ---------------------------------------------------------------------------

def test_looks_like_gherkin_detects_feature_keyword():
    assert _looks_like_gherkin("Feature: Login\nScenario: Valid login")


def test_looks_like_gherkin_detects_scenario_keyword():
    assert _looks_like_gherkin("Scenario: User resets password")


def test_looks_like_gherkin_detects_given_keyword():
    assert _looks_like_gherkin("Given the user is on the login page")


def test_looks_like_gherkin_detects_when_keyword():
    assert _looks_like_gherkin("When the user clicks Submit")


def test_looks_like_gherkin_detects_then_keyword():
    assert _looks_like_gherkin("Then an error message is shown")


def test_looks_like_gherkin_detects_and_keyword():
    text = "Given the user is logged out\nAnd the session has expired"
    assert _looks_like_gherkin(text)


def test_looks_like_gherkin_rejects_plain_prose():
    assert not _looks_like_gherkin(
        "The user opens the login page and enters their credentials."
    )


def test_looks_like_gherkin_rejects_numbered_steps():
    assert not _looks_like_gherkin(
        "1. Navigate to /login\n2. Enter username\n3. Click Submit"
    )


def test_looks_like_gherkin_is_case_insensitive():
    assert _looks_like_gherkin("given the user is on the page")


# ---------------------------------------------------------------------------
# _is_lossless_gherkin — the guard that prevents silent data loss
# ---------------------------------------------------------------------------

def test_is_lossless_gherkin_true_for_pure_step_lines():
    assert _is_lossless_gherkin("Given A\nWhen B\nThen C")


def test_is_lossless_gherkin_true_with_headers_tags_comments():
    text = (
        "@smoke\n"
        "Feature: Login\n"
        "  # a comment\n"
        "  Scenario: Valid login\n"
        "    Given the user is on the login page\n"
        "    Then the dashboard is visible\n"
    )
    assert _is_lossless_gherkin(text)


def test_is_lossless_gherkin_false_when_prose_line_present():
    # One real Gherkin step plus prose that would be dropped by step extraction.
    text = "The user opens the app.\nThen the dashboard loads.\nThey click Settings."
    assert not _is_lossless_gherkin(text)


def test_is_lossless_gherkin_false_for_data_table():
    text = (
        "When the user enters credentials\n"
        "  | field | value     |\n"
        "  | email | a@b.test  |\n"
    )
    assert not _is_lossless_gherkin(text)


def test_is_lossless_gherkin_false_when_no_step_lines():
    assert not _is_lossless_gherkin("Feature: Login\nScenario: Valid login")


def test_is_lossless_gherkin_false_for_plain_prose():
    assert not _is_lossless_gherkin("The user logs in and sees the dashboard.")


# ---------------------------------------------------------------------------
# _extract_gherkin_scenario_title
# ---------------------------------------------------------------------------

def test_extract_scenario_title_from_scenario_line():
    text = "Feature: Auth\nScenario: Login with invalid password"
    assert _extract_gherkin_scenario_title(text) == "Login with invalid password"


def test_extract_scenario_title_prefers_scenario_over_feature():
    text = "Feature: Auth\nScenario: Password reset"
    assert _extract_gherkin_scenario_title(text) == "Password reset"


def test_extract_scenario_title_falls_back_to_feature():
    text = "Feature: Authentication flow"
    assert _extract_gherkin_scenario_title(text) == "Authentication flow"


def test_extract_scenario_title_returns_none_when_absent():
    assert _extract_gherkin_scenario_title("Given something happens") is None


def test_extract_scenario_title_handles_scenario_outline():
    text = "Scenario Outline: Login with <role>"
    assert _extract_gherkin_scenario_title(text) == "Login with <role>"


def test_extract_scenario_title_strips_whitespace():
    text = "Scenario:   Trailing spaces   "
    assert _extract_gherkin_scenario_title(text) == "Trailing spaces"


# ---------------------------------------------------------------------------
# _extract_gherkin_steps
# ---------------------------------------------------------------------------

def test_extract_gherkin_steps_returns_step_texts_only():
    text = (
        "Feature: Login\n"
        "  Scenario: Invalid password\n"
        "    Given the user is on the login page\n"
        "    When the user enters 'wrong' in the password field\n"
        "    Then an error message is displayed\n"
    )
    steps = _extract_gherkin_steps(text)
    assert steps == [
        "the user is on the login page",
        "the user enters 'wrong' in the password field",
        "an error message is displayed",
    ]


def test_extract_gherkin_steps_includes_and_but():
    text = (
        "Given the user is logged out\n"
        "And the session has expired\n"
        "But the remember-me cookie is set\n"
    )
    steps = _extract_gherkin_steps(text)
    assert len(steps) == 3
    assert steps[1] == "the session has expired"


def test_extract_gherkin_steps_excludes_header_lines():
    text = "Feature: Auth\nScenario: Login\nGiven the page is loaded"
    steps = _extract_gherkin_steps(text)
    assert steps == ["the page is loaded"]


def test_extract_gherkin_steps_returns_empty_for_no_steps():
    assert _extract_gherkin_steps("Feature: Auth\nScenario: Empty") == []


def test_extract_gherkin_steps_preserves_order():
    text = "Given A\nWhen B\nThen C"
    steps = _extract_gherkin_steps(text)
    assert steps == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# _build_expected_trace
# ---------------------------------------------------------------------------

def test_build_expected_trace_reindexes_sequentially():
    llm_trace = _make_llm_trace()
    trace = _build_expected_trace(llm_trace)
    assert [s.index for s in trace.steps] == [1, 2, 3]


def test_build_expected_trace_preserves_scenario_title():
    llm_trace = _make_llm_trace(title="My scenario")
    trace = _build_expected_trace(llm_trace)
    assert trace.scenario_title == "My scenario"


def test_build_expected_trace_preserves_action_types():
    llm_trace = _make_llm_trace()
    trace = _build_expected_trace(llm_trace)
    assert trace.steps[0].action == ActionType.navigate
    assert trace.steps[1].action == ActionType.type_text
    assert trace.steps[2].action == ActionType.assert_text


def test_build_expected_trace_returns_expected_trace_instance():
    assert isinstance(_build_expected_trace(_make_llm_trace()), ExpectedTrace)


def test_build_expected_trace_normalises_blank_target_to_none():
    step = _LLMStep(action=ActionType.click, target="   ")
    trace = _build_expected_trace(_LLMTrace(scenario_title="t", steps=[step]))
    assert trace.steps[0].target is None


def test_build_expected_trace_empty_steps_list():
    trace = _build_expected_trace(_LLMTrace(scenario_title="empty", steps=[]))
    assert trace.steps == []


# ---------------------------------------------------------------------------
# parse_scenario — free text path
# ---------------------------------------------------------------------------

FREE_TEXT_SCENARIO = """\
The user navigates to the login page.
They enter 'user@example.invalid' in the email field.
They enter 'wrongpass' in the password field.
They click the Sign in button.
The error message 'Invalid email or password' should be visible.
"""

STRUCTURED_STEPS_SCENARIO = """\
1. Navigate to /login
2. Enter 'user@example.invalid' in the Email field
3. Enter 'wrongpass' in the Password field
4. Click the Sign in button
5. Verify 'Invalid email or password' error message is shown
"""


def test_parse_scenario_free_text_returns_expected_trace():
    client = _make_instructor_client(_make_llm_trace())
    result = parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)
    assert isinstance(result, ExpectedTrace)


def test_parse_scenario_free_text_uses_free_text_system_prompt():
    llm_trace = _make_llm_trace()
    client = _make_instructor_client(llm_trace)
    parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert system_msgs and _FREE_TEXT_SYSTEM in system_msgs[0]["content"]


def test_parse_scenario_free_text_sends_scenario_as_user_message():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert user_msgs and FREE_TEXT_SCENARIO in user_msgs[0]["content"]


def test_parse_scenario_free_text_uses_llm_trace_response_model():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_model"] is _LLMTrace


def test_parse_scenario_structured_steps_uses_free_text_path():
    client = _make_instructor_client(_make_llm_trace())
    result = parse_scenario(STRUCTURED_STEPS_SCENARIO, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert system_msgs and _FREE_TEXT_SYSTEM in system_msgs[0]["content"]
    assert isinstance(result, ExpectedTrace)


def test_parse_scenario_keeps_wait_constraint_with_preceding_action():
    scenario = """\
1. Open GitHub homepage.
2. Click "Sign in".
3. Verify the GitHub login page is displayed.
4. Enter username in the username field.
5. Enter password in the password field.
6. Click "Sign in" and wait for login to complete.
7. Verify the GitHub Dashboard is displayed.
"""
    llm_trace = _LLMTrace(
        scenario_title="GitHub login",
        steps=[
            _LLMStep(action=ActionType.navigate, target="GitHub homepage"),
            _LLMStep(action=ActionType.click, target="Sign in"),
            _LLMStep(
                action=ActionType.assert_visible,
                target="GitHub login page",
                assertion="The GitHub login page is displayed.",
            ),
            _LLMStep(action=ActionType.type_text, target="username field"),
            _LLMStep(action=ActionType.type_text, target="password field"),
            _LLMStep(
                action=ActionType.click,
                target="Sign in",
                raw_text='Click "Sign in" and wait for login to complete.',
            ),
            _LLMStep(
                action=ActionType.assert_visible,
                target="GitHub Dashboard",
                assertion="The GitHub Dashboard is displayed.",
            ),
        ],
    )
    client = _make_instructor_client(llm_trace)

    result = parse_scenario(scenario, openai_client=client)

    assert len(result.steps) == 7
    assert result.steps[5].action == ActionType.click
    assert result.steps[5].target == "Sign in"
    assert result.steps[6].action == ActionType.assert_visible
    assert result.steps[6].target == "GitHub Dashboard"
    assert all(step.action != ActionType.other for step in result.steps)
    system_message = client.chat.completions.create.call_args.kwargs["messages"][0]
    assert "do not emit a separate `other` step" in system_message["content"]


def test_parse_scenario_numbered_steps_preserve_original_raw_text():
    scenario = (
        '1. Enter "tomsmith" as the username.\n'
        '2. Enter "wrongpassword" as the password.\n'
        "3. Click the Login button.\n"
        '4. Verify that "Your password is invalid!" is displayed.\n'
    )

    # Simulate the model dropping the list numbers from raw_text.
    llm_trace = _LLMTrace(
        scenario_title="Numbered login",
        steps=[
            _LLMStep(
                action=ActionType.type_text,
                target="username",
                input_value="tomsmith",
                raw_text='Enter "tomsmith" as the username.',
            ),
            _LLMStep(
                action=ActionType.type_text,
                target="password",
                input_value="wrongpassword",
                raw_text='Enter "wrongpassword" as the password.',
            ),
            _LLMStep(
                action=ActionType.click,
                target="Login button",
                raw_text="Click the Login button.",
            ),
            _LLMStep(
                action=ActionType.assert_text,
                assertion="Your password is invalid!",
                raw_text='Verify that "Your password is invalid!" is displayed.',
            ),
        ],
    )

    client = _make_instructor_client(llm_trace)

    result = parse_scenario(
        scenario,
        openai_client=client,
    )

    assert [step.raw_text for step in result.steps] == [
        '1. Enter "tomsmith" as the username.',
        '2. Enter "wrongpassword" as the password.',
        "3. Click the Login button.",
        '4. Verify that "Your password is invalid!" is displayed.',
    ]


def test_parse_scenario_normalizes_imperative_dismiss_to_click():
    scenario = "2. Dismiss terms after reading"
    client = _make_instructor_client(_LLMTrace(
        scenario_title="Terms and Conditions Navigation",
        steps=[
            _LLMStep(
                action=ActionType.other,
                target="terms",
                raw_text="Dismiss terms after reading",
            ),
        ],
    ))

    result = parse_scenario(scenario, openai_client=client)

    assert result.steps[0].action == ActionType.click
    assert result.steps[0].target == "terms"
    assert result.steps[0].raw_text == scenario


def test_parse_scenario_does_not_normalize_passive_dismissal():
    scenario = "Verify the dialog was dismissed"
    client = _make_instructor_client(_LLMTrace(
        scenario_title="Dialog dismissal",
        steps=[
            _LLMStep(
                action=ActionType.other,
                target="dialog",
                assertion="the dialog was dismissed",
                raw_text=scenario,
            ),
        ],
    ))

    result = parse_scenario(scenario, openai_client=client)

    assert result.steps[0].action == ActionType.other
    assert result.steps[0].raw_text == scenario


def test_parse_scenario_uses_custom_model_name():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(FREE_TEXT_SCENARIO, openai_client=client, model="gpt-4-turbo")

    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4-turbo"


def test_parse_scenario_steps_are_sequentially_indexed():
    client = _make_instructor_client(_make_llm_trace())
    result = parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)
    assert [s.index for s in result.steps] == list(range(1, len(result.steps) + 1))


# ---------------------------------------------------------------------------
# parse_scenario — Gherkin path
# ---------------------------------------------------------------------------

GHERKIN_SCENARIO = """\
Feature: Authentication
  Scenario: Login with invalid password shows error message
    Given the user is on the login page
    When the user enters 'user@example.invalid' in the email field
    And enters 'wrongpass' in the password field
    And clicks the Sign in button
    Then the error message 'Invalid email or password' is displayed
"""


def test_parse_scenario_gherkin_returns_expected_trace():
    client = _make_instructor_client(_make_llm_trace())
    result = parse_scenario(GHERKIN_SCENARIO, openai_client=client)
    assert isinstance(result, ExpectedTrace)


def test_parse_scenario_gherkin_uses_gherkin_system_prompt():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(GHERKIN_SCENARIO, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert system_msgs and _GHERKIN_SYSTEM in system_msgs[0]["content"]


def test_parse_scenario_gherkin_sends_step_lines_not_full_blob():
    """The LLM must not see Feature:/Scenario: headers — only step lines."""
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(GHERKIN_SCENARIO, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
    assert "Feature:" not in user_content


def test_parse_scenario_gherkin_sends_extracted_step_texts():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(GHERKIN_SCENARIO, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
    assert "the user is on the login page" in user_content


def test_parse_scenario_gherkin_preserves_title_from_source_not_llm():
    """Title must come from the Gherkin text, not from whatever the LLM emits."""
    llm_trace = _make_llm_trace(title="LLM invented title — should be overridden")
    client = _make_instructor_client(llm_trace)
    result = parse_scenario(GHERKIN_SCENARIO, openai_client=client)
    assert result.scenario_title == "Login with invalid password shows error message"


def test_parse_scenario_gherkin_step_count_matches_step_lines():
    step_lines = _extract_gherkin_steps(GHERKIN_SCENARIO)
    llm_steps = [_LLMStep(action=ActionType.click, raw_text=line) for line in step_lines]
    client = _make_instructor_client(_LLMTrace(scenario_title="t", steps=llm_steps))
    result = parse_scenario(GHERKIN_SCENARIO, openai_client=client)
    assert len(result.steps) == len(step_lines)


def test_parse_scenario_gherkin_steps_sequentially_indexed():
    step_lines = _extract_gherkin_steps(GHERKIN_SCENARIO)
    llm_steps = [_LLMStep(action=ActionType.navigate, raw_text=line) for line in step_lines]
    client = _make_instructor_client(_LLMTrace(scenario_title="t", steps=llm_steps))
    result = parse_scenario(GHERKIN_SCENARIO, openai_client=client)
    assert [s.index for s in result.steps] == list(range(1, len(step_lines) + 1))


def test_parse_scenario_gherkin_without_feature_header():
    """Bare Given/When/Then lines (no Feature: or Scenario: header) are still Gherkin."""
    bare_gherkin = (
        "Given the user is on the home page\n"
        "When the user clicks the menu\n"
        "Then the dropdown is visible\n"
    )
    llm_steps = [
        _LLMStep(action=ActionType.navigate, raw_text="the user is on the home page"),
        _LLMStep(action=ActionType.click, raw_text="the user clicks the menu"),
        _LLMStep(action=ActionType.assert_visible, raw_text="the dropdown is visible"),
    ]
    client = _make_instructor_client(_LLMTrace(scenario_title="Scenario", steps=llm_steps))
    result = parse_scenario(bare_gherkin, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert system_msgs and _GHERKIN_SYSTEM in system_msgs[0]["content"]
    assert isinstance(result, ExpectedTrace)


def test_parse_scenario_gherkin_with_no_steps_falls_back_to_free_text():
    """A Gherkin file with only headers and no step lines falls back to free-text."""
    header_only = "Feature: Empty\n  Scenario: Nothing to do\n"
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(header_only, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert system_msgs and _FREE_TEXT_SYSTEM in system_msgs[0]["content"]


# ---------------------------------------------------------------------------
# parse_scenario — schema validation on output
# ---------------------------------------------------------------------------

def test_parse_scenario_output_validates_against_expected_trace_schema():
    client = _make_instructor_client(_make_llm_trace())
    result = parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)
    # Re-validate through Pydantic to confirm the output is schema-compliant.
    validated = ExpectedTrace.model_validate(result.model_dump())
    assert validated.scenario_title == result.scenario_title


def test_parse_scenario_blank_target_from_llm_becomes_none():
    llm_trace = _LLMTrace(
        scenario_title="t",
        steps=[_LLMStep(action=ActionType.click, target="   ")],
    )
    client = _make_instructor_client(llm_trace)
    result = parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)
    assert result.steps[0].target is None


def test_parse_scenario_blank_assertion_from_llm_becomes_none():
    llm_trace = _LLMTrace(
        scenario_title="t",
        steps=[_LLMStep(action=ActionType.assert_text, assertion="")],
    )
    client = _make_instructor_client(llm_trace)
    result = parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)
    assert result.steps[0].assertion is None


def test_parse_scenario_single_step_trace_is_valid():
    llm_trace = _LLMTrace(
        scenario_title="one step",
        steps=[_LLMStep(action=ActionType.navigate, target="/home")],
    )
    client = _make_instructor_client(llm_trace)
    result = parse_scenario("Navigate to home.", openai_client=client)
    assert len(result.steps) == 1
    assert result.steps[0].index == 1


def test_parse_scenario_makes_exactly_one_llm_call():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)
    assert client.chat.completions.create.call_count == 1


def test_parse_scenario_gherkin_makes_exactly_one_llm_call():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(GHERKIN_SCENARIO, openai_client=client)
    assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Data-loss guard: free text containing a stray Gherkin keyword
# ---------------------------------------------------------------------------

PROSE_WITH_STEP_KEYWORD = """\
The user opens the application.
Then they navigate to the settings page.
They toggle the dark mode switch.
"""


def test_parse_scenario_prose_with_keyword_uses_free_text_path():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(PROSE_WITH_STEP_KEYWORD, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert system_msgs and _FREE_TEXT_SYSTEM in system_msgs[0]["content"]


def test_parse_scenario_prose_with_keyword_sends_full_text_no_data_loss():
    # The whole scenario must reach the model — no line silently dropped.
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(PROSE_WITH_STEP_KEYWORD, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
    assert "The user opens the application." in user_content
    assert "They toggle the dark mode switch." in user_content


def test_parse_scenario_gherkin_with_data_table_uses_free_text_path():
    # Data tables carry input values; dropping them would lose data, so the whole
    # text goes to the model via the free-text path.
    gherkin_with_table = (
        "Scenario: Login with table data\n"
        "  When the user enters credentials\n"
        "    | field    | value    |\n"
        "    | email    | a@b.test |\n"
        "  Then the dashboard is visible\n"
    )
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(gherkin_with_table, openai_client=client)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert system_msgs and _FREE_TEXT_SYSTEM in system_msgs[0]["content"]
    user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
    assert "a@b.test" in user_content


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------

def test_parse_scenario_empty_string_raises():
    client = _make_instructor_client(_make_llm_trace())
    with pytest.raises(ValueError, match="empty"):
        parse_scenario("", openai_client=client)
    client.chat.completions.create.assert_not_called()


def test_parse_scenario_whitespace_only_raises():
    client = _make_instructor_client(_make_llm_trace())
    with pytest.raises(ValueError, match="empty"):
        parse_scenario("   \n  \t ", openai_client=client)
    client.chat.completions.create.assert_not_called()


def test_parse_scenario_multiple_scenarios_raises():
    multi = (
        "Feature: Auth\n"
        "  Scenario: Valid login\n"
        "    Given the user is on the login page\n"
        "    Then the dashboard is visible\n"
        "  Scenario: Invalid login\n"
        "    Given the user is on the login page\n"
        "    Then an error is shown\n"
    )
    client = _make_instructor_client(_make_llm_trace())
    with pytest.raises(ValueError, match="multiple scenarios"):
        parse_scenario(multi, openai_client=client)
    client.chat.completions.create.assert_not_called()


def test_parse_scenario_single_scenario_outline_is_allowed():
    # One Scenario Outline (no Examples table) must not trip the multi-scenario guard.
    outline = (
        "Scenario Outline: Login attempt\n"
        "  Given the user is on the login page\n"
        "  Then the result is shown\n"
    )
    step_lines = _extract_gherkin_steps(outline)
    llm_steps = [_LLMStep(action=ActionType.navigate, raw_text=s) for s in step_lines]
    client = _make_instructor_client(_LLMTrace(scenario_title="t", steps=llm_steps))
    result = parse_scenario(outline, openai_client=client)
    assert isinstance(result, ExpectedTrace)


# ---------------------------------------------------------------------------
# Temperature: omitted by default (GPT-5-family models reject explicit values),
# still overridable for models that support it.
# ---------------------------------------------------------------------------

def test_parse_scenario_free_text_omits_temperature_by_default():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(FREE_TEXT_SCENARIO, openai_client=client)
    assert "temperature" not in client.chat.completions.create.call_args.kwargs


def test_parse_scenario_gherkin_omits_temperature_by_default():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(GHERKIN_SCENARIO, openai_client=client)
    assert "temperature" not in client.chat.completions.create.call_args.kwargs


def test_default_temperature_is_none():
    assert DEFAULT_TEMPERATURE is None


def test_parse_scenario_temperature_is_overridable():
    client = _make_instructor_client(_make_llm_trace())
    parse_scenario(FREE_TEXT_SCENARIO, openai_client=client, temperature=0.4)
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.4

def test_free_text_system_uses_shared_assertion_rules():
    assert _ASSERTION_RULES.strip() in _FREE_TEXT_SYSTEM

def test_gherkin_system_uses_shared_assertion_rules():
    assert _ASSERTION_RULES.strip() in _GHERKIN_SYSTEM