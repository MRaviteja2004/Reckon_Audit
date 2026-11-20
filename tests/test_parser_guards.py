from reckon.parser import _build_expected_trace, _LLMStep, _LLMTrace
from reckon.parser_guards import (
    drop_dangling_qualifier_steps,
    normalize_action_and_target,
    restore_source_order,
)
from reckon.schema import ActionType

SOURCE_ORDER_SCENARIO = (
    "1. Click the Terms link\n"
    "2. Dismiss terms after reading\n"
    "3. Terms page opens\n"
    "4. Return to language page"
)


def _step(
    action: ActionType,
    raw_text: str,
    *,
    target: str | None = None,
    assertion: str | None = None,
    input_value: str | None = None,
) -> _LLMStep:
    return _LLMStep(
        action=action,
        target=target,
        assertion=assertion,
        input_value=input_value,
        raw_text=raw_text,
    )


def _trace(*steps: _LLMStep) -> _LLMTrace:
    return _LLMTrace(scenario_title="Guards", steps=list(steps))


# --- 1. dangling qualifiers -------------------------------------------------


def test_dangling_qualifier_step_dropped_when_line_has_sibling():
    title = _step(
        ActionType.assert_text,
        "Earn when you mentor new drivers with description",
        target="title",
        assertion="Earn when you mentor new drivers",
    )
    qualifier = _step(
        ActionType.assert_visible,
        "Earn when you mentor new drivers with description",
        target="Title description",
        assertion="with description",
    )

    assert drop_dangling_qualifier_steps([title, qualifier]) == [title]


def test_dangling_qualifier_step_kept_when_it_is_the_only_step_for_the_line():
    only = _step(
        ActionType.assert_visible,
        "Banner with description",
        target="banner",
        assertion="with description",
    )

    assert drop_dangling_qualifier_steps([only]) == [only]


def test_qualifier_with_concrete_value_is_kept():
    title = _step(ActionType.assert_text, "Header with text Welcome", assertion="Welcome")
    qualified = _step(
        ActionType.assert_text,
        "Header with text Welcome",
        assertion="with description Welcome back",
    )

    assert drop_dangling_qualifier_steps([title, qualified]) == [title, qualified]


def test_qualifier_wording_variants_are_dropped():
    anchor = _step(ActionType.assert_text, "Card with a short caption", assertion="Card")
    variant = _step(
        ActionType.assert_visible,
        "Card with a short caption",
        assertion="having a short caption",
    )

    assert drop_dangling_qualifier_steps([anchor, variant]) == [anchor]


def test_non_assertion_step_with_qualifier_text_is_kept():
    anchor = _step(ActionType.click, "Open card with details", target="card")
    other = _step(ActionType.other, "Open card with details", assertion="with details")

    assert drop_dangling_qualifier_steps([anchor, other]) == [anchor, other]


# --- 2. source order --------------------------------------------------------


def _mixed_order_steps() -> list[_LLMStep]:
    link = _step(ActionType.click, "1. Click the Terms link", target="Terms link")
    dismiss = _step(
        ActionType.click, "2. Dismiss terms after reading", target="Terms page"
    )
    opens = _step(
        ActionType.assert_visible,
        "3. Terms page opens",
        target="Terms page",
        assertion="Terms page opens",
    )
    back = _step(
        ActionType.assert_visible,
        "4. Return to language page",
        target="language page",
        assertion="language page",
    )
    return [link, opens, dismiss, back]


def test_restore_source_order_undoes_action_assertion_pairing():
    link, opens, dismiss, back = _mixed_order_steps()

    ordered = restore_source_order([link, opens, dismiss, back], SOURCE_ORDER_SCENARIO)

    assert ordered == [link, dismiss, opens, back]


def test_restore_source_order_is_stable_for_steps_from_one_line():
    first = _step(ActionType.click, "1. Click A and click B", target="A")
    second = _step(ActionType.click, "1. Click A and click B", target="B")
    later = _step(ActionType.click, "2. Click C", target="C")

    ordered = restore_source_order(
        [later, first, second], "1. Click A and click B\n2. Click C"
    )

    assert ordered == [first, second, later]


def test_restore_source_order_keeps_model_order_when_raw_text_is_not_in_scenario():
    steps = [
        _step(ActionType.click, "paraphrased two", target="B"),
        _step(ActionType.click, "1. Click A", target="A"),
    ]

    assert restore_source_order(steps, "1. Click A\n2. Click B") == steps


def test_restore_source_order_keeps_model_order_when_raw_text_is_ambiguous():
    steps = [
        _step(ActionType.click, "Click Next", target="Next"),
        _step(ActionType.click, "Click Next now", target="Next"),
        _step(ActionType.click, "Click Next", target="Next"),
    ]

    assert restore_source_order(steps, "Click Next now\nClick Next now") == steps


def test_restore_source_order_handles_repeated_identical_lines():
    first = _step(ActionType.click, "Click Next", target="Next")
    second = _step(ActionType.click, "Click Next", target="Next")
    done = _step(ActionType.click, "Click Done", target="Done")

    ordered = restore_source_order(
        [done, first, second], "Click Next\nClick Next\nClick Done"
    )

    assert ordered == [first, second, done]


def test_restore_source_order_without_scenario_text_keeps_order():
    steps = [_step(ActionType.click, "B", target="B"), _step(ActionType.click, "A", target="A")]

    assert restore_source_order(steps, "") == steps


def test_build_expected_trace_preserves_source_order_and_citation_binding():
    link, opens, dismiss, back = _mixed_order_steps()

    trace = _build_expected_trace(
        _trace(link, opens, dismiss, back), scenario_text=SOURCE_ORDER_SCENARIO
    )

    assert [step.raw_text for step in trace.steps] == [
        "1. Click the Terms link",
        "2. Dismiss terms after reading",
        "3. Terms page opens",
        "4. Return to language page",
    ]
    assert [step.index for step in trace.steps] == [1, 2, 3, 4]


def test_build_expected_trace_keeps_multiple_actions_then_multiple_assertions():
    scenario = (
        "1. Enter username\n2. Enter password\n3. Click Login\n"
        "4. Verify welcome banner\n5. Verify logout link"
    )
    steps = [
        _step(ActionType.type_text, "1. Enter username", target="username"),
        _step(ActionType.type_text, "2. Enter password", target="password"),
        _step(ActionType.click, "3. Click Login", target="Login button"),
        _step(
            ActionType.assert_visible,
            "4. Verify welcome banner",
            target="welcome banner",
        ),
        _step(ActionType.assert_visible, "5. Verify logout link", target="logout link"),
    ]

    trace = _build_expected_trace(_trace(*steps), scenario_text=scenario)

    assert [step.action for step in trace.steps] == [
        ActionType.type_text,
        ActionType.type_text,
        ActionType.click,
        ActionType.assert_visible,
        ActionType.assert_visible,
    ]
    assert [step.target for step in trace.steps] == [
        "username",
        "password",
        "Login button",
        "welcome banner",
        "logout link",
    ]


# --- 3. dismissal target ----------------------------------------------------


def test_dismiss_without_named_control_leaves_target_unresolved():
    step = _step(
        ActionType.click,
        "2. Dismiss terms after reading",
        target="Terms and conditions page",
    )

    assert normalize_action_and_target(step).target is None


def test_dismiss_keeps_source_text_when_target_is_cleared():
    step = _step(ActionType.click, "Dismiss terms after reading", target="terms page")

    assert normalize_action_and_target(step).raw_text == "Dismiss terms after reading"


def test_dismiss_with_named_control_keeps_target():
    step = _step(
        ActionType.click,
        "Dismiss the dialog by clicking the Close button",
        target="Close button",
    )

    assert normalize_action_and_target(step).target == "Close button"


def test_dismiss_clears_an_invented_control_name():
    step = _step(ActionType.click, "Close the popup", target="Close button")

    assert normalize_action_and_target(step).target is None


def test_click_on_close_button_keeps_target():
    step = _step(ActionType.click, "Click the Close button", target="Close button")

    assert normalize_action_and_target(step).target == "Close button"


def test_dismiss_other_action_is_cleared_and_then_classified_click():
    trace = _build_expected_trace(
        _trace(_step(ActionType.other, "Close the modal", target="modal")),
        scenario_text="Close the modal",
    )

    assert trace.steps[0].action == ActionType.click
    assert trace.steps[0].target is None


def test_dismissal_assertion_is_not_given_a_cleared_target():
    step = _step(
        ActionType.assert_visible,
        "Dismissing terms lands on language page",
        target="language page",
        assertion="language page",
    )

    assert normalize_action_and_target(step) == step


# --- 4. select versus click -------------------------------------------------


def test_select_verb_on_checkbox_becomes_select():
    step = _step(
        ActionType.click,
        "2.Select the terms and conditions checkbox and tap on join cta.",
        target="terms and conditions checkbox",
    )

    assert normalize_action_and_target(step).action == ActionType.select


def test_choose_verb_on_option_becomes_select():
    step = _step(ActionType.click, "Choose the monthly option", target="monthly option")

    assert normalize_action_and_target(step).action == ActionType.select


def test_tap_verb_on_cta_stays_click_in_same_sentence_as_a_select():
    step = _step(
        ActionType.click,
        "Select the terms checkbox and tap on join mentorship/Done cta.",
        target="join mentorship/Done cta",
    )

    assert normalize_action_and_target(step).action == ActionType.click


def test_click_verb_on_button_downgrades_select_to_click():
    step = _step(ActionType.select, "Click the Continue button", target="Continue button")

    assert normalize_action_and_target(step).action == ActionType.click


def test_select_with_input_value_is_not_downgraded():
    step = _step(
        ActionType.select,
        "Click the language button and pick English",
        target="language button",
        input_value="English",
    )

    assert normalize_action_and_target(step).action == ActionType.select


def test_select_verb_on_button_stays_click():
    step = _step(ActionType.click, "Select the Done button", target="Done button")

    assert normalize_action_and_target(step).action == ActionType.click


def test_click_verb_on_checkbox_stays_click():
    step = _step(ActionType.click, "Click the terms checkbox", target="terms checkbox")

    assert normalize_action_and_target(step).action == ActionType.click


def test_ambiguous_verbs_with_unmatched_target_leave_action_alone():
    step = _step(
        ActionType.click,
        "Tap Next then select the agree checkbox",
        target="paraphrased control",
    )

    assert normalize_action_and_target(step).action == ActionType.click


def test_assertion_actions_are_never_reclassified():
    step = _step(
        ActionType.assert_state,
        "Verify the terms checkbox is selected",
        target="terms checkbox",
        assertion="selected",
    )

    assert normalize_action_and_target(step) == step
