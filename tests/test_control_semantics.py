import pytest

from reckon.control_semantics import (
    choice_identity_conflicts,
    choice_pair_supports_step,
    choice_role,
    identity_tokens,
    state_contradicts_instruction,
    targetless_dismissal_supported,
)
from reckon.schema import ActionType, ExpectedStep, ObservedAction


def _obs(
    action: ActionType,
    target: str | None,
    *,
    index: int = 1,
    screen_description: str | None = None,
) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=index * 1000,
        action=action,
        target=target,
        confidence=0.9,
        screen_description=screen_description,
    )


def _step(
    action: ActionType,
    target: str | None,
    *,
    input_value: str | None = None,
    raw_text: str | None = None,
) -> ExpectedStep:
    return ExpectedStep(
        index=1,
        action=action,
        target=target,
        input_value=input_value,
        raw_text=raw_text,
    )


@pytest.mark.parametrize(
    ("target", "role"),
    [
        ("English checkbox", "checkbox"),
        ("Yes radio button", "radio"),
        ("Notifications switch", "switch"),
        ("Dark mode toggle", "switch"),
        ("India option", "option"),
        ("English option row", "option"),
        ("Country dropdown", "dropdown"),
        ("Switch account button", None),
        ("Add option button", None),
        ("Terms and Conditions link", None),
        ("Next arrow", None),
        ("English", None),
    ],
)
def test_choice_role_uses_head_noun(target: str, role: str | None):
    assert choice_role(target) == role


def test_identity_tokens_keep_single_character_identifier():
    assert identity_tokens("Plan A radio button") == ("plan", "a")


def test_identity_tokens_drop_only_leading_article():
    assert identity_tokens("the Terms and Conditions checkbox") == (
        "terms",
        "and",
        "conditions",
    )


@pytest.mark.parametrize(
    ("value", "observed_target"),
    [
        ("English", "English checkbox"),
        ("India", "India option"),
        ("monthly plan", "Monthly plan radio button"),
        ("notifications", "Notifications switch"),
    ],
)
def test_value_only_select_matches_choice_control(value: str, observed_target: str):
    step = _step(ActionType.select, value)

    assert choice_pair_supports_step(step, _obs(ActionType.click, observed_target))


@pytest.mark.parametrize(
    ("value", "observed_target"),
    [
        ("English", "Spanish checkbox"),
        ("Plan A", "Plan B radio button"),
        ("Dark mode", "Light mode switch"),
        ("Marketing emails", "SMS notifications checkbox"),
        ("English", "English button"),
        ("English", "English link"),
    ],
)
def test_select_rejects_sibling_or_command_control(value: str, observed_target: str):
    step = _step(ActionType.select, value)

    assert not choice_pair_supports_step(step, _obs(ActionType.click, observed_target))


def test_select_with_value_rejects_opening_the_dropdown():
    step = _step(ActionType.select, "language dropdown", input_value="English")

    assert not choice_pair_supports_step(
        step, _obs(ActionType.click, "Language dropdown")
    )


def test_select_is_ambiguous_when_two_controls_contain_the_value():
    step = _step(ActionType.select, "notifications")
    email = _obs(ActionType.click, "Email notifications switch", index=1)
    sms = _obs(ActionType.click, "SMS notifications switch", index=2)

    assert not choice_pair_supports_step(step, email, observed_actions=[email, sms])


def test_select_stays_unambiguous_when_the_same_control_repeats():
    step = _step(ActionType.select, "English checkbox")
    first = _obs(ActionType.click, "English checkbox", index=1)
    second = _obs(ActionType.click, "English checkbox", index=2)

    assert choice_pair_supports_step(step, first, observed_actions=[first, second])


def test_identity_conflict_detects_sibling_click_targets():
    step = _step(ActionType.click, "English checkbox")

    assert choice_identity_conflicts(step, _obs(ActionType.click, "Spanish checkbox"))


def test_identity_conflict_ignores_assertions():
    step = _step(ActionType.assert_visible, "English checkbox")

    assert not choice_identity_conflicts(
        step, _obs(ActionType.assert_visible, "Spanish checkbox")
    )


def test_identity_conflict_allows_observed_label_with_extra_context():
    step = _step(ActionType.click, "terms and conditions checkbox")

    assert not choice_identity_conflicts(
        step,
        _obs(ActionType.click, "I agree to the Terms and Conditions checkbox"),
    )


def test_missing_state_does_not_contradict_instruction():
    step = _step(ActionType.click, "Notifications switch", raw_text="Turn off notifications")

    assert not state_contradicts_instruction(
        step, _obs(ActionType.click, "Notifications switch")
    )


def test_mixed_state_does_not_contradict_instruction():
    step = _step(ActionType.click, "English checkbox", raw_text="Check English")

    assert not state_contradicts_instruction(
        step,
        _obs(
            ActionType.click,
            "English checkbox",
            screen_description="English checked, Spanish unchecked",
        ),
    )


def test_targetless_dismissal_requires_close_control():
    step = _step(ActionType.click, None, raw_text="Dismiss terms after reading")

    assert not targetless_dismissal_supported(
        step,
        _obs(
            ActionType.click,
            "Next arrow button",
            screen_description="Terms page dismissed",
        ),
    )


def test_targetless_dismissal_accepts_close_with_dismissal_clause():
    step = _step(ActionType.click, None, raw_text="Dismiss terms after reading")

    assert targetless_dismissal_supported(
        step,
        _obs(
            ActionType.click,
            "X close button",
            screen_description="Terms page dismissed, back to language selection",
        ),
    )
