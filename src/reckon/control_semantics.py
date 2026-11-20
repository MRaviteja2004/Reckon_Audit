"""Deterministic control semantics for baseline alignment.

Action labels from Stage 1 and Stage 2 are constraints, not proof. A pair whose
labels differ can still be the same event (a checkbox tap is a selection), and a
pair whose labels agree can still be wrong (a tap on a sibling checkbox). These
rules decide both from target identity and recorded state, never from fuzzy
similarity alone.

Only checkbox, radio, switch, option and dropdown roles are handled, and only
the role named by a control's head noun counts.
"""

import re
from collections.abc import Callable

from reckon.schema import ActionType, ExpectedStep, ObservedAction

DISMISS_ACTION_RE = re.compile(
    r"^\s*(?:(?:\d+[.)]|[-*•])\s+)?(?:dismiss|close)\b",
    re.IGNORECASE,
)
CLOSE_CONTROL_RE = re.compile(r"\b(?:close|dismiss)\b", re.IGNORECASE)

_DISMISS_OBJECT_RE = re.compile(
    r"^\s*(?:(?:\d+[.)]|[-*•])\s+)?(?:dismiss|close)\w*\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_DISMISS_MANNER_RE = re.compile(
    r"\s+\b(?:after|once|when|while|before)\b.*$",
    re.IGNORECASE,
)
# Words that name a surface but not which one. "Dismiss the page" cannot be
# tied to a particular overlay, so those tokens are not an identity.
_DISMISS_FUNCTION_WORDS = frozenset({"the", "a", "an", "this", "that", "it", "its"})
_GENERIC_SURFACE_WORDS = frozenset({"page", "screen", "view"})

# Head nouns that make a label a command. A choice role has to be the head,
# so "switch" or "option" earlier in "Switch account button" does not count.
_COMMAND_HEADS = frozenset({"button", "link", "cta", "icon", "tab", "arrow", "banner"})
_CHOICE_ROLE_WORDS = frozenset({
    "checkbox", "check", "box", "radio", "button", "switch", "toggle",
    "dropdown", "drop", "down", "option", "options", "row", "rows", "combo",
})
_CHOICE_VERB_WORDS = frozenset({
    "select", "selects", "selected", "choose", "chooses", "chose", "chosen",
    "pick", "picks", "picked", "tick", "ticks", "ticked",
    "tap", "taps", "tapped", "click", "clicks", "clicked",
})
ROLE_DROPDOWN = "dropdown"
_CHOICE_EVENT_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.click, ActionType.select}
)

_NEGATIVE_INSTRUCTION_RE = re.compile(
    r"\b(?:uncheck(?:ed|ing)?|deselect(?:ed|ing)?|untick(?:ed|ing)?|"
    r"disabl(?:e|es|ed|ing)|"
    r"turn(?:s|ed|ing)?\s+off|switch(?:es|ed|ing)?\s+off|"
    r"toggle(?:s|d|ing)?\s+off)\b",
    re.IGNORECASE,
)
_POSITIVE_INSTRUCTION_RE = re.compile(
    r"\b(?:check(?:s|ed|ing)?|tick(?:s|ed|ing)?|select(?:s|ed|ing)?|"
    r"choos(?:e|es|ing)|enabl(?:e|es|ed|ing)|"
    r"turn(?:s|ed|ing)?\s+on|switch(?:es|ed|ing)?\s+on|"
    r"toggle(?:s|d|ing)?\s+on)\b",
    re.IGNORECASE,
)
# Word boundaries keep "unchecked" from also counting as "checked".
_NEGATIVE_STATE_RE = re.compile(
    r"\b(?:unchecked|deselected|turned\s+off|toggled\s+off|switched\s+off|disabled)\b"
    r"|\bnot\s+(?:checked|selected|enabled|on)\b",
    re.IGNORECASE,
)
_POSITIVE_STATE_RE = re.compile(
    r"\b(?:checked|selected|turned\s+on|toggled\s+on|switched\s+on|enabled)\b",
    re.IGNORECASE,
)
_ON = "on"
_OFF = "off"

_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[.;,]|\band\b)\s*", re.IGNORECASE)
_CLAUSE_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|failed|unable|cannot|can't|didn't|did\s+not)\b",
    re.IGNORECASE,
)
_STILL_OPEN_RE = re.compile(
    r"\b(?:remains?|remained|still|stays?|left)\s+(?:open|visible)\b",
    re.IGNORECASE,
)
# Result words only. "back to the terms page" names the destination, not the
# overlay that closed, so it is not a dismissal clause.
_DISMISS_RESULT_RE = re.compile(
    r"\b(?:dismiss(?:ed|es|ing)|clos(?:ed|es|ing)|disappear(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)


def _text_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def identity_tokens(text: str) -> tuple[str, ...]:
    """Distinctive tokens of a choice value.

    Leading articles are dropped. Role and verb words are dropped wherever
    they sit. An internal article, including the identifier in "Plan A", stays.
    A lone "A" or "B" is an identifier, not an article.
    """
    tokens = _text_tokens(text)
    while tokens and tokens[0] in {"the", "an"}:
        tokens.pop(0)
    if len(tokens) > 1 and tokens[0] == "a":
        tokens.pop(0)
    return tuple(
        token
        for token in tokens
        if token not in _CHOICE_ROLE_WORDS and token not in _CHOICE_VERB_WORDS
    )


def choice_role(target: str | None) -> str | None:
    """Role of a choice control from its head noun, or None for a command.

    "radio button" is a choice role. "Switch account button" is a command
    because its head is "button", not "switch".
    """
    tokens = _text_tokens(target or "")
    if not tokens:
        return None
    tail = tuple(tokens[-2:]) if len(tokens) >= 2 else ()
    if tail == ("radio", "button") or tokens[-1] == "radio":
        return "radio"
    if tail == ("check", "box") or tokens[-1] == "checkbox":
        return "checkbox"
    if tail in {("combo", "box"), ("drop", "down")} or tokens[-1] == "combobox":
        return ROLE_DROPDOWN
    if tokens[-1] in _COMMAND_HEADS:
        return None
    if tokens[-1] in {"switch", "toggle"}:
        return "switch"
    if tokens[-1] == "option" or tail == ("option", "row"):
        return "option"
    if "option" in tokens and tokens[-1] in {"dropdown", "down"}:
        return "option"
    if tokens[-1] in {"dropdown", "down"}:
        return ROLE_DROPDOWN
    return None


def _instruction_polarity(raw_text: str | None) -> str | None:
    """"off" for uncheck/turn off wording, "on" for select/turn on, else None."""
    text = raw_text or ""
    if _NEGATIVE_INSTRUCTION_RE.search(text):
        return _OFF
    if _POSITIVE_INSTRUCTION_RE.search(text):
        return _ON
    return None


def _observed_state(obs: ObservedAction) -> str | None:
    """Recorded resulting state, or None when absent or mixed."""
    parts = [obs.screen_description or ""]
    if obs.assertion_evidence is not None:
        parts.append(obs.assertion_evidence.description or "")
    text = " ".join(parts)
    negative = _NEGATIVE_STATE_RE.search(text) is not None
    positive = _POSITIVE_STATE_RE.search(text) is not None
    if negative and not positive:
        return _OFF
    if positive and not negative:
        return _ON
    return None


def state_contradicts_instruction(exp: ExpectedStep, obs: ObservedAction) -> bool:
    """A choice control whose recorded state is the opposite of the instruction.

    "Turn on notifications" against "notifications switched off" contradicts.
    Missing or mixed state never contradicts. Assertion steps are judged by
    their asserted text instead.
    """
    if exp.action not in _CHOICE_EVENT_ACTIONS or obs.action not in _CHOICE_EVENT_ACTIONS:
        return False
    if choice_role(obs.target) is None:
        return False
    polarity = _instruction_polarity(exp.raw_text)
    state = _observed_state(obs)
    return polarity is not None and state is not None and polarity != state


def choice_identity_conflicts(exp: ExpectedStep, obs: ObservedAction) -> bool:
    """A click or select landed on a different choice control.

    "English checkbox" against "Spanish checkbox" conflicts even though the
    shared role word makes them look alike.
    """
    if exp.action not in _CHOICE_EVENT_ACTIONS or obs.action not in _CHOICE_EVENT_ACTIONS:
        return False
    if choice_role(exp.target) is None or choice_role(obs.target) is None:
        return False
    expected = set(identity_tokens(exp.target or ""))
    observed = set(identity_tokens(obs.target or ""))
    return bool(expected) and bool(observed) and not expected <= observed


def _identity_contained(value: str, observed_target: str) -> bool:
    identity = identity_tokens(value)
    return bool(identity) and set(identity) <= set(identity_tokens(observed_target))


def _select_by_click(exp: ExpectedStep, obs: ObservedAction) -> bool:
    if (
        exp.action != ActionType.select
        or obs.action != ActionType.click
        or obs.target is None
        or (exp.input_value is None and exp.target is None)
    ):
        return False
    if _instruction_polarity(exp.raw_text) == _OFF:
        return False
    role = choice_role(obs.target)
    if role is None:
        return False
    # Opening the dropdown shows the list. It does not show that a value was chosen.
    if exp.input_value is not None and role == ROLE_DROPDOWN:
        return False
    if not _identity_contained(exp.input_value or exp.target or "", obs.target):
        return False
    return _observed_state(obs) != _OFF


def _click_by_select(exp: ExpectedStep, obs: ObservedAction) -> bool:
    """A tap on a checkbox, radio, switch or option recorded as ``select``.

    A dropdown is excluded: a select on the list does not show which control the
    expected click named.
    """
    if (
        exp.action != ActionType.click
        or obs.action != ActionType.select
        or exp.target is None
        or obs.target is None
        or exp.input_value is not None
    ):
        return False
    if _instruction_polarity(exp.raw_text) == _OFF:
        return False
    expected_role = choice_role(exp.target)
    observed_role = choice_role(obs.target)
    if expected_role in {None, ROLE_DROPDOWN} or observed_role in {None, ROLE_DROPDOWN}:
        return False
    if not _identity_contained(exp.target, obs.target):
        return False
    return _observed_state(obs) != _OFF


def choice_pair_supported(exp: ExpectedStep, obs: ObservedAction) -> bool:
    """This click/select pair is one choice event, ignoring other observations."""
    return _select_by_click(exp, obs) or _click_by_select(exp, obs)


def _ambiguous(
    exp: ExpectedStep,
    observed_actions: list[ObservedAction],
    supports: Callable[[ExpectedStep, ObservedAction], bool],
) -> bool:
    """True when two different controls both satisfy the step."""
    identities = {
        identity_tokens(action.target)
        for action in observed_actions
        if action.target is not None and supports(exp, action)
    }
    return len(identities) > 1


def choice_pair_supports_step(
    exp: ExpectedStep,
    obs: ObservedAction,
    *,
    observed_actions: list[ObservedAction] | None = None,
) -> bool:
    """A choice click or select proves the step, and no other control also does.

    The observed head noun has to be a choice role and every identity token has
    to occur in that control. When two distinct controls qualify, neither is
    accepted.
    """
    if not choice_pair_supported(exp, obs):
        return False
    return observed_actions is None or not _ambiguous(
        exp, observed_actions, choice_pair_supported
    )


def is_targetless_dismiss(exp: ExpectedStep) -> bool:
    """A dismiss/close step whose scenario names no control."""
    return (
        exp.action == ActionType.click
        and exp.target is None
        and exp.raw_text is not None
        and DISMISS_ACTION_RE.search(exp.raw_text) is not None
    )


def _dismissed_object_tokens(raw_text: str) -> list[str]:
    """Tokens naming what is dismissed, taken from the step sentence.

    The manner clause ("after reading") is not the overlay. A sentence that
    only says "page" or "screen" does not identify one either.
    """
    match = _DISMISS_OBJECT_RE.match(raw_text.strip())
    if match is None:
        return []
    body = _DISMISS_MANNER_RE.sub("", match.group("body"))
    return [
        token
        for token in _text_tokens(body)
        if token not in _DISMISS_FUNCTION_WORDS
        and token not in _GENERIC_SURFACE_WORDS
    ]


def _positive_dismissal_clauses(text: str) -> list[str]:
    """Clauses that record a dismissal and are not negated or still open."""
    clauses: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        piece = clause.strip()
        if not piece or _STILL_OPEN_RE.search(piece):
            continue
        if _DISMISS_RESULT_RE.search(piece) is None:
            continue
        if _CLAUSE_NEGATION_RE.search(piece):
            continue
        clauses.append(piece)
    return clauses


def targetless_dismissal_supported(exp: ExpectedStep, obs: ObservedAction) -> bool:
    """A close control whose dismissal clause names the overlay.

    The dismissed page stays out of ``exp.target``. Identity is read from the
    clause that records the close, not from the rest of the description.
    """
    if not is_targetless_dismiss(exp) or exp.raw_text is None:
        return False
    if (
        obs.action != ActionType.click
        or obs.target is None
        or CLOSE_CONTROL_RE.search(obs.target) is None
    ):
        return False
    object_tokens = _dismissed_object_tokens(exp.raw_text)
    if not object_tokens:
        return False
    texts = [obs.screen_description or ""]
    if obs.assertion_evidence is not None:
        texts.append(obs.assertion_evidence.description or "")
    for text in texts:
        for clause in _positive_dismissal_clauses(text):
            clause_tokens = _text_tokens(clause)
            if all(token in clause_tokens for token in object_tokens):
                return True
    return False
