"""Deterministic guards over the steps an LLM returned for Stage 1.

The model is asked to follow these rules in its prompt, but a prompt is a request,
not a guarantee. Each guard here re-checks one rule against the scenario text itself
and only ever *removes* or *reorders* what the model produced, or clears a field it
could not have grounded. None of them adds content, so nothing is invented.

The guards are generic over any step object that carries the fields they read
(``action``, ``target``, ``input_value``, ``assertion``, ``raw_text``), so they apply
to the parser's pre-index step model without importing it.
"""

import logging
import re
from typing import Protocol

from reckon.macros.textutils import strip_step_number_prefix
from reckon.schema import ActionType
from reckon.tolerance import is_choice_control_target

logger = logging.getLogger(__name__)

_ASSERTION_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.assert_visible, ActionType.assert_text, ActionType.assert_state}
)

_QUALIFIER_LEADS = r"(?:with|having|containing|including|plus|along\s+with)"
_QUALIFIER_DETERMINERS = r"(?:(?:an?|the|some|its|their|any)\s+)?"
_QUALIFIER_ADJECTIVES = r"(?:(?:short|brief|full|detailed|relevant|valid)\s+)?"
_GENERIC_NOUNS = (
    r"(?:description|details?|content|text|info|information|body|caption|"
    r"subtitle|label|message|value|copy)s?"
)
# A qualifier that names a kind of content but not the content itself, e.g.
# "with description". Without the actual value it is not a checkable assertion.
_DANGLING_QUALIFIER_RE = re.compile(
    rf"^\s*{_QUALIFIER_LEADS}\s+{_QUALIFIER_DETERMINERS}{_QUALIFIER_ADJECTIVES}"
    rf"{_GENERIC_NOUNS}\s*[.,;:]?\s*$",
    re.IGNORECASE,
)

_VERB_RE = re.compile(
    r"\b(?:"
    r"select(?:s|ed|ing)?|choose|chooses|choosing|chose|pick(?:s|ed|ing)?|"
    r"tick(?:s|ed|ing)?|"
    r"click(?:s|ed|ing)?|tap(?:s|ped|ping)?|press(?:es|ed|ing)?|"
    r"dismiss(?:es|ed|ing)?|close|closes|closed|closing|exit(?:s|ed|ing)?"
    r")\b",
    re.IGNORECASE,
)
_SELECT_PREFIXES = ("select", "choo", "chos", "pick", "tick")
_CLICK_PREFIXES = ("click", "tap", "press")
_DISMISS_PREFIXES = ("dismiss", "clos", "exit")

# Words that make a control explicit. When none appear in a dismissal's text, the
# scenario never says *what* is pressed to dismiss, so the step has no control target.
_CONTROL_EVIDENCE_RE = re.compile(
    r"\b(?:by|using|via|through|with)\b"
    r"|\b(?:click|tap|press)(?:s|ed|ing|ped|ping|es)?\b"
    r"|\b(?:button|icon|link|hyperlink|cta|arrow|cross|x)\b",
    re.IGNORECASE,
)
# Targets that are commands rather than choices: pressing one acts, it does not pick.
_COMMAND_TARGET_RE = re.compile(
    r"\b(?:button|link|hyperlink|cta|icon|arrow|tab)\b",
    re.IGNORECASE,
)

_VERB_SELECT = "select"
_VERB_CLICK = "click"
_VERB_DISMISS = "dismiss"
_CONTROL_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.click, ActionType.select}
)


class _StepFields(Protocol):
    action: ActionType
    target: str | None
    input_value: str | None
    assertion: str | None
    raw_text: str | None

    def model_copy(self, *, update: dict[str, object]) -> "_StepFields": ...



def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def drop_dangling_qualifier_steps[StepT: _StepFields](
    steps: list[StepT],
) -> list[StepT]:
    """Drop assertion steps whose whole assertion is an unfinished qualifier.

    "Earn when you mentor new drivers with description" says a description exists but
    never what it says; the model turned the tail into its own assertion. The step is
    removed only when a sibling step from the same source line survives, so the line
    stays covered and traceable.
    """
    kept: list[StepT] = []
    for step in steps:
        is_dangling = (
            step.action in _ASSERTION_ACTIONS
            and step.assertion is not None
            and _DANGLING_QUALIFIER_RE.match(step.assertion) is not None
        )
        has_sibling = step.raw_text is not None and any(
            other is not step and other.raw_text == step.raw_text for other in steps
        )
        if is_dangling and has_sibling:
            logger.info("Dropped step with dangling qualifier %r", step.assertion)
            continue
        kept.append(step)
    return kept


def _source_positions[StepT: _StepFields](
    steps: list[StepT],
    scenario_text: str,
) -> list[int] | None:
    """Position of each step's source line in the scenario, or None if any is unsure."""
    haystack = _collapse(scenario_text)
    if not haystack:
        return None
    steps_per_source: dict[str, int] = {}
    for step in steps:
        if not step.raw_text:
            return None
        key = _collapse(step.raw_text)
        steps_per_source[key] = steps_per_source.get(key, 0) + 1

    occurrences: dict[str, list[int]] = {}
    for key, step_count in steps_per_source.items():
        found = [m.start() for m in re.finditer(re.escape(key), haystack)]
        if len(found) != 1 and len(found) != step_count:
            return None
        occurrences[key] = found

    seen: dict[str, int] = {}
    positions: list[int] = []
    for step in steps:
        key = _collapse(step.raw_text or "")
        found = occurrences[key]
        nth = seen.get(key, 0)
        seen[key] = nth + 1
        positions.append(found[0] if len(found) == 1 else found[nth])
    return positions


def restore_source_order[StepT: _StepFields](
    steps: list[StepT],
    scenario_text: str,
) -> list[StepT]:
    """Return the steps in the order their source lines appear in the scenario.

    Steps are only moved when every step's ``raw_text`` can be located unambiguously
    in the scenario; otherwise the model's order is kept as-is. The sort is stable,
    so steps split from one line keep the order the model gave them.
    """
    positions = _source_positions(steps, scenario_text)
    if positions is None or positions == sorted(positions):
        return steps
    logger.info("Reordered steps to match the scenario's source order")
    order = sorted(range(len(steps)), key=lambda i: positions[i])
    return [steps[i] for i in order]


def _verb_class(word: str) -> str:
    lowered = word.lower()
    if lowered.startswith(_SELECT_PREFIXES):
        return _VERB_SELECT
    if lowered.startswith(_CLICK_PREFIXES):
        return _VERB_CLICK
    return _VERB_DISMISS


def _governing_verb_class(step: _StepFields) -> str | None:
    """Class of the verb that governs the step's target within its own source text.

    Uses the nearest verb before the target. When the target is not quoted verbatim
    from the text, a class is returned only if every verb in the text agrees.
    """
    text = strip_step_number_prefix(step.raw_text or "")
    verbs = [(m.start(), _verb_class(m.group())) for m in _VERB_RE.finditer(text)]
    if not verbs:
        return None
    target_at = text.lower().find(step.target.lower()) if step.target else -1
    if target_at >= 0:
        before = [cls for start, cls in verbs if start < target_at]
        return before[-1] if before else None
    classes = {cls for _, cls in verbs}
    return classes.pop() if len(classes) == 1 else None


def normalize_action_and_target[StepT: _StepFields](step: StepT) -> StepT:
    """Fix select/click classification and unresolved dismissal targets from the verb.

    - a select-class verb ("select", "choose", "pick") on a checkbox, radio, switch,
      dropdown or option is ``select``;
    - a click-class verb ("click", "tap", "press") on a button, link or similar
      command is ``click``;
    - a dismissal that never says what is pressed has no control, so its target is
      cleared instead of naming the dismissed page or dialog as if it were a control.
    """
    if step.action not in _CONTROL_ACTIONS | {ActionType.other}:
        return step
    verb_class = _governing_verb_class(step)
    if verb_class is None:
        return step

    if (
        step.action == ActionType.click
        and verb_class == _VERB_SELECT
        and is_choice_control_target(step.target)
    ):
        return step.model_copy(update={"action": ActionType.select})

    is_command = bool(step.target and _COMMAND_TARGET_RE.search(step.target))
    if (
        step.action == ActionType.select
        and verb_class == _VERB_CLICK
        and is_command
        and step.input_value is None
    ):
        return step.model_copy(update={"action": ActionType.click})

    text = strip_step_number_prefix(step.raw_text or "")
    if (
        verb_class == _VERB_DISMISS
        and step.action != ActionType.select
        and step.assertion is None
        and step.target is not None
        and _CONTROL_EVIDENCE_RE.search(text) is None
    ):
        return step.model_copy(update={"target": None})
    return step
