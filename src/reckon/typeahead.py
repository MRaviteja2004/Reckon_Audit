"""Deterministic recovery of expected ``select`` via typeahead sequences.

Does not change global ``actions_compatible``. A select is proved only when the
ObservedTrace shows type → matching suggestion click (optionally with an
intermediate suggestion list and a confirming field state) in order, inside a
bounded time span, with target and value checks. Typing alone never passes.
"""

import re
from dataclasses import dataclass

from reckon.schema import ActionType, ExpectedStep, ObservedAction, ObservedTrace
from reckon.tolerance import values_equivalent

# Max wall-clock span from first type_text to the completing click/confirm.
TYPEAHEAD_MAX_SPAN_MS = 30_000

# Field nouns that may stand in for a scenario "Location" / city select target.
_LOCATION_FIELD_TOKENS = frozenset(
    {
        "location",
        "city",
        "place",
        "where",
        "earn",
        "destination",
        "region",
        "area",
    }
)

_STRUCTURAL_FIELD_TOKENS = frozenset(
    {"input", "field", "box", "control", "optional"}
)

_SUGGESTION_TOKENS = frozenset(
    {"suggestion", "suggestions", "dropdown", "autocomplete", "result", "results"}
)


@dataclass(frozen=True)
class TypeaheadProof:
    """Indices that jointly prove an expected select via typeahead."""

    type_index: int
    click_index: int
    suggestion_index: int | None
    confirm_index: int | None

    @property
    def terminal_index(self) -> int:
        if self.confirm_index is not None:
            return self.confirm_index
        return self.click_index

    @property
    def claimed_indices(self) -> frozenset[int]:
        claimed = {self.type_index, self.click_index}
        if self.suggestion_index is not None:
            claimed.add(self.suggestion_index)
        if self.confirm_index is not None:
            claimed.add(self.confirm_index)
        return frozenset(claimed)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _selection_value_reflected(expected_value: str, text: str) -> bool:
    """True when suggestion/field text reflects the expected selected value.

    Exact (or ``values_equivalent``) full-string match is accepted. Suggestions
    may append locality after a comma ("Austin" → "Austin, TX, USA"), matched by
    an exact first-segment comparison only — fuzzy city-name near-misses such as
    Austin≈Houston must not pass.
    """
    if values_equivalent(expected_value, text):
        return True
    expected = expected_value.strip().lower()
    observed = text.strip().lower()
    if not expected or not observed:
        return False
    for part in re.split(r"[\n;/|]+", observed):
        head = part.split(",", 1)[0].strip()
        if head == expected:
            return True
    return (
        re.search(
            rf"(?:^|[\n\s]){re.escape(expected)}(?:,|\s|$)",
            observed,
        )
        is not None
    )


def _field_matches_select_target(expected_target: str, observed_target: str) -> bool:
    """True when a typed field can belong to the expected select control."""
    expected_tokens = _tokens(expected_target) - _STRUCTURAL_FIELD_TOKENS
    observed_tokens = _tokens(observed_target) - _STRUCTURAL_FIELD_TOKENS
    if not expected_tokens or not observed_tokens:
        return False
    if expected_tokens & observed_tokens:
        return True
    # Location-style selects are often typed into city / "where to earn" fields.
    if expected_tokens & _LOCATION_FIELD_TOKENS:
        return bool(observed_tokens & _LOCATION_FIELD_TOKENS)
    return False


def _is_suggestion_click(action: ObservedAction, expected_value: str) -> bool:
    if action.action != ActionType.click or not action.target:
        return False
    target_tokens = _tokens(action.target)
    # Prefer explicit suggestion chrome; otherwise require the click label itself
    # to carry the selected value (common for "Austin, TX, USA suggestion").
    if not (
        target_tokens & _SUGGESTION_TOKENS
        or _selection_value_reflected(expected_value, action.target)
    ):
        return False
    return _selection_value_reflected(expected_value, action.target)


def _is_suggestion_list(action: ObservedAction, expected_value: str) -> bool:
    if action.action not in {
        ActionType.assert_visible,
        ActionType.assert_text,
        ActionType.assert_state,
    }:
        return False
    blob_parts = [action.target or ""]
    if action.assertion_evidence is not None:
        blob_parts.append(action.assertion_evidence.visible_text or "")
        blob_parts.append(action.assertion_evidence.description or "")
    if action.screen_description:
        blob_parts.append(action.screen_description)
    blob = " ".join(blob_parts)
    tokens = _tokens(blob)
    if not (tokens & _SUGGESTION_TOKENS):
        # Still allow a plain list observation that names the value.
        if action.assertion_evidence is None:
            return False
    return _selection_value_reflected(expected_value, blob)


def _is_field_confirmation(
    action: ObservedAction,
    *,
    expected_target: str,
    expected_value: str,
) -> bool:
    if action.action not in {
        ActionType.assert_visible,
        ActionType.assert_text,
        ActionType.assert_state,
    }:
        return False
    if action.target and not _field_matches_select_target(
        expected_target, action.target
    ):
        return False
    texts: list[str] = []
    if action.input_value:
        texts.append(action.input_value)
    if action.assertion_evidence is not None:
        if action.assertion_evidence.visible_text:
            texts.append(action.assertion_evidence.visible_text)
        if action.assertion_evidence.description:
            texts.append(action.assertion_evidence.description)
    return any(_selection_value_reflected(expected_value, text) for text in texts)


def find_typeahead_selection_proof(
    step: ExpectedStep,
    observed: ObservedTrace,
    *,
    window: tuple[int, int],
    claimed_obs_indices: set[int],
    max_span_ms: int = TYPEAHEAD_MAX_SPAN_MS,
) -> TypeaheadProof | None:
    """Return a typeahead proof for an expected select, or None.

    Requires chronological ``type_text`` then suggestion ``click`` reflecting
    ``step.input_value``. An intermediate suggestion list and a later field
    confirmation strengthen the proof but typing alone is never enough.
    """
    if step.action != ActionType.select:
        return None
    if not step.target or not step.input_value:
        return None

    start, end = window
    in_window = [
        action
        for action in observed.actions
        if start <= action.index <= end and action.index not in claimed_obs_indices
    ]
    if not in_window:
        return None

    type_candidates = [
        action
        for action in in_window
        if action.action == ActionType.type_text
        and action.target
        and _field_matches_select_target(step.target, action.target)
        and (
            action.input_value is None
            or _selection_value_reflected(step.input_value, action.input_value)
            or step.input_value.lower().startswith(action.input_value.lower())
            or action.input_value.lower().startswith(step.input_value.lower())
        )
    ]
    if not type_candidates:
        return None

    for type_action in type_candidates:
        later = [action for action in in_window if action.index > type_action.index]
        click_candidates = [
            action
            for action in later
            if _is_suggestion_click(action, step.input_value)
            and action.timestamp_ms - type_action.timestamp_ms <= max_span_ms
        ]
        for click_action in click_candidates:
            between = [
                action
                for action in later
                if type_action.index < action.index < click_action.index
            ]
            suggestion = next(
                (
                    action
                    for action in between
                    if _is_suggestion_list(action, step.input_value)
                ),
                None,
            )
            after_click = [
                action
                for action in later
                if action.index > click_action.index
                and action.timestamp_ms - type_action.timestamp_ms <= max_span_ms
            ]
            confirm = next(
                (
                    action
                    for action in after_click
                    if _is_field_confirmation(
                        action,
                        expected_target=step.target,
                        expected_value=step.input_value,
                    )
                ),
                None,
            )
            # Prefer proofs that include confirmation when available; still accept
            # type + matching suggestion click without a later assert.
            return TypeaheadProof(
                type_index=type_action.index,
                click_index=click_action.index,
                suggestion_index=(
                    suggestion.index if suggestion is not None else None
                ),
                confirm_index=confirm.index if confirm is not None else None,
            )
    return None
