"""The Stage 2 interaction contract: what the extraction model may report and how.

Stage 2 describes what was visibly done to an on-screen element; Stage 1 describes what
a scenario asks for. They share IntentOperation, but the observed element kind is an
ElementKind, a separate enum the equivalence table relates to Stage 1's ObjectKind.
All allowed values are derived from those enums and CitationSurface, so the enums stay
the only place a value can be added. "unspecified" is never offered to the model.

An interaction's operation must be one its action type denotes, so typing, selection,
navigation, activation, dismissal and assertion evidence never stand in for each other.
A dismissal is seen as a click, so a click may be activate or dismiss. Action types that
denote no operation (scroll, hover, other) carry no interaction.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from reckon.schema import ActionType, CitationSurface, ElementKind, IntentOperation

INTERACTION_CONTRACT_VERSION = "1"

OBSERVED_OPERATIONS: tuple[str, ...] = tuple(
    member.value for member in IntentOperation if member != IntentOperation.unspecified
)
OBSERVED_KINDS: tuple[str, ...] = tuple(
    member.value for member in ElementKind if member != ElementKind.unspecified
)
CITATION_SURFACES: tuple[str, ...] = tuple(member.value for member in CitationSurface)

OPERATIONS_FOR_ACTION: dict[ActionType, tuple[IntentOperation, ...]] = {
    ActionType.click: (IntentOperation.activate, IntentOperation.dismiss),
    ActionType.type_text: (IntentOperation.enter,),
    ActionType.select: (IntentOperation.choose,),
    ActionType.navigate: (IntentOperation.traverse,),
    ActionType.assert_visible: (IntentOperation.assert_,),
    ActionType.assert_text: (IntentOperation.assert_,),
    ActionType.assert_state: (IntentOperation.assert_,),
}


def _operation_pairs() -> str:
    return "; ".join(
        f"{action.value}: {' or '.join(op.value for op in operations)}"
        for action, operations in OPERATIONS_FOR_ACTION.items()
    )

MAX_VOCABULARY_TOKEN_LENGTH = 32
_VOCABULARY_TOKEN = re.compile(rf"[a-z0-9_ -]{{1,{MAX_VOCABULARY_TOKEN_LENGTH}}}")


def normalise_token(value: str | None) -> str | None:
    """Casing and surrounding whitespace only. Any other difference is a different value."""
    if value is None:
        return None
    token = value.strip().lower()
    return token or None


def loggable_token(value: str | None) -> str | None:
    """A normalised token short and plain enough to record without leaking screen text."""
    token = normalise_token(value)
    return token if token is not None and _VOCABULARY_TOKEN.fullmatch(token) else None


def _allowed_values(values: tuple[str, ...]):
    def extend(schema: dict[str, Any]) -> None:
        for branch in schema.get("anyOf", []):
            if branch.get("type") == "string":
                branch["enum"] = list(values)

    return extend


class ReportedInteraction(BaseModel):
    """Interaction slots as the extraction model reports them, flat on each action.

    operation, kind and surface are listed as enums in the response schema but typed as
    strings, so an unsupported value rejects one interaction with a recorded reason
    instead of failing the whole window's extraction.
    """

    interaction_operation: str | None = Field(
        default=None,
        description=(
            f"One of: {', '.join(OBSERVED_OPERATIONS)}. Must match the action "
            f"({_operation_pairs()})."
        ),
        json_schema_extra=_allowed_values(OBSERVED_OPERATIONS),
    )
    interaction_operation_text: str | None = Field(
        default=None, description="Words in the cited surface naming the operation, verbatim."
    )
    interaction_kind: str | None = Field(
        default=None,
        description=f"Kind of element acted on. One of: {', '.join(OBSERVED_KINDS)}.",
        json_schema_extra=_allowed_values(OBSERVED_KINDS),
    )
    interaction_kind_text: str | None = Field(
        default=None,
        description="Words in the cited surface naming the element kind, verbatim.",
    )
    interaction_name: str | None = Field(
        default=None, description="Element name, verbatim from the cited surface."
    )
    interaction_product: str | None = Field(
        default=None, description="Product name, verbatim from the cited surface."
    )
    interaction_instance: str | None = Field(
        default=None, description="Instance label, verbatim from the cited surface."
    )
    interaction_value: str | None = Field(
        default=None, description="Literal value, verbatim from the cited surface."
    )
    interaction_surface: str | None = Field(
        default=None,
        description=f"Field every interaction text is cited from. One of: {', '.join(CITATION_SURFACES)}.",
        json_schema_extra=_allowed_values(CITATION_SURFACES),
    )


def interaction_prompt_section() -> str:
    """Prompt wording for interactions, built from the same constants as the schema."""
    return (
        "Optional interaction, only for what is visibly done to one element:\n"
        f"- interaction_operation is one of: {', '.join(OBSERVED_OPERATIONS)}, and must "
        f"match the action ({_operation_pairs()}). Other actions have no interaction.\n"
        f"- interaction_kind is one of: {', '.join(OBSERVED_KINDS)}.\n"
        f"- interaction_surface is one of: {', '.join(CITATION_SURFACES)}; every "
        "interaction text (operation_text, kind_text, name, product, instance, value) "
        "is copied verbatim from that one field. Never cite screen_description.\n"
        "- interaction_kind_text must be words in that field that name the element's "
        "kind. When the field has no such words, leave the whole interaction empty.\n"
        "- Leave the interaction empty when you cannot cite it. Never invent a value "
        "outside the lists above.\n"
    )
