"""The Stage 1 intent contract: what the model may offer and how it is described to it.

The allowed values are derived from IntentOperation and ObjectKind, so the enums stay
the only place a value can be added. "unspecified" is a value the application assigns
to an intent it cannot classify; the model is never offered it.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from reckon.schema import IntentOperation, ObjectKind

INTENT_CONTRACT_VERSION = "2"

EXPECTED_INTENT_OPERATIONS: tuple[str, ...] = tuple(
    member.value for member in IntentOperation if member != IntentOperation.unspecified
)
EXPECTED_INTENT_KINDS: tuple[str, ...] = tuple(
    member.value for member in ObjectKind if member != ObjectKind.unspecified
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
    """A normalised token short and plain enough to record without leaking scenario text."""
    token = normalise_token(value)
    return token if token is not None and _VOCABULARY_TOKEN.fullmatch(token) else None


def _allowed_values(values: tuple[str, ...]):
    def extend(schema: dict[str, Any]) -> None:
        for branch in schema.get("anyOf", []):
            if branch.get("type") == "string":
                branch["enum"] = list(values)

    return extend


class ReportedIntent(BaseModel):
    """Slot strings as the model reports them. Python validates them, or drops the intent.

    operation and kind are listed as enums in the response schema but typed as strings, so
    an unsupported value rejects one intent with a recorded reason instead of failing the
    whole Stage 1 response.
    """

    operation: str | None = Field(
        default=None,
        description=f"UI operation. One of: {', '.join(EXPECTED_INTENT_OPERATIONS)}.",
        json_schema_extra=_allowed_values(EXPECTED_INTENT_OPERATIONS),
    )
    operation_text: str | None = Field(
        default=None,
        description="The words in the step that express the operation, copied verbatim.",
    )
    kind: str | None = Field(
        default=None,
        description=f"Kind of object acted on. One of: {', '.join(EXPECTED_INTENT_KINDS)}.",
        json_schema_extra=_allowed_values(EXPECTED_INTENT_KINDS),
    )
    kind_text: str | None = Field(
        default=None,
        description="The words in the step that name the kind of object, copied verbatim.",
    )
    name: str | None = Field(default=None, description="Object name, copied verbatim.")
    product: str | None = Field(default=None, description="Product name, copied verbatim.")
    instance: str | None = Field(default=None, description="Instance label, copied verbatim.")
    value: str | None = Field(default=None, description="Literal value, copied verbatim.")


def intent_prompt_section() -> str:
    """The prompt wording for the intent block, built from the same constants as the schema."""
    return (
        "Structured intent, when you can cite it from the step line:\n"
        f"- operation is one of: {', '.join(EXPECTED_INTENT_OPERATIONS)}.\n"
        f"- kind is one of: {', '.join(EXPECTED_INTENT_KINDS)}.\n"
        "- operation_text, kind_text, name, product, instance, and value are copied "
        "verbatim from that step line; do not paraphrase them.\n"
        "- Leave the whole intent empty when you cannot cite it. Never invent a "
        "value outside the lists above.\n"
    )
