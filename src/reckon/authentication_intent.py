"""Whether an expected step claims authentication.

Launching an application shows a window opening, not an account being signed in,
so a launch cannot stand in for a step whose claim is the sign-in itself. The
claim is read from the step's own wording, not from its parsed action or intent:
Stage 1 may file "log in to the app" under ``other`` with no operation at all.
"""

import re

from reckon.schema import ExpectedStep

_AUTHENTICATION_RE = re.compile(
    r"\b(?:"
    r"log(?:s|ged|ging)?[\s-]*in"
    r"|login"
    r"|sign(?:s|ed|ing)?[\s-]*(?:in|on)"
    r"|authenticat\w*"
    r")\b",
    re.IGNORECASE,
)


def claims_authentication(step: ExpectedStep) -> bool:
    """True when the step's wording asks for a login, sign-in or authentication."""
    return any(
        text is not None and _AUTHENTICATION_RE.search(text) is not None
        for text in (step.raw_text, step.target, step.assertion)
    )
