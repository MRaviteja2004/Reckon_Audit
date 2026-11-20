"""Strip credentials from free text before it is persisted."""

import os
import re

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8

_SECRET_ENV_NAME_RE = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|USSO", re.IGNORECASE)
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_API_KEY_RE = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|usso[_-]?token|password|secret)"
    r"(\"?\s*[:=]\s*)(\"?)[^\s\"',}]{4,}"
)


def _secret_env_values() -> list[str]:
    return sorted(
        {
            value
            for name, value in os.environ.items()
            if _SECRET_ENV_NAME_RE.search(name) and len(value) >= MIN_SECRET_LENGTH
        },
        key=len,
        reverse=True,
    )


def redact_secrets(text: str) -> str:
    """Replace credential values from the environment and common credential shapes."""
    for value in _secret_env_values():
        text = text.replace(value, REDACTED)
    text = _BEARER_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    text = _API_KEY_RE.sub(REDACTED, text)
    return _ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}", text)
