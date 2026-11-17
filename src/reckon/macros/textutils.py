"""Shared text helpers for Domain Macro triggers and onboarding lists."""

import re

_HUB_LANDING_RE = re.compile(
    r"\bland(?:s|ed)?\s+on\s+(?:the\s+)?hub\b",
    re.IGNORECASE,
)

_ONBOARDING_PROCESS_RE = re.compile(
    r"onboarding\s+process\s+(?:like[-–—:\s]+)(.+?)"
    r"(?:\s+(?:user\s+)?should\s+land|\s+and\s+(?:then\s+)?(?:the\s+)?user\s+lands|"
    r"\s+then\s+land|\.|$)",
    re.IGNORECASE | re.DOTALL,
)


def strip_step_number_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:(?:\d+[.)]|[-*•])\s+)?", "", text or "")


def onboarding_milestones_from_text(text: str) -> list[str] | None:
    """Return explicitly listed onboarding milestones, or None if not listed."""
    match = _ONBOARDING_PROCESS_RE.search(strip_step_number_prefix(text))
    if match is None:
        return None
    chunk = match.group(1).strip().strip("-–—: ")
    parts = [
        part.strip(" \t-–—")
        for part in re.split(r"\s*,\s*|\s+and\s+", chunk)
        if part.strip(" \t-–—")
    ]
    cleaned: list[str] = []
    for part in parts:
        cleaned.append(re.split(r"\s+user\s+", part, maxsplit=1)[0].strip())
    milestones = [part for part in cleaned if part]
    if len(milestones) < 2:
        return None
    return milestones
