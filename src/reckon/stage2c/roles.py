"""Semantic roles for Stage 2C ACTION evidence validation.

Distinguishes application-launch obligations from authentication/login
obligations so password-login frames cannot prove a pure app-launch step.
Unknown / unrelated actions keep the OTHER role and existing behavior.
"""

import re
from enum import Enum

from reckon.schema import ActionType, ExpectedStep, ExpectedTrace

__all__ = [
    "SemanticRole",
    "expectation_semantic_role",
    "evidence_semantic_role",
    "role_mismatch_reason",
]


class SemanticRole(str, Enum):
    """Explicit Stage 2C obligation / evidence roles."""

    APP_LAUNCH = "app_launch"
    AUTHENTICATION = "authentication"
    OTHER = "other"


_LOGIN_CUE_RE = re.compile(
    r"\b(?:log\s*in|login|sign\s*in|signin|authenticate|credentials?)\b",
    re.IGNORECASE,
)
_APP_TARGET_RE = re.compile(r"\b(?:app|application)\b", re.IGNORECASE)
_APP_GENERIC_TOKENS = frozenset({"app", "application", "the", "a", "an"})
_ACCOUNT_TARGET_RE = re.compile(r"\b(?:test\s+)?account\b", re.IGNORECASE)

# Evidence that proves authentication / credential entry.
_LOGIN_EVIDENCE_RE = re.compile(
    r"\b(?:"
    r"login\s+with\s+password|"
    r"enter\s+your\s+password|"
    r"password\s+field|"
    r"masked\s+password|"
    r"otp|"
    r"welcome\s+back|"
    r"sign\s*in|"
    r"log\s*in|"
    r"credentials?"
    r")\b",
    re.IGNORECASE,
)

# Evidence that proves launching / opening an application.
_APP_LAUNCH_EVIDENCE_RE = re.compile(
    r"\b(?:"
    r"app\s+icon|"
    r"application\s+icon|"
    r"launching|"
    r"splash|"
    r"open(?:s|ing)?\s+the\s+(?:\w+\s+){0,2}app|"
    r"tap(?:s|ped)?\s+(?:the\s+)?(?:\w+\s+){0,2}app\s+icon"
    r")\b",
    re.IGNORECASE,
)


def _app_named_target(target: str | None) -> bool:
    if not target or not _APP_TARGET_RE.search(target):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", target.lower()))
    return bool(tokens - _APP_GENERIC_TOKENS)


def _scenario_has_split_account_step(
    expected: ExpectedTrace,
    step: ExpectedStep,
) -> bool:
    for other in expected.steps:
        if other.index == step.index or not other.target:
            continue
        if _ACCOUNT_TARGET_RE.search(other.target):
            return True
    return False


def expectation_semantic_role(
    step: ExpectedStep,
    expected: ExpectedTrace | None = None,
) -> SemanticRole:
    """Classify the expected step's Stage 2C obligation.

    Pure app-named targets (``Driver app``) are APP_LAUNCH even when raw_text
    mentions login — that keeps coarse ``Login to Driver app`` from accepting
    password frames as proof of launching the app. Explicit login in the
    *target* (``Driver app login``) is AUTHENTICATION. Split account macros
    keep the app half as APP_LAUNCH.
    """
    target = step.target or ""
    raw = step.raw_text or ""
    target_login = bool(_LOGIN_CUE_RE.search(target))
    raw_login = bool(_LOGIN_CUE_RE.search(raw))
    appish = _app_named_target(target)

    if appish:
        if expected is not None and _scenario_has_split_account_step(expected, step):
            return SemanticRole.APP_LAUNCH
        if target_login:
            return SemanticRole.AUTHENTICATION
        return SemanticRole.APP_LAUNCH

    if target_login or raw_login:
        return SemanticRole.AUTHENTICATION

    if step.action != ActionType.other:
        return SemanticRole.OTHER
    return SemanticRole.OTHER


def evidence_semantic_role(
    observed_description: str,
    visible_text: list[str] | None = None,
) -> SemanticRole:
    """Classify candidate frame evidence into a semantic role."""
    parts = [observed_description, *(visible_text or [])]
    combined = " ".join(part for part in parts if part)
    has_login = bool(_LOGIN_EVIDENCE_RE.search(combined))
    has_launch = bool(_APP_LAUNCH_EVIDENCE_RE.search(combined))
    if has_launch and not has_login:
        return SemanticRole.APP_LAUNCH
    if has_login and not has_launch:
        return SemanticRole.AUTHENTICATION
    if has_login and has_launch:
        # Prefer the more specific auth signal when both appear.
        return SemanticRole.AUTHENTICATION
    return SemanticRole.OTHER


def role_mismatch_reason(
    expectation_role: SemanticRole,
    evidence_role: SemanticRole,
) -> str | None:
    """Return a rejection reason when roles are exclusive; else None.

    OTHER never rejects: unknown roles retain existing Stage 2C behavior.
    """
    if expectation_role == SemanticRole.APP_LAUNCH:
        if evidence_role == SemanticRole.AUTHENTICATION:
            return "login_evidence_for_app_launch"
        return None
    if expectation_role == SemanticRole.AUTHENTICATION:
        if evidence_role == SemanticRole.APP_LAUNCH:
            return "app_launch_evidence_for_login"
        return None
    return None
