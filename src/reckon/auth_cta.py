"""Narrow authentication CTA synonym matching for Stage 3 recovery.

Explicit Sign in / Login equivalence only — not a general thesaurus.
"""

import re

from reckon.schema import ActionType, ExpectedStep, ObservedAction, ObservedTrace

AUTH_CTA_RECOVERY_REASON = "auth_cta_synonym_recovery"

# Contiguous token phrases that identify an authentication CTA label.
_AUTH_CTA_PHRASES: tuple[tuple[str, ...], ...] = (
    ("sign", "in"),
    ("log", "in"),
    ("login",),
)

# Widget chrome stripped before phrase detection.
_AUTH_CTA_CHROME = frozenset({
    "button",
    "btn",
    "cta",
    "link",
    "control",
})

# Tokens that mean the control is not a bare auth CTA (other roles / fields).
_AUTH_CTA_CONFLICT_TOKENS = frozenset({
    "cancel",
    "submit",
    "continue",
    "next",
    "back",
    "close",
    "dismiss",
    "username",
    "password",
    "email",
    "field",
    "input",
    "title",
    "heading",
    "checkbox",
    "radio",
    "toggle",
    "option",
    "search",
})

# Descriptive leftovers allowed after removing the auth phrase + chrome
# (e.g. "Sign in button in GitHub header").
_AUTH_CTA_ALLOWED_CONTEXT = frozenset({
    "in",
    "on",
    "the",
    "a",
    "an",
    "github",
    "header",
    "homepage",
    "home",
    "nav",
    "navbar",
    "menu",
    "app",
    "page",
    "screen",
    "form",
    "dialog",
    "modal",
    "primary",
    "secondary",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def auth_cta_canonical(target: str | None) -> str | None:
    """Return canonical auth-CTA id, or None when the target is not an auth CTA.

    A target qualifies only when it contains exactly one supported auth phrase
    (Sign in / Log in / Login) and any leftover tokens are chrome or light
    descriptive context — not a conflicting control role or text-field noun.
    """
    if not target:
        return None
    tokens = _tokens(target)
    if not tokens:
        return None

    matched_phrase: tuple[str, ...] | None = None
    match_at: int | None = None
    for phrase in _AUTH_CTA_PHRASES:
        length = len(phrase)
        for index in range(0, len(tokens) - length + 1):
            if tuple(tokens[index:index + length]) == phrase:
                if matched_phrase is not None:
                    return None
                matched_phrase = phrase
                match_at = index
                break
        if matched_phrase is not None and match_at is not None:
            # Prefer the first phrase family hit; do not also accept another.
            break
    if matched_phrase is None or match_at is None:
        return None

    consumed = set(range(match_at, match_at + len(matched_phrase)))
    leftovers = [
        token
        for index, token in enumerate(tokens)
        if index not in consumed and token not in _AUTH_CTA_CHROME
    ]
    for token in leftovers:
        if token in _AUTH_CTA_CONFLICT_TOKENS and token not in _AUTH_CTA_ALLOWED_CONTEXT:
            return None
        if token not in _AUTH_CTA_ALLOWED_CONTEXT and token not in _AUTH_CTA_CHROME:
            # Unknown leftover — reject rather than invent synonyms.
            return None
    return "auth.signin"


def auth_ctas_equivalent(expected_target: str | None, observed_target: str | None) -> bool:
    """True when both targets canonicalize to the same auth CTA synonym class."""
    left = auth_cta_canonical(expected_target)
    right = auth_cta_canonical(observed_target)
    return left is not None and left == right


def find_auth_cta_synonym_match(
    step: ExpectedStep,
    observed: ObservedTrace,
    *,
    window: tuple[int, int],
    claimed_obs_indices: set[int],
    preferred_obs_index: int | None = None,
) -> ObservedAction | None:
    """Return a click in-window whose auth-CTA label is synonymous with ``step``.

    Requires click↔click. Typing-only / field controls never match.
    """
    if step.action != ActionType.click:
        return None
    if auth_cta_canonical(step.target) is None:
        return None

    start, end = window
    candidates = [
        action
        for action in observed.actions
        if start <= action.index <= end
        and action.index not in claimed_obs_indices
        and action.action == ActionType.click
        and auth_ctas_equivalent(step.target, action.target)
    ]
    if not candidates:
        return None

    if preferred_obs_index is not None:
        preferred = next(
            (action for action in candidates if action.index == preferred_obs_index),
            None,
        )
        if preferred is not None:
            return preferred
    return candidates[0]
