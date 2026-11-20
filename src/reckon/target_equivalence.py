"""Exact target equivalences that fuzzy similarity cannot see.

Each rule is an exact match over a closed vocabulary, never a looser threshold:
a pair either meets the rule word for word or the ordinary similarity check
decides it as before.
"""

import re

from reckon.schema import ActionType, ExpectedStep, ObservedAction

_ALTERNATIVE_ACTIONS = frozenset({ActionType.type_text, ActionType.select})
_FIELD_WORDS = frozenset({"field", "input"})
_ALTERNATIVE_SEPARATOR = re.compile(r"\s+or\s+|/", re.IGNORECASE)

_BROWSER_CHROME_ACTIONS = frozenset({ActionType.click, ActionType.type_text})
_BROWSER_CHROME_PHRASES = frozenset({
    ("address", "bar"),
    ("url", "bar"),
    ("location", "bar"),
    ("omnibox",),
    ("browser", "search", "bar"),
    ("browser", "address", "bar"),
})

_SITE_NOUNS = frozenset({"website", "site", "homepage"})
_IGNORED_HOST_LABELS = frozenset({"www", "m"})
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://")
_HOST = re.compile(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _alternative_label_matches(expected: str, observed: str) -> bool:
    alternatives = _ALTERNATIVE_SEPARATOR.split(observed)
    if len(alternatives) < 2:
        return False
    meaning = set(_words(expected)) - _FIELD_WORDS
    if not meaning:
        return False
    return any(set(_words(label)) - _FIELD_WORDS == meaning for label in alternatives)


def _is_browser_chrome(target: str) -> bool:
    words = [word for word in _words(target) if word != "the"]
    return tuple(words) in _BROWSER_CHROME_PHRASES


def _registrable_label(url: str) -> str | None:
    """Return the label left of the public suffix, read from the right of the host.

    Multi-part suffixes such as co.uk yield the wrong label, which can only fail a
    match, never create one.
    """
    rest = _SCHEME.sub("", url.strip().lower())
    host = re.split(r"[/?#:]", rest, maxsplit=1)[0]
    if "@" in host or not _HOST.fullmatch(host):
        return None
    labels = [label for label in host.split(".") if label not in _IGNORED_HOST_LABELS]
    return labels[-2] if len(labels) >= 2 else None


def _site_navigation_matches(expected: str, observed_url: str) -> bool:
    words = _words(expected)
    brand = [word for word in words if word not in _SITE_NOUNS]
    if not brand or len(brand) == len(words):
        return False
    label = _registrable_label(observed_url)
    return label is not None and label.replace("-", "") == "".join(brand)


def target_exactly_equivalent(exp: ExpectedStep, obs: ObservedAction) -> bool:
    """True when exp and obs name the same target under one of the exact rules."""
    if exp.target is None or obs.target is None:
        return False
    alternatives = (
        exp.action in _ALTERNATIVE_ACTIONS
        and obs.action in _ALTERNATIVE_ACTIONS
        and _alternative_label_matches(exp.target, obs.target)
    )
    browser_chrome = (
        exp.action == obs.action
        and exp.action in _BROWSER_CHROME_ACTIONS
        and _is_browser_chrome(exp.target)
        and _is_browser_chrome(obs.target)
    )
    site_navigation = exp.action == obs.action == ActionType.navigate and (
        _site_navigation_matches(exp.target, obs.target)
    )
    return alternatives or browser_chrome or site_navigation
