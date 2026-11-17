"""Versioned Domain Macro definitions (data, not LLM prompts)."""

import re
from dataclasses import dataclass, field
from typing import Pattern

from reckon.schema import ActionType, StepObligation


@dataclass(frozen=True)
class MacroMember:
    """One slot a macro may emit into ExpectedTrace.steps."""

    member_id: str
    obligation: StepObligation
    action: ActionType
    target: str
    assertion: str | None = None
    # If True, member is only considered when its cue appears in source text.
    requires_cue: bool = False
    cue_pattern: Pattern[str] | None = None


@dataclass(frozen=True)
class DomainMacro:
    """A controlled, versioned expansion recipe for a coarse scenario phrase."""

    macro_id: str
    version: str
    domain: str
    description: str
    trigger_patterns: tuple[Pattern[str], ...]
    members: tuple[MacroMember, ...]
    # Explicit scenario cues that mean "do not invent login sub-steps".
    block_expansion_patterns: tuple[Pattern[str], ...] = field(default_factory=tuple)
    # Negation patterns keyed by member_id → suppress that member.
    negation_patterns: dict[str, tuple[Pattern[str], ...]] = field(default_factory=dict)


def _re(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


AUTH_LOGIN_COARSE = DomainMacro(
    macro_id="auth.login.coarse",
    version="1.0.0",
    domain="auth",
    description=(
        "Coarse login remains a single other step when the scenario does not "
        "name credential fields, Next/Submit, or concrete values."
    ),
    trigger_patterns=(
        _re(r"\blog\s*in\b"),
        _re(r"\blogin\b"),
    ),
    members=(
        MacroMember(
            member_id="coarse_login",
            obligation=StepObligation.mandatory,
            action=ActionType.other,
            target="Driver app login",
        ),
    ),
    block_expansion_patterns=(
        _re(r"\b(?:enter|type)\b.+\bpassword\b"),
        _re(r"\b(?:enter|type)\b.+\b(?:username|email|credential)"),
        _re(r"\b(?:tap|click)\b.+\b(?:next|submit|login|sign\s*in)\b"),
        _re(r"\bpassword\b.+\b(?:tap|click)\b.+\bnext\b"),
    ),
)

DRIVER_ONBOARDING_MILESTONES = DomainMacro(
    macro_id="driver.onboarding.milestones",
    version="1.0.0",
    domain="driver_onboarding",
    description=(
        "Expand an explicitly listed onboarding process into one step per "
        "named milestone; Hub landing is conditional on landing language."
    ),
    trigger_patterns=(
        _re(r"onboarding\s+process\s+(?:like[-–—:\s]+)"),
    ),
    members=(
        MacroMember(
            member_id="security_number",
            obligation=StepObligation.mandatory,
            action=ActionType.other,
            target="Security Number",
            cue_pattern=_re(r"security\s+number|\bssn\b"),
            requires_cue=True,
        ),
        MacroMember(
            member_id="start_earn",
            obligation=StepObligation.mandatory,
            action=ActionType.other,
            target="Start Earn",
            cue_pattern=_re(r"start\s+earn"),
            requires_cue=True,
        ),
        MacroMember(
            member_id="vehicle",
            obligation=StepObligation.mandatory,
            action=ActionType.other,
            target="vehicle",
            cue_pattern=_re(r"\bvehicle\b"),
            requires_cue=True,
        ),
        MacroMember(
            member_id="hub_landing",
            obligation=StepObligation.conditional,
            action=ActionType.assert_visible,
            target="Hub screen",
            assertion="user should land on Hub screen.",
            requires_cue=True,
            cue_pattern=_re(r"\bland(?:s|ed)?\s+on\s+(?:the\s+)?hub\b"),
        ),
    ),
    negation_patterns={
        "vehicle": (
            _re(r"\b(?:skip|without|no|exclude|excluding)\b.{0,40}\bvehicle\b"),
            _re(r"\bvehicle\b.{0,40}\b(?:not\s+required|optional|skipped)\b"),
        ),
        "hub_landing": (
            _re(r"\b(?:skip|without|no)\b.{0,40}\bhub\b"),
            _re(r"\bdo\s+not\s+land\s+on\s+(?:the\s+)?hub\b"),
        ),
    },
)

DOMAIN_MACROS: tuple[DomainMacro, ...] = (
    AUTH_LOGIN_COARSE,
    DRIVER_ONBOARDING_MILESTONES,
)

_BY_ID: dict[str, DomainMacro] = {macro.macro_id: macro for macro in DOMAIN_MACROS}


def get_macro(macro_id: str) -> DomainMacro:
    """Return a registered macro or raise KeyError."""
    return _BY_ID[macro_id]


def list_macros() -> tuple[DomainMacro, ...]:
    """Return all registered macros in registration order."""
    return DOMAIN_MACROS
