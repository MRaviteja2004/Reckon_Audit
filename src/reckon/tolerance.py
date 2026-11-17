"""What the audit forgives, and why.

Alignment mechanics and alignment *policy* are different concerns. The mechanics
(sequence alignment, scoring, finding classification) live in aligner.py and rarely
change. The judgements here change whenever a new recording shows a way the pipeline
can be right in substance and wrong on a technicality, so they are collected in one
place rather than scattered through the alignment code.

Every rule in this module exists to stop a *false finding* — the audit blaming a
faithful recording for something it did demonstrate. Each is deliberately narrow, and
each carries the condition that keeps it from ever producing the opposite error, a
false pass. That asymmetry is the point: this file may only ever forgive things the
recording genuinely shows.
"""

import re

from rapidfuzz import fuzz

from reckon.schema import (
    ActionType,
    AuditVerdict,
    ExpectedStep,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
)

_DIGIT_RUN = re.compile(r"\d+")

# The assertion action types describe the same intent — "verify something on screen" —
# and Stage 1 (scenario parse) and Stage 2 (video extraction) routinely disagree on
# which one applies: a scenario's "assert the error is shown" parses as assert_text
# while the model labels the same banner assert_visible. Treating them as
# interchangeable lets the assertion be confirmed by evidence rather than reported as a
# missing action on a technicality.
ASSERTION_ACTIONS = frozenset({
    ActionType.assert_visible,
    ActionType.assert_text,
    ActionType.assert_state,
})

# Actions that cannot be performed without first focusing their target, and so stand as
# evidence that the focusing click happened even when the click itself left no trace.
FOCUS_IMPLYING_ACTIONS = frozenset({
    ActionType.type_text,
    ActionType.select,
})

# Minimum RapidFuzz ratio (0–100) for two strings to be considered equivalent.
FUZZY_THRESHOLD = 60.0

# Structural chrome stripped before comparing assertion targets.
_TARGET_STRUCTURAL_TERMS = frozenset({
    "input",
    "field",
    "button",
    "btn",
    "cta",
    "link",
    "control",
})

# Targets that do not name a distinct widget — text evidence alone may pass.
_GENERIC_ASSERTION_TARGETS = frozenset({
    "message",
    "banner",
    "toast",
    "notification",
    "confirmation",
    "app",
    "screen",
    "page",
    "ui",
    "view",
    "window",
})

# Legitimate component-role aliases (carousel ↔ slider, sheet ↔ drawer, …).
_ASSERTION_TARGET_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"carousel", "slider", "pager", "gallery"}),
    frozenset({"sheet", "drawer", "bottomsheet"}),
    frozenset({"modal", "dialog", "popup"}),
    frozenset({"menu", "overflow", "kebab"}),
)

# Role families that must not satisfy each other via fuzzy target match or
# shared label text. Detected from raw target tokens (including structural
# chrome such as button/cta) so "Mentorship title" cannot be proved by
# "Join Mentorship button", and a dashboard CTA/option cannot be proved by a
# carousel/page surface. Options live on sheets/menus, so they share that family.
_ASSERTION_ROLE_CONFLICT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"carousel", "slider", "pager", "gallery", "page"}),
    frozenset({
        "sheet",
        "drawer",
        "bottomsheet",
        "menu",
        "modal",
        "dialog",
        "popup",
        "option",
        "item",
        "entry",
    }),
    frozenset({"title", "heading", "header"}),
    frozenset({"button", "btn", "cta"}),
    frozenset({"checkbox", "radio", "toggle", "switch"}),
)

_ALIAS_CANONICAL: dict[str, str] = {
    token: sorted(group)[0]
    for group in _ASSERTION_TARGET_ALIAS_GROUPS
    for token in group
}


def values_equivalent(expected: str, observed: str) -> bool:
    """Return True when an observed input value matches the expected one.

    Fuzzy similarity is right for the words humans use for UI elements and wrong for the
    data under test. Measured on rapidfuzz, '89.00' scores 67 against '8.90', '4211'
    scores 75 against '4212', and '100' scores 86 against '1000' — all comfortably above
    any workable text threshold, so a refund off by a factor of ten, or the wrong order
    entirely, would be reported as a pass. That is precisely the false pass this audit
    exists to prevent.

    Digits are therefore compared exactly and only the surrounding text fuzzily, which
    still tolerates the differences that do not change meaning ('89.00' against
    '£89.00') while refusing the ones that do.
    """
    if _DIGIT_RUN.findall(expected) != _DIGIT_RUN.findall(observed):
        return False
    if _is_mid_entry(expected, observed):
        return True
    return fuzz.ratio(expected.lower(), observed.lower()) >= FUZZY_THRESHOLD


def _is_mid_entry(expected: str, observed: str) -> bool:
    """Return True when the observed value is the expected one caught part-typed.

    Frames are captured while a field is still being filled, so "tomsmith" is routinely
    observed as "to" — which on a real recording was reported as a wrong input value
    against a scenario the recording went on to satisfy exactly.

    Only this direction counts: a value the recording never finished typing is still the
    value being typed, whereas a recording that entered *more* than the scenario claimed
    has genuinely done something else. The digit check above runs first, so this can
    never wave through a figure that stopped short — "8" against "89.00" is rejected on
    its digits before reaching here.
    """
    return bool(observed) and expected.lower().startswith(observed.lower())


# Words that flip an outcome. Evidence may carry these only if the assertion did.
_NEGATION_TOKENS = frozenset({"not", "no", "never", "fail", "failed", "unable"})

# Assertion language that expects an empty / zero / absent UI state.
_EMPTY_STATE_ASSERTION_RE = re.compile(
    r"\b(?:"
    r"empty|"
    r"no\s+(?:learners?|mentees?|items?|results?|data)|"
    r"zero|"
    r"0\s+(?:active|learners?|mentees?)|"
    r"without\s+(?:any\s+)?(?:learners?|mentees?|items?)"
    r")\b",
    re.IGNORECASE,
)

# Evidence that supports an empty-state assertion (not a failure banner).
_EMPTY_STATE_EVIDENCE_RE = re.compile(
    r"\b(?:"
    r"no\s+(?:learners?|mentees?|items?|results?)\s+yet|"
    r"empty(?:\s+state)?|"
    r"0\s+(?:active\s+)?(?:learners?|mentees?)|"
    r"you(?:'ll| will)\s+be\s+alerted|"
    r"matched\s+learners?\s+0|"
    r"earned\s+\$?0"
    r")\b",
    re.IGNORECASE,
)

# Bounded terminology aliases for target/screen identity (not open fuzzy match).
_TARGET_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"mentee", "mentees", "learner", "learners"}),
    frozenset({"mentor", "mentorship", "mentoring"}),
    frozenset({"mentorship", "program", "programme"}),
    frozenset({"setting", "settings"}),
    frozenset({"dashboard", "screen", "page", "home"}),
    frozenset({"past", "previous", "prior"}),
    frozenset({"landing", "signup"}),
)


def _normalise_assertion_text(text: str) -> list[str]:
    """Reduce assertion or evidence text to lowercase alphanumeric tokens."""
    lowered = re.sub(r"n['\u2019]t\b", " not", text.lower())
    return re.findall(r"[a-z0-9]+", lowered)


def empty_state_evidence_supports_assertion(assertion: str, evidence: str) -> bool:
    """True when empty-state UI text supports an empty/zero assertion claim."""
    if not assertion.strip() or not evidence.strip():
        return False
    if not _EMPTY_STATE_ASSERTION_RE.search(assertion):
        return False
    return bool(_EMPTY_STATE_EVIDENCE_RE.search(evidence))


def _introduces_negation(assertion: str, evidence: str) -> bool:
    """Return True when the evidence negates something the assertion did not.

    Token containment alone cannot catch this: every word of "Login successful"
    is present in "Login not successful", so the reverse outcome would confirm
    the assertion. The added negation is the whole meaning.

    Empty-state evidence ("No learners yet") supporting an empty-dashboard
    assertion is not treated as negation.
    """
    if empty_state_evidence_supports_assertion(assertion, evidence):
        return False
    assertion_negations = set(_normalise_assertion_text(assertion)) & _NEGATION_TOKENS
    evidence_negations = set(_normalise_assertion_text(evidence)) & _NEGATION_TOKENS
    return bool(evidence_negations - assertion_negations)


def _synonym_set(token: str) -> frozenset[str]:
    matched = [group for group in _TARGET_SYNONYM_GROUPS if token in group]
    if not matched:
        return frozenset({token})
    merged: set[str] = {token}
    for group in matched:
        merged |= set(group)
    return frozenset(merged)


def target_tokens_compatible(expected_token: str, observed_token: str) -> bool:
    """Stem or explicit synonym compatibility for one target token pair."""
    if _tokens_share_stem(expected_token, observed_token):
        return True
    return bool(_synonym_set(expected_token) & _synonym_set(observed_token))


def _contains_assertion(assertion: str, evidence: str) -> bool:
    """Return True when the evidence contains every token of the assertion.

    Containment rather than similarity, and counted rather than set-based. Substring
    similarity cannot tell an outcome from its opposite — "Refund issued" scores 77
    against "Refund failed", because the shared word dominates and the differing word
    is the one carrying the meaning. Requiring each token to actually be present
    separates them, while still allowing evidence to add context the assertion never
    mentioned ("Error: Invalid email" satisfies "Invalid email").

    Digits tokenise like any other word, so a figure that differs — £89.00 against
    £8.90 — fails containment on the figure itself.
    """
    required: dict[str, int] = {}
    for token in _normalise_assertion_text(assertion):
        required[token] = required.get(token, 0) + 1
    if not required:
        return False

    available: dict[str, int] = {}
    for token in _normalise_assertion_text(evidence):
        available[token] = available.get(token, 0) + 1

    return all(available.get(token, 0) >= count for token, count in required.items())


# Click/select interchange is opt-in for choice controls only (assist remapping).
# Baseline align() must not treat them as interchangeable by default.
_CHOICE_CONTROL_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.click, ActionType.select}
)

_CHOICE_TARGET_RE = re.compile(
    r"\b(?:"
    r"check\s*box|checkbox|"
    r"radio(?:\s*button)?|"
    r"switch|"
    r"drop\s*down|dropdown|"
    r"option|"
    r"combo(?:\s*box)?"
    r")\b",
    re.IGNORECASE,
)


def is_choice_control_target(target: str | None) -> bool:
    """True when the target names a checkbox, radio, switch, dropdown, or option."""
    return bool(target and _CHOICE_TARGET_RE.search(target))


def actions_compatible(
    exp_action: ActionType,
    obs_action: ActionType,
    *,
    allow_choice_control: bool = False,
    expected_target: str | None = None,
    observed_target: str | None = None,
) -> bool:
    """Return True when two action types may be paired in alignment.

    Identical types are always compatible. The assertion family is treated as
    internally interchangeable; see ASSERTION_ACTIONS.

    Click ↔ select is NOT allowed by default (baseline DP must stay strict).
    Opt in with ``allow_choice_control=True`` only when *both* targets denote
    choice controls (checkbox / radio / switch / dropdown). A bare ``select``
    must never remap onto a standard button click.
    """
    if exp_action == obs_action:
        return True
    if exp_action in ASSERTION_ACTIONS and obs_action in ASSERTION_ACTIONS:
        return True
    if not allow_choice_control:
        return False
    if not (
        exp_action in _CHOICE_CONTROL_ACTIONS
        and obs_action in _CHOICE_CONTROL_ACTIONS
    ):
        return False
    return is_choice_control_target(expected_target) and is_choice_control_target(
        observed_target
    )


def _assertion_meaning_tokens(target: str) -> list[str]:
    """Lowercase tokens with structural chrome removed."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", target.lower())
        if token not in _TARGET_STRUCTURAL_TERMS
    ]


def _canonicalise_target_tokens(tokens: list[str]) -> list[str]:
    return [_ALIAS_CANONICAL.get(token, token) for token in tokens]


def _assertion_target_similarity(expected: str, observed: str) -> float:
    """Fuzzy similarity of assertion targets after structural + alias normalisation."""
    expected_meaning = _canonicalise_target_tokens(_assertion_meaning_tokens(expected))
    observed_meaning = _canonicalise_target_tokens(_assertion_meaning_tokens(observed))
    if expected_meaning and observed_meaning:
        if expected_meaning == observed_meaning:
            return 100.0
        return float(
            fuzz.ratio(" ".join(expected_meaning), " ".join(observed_meaning))
        )
    expected_tokens = re.findall(r"[a-z0-9]+", expected.lower())
    observed_tokens = re.findall(r"[a-z0-9]+", observed.lower())
    if expected_tokens == observed_tokens:
        return 100.0
    return 0.0


def _raw_target_tokens(target: str) -> set[str]:
    """Lowercase alphanumeric tokens including structural chrome (button, cta, …)."""
    return set(re.findall(r"[a-z0-9]+", target.lower()))


def _assertion_role_groups(target: str) -> list[frozenset[str]]:
    """Map target text onto role-conflict families."""
    tokens = _raw_target_tokens(target)
    return [group for group in _ASSERTION_ROLE_CONFLICT_GROUPS if tokens & group]


def _assertion_roles_conflict(expected_target: str, observed_text: str) -> bool:
    """True when expected and observed name incompatible UI component roles."""
    expected_groups = _assertion_role_groups(expected_target)
    observed_groups = _assertion_role_groups(observed_text)
    if not expected_groups or not observed_groups:
        return False
    return all(
        expected_group != observed_group
        for expected_group in expected_groups
        for observed_group in observed_groups
    )


def assertion_roles_conflict_with_observation(
    expected_target: str | None,
    obs: ObservedAction,
) -> bool:
    """True when the observation's named UI role cannot prove ``expected_target``.

    Role conflict is judged from ``obs.target`` only. Free-form screen prose
    often mentions "dialog", "page", or "menu" as ambient chrome and must not
    veto a compatible control on that screen (e.g. an X button on a dialog).
    """
    if expected_target is None or not obs.target:
        return False
    return _assertion_roles_conflict(expected_target, obs.target)


_MIN_SCREEN_STEM_LEN = 4


def _tokens_share_stem(expected_token: str, observed_token: str) -> bool:
    """True when tokens match or one extends the other (mentor ↔ mentorship)."""
    if expected_token == observed_token:
        return True
    shorter, longer = sorted(
        (expected_token, observed_token), key=len
    )
    return len(shorter) >= _MIN_SCREEN_STEM_LEN and longer.startswith(shorter)


def assertion_screen_compatible(
    expected_target: str | None,
    obs: ObservedAction,
) -> bool:
    """True when observation screen identity can support the expected target.

    Role conflicts always fail. When the expected target names a distinct screen
    or component, at least one distinctive token must relate to the observed
    target/screen (stem-aware), so a join-confirmation action cannot stand in
    for a Mentor Dashboard proof merely by lacking an explicit role conflict.
    """
    if expected_target is None:
        return True
    if assertion_roles_conflict_with_observation(expected_target, obs):
        return False

    expected_tokens = [
        token
        for token in _canonicalise_target_tokens(
            _assertion_meaning_tokens(expected_target)
        )
        if token not in _GENERIC_ASSERTION_TARGETS
    ]
    if not expected_tokens:
        return True

    observed_parts = [obs.target or "", obs.screen_description or ""]
    if obs.assertion_evidence is not None:
        observed_parts.append(obs.assertion_evidence.description or "")
    observed_tokens = _canonicalise_target_tokens(
        [
            token
            for token in re.findall(r"[a-z0-9]+", " ".join(observed_parts).lower())
            if token not in _TARGET_STRUCTURAL_TERMS
        ]
    )
    if not observed_tokens:
        return False
    return any(
        target_tokens_compatible(expected_token, observed_token)
        for expected_token in expected_tokens
        for observed_token in observed_tokens
    )


def assertion_target_relevant(
    expected_target: str | None,
    obs: ObservedAction,
) -> bool:
    """Require a distinct expected assertion target to match the observed UI context.

    Screen-level ``visible_text`` alone must never prove a named component (e.g.
    a carousel) merely because a menu or sheet happens to display the same label.
    Substring / fuzzy overlap across incompatible roles (title↔button,
    option/CTA↔carousel page) is also rejected. Generic message-region targets
    (banner/toast/…) remain text-driven.
    """
    if expected_target is None:
        return True

    expected_tokens = _assertion_meaning_tokens(expected_target)
    if not expected_tokens:
        return True
    # Generic / non-distinct targets (banner, app, …) stay text-driven.
    if all(token in _GENERIC_ASSERTION_TARGETS for token in expected_tokens):
        return True

    if assertion_roles_conflict_with_observation(expected_target, obs):
        return False

    if obs.target is not None:
        if _assertion_target_similarity(expected_target, obs.target) >= FUZZY_THRESHOLD:
            return True

    # Target context deliberately excludes visible_text for most components —
    # labels on an unrelated sheet must not stand in for a carousel/title/CTA.
    # Menu options are the exception: the sheet is the surface, and the option
    # label lives in visible_text.
    context_parts = [obs.target or "", obs.screen_description or ""]
    if obs.assertion_evidence is not None:
        context_parts.append(obs.assertion_evidence.description or "")
    required = {
        token
        for token in _canonicalise_target_tokens(expected_tokens)
        if token not in _GENERIC_ASSERTION_TARGETS
    }
    expected_raw = _raw_target_tokens(expected_target)
    option_role_tokens = frozenset({"option", "item", "entry"})
    sheet_surface_tokens = frozenset(
        {"sheet", "drawer", "bottomsheet", "menu", "modal", "dialog", "popup"}
    )
    observed_raw = _raw_target_tokens(obs.target or "")
    if (
        expected_raw & option_role_tokens
        and observed_raw & sheet_surface_tokens
        and obs.assertion_evidence is not None
        and obs.assertion_evidence.visible_text
    ):
        context_parts.append(obs.assertion_evidence.visible_text)
        required -= option_role_tokens
    if not required:
        return True
    context_tokens = set(
        _canonicalise_target_tokens(
            re.findall(r"[a-z0-9]+", " ".join(context_parts).lower())
        )
    )
    # Each required token must appear in context via exact/stem/synonym match.
    return all(
        any(
            target_tokens_compatible(required_token, context_token)
            for context_token in context_tokens
        )
        for required_token in required
    )


# Surface nouns that mark an expected target as a named screen identity.
_SCREEN_SURFACE_TOKENS = frozenset({"screen", "page", "view", "dashboard"})

# Landing / arrival cues required before screen-identity may replace full
# assertion token containment (avoids "X passed successfully" false passes).
_LANDING_ASSERTION_CUE_RE = re.compile(
    r"\b(?:"
    r"land|lands|landed|"
    r"reach|reaches|reached|"
    r"arrive|arrives|arrived|"
    r"open|opens|opened|"
    r"shown|display|displayed|displays|"
    r"appear|appears|appeared|"
    r"visible|"
    r"on\s+the"
    r")\b",
    re.IGNORECASE,
)


def _screen_identity_tokens(expected_target: str) -> list[str]:
    """Distinctive tokens naming the screen (excludes surface nouns like screen/page)."""
    return [
        token
        for token in _canonicalise_target_tokens(
            _assertion_meaning_tokens(expected_target)
        )
        if token not in _GENERIC_ASSERTION_TARGETS
        and token not in _SCREEN_SURFACE_TOKENS
    ]


def _is_screen_landing_assertion(
    assertion: str,
    expected_target: str | None,
) -> bool:
    """True when the assertion is a land-on / reach-screen claim for a named surface."""
    if expected_target is None:
        return False
    if not (_raw_target_tokens(expected_target) & _SCREEN_SURFACE_TOKENS):
        return False
    if not _LANDING_ASSERTION_CUE_RE.search(assertion):
        return False
    identity = _screen_identity_tokens(expected_target)
    if not identity:
        return False
    assertion_tokens = set(_normalise_assertion_text(assertion))
    return all(token in assertion_tokens for token in identity)


def _screen_identity_supported_by_observation(
    identity: list[str],
    obs: ObservedAction,
) -> bool:
    """True when structured observation text names the expected screen identity.

    Multi-token identities (e.g. Background check) may appear across target,
    visible_text, description, or screen_description. A single-token identity
    (e.g. Hub) must appear in the observed target itself so ambient copy cannot
    prove the wrong screen.
    """
    if not identity:
        return False
    if len(identity) == 1:
        if not obs.target:
            return False
        return identity[0] in set(
            _canonicalise_target_tokens(_assertion_meaning_tokens(obs.target))
        )

    parts = [obs.target or "", obs.screen_description or ""]
    if obs.assertion_evidence is not None:
        parts.append(obs.assertion_evidence.visible_text or "")
        parts.append(obs.assertion_evidence.description or "")
    blob_tokens = set(
        _canonicalise_target_tokens(
            re.findall(r"[a-z0-9]+", " ".join(parts).lower())
        )
    )
    return all(token in blob_tokens for token in identity)


def assertion_satisfied(
    assertion: str,
    obs: ObservedAction,
    *,
    expected_target: str | None = None,
) -> bool:
    """Return True when the observed action provides evidence for the assertion.

    Evidence is checked only against the single matched observed action, not over a
    timestamp window. This is deliberate: the recording's whole point is to catch a
    false pass, so an assertion is confirmed only where it lines up with the step that
    should produce it — not anywhere the text happens to appear. Widening the search
    would reintroduce the false-pass risk the audit exists to catch.

    When ``expected_target`` names a distinct UI component, target relevance is
    required before text containment can establish a pass.

    Narrow exception: a land-on-screen assertion whose expected target names that
    screen may also pass when structured screen-identity evidence agrees, without
    requiring boilerplate words like "user should land on" to appear in the OCR.
    Navigate / screen-change observations may supply that identity via target or
    screen_description even when Stage 2 omitted ``assertion_evidence``.
    """
    if not assertion_target_relevant(expected_target, obs):
        return False

    evidence = obs.assertion_evidence
    if evidence is not None:
        for text in (evidence.visible_text, evidence.description):
            if not text:
                continue
            if _introduces_negation(assertion, text):
                continue
            if _contains_assertion(assertion, text):
                return True

    # Screen-landing path: identity tokens only, after target relevance + polarity.
    # Without assertion_evidence, only passive/navigate observations may prove a
    # land-on-screen claim. A click/type/select is the action that *opens* a page,
    # not evidence the page is displayed — that belongs to the action step.
    if not _is_screen_landing_assertion(assertion, expected_target):
        return False
    if evidence is None and obs.action not in (
        ASSERTION_ACTIONS | {ActionType.navigate, ActionType.other}
    ):
        return False
    identity = _screen_identity_tokens(expected_target or "")
    polarity_texts = [
        text
        for text in (
            evidence.visible_text if evidence is not None else None,
            evidence.description if evidence is not None else None,
            obs.screen_description,
            obs.target,
        )
        if text
    ]
    if any(_introduces_negation(assertion, text) for text in polarity_texts):
        return False
    return _screen_identity_supported_by_observation(identity, obs)


def focus_click_evidence(
    exp: ExpectedStep,
    next_step: ExpectedStep | None,
    matched_by_step: dict[int, ObservedAction],
) -> ObservedAction | None:
    """Return the observed action that proves an unmatched focusing click occurred.

    Clicking into a field to focus it changes almost nothing on screen — a thin ring and
    a caret, measured on the synthetic corpus at roughly the same magnitude as the mouse
    pointer's own movement. Such a click is therefore routinely absent from the observed
    trace even on a perfectly faithful recording, and reporting it as a missing action
    penalises the recording for the limits of frame differencing rather than for
    anything it failed to demonstrate.

    Text appearing in that same field is proof the focus happened, so the following step
    is accepted as evidence. Crucially the *typing* must itself have been observed: if
    the field was never filled, nothing is implied and the click still fails. That keeps
    the leniency from ever manufacturing a pass for a step the recording does not show.
    """
    if exp.action != ActionType.click or exp.target is None:
        return None
    if next_step is None or next_step.action not in FOCUS_IMPLYING_ACTIONS:
        return None
    if next_step.target is None:
        return None
    if fuzz.ratio(exp.target.lower(), next_step.target.lower()) < FUZZY_THRESHOLD:
        return None
    return matched_by_step.get(next_step.index)


def finding_severity(finding_type: FindingType, expected_step: ExpectedStep | None) -> FindingSeverity:
    """Classify whether a finding bears on fidelity or only on wording and grain.

    The distinction is drawn from measurement, not taste. Across repeated runs of the
    synthetic corpus, assertion_not_observed and wrong_input_value appeared only on
    recordings that genuinely failed their scenario. wrong_target and extra_action
    appeared on every long run including the faithful ones, because Stage 2 names
    elements in its own words and reports actions a scenario leaves implicit.

    missing_action is split. On a step carrying an assertion it means the claimed
    outcome was never matched at all — the false pass this audit exists to catch. On a
    plain action step it fires constantly from grain mismatch, so it is advisory. That
    trade is deliberate and it is not free: a recording that genuinely skips a plain
    action is reported without being condemned. Blocking on it instead would condemn
    faithful recordings far more often, and the assertions — the substance of any
    scenario — stay critical either way.
    """
    if finding_type in (FindingType.assertion_not_observed, FindingType.wrong_input_value):
        return FindingSeverity.critical
    if finding_type == FindingType.missing_action:
        if expected_step is not None and expected_step.assertion is not None:
            return FindingSeverity.critical
    return FindingSeverity.advisory


def verdict_for(findings: list[Finding]) -> AuditVerdict:
    """Return the headline verdict: unfaithful when any critical finding survives."""
    if any(f.severity == FindingSeverity.critical for f in findings):
        return AuditVerdict.unfaithful
    return AuditVerdict.faithful


def verdict_for_steps(
    steps: list[ExpectedStep],
    findings: list[Finding],
    verdicts: list,
) -> AuditVerdict:
    """Unfaithful when a critical finding is still on the audit or on a failed step."""
    if verdict_for(findings) == AuditVerdict.unfaithful:
        return AuditVerdict.unfaithful
    by_index = {step.index: step for step in steps}
    for verdict in verdicts:
        if verdict.passed:
            continue
        step = by_index.get(verdict.expected_step_index)
        for finding_type in verdict.finding_types:
            if finding_severity(finding_type, step) == FindingSeverity.critical:
                return AuditVerdict.unfaithful
    return AuditVerdict.faithful
