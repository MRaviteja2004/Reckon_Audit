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


def _normalise_assertion_text(text: str) -> list[str]:
    """Reduce assertion or evidence text to lowercase alphanumeric tokens."""
    lowered = re.sub(r"n['\u2019]t\b", " not", text.lower())
    return re.findall(r"[a-z0-9]+", lowered)


def _introduces_negation(assertion: str, evidence: str) -> bool:
    """Return True when the evidence negates something the assertion did not.

    Token containment alone cannot catch this: every word of "Login successful"
    is present in "Login not successful", so the reverse outcome would confirm
    the assertion. The added negation is the whole meaning.
    """
    assertion_negations = set(_normalise_assertion_text(assertion)) & _NEGATION_TOKENS
    evidence_negations = set(_normalise_assertion_text(evidence)) & _NEGATION_TOKENS
    return bool(evidence_negations - assertion_negations)


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


def actions_compatible(exp_action: ActionType, obs_action: ActionType) -> bool:
    """Return True when two action types may be paired in alignment.

    Identical types are always compatible. The assertion family is treated as
    internally interchangeable; see ASSERTION_ACTIONS.
    """
    if exp_action == obs_action:
        return True
    return exp_action in ASSERTION_ACTIONS and obs_action in ASSERTION_ACTIONS


def assertion_satisfied(assertion: str, obs: ObservedAction) -> bool:
    """Return True when the observed action provides evidence for the assertion.

    Evidence is checked only against the single matched observed action, not over a
    timestamp window. This is deliberate: the recording's whole point is to catch a
    false pass, so an assertion is confirmed only where it lines up with the step that
    should produce it — not anywhere the text happens to appear. Widening the search
    would reintroduce the false-pass risk the audit exists to catch.
    """
    if obs.assertion_evidence is None:
        return False
    evidence = obs.assertion_evidence
    for text in (evidence.visible_text, evidence.description):
        if not text:
            continue
        if _introduces_negation(assertion, text):
            continue
        if _contains_assertion(assertion, text):
            return True
    return False


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
