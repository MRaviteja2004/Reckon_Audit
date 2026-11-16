import re

from rapidfuzz import fuzz

from reckon.schema import (
    ActionType,
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)
from reckon.tolerance import (
    ASSERTION_ACTIONS,
    FUZZY_THRESHOLD,
    _introduces_negation,
    actions_compatible,
    assertion_satisfied,
    finding_severity,
    focus_click_evidence,
    values_equivalent,
    verdict_for,
)

# Minimum _action_match_score for two steps to be paired in the DP alignment.
# Below this, pairing is forbidden and a gap is preferred.
_MATCH_FLOOR = 0.30

# Cost of leaving an expected step (missing) or observed action (extra) unmatched.
_GAP_COST = -0.30

_CHOICE_MATCH = 0
_CHOICE_GAP_OBS = 1   # skip observed action → extra_action finding
_CHOICE_GAP_EXP = 2   # skip expected step  → missing_action finding
_TARGET_STRUCTURAL_TERMS = frozenset({"input", "field"})
_OPEN_ORDER_RE = re.compile(r"\bopen\s+(?:the\s+)?order\b", re.IGNORECASE)
_ORDER_TARGET_RE = re.compile(r"^\s*order\s*#\s*(\d+)\s*$", re.IGNORECASE)
_ORDER_ROUTE_RE = re.compile(r"(?:^|/)orders/(\d+)/?$", re.IGNORECASE)
_ORDER_DETAILS_RE = re.compile(r"\border\s*#\s*(\d+)\b", re.IGNORECASE)
_DISMISS_ACTION_RE = re.compile(
    r"^\s*(?:(?:\d+[.)]|[-*•])\s+)?(?:dismiss|close)\b",
    re.IGNORECASE,
)
_CLOSE_CONTROL_RE = re.compile(r"\b(?:close|dismiss)\b", re.IGNORECASE)
_VISIBLE_STATE_TERMS = frozenset({
    "visible", "displayed", "shown", "open", "opened", "loaded", "present",
})
_ABSENT_STATE_TERMS = frozenset({
    "absent", "hidden", "missing", "closed", "dismissed",
})
_GENERIC_ASSERTION_TARGETS = frozenset({
    "message", "banner", "toast", "notification", "confirmation",
})
_DIRECT_ASSERTION_BONUS = 0.2

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _target_similarity(expected: str, observed: str) -> float:
    """Return target similarity after removing only generic UI structure terms."""
    expected_tokens = re.findall(r"[a-z0-9]+", expected.lower())
    observed_tokens = re.findall(r"[a-z0-9]+", observed.lower())
    expected_meaning = [
        token for token in expected_tokens if token not in _TARGET_STRUCTURAL_TERMS
    ]
    observed_meaning = [
        token for token in observed_tokens if token not in _TARGET_STRUCTURAL_TERMS
    ]

    if expected_meaning and observed_meaning:
        if expected_meaning == observed_meaning:
            return 100.0
        return fuzz.ratio(" ".join(expected_meaning), " ".join(observed_meaning))
    elif expected_tokens == observed_tokens:
        return 100.0

    return 0.0


def _assertion_target_relevant(
    exp: ExpectedStep,
    obs: ObservedAction,
) -> bool:
    """Require concrete assertion targets to match the observed UI context."""
    if exp.target is None:
        return True

    expected_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", exp.target.lower())
        if token not in _TARGET_STRUCTURAL_TERMS
    ]
    if (
        len(expected_tokens) == 1
        and expected_tokens[0] in _GENERIC_ASSERTION_TARGETS
    ):
        return True
    if (
        obs.target is not None
        and _target_similarity(exp.target, obs.target) >= FUZZY_THRESHOLD
    ):
        return True

    context_parts = [obs.target or "", obs.screen_description or ""]
    if obs.assertion_evidence is not None:
        context_parts.append(obs.assertion_evidence.description)
    context_tokens = re.findall(
        r"[a-z0-9]+",
        " ".join(context_parts).lower(),
    )
    return bool(expected_tokens) and all(
        token in context_tokens for token in expected_tokens
    )


def _screen_description_supports_assertion(
    exp: ExpectedStep,
    obs: ObservedAction,
) -> bool:
    if (
        exp.action not in ASSERTION_ACTIONS
        or exp.target is None
        or exp.assertion is None
        or obs.screen_description is None
    ):
        return False

    support_texts = [obs.screen_description]
    if obs.assertion_evidence is not None:
        support_texts.extend(
            text
            for text in (
                obs.assertion_evidence.description,
                obs.assertion_evidence.visible_text,
            )
            if text
        )
    support_tokens = {
        token
        for text in support_texts
        for token in re.findall(r"[a-z0-9]+", text.lower())
    }
    if (
        any(_introduces_negation(exp.assertion, text) for text in support_texts)
        or support_tokens & _ABSENT_STATE_TERMS
    ):
        return False

    if obs.assertion_evidence is not None:
        return (
            exp.action == ActionType.assert_visible
            and obs.target is not None
            and _target_similarity(exp.target, obs.target) >= FUZZY_THRESHOLD
            and bool(support_tokens & _VISIBLE_STATE_TERMS)
        )

    target_tokens = re.findall(r"[a-z0-9]+", exp.target.lower())
    screen_tokens = re.findall(r"[a-z0-9]+", obs.screen_description.lower())
    width = len(target_tokens)
    exact_screen_match = bool(target_tokens) and any(
        screen_tokens[start:start + width] == target_tokens
        for start in range(len(screen_tokens) - width + 1)
    )
    if exact_screen_match:
        return True

    meaningful_tokens = [
        token for token in target_tokens
        if token not in _TARGET_STRUCTURAL_TERMS
    ]
    observed_target_tokens = re.findall(
        r"[a-z0-9]+", (obs.target or "").lower()
    )
    context_tokens = observed_target_tokens + screen_tokens
    return bool(meaningful_tokens) and all(
        token in context_tokens for token in meaningful_tokens
    )


def _order_navigation_supports_open(
    exp: ExpectedStep,
    obs: ObservedAction,
) -> bool:
    """Recognize exact order-detail navigation caused by opening that order."""
    if (
        exp.action != ActionType.click
        or obs.action != ActionType.navigate
        or exp.target is None
        or exp.raw_text is None
        or obs.target is None
        or obs.screen_description is None
        or _OPEN_ORDER_RE.search(exp.raw_text) is None
    ):
        return False

    expected_match = _ORDER_TARGET_RE.fullmatch(exp.target)
    route_match = _ORDER_ROUTE_RE.search(obs.target)
    if expected_match is None or route_match is None:
        return False

    expected_id = expected_match.group(1)
    detail_ids = set(_ORDER_DETAILS_RE.findall(obs.screen_description))
    return route_match.group(1) == expected_id and detail_ids == {expected_id}


def _dismissal_click_supports_target(
    exp: ExpectedStep,
    obs: ObservedAction,
) -> bool:
    """Recognize a close control acting on the exact object being dismissed."""
    if (
        exp.action != ActionType.click
        or obs.action != ActionType.click
        or exp.target is None
        or exp.raw_text is None
        or obs.target is None
        or _DISMISS_ACTION_RE.search(exp.raw_text) is None
        or _CLOSE_CONTROL_RE.search(obs.target) is None
    ):
        return False

    expected_tokens = re.findall(r"[a-z0-9]+", exp.target.lower())
    context_tokens = re.findall(
        r"[a-z0-9]+",
        f"{obs.target} {obs.screen_description or ''}".lower(),
    )
    meaningful_tokens = [
        token for token in expected_tokens
        if token not in _TARGET_STRUCTURAL_TERMS
    ]
    return bool(meaningful_tokens) and all(
        token in context_tokens for token in meaningful_tokens
    )


def _action_match_score(exp: ExpectedStep, obs: ObservedAction) -> float:
    """Score 0.0–1.0 for how well exp matches obs; 0.0 when action types are incompatible."""
    order_navigation_support = _order_navigation_supports_open(exp, obs)
    dismissal_target_support = _dismissal_click_supports_target(exp, obs)
    compatible_actions = (
        actions_compatible(exp.action, obs.action)
        or order_navigation_support
    )
    screen_supports_assertion = _screen_description_supports_assertion(exp, obs)
    if not compatible_actions and not screen_supports_assertion:
        return 0.0

    score = 0.4 if compatible_actions else 0.3
    if screen_supports_assertion and obs.action == ActionType.navigate:
        score += 0.1

    # Target similarity contributes up to 0.4.
    direct_assertion_support = (
        exp.assertion is not None
        and assertion_satisfied(exp.assertion, obs)
        and _assertion_target_relevant(exp, obs)
    )
    if (
        screen_supports_assertion
        or order_navigation_support
        or dismissal_target_support
    ):
        score += 0.4
    elif exp.target is not None and obs.target is not None:
        score += 0.4 * _target_similarity(exp.target, obs.target) / 100.0
    elif exp.target is None and obs.target is None:
        score += 0.4

    if direct_assertion_support:
        score += _DIRECT_ASSERTION_BONUS

    # Input value similarity contributes up to 0.2.
    if exp.input_value is not None and obs.input_value is not None:
        score += 0.2 * fuzz.ratio(exp.input_value.lower(), obs.input_value.lower()) / 100.0
    elif exp.input_value is None and obs.input_value is None:
        score += 0.2

    return min(score, 1.0)

# ---------------------------------------------------------------------------
# DP alignment
# ---------------------------------------------------------------------------

def _dp_align(
    expected: list[ExpectedStep],
    observed: list[ObservedAction],
) -> list[tuple[ExpectedStep | None, ObservedAction | None]]:
    """Align two ordered sequences using Needleman-Wunsch-style DP.

    Returns a list of pairs where one element may be None:
    - (ExpectedStep, ObservedAction)  → matched pair
    - (ExpectedStep, None)            → missing action
    - (None, ObservedAction)          → extra action
    """
    n = len(expected)
    m = len(observed)

    NEG_INF = float("-inf")
    dp = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    choice = [[_CHOICE_MATCH] * (m + 1) for _ in range(n + 1)]

    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = i * _GAP_COST
        choice[i][0] = _CHOICE_GAP_EXP
    for j in range(1, m + 1):
        dp[0][j] = j * _GAP_COST
        choice[0][j] = _CHOICE_GAP_OBS

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = _action_match_score(expected[i - 1], observed[j - 1])

            options: list[tuple[float, int]] = []
            if score >= _MATCH_FLOOR:
                options.append((dp[i - 1][j - 1] + score, _CHOICE_MATCH))
            options.append((dp[i][j - 1] + _GAP_COST, _CHOICE_GAP_OBS))
            options.append((dp[i - 1][j] + _GAP_COST, _CHOICE_GAP_EXP))

            best_score, best_choice = max(options, key=lambda x: x[0])
            dp[i][j] = best_score
            choice[i][j] = best_choice

    # Traceback
    result: list[tuple[ExpectedStep | None, ObservedAction | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            result.append((None, observed[j - 1]))
            j -= 1
        elif j == 0:
            result.append((expected[i - 1], None))
            i -= 1
        else:
            c = choice[i][j]
            if c == _CHOICE_MATCH:
                result.append((expected[i - 1], observed[j - 1]))
                i -= 1
                j -= 1
            elif c == _CHOICE_GAP_OBS:
                result.append((None, observed[j - 1]))
                j -= 1
            else:
                result.append((expected[i - 1], None))
                i -= 1

    result.reverse()
    return result


def _is_complete_repeated_flow(
    expected: list[ExpectedStep],
    observed: list[ObservedAction],
) -> bool:
    """Return True when observed is two or more clean, exact copies of expected.

    This intentionally handles only clean, exact repeats: the observed action
    count must be an exact multiple of the expected step count, and every block
    must match the expected flow in order without classification findings.

    Partial or messy repeats fall back to normal alignment so unexpected actions
    are not hidden. A one-step scenario repeated multiple times also qualifies
    as a complete repeated flow and its repeated actions are suppressed.
    """
    if not expected:
        return False

    expected_count = len(expected)
    observed_count = len(observed)

    # A repeated flow requires at least two complete executions.
    if observed_count < expected_count * 2:
        return False

    # Partial trailing repetitions are not considered valid repeats.
    if observed_count % expected_count != 0:
        return False

    for block_start in range(0, observed_count, expected_count):
        block = observed[
            block_start:block_start + expected_count
        ]

        for exp, obs in zip(expected, block):
            if not actions_compatible(exp.action, obs.action):
                return False

            if _classify_pair(exp, obs):
                return False

    return True


# ---------------------------------------------------------------------------
# Finding classification
# ---------------------------------------------------------------------------

def _classify_pair(exp: ExpectedStep, obs: ObservedAction) -> list[FindingType]:
    """Return finding types for a matched (exp, obs) pair; empty list means passed."""
    findings: list[FindingType] = []

    assertion_confirmed = exp.assertion is not None and (
        (
            assertion_satisfied(exp.assertion, obs)
            and _assertion_target_relevant(exp, obs)
        )
        or _screen_description_supports_assertion(exp, obs)
    )
    target_confirmed = (
        assertion_confirmed
        or _order_navigation_supports_open(exp, obs)
        or _dismissal_click_supports_target(exp, obs)
    )

    # Wrong target: expected a target but observed differs or is absent.
    #
    # Skipped once an assertion has been confirmed by evidence. The "target" of an
    # assertion step is whatever noun the scenario happened to use — "the message", "a
    # banner" — not a real element name, so comparing it to the element the model named
    # fails on wording rather than on fidelity. The asserted text is the substantive
    # check, and it has already passed. An assertion that was *not* confirmed still
    # reports the target mismatch, where it remains diagnostic.
    if exp.target is not None and not target_confirmed:
        if obs.target is None:
            findings.append(FindingType.wrong_target)
        elif _target_similarity(exp.target, obs.target) < FUZZY_THRESHOLD:
            findings.append(FindingType.wrong_target)

    # Wrong input value: compared only when both values are observable.
    #
    # An observed None means the value could not be read — a masked password field, or
    # a frame that caught the field before it was filled — not that the field was wrong.
    # Reporting a mismatch against something unverifiable is a false finding.
    #
    # When both are readable the comparison is digit-exact: fuzzy similarity scored
    # '89.00' against '8.90' at 67, so a refund off by a factor of ten passed as a match.
    if exp.input_value is not None and obs.input_value is not None:
        if not values_equivalent(exp.input_value, obs.input_value):
            findings.append(FindingType.wrong_input_value)

    # Assertion not observed: every step with an assertion must be verified.
    if exp.assertion is not None and not assertion_confirmed:
        findings.append(FindingType.assertion_not_observed)

    return findings


def _describe_finding(ft: FindingType, exp: ExpectedStep, obs: ObservedAction) -> str:
    if ft == FindingType.wrong_target:
        return (
            f"Step {exp.index}: expected {exp.action.value} on {exp.target!r}, "
            f"but observed on {obs.target!r} at {obs.timestamp_ms}ms."
        )
    if ft == FindingType.wrong_input_value:
        return (
            f"Step {exp.index}: expected input value {exp.input_value!r}, "
            f"but observed {obs.input_value!r} at {obs.timestamp_ms}ms."
        )
    if ft == FindingType.assertion_not_observed:
        evidence_detail = (
            f"'{obs.assertion_evidence.visible_text}'"
            if obs.assertion_evidence and obs.assertion_evidence.visible_text
            else "no visible text"
        )
        return (
            f"Step {exp.index}: assertion '{exp.assertion}' was not confirmed — "
            f"observed evidence at {obs.timestamp_ms}ms shows {evidence_detail}."
        )
    return f"Step {exp.index}: unexpected finding type {ft.value}."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def align(expected: ExpectedTrace, observed: ObservedTrace) -> AuditResult:
    """Align expected and observed traces and return a scored audit result.

    No model calls are made — alignment and scoring are purely deterministic.

    fidelity_score is the fraction of expected steps that passed. Extra actions
    (observed but not in the scenario) are reported as findings but do not lower
    the score, since the score measures fidelity to the expected scenario, not the
    absence of additional activity.

    `verdict` is the headline: unfaithful when any critical finding survives. Read that,
    or `findings`, to decide whether a recording is faithful — never `fidelity_score`.
    On a 90-second synthetic flow run five times, the faithful recording scored between
    0.69 and 0.85 while the three deliberately broken variants scored between 0.61 and
    0.77 — overlapping ranges, because extraction is not deterministic and a longer
    recording gives it more places to describe an action at a different grain than the
    scenario was written at. Across those same runs the *finding types* separated
    faithful from broken every time. The score is a summary for a human; the findings
    are the verdict.
    """
    pairs = _dp_align(expected.steps, observed.actions)

    repeated_flow = _is_complete_repeated_flow(
        expected.steps,
        observed.actions,
    )

    all_findings: list[Finding] = []
    step_verdicts: list[StepVerdict] = []

    matched_by_step = {
        exp.index: obs for exp, obs in pairs if exp is not None and obs is not None
    }
    next_step_of = {
        step.index: expected.steps[position + 1] if position + 1 < len(expected.steps) else None
        for position, step in enumerate(expected.steps)
    }

    for exp, obs in pairs:
        if exp is None:
            assert obs is not None

            # When the recording contains complete repeated executions of the
            # expected flow, unmatched copies are intentional repetitions rather
            # than unexpected actions.
            if repeated_flow:
                continue

            all_findings.append(Finding(
                type=FindingType.extra_action,
                severity=finding_severity(FindingType.extra_action, None),
                expected_step_index=None,
                observed_action_index=obs.index,
                evidence_timestamp_ms=obs.timestamp_ms,
                description=(
                    f"Observed action {obs.index} "
                    f"({obs.action.value} on {obs.target!r}) "
                    "was not in the expected scenario."
                ),
            ))
        elif obs is None:
            implied_by = focus_click_evidence(exp, next_step_of[exp.index], matched_by_step)
            if implied_by is not None:
                step_verdicts.append(StepVerdict(
                    expected_step_index=exp.index,
                    passed=True,
                    matched_observed_index=implied_by.index,
                    evidence_timestamp_ms=implied_by.timestamp_ms,
                    finding_types=[],
                ))
                continue
            all_findings.append(Finding(
                type=FindingType.missing_action,
                severity=finding_severity(FindingType.missing_action, exp),
                expected_step_index=exp.index,
                observed_action_index=None,
                evidence_timestamp_ms=None,
                description=(
                    f"Expected step {exp.index} ({exp.action.value} on {exp.target!r}) "
                    "was not observed in the recording."
                ),
            ))
            step_verdicts.append(StepVerdict(
                expected_step_index=exp.index,
                passed=False,
                matched_observed_index=None,
                evidence_timestamp_ms=None,
                finding_types=[FindingType.missing_action],
            ))
        else:
            pair_findings = _classify_pair(exp, obs)
            for ft in pair_findings:
                all_findings.append(Finding(
                    type=ft,
                    severity=finding_severity(ft, exp),
                    expected_step_index=exp.index,
                    observed_action_index=obs.index,
                    evidence_timestamp_ms=obs.timestamp_ms,
                    description=_describe_finding(ft, exp, obs),
                ))
            step_verdicts.append(StepVerdict(
                expected_step_index=exp.index,
                passed=len(pair_findings) == 0,
                matched_observed_index=obs.index,
                evidence_timestamp_ms=obs.timestamp_ms,
                finding_types=pair_findings,
            ))

    total = len(expected.steps)
    passed = sum(1 for v in step_verdicts if v.passed)
    fidelity_score = passed / total if total > 0 else 1.0

    return AuditResult(
        verdict=verdict_for(all_findings),
        fidelity_score=fidelity_score,
        findings=all_findings,
        step_verdicts=step_verdicts,
    )
