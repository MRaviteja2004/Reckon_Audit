import json
from pathlib import Path

import pytest

from reckon.aligner import (
    _MATCH_FLOOR,
    _action_match_score,
    _classify_pair,
    _dp_align,
    _screen_description_supports_assertion,
    _target_similarity,
    align,
)
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditResult,
    AuditVerdict,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)
from reckon.tolerance import FUZZY_THRESHOLD

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp(index: int, action: ActionType, target: str | None = None,
         input_value: str | None = None, assertion: str | None = None) -> ExpectedStep:
    return ExpectedStep(index=index, action=action, target=target,
                        input_value=input_value, assertion=assertion)


def _obs(index: int, action: ActionType, target: str | None = None,
         input_value: str | None = None, timestamp_ms: int = 1000,
         evidence: AssertionEvidence | None = None, confidence: float = 0.9) -> ObservedAction:
    return ObservedAction(index=index, action=action, target=target,
                          input_value=input_value, timestamp_ms=timestamp_ms,
                          assertion_evidence=evidence, confidence=confidence)


def _make_expected(steps: list[ExpectedStep]) -> ExpectedTrace:
    return ExpectedTrace(scenario_title="test", steps=steps)


def _make_observed(actions: list[ObservedAction]) -> ObservedTrace:
    return ObservedTrace(source_file="test.mp4", actions=actions)


def _load_expected() -> ExpectedTrace:
    data = json.loads((FIXTURES / "login_expected.json").read_text())
    return ExpectedTrace.model_validate(data)


def _load_observed(filename: str) -> ObservedTrace:
    data = json.loads((FIXTURES / filename).read_text())
    return ObservedTrace.model_validate(data)


# ---------------------------------------------------------------------------
# _action_match_score
# ---------------------------------------------------------------------------

def test_match_score_zero_when_action_types_differ():
    exp = _exp(1, ActionType.click, "Submit")
    obs = _obs(1, ActionType.type_text, "Submit")
    assert _action_match_score(exp, obs) == 0.0


def test_match_score_assert_text_compatible_with_assert_visible():
    # Stage 1 parses "assert error is shown" as assert_text; Stage 2 (the model)
    # commonly labels the same event assert_visible. They must be matchable so the
    # assertion can be verified by evidence rather than reported as missing.
    exp = _exp(1, ActionType.assert_text, "Error banner")
    obs = _obs(1, ActionType.assert_visible, "Error banner")
    assert _action_match_score(exp, obs) >= _MATCH_FLOOR


def test_match_score_assert_text_compatible_with_assert_state():
    exp = _exp(1, ActionType.assert_text, "Checkbox")
    obs = _obs(1, ActionType.assert_state, "Checkbox")
    assert _action_match_score(exp, obs) >= _MATCH_FLOOR


def test_match_score_assert_not_compatible_with_non_assert_action():
    exp = _exp(1, ActionType.assert_text, "Error banner")
    obs = _obs(1, ActionType.navigate, "Error banner")
    assert _action_match_score(exp, obs) == 0.0


def test_match_score_above_floor_when_action_and_target_match():
    exp = _exp(1, ActionType.click, "Sign in button")
    obs = _obs(1, ActionType.click, "Sign in button")
    score = _action_match_score(exp, obs)
    assert score >= 0.7


def test_match_score_penalised_when_target_differs():
    exp = _exp(1, ActionType.click, "Sign in button")
    obs = _obs(1, ActionType.click, "Cancel button")
    score_match = _action_match_score(exp, _obs(1, ActionType.click, "Sign in button"))
    score_mismatch = _action_match_score(exp, obs)
    assert score_mismatch < score_match


def test_match_score_both_targets_none_contributes_full_target_weight():
    exp = _exp(1, ActionType.click, None)
    obs = _obs(1, ActionType.click, None)
    score = _action_match_score(exp, obs)
    assert score >= 0.7


def test_match_score_one_target_none_does_not_add_target_bonus():
    exp = _exp(1, ActionType.click, "Submit")
    obs = _obs(1, ActionType.click, None)
    score = _action_match_score(exp, obs)
    assert score < 0.7  # only base + input bonus (both None here)


def test_match_score_input_value_match_adds_to_score():
    exp = _exp(1, ActionType.type_text, "Email", input_value="user@example.invalid")
    obs_match = _obs(1, ActionType.type_text, "Email", input_value="user@example.invalid")
    obs_no_input = _obs(1, ActionType.type_text, "Email")
    assert _action_match_score(exp, obs_match) > _action_match_score(exp, obs_no_input)


@pytest.mark.parametrize(
    ("expected_target", "observed_target"),
    [
        ("username", "Username input field"),
        ("password", "Password input field"),
    ],
)
def test_target_similarity_ignores_generic_ui_structure(
    expected_target: str,
    observed_target: str,
):
    assert _target_similarity(expected_target, observed_target) == 100.0
    exp = _exp(1, ActionType.type_text, expected_target)
    obs = _obs(1, ActionType.type_text, observed_target)
    exact_obs = _obs(1, ActionType.type_text, expected_target)
    assert _action_match_score(exp, obs) == _action_match_score(exp, exact_obs)
    assert FindingType.wrong_target not in _classify_pair(exp, obs)


def test_target_similarity_ignores_case_and_whitespace():
    assert _target_similarity("  Username  ", "username") == 100.0


def test_target_similarity_keeps_different_targets_distinct():
    assert _target_similarity("username", "password") < FUZZY_THRESHOLD


@pytest.mark.parametrize(
    ("expected_target", "observed_target"),
    [
        ("input", "field"),
        ("input field", "field input"),
        ("username input", "password input"),
    ],
)
def test_target_similarity_does_not_equate_structural_words_alone(
    expected_target: str,
    observed_target: str,
):
    assert _target_similarity(expected_target, observed_target) < FUZZY_THRESHOLD


@pytest.mark.parametrize(
    ("expected_target", "observed_target"),
    [
        ("Next", "Next button"),
        ("Login", "Login button"),
        ("Save", "Save button"),
        ("Next", "Next btn"),
        ("Continue", "Continue button"),
        ("Sign in", "Sign in button"),
    ],
)
def test_target_similarity_control_chrome_equality(
    expected_target: str,
    observed_target: str,
):
    """Bare CTA labels match the same label with button/btn chrome exactly."""
    assert _target_similarity(expected_target, observed_target) == 100.0
    assert _target_similarity(observed_target, expected_target) == 100.0


def test_chrome_equality_requires_non_empty_meaningful_tokens():
    """Pure chrome labels must not get chrome-equivalence 100.0."""
    assert _target_similarity("button", "btn") != 100.0
    assert _target_similarity("btn", "button") != 100.0
    assert _target_similarity("button", "button btn") != 100.0


def test_chrome_equality_next_vs_next_button():
    assert _target_similarity("Next", "Next button") == 100.0


def test_chrome_equality_login_vs_login_button():
    assert _target_similarity("Login", "Login button") == 100.0


@pytest.mark.parametrize(
    ("expected_target", "observed_target"),
    [
        ("Cancel", "Submit order"),
        ("OK", "Cancel button"),
        ("Next", "Back button"),
    ],
)
def test_target_similarity_control_chrome_keeps_distinct_controls_below_threshold(
    expected_target: str,
    observed_target: str,
):
    assert _target_similarity(expected_target, observed_target) < FUZZY_THRESHOLD


@pytest.mark.parametrize(
    ("expected_target", "observed_target"),
    [
        ("Go button", "Google button"),
        ("Mentorship title", "Join Mentorship button"),
        ("Log button", "Logout button"),
    ],
)
def test_control_chrome_equality_does_not_force_unrelated_stems(
    expected_target: str,
    observed_target: str,
):
    """Chrome strip must not invent exact equality for different meaning tokens.

    Some of these pairs already score above FUZZY_THRESHOLD via pre-existing
    fuzzy overlap; chrome normalization must not upgrade them to 100.0.
    """
    assert _target_similarity(expected_target, observed_target) < 100.0


def test_control_chrome_clears_wrong_target_on_click_pair():
    exp = _exp(1, ActionType.click, "Next")
    obs = _obs(1, ActionType.click, "Next button")
    assert FindingType.wrong_target not in _classify_pair(exp, obs)


def test_control_chrome_does_not_broaden_assert_target_role_conflict():
    """Assertion path still rejects Mentorship title ← Join Mentorship button."""
    from reckon.tolerance import assertion_satisfied, assertion_target_relevant

    obs = ObservedAction(
        index=1,
        action=ActionType.assert_visible,
        target="Join Mentorship button",
        timestamp_ms=1000,
        assertion_evidence=AssertionEvidence(
            description="Join Mentorship CTA",
            visible_text="Join Mentorship",
        ),
        confidence=0.9,
    )
    # Chrome path must not declare exact equality for title vs CTA.
    assert _target_similarity("Mentorship title", "Join Mentorship button") < 100.0
    assert not assertion_target_relevant("Mentorship title", obs)
    assert not assertion_satisfied(
        "Mentorship",
        obs,
        expected_target="Mentorship title",
    )


def test_control_chrome_does_not_equate_unrelated_assert_targets():
    """Non-click actions do not gain chrome synonym matching."""
    exp = _exp(1, ActionType.assert_visible, "Hub screen")
    obs = _obs(1, ActionType.assert_visible, "Welcome page button")
    assert _target_similarity(exp.target or "", obs.target or "") < FUZZY_THRESHOLD
    assert FindingType.wrong_target in _classify_pair(exp, obs)


def test_align_next_vs_next_button_does_not_report_wrong_target():
    expected = ExpectedTrace(
        scenario_title="Next chrome",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Next"),
        ],
    )
    observed = ObservedTrace(
        source_file="chrome.mp4",
        actions=[
            ObservedAction(
                index=1,
                action=ActionType.click,
                target="Next button",
                timestamp_ms=1000,
                confidence=0.9,
            ),
        ],
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed
    assert not any(f.type == FindingType.wrong_target for f in result.findings)


# ---------------------------------------------------------------------------
# _dp_align — structure
# ---------------------------------------------------------------------------

def test_dp_align_empty_both_returns_empty():
    assert _dp_align([], []) == []


def test_dp_align_empty_expected_all_extra():
    obs = [_obs(1, ActionType.click, "Submit")]
    pairs = _dp_align([], obs)
    assert len(pairs) == 1
    assert pairs[0][0] is None
    assert pairs[0][1] is obs[0]


def test_dp_align_empty_observed_all_missing():
    exp = [_exp(1, ActionType.click, "Submit")]
    pairs = _dp_align(exp, [])
    assert len(pairs) == 1
    assert pairs[0][0] is exp[0]
    assert pairs[0][1] is None


def test_dp_align_identical_sequences_all_matched():
    exps = [_exp(i, ActionType.click, "Button") for i in range(1, 4)]
    obss = [_obs(i, ActionType.click, "Button") for i in range(1, 4)]
    pairs = _dp_align(exps, obss)
    assert all(e is not None and o is not None for e, o in pairs)
    assert len(pairs) == 3


def test_dp_align_preserves_order():
    exps = [_exp(1, ActionType.navigate, "/login"), _exp(2, ActionType.click, "Submit")]
    obss = [_obs(1, ActionType.navigate, "/login"), _obs(2, ActionType.click, "Submit")]
    pairs = _dp_align(exps, obss)
    matched = [(e, o) for e, o in pairs if e and o]
    assert matched[0][0].action == ActionType.navigate
    assert matched[1][0].action == ActionType.click


def test_align_open_order_prefers_exact_detail_navigation():
    expected = _make_expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="order #4211",
            raw_text="Open order #4211",
        ),
    ])
    observed = _make_observed([
        ObservedAction(
            index=1,
            timestamp_ms=48_000,
            action=ActionType.navigate,
            target="shop.test/orders/4211",
            confidence=0.95,
            screen_description="Order details page for Order #4211 is displayed.",
        ),
        ObservedAction(
            index=2,
            timestamp_ms=51_000,
            action=ActionType.click,
            target="Order total field",
            confidence=0.55,
            screen_description="Order #4211 detail page is displayed.",
        ),
    ])

    result = align(expected, observed)

    verdict = result.step_verdicts[0]
    assert verdict.passed
    assert verdict.matched_observed_index == 1
    assert verdict.evidence_timestamp_ms == 48_000


def test_align_open_order_rejects_different_identifier():
    expected = _make_expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="order #4211",
            raw_text="Open order #4211",
        ),
    ])
    observed = _make_observed([
        ObservedAction(
            index=1,
            timestamp_ms=48_000,
            action=ActionType.navigate,
            target="shop.test/orders/42110",
            confidence=0.95,
            screen_description="Order details page for Order #42110 is displayed.",
        ),
    ])

    result = align(expected, observed)

    assert not result.step_verdicts[0].passed
    assert result.step_verdicts[0].matched_observed_index is None


def test_align_open_order_rejects_unrelated_navigation():
    expected = _make_expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="order #4211",
            raw_text="Open order #4211",
        ),
    ])
    observed = _make_observed([
        ObservedAction(
            index=1,
            timestamp_ms=48_000,
            action=ActionType.navigate,
            target="shop.test/dashboard",
            confidence=0.95,
            screen_description="Dashboard page is displayed.",
        ),
    ])

    result = align(expected, observed)

    assert not result.step_verdicts[0].passed
    assert result.step_verdicts[0].matched_observed_index is None


def test_align_assertion_prefers_direct_evidence_over_later_similar_toast():
    expected = _make_expected([
        _exp(
            1,
            ActionType.assert_text,
            "confirmation",
            assertion="Refund issued",
        ),
    ])
    observed = _make_observed([
        _obs(
            1,
            ActionType.assert_visible,
            "Success banner",
            timestamp_ms=74_000,
            evidence=AssertionEvidence(
                description="Refund success message shown.",
                visible_text="Refund issued",
            ),
        ),
        _obs(
            2,
            ActionType.assert_visible,
            "green success banner",
            timestamp_ms=78_000,
            evidence=AssertionEvidence(
                description="Green banner confirming refund issuance.",
                visible_text="Refund issued",
            ),
        ),
        _obs(
            3,
            ActionType.assert_visible,
            "toast notification",
            timestamp_ms=80_000,
            evidence=AssertionEvidence(
                description="Toast confirming refund was recorded.",
                visible_text="Refund recorded",
            ),
        ),
    ])

    result = align(expected, observed)

    verdict = result.step_verdicts[0]
    assert verdict.passed
    assert verdict.matched_observed_index in {1, 2}
    assert verdict.evidence_timestamp_ms in {74_000, 78_000}


def test_align_assertion_prefers_relevant_target_over_unrelated_exact_text():
    expected = _make_expected([
        _exp(
            1,
            ActionType.assert_visible,
            "Refund success banner",
            assertion="Refund issued",
        ),
    ])
    observed = _make_observed([
        _obs(
            1,
            ActionType.assert_visible,
            "Account history panel",
            timestamp_ms=1000,
            evidence=AssertionEvidence(
                description="Earlier refund record.",
                visible_text="Refund issued",
            ),
        ),
        _obs(
            2,
            ActionType.assert_visible,
            "green success banner",
            timestamp_ms=2000,
            evidence=AssertionEvidence(
                description="Current refund status.",
                visible_text="Refund recorded",
            ),
        ),
    ])

    result = align(expected, observed)

    verdict = result.step_verdicts[0]
    assert verdict.matched_observed_index == 2
    assert verdict.passed is False
    assert FindingType.assertion_not_observed in verdict.finding_types
    assert result.verdict == AuditVerdict.unfaithful


def test_align_assertion_rejects_exact_text_on_only_unrelated_target():
    expected = _make_expected([
        _exp(
            1,
            ActionType.assert_visible,
            "Refund success banner",
            assertion="Refund issued",
        ),
    ])
    observed = _make_observed([
        _obs(
            1,
            ActionType.assert_visible,
            "Account history panel",
            evidence=AssertionEvidence(
                description="Earlier refund record.",
                visible_text="Refund issued",
            ),
        ),
    ])

    result = align(expected, observed)

    findings = {
        finding.type: finding
        for finding in result.findings
        if finding.expected_step_index == 1
    }
    assert FindingType.wrong_target in findings
    assert FindingType.assertion_not_observed in findings
    assert findings[FindingType.assertion_not_observed].severity == FindingSeverity.critical
    assert result.step_verdicts[0].passed is False
    assert result.verdict == AuditVerdict.unfaithful


@pytest.mark.parametrize("target", ["message", "confirmation"])
def test_align_assertion_accepts_valid_direct_evidence_for_generic_target(
    target: str,
):
    expected = _make_expected([
        _exp(
            1,
            ActionType.assert_visible,
            target,
            assertion="Refund issued",
        ),
    ])
    observed = _make_observed([
        _obs(
            1,
            ActionType.assert_visible,
            "Success banner",
            evidence=AssertionEvidence(
                description="Current refund confirmation.",
                visible_text="Refund issued",
            ),
        ),
    ])

    result = align(expected, observed)

    assert result.step_verdicts[0].passed is True
    assert result.findings == []


def test_align_dismissal_prefers_close_control_over_unrelated_click():
    expected = _make_expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="account panel",
            raw_text="Dismiss account panel",
        ),
    ])
    observed = _make_observed([
        _obs(1, ActionType.click, "Account panel details link"),
        ObservedAction(
            index=2,
            timestamp_ms=2000,
            action=ActionType.click,
            target="X close button on account panel",
            confidence=0.9,
            screen_description="The account panel is dismissed.",
        ),
    ])

    result = align(expected, observed)

    assert result.step_verdicts[0].passed is True
    assert result.step_verdicts[0].matched_observed_index == 2
    assert result.step_verdicts[0].evidence_timestamp_ms == 2000


def test_align_dismissal_does_not_pair_with_scroll_only_observation():
    expected = _make_expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="account panel",
            raw_text="Close account panel",
        ),
    ])
    observed = _make_observed([
        _obs(1, ActionType.scroll, "account panel"),
    ])

    result = align(expected, observed)

    assert result.step_verdicts[0].matched_observed_index is None
    assert result.step_verdicts[0].finding_types == [FindingType.missing_action]
    assert {finding.type for finding in result.findings} == {
        FindingType.missing_action,
        FindingType.extra_action,
    }


def test_align_repeated_dismissal_clicks_do_not_reuse_close_observation():
    expected = _make_expected([
        ExpectedStep(
            index=1,
            action=ActionType.click,
            target="first panel",
            raw_text="Close first panel",
        ),
        ExpectedStep(
            index=2,
            action=ActionType.click,
            target="second panel",
            raw_text="Close second panel",
        ),
    ])
    observed = _make_observed([
        ObservedAction(
            index=1,
            timestamp_ms=2000,
            action=ActionType.click,
            target="X close button on second panel",
            confidence=0.9,
            screen_description="The second panel is dismissed.",
        ),
    ])

    result = align(expected, observed)

    assert result.step_verdicts[0].matched_observed_index is None
    assert result.step_verdicts[1].matched_observed_index == 1
    assert sum(
        verdict.matched_observed_index == 1
        for verdict in result.step_verdicts
    ) == 1


@pytest.mark.parametrize(
    (
        "assertion_before_dismissal",
        "expected_matches",
        "expected_passes",
    ),
    [
        (False, [1, 3, 4, None], [True, True, False, False]),
        (True, [1, 2, 3, 4], [True, True, True, False]),
    ],
)
def test_align_state_assertion_respects_temporal_order(
    assertion_before_dismissal: bool,
    expected_matches: list[int | None],
    expected_passes: list[bool],
):
    opening = ExpectedStep(
        index=1,
        action=ActionType.click,
        target="Policy notice link",
    )
    assertion = ExpectedStep(
        index=2 if assertion_before_dismissal else 3,
        action=ActionType.assert_visible,
        target="notice page",
        assertion="Policy notice is visible",
    )
    dismissal = ExpectedStep(
        index=3 if assertion_before_dismissal else 2,
        action=ActionType.click,
        target="notice page",
        raw_text="Close notice page",
    )
    destination = ExpectedStep(
        index=4,
        action=ActionType.assert_visible,
        target="language selection page",
        assertion="Language selection page is visible",
    )
    middle_steps = (
        [assertion, dismissal]
        if assertion_before_dismissal
        else [dismissal, assertion]
    )
    expected = _make_expected([opening, *middle_steps, destination])
    observed = _make_observed([
        _obs(
            1,
            ActionType.click,
            "Policy notice link",
            timestamp_ms=100,
        ),
        ObservedAction(
            index=2,
            timestamp_ms=500,
            action=ActionType.assert_visible,
            target="notice page",
            assertion_evidence=AssertionEvidence(
                description="Policy notice is visible.",
                visible_text="Policy notice",
            ),
            confidence=0.9,
            screen_description="The policy notice is visible.",
        ),
        ObservedAction(
            index=3,
            timestamp_ms=1000,
            action=ActionType.click,
            target="X close button on notice page",
            confidence=0.9,
            screen_description=(
                "The notice page is dismissed and the Mentorship "
                "information page appears."
            ),
        ),
        ObservedAction(
            index=4,
            timestamp_ms=1000,
            action=ActionType.assert_visible,
            target="Mentorship information page",
            assertion_evidence=AssertionEvidence(
                description="Mentorship details are visible.",
                visible_text="Earn when you mentor new drivers",
            ),
            confidence=0.9,
            screen_description="The Mentorship information page is visible.",
        ),
    ])

    result = align(expected, observed)

    matches = [
        verdict.matched_observed_index
        for verdict in result.step_verdicts
    ]
    assert matches == expected_matches
    assert [
        verdict.passed for verdict in result.step_verdicts
    ] == expected_passes

    used_matches = [match for match in matches if match is not None]
    assert used_matches == sorted(set(used_matches))

    assertion_verdict = result.step_verdicts[assertion.index - 1]
    dismissal_verdict = result.step_verdicts[dismissal.index - 1]
    if assertion_before_dismissal:
        assert assertion_verdict.matched_observed_index == 2
        assert assertion_verdict.evidence_timestamp_ms == 500
        assert dismissal_verdict.matched_observed_index == 3
        assert dismissal_verdict.evidence_timestamp_ms == 1000
    else:
        assert dismissal_verdict.matched_observed_index == 3
        assert assertion_verdict.matched_observed_index == 4
        assert assertion_verdict.passed is False
        assert FindingType.assertion_not_observed in assertion_verdict.finding_types

    language_verdict = result.step_verdicts[3]
    assert language_verdict.passed is False
    if assertion_before_dismissal:
        assert language_verdict.matched_observed_index == 4
        assert set(language_verdict.finding_types) == {
            FindingType.wrong_target,
            FindingType.assertion_not_observed,
        }
    else:
        assert language_verdict.matched_observed_index is None
        assert language_verdict.finding_types == [FindingType.missing_action]


def test_dp_align_extra_observed_produces_gap():
    exps = [_exp(1, ActionType.click, "Submit")]
    obss = [_obs(1, ActionType.scroll, "Page"), _obs(2, ActionType.click, "Submit")]
    pairs = _dp_align(exps, obss)
    extras = [(e, o) for e, o in pairs if e is None]
    assert len(extras) == 1
    assert extras[0][1].action == ActionType.scroll


def test_dp_align_missing_expected_produces_gap():
    exps = [_exp(1, ActionType.navigate, "/login"), _exp(2, ActionType.type_text, "Email")]
    obss = [_obs(1, ActionType.navigate, "/login")]
    pairs = _dp_align(exps, obss)
    missing = [(e, o) for e, o in pairs if o is None]
    assert len(missing) == 1
    assert missing[0][0].action == ActionType.type_text


# ---------------------------------------------------------------------------
# _classify_pair
# ---------------------------------------------------------------------------

def test_classify_pair_no_findings_on_perfect_match():
    exp = _exp(1, ActionType.click, "Submit", input_value=None, assertion=None)
    obs = _obs(1, ActionType.click, "Submit")
    assert _classify_pair(exp, obs) == []


def test_classify_pair_wrong_target_when_targets_dissimilar():
    exp = _exp(1, ActionType.click, "Sign in button")
    obs = _obs(1, ActionType.click, "Logout button")
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_target in findings


def test_classify_pair_wrong_target_when_observed_target_absent():
    exp = _exp(1, ActionType.click, "Submit")
    obs = _obs(1, ActionType.click, None)
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_target in findings


def test_classify_pair_no_wrong_target_when_expected_target_absent():
    exp = _exp(1, ActionType.click, None)
    obs = _obs(1, ActionType.click, "Anything")
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_target not in findings


def test_classify_pair_wrong_input_value_when_values_dissimilar():
    exp = _exp(1, ActionType.type_text, "Email", input_value="correct_user@example.invalid")
    obs = _obs(1, ActionType.type_text, "Email", input_value="xyz")
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_input_value in findings


def test_classify_pair_wrong_input_value_when_digits_differ():
    exp = _exp(1, ActionType.type_text, "Amount", input_value="4211")
    obs = _obs(1, ActionType.type_text, "Amount", input_value="4212")
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_input_value in findings


def test_classify_pair_wrong_input_value_when_observed_value_absent():
    exp = _exp(1, ActionType.type_text, "Email", input_value="user@example.invalid")
    obs = _obs(1, ActionType.type_text, "Email", input_value=None)
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_input_value not in findings


def test_classify_pair_no_wrong_input_when_expected_value_absent():
    exp = _exp(1, ActionType.click, "Submit", input_value=None)
    obs = _obs(1, ActionType.click, "Submit", input_value="extra")
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_input_value not in findings


def test_classify_pair_assertion_not_observed_when_evidence_missing():
    exp = _exp(1, ActionType.assert_text, "Error banner", assertion="Invalid email or password")
    obs = _obs(1, ActionType.assert_text, "Error banner")
    findings = _classify_pair(exp, obs)
    assert FindingType.assertion_not_observed in findings


def test_classify_pair_assertion_passes_with_matching_evidence():
    ev = AssertionEvidence(description="Error shown", visible_text="Invalid email or password")
    exp = _exp(1, ActionType.assert_text, "Error banner", assertion="Invalid email or password")
    obs = _obs(1, ActionType.assert_text, "Error banner", evidence=ev)
    findings = _classify_pair(exp, obs)
    assert FindingType.assertion_not_observed not in findings


@pytest.mark.parametrize(
    ("assertion", "visible_text"),
    [
        ("Login successful", "Login failed"),
        ("Login successful", "Login not successful"),
        ("Refund issued", "Refund failed"),
        ("Refund issued", "Refund recorded"),
    ],
)
def test_classify_pair_assertion_contradictions_remain_findings(
    assertion: str,
    visible_text: str,
):
    evidence = AssertionEvidence(description=visible_text, visible_text=visible_text)
    exp = _exp(1, ActionType.assert_text, "Status", assertion=assertion)
    obs = _obs(1, ActionType.assert_text, "Status", evidence=evidence)
    findings = _classify_pair(exp, obs)
    assert FindingType.assertion_not_observed in findings


def test_classify_pair_terms_page_assertion_requires_evidence_text_match():
    """Screen/target nouns alone do not confirm an assertion; evidence text must."""
    exp = _exp(
        1,
        ActionType.assert_visible,
        "terms and condtions page",
        assertion="User Should able to launch terms and condtions page",
    )
    obs_screen_only = ObservedAction(
        index=1,
        timestamp_ms=500,
        action=ActionType.assert_visible,
        target="Terms and Conditions text",
        assertion_evidence=AssertionEvidence(
            description="Terms and conditions overlay text is visible.",
            visible_text="Terms and Conditions",
        ),
        confidence=0.9,
        screen_description="Terms and conditions text is visible in an overlay.",
    )
    assert FindingType.assertion_not_observed in _classify_pair(exp, obs_screen_only)

    obs_with_claim = ObservedAction(
        index=1,
        timestamp_ms=500,
        action=ActionType.assert_visible,
        target="Terms and Conditions page",
        assertion_evidence=AssertionEvidence(
            description="User launched terms and conditions page",
            visible_text="User Should able to launch terms and condtions page",
        ),
        confidence=0.9,
        screen_description="Terms and conditions text is visible in an overlay.",
    )
    assert _classify_pair(exp, obs_with_claim) == []


def test_classify_pair_terms_page_assertion_rejects_unrelated_page():
    exp = _exp(
        1,
        ActionType.assert_visible,
        "terms and condtions page",
        assertion="User Should able to launch terms and condtions page",
    )
    obs = ObservedAction(
        index=1,
        timestamp_ms=500,
        action=ActionType.assert_visible,
        target="Mentorship information page",
        assertion_evidence=AssertionEvidence(
            description="Mentorship program details are visible.",
            visible_text="Earn when you mentor new drivers",
        ),
        confidence=0.95,
        screen_description="The mentorship information page is visible.",
    )

    findings = _classify_pair(exp, obs)

    assert FindingType.assertion_not_observed in findings


def test_classify_pair_terms_page_assertion_rejects_contradictory_state():
    exp = _exp(
        1,
        ActionType.assert_visible,
        "terms and condtions page",
        assertion="User Should able to launch terms and condtions page",
    )
    obs = ObservedAction(
        index=1,
        timestamp_ms=500,
        action=ActionType.assert_visible,
        target="Terms and Conditions text",
        assertion_evidence=AssertionEvidence(
            description="Terms page not visible.",
            visible_text=None,
        ),
        confidence=0.95,
        screen_description="The terms page is not visible.",
    )

    findings = _classify_pair(exp, obs)

    assert FindingType.assertion_not_observed in findings


def test_classify_pair_dismiss_terms_accepts_close_control():
    exp = ExpectedStep(
        index=1,
        action=ActionType.click,
        target="terms",
        raw_text="2. Dismiss terms after reading",
    )
    obs = ObservedAction(
        index=1,
        timestamp_ms=3766,
        action=ActionType.click,
        target="X close button on terms overlay",
        confidence=0.72,
        screen_description="The terms overlay is dismissed.",
    )

    assert FindingType.wrong_target not in _classify_pair(exp, obs)


def test_classify_pair_dismiss_terms_rejects_different_dialog_close():
    exp = ExpectedStep(
        index=1,
        action=ActionType.click,
        target="terms",
        raw_text="Dismiss terms after reading",
    )
    obs = ObservedAction(
        index=1,
        timestamp_ms=3766,
        action=ActionType.click,
        target="Close button on privacy dialog",
        confidence=0.9,
        screen_description="The privacy dialog is dismissed.",
    )

    assert FindingType.wrong_target in _classify_pair(exp, obs)


def test_classify_pair_dismiss_terms_rejects_unrelated_overlay_click():
    exp = ExpectedStep(
        index=1,
        action=ActionType.click,
        target="terms",
        raw_text="Dismiss terms after reading",
    )
    obs = ObservedAction(
        index=1,
        timestamp_ms=3766,
        action=ActionType.click,
        target="Read more link in terms overlay",
        confidence=0.9,
        screen_description="The terms overlay remains open.",
    )

    assert FindingType.wrong_target in _classify_pair(exp, obs)


def test_classify_pair_language_selection_still_rejects_mentorship_page():
    exp = _exp(
        1,
        ActionType.assert_visible,
        "language selection page",
        assertion="Dismissing terms User should land back on language selection page",
    )
    obs = ObservedAction(
        index=1,
        timestamp_ms=3766,
        action=ActionType.assert_visible,
        target="Mentorship information page",
        assertion_evidence=AssertionEvidence(
            description="Mentorship program details are visible.",
            visible_text="Earn when you mentor new drivers",
        ),
        confidence=0.95,
        screen_description="The mentorship information page is visible.",
    )

    findings = set(_classify_pair(exp, obs))

    assert FindingType.wrong_target in findings
    assert FindingType.assertion_not_observed in findings


def test_classify_pair_multiple_findings_on_same_step():
    exp = _exp(1, ActionType.type_text, "Email", input_value="correct_user@example.invalid",
               assertion="Field highlighted in red")
    obs = _obs(1, ActionType.type_text, "Username field", input_value="xyz")
    findings = _classify_pair(exp, obs)
    assert FindingType.wrong_target in findings
    assert FindingType.wrong_input_value in findings
    assert FindingType.assertion_not_observed in findings


# ---------------------------------------------------------------------------
# align — passing fixture
# ---------------------------------------------------------------------------

def test_align_passing_trace_has_no_findings():
    result = align(_load_expected(), _load_observed("login_observed_passing.json"))
    assert result.findings == []


def test_align_passing_trace_score_is_one():
    result = align(_load_expected(), _load_observed("login_observed_passing.json"))
    assert result.fidelity_score == 1.0


def test_align_passing_trace_all_steps_have_verdicts():
    expected = _load_expected()
    result = align(expected, _load_observed("login_observed_passing.json"))
    assert len(result.step_verdicts) == len(expected.steps)


def test_align_passing_trace_all_verdicts_passed():
    result = align(_load_expected(), _load_observed("login_observed_passing.json"))
    assert all(v.passed for v in result.step_verdicts)


def test_align_passing_trace_evidence_timestamps_present():
    result = align(_load_expected(), _load_observed("login_observed_passing.json"))
    assert all(v.evidence_timestamp_ms is not None for v in result.step_verdicts)


# ---------------------------------------------------------------------------
# align — broken fixture (recording navigates to dashboard instead of showing error)
# ---------------------------------------------------------------------------

def test_align_broken_trace_has_findings():
    result = align(_load_expected(), _load_observed("login_observed_broken.json"))
    assert len(result.findings) > 0


def test_align_broken_trace_score_below_one():
    result = align(_load_expected(), _load_observed("login_observed_broken.json"))
    assert result.fidelity_score < 1.0


def test_align_broken_trace_missing_action_finding_present():
    result = align(_load_expected(), _load_observed("login_observed_broken.json"))
    types = [f.type for f in result.findings]
    assert FindingType.missing_action in types


def test_align_broken_trace_extra_action_finding_present():
    result = align(_load_expected(), _load_observed("login_observed_broken.json"))
    types = [f.type for f in result.findings]
    assert FindingType.extra_action in types


def test_align_broken_trace_missing_action_has_no_timestamp():
    result = align(_load_expected(), _load_observed("login_observed_broken.json"))
    missing = [f for f in result.findings if f.type == FindingType.missing_action]
    assert all(f.evidence_timestamp_ms is None for f in missing)


def test_align_broken_trace_extra_action_has_timestamp():
    result = align(_load_expected(), _load_observed("login_observed_broken.json"))
    extras = [f for f in result.findings if f.type == FindingType.extra_action]
    assert all(f.evidence_timestamp_ms is not None for f in extras)


def test_align_broken_trace_expected_step_index_on_missing():
    expected = _load_expected()
    result = align(expected, _load_observed("login_observed_broken.json"))
    missing = [f for f in result.findings if f.type == FindingType.missing_action]
    # The assert_text step (index 5) should be the missing one.
    assert any(f.expected_step_index == 5 for f in missing)


def test_align_broken_trace_fidelity_score_reflects_passing_steps():
    expected = _load_expected()
    result = align(expected, _load_observed("login_observed_broken.json"))
    passed = sum(1 for v in result.step_verdicts if v.passed)
    assert result.fidelity_score == pytest.approx(passed / len(expected.steps))


# ---------------------------------------------------------------------------
# align — fidelity score edge cases
# ---------------------------------------------------------------------------

def test_align_empty_both_returns_perfect_score():
    result = align(_make_expected([]), _make_observed([]))
    assert result.fidelity_score == 1.0
    assert result.findings == []
    assert result.step_verdicts == []


def test_align_all_missing_gives_score_zero():
    exps = [_exp(1, ActionType.click, "A"), _exp(2, ActionType.click, "B")]
    result = align(_make_expected(exps), _make_observed([]))
    assert result.fidelity_score == 0.0


def test_align_all_extra_gives_perfect_score_no_verdicts():
    obss = [_obs(1, ActionType.click, "X"), _obs(2, ActionType.scroll, "Page")]
    result = align(_make_expected([]), _make_observed(obss))
    assert result.fidelity_score == 1.0
    assert result.step_verdicts == []
    extra_findings = [f for f in result.findings if f.type == FindingType.extra_action]
    assert len(extra_findings) == 2


def test_align_fidelity_score_is_fraction_passed():
    exps = [_exp(i, ActionType.click, "Btn") for i in range(1, 5)]
    # Provide 2 matching observed actions, leave 2 missing.
    obss = [_obs(i, ActionType.click, "Btn") for i in range(1, 3)]
    result = align(_make_expected(exps), _make_observed(obss))
    assert result.fidelity_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# align — wrong_target and wrong_input_value findings
# ---------------------------------------------------------------------------

def test_align_wrong_target_produces_finding():
    exp = _make_expected([_exp(1, ActionType.click, "Sign in button")])
    obs = _make_observed([_obs(1, ActionType.click, "Logout button")])
    result = align(exp, obs)
    types = [f.type for f in result.findings]
    assert FindingType.wrong_target in types


def test_align_wrong_target_verdict_not_passed():
    exp = _make_expected([_exp(1, ActionType.click, "Sign in button")])
    obs = _make_observed([_obs(1, ActionType.click, "Logout button")])
    result = align(exp, obs)
    assert not result.step_verdicts[0].passed


def test_align_wrong_input_value_produces_finding():
    exp = _make_expected([_exp(1, ActionType.type_text, "Email", input_value="correct_user@example.invalid")])
    obs = _make_observed([_obs(1, ActionType.type_text, "Email", input_value="xyz")])
    result = align(exp, obs)
    types = [f.type for f in result.findings]
    assert FindingType.wrong_input_value in types


def test_align_wrong_input_finding_carries_timestamp():
    exp = _make_expected([_exp(1, ActionType.type_text, "Email", input_value="correct_user@example.invalid")])
    obs = _make_observed([_obs(1, ActionType.type_text, "Email", input_value="xyz",
                                timestamp_ms=4200)])
    result = align(exp, obs)
    wrong_input = [f for f in result.findings if f.type == FindingType.wrong_input_value]
    assert wrong_input[0].evidence_timestamp_ms == 4200


# ---------------------------------------------------------------------------
# align — assertion_not_observed findings
# ---------------------------------------------------------------------------

def test_align_assertion_not_observed_when_evidence_absent():
    exp = _make_expected([
        _exp(1, ActionType.assert_text, "Error banner", assertion="Invalid email or password"),
    ])
    obs = _make_observed([_obs(1, ActionType.assert_text, "Error banner")])
    result = align(exp, obs)
    types = [f.type for f in result.findings]
    assert FindingType.assertion_not_observed in types


def test_align_assertion_not_observed_has_timestamp():
    exp = _make_expected([
        _exp(1, ActionType.assert_text, "Error banner", assertion="Invalid email or password"),
    ])
    obs = _make_observed([_obs(1, ActionType.assert_text, "Error banner", timestamp_ms=8000)])
    result = align(exp, obs)
    findings = [f for f in result.findings if f.type == FindingType.assertion_not_observed]
    assert findings[0].evidence_timestamp_ms == 8000


def test_align_page_assertion_requires_evidence_not_screen_tokens():
    """Screen copy that only shares target nouns must not pass an assertion."""
    expected = _make_expected([
        _exp(1, ActionType.navigate, "Start page"),
        _exp(2, ActionType.click, "Sign in"),
        _exp(
            3,
            ActionType.assert_visible,
            "GitHub login page",
            assertion="the GitHub login page is displayed.",
        ),
    ])
    observed = _make_observed([
        _obs(1, ActionType.navigate, "Start page", timestamp_ms=1000),
        _obs(2, ActionType.click, "Sign in", timestamp_ms=2000),
        ObservedAction(
            index=5,
            timestamp_ms=6500,
            action=ActionType.navigate,
            target="github.com/login",
            confidence=0.95,
            screen_description=(
                "GitHub sign-in page is loaded with the username/email field focused."
            ),
        ),
        ObservedAction(
            index=6,
            timestamp_ms=7500,
            action=ActionType.click,
            target="Username or email address field",
            confidence=0.95,
            screen_description=(
                "GitHub sign-in page with the username/email field focused and "
                "browser autofill suggestions visible."
            ),
        ),
        ObservedAction(
            index=11,
            timestamp_ms=13_266,
            action=ActionType.assert_visible,
            target="Chrome autofill suggestion popup",
            assertion_evidence=AssertionEvidence(
                description="Chrome autofill suggestion popup",
                visible_text="RaviTeja Mukkamula\nManage addresses...",
            ),
            confidence=0.95,
        ),
    ])

    assert not _screen_description_supports_assertion(
        expected.steps[2], observed.actions[2]
    )
    result = align(expected, observed)

    step = result.step_verdicts[2]
    assert step.passed is False
    assert FindingType.assertion_not_observed in step.finding_types


def test_screen_description_support_is_deprecated_noop():
    expected = _exp(
        1,
        ActionType.assert_visible,
        "Account settings page",
        assertion="the Account settings page is displayed.",
    )
    observed = ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=ActionType.navigate,
        target="example.invalid/settings",
        confidence=0.95,
        screen_description="Account settings page is loaded.",
    )

    assert not _screen_description_supports_assertion(expected, observed)


def test_align_assertion_passes_with_visible_text_evidence():
    expected = _make_expected([
        _exp(
            1,
            ActionType.assert_visible,
            "GitHub login page",
            assertion="the GitHub login page is displayed",
        ),
    ])
    observed = _make_observed([
        ObservedAction(
            index=1,
            timestamp_ms=6500,
            action=ActionType.assert_visible,
            target="GitHub login page",
            assertion_evidence=AssertionEvidence(
                description="Sign-in form visible",
                visible_text="the GitHub login page is displayed",
            ),
            confidence=0.95,
            screen_description="GitHub sign-in page is loaded.",
        ),
    ])

    result = align(expected, observed)
    assert result.step_verdicts[0].passed is True
    assert result.verdict == AuditVerdict.faithful


def test_align_studio_settings_distractor_does_not_false_pass_login():
    case = Path(__file__).resolve().parents[1] / (
        "eval/aligner_evidence/ea_coarse_login_spinner_distract"
    )
    expected = ExpectedTrace.model_validate_json(
        (case / "expected.json").read_text(encoding="utf-8")
    )
    observed = ObservedTrace.model_validate_json(
        (case / "observed.json").read_text(encoding="utf-8")
    )

    result = align(expected, observed)
    step2 = result.step_verdicts[1]
    assert step2.passed is False
    assert FindingType.assertion_not_observed in step2.finding_types
    assert result.verdict == AuditVerdict.unfaithful
    # Studio Settings must not be the assertion match.
    matched = {
        action.index: action for action in observed.actions
    }.get(step2.matched_observed_index or -1)
    if matched is not None:
        assert "studio" not in (matched.target or "").lower()


def test_align_preserves_extra_action_after_valid_match():
    expected = _make_expected([
        _exp(1, ActionType.click, "Submit"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.click, "Submit", timestamp_ms=1000),
        _obs(2, ActionType.hover, "Help", timestamp_ms=2000),
    ])

    result = align(expected, observed)

    extras = [
        finding for finding in result.findings
        if finding.type == FindingType.extra_action
    ]
    assert result.step_verdicts[0].matched_observed_index == 1
    assert [finding.observed_action_index for finding in extras] == [2]
    assert result.verdict == AuditVerdict.faithful


def test_align_screen_tokens_do_not_confirm_contradictory_assertion():
    expected = _make_expected([
        _exp(
            1,
            ActionType.assert_visible,
            "Login page",
            assertion="Login successful",
        ),
    ])
    observed = _make_observed([
        ObservedAction(
            index=1,
            timestamp_ms=3000,
            action=ActionType.navigate,
            target="example.invalid/login",
            confidence=0.95,
            screen_description="Login page is visible. Login failed.",
        ),
    ])

    result = align(expected, observed)

    # Navigate is incompatible with assert_visible once screen-support pairing
    # is disabled, so the assertion step is unmatched rather than false-passed.
    assert FindingType.missing_action in {
        finding.type for finding in result.findings
    }
    assert result.step_verdicts[0].matched_observed_index is None
    assert result.step_verdicts[0].passed is False
    assert result.verdict == AuditVerdict.unfaithful


def test_align_assertion_passes_when_evidence_matches():
    ev = AssertionEvidence(description="Error banner shown", visible_text="Invalid email or password")
    exp = _make_expected([
        _exp(1, ActionType.assert_text, "Error banner", assertion="Invalid email or password"),
    ])
    obs = _make_observed([_obs(1, ActionType.assert_text, "Error banner", evidence=ev)])
    result = align(exp, obs)
    assert result.fidelity_score == 1.0
    assert result.findings == []


def test_align_assert_text_passes_against_observed_assert_visible():
    # Regression: an assert_text step must be confirmed by an assert_visible
    # observation carrying matching evidence, not reported as missing_action.
    ev = AssertionEvidence(description="Error banner", visible_text="Your username is invalid!")
    exp = _make_expected([
        _exp(1, ActionType.assert_text, "error message", assertion="Your username is invalid!"),
    ])
    obs = _make_observed([_obs(1, ActionType.assert_visible, "Error banner", evidence=ev)])
    result = align(exp, obs)
    assert result.fidelity_score == 1.0
    assert result.findings == []


# ---------------------------------------------------------------------------
# align — blank strings from the LLM must not fabricate findings
# ---------------------------------------------------------------------------

def test_align_blank_assertion_does_not_fabricate_finding():
    # An LLM may emit "" for an assertion it did not intend; schema normalises it
    # to None, so no assertion_not_observed should be raised.
    exp = ExpectedTrace.model_validate({
        "scenario_title": "test",
        "steps": [{"index": 1, "action": "click", "target": "Submit", "assertion": ""}],
    })
    obs = ObservedTrace.model_validate({
        "source_file": "test.mp4",
        "actions": [{"index": 1, "timestamp_ms": 1000, "action": "click",
                     "target": "Submit", "confidence": 0.9}],
    })
    result = align(exp, obs)
    assert result.findings == []
    assert result.fidelity_score == 1.0


def test_align_blank_target_does_not_fabricate_wrong_target():
    exp = ExpectedTrace.model_validate({
        "scenario_title": "test",
        "steps": [{"index": 1, "action": "click", "target": ""}],
    })
    obs = ObservedTrace.model_validate({
        "source_file": "test.mp4",
        "actions": [{"index": 1, "timestamp_ms": 1000, "action": "click",
                     "target": "", "confidence": 0.9}],
    })
    result = align(exp, obs)
    assert result.findings == []


# ---------------------------------------------------------------------------
# AuditResult schema
# ---------------------------------------------------------------------------

def test_audit_result_is_pydantic_model():
    result = align(_load_expected(), _load_observed("login_observed_passing.json"))
    assert isinstance(result, AuditResult)


def test_audit_result_findings_are_finding_instances():
    result = align(_load_expected(), _load_observed("login_observed_broken.json"))
    assert all(isinstance(f, Finding) for f in result.findings)


def test_audit_result_verdicts_are_step_verdict_instances():
    result = align(_load_expected(), _load_observed("login_observed_passing.json"))
    assert all(isinstance(v, StepVerdict) for v in result.step_verdicts)


def test_audit_result_verdict_indices_match_expected_step_indices():
    expected = _load_expected()
    result = align(expected, _load_observed("login_observed_passing.json"))
    verdict_indices = [v.expected_step_index for v in result.step_verdicts]
    expected_indices = [s.index for s in expected.steps]
    assert verdict_indices == expected_indices


# ---------------------------------------------------------------------------
# Focusing clicks implied by the typing that follows them
# ---------------------------------------------------------------------------

def test_focus_click_passes_when_following_typing_is_observed():
    expected = _make_expected([
        _exp(1, ActionType.click, "Username field"),
        _exp(2, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    result = align(expected, observed)
    assert result.fidelity_score == 1.0
    assert result.findings == []


def test_focus_click_verdict_points_at_the_typing_that_evidenced_it():
    expected = _make_expected([
        _exp(1, ActionType.click, "Username field"),
        _exp(2, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.type_text, "Username field", input_value="tomsmith", timestamp_ms=4200),
    ])
    verdict = next(v for v in align(expected, observed).step_verdicts if v.expected_step_index == 1)
    assert verdict.passed
    assert verdict.evidence_timestamp_ms == 4200


def test_focus_click_still_missing_when_the_typing_was_not_observed():
    """The guard that stops this leniency from manufacturing a pass.

    If the field was never filled, nothing evidences the click and it must still fail.
    """
    expected = _make_expected([
        _exp(1, ActionType.click, "Username field"),
        _exp(2, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    observed = _make_observed([_obs(1, ActionType.click, "Login button")])
    findings = align(expected, observed).findings
    assert FindingType.missing_action in {f.type for f in findings}


def test_focus_click_not_implied_for_a_different_target():
    expected = _make_expected([
        _exp(1, ActionType.click, "Password field"),
        _exp(2, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    missing = [f for f in align(expected, observed).findings if f.type == FindingType.missing_action]
    assert [f.expected_step_index for f in missing] == [1]


def test_focus_click_not_implied_when_it_is_the_final_step():
    expected = _make_expected([
        _exp(1, ActionType.type_text, "Username field", input_value="tomsmith"),
        _exp(2, ActionType.click, "Login button"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    missing = [f for f in align(expected, observed).findings if f.type == FindingType.missing_action]
    assert [f.expected_step_index for f in missing] == [2]


def test_focus_click_implied_by_a_following_select():
    expected = _make_expected([
        _exp(1, ActionType.click, "Country dropdown"),
        _exp(2, ActionType.select, "Country dropdown", input_value="Germany"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.select, "Country dropdown", input_value="Germany"),
    ])
    assert align(expected, observed).fidelity_score == 1.0


def test_unmatched_non_click_step_is_still_missing():
    expected = _make_expected([
        _exp(1, ActionType.type_text, "Username field", input_value="tomsmith"),
        _exp(2, ActionType.type_text, "Password field", input_value="hunter2"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.type_text, "Username field", input_value="tomsmith"),
    ])
    missing = [f for f in align(expected, observed).findings if f.type == FindingType.missing_action]
    assert [f.expected_step_index for f in missing] == [2]


# ---------------------------------------------------------------------------
# Target wording on a confirmed assertion
# ---------------------------------------------------------------------------

def test_confirmed_assertion_does_not_report_wrong_target():
    """A scenario's noun for an assertion is not an element name.

    "The message ... should be visible" parses to target 'message'; the model names the
    same thing 'Success banner'. With the asserted text confirmed, that difference is
    wording, not a fidelity failure.
    """
    expected = _make_expected([
        _exp(1, ActionType.assert_visible, "message", assertion="Welcome back, tomsmith!"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.assert_visible, "Success banner",
             evidence=AssertionEvidence(description="green banner",
                                        visible_text="Welcome back, tomsmith!")),
    ])
    result = align(expected, observed)
    assert result.findings == []
    assert result.fidelity_score == 1.0


def test_unconfirmed_assertion_still_reports_wrong_target():
    expected = _make_expected([
        _exp(1, ActionType.assert_visible, "message", assertion="Welcome back, tomsmith!"),
    ])
    observed = _make_observed([
        _obs(1, ActionType.assert_visible, "Error banner",
             evidence=AssertionEvidence(description="red banner",
                                        visible_text="Your password is invalid!")),
    ])
    found = {f.type for f in align(expected, observed).findings}
    assert FindingType.wrong_target in found
    assert FindingType.assertion_not_observed in found


def test_wrong_target_still_reported_on_a_step_without_an_assertion():
    expected = _make_expected([_exp(1, ActionType.click, "Login button")])
    observed = _make_observed([_obs(1, ActionType.click, "Cancel link")])
    found = {f.type for f in align(expected, observed).findings}
    assert FindingType.wrong_target in found
