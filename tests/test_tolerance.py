from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditVerdict,
    ExpectedStep,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
)
from reckon.tolerance import (
    FUZZY_THRESHOLD,
    actions_compatible,
    assertion_satisfied,
    assertion_target_relevant,
    finding_severity,
    focus_click_evidence,
    values_equivalent,
    verdict_for,
)


def _exp(index: int, action: ActionType, target: str | None = None) -> ExpectedStep:
    return ExpectedStep(index=index, action=action, target=target)


def _obs(index: int, action: ActionType, target: str | None = None,
         evidence: AssertionEvidence | None = None) -> ObservedAction:
    return ObservedAction(index=index, action=action, target=target, timestamp_ms=1000,
                          assertion_evidence=evidence, confidence=0.9)


# ---------------------------------------------------------------------------
# assertion_satisfied
# ---------------------------------------------------------------------------

def test_assertion_satisfied_false_when_no_evidence():
    obs = _obs(1, ActionType.assert_text, "Error banner")
    assert not assertion_satisfied("Invalid email or password", obs)


def test_assertion_satisfied_true_when_visible_text_matches():
    ev = AssertionEvidence(description="Error shown", visible_text="Invalid email or password")
    obs = _obs(1, ActionType.assert_text, "Error banner", evidence=ev)
    assert assertion_satisfied("Invalid email or password", obs)


def test_assertion_satisfied_true_on_partial_match_in_description():
    ev = AssertionEvidence(description="Error banner with text: Invalid email or password")
    obs = _obs(1, ActionType.assert_text, "Error banner", evidence=ev)
    assert assertion_satisfied("Invalid email or password", obs)


def test_assertion_satisfied_false_when_text_is_unrelated():
    ev = AssertionEvidence(description="Dashboard loaded", visible_text="Welcome back")
    obs = _obs(1, ActionType.assert_text, "Error banner", evidence=ev)
    assert not assertion_satisfied("Invalid email or password", obs)


def test_assertion_rejects_carousel_via_menu_label_visible_text():
    """Sheet/menu chrome must not prove a named carousel via shared label text."""
    ev = AssertionEvidence(
        description="Options bottom sheet",
        visible_text="Mentorship settings\nHow it works\nLanguage\nDelete",
    )
    obs = _obs(78, ActionType.assert_visible, "Options bottom sheet", evidence=ev)
    assert not assertion_target_relevant("How it works carousel", obs)
    assert not assertion_satisfied(
        "How it works",
        obs,
        expected_target="How it works carousel",
    )


def test_assertion_accepts_matching_carousel_with_text():
    ev = AssertionEvidence(
        description="How it works carousel page",
        visible_text="How it works\nSwipe to learn more",
    )
    obs = _obs(12, ActionType.assert_visible, "How it works carousel", evidence=ev)
    assert assertion_satisfied(
        "How it works",
        obs,
        expected_target="How it works carousel",
    )


def test_assertion_accepts_carousel_target_synonym():
    """carousel ↔ slider is a configured component-role alias."""
    ev = AssertionEvidence(
        description="Onboarding slider showing How it works",
        visible_text="How it works",
    )
    obs = _obs(12, ActionType.assert_visible, "How it works slider", evidence=ev)
    assert assertion_target_relevant("How it works carousel", obs)
    assert assertion_satisfied(
        "How it works",
        obs,
        expected_target="How it works carousel",
    )


def test_mentorship_title_not_proved_by_join_mentorship_button():
    """E29: language-form Join Mentorship must not clear Mentorship title."""
    ev = AssertionEvidence(
        description="Join Mentorship button becomes enabled after agreeing to terms",
        visible_text="Join Mentorship",
    )
    obs = ObservedAction(
        index=58,
        action=ActionType.assert_state,
        target="Join Mentorship button",
        timestamp_ms=189513,
        assertion_evidence=ev,
        confidence=0.85,
        screen_description="Join Mentorship button changed from disabled grey to enabled black",
    )
    assert not assertion_target_relevant("Mentorship title", obs)
    assert not assertion_satisfied(
        "Mentorship",
        obs,
        expected_target="Mentorship title",
    )


def test_mentorship_title_accepts_dashboard_title_evidence():
    """Legitimate Mentorship title on the dashboard still passes."""
    ev = AssertionEvidence(
        description="Dashboard header title",
        visible_text="Mentorship",
    )
    obs = ObservedAction(
        index=70,
        action=ActionType.assert_visible,
        target="Mentorship title",
        timestamp_ms=240750,
        assertion_evidence=ev,
        confidence=0.9,
        screen_description="Mentor Dashboard empty state / Mentorship page",
    )
    assert assertion_target_relevant("Mentorship title", obs)
    assert assertion_satisfied(
        "Mentorship",
        obs,
        expected_target="Mentorship title",
    )


def test_dashboard_cta_not_proved_by_carousel_page():
    """E32: How it works onboarding page must not clear dashboard CTA."""
    ev = AssertionEvidence(
        description="Onboarding page title and pager indicator",
        visible_text="How it works ... 1 of 6",
    )
    obs = ObservedAction(
        index=64,
        action=ActionType.assert_visible,
        target="How it works page",
        timestamp_ms=213765,
        assertion_evidence=ev,
        confidence=0.85,
    )
    assert not assertion_target_relevant("How it works CTA", obs)
    assert not assertion_satisfied(
        "How it works",
        obs,
        expected_target="How it works CTA",
    )


def test_menu_option_not_proved_by_carousel_page():
    """E39: carousel page must not clear How it works menu option."""
    ev = AssertionEvidence(
        description="Onboarding page title and pager indicator",
        visible_text="How it works ... 1 of 6",
    )
    obs = ObservedAction(
        index=64,
        action=ActionType.assert_visible,
        target="How it works page",
        timestamp_ms=213765,
        assertion_evidence=ev,
        confidence=0.85,
    )
    assert not assertion_target_relevant("How it works option", obs)
    assert not assertion_satisfied(
        "How it works",
        obs,
        expected_target="How it works option",
    )


def test_menu_option_accepts_action_sheet_evidence():
    """Legitimate How it works option on the Mentorship action sheet still passes."""
    ev = AssertionEvidence(
        description="Bottom sheet with options shown",
        visible_text="Mentorship settings, How it works, Past learners, Cancel",
    )
    obs = ObservedAction(
        index=79,
        action=ActionType.assert_visible,
        target="Action sheet menu",
        timestamp_ms=283753,
        assertion_evidence=ev,
        confidence=0.9,
        screen_description="Mentorship page dimmed with bottom action sheet",
    )
    assert assertion_target_relevant("How it works option", obs)
    assert assertion_satisfied(
        "How it works",
        obs,
        expected_target="How it works option",
    )


def test_dashboard_cta_accepts_dashboard_button_evidence():
    """Legitimate How it works CTA on the Mentor Dashboard still passes."""
    ev = AssertionEvidence(
        description="Dashboard CTA",
        visible_text="How it works",
    )
    obs = ObservedAction(
        index=71,
        action=ActionType.assert_visible,
        target="How it works button",
        timestamp_ms=240800,
        assertion_evidence=ev,
        confidence=0.9,
        screen_description="Mentor Dashboard",
    )
    assert assertion_target_relevant("How it works CTA", obs)
    assert assertion_satisfied(
        "How it works",
        obs,
        expected_target="How it works CTA",
    )


def test_background_check_screen_landing_assertion_passes_with_identity_evidence():
    """Land-on-screen claims may use structured screen identity without boilerplate OCR."""
    ev = AssertionEvidence(
        description="Background check page asking for SSN",
        visible_text="Background check",
    )
    obs = ObservedAction(
        index=30,
        action=ActionType.assert_visible,
        target="Background check page",
        timestamp_ms=51266,
        assertion_evidence=ev,
        confidence=0.85,
        screen_description="SSN field and 'I agree and acknowledge' button",
    )
    assert assertion_satisfied(
        "User should land on Background check screen.",
        obs,
        expected_target="Background check screen",
    )


def test_screen_landing_assertion_rejects_different_screen_identity():
    ev = AssertionEvidence(
        description="Vehicle information form",
        visible_text="Add your vehicle",
    )
    obs = ObservedAction(
        index=40,
        action=ActionType.assert_visible,
        target="Vehicle information page",
        timestamp_ms=60000,
        assertion_evidence=ev,
        confidence=0.85,
    )
    assert not assertion_satisfied(
        "User should land on Background check screen.",
        obs,
        expected_target="Background check screen",
    )


def test_non_landing_assertion_still_requires_full_token_containment():
    """Screen-identity shortcut must not weaken ordinary outcome assertions."""
    ev = AssertionEvidence(
        description="Background check page asking for SSN",
        visible_text="Background check",
    )
    obs = ObservedAction(
        index=30,
        action=ActionType.assert_visible,
        target="Background check page",
        timestamp_ms=51266,
        assertion_evidence=ev,
        confidence=0.85,
    )
    assert not assertion_satisfied(
        "Background check passed successfully",
        obs,
        expected_target="Background check screen",
    )


# ---------------------------------------------------------------------------
# actions_compatible
# ---------------------------------------------------------------------------

def test_actions_compatible_identical_types():
    assert actions_compatible(ActionType.click, ActionType.click)


def test_actions_compatible_across_the_assertion_family():
    assert actions_compatible(ActionType.assert_text, ActionType.assert_visible)


def test_actions_compatible_rejects_unrelated_types():
    assert not actions_compatible(ActionType.click, ActionType.type_text)


def test_actions_compatible_rejects_assertion_paired_with_action():
    assert not actions_compatible(ActionType.assert_visible, ActionType.click)


def test_actions_compatible_click_and_select_strict_by_default():
    assert not actions_compatible(ActionType.select, ActionType.click)
    assert not actions_compatible(ActionType.click, ActionType.select)


def test_actions_compatible_click_select_opt_in_for_choice_controls():
    assert actions_compatible(
        ActionType.select,
        ActionType.click,
        allow_choice_control=True,
        expected_target="Language dropdown",
        observed_target="English checkbox",
    )
    # One-sided choice control is not enough — both sides must denote choice UI.
    assert not actions_compatible(
        ActionType.select,
        ActionType.click,
        allow_choice_control=True,
        expected_target="Language",
        observed_target="English checkbox",
    )
    assert not actions_compatible(
        ActionType.select,
        ActionType.click,
        allow_choice_control=True,
        expected_target="Language dropdown",
        observed_target="Submit button",
    )


# ---------------------------------------------------------------------------
# focus_click_evidence
# ---------------------------------------------------------------------------

def test_focus_click_evidence_returns_the_observed_typing():
    typing = _obs(3, ActionType.type_text, "Username field")
    result = focus_click_evidence(
        _exp(1, ActionType.click, "Username field"),
        _exp(2, ActionType.type_text, "Username field"),
        {2: typing},
    )
    assert result is typing


def test_focus_click_evidence_none_when_typing_unobserved():
    result = focus_click_evidence(
        _exp(1, ActionType.click, "Username field"),
        _exp(2, ActionType.type_text, "Username field"),
        {},
    )
    assert result is None


def test_focus_click_evidence_none_for_a_different_target():
    result = focus_click_evidence(
        _exp(1, ActionType.click, "Password field"),
        _exp(2, ActionType.type_text, "Username field"),
        {2: _obs(3, ActionType.type_text, "Username field")},
    )
    assert result is None


def test_focus_click_evidence_none_when_step_is_not_a_click():
    result = focus_click_evidence(
        _exp(1, ActionType.scroll, "Username field"),
        _exp(2, ActionType.type_text, "Username field"),
        {2: _obs(3, ActionType.type_text, "Username field")},
    )
    assert result is None


def test_focus_click_evidence_none_without_a_following_step():
    result = focus_click_evidence(_exp(1, ActionType.click, "Username field"), None, {})
    assert result is None


def test_fuzzy_threshold_is_a_percentage():
    assert 0 < FUZZY_THRESHOLD <= 100


# ---------------------------------------------------------------------------
# values_equivalent — digits are data, not prose
# ---------------------------------------------------------------------------

def test_values_equivalent_rejects_decimal_shift():
    """The false pass that motivated this rule: a refund off by a factor of ten."""
    assert not values_equivalent("89.00", "8.90")


def test_values_equivalent_rejects_adjacent_identifier():
    assert not values_equivalent("4211", "4212")


def test_values_equivalent_rejects_trailing_zero_shift():
    assert not values_equivalent("100", "1000")


def test_values_equivalent_allows_currency_symbol():
    assert values_equivalent("89.00", "£89.00")


def test_values_equivalent_allows_case_and_spacing_on_text():
    assert values_equivalent("tomsmith", "TomSmith")


def test_values_equivalent_rejects_unrelated_text():
    assert not values_equivalent("tomsmith", "completely other")


def test_values_equivalent_identical_values_match():
    assert values_equivalent("89.00", "89.00")


# ---------------------------------------------------------------------------
# assertion_satisfied — figures named in an assertion must actually appear
# ---------------------------------------------------------------------------

def test_assertion_not_satisfied_when_figure_differs():
    obs = _obs(1, ActionType.assert_visible,
               evidence=AssertionEvidence(description="banner", visible_text="Refunded £8.90"))
    assert not assertion_satisfied("Refunded £89.00", obs)


def test_assertion_satisfied_when_evidence_carries_extra_figures():
    """Evidence legitimately holds numbers the assertion never mentioned."""
    obs = _obs(1, ActionType.assert_visible,
               evidence=AssertionEvidence(description="banner",
                                          visible_text="Order #4211 refunded £89.00"))
    assert assertion_satisfied("refunded £89.00", obs)


def test_assertion_not_satisfied_by_the_opposite_outcome():
    """The worst failure the audit can have: confirming the reverse of the scenario.

    'Refund issued' scores 77 against 'Refund failed' on substring similarity, because
    the shared word dominates and the differing word is the one that carries the meaning.
    """
    obs = _obs(1, ActionType.assert_visible,
               evidence=AssertionEvidence(description="red banner", visible_text="Refund failed"))
    assert not assertion_satisfied("Refund issued", obs)


def test_assertion_not_satisfied_by_a_pending_outcome():
    obs = _obs(1, ActionType.assert_visible,
               evidence=AssertionEvidence(description="banner", visible_text="Refund pending"))
    assert not assertion_satisfied("Refund issued", obs)


def test_assertion_satisfied_when_evidence_wraps_the_asserted_text():
    obs = _obs(1, ActionType.assert_visible,
               evidence=AssertionEvidence(description="green banner",
                                          visible_text="Refund issued successfully"))
    assert assertion_satisfied("Refund issued", obs)


# ---------------------------------------------------------------------------
# Mid-entry values — a field caught part-typed
# ---------------------------------------------------------------------------

def test_part_typed_text_is_not_a_wrong_value():
    """Observed on the real recording: 'tomsmith' caught as 'to' mid-entry."""
    assert values_equivalent("tomsmith", "to")


def test_part_typed_figure_is_still_rejected():
    """The digit rule runs first, so a figure that stopped short is not waved through."""
    assert not values_equivalent("89.00", "8")


def test_typing_more_than_expected_is_still_a_wrong_value():
    assert not values_equivalent("to", "tomsmith")


def test_empty_observed_value_is_not_mid_entry():
    assert not values_equivalent("tomsmith", "")


# ---------------------------------------------------------------------------
# finding_severity — which differences bear on fidelity
# ---------------------------------------------------------------------------

def test_assertion_not_observed_is_critical():
    assert finding_severity(FindingType.assertion_not_observed, None) == FindingSeverity.critical


def test_wrong_input_value_is_critical():
    assert finding_severity(FindingType.wrong_input_value, None) == FindingSeverity.critical


def test_missing_action_on_an_assertion_step_is_critical():
    step = ExpectedStep(index=1, action=ActionType.assert_visible, assertion="Refund issued")
    assert finding_severity(FindingType.missing_action, step) == FindingSeverity.critical


def test_missing_action_on_a_plain_step_is_advisory():
    step = ExpectedStep(index=1, action=ActionType.click, target="Login button")
    assert finding_severity(FindingType.missing_action, step) == FindingSeverity.advisory


def test_wrong_target_is_advisory():
    step = ExpectedStep(index=1, action=ActionType.click, target="Login button")
    assert finding_severity(FindingType.wrong_target, step) == FindingSeverity.advisory


def test_extra_action_is_advisory():
    assert finding_severity(FindingType.extra_action, None) == FindingSeverity.advisory


# ---------------------------------------------------------------------------
# verdict_for
# ---------------------------------------------------------------------------

def _finding(finding_type: FindingType, severity: FindingSeverity) -> Finding:
    return Finding(type=finding_type, severity=severity, description="x")


def test_verdict_is_faithful_without_findings():
    assert verdict_for([]) == AuditVerdict.faithful


def test_verdict_is_faithful_when_only_advisory_findings():
    findings = [
        _finding(FindingType.wrong_target, FindingSeverity.advisory),
        _finding(FindingType.extra_action, FindingSeverity.advisory),
    ]
    assert verdict_for(findings) == AuditVerdict.faithful


def test_verdict_is_unfaithful_on_a_single_critical_finding():
    findings = [
        _finding(FindingType.wrong_target, FindingSeverity.advisory),
        _finding(FindingType.assertion_not_observed, FindingSeverity.critical),
    ]
    assert verdict_for(findings) == AuditVerdict.unfaithful


# ---------------------------------------------------------------------------
# Introduced negation — evidence that reverses the asserted outcome
# ---------------------------------------------------------------------------

def test_assertion_not_satisfied_when_observed_introduces_not():
    """Token containment alone cannot catch this.

    Every word of "Payment successful" appears in "Payment was not successful", so
    without a negation guard the reverse outcome confirms the assertion.
    """
    ev = AssertionEvidence(description="Payment status shown",
                           visible_text="Payment was not successful")
    obs = _obs(1, ActionType.assert_text, "Payment status", evidence=ev)
    assert not assertion_satisfied("Payment successful", obs)


def test_assertion_not_satisfied_when_valid_is_negated():
    ev = AssertionEvidence(description="Password validation shown",
                           visible_text="Password is not valid")
    obs = _obs(1, ActionType.assert_text, "Password status", evidence=ev)
    assert not assertion_satisfied("Password is valid", obs)


def test_assertion_not_satisfied_when_contraction_negates():
    ev = AssertionEvidence(description="Login status shown",
                           visible_text="Login isn't successful")
    obs = _obs(1, ActionType.assert_text, "Login status", evidence=ev)
    assert not assertion_satisfied("Login successful", obs)


def test_assertion_satisfied_when_assertion_itself_is_negative():
    """A negative assertion may legitimately be met by negative evidence."""
    ev = AssertionEvidence(description="banner", visible_text="Login not successful")
    obs = _obs(1, ActionType.assert_text, "Login status", evidence=ev)
    assert assertion_satisfied("Login not successful", obs)
