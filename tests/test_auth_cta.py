"""Regression tests for narrow authentication CTA synonym recovery."""

from reckon.aligner import align
from reckon.auth_cta import (
    AUTH_CTA_RECOVERY_REASON,
    auth_cta_canonical,
    auth_ctas_equivalent,
    find_auth_cta_synonym_match,
)
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedAction,
    ObservedTrace,
)
from reckon.tolerance import actions_compatible


def _click(index: int, target: str, *, timestamp_ms: int | None = None) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms if timestamp_ms is not None else index * 100,
        action=ActionType.click,
        target=target,
        confidence=0.95,
    )


def test_auth_cta_canonical_accepts_sign_in_and_login_variants():
    assert auth_cta_canonical("Sign in") == "auth.signin"
    assert auth_cta_canonical("Sign in button") == "auth.signin"
    assert auth_cta_canonical("Login") == "auth.signin"
    assert auth_cta_canonical("Login CTA") == "auth.signin"
    assert auth_cta_canonical("Log in") == "auth.signin"
    assert auth_cta_canonical("Sign in button in GitHub header") == "auth.signin"


def test_auth_cta_canonical_rejects_non_auth_and_conflict_roles():
    assert auth_cta_canonical("Cancel") is None
    assert auth_cta_canonical("Submit") is None
    assert auth_cta_canonical("Continue") is None
    assert auth_cta_canonical("Continue button") is None
    assert auth_cta_canonical("Join Mentorship button") is None
    assert auth_cta_canonical("username field") is None
    assert auth_cta_canonical("Sign in username field") is None
    assert auth_cta_canonical("Pay order") is None


def test_auth_ctas_equivalent_bidirectional():
    assert auth_ctas_equivalent("Sign in button", "Login CTA")
    assert auth_ctas_equivalent("Login CTA", "Sign in button")
    assert not auth_ctas_equivalent("Cancel", "Submit")
    assert not auth_ctas_equivalent("Sign in", "Continue")
    assert not auth_ctas_equivalent("Sign in", "username field")


def test_auth_cta_does_not_change_global_action_compatibility():
    assert actions_compatible(ActionType.click, ActionType.click)
    assert not actions_compatible(ActionType.click, ActionType.type_text)


def test_find_auth_cta_synonym_match_positive_sign_in_login():
    step = ExpectedStep(index=1, action=ActionType.click, target="Sign in button")
    observed = ObservedTrace(
        source_file="syn.mp4",
        actions=[_click(1, "Login CTA", timestamp_ms=100)],
    )
    match = find_auth_cta_synonym_match(
        step,
        observed,
        window=(1, 1),
        claimed_obs_indices=set(),
    )
    assert match is not None
    assert match.index == 1
    assert match.timestamp_ms == 100


def test_find_auth_cta_synonym_match_rejects_cancel_submit():
    step = ExpectedStep(index=1, action=ActionType.click, target="Cancel")
    observed = ObservedTrace(
        source_file="syn.mp4",
        actions=[_click(1, "Submit")],
    )
    assert (
        find_auth_cta_synonym_match(
            step,
            observed,
            window=(1, 1),
            claimed_obs_indices=set(),
        )
        is None
    )


def test_find_auth_cta_synonym_match_rejects_text_input():
    step = ExpectedStep(index=1, action=ActionType.click, target="Sign in")
    observed = ObservedTrace(
        source_file="syn.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=100,
                action=ActionType.type_text,
                target="Login",
                input_value="x",
                confidence=0.9,
            )
        ],
    )
    assert (
        find_auth_cta_synonym_match(
            step,
            observed,
            window=(1, 1),
            claimed_obs_indices=set(),
        )
        is None
    )


def test_find_auth_cta_synonym_match_skips_claimed_observation():
    step = ExpectedStep(index=1, action=ActionType.click, target="Sign in")
    observed = ObservedTrace(
        source_file="syn.mp4",
        actions=[_click(1, "Login")],
    )
    assert (
        find_auth_cta_synonym_match(
            step,
            observed,
            window=(1, 1),
            claimed_obs_indices={1},
        )
        is None
    )


def test_align_recovers_sign_in_to_login():
    expected = ExpectedTrace(
        scenario_title="Sign in",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Sign in button",
                raw_text="Click Sign in",
            )
        ],
    )
    observed = ObservedTrace(
        source_file="wt_synonym_sign_in.mp4",
        actions=[_click(1, "Login CTA", timestamp_ms=100)],
    )
    result = align(expected, observed)
    verdict = result.step_verdicts[0]
    assert verdict.passed is True
    assert verdict.finding_types == []
    assert verdict.matched_observed_index == 1
    assert verdict.evidence_timestamp_ms == 100
    assert verdict.recovery_reason == AUTH_CTA_RECOVERY_REASON
    assert verdict.cleared_by_assist is False
    assert not any(f.type == FindingType.wrong_target for f in result.findings)


def test_align_recovers_login_to_sign_in():
    expected = ExpectedTrace(
        scenario_title="Login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Login CTA",
                raw_text="Click Login",
            )
        ],
    )
    observed = ObservedTrace(
        source_file="login.mp4",
        actions=[_click(1, "Sign in button", timestamp_ms=250)],
    )
    result = align(expected, observed)
    verdict = result.step_verdicts[0]
    assert verdict.passed is True
    assert verdict.evidence_timestamp_ms == 250
    assert verdict.recovery_reason == AUTH_CTA_RECOVERY_REASON


def test_align_recovers_github_header_sign_in():
    expected = ExpectedTrace(
        scenario_title="GitHub",
        steps=[
            ExpectedStep(index=1, action=ActionType.navigate, target="GitHub homepage"),
            ExpectedStep(index=2, action=ActionType.click, target="Sign in"),
        ],
    )
    observed = ObservedTrace(
        source_file="github.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=50,
                action=ActionType.navigate,
                target="github.com",
                confidence=0.9,
            ),
            _click(2, "Sign in button in GitHub header", timestamp_ms=200),
        ],
    )
    result = align(expected, observed)
    assert result.step_verdicts[1].passed is True
    assert result.step_verdicts[1].matched_observed_index == 2
    assert result.step_verdicts[1].evidence_timestamp_ms == 200
    # Same auth phrase may already score high enough that recovery is a no-op.
    assert result.step_verdicts[1].recovery_reason in {
        None,
        AUTH_CTA_RECOVERY_REASON,
    }


def test_align_does_not_equate_cancel_and_submit():
    expected = ExpectedTrace(
        scenario_title="Dialog",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Cancel"),
        ],
    )
    observed = ObservedTrace(
        source_file="dlg.mp4",
        actions=[_click(1, "Submit")],
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert FindingType.wrong_target in result.step_verdicts[0].finding_types
    assert result.step_verdicts[0].recovery_reason is None


def test_align_does_not_equate_sign_in_with_unrelated_button():
    expected = ExpectedTrace(
        scenario_title="Sign in",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Sign in button"),
        ],
    )
    observed = ObservedTrace(
        source_file="unrelated.mp4",
        actions=[_click(1, "Download report button")],
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert FindingType.wrong_target in result.step_verdicts[0].finding_types
    assert result.step_verdicts[0].recovery_reason is None


def test_align_does_not_equate_auth_cta_with_text_input():
    expected = ExpectedTrace(
        scenario_title="Sign in",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Sign in"),
        ],
    )
    observed = ObservedTrace(
        source_file="field.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=100,
                action=ActionType.type_text,
                target="Login field",
                input_value="user",
                confidence=0.9,
            )
        ],
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert result.step_verdicts[0].recovery_reason is None


def test_align_does_not_equate_auth_cta_with_continue():
    expected = ExpectedTrace(
        scenario_title="Sign in",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Sign in"),
        ],
    )
    observed = ObservedTrace(
        source_file="continue.mp4",
        actions=[_click(1, "Continue")],
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert FindingType.wrong_target in result.step_verdicts[0].finding_types
    assert result.step_verdicts[0].recovery_reason is None


def test_align_claimed_observation_not_reused_for_second_auth_cta():
    expected = ExpectedTrace(
        scenario_title="Double sign in",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Sign in"),
            ExpectedStep(index=2, action=ActionType.click, target="Login"),
        ],
    )
    observed = ObservedTrace(
        source_file="once.mp4",
        actions=[_click(1, "Login CTA", timestamp_ms=100)],
    )
    result = align(expected, observed)
    passed = [v for v in result.step_verdicts if v.passed]
    failed = [v for v in result.step_verdicts if not v.passed]
    assert len(passed) == 1
    assert len(failed) == 1
    assert passed[0].matched_observed_index == 1
    assert failed[0].recovery_reason is None
    assert FindingType.missing_action in failed[0].finding_types or (
        FindingType.wrong_target in failed[0].finding_types
    )


def test_align_out_of_window_auth_cta_not_recovered():
    """Login after the next trusted assertion stays outside the search window."""
    expected = ExpectedTrace(
        scenario_title="Anchored",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Sign in"),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="Dashboard",
                assertion="Dashboard is visible",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="window.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=100,
                action=ActionType.assert_visible,
                target="Dashboard",
                confidence=0.95,
                assertion_evidence=AssertionEvidence(
                    description="Dashboard is visible",
                    visible_text="Dashboard",
                ),
                screen_description="Dashboard home",
            ),
            _click(2, "Login CTA", timestamp_ms=300),
        ],
    )
    result = align(expected, observed)
    sign_in = result.step_verdicts[0]
    assert sign_in.passed is False
    assert FindingType.missing_action in sign_in.finding_types
    assert sign_in.recovery_reason is None
    assert result.step_verdicts[1].passed is True
