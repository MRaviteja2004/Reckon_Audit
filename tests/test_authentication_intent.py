"""Opening an application is not evidence that a login completed."""

from reckon.aligner import align
from reckon.authentication_intent import claims_authentication
from reckon.episode_interaction import remap_interactions_against_episodes
from reckon.episodes import cluster_observed_episodes
from reckon.equivalence.citations import bind_expected_intent
from reckon.equivalence.engine import evaluate
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    ElementKind,
    ExpectedStep,
    ExpectedTrace,
    IntentOperation,
    ObservedAction,
    ObservedTrace,
)
from tests.test_equivalence_engine import _interaction, _obs


def _icon() -> ObservedAction:
    return ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=ActionType.click,
        target="Driver application icon",
        confidence=0.9,
        screen_description="Driver application launching",
    )


def _remap(expected: ExpectedTrace, observed: ObservedTrace):
    return remap_interactions_against_episodes(
        expected,
        observed,
        align(expected, observed),
        cluster_observed_episodes(observed),
    )


def test_claims_authentication_reads_wording_when_the_action_is_other():
    step = ExpectedStep(
        index=1,
        action=ActionType.other,
        target="driver app",
        raw_text="Log in to the driver app.",
    )
    opening = step.model_copy(update={"raw_text": "Open the driver app."})
    assert claims_authentication(step) is True
    assert claims_authentication(opening) is False


def test_app_icon_tap_does_not_pass_a_login_step():
    expected = ExpectedTrace(
        scenario_title="Sign in",
        steps=[
            ExpectedStep(index=1, action=ActionType.other, target="test account"),
            ExpectedStep(
                index=2,
                action=ActionType.other,
                target="driver app",
                raw_text="Log in to the driver app.",
            ),
        ],
    )
    records = _remap(expected, ObservedTrace(source_file="login.mp4", actions=[_icon()]))
    assert not any(record.status.value == "accepted" for record in records)
    assert align(expected, ObservedTrace(source_file="login.mp4", actions=[_icon()])).step_verdicts[1].passed is False


def test_app_icon_tap_supports_an_open_step():
    expected = ExpectedTrace(
        scenario_title="Open",
        steps=[
            ExpectedStep(index=1, action=ActionType.other, target="test account"),
            ExpectedStep(
                index=2,
                action=ActionType.other,
                target="driver app",
                raw_text="Open the driver app.",
            ),
        ],
    )
    records = _remap(expected, ObservedTrace(source_file="open.mp4", actions=[_icon()]))
    assert any(
        record.expected_step_index == 2
        and record.status.value == "accepted"
        and record.supporting_observed_index == 1
        for record in records
    )


def test_login_completion_requires_its_own_evidence():
    expected = ExpectedTrace(
        scenario_title="Signed in",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="driver app",
                raw_text="Log in to the driver app.",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_state,
                target="session",
                assertion="the user is signed in",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="session.mp4",
        actions=[
            _icon(),
            ObservedAction(
                index=2,
                timestamp_ms=2000,
                action=ActionType.assert_visible,
                target="splash screen",
                confidence=0.9,
                screen_description="Application launch splash",
                assertion_evidence=AssertionEvidence(
                    description="Splash",
                    visible_text="Loading",
                ),
            ),
        ],
    )
    verdicts = align(expected, observed).step_verdicts
    assert verdicts[0].passed is False
    assert verdicts[1].passed is False


def test_login_wording_blocks_application_icon_equivalence():
    scenario = "Log in to the Driver application."
    intent = bind_expected_intent(
        scenario,
        sentence_start=0,
        sentence_end=len(scenario),
        operation="traverse",
        operation_text="Log in",
        kind="application",
        kind_text="application",
        name="Driver",
    )
    assert intent is not None
    target = "Driver application icon"
    step = ExpectedStep(
        index=1,
        action=ActionType.other,
        target="Driver application",
        raw_text=scenario,
        action_intent=intent,
    )
    observed = _obs(1, 1000, target, _interaction(
        target,
        operation=IntentOperation.activate,
        kind=ElementKind.icon,
        kind_text="icon",
        name="Driver",
    ))
    assert evaluate(
        step,
        observed,
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    ) == "type_not_realized"
