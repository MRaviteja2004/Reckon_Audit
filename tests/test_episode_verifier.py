"""Unit tests for Phase 3 episode-level state verification."""

from unittest.mock import MagicMock

from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.episodes import cluster_observed_episodes
from reckon.episode_verifier import (
    candidate_episodes_for_step,
    episode_has_negative_invariant,
    fast_path_verify_episode,
    verify_step_against_episodes,
    verify_trace_against_episodes,
)
from reckon.episode_verify_fast import best_proof_action, search_start_observed_index
from reckon.episode_verify_propose import EpisodeVerifyDecision, EpisodeVerifyVerdict
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalStatus,
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


def _obs(
    index: int,
    *,
    timestamp_ms: int,
    action: ActionType,
    target: str,
    screen: str,
    visible_text: str | None = None,
    description: str | None = None,
) -> ObservedAction:
    evidence = None
    if visible_text is not None or description is not None:
        evidence = AssertionEvidence(
            description=description or visible_text or "",
            visible_text=visible_text,
        )
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=action,
        target=target,
        confidence=0.9,
        screen_description=screen,
        assertion_evidence=evidence,
    )


def _coarse_login_observed() -> ObservedTrace:
    """10 micro-actions: form fill → click → spinner → home with offline banner."""
    return ObservedTrace(
        source_file="coarse_login.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.click,
                target="App icon",
                screen="Launcher",
            ),
            _obs(
                2,
                timestamp_ms=400,
                action=ActionType.navigate,
                target="Splash",
                screen="Splash screen",
            ),
            _obs(
                3,
                timestamp_ms=800,
                action=ActionType.type_text,
                target="Email field",
                screen="Login form",
            ),
            _obs(
                4,
                timestamp_ms=1100,
                action=ActionType.type_text,
                target="Password field",
                screen="Login form",
            ),
            _obs(
                5,
                timestamp_ms=1400,
                action=ActionType.click,
                target="Sign in",
                screen="Login form",
            ),
            _obs(
                6,
                timestamp_ms=1700,
                action=ActionType.other,
                target="Loading spinner",
                screen="Loading spinner",
            ),
            _obs(
                7,
                timestamp_ms=2000,
                action=ActionType.other,
                target="Loading spinner",
                screen="Loading spinner",
            ),
            _obs(
                8,
                timestamp_ms=4500,
                action=ActionType.navigate,
                target="Home map",
                screen="Driver home map",
            ),
            _obs(
                9,
                timestamp_ms=4800,
                action=ActionType.assert_visible,
                target="Offline banner",
                screen="Driver home map with offline sheet",
                visible_text="You're offline",
                description="Home shows offline status after login",
            ),
            _obs(
                10,
                timestamp_ms=5100,
                action=ActionType.other,
                target="Home chrome",
                screen="Driver home map with offline sheet",
            ),
        ],
    )


def test_coarse_login_remaps_to_home_screen_and_clears_critical_findings():
    expected = ExpectedTrace(
        scenario_title="Coarse login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="app",
                assertion="You're offline",
                raw_text="Log into the app",
            ),
        ],
    )
    observed = _coarse_login_observed()
    baseline = align(expected, observed)
    assert baseline.step_verdicts[0].passed is True
    assert baseline.step_verdicts[0].matched_observed_index == 9
    assert baseline.step_verdicts[0].evidence_id == "O9"

    client = MagicMock()
    client.chat.completions.create.side_effect = AssertionError(
        "LLM must not be called on deterministic episode fast-path"
    )
    assisted = align_with_assist(expected, observed, openai_client=client)

    assert assisted.deterministic_result.model_dump() == baseline.model_dump()
    step = assisted.result.step_verdicts[0]
    assert step.passed is True
    assert step.matched_observed_index == 9
    assert step.evidence_timestamp_ms == 4800
    assert step.finding_types == []
    assert assisted.result.verdict == AuditVerdict.faithful
    assert not any(
        finding.severity == FindingSeverity.critical
        for finding in assisted.result.findings
    )
    accepted = [
        proposal
        for proposal in assisted.assist.proposals
        if proposal.kind == AssistCandidateKind.episode_verification
        and proposal.status == AssistProposalStatus.accepted
    ]
    assert accepted == []
    assert step.recovery_reason == "later_observed_action"
    assert step.evidence_id == "O9"
    client.chat.completions.create.assert_not_called()


def test_distractor_settings_screen_remains_unfaithful():
    expected = ExpectedTrace(
        scenario_title="Login required",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_state,
                target="app",
                assertion="You're offline",
                raw_text="User should be logged in successfully",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="distractor.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.click,
                target="App icon",
                screen="Launcher",
            ),
            _obs(
                2,
                timestamp_ms=500,
                action=ActionType.other,
                target="Loading spinner",
                screen="Loading spinner",
            ),
            _obs(
                3,
                timestamp_ms=3000,
                action=ActionType.navigate,
                target="Studio Dev Settings",
                screen="Studio Dev Settings with Launch Deeplink field focused",
            ),
        ],
    )
    baseline = align(expected, observed)
    episodes = cluster_observed_episodes(observed)
    record = verify_step_against_episodes(
        expected.steps[0],
        expected,
        observed,
        baseline,
        episodes,
        allow_llm=False,
    )
    assert record.status != AssistProposalStatus.accepted
    assert record.reason in {
        "insufficient_evidence",
        "no_candidate_episodes",
        "negative_invariant",
    }

    client = MagicMock()
    client.chat.completions.create.return_value = EpisodeVerifyDecision(
        verdict=EpisodeVerifyVerdict.insufficient_evidence,
        best_action_index=None,
        evidence_quote=None,
        confidence=0.9,
        rationale="settings screen does not prove offline home",
    )
    assisted = align_with_assist(expected, observed, openai_client=client)
    assert assisted.result.step_verdicts[0].passed is False
    assert assisted.result.verdict == AuditVerdict.unfaithful
    assert {
        FindingType.assertion_not_observed,
        FindingType.missing_action,
    } & set(assisted.result.step_verdicts[0].finding_types)


def test_negative_invariant_rejects_error_alert():
    expected = ExpectedTrace(
        scenario_title="Successful login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_text,
                target="status",
                assertion="Login successful",
                raw_text="Assert login successful",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="neg.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.click,
                target="Sign in",
                screen="Login form",
            ),
            _obs(
                2,
                timestamp_ms=800,
                action=ActionType.assert_visible,
                target="Error alert",
                screen="Login form with error dialog",
                visible_text="Invalid credentials",
                description="Error dialog: Invalid credentials",
            ),
        ],
    )
    baseline = align(expected, observed)
    episodes = cluster_observed_episodes(observed)
    assert any(
        episode_has_negative_invariant(expected.steps[0], episode, observed)
        for episode in episodes
    )
    record = verify_step_against_episodes(
        expected.steps[0],
        expected,
        observed,
        baseline,
        episodes,
        allow_llm=False,
    )
    assert record.status == AssistProposalStatus.rejected
    assert record.reason == "negative_invariant"

    assisted = align_with_assist(
        expected,
        observed,
        openai_client=MagicMock(),
    )
    assert assisted.result.step_verdicts[0].passed is False
    assert assisted.result.verdict == AuditVerdict.unfaithful


def test_fast_path_finds_terminal_proof_inside_home_episode():
    expected = ExpectedTrace(
        scenario_title="t",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.other,
                target="app",
                assertion="You're offline",
                raw_text="Log into the app",
            ),
        ],
    )
    observed = _coarse_login_observed()
    audit = align(expected, observed)
    episodes = cluster_observed_episodes(observed)
    home = next(
        episode
        for episode in episodes
        if any("offline" in text.lower() for text in episode.visible_texts)
    )
    proof = fast_path_verify_episode(
        expected.steps[0], home, observed, expected=expected
    )
    assert proof is not None
    assert proof.index == 9

    proposals, llm_requests = verify_trace_against_episodes(
        expected,
        observed,
        audit,
        episodes,
        allow_llm=False,
    )
    assert llm_requests == 0
    assert proposals == []


def test_llm_verified_rejects_invented_evidence_quote():
    expected_hard = ExpectedTrace(
        scenario_title="t",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_state,
                target="home",
                assertion="session ready for dispatch",
                raw_text="Assert session ready",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.assert_state,
                target="Home",
                screen="Home dashboard",
                visible_text="Welcome back",
                description="Home greeting",
            ),
        ],
    )
    baseline_hard = align(expected_hard, observed)
    assert baseline_hard.step_verdicts[0].passed is False
    episodes = cluster_observed_episodes(observed)
    client = MagicMock()
    client.chat.completions.create.return_value = EpisodeVerifyDecision(
        verdict=EpisodeVerifyVerdict.verified,
        best_action_index=1,
        evidence_quote="session ready for dispatch",
        confidence=0.95,
        rationale="paraphrased claim not present on screen",
    )
    record = verify_step_against_episodes(
        expected_hard.steps[0],
        expected_hard,
        observed,
        baseline_hard,
        episodes,
        openai_client=client,
        allow_llm=True,
    )
    # Invented quote (not a substring of episode text) → reject.
    assert record.status == AssistProposalStatus.rejected
    assert record.reason == "ungrounded_verification"


def test_llm_verified_accepts_semantic_quote_from_visible_text():
    """Regression: login claim verified via 'You're offline' must not be ungrounded."""
    expected = ExpectedTrace(
        scenario_title="Login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_state,
                target="driver app",
                assertion="User should able to log in succesfully",
                raw_text="User should able to log in succesfully",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="login.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=1000,
                action=ActionType.other,
                target="Loading spinner",
                screen="Loading spinner",
            ),
            _obs(
                26,
                timestamp_ms=8000,
                action=ActionType.assert_visible,
                target="Offline banner",
                screen="Driver home map",
                visible_text="You're offline",
                description="Home shows offline status",
            ),
        ],
    )
    baseline = align(expected, observed)
    assert baseline.step_verdicts[0].passed is False
    episodes = cluster_observed_episodes(observed)
    client = MagicMock()
    client.chat.completions.create.return_value = EpisodeVerifyDecision(
        verdict=EpisodeVerifyVerdict.verified,
        best_action_index=26,
        evidence_quote="You're offline",
        confidence=0.85,
        rationale="Offline home banner proves the login session landed",
    )
    record = verify_step_against_episodes(
        expected.steps[0],
        expected,
        observed,
        baseline,
        episodes,
        openai_client=client,
        allow_llm=True,
    )
    assert record.status == AssistProposalStatus.accepted
    assert record.reason == "episode_verified"
    assert record.matched_observed_index == 26
    assert record.supporting_observed_index == 26
    assert record.confidence == 0.85


def test_disabled_button_is_not_a_fatal_negative_invariant():
    expected = ExpectedTrace(
        scenario_title="Join programme",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_text,
                target="confirmation",
                assertion="You've now joined the Uber mentorship programme",
                raw_text="Assert joined confirmation",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="form.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.assert_visible,
                target="Continue button",
                screen="Enrollment form with Continue disabled and greyed out",
                visible_text="Continue",
                description="Continue button is disabled until terms are checked",
            ),
            _obs(
                2,
                timestamp_ms=5000,
                action=ActionType.assert_visible,
                target="Success banner",
                screen="Mentorship confirmation",
                visible_text="You've now joined the Uber mentorship programme",
                description="Joined confirmation toast",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    assert len(episodes) >= 2
    form_episode = episodes[0]
    assert "disabled" in form_episode.screen_summary.lower()
    assert episode_has_negative_invariant(expected.steps[0], form_episode, observed) is False

    baseline = align(expected, observed)
    record = verify_step_against_episodes(
        expected.steps[0],
        expected,
        observed,
        baseline,
        episodes,
        allow_llm=False,
    )
    assert record.status == AssistProposalStatus.accepted
    assert record.matched_observed_index == 2


def test_mentorship_title_fast_path_rejects_join_mentorship_button():
    """E29: episode verify must not clear Mentorship title via Join Mentorship."""
    expected = ExpectedTrace(
        scenario_title="Mentorship dashboard",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_text,
                target="Mentorship title",
                assertion="Mentorship",
                raw_text="Mentorship title",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="mentorship.mp4",
        actions=[
            _obs(
                58,
                timestamp_ms=189513,
                action=ActionType.assert_state,
                target="Join Mentorship button",
                screen="Choose a language screen with Join Mentorship enabled",
                visible_text="Join Mentorship",
                description="Join Mentorship button becomes enabled after agreeing to terms",
            ),
        ],
    )
    baseline = align(expected, observed)
    episodes = cluster_observed_episodes(observed)
    record = verify_step_against_episodes(
        expected.steps[0],
        expected,
        observed,
        baseline,
        episodes,
        allow_llm=False,
    )
    assert record.status != AssistProposalStatus.accepted


def test_dashboard_cta_and_menu_option_fast_path_reject_carousel_page():
    """E32/E39: carousel page must not clear dashboard CTA or menu option."""
    carousel = _obs(
        64,
        timestamp_ms=213765,
        action=ActionType.assert_visible,
        target="How it works page",
        screen="How it works onboarding carousel page 1 of 6",
        visible_text="How it works ... 1 of 6",
        description="Onboarding page title and pager indicator",
    )
    observed = ObservedTrace(source_file="mentorship.mp4", actions=[carousel])
    episodes = cluster_observed_episodes(observed)

    for target in ("How it works CTA", "How it works option"):
        expected = ExpectedTrace(
            scenario_title="Mentorship dashboard",
            steps=[
                ExpectedStep(
                    index=1,
                    action=ActionType.assert_visible,
                    target=target,
                    assertion="How it works",
                    raw_text=target,
                ),
            ],
        )
        baseline = align(expected, observed)
        record = verify_step_against_episodes(
            expected.steps[0],
            expected,
            observed,
            baseline,
            episodes,
            allow_llm=False,
        )
        assert record.status != AssistProposalStatus.accepted, target


def test_menu_option_fast_path_accepts_action_sheet():
    """Legitimate How it works option evidence from the action sheet still passes."""
    expected = ExpectedTrace(
        scenario_title="Mentorship dashboard",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_visible,
                target="How it works option",
                assertion="How it works",
                raw_text="How it works",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="mentorship.mp4",
        actions=[
            _obs(
                79,
                timestamp_ms=283753,
                action=ActionType.assert_visible,
                target="Action sheet menu",
                screen="Mentorship page dimmed with bottom action sheet",
                visible_text="Mentorship settings, How it works, Past learners, Cancel",
                description="Bottom sheet with options shown",
            ),
        ],
    )
    baseline = align(expected, observed)
    episodes = cluster_observed_episodes(observed)
    record = verify_step_against_episodes(
        expected.steps[0],
        expected,
        observed,
        baseline,
        episodes,
        allow_llm=False,
    )
    assert record.status == AssistProposalStatus.accepted
    assert record.matched_observed_index == 79


def test_assertion_search_is_not_capped_by_next_step_dp_match():
    """Evidence later in the trace must stay in the candidate window."""
    expected = ExpectedTrace(
        scenario_title="How it works",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Start",
                raw_text="Click Start",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_visible,
                target="How it works",
                assertion="How it works",
                raw_text="Assert How it works is visible",
            ),
            ExpectedStep(
                index=3,
                action=ActionType.click,
                target="Continue",
                raw_text="Click Continue",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="how.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.click,
                target="Start",
                screen="Start screen",
            ),
            _obs(
                2,
                timestamp_ms=500,
                action=ActionType.click,
                target="Unrelated settings",
                screen="Settings distractor",
            ),
            _obs(
                3,
                timestamp_ms=4000,
                action=ActionType.assert_visible,
                target="How it works heading",
                screen="How it works page",
                visible_text="How it works",
                description="How it works section visible",
            ),
            _obs(
                4,
                timestamp_ms=5000,
                action=ActionType.click,
                target="Continue",
                screen="How it works page",
            ),
        ],
    )
    # Craft an audit where DP misaligned step 2 to O2 and step 3 also to O2,
    # which previously capped the assertion window before O3.
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.33,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=2,
                observed_action_index=2,
                evidence_timestamp_ms=500,
                description="misaligned",
            ),
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=2,
                observed_action_index=2,
                evidence_timestamp_ms=500,
                description="missing proof",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=100,
                finding_types=[],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                matched_observed_index=2,
                evidence_timestamp_ms=500,
                finding_types=[
                    FindingType.wrong_target,
                    FindingType.assertion_not_observed,
                ],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=True,
                matched_observed_index=2,
                evidence_timestamp_ms=500,
                finding_types=[],
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    assert search_start_observed_index(
        expected.steps[1], expected, audit, episodes
    ) == 1
    candidates = candidate_episodes_for_step(
        expected.steps[1], expected, observed, audit, episodes
    )
    covered = {
        index
        for episode in candidates
        for index in range(episode.start_index, episode.end_index + 1)
    }
    assert 3 in covered

    record = verify_step_against_episodes(
        expected.steps[1],
        expected,
        observed,
        audit,
        episodes,
        allow_llm=False,
    )
    assert record.status == AssistProposalStatus.accepted
    assert record.matched_observed_index == 3


def test_assertion_prefers_passive_observation_over_click():
    """assert_visible must not consume a click when a passive proof exists."""
    step = ExpectedStep(
        index=1,
        action=ActionType.assert_visible,
        target="X button",
        assertion="X button",
        raw_text="Assert X button visible",
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.assert_visible,
                target="X button",
                screen="Dialog open",
                visible_text="X button",
                description="Close control visible",
            ),
            _obs(
                2,
                timestamp_ms=200,
                action=ActionType.click,
                target="Close (X) button",
                screen="Dialog open",
                visible_text="X button",
                description="User clicks close",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    proof = best_proof_action(step, episodes[0], observed)
    assert proof is not None
    assert proof.index == 1
    assert proof.action == ActionType.assert_visible


def test_assertion_may_use_click_when_no_passive_proof_exists():
    step = ExpectedStep(
        index=1,
        action=ActionType.assert_visible,
        target="X button",
        assertion="X button",
        raw_text="Assert X button visible",
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.click,
                target="Close (X) button",
                screen="Dialog",
                visible_text="X button",
                description="Close control visible on click",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    proof = best_proof_action(step, episodes[0], observed)
    assert proof is not None
    assert proof.index == 1
    assert proof.action == ActionType.click


def test_interaction_remap_recovers_click_inside_verified_window():
    """wrong_target click remaps to a later semantic match between verified steps."""
    expected = ExpectedTrace(
        scenario_title="interest",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Start",
                raw_text="Click Start",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.click,
                target="I'm interested",
                raw_text="Click I'm interested",
            ),
            ExpectedStep(
                index=3,
                action=ActionType.assert_visible,
                target="Confirmation",
                assertion="Thanks for your interest",
                raw_text="Assert confirmation",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="interest.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.click,
                target="Start",
                screen="Start",
            ),
            _obs(
                2,
                timestamp_ms=400,
                action=ActionType.click,
                target="Unrelated banner",
                screen="Noise",
            ),
            _obs(
                3,
                timestamp_ms=800,
                action=ActionType.click,
                target="I'm interested",
                screen="Interest form",
            ),
            _obs(
                4,
                timestamp_ms=1200,
                action=ActionType.assert_visible,
                target="Confirmation",
                screen="Done",
                visible_text="Thanks for your interest",
                description="Confirmation banner",
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.33,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=2,
                observed_action_index=2,
                evidence_timestamp_ms=400,
                description="DP paired with noise",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=100,
                finding_types=[],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                matched_observed_index=2,
                evidence_timestamp_ms=400,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=True,
                matched_observed_index=4,
                evidence_timestamp_ms=1200,
                finding_types=[],
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    proposals, _ = verify_trace_against_episodes(
        expected, observed, audit, episodes, allow_llm=False
    )
    interaction = next(
        (
            p
            for p in proposals
            if p.expected_step_index == 2
            and p.status == AssistProposalStatus.accepted
        ),
        None,
    )
    assert interaction is not None
    assert interaction.matched_observed_index == 3
    assert interaction.reason == "interaction_temporal_remap"


def test_select_remaps_to_click_on_checkbox_option():
    """select on field label remaps to click on a choice-control option."""
    expected = ExpectedTrace(
        scenario_title="lang",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Settings",
                raw_text="Open settings",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.select,
                target="Language dropdown",
                raw_text="Select Language dropdown",
            ),
            ExpectedStep(
                index=3,
                action=ActionType.assert_visible,
                target="Saved",
                assertion="Language saved",
                raw_text="Assert saved",
            ),
        ],
    )

    observed = ObservedTrace(
        source_file="lang.mp4",
        actions=[
            _obs(
                1,
                timestamp_ms=100,
                action=ActionType.click,
                target="Settings",
                screen="Settings",
            ),
            _obs(
                2,
                timestamp_ms=400,
                action=ActionType.click,
                target="Unrelated toggle",
                screen="Settings",
            ),
            _obs(
                3,
                timestamp_ms=800,
                action=ActionType.click,
                target="English checkbox",
                screen="Language dropdown picker",
            ),
            _obs(
                4,
                timestamp_ms=1200,
                action=ActionType.assert_visible,
                target="Saved",
                screen="Settings",
                visible_text="Language saved",
                description="Saved toast",
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.33,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=2,
                observed_action_index=2,
                evidence_timestamp_ms=400,
                description="DP drift",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=100,
                finding_types=[],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                matched_observed_index=2,
                evidence_timestamp_ms=400,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=True,
                matched_observed_index=4,
                evidence_timestamp_ms=1200,
                finding_types=[],
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    proposals, _ = verify_trace_against_episodes(
        expected, observed, audit, episodes, allow_llm=False
    )
    interaction = next(
        (
            p
            for p in proposals
            if p.expected_step_index == 2
            and p.status == AssistProposalStatus.accepted
        ),
        None,
    )
    assert interaction is not None
    assert interaction.matched_observed_index == 3


def test_interaction_remap_ignores_dp_match_outside_trusted_bracket():
    """Drifted DP indices outside (prev, next) must not skip in-bracket search.

    Reproduces Steps 16/17/19: DP mapped interactions past the next verified
    assertion, while the true clicks sit inside the chronological bracket.
    """
    expected = ExpectedTrace(
        scenario_title="mentorship form",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_visible,
                target="Form",
                assertion="Interest form",
                raw_text="Assert form",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.click,
                target="terms and conditions checkbox",
                raw_text="Click terms",
            ),
            ExpectedStep(
                index=3,
                action=ActionType.click,
                target="join mentorship/Done cta",
                raw_text="Click join",
            ),
            ExpectedStep(
                index=4,
                action=ActionType.assert_visible,
                target="Success",
                assertion="You're in",
                raw_text="Assert success",
            ),
            ExpectedStep(
                index=5,
                action=ActionType.click,
                target="continue",
                raw_text="Click continue",
            ),
            ExpectedStep(
                index=6,
                action=ActionType.assert_visible,
                target="Home",
                assertion="Home screen",
                raw_text="Assert home",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="mentorship.mp4",
        actions=[
            _obs(
                10,
                timestamp_ms=1000,
                action=ActionType.assert_visible,
                target="Form",
                screen="Form",
                visible_text="Interest form",
                description="Form visible",
            ),
            _obs(
                11,
                timestamp_ms=1100,
                action=ActionType.click,
                target="Unrelated chip",
                screen="Form",
            ),
            _obs(
                13,
                timestamp_ms=1300,
                action=ActionType.click,
                target="I agree to the Terms and Conditions checkbox",
                screen="Form",
            ),
            _obs(
                15,
                timestamp_ms=1500,
                action=ActionType.click,
                target="Join Mentorship button",
                screen="Form",
            ),
            _obs(
                16,
                timestamp_ms=1600,
                action=ActionType.assert_visible,
                target="Success",
                screen="Success",
                visible_text="You're in",
                description="Success banner",
            ),
            _obs(
                18,
                timestamp_ms=1800,
                action=ActionType.click,
                target="Continue button",
                screen="Success",
            ),
            _obs(
                20,
                timestamp_ms=2000,
                action=ActionType.assert_visible,
                target="Home",
                screen="Home",
                visible_text="Home screen",
                description="Home",
            ),
        ],
    )
    # DP cascaded every interaction past the next assertion (O16 / O20).
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.2,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=2,
                observed_action_index=18,
                evidence_timestamp_ms=1800,
                description="DP outside bracket",
            ),
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=3,
                observed_action_index=18,
                evidence_timestamp_ms=1800,
                description="DP outside bracket",
            ),
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=5,
                observed_action_index=20,
                evidence_timestamp_ms=2000,
                description="DP outside bracket",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=10,
                evidence_timestamp_ms=1000,
                finding_types=[],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                matched_observed_index=18,
                evidence_timestamp_ms=1800,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=False,
                matched_observed_index=18,
                evidence_timestamp_ms=1800,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=4,
                passed=True,
                matched_observed_index=16,
                evidence_timestamp_ms=1600,
                finding_types=[],
            ),
            StepVerdict(
                expected_step_index=5,
                passed=False,
                matched_observed_index=20,
                evidence_timestamp_ms=2000,
                finding_types=[FindingType.wrong_target],
            ),
            StepVerdict(
                expected_step_index=6,
                passed=True,
                matched_observed_index=20,
                evidence_timestamp_ms=2000,
                finding_types=[],
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    proposals, _ = verify_trace_against_episodes(
        expected, observed, audit, episodes, allow_llm=False
    )
    by_step = {
        p.expected_step_index: p
        for p in proposals
        if p.status == AssistProposalStatus.accepted
        and p.reason == "interaction_temporal_remap"
    }
    assert by_step[2].matched_observed_index == 13
    assert by_step[3].matched_observed_index == 15
    assert by_step[5].matched_observed_index == 18
