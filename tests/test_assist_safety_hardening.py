"""Safety hardening regressions for Stage 3 assist false-pass vectors."""

from pathlib import Path

from reckon.aligner import align
from reckon.assist_apply import apply_assist_proposals
from reckon.assist_propose import _AssistDecision
from reckon.assist_select import select_assist_candidates
from reckon.assist_validate import validate_assist_decision
from reckon.crosscheck import select_crosscheck_targets
from reckon.episode_interaction import (
    REMAP_MISSING_ACTION_FLOOR,
    find_matching_interaction,
    remap_interactions_against_episodes,
    remap_targets_compatible,
    target_similarity,
)
from reckon.episodes import cluster_observed_episodes
from reckon.episode_verify_fast import (
    fast_path_verify_episode,
    is_transient_evidence_text,
)
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
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


def _exp(
    index: int,
    action: ActionType,
    target: str | None = None,
    *,
    assertion: str | None = None,
    input_value: str | None = None,
) -> ExpectedStep:
    return ExpectedStep(
        index=index,
        action=action,
        target=target,
        assertion=assertion,
        input_value=input_value,
        raw_text=f"step {index}",
    )


def _obs(
    index: int,
    action: ActionType,
    target: str | None = None,
    *,
    timestamp_ms: int = 1000,
    input_value: str | None = None,
    visible_text: str | None = None,
    description: str | None = None,
    screen: str | None = None,
    confidence: float = 0.95,
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
        input_value=input_value,
        confidence=confidence,
        screen_description=screen,
        assertion_evidence=evidence,
    )


def test_delayed_synonym_rejected_without_assertion_satisfied():
    """LLM equivalence without assertion_satisfied must keep assertion_not_observed."""
    from reckon.assist_select import AssistCandidate, AssistEvidenceEvent

    expected = ExpectedTrace(
        scenario_title="saved",
        steps=[_exp(1, ActionType.assert_text, "banner", assertion="Settings saved")],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                ActionType.assert_text,
                "banner",
                timestamp_ms=100,
                visible_text="Working",
                description="Working",
            ),
            _obs(
                2,
                ActionType.assert_text,
                "banner",
                timestamp_ms=400,
                visible_text="Preferences updated",
                description="Preferences updated",
            ),
        ],
    )
    candidate = AssistCandidate(
        candidate_id="C1",
        kind=AssistCandidateKind.delayed_assertion,
        expected_step_index=1,
        matched_observed_index=1,
        finding_types=(FindingType.assertion_not_observed,),
        expected_action=ActionType.assert_text.value,
        observed_action=ActionType.assert_text.value,
        expected_text="Settings saved",
        observed_text="Working",
        delayed_events=(
            AssistEvidenceEvent(
                observed_index=2,
                timestamp_ms=400,
                action=ActionType.assert_text.value,
                target="banner",
                visible_text="Preferences updated",
                evidence_description="Preferences updated",
                screen_description=None,
            ),
        ),
        deterministic_support_index=None,
    )

    record = validate_assist_decision(
        candidate,
        _AssistDecision(
            candidate_id="C1",
            answer="equivalent",
            confidence=0.99,
            rationale="synonym",
            supporting_event_id="O2",
        ),
        expected,
        observed,
    )
    assert record.status == AssistProposalStatus.rejected
    assert record.reason == "llm_accepted_without_assertion_evidence_support"

    # Rejected proposal must not clear the baseline critical finding.
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=100,
                description="missing",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=100,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
    )
    refined = apply_assist_proposals(deterministic, [], expected, observed)
    assert refined.step_verdicts[0].passed is False
    assert FindingType.assertion_not_observed in refined.step_verdicts[0].finding_types


def test_delayed_synonym_rejects_polarity_contradiction():
    expected = ExpectedTrace(
        scenario_title="saved",
        steps=[_exp(1, ActionType.assert_text, "banner", assertion="Settings saved")],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                ActionType.assert_text,
                "banner",
                timestamp_ms=10,
                visible_text="Working",
                description="Working",
            ),
            _obs(
                2,
                ActionType.assert_text,
                "banner",
                timestamp_ms=300,
                visible_text="Save failed",
                description="Save failed",
            ),
        ],
    )
    audit = align(expected, observed)
    # Protected contradiction filtering may drop the candidate entirely.
    candidates = select_assist_candidates(expected, observed, audit)
    if not candidates:
        assert any(
            f.type == FindingType.assertion_not_observed for f in audit.findings
        )
        return

    candidate = candidates[0]
    record = validate_assist_decision(
        candidate,
        _AssistDecision(
            candidate_id=candidate.candidate_id,
            answer="equivalent",
            confidence=0.99,
            rationale="ignore failure",
            supporting_event_id="O2",
        ),
        expected,
        observed,
    )
    assert record.status == AssistProposalStatus.rejected
    assert record.reason in {
        "negative_or_contradictory_evidence",
        "supporting_event_not_offered",
        "llm_accepted_without_assertion_evidence_support",
    }


def test_remap_rejects_mismatched_input_value():
    expected = ExpectedTrace(
        scenario_title="amount",
        steps=[
            _exp(1, ActionType.click, "Start"),
            _exp(2, ActionType.type_text, "Amount", input_value="100.00"),
            _exp(3, ActionType.assert_text, "Done", assertion="Submitted"),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(1, ActionType.click, "Start", timestamp_ms=100, screen="Start"),
            _obs(
                2,
                ActionType.type_text,
                "Amount",
                timestamp_ms=400,
                input_value="10.00",
                screen="Form",
            ),
            _obs(
                3,
                ActionType.assert_text,
                "Done",
                timestamp_ms=800,
                visible_text="Submitted",
                screen="Done",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    match = find_matching_interaction(
        expected_action=ActionType.type_text,
        expected_target="Amount",
        expected_input_value="100.00",
        observed=observed,
        episodes=episodes,
        window=(2, 2),
        claimed_obs_indices=set(),
        min_score=60.0,
        require_canonical_nouns=False,
    )
    assert match is None


def test_remap_does_not_clear_missing_action_below_strict_floor():
    expected = ExpectedTrace(
        scenario_title="interest",
        steps=[
            _exp(1, ActionType.click, "Start"),
            _exp(2, ActionType.click, "I'm interested cta"),
            _exp(3, ActionType.assert_text, "Done", assertion="Thanks for your interest"),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(1, ActionType.click, "Start", timestamp_ms=100, screen="Start"),
            _obs(
                2,
                ActionType.click,
                "Learn more link",
                timestamp_ms=400,
                screen="Promo",
            ),
            _obs(
                3,
                ActionType.assert_text,
                "Done",
                timestamp_ms=800,
                visible_text="Thanks for your interest",
                screen="Done",
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.33,
        findings=[
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.critical,
                expected_step_index=2,
                observed_action_index=None,
                evidence_timestamp_ms=None,
                description="missing",
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
                matched_observed_index=None,
                evidence_timestamp_ms=None,
                finding_types=[FindingType.missing_action],
            ),
            StepVerdict(
                expected_step_index=3,
                passed=True,
                matched_observed_index=3,
                evidence_timestamp_ms=800,
                finding_types=[],
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    # Mid-60s fuzzy should not clear missing_action (needs >= 80 + shared nouns).
    from rapidfuzz import fuzz

    score = fuzz.token_sort_ratio("i'm interested cta", "learn more link")
    assert score < REMAP_MISSING_ACTION_FLOOR

    proposals = remap_interactions_against_episodes(
        expected, observed, audit, episodes
    )
    assert proposals == []
    assert FindingType.missing_action in {
        f.type for f in audit.findings
    }


def test_transient_spinner_cannot_satisfy_assertion():
    assert is_transient_evidence_text("Loading spinner")
    assert is_transient_evidence_text("please wait")
    assert not is_transient_evidence_text("You're offline")

    step = _exp(
        1,
        ActionType.assert_state,
        "home",
        assertion="You're offline",
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                ActionType.other,
                "Loading spinner",
                timestamp_ms=100,
                visible_text="Loading spinner",
                description="Loading spinner",
                screen="Loading spinner",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    assert fast_path_verify_episode(step, episodes[0], observed) is None


def test_baseline_align_strictly_identical_to_golden():
    """Frozen login_faithful fixture: baseline align() must stay byte-stable."""
    case = Path(__file__).resolve().parents[1] / "eval" / "aligner" / "login_faithful"
    expected = ExpectedTrace.model_validate_json((case / "expected.json").read_text())
    observed = ObservedTrace.model_validate_json((case / "observed.json").read_text())

    first = align(expected, observed)
    second = align(expected, observed)
    assert first.model_dump() == second.model_dump()

    # click↔select must not be interchangeable on the baseline path.
    from reckon.tolerance import actions_compatible

    assert not actions_compatible(ActionType.click, ActionType.select)


def test_low_confidence_assist_clear_still_queued_for_visual_review():
    expected = ExpectedTrace(
        scenario_title="banner",
        steps=[_exp(1, ActionType.assert_text, "banner", assertion="Refund issued")],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                ActionType.assert_text,
                "banner",
                timestamp_ms=500,
                visible_text="Refund issued",
                confidence=0.95,
            ),
        ],
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=500,
                description="missing",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=500,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
    )
    accepted = [
        AssistProposalRecord(
            kind=AssistCandidateKind.delayed_assertion,
            status=AssistProposalStatus.accepted,
            expected_step_index=1,
            matched_observed_index=1,
            supporting_observed_index=1,
            finding_types=[FindingType.assertion_not_observed],
            confidence=0.70,
            matching_method=AssistMatchMethod.llm_assisted,
        )
    ]
    refined = apply_assist_proposals(deterministic, accepted, expected, observed)
    assert refined.step_verdicts[0].cleared_by_assist is True
    assert refined.step_verdicts[0].assist_confidence == 0.70
    assert refined.step_verdicts[0].passed is True

    targets = select_crosscheck_targets(expected, observed, refined)
    assert any(
        t.expected_step_index == 1 and t.kind.value == "assertion" for t in targets
    )


def test_high_confidence_llm_cannot_bypass_python_evidence_failure():
    """High LLM confidence never populates skip_semantic_pairs without Python evidence."""
    from reckon.assist import _reviewed_pairs_from_proposals
    from reckon.assist_select import AssistCandidate, AssistEvidenceEvent
    from reckon.tolerance import assertion_satisfied

    expected = ExpectedTrace(
        scenario_title="saved",
        steps=[_exp(1, ActionType.assert_text, "banner", assertion="Settings saved")],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                ActionType.assert_text,
                "banner",
                timestamp_ms=100,
                visible_text="Working",
                description="Working",
            ),
            _obs(
                2,
                ActionType.assert_text,
                "banner",
                timestamp_ms=400,
                visible_text="Preferences updated",
                description="Preferences updated",
            ),
        ],
    )
    assert not assertion_satisfied("Settings saved", observed.actions[1])

    candidate = AssistCandidate(
        candidate_id="C1",
        kind=AssistCandidateKind.delayed_assertion,
        expected_step_index=1,
        matched_observed_index=1,
        finding_types=(FindingType.assertion_not_observed,),
        expected_action=ActionType.assert_text.value,
        observed_action=ActionType.assert_text.value,
        expected_text="Settings saved",
        observed_text="Working",
        delayed_events=(
            AssistEvidenceEvent(
                observed_index=2,
                timestamp_ms=400,
                action=ActionType.assert_text.value,
                target="banner",
                visible_text="Preferences updated",
                evidence_description="Preferences updated",
                screen_description=None,
            ),
        ),
    )
    record = validate_assist_decision(
        candidate,
        _AssistDecision(
            candidate_id="C1",
            answer="equivalent",
            confidence=0.99,
            rationale="high confidence synonym",
            supporting_event_id="O2",
        ),
        expected,
        observed,
    )
    assert record.status == AssistProposalStatus.rejected
    assert record.reason == "llm_accepted_without_assertion_evidence_support"
    assert _reviewed_pairs_from_proposals([record]) == []


def test_delayed_toast_rejects_unrelated_later_success_banner():
    """Delayed toast search stays within 10s and requires target-token relevance."""
    from reckon.episode_verify_fast import (
        DELAYED_TOAST_WINDOW_MS,
        candidate_episodes_for_step,
        delayed_toast_target_relevant,
    )
    from reckon.episode_verifier import verify_step_against_episodes

    expected = ExpectedTrace(
        scenario_title="refund toast",
        steps=[
            _exp(1, ActionType.click, "Submit refund"),
            _exp(
                2,
                ActionType.assert_text,
                "Refund toast",
                assertion="Refund issued",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(
                1,
                ActionType.click,
                "Submit refund",
                timestamp_ms=1000,
                screen="Checkout",
            ),
            _obs(
                2,
                ActionType.assert_text,
                "Profile banner",
                timestamp_ms=1000 + DELAYED_TOAST_WINDOW_MS + 5000,
                visible_text="Profile saved successfully",
                description="Unrelated success banner",
                screen="Profile",
            ),
            _obs(
                3,
                ActionType.assert_text,
                "Order banner",
                timestamp_ms=1000 + DELAYED_TOAST_WINDOW_MS + 8000,
                visible_text="Refund issued",
                description="Late refund text outside window",
                screen="Orders",
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.5,
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=2,
                observed_action_index=None,
                evidence_timestamp_ms=None,
                description="missing toast",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
                finding_types=[],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                matched_observed_index=None,
                evidence_timestamp_ms=None,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    candidates = candidate_episodes_for_step(
        expected.steps[1], expected, observed, audit, episodes
    )
    covered = {
        index
        for episode in candidates
        for index in range(episode.start_index, episode.end_index + 1)
    }
    assert 2 not in covered
    assert 3 not in covered
    assert not delayed_toast_target_relevant(expected.steps[1], observed.actions[1])

    record = verify_step_against_episodes(
        expected.steps[1],
        expected,
        observed,
        audit,
        episodes,
        allow_llm=False,
    )
    assert record.status != AssistProposalStatus.accepted


def test_minor_ocr_noise_does_not_split_static_screen():
    """Soft Jaccard band keeps static screens together when no mutator occurs."""
    screen_a = "Driver home map offline sheet view"
    screen_b = "Driver home map offlne sheet panel"
    from reckon.episodes import _jaccard, _tokens

    overlap = _jaccard(_tokens(screen_a), _tokens(screen_b))
    assert 0.45 <= overlap < 0.65

    observed = ObservedTrace(
        source_file="rec.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.assert_visible,
                target="Status",
                confidence=0.9,
                screen_description=screen_a,
                assertion_evidence=AssertionEvidence(
                    description="You're offline",
                    visible_text="You're offline",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=1300,
                action=ActionType.assert_visible,
                target="Status",
                confidence=0.9,
                screen_description=screen_b,
                assertion_evidence=AssertionEvidence(
                    description="You're offline",
                    visible_text="You're offline",
                ),
            ),
        ],
    )

    episodes = cluster_observed_episodes(observed)
    assert len(episodes) == 1
    assert episodes[0].start_index == 1
    assert episodes[0].end_index == 2


def test_select_cannot_remap_to_click_on_standard_button():
    """select↔click remapping requires both sides to be choice controls."""
    expected = ExpectedTrace(
        scenario_title="lang",
        steps=[
            _exp(1, ActionType.click, "Settings"),
            _exp(2, ActionType.select, "Language"),
            _exp(3, ActionType.assert_text, "Saved", assertion="Language saved"),
        ],
    )
    observed = ObservedTrace(
        source_file="lang.mp4",
        actions=[
            _obs(1, ActionType.click, "Settings", timestamp_ms=100, screen="Settings"),
            _obs(
                2,
                ActionType.click,
                "Done button",
                timestamp_ms=400,
                screen="Settings",
            ),
            _obs(
                3,
                ActionType.assert_text,
                "Saved",
                timestamp_ms=800,
                visible_text="Language saved",
                screen="Settings",
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
                matched_observed_index=3,
                evidence_timestamp_ms=800,
                finding_types=[],
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    match = find_matching_interaction(
        expected_action=ActionType.select,
        expected_target="Language",
        expected_input_value=None,
        observed=observed,
        episodes=episodes,
        window=(2, 2),
        claimed_obs_indices={1, 3},
        min_score=60.0,
        require_canonical_nouns=False,
    )
    assert match is None

    proposals = remap_interactions_against_episodes(
        expected, observed, audit, episodes
    )
    assert not any(
        p.expected_step_index == 2 and p.status == AssistProposalStatus.accepted
        for p in proposals
    )


def test_remap_rejects_terms_acceptance_onto_continue_button():
    """E21-style false pass: fuzzy partial_ratio clears 60 via conditions≈continue."""
    expected_target = "Terms and Conditions acceptance"
    observed_target = "Continue button"
    assert target_similarity(expected_target, observed_target) >= 60.0
    assert not remap_targets_compatible(expected_target, observed_target)

    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(41, ActionType.click, "Language", timestamp_ms=4100, screen="Form"),
            _obs(
                44,
                ActionType.click,
                "Continue button",
                timestamp_ms=4400,
                screen="Form",
            ),
            _obs(
                45,
                ActionType.assert_text,
                "Joined",
                timestamp_ms=4500,
                visible_text="Joined",
                screen="Done",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    match = find_matching_interaction(
        expected_action=ActionType.click,
        expected_target=expected_target,
        expected_input_value=None,
        observed=observed,
        episodes=episodes,
        window=(41, 45),
        claimed_obs_indices={41, 45},
        min_score=60.0,
        require_canonical_nouns=False,
    )
    assert match is None


def test_remap_accepts_legitimate_terms_alias():
    assert remap_targets_compatible(
        "Terms and Conditions acceptance",
        "I agree to the Terms checkbox",
    )
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(1, ActionType.click, "Start", timestamp_ms=100, screen="Start"),
            _obs(
                2,
                ActionType.click,
                "I agree to the Terms checkbox",
                timestamp_ms=400,
                screen="Form",
            ),
            _obs(
                3,
                ActionType.assert_text,
                "Done",
                timestamp_ms=800,
                visible_text="Done",
                screen="Done",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    match = find_matching_interaction(
        expected_action=ActionType.click,
        expected_target="Terms and Conditions acceptance",
        expected_input_value=None,
        observed=observed,
        episodes=episodes,
        window=(2, 2),
        claimed_obs_indices={1, 3},
        min_score=60.0,
        require_canonical_nouns=False,
    )
    assert match is not None
    assert match.index == 2


def test_remap_rejects_unrelated_button_inside_valid_temporal_window():
    observed = ObservedTrace(
        source_file="x.mp4",
        actions=[
            _obs(1, ActionType.click, "Start", timestamp_ms=100, screen="Start"),
            _obs(
                2,
                ActionType.click,
                "Learn more link",
                timestamp_ms=400,
                screen="Promo",
            ),
            _obs(
                3,
                ActionType.assert_text,
                "Done",
                timestamp_ms=800,
                visible_text="Done",
                screen="Done",
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    match = find_matching_interaction(
        expected_action=ActionType.click,
        expected_target="Terms and Conditions acceptance",
        expected_input_value=None,
        observed=observed,
        episodes=episodes,
        window=(1, 3),
        claimed_obs_indices={1, 3},
        min_score=60.0,
        require_canonical_nouns=False,
    )
    assert match is None
