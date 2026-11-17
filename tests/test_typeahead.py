"""Regression tests for typeahead select recovery."""

from reckon.aligner import align
from reckon.episodes import cluster_observed_episodes
from reckon.episode_interaction import remap_interactions_against_episodes
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
from reckon.typeahead import (
    TYPEAHEAD_MAX_SPAN_MS,
    find_typeahead_selection_proof,
)
from reckon.tolerance import actions_compatible


def _select_step() -> ExpectedStep:
    return ExpectedStep(
        index=1,
        action=ActionType.select,
        target="Location",
        input_value="Austin",
        raw_text="Set a Location to Austin.",
    )


def _austin_sequence(
    *,
    typed: str = "austin",
    suggestion_click: str = "Austin, TX, USA suggestion",
    confirm_text: str = "Austin, TX, USA",
    click_ts: int = 27500,
) -> list[ObservedAction]:
    return [
        ObservedAction(
            index=1,
            timestamp_ms=25500,
            action=ActionType.type_text,
            target="Where would you like to earn? field",
            input_value=typed,
            confidence=0.9,
        ),
        ObservedAction(
            index=2,
            timestamp_ms=27000,
            action=ActionType.assert_visible,
            target="Location suggestions dropdown",
            confidence=0.85,
            assertion_evidence=AssertionEvidence(
                description="Suggestions list appears",
                visible_text="Austin, TX, USA",
            ),
        ),
        ObservedAction(
            index=3,
            timestamp_ms=click_ts,
            action=ActionType.click,
            target=suggestion_click,
            confidence=0.85,
        ),
        ObservedAction(
            index=4,
            timestamp_ms=click_ts + 1000,
            action=ActionType.assert_text,
            target="Where would you like to earn? field",
            confidence=0.85,
            assertion_evidence=AssertionEvidence(
                description="Location field filled with chosen city",
                visible_text=confirm_text,
            ),
        ),
    ]


def test_typeahead_valid_sequence_recovers_select():
    step = _select_step()
    observed = ObservedTrace(source_file="austin.mp4", actions=_austin_sequence())
    proof = find_typeahead_selection_proof(
        step,
        observed,
        window=(1, 4),
        claimed_obs_indices=set(),
    )
    assert proof is not None
    assert proof.type_index == 1
    assert proof.click_index == 3
    assert proof.confirm_index == 4
    assert proof.type_index in proof.claimed_indices
    assert proof.click_index in proof.claimed_indices
    assert proof.confirm_index in proof.claimed_indices
    # Suggestion-list observation is optional supporting evidence when present.
    assert proof.suggestion_index in {2, None}


def test_typeahead_typing_without_suggestion_click_does_not_pass():
    step = _select_step()
    actions = _austin_sequence()[:2]  # type + suggestion list only
    observed = ObservedTrace(source_file="austin.mp4", actions=actions)
    proof = find_typeahead_selection_proof(
        step,
        observed,
        window=(1, 2),
        claimed_obs_indices=set(),
    )
    assert proof is None


def test_typeahead_wrong_suggestion_value_does_not_pass():
    step = _select_step()
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=_austin_sequence(
            suggestion_click="Houston, TX, USA suggestion",
            confirm_text="Houston, TX, USA",
        ),
    )
    proof = find_typeahead_selection_proof(
        step,
        observed,
        window=(1, 4),
        claimed_obs_indices=set(),
    )
    assert proof is None


def test_typeahead_wrong_target_field_does_not_pass():
    step = _select_step()
    actions = _austin_sequence()
    actions[0] = actions[0].model_copy(
        update={"target": "Referral code field", "input_value": "Austin"}
    )
    observed = ObservedTrace(source_file="austin.mp4", actions=actions)
    proof = find_typeahead_selection_proof(
        step,
        observed,
        window=(1, 4),
        claimed_obs_indices=set(),
    )
    assert proof is None


def test_typeahead_out_of_order_does_not_pass():
    step = _select_step()
    # Click before type — indices still ascending but chronologically swapped via
    # reconstructing with click at lower index than type.
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=27000,
                action=ActionType.click,
                target="Austin, TX, USA suggestion",
                confidence=0.85,
            ),
            ObservedAction(
                index=2,
                timestamp_ms=28000,
                action=ActionType.type_text,
                target="Where would you like to earn? field",
                input_value="austin",
                confidence=0.9,
            ),
        ],
    )
    proof = find_typeahead_selection_proof(
        step,
        observed,
        window=(1, 2),
        claimed_obs_indices=set(),
    )
    assert proof is None


def test_typeahead_excessively_delayed_click_does_not_pass():
    step = _select_step()
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=_austin_sequence(click_ts=25500 + TYPEAHEAD_MAX_SPAN_MS + 1),
    )
    proof = find_typeahead_selection_proof(
        step,
        observed,
        window=(1, 4),
        claimed_obs_indices=set(),
    )
    assert proof is None


def test_typeahead_claimed_indices_not_reused_for_second_select():
    step_a = _select_step()
    step_b = ExpectedStep(
        index=2,
        action=ActionType.select,
        target="Location",
        input_value="Austin",
    )
    observed = ObservedTrace(source_file="austin.mp4", actions=_austin_sequence())
    first = find_typeahead_selection_proof(
        step_a,
        observed,
        window=(1, 4),
        claimed_obs_indices=set(),
    )
    assert first is not None
    second = find_typeahead_selection_proof(
        step_b,
        observed,
        window=(1, 4),
        claimed_obs_indices=set(first.claimed_indices),
    )
    assert second is None


def test_typeahead_does_not_change_global_action_compatibility():
    assert not actions_compatible(ActionType.select, ActionType.type_text)
    assert not actions_compatible(ActionType.select, ActionType.click)
    assert not actions_compatible(
        ActionType.select,
        ActionType.click,
        allow_choice_control=True,
        expected_target="Location",
        observed_target="Austin, TX, USA suggestion",
    )


def test_typeahead_align_recovers_austin_style_select_in_baseline_audit():
    expected = ExpectedTrace(
        scenario_title="Austin location",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.select,
                target="Location",
                input_value="Austin",
                raw_text="Set Location to Austin",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.type_text,
                target="Referral code",
                input_value="1",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=_austin_sequence()
        + [
            ObservedAction(
                index=5,
                timestamp_ms=31500,
                action=ActionType.type_text,
                target="Referral code field",
                input_value="1",
                confidence=0.9,
            ),
        ],
    )
    baseline = align(expected, observed)
    select_verdict = baseline.step_verdicts[0]
    assert select_verdict.passed is True
    assert select_verdict.finding_types == []
    assert select_verdict.matched_observed_index == 4
    assert select_verdict.evidence_timestamp_ms == 28500
    assert select_verdict.recovery_reason == "typeahead_selection_recovery"
    assert select_verdict.cleared_by_assist is False
    assert not any(
        f.type == FindingType.missing_action and f.expected_step_index == 1
        for f in baseline.findings
    )
    assert baseline.step_verdicts[1].passed is True


def test_typeahead_align_typing_without_click_remains_missing():
    expected = ExpectedTrace(
        scenario_title="Typing only",
        steps=[_select_step()],
    )
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=_austin_sequence()[:2],
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert FindingType.missing_action in result.step_verdicts[0].finding_types
    assert result.step_verdicts[0].recovery_reason is None


def test_typeahead_align_wrong_suggestion_remains_missing():
    expected = ExpectedTrace(
        scenario_title="Wrong city",
        steps=[_select_step()],
    )
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=_austin_sequence(
            suggestion_click="Houston, TX, USA suggestion",
            confirm_text="Houston, TX, USA",
        ),
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert FindingType.missing_action in result.step_verdicts[0].finding_types


def test_typeahead_align_wrong_field_remains_missing():
    expected = ExpectedTrace(
        scenario_title="Wrong field",
        steps=[_select_step()],
    )
    actions = _austin_sequence()
    actions[0] = actions[0].model_copy(
        update={"target": "Referral code field", "input_value": "Austin"}
    )
    observed = ObservedTrace(source_file="austin.mp4", actions=actions)
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert FindingType.missing_action in result.step_verdicts[0].finding_types


def test_typeahead_align_preselected_earning_option_remains_missing():
    """Visible earning page without an explicit option selection is not enough."""
    expected = ExpectedTrace(
        scenario_title="Earning mode",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.select,
                target="Rides and Delivery with car",
                input_value="Rides and Delivery with car",
                raw_text='Choose "Rides and Delivery with car" and tap Continue.',
            ),
            ExpectedStep(
                index=2,
                action=ActionType.click,
                target="Continue",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=44000,
                action=ActionType.assert_visible,
                target="Choose how you want to earn page",
                confidence=0.9,
                assertion_evidence=AssertionEvidence(
                    description="Earning options",
                    visible_text="Rides and Delivery with car",
                ),
            ),
            ObservedAction(
                index=2,
                timestamp_ms=44766,
                action=ActionType.click,
                target="Continue button",
                confidence=0.9,
            ),
        ],
    )
    result = align(expected, observed)
    assert result.step_verdicts[0].passed is False
    assert FindingType.missing_action in result.step_verdicts[0].finding_types
    assert result.step_verdicts[0].recovery_reason is None


def test_typeahead_remap_still_emits_proposal_when_baseline_not_yet_recovered():
    """Remap path remains available for assist pipelines using a raw audit."""
    expected = ExpectedTrace(
        scenario_title="Austin location",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.select,
                target="Location",
                input_value="Austin",
                raw_text="Set Location to Austin",
            ),
            ExpectedStep(
                index=2,
                action=ActionType.type_text,
                target="Referral code",
                input_value="1",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="austin.mp4",
        actions=_austin_sequence()
        + [
            ObservedAction(
                index=5,
                timestamp_ms=31500,
                action=ActionType.type_text,
                target="Referral code field",
                input_value="1",
                confidence=0.9,
            ),
        ],
    )
    episodes = cluster_observed_episodes(observed)
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.advisory,
                expected_step_index=1,
                description="select missing",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=None,
                finding_types=[FindingType.missing_action],
            ),
            StepVerdict(
                expected_step_index=2,
                passed=True,
                matched_observed_index=5,
                evidence_timestamp_ms=31500,
                finding_types=[],
            ),
        ],
    )
    proposals = remap_interactions_against_episodes(
        expected, observed, audit, episodes
    )
    assert len(proposals) == 1
    assert proposals[0].reason == "typeahead_selection_recovery"
    assert proposals[0].matched_observed_index == 4
