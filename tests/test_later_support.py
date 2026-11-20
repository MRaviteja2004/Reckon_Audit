"""A failed alignment pair must not hide the one observation that supports the step."""

from reckon.aligner import align
from reckon.assist_select import select_assist_candidates
from reckon.later_support import recover_later_support
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditResult,
    AuditVerdict,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)


def _step(
    index: int,
    target: str,
    *,
    action: ActionType = ActionType.click,
    assertion: str | None = None,
    raw_text: str | None = None,
) -> ExpectedStep:
    return ExpectedStep(
        index=index,
        action=action,
        target=target,
        assertion=assertion,
        raw_text=raw_text,
    )


def _click(
    index: int,
    target: str,
    timestamp_ms: int,
    *,
    screen_description: str | None = None,
) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=ActionType.click,
        target=target,
        confidence=0.9,
        screen_description=screen_description,
    )


def _text(
    index: int,
    timestamp_ms: int,
    *,
    visible_text: str | None,
    description: str,
    target: str = "banner",
) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=ActionType.assert_text,
        target=target,
        confidence=0.9,
        assertion_evidence=AssertionEvidence(
            description=description,
            visible_text=visible_text,
        ),
    )


def _failed(
    step: int,
    matched: int | None,
    timestamp_ms: int | None,
    finding: FindingType = FindingType.wrong_target,
) -> StepVerdict:
    return StepVerdict(
        expected_step_index=step,
        passed=False,
        matched_observed_index=matched,
        evidence_timestamp_ms=timestamp_ms,
        finding_types=[finding],
        evidence_id=None if matched is None else f"O{matched}",
    )


def _passed(step: int, matched: int, timestamp_ms: int) -> StepVerdict:
    return StepVerdict(
        expected_step_index=step,
        passed=True,
        matched_observed_index=matched,
        evidence_timestamp_ms=timestamp_ms,
        finding_types=[],
        evidence_id=f"O{matched}",
        explanation=(
            f"Observed action O{matched} at {timestamp_ms} ms supports this step."
        ),
    )


def _recover(
    steps: list[ExpectedStep],
    actions: list[ObservedAction],
    verdicts: list[StepVerdict],
) -> AuditResult:
    expected = ExpectedTrace(scenario_title="Fence", steps=steps)
    observed = ObservedTrace(source_file="fence.mp4", actions=actions)
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[],
        step_verdicts=verdicts,
    )
    return recover_later_support(expected, observed, audit)


def _assert_lineage(
    verdict: StepVerdict,
    index: int,
    timestamp_ms: int,
    reason: str,
) -> None:
    assert verdict.passed is True
    assert verdict.matched_observed_index == index
    assert verdict.evidence_timestamp_ms == timestamp_ms
    assert verdict.evidence_id == f"O{index}"
    assert verdict.recovery_reason == reason
    assert verdict.finding_types == []
    assert verdict.explanation == (
        f"Observed action O{index} at {timestamp_ms} ms supports this step."
    )


def test_evidence_before_invalid_pair_inside_fence_is_fenced() -> None:
    result = _recover(
        [
            _step(1, "Open settings"),
            _step(2, "Save button"),
            _step(3, "Close button"),
        ],
        [
            _click(1, "Open settings", 1000),
            _click(2, "Save button", 2000),
            _click(3, "Delete button", 3000),
            _click(4, "Close button", 5000),
        ],
        [
            _passed(1, 1, 1000),
            _failed(2, 3, 3000),
            _passed(3, 4, 5000),
        ],
    )

    _assert_lineage(result.step_verdicts[1], 2, 2000, "fenced_observed_action")
    assert result.step_verdicts[0].recovery_reason is None
    assert result.step_verdicts[2].matched_observed_index == 4
    assert any(
        finding.type == FindingType.extra_action
        and finding.observed_action_index == 3
        for finding in result.findings
    )


def test_evidence_after_invalid_pair_uses_later_reason() -> None:
    result = _recover(
        [
            _step(1, "Open settings"),
            _step(2, "Save button"),
            _step(3, "Close button"),
        ],
        [
            _click(1, "Open settings", 1000),
            _click(2, "Delete button", 2000),
            _click(3, "Save button", 3500),
            _click(4, "Close button", 5000),
        ],
        [
            _passed(1, 1, 1000),
            _failed(2, 2, 2000),
            _passed(3, 4, 5000),
        ],
    )

    _assert_lineage(result.step_verdicts[1], 3, 3500, "later_observed_action")


def test_evidence_before_the_previous_passed_step_is_not_recovered() -> None:
    result = _recover(
        [_step(1, "Open settings"), _step(2, "Save button")],
        [
            _click(1, "Save button", 400),
            _click(2, "Open settings", 1000),
            _click(3, "Delete button", 3000),
        ],
        [_passed(1, 2, 1000), _failed(2, 3, 3000)],
    )

    failed = result.step_verdicts[1]
    assert failed.passed is False
    assert failed.matched_observed_index == 3
    assert failed.recovery_reason is None


def test_evidence_after_the_next_passed_step_is_not_recovered() -> None:
    result = _recover(
        [_step(1, "Save button"), _step(2, "Close button")],
        [
            _click(1, "Delete button", 2000),
            _click(2, "Close button", 3000),
            _click(3, "Save button", 4000),
        ],
        [_failed(1, 1, 2000), _passed(2, 2, 3000)],
    )

    failed = result.step_verdicts[0]
    assert failed.passed is False
    assert failed.matched_observed_index == 1
    assert failed.recovery_reason is None


def test_two_valid_observations_in_the_fence_stay_unmatched() -> None:
    result = _recover(
        [
            _step(1, "Open settings"),
            _step(2, "Save button"),
            _step(3, "Close button"),
        ],
        [
            _click(1, "Open settings", 1000),
            _click(2, "Save button", 1500),
            _click(3, "Save button", 2500),
            _click(4, "Delete button", 3000),
            _click(5, "Close button", 5000),
        ],
        [
            _passed(1, 1, 1000),
            _failed(2, 4, 3000),
            _passed(3, 5, 5000),
        ],
    )

    failed = result.step_verdicts[1]
    assert failed.passed is False
    assert failed.matched_observed_index == 4
    assert failed.recovery_reason is None


def test_contradicting_state_does_not_count_beside_a_valid_observation() -> None:
    result = _recover(
        [
            _step(1, "Notifications switch", raw_text="Turn on notifications"),
        ],
        [
            _click(1, "Notifications switch", 1500, screen_description="Notifications switched off"),
            _click(2, "Notifications switch", 2200, screen_description="Notifications switched on"),
            _click(3, "Delete button", 3000),
        ],
        [_failed(1, 3, 3000)],
    )

    _assert_lineage(result.step_verdicts[0], 2, 2200, "fenced_observed_action")


def test_sibling_choice_control_is_not_recovered() -> None:
    result = _recover(
        [_step(1, "English checkbox")],
        [
            _click(1, "Spanish checkbox", 1500),
            _click(2, "Delete button", 3000),
        ],
        [_failed(1, 2, 3000)],
    )

    failed = result.step_verdicts[0]
    assert failed.passed is False
    assert failed.matched_observed_index == 2


def test_matching_choice_control_is_recovered_when_a_sibling_is_present() -> None:
    result = _recover(
        [_step(1, "English checkbox")],
        [
            _click(1, "Spanish checkbox", 1500),
            _click(2, "English checkbox", 2200),
            _click(3, "Delete button", 3000),
        ],
        [_failed(1, 3, 3000)],
    )

    _assert_lineage(result.step_verdicts[0], 2, 2200, "fenced_observed_action")


def test_model_description_alone_does_not_recover_an_assertion() -> None:
    result = _recover(
        [
            _step(
                1,
                "banner",
                action=ActionType.assert_text,
                assertion="Invalid username",
            ),
        ],
        [
            _text(1, 1500, visible_text="Please wait", description="Invalid username"),
            _text(2, 3000, visible_text="Please wait", description="Please wait"),
        ],
        [_failed(1, 2, 3000, FindingType.assertion_not_observed)],
    )

    failed = result.step_verdicts[0]
    assert failed.passed is False
    assert failed.matched_observed_index == 2


def test_visible_assertion_text_before_the_invalid_pair_is_recovered() -> None:
    result = _recover(
        [
            _step(
                1,
                "banner",
                action=ActionType.assert_text,
                assertion="Invalid username",
            ),
        ],
        [
            _text(1, 1500, visible_text="Invalid username", description="model prose"),
            _text(2, 3000, visible_text="Please wait", description="Please wait"),
        ],
        [_failed(1, 2, 3000, FindingType.assertion_not_observed)],
    )

    _assert_lineage(result.step_verdicts[0], 1, 1500, "fenced_observed_action")


def test_click_without_assertion_text_does_not_prove_an_assertion() -> None:
    result = _recover(
        [
            _step(
                1,
                "banner",
                action=ActionType.assert_text,
                assertion="Invalid username",
            ),
        ],
        [
            _click(1, "banner", 1500, screen_description="Invalid username"),
            _text(2, 3000, visible_text="Please wait", description="Please wait"),
        ],
        [_failed(1, 2, 3000, FindingType.assertion_not_observed)],
    )

    assert result.step_verdicts[0].passed is False
    assert result.step_verdicts[0].matched_observed_index == 2


def test_earlier_iteration_outside_the_fence_is_not_recovered() -> None:
    result = _recover(
        [_step(1, "Open settings"), _step(2, "Save button")],
        [
            _click(1, "Save button", 1000),
            _click(2, "Open settings", 3000),
            _click(3, "Save button", 4000),
            _click(4, "Delete button", 4500),
        ],
        [_passed(1, 2, 3000), _failed(2, 4, 4500)],
    )

    _assert_lineage(result.step_verdicts[1], 3, 4000, "fenced_observed_action")


def test_two_iterations_inside_the_same_fence_stay_unmatched() -> None:
    result = _recover(
        [_step(1, "Save button")],
        [
            _click(1, "Save button", 1000),
            _click(2, "Save button", 2000),
            _click(3, "Delete button", 3000),
        ],
        [_failed(1, 3, 3000)],
    )

    failed = result.step_verdicts[0]
    assert failed.passed is False
    assert failed.matched_observed_index == 3
    assert failed.recovery_reason is None


def test_observation_claimed_by_passed_step_is_not_reused() -> None:
    result = _recover(
        [_step(1, "Save button"), _step(2, "Save button")],
        [
            _click(1, "Save button", 2000),
            _click(2, "Delete button", 3000),
        ],
        [_passed(1, 1, 2000), _failed(2, 2, 3000)],
    )

    failed = result.step_verdicts[1]
    assert failed.passed is False
    assert failed.matched_observed_index == 2
    assert result.step_verdicts[0].matched_observed_index == 1


def test_failed_pair_does_not_reserve_its_observation() -> None:
    result = _recover(
        [_step(1, "English checkbox"), _step(2, "Spanish checkbox")],
        [
            _click(1, "Spanish checkbox", 1000),
            _click(2, "Delete button", 3000),
        ],
        [_failed(1, 1, 1000), _failed(2, 2, 3000)],
    )

    assert result.step_verdicts[0].passed is False
    _assert_lineage(result.step_verdicts[1], 1, 1000, "fenced_observed_action")


def test_in_order_evidence_before_each_invalid_pair_recovers_both_steps() -> None:
    result = _recover(
        [_step(1, "Save button"), _step(2, "Continue button")],
        [
            _click(1, "Save button", 200),
            _click(2, "Delete button", 800),
            _click(3, "Continue button", 400),
            _click(4, "Cancel button", 900),
        ],
        [_failed(1, 2, 800), _failed(2, 4, 900)],
    )

    _assert_lineage(result.step_verdicts[0], 1, 200, "fenced_observed_action")
    _assert_lineage(result.step_verdicts[1], 3, 400, "fenced_observed_action")


def test_align_swapped_choice_controls_recover_only_the_later_valid_occurrence() -> None:
    expected = ExpectedTrace(
        scenario_title="Languages",
        steps=[
            _step(1, "English checkbox"),
            _step(2, "Spanish checkbox"),
        ],
    )
    observed = ObservedTrace(
        source_file="languages.mp4",
        actions=[
            _click(1, "Spanish checkbox", 100),
            _click(2, "English checkbox", 200),
        ],
    )

    result = align(expected, observed)
    english, spanish = result.step_verdicts

    _assert_lineage(english, 2, 200, "later_observed_action")
    assert spanish.passed is False
    assert spanish.recovery_reason is None
    assert FindingType.wrong_target in spanish.finding_types


def test_align_choice_controls_in_recorded_order_pass_without_recovery() -> None:
    expected = ExpectedTrace(
        scenario_title="Languages",
        steps=[
            _step(1, "English checkbox"),
            _step(2, "Spanish checkbox"),
        ],
    )
    observed = ObservedTrace(
        source_file="languages.mp4",
        actions=[
            _click(1, "English checkbox", 100),
            _click(2, "Spanish checkbox", 200),
        ],
    )

    result = align(expected, observed)

    assert [verdict.passed for verdict in result.step_verdicts] == [True, True]
    assert [verdict.matched_observed_index for verdict in result.step_verdicts] == [1, 2]
    assert [verdict.recovery_reason for verdict in result.step_verdicts] == [None, None]


def _languages(observed_targets: list[tuple[str, int]]) -> tuple[ExpectedTrace, ObservedTrace]:
    expected = ExpectedTrace(
        scenario_title="Languages",
        steps=[
            _step(1, "English checkbox"),
            _step(2, "Spanish checkbox"),
        ],
    )
    observed = ObservedTrace(
        source_file="languages.mp4",
        actions=[
            _click(index, target, timestamp_ms)
            for index, (target, timestamp_ms) in enumerate(observed_targets, start=1)
        ],
    )
    return expected, observed


def _assert_passed_observations_are_unique(result: AuditResult) -> None:
    claimed = [
        verdict.matched_observed_index
        for verdict in result.step_verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    ]
    assert len(claimed) == len(set(claimed))


def test_recovered_observation_is_not_also_cited_by_the_failed_step() -> None:
    expected, observed = _languages([
        ("Spanish checkbox", 100),
        ("English checkbox", 200),
    ])
    result = align(expected, observed)
    english, spanish = result.step_verdicts

    _assert_lineage(english, 2, 200, "later_observed_action")
    assert spanish.passed is False
    assert spanish.matched_observed_index is None
    assert spanish.evidence_id is None
    assert spanish.evidence_timestamp_ms == 200
    assert spanish.finding_types == [FindingType.wrong_target]
    assert spanish.explanation is None
    assert spanish.recovery_reason is None
    wrong_target = next(
        finding for finding in result.findings
        if finding.type == FindingType.wrong_target
    )
    assert wrong_target.expected_step_index == 2
    assert wrong_target.observed_action_index is None
    assert wrong_target.evidence_timestamp_ms == 200


def test_unclaimed_failed_citation_stays() -> None:
    result = _recover(
        [_step(1, "Open settings"), _step(2, "Save button")],
        [
            _click(1, "Open settings", 1000),
            _click(2, "Delete button", 3000),
        ],
        [_passed(1, 1, 1000), _failed(2, 2, 3000)],
    )

    failed = result.step_verdicts[1]
    assert failed.passed is False
    assert failed.matched_observed_index == 2
    assert failed.evidence_id == "O2"
    assert failed.evidence_timestamp_ms == 3000
    assert failed.finding_types == [FindingType.wrong_target]
    assert result.step_verdicts[0].matched_observed_index == 1


def test_passed_steps_never_claim_the_same_observation() -> None:
    swapped = align(*_languages([
        ("Spanish checkbox", 100),
        ("English checkbox", 200),
    ]))
    in_order = align(*_languages([
        ("English checkbox", 100),
        ("Spanish checkbox", 200),
    ]))
    recovered = _recover(
        [_step(1, "Save button"), _step(2, "Continue button")],
        [
            _click(1, "Save button", 200),
            _click(2, "Delete button", 800),
            _click(3, "Continue button", 400),
            _click(4, "Cancel button", 900),
        ],
        [_failed(1, 2, 800), _failed(2, 4, 900)],
    )

    _assert_passed_observations_are_unique(swapped)
    _assert_passed_observations_are_unique(in_order)
    _assert_passed_observations_are_unique(recovered)
    assert [verdict.matched_observed_index for verdict in recovered.step_verdicts] == [1, 3]


def test_early_return_clears_a_citation_already_claimed_by_a_passed_step() -> None:
    finding = Finding(
        type=FindingType.wrong_target,
        expected_step_index=2,
        observed_action_index=1,
        evidence_timestamp_ms=1000,
        description="Step 2 was paired with the English click.",
    )
    failed = StepVerdict(
        expected_step_index=2,
        passed=False,
        matched_observed_index=1,
        evidence_timestamp_ms=1000,
        finding_types=[FindingType.wrong_target],
        evidence_id="O1",
        explanation="Diagnostic pair only.",
    )
    expected = ExpectedTrace(
        scenario_title="Languages",
        steps=[_step(1, "English checkbox"), _step(2, "Spanish checkbox")],
    )
    observed = ObservedTrace(
        source_file="languages.mp4",
        actions=[_click(1, "English checkbox", 1000)],
    )
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.5,
        findings=[finding],
        step_verdicts=[_passed(1, 1, 1000), failed],
    )

    result = recover_later_support(expected, observed, audit)
    kept = result.step_verdicts[1]

    assert result.step_verdicts[0].matched_observed_index == 1
    assert kept.passed is False
    assert kept.matched_observed_index is None
    assert kept.evidence_id is None
    assert kept.evidence_timestamp_ms == 1000
    assert kept.finding_types == [FindingType.wrong_target]
    assert kept.explanation == "Diagnostic pair only."
    assert kept.recovery_reason is None
    assert result.findings == [finding]
    assert result.findings[0] is finding


def test_assist_does_not_review_an_observation_owned_by_a_passed_step() -> None:
    expected, observed = _languages([
        ("Spanish checkbox", 100),
        ("English checkbox", 200),
    ])
    result = align(expected, observed)

    assert select_assist_candidates(expected, observed, result) == []


def test_unclaimed_wrong_target_remains_an_assist_candidate() -> None:
    expected = ExpectedTrace(
        scenario_title="Languages",
        steps=[_step(1, "English checkbox")],
    )
    observed = ObservedTrace(
        source_file="languages.mp4",
        actions=[_click(1, "Spanish checkbox", 100)],
    )
    result = align(expected, observed)

    assert result.step_verdicts[0].matched_observed_index == 1
    candidates = select_assist_candidates(expected, observed, result)
    assert [(item.expected_step_index, item.matched_observed_index) for item in candidates] == [
        (1, 1),
    ]
