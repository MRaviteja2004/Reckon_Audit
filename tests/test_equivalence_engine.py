"""Predicate engine: citations, assignment, and legacy compatibility."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.assist_propose import _AssistBatchResponse
from reckon.equivalence.citations import bind_expected_intent, bind_observed_interaction
from reckon.equivalence.engine import (
    apply_agent_equivalence,
    apply_base_equivalence,
    evaluate,
)
from reckon.equivalence.table import TABLE_VERSION, TYPE_REALIZATIONS
from reckon.intent_contract import ReportedIntent
from reckon.parser import _build_expected_trace, _LLMStep, _LLMTrace
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditResult,
    CandidateEvidenceType,
    CandidateSearchStatus,
    ElementKind,
    EquivalenceDecision,
    ExpectedActionIntent,
    ExpectedObject,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    IntentOperation,
    ObjectKind,
    ObservedAction,
    ObservedElement,
    ObservedInteraction,
    ObservedTrace,
    SlotCitation,
    StepVerdict,
)
from reckon.stage2.interaction_binding import bind_reported_interaction
from reckon.stage2.interaction_contract import ReportedInteraction
from reckon.stage2c.manifest import build_frame_manifest
from reckon.stage2c.propose import RawFrameEvidenceBatch, RawFrameEvidenceCandidate
from reckon.tolerance import actions_compatible

SCENARIO = "Open the Driver application."


def _cite(source: str, text: str) -> SlotCitation:
    start = source.index(text)
    return SlotCitation(start=start, end=start + len(text), text=text)


def _intent(
    scenario: str,
    *,
    operation: IntentOperation,
    operation_text: str,
    kind: ObjectKind,
    kind_text: str,
    name: str | None = None,
    product: str | None = None,
    instance: str | None = None,
    value: str | None = None,
    sentence: tuple[int, int] | None = None,
) -> ExpectedActionIntent:
    start, end = (0, len(scenario)) if sentence is None else sentence
    return ExpectedActionIntent(
        operation=operation,
        operation_citation=_cite(scenario, operation_text),
        object=ExpectedObject(
            kind=kind,
            kind_citation=_cite(scenario, kind_text),
            name=name,
            name_citation=None if name is None else _cite(scenario, name),
            product=product,
            product_citation=None if product is None else _cite(scenario, product),
            instance=instance,
            instance_citation=None if instance is None else _cite(scenario, instance),
            value=value,
            value_citation=None if value is None else _cite(scenario, value),
        ),
        sentence_start=start,
        sentence_end=end,
    )


def _interaction(
    surface: str,
    *,
    operation: IntentOperation,
    kind: ElementKind,
    kind_text: str,
    name: str | None = None,
    product: str | None = None,
    instance: str | None = None,
    value: str | None = None,
    frame: int = 1,
    episode: int | None = None,
) -> ObservedInteraction:
    return ObservedInteraction(
        operation=operation,
        element=ObservedElement(
            kind=kind,
            kind_citation=_cite(surface, kind_text),
            name=name,
            name_citation=None if name is None else _cite(surface, name),
            product=product,
            product_citation=None if product is None else _cite(surface, product),
            instance=instance,
            instance_citation=None if instance is None else _cite(surface, instance),
            value=value,
            value_citation=None if value is None else _cite(surface, value),
        ),
        citation_surface="target",
        cited_frame_index=frame,
        context_episode_id=episode,
    )


def _obs(
    index: int,
    timestamp: int,
    target: str,
    interaction: ObservedInteraction | None,
    *,
    action: ActionType = ActionType.click,
) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp,
        action=action,
        target=target,
        confidence=0.4,
        screen_description="must not be read",
        interaction=interaction,
    )


def _step(index: int, intent: ExpectedActionIntent | None, **kwargs) -> ExpectedStep:
    return ExpectedStep(index=index, action=ActionType.other, action_intent=intent, **kwargs)


def _failed_audit(steps: list[ExpectedStep], actions: list[ObservedAction]) -> AuditResult:
    return AuditResult(
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.advisory,
                expected_step_index=step.index,
                description="missing",
            )
            for step in steps
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=step.index,
                passed=False,
                finding_types=[FindingType.missing_action],
            )
            for step in steps
        ],
    )


def _open_intent(scenario: str = SCENARIO, **kwargs) -> ExpectedActionIntent:
    return _intent(
        scenario,
        operation=IntentOperation.traverse,
        operation_text="Open",
        kind=ObjectKind.application,
        kind_text="application",
        name="Driver",
        **kwargs,
    )


def test_table_version_and_cells_are_enums():
    assert TABLE_VERSION == "1"
    ids = [row.rule_id for row in TYPE_REALIZATIONS]
    assert ids == ["same_kind", "application_icon", "control_button", "control_link"]
    for row in TYPE_REALIZATIONS:
        for cell in (
            row.expected_operation,
            row.observed_operation,
            row.expected_kind,
            row.observed_kind,
        ):
            assert cell is None or isinstance(cell, (IntentOperation, ObjectKind, ElementKind))


def test_name_adjacent_to_kind_in_driver_application():
    scenario = "Open the Driver application."
    intent = bind_expected_intent(
        scenario,
        sentence_start=0,
        sentence_end=len(scenario),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    )
    assert intent is not None
    target = "Uber Driver application icon"
    observed = _obs(1, 1000, target, _interaction(
        target,
        operation=IntentOperation.activate,
        kind=ElementKind.icon,
        kind_text="icon",
        name="Driver",
        product="Uber",
    ))
    match = evaluate(
        _step(1, intent),
        observed,
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    )
    assert getattr(match, "relation_id", None) == "application_icon@1"
    assert match.evidence_id == "O1"
    assert match.evidence_timestamp_ms == 1000


def test_drivers_modifier_is_not_adjacent_to_application():
    scenario = "Open the application used by drivers."
    intent = bind_expected_intent(
        scenario,
        sentence_start=0,
        sentence_end=len(scenario),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="drivers",
    )
    assert intent is not None
    target = "drivers icon"
    result = evaluate(
        _step(1, intent),
        _obs(1, 1000, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="drivers",
        )),
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    )
    assert result == "bad_citation"


def test_driver_accounts_is_not_the_application_name():
    scenario = "Open the Uber application for Driver accounts."
    adjacent = bind_expected_intent(
        scenario,
        sentence_start=0,
        sentence_end=len(scenario),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Uber",
    )
    assert adjacent is not None
    drifted = bind_expected_intent(
        scenario,
        sentence_start=0,
        sentence_end=len(scenario),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    )
    assert drifted is not None
    target = "Uber icon"
    uber = evaluate(
        _step(1, adjacent),
        _obs(1, 1000, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Uber",
        )),
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    )
    assert getattr(uber, "relation_id", None) == "application_icon@1"
    result = evaluate(
        _step(1, drifted),
        _obs(1, 1000, "Driver icon", _interaction(
            "Driver icon", operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    )
    assert result == "bad_citation"


def test_exclusive_end_and_altered_citation_text():
    scenario = SCENARIO
    start = scenario.index("Driver")
    altered = _open_intent()
    altered.object.name_citation = SlotCitation(start=start, end=start + len("Draver"), text="Draver")
    altered.object.name = "Draver"
    result = evaluate(
        _step(1, altered),
        _obs(1, 1000, "Driver icon", _interaction(
            "Driver icon", operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    )
    assert result == "bad_citation"
    citation = _cite(scenario, "Driver")
    assert scenario[citation.start:citation.end] == "Driver"
    assert scenario[citation.start:citation.end + 1] != "Driver"


def test_citation_from_another_sentence_is_rejected():
    scenario = "Open the Driver application. Then close the other application."
    first = scenario.index(".") + 1
    intent = _open_intent(scenario, sentence=(0, first))
    intent.object.name_citation = _cite(scenario, "other")
    intent.object.name = "other"
    result = evaluate(
        _step(1, intent),
        _obs(1, 1000, "other icon", _interaction(
            "other icon", operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="other",
        )),
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    )
    assert result == "bad_citation"


def test_invalid_offsets_are_rejected():
    with pytest.raises(ValidationError):
        SlotCitation(start=5, end=5, text="")
    scenario = SCENARIO
    intent = _open_intent()
    intent.object.name_citation = SlotCitation(start=50, end=56, text="Driver")
    result = evaluate(
        _step(1, intent),
        _obs(1, 1000, "Driver icon", _interaction(
            "Driver icon", operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
        scenario_text=scenario,
        fence=(None, None),
        claimed=set(),
    )
    assert result == "bad_citation"


def test_parser_binds_citations_or_drops_the_intent():
    intent = ReportedIntent(
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    )
    step = _LLMStep(
        action=ActionType.other,
        raw_text="Open the Driver application.",
        intent=intent,
    )
    gherkin = "Feature: Apps\nScenario: Open\n  When Open the Driver application.\n"
    parsed = _build_expected_trace(
        _LLMTrace(scenario_title="t", steps=[step]),
        scenario_text=gherkin,
    )
    assert parsed.source_text == gherkin
    assert parsed.steps[0].action_intent is not None
    assert parsed.steps[0].action_intent.object.name == "Driver"
    numbered = _build_expected_trace(
        _LLMTrace(scenario_title="t", steps=[step]),
        scenario_text="1. Open the Driver application.\n",
    )
    assert numbered.steps[0].action_intent is not None
    prose = _build_expected_trace(
        _LLMTrace(scenario_title="t", steps=[step]),
        scenario_text="Please start. Open the Driver application.",
    )
    assert prose.steps[0].action_intent is not None
    uncited = _build_expected_trace(
        _LLMTrace(scenario_title="t", steps=[step]),
        scenario_text="No matching sentence here.",
    )
    assert uncited.steps[0].action_intent is None


def test_gherkin_and_numbered_spans_bind_exact_slices():
    gherkin = "Feature: Apps\nScenario: Open\n  When Open the Driver application.\n"
    body = "Open the Driver application."
    start = gherkin.index(body)
    intent = bind_expected_intent(
        gherkin,
        sentence_start=start,
        sentence_end=start + len(body),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    )
    assert intent is not None
    assert gherkin[intent.object.name_citation.start:intent.object.name_citation.end] == "Driver"
    numbered = "1. Open the Driver application.\n"
    body_start = numbered.index(body)
    numbered_intent = bind_expected_intent(
        numbered,
        sentence_start=body_start,
        sentence_end=body_start + len(body),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    )
    assert numbered_intent is not None
    prose = "First do nothing. Open the Driver application. Stop."
    prose_start = prose.index(body)
    assert bind_expected_intent(
        prose,
        sentence_start=prose_start,
        sentence_end=prose_start + len(body),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    ) is not None
    assert bind_expected_intent(
        prose,
        sentence_start=0,
        sentence_end=len("First do nothing."),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    ) is None


def test_two_submit_buttons_stay_ambiguous_regardless_of_episode():
    scenario = "Activate the Submit control."
    intent = _intent(
        scenario,
        operation=IntentOperation.activate,
        operation_text="Activate",
        kind=ObjectKind.control,
        kind_text="control",
        name="Submit",
    )
    trace = ExpectedTrace(scenario_title="t", source_text=scenario, steps=[_step(1, intent)])
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 1000, "Submit button", _interaction(
            "Submit button", operation=IntentOperation.activate, kind=ElementKind.button,
            kind_text="button", name="Submit", episode=1,
        )),
        _obs(2, 1100, "Submit button", _interaction(
            "Submit button", operation=IntentOperation.activate, kind=ElementKind.button,
            kind_text="button", name="Submit", episode=1,
        )),
    ])
    audit = apply_base_equivalence(trace, observed, _failed_audit(trace.steps, observed.actions), scenario_text=scenario)
    assert audit.equivalence_decisions[0].status == "ambiguous_candidates"
    assert audit.step_verdicts[0].passed is False
    observed.actions[1].interaction.context_episode_id = 9
    again = apply_base_equivalence(trace, observed, _failed_audit(trace.steps, observed.actions), scenario_text=scenario)
    assert again.equivalence_decisions[0].status == "ambiguous_candidates"


def test_duplicate_observations_of_one_button_stay_ambiguous():
    scenario = "Activate the Submit control."
    intent = _intent(
        scenario,
        operation=IntentOperation.activate,
        operation_text="Activate",
        kind=ObjectKind.control,
        kind_text="control",
        name="Submit",
    )
    interaction = _interaction(
        "Submit button", operation=IntentOperation.activate, kind=ElementKind.button,
        kind_text="button", name="Submit", episode=3,
    )
    trace = ExpectedTrace(scenario_title="t", source_text=scenario, steps=[_step(1, intent)])
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 1000, "Submit button", interaction),
        _obs(2, 1200, "Submit button", interaction.model_copy(deep=True)),
    ])
    audit = apply_base_equivalence(trace, observed, _failed_audit(trace.steps, observed.actions), scenario_text=scenario)
    assert audit.equivalence_decisions[0].status == "ambiguous_candidates"
    assert audit.equivalence_decisions[0].observed_index is None


def test_base_match_survives_later_agent_proposals():
    scenario = SCENARIO
    intent = _open_intent()
    target = "Driver icon"
    trace = ExpectedTrace(scenario_title="t", source_text=scenario, steps=[_step(1, intent)])
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 1000, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
        _obs(2, 1500, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
    ])
    # Two icons are ambiguous at Base, so use a single icon for the committed match.
    observed.actions = observed.actions[:1]
    base = apply_base_equivalence(trace, observed, _failed_audit(trace.steps, observed.actions), scenario_text=scenario)
    assert base.equivalence_decisions[0].status == "committed"
    extra = ObservedTrace(source_file="r.mp4", actions=[
        observed.actions[0],
        _obs(2, 1500, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
    ])
    verified = apply_agent_equivalence(trace, extra, base, {1: [2]}, scenario_text=scenario)
    assert verified.equivalence_decisions[0].status == "committed"
    assert verified.equivalence_decisions[0].agent_status == "ignored_after_base_commit"
    assert verified.step_verdicts[0].matched_observed_index == 1
    rejected = apply_agent_equivalence(trace, extra, base, {1: [99]}, scenario_text=scenario)
    assert rejected.equivalence_decisions[0].status == "committed"
    assert rejected.equivalence_decisions[0].agent_status == "ignored_after_base_commit"
    assert rejected.step_verdicts[0].evidence_timestamp_ms == 1000


def test_agent_fills_only_a_base_gap_and_not_a_tie():
    scenario = "Activate the Submit control."
    intent = _intent(
        scenario,
        operation=IntentOperation.activate,
        operation_text="Activate",
        kind=ObjectKind.control,
        kind_text="control",
        name="Submit",
    )
    trace = ExpectedTrace(scenario_title="t", source_text=scenario, steps=[_step(1, intent)])
    target = "Submit button"
    action = _obs(1, 1000, target, _interaction(
        target, operation=IntentOperation.activate, kind=ElementKind.button,
        kind_text="button", name="Submit",
    ))
    observed = ObservedTrace(source_file="r.mp4", actions=[action])
    gap = _failed_audit(trace.steps, observed.actions)
    gap.equivalence_decisions = [EquivalenceDecision(expected_step_index=1, status="no_candidate")]
    filled = apply_agent_equivalence(trace, observed, gap, {1: [1]}, scenario_text=scenario)
    assert filled.equivalence_decisions[0].status == "no_candidate"
    assert filled.equivalence_decisions[0].agent_status == "committed"
    assert filled.step_verdicts[0].passed is True
    assert filled.step_verdicts[0].evidence_id == "O1"

    second = _obs(2, 1300, target, action.interaction.model_copy(deep=True))
    two = apply_agent_equivalence(
        trace,
        ObservedTrace(source_file="r.mp4", actions=[action, second]),
        gap,
        {1: [1, 2]},
        scenario_text=scenario,
    )
    assert two.step_verdicts[0].passed is False
    assert two.equivalence_decisions[0].agent_rejection == "ambiguous_candidates"

    ambiguous = gap.model_copy(update={"equivalence_decisions": [
        EquivalenceDecision(expected_step_index=1, status="ambiguous_candidates"),
    ]})
    stuck = apply_agent_equivalence(trace, observed, ambiguous, {1: [1]}, scenario_text=scenario)
    assert stuck.equivalence_decisions[0].status == "ambiguous_candidates"
    assert stuck.step_verdicts[0].passed is False


def test_contested_observation_is_awarded_to_neither_step():
    scenario = "Open the Driver application. Open the Driver application."
    first_end = scenario.index(".") + 1
    intents = [
        _open_intent(scenario, sentence=(0, first_end)),
        _open_intent(scenario, sentence=(first_end, len(scenario))),
    ]
    # The second sentence's "Open" / "Driver" / "application" still cite the first
    # occurrence via _cite. Build the second sentence's citations explicitly.
    second = scenario[first_end:]
    second_intent = bind_expected_intent(
        scenario,
        sentence_start=first_end,
        sentence_end=len(scenario),
        operation="traverse",
        operation_text="Open",
        kind="application",
        kind_text="application",
        name="Driver",
    )
    assert second_intent is not None
    intents[1] = second_intent
    trace = ExpectedTrace(
        scenario_title="t",
        source_text=scenario,
        steps=[_step(1, intents[0]), _step(2, intents[1])],
    )
    target = "Driver icon"
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 1000, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
    ])
    audit = apply_base_equivalence(trace, observed, _failed_audit(trace.steps, observed.actions), scenario_text=scenario)
    assert {item.status for item in audit.equivalence_decisions} == {"contested_observation"}
    assert all(verdict.passed is False for verdict in audit.step_verdicts)


def test_chronology_violation_keeps_the_earlier_commit():
    scenario = "Open the Driver application. Open the Maps application."
    first_end = scenario.index(".") + 1
    first = bind_expected_intent(
        scenario, sentence_start=0, sentence_end=first_end,
        operation="traverse", operation_text="Open", kind="application",
        kind_text="application", name="Driver",
    )
    second = bind_expected_intent(
        scenario, sentence_start=first_end, sentence_end=len(scenario),
        operation="traverse", operation_text="Open", kind="application",
        kind_text="application", name="Maps",
    )
    trace = ExpectedTrace(
        scenario_title="t", source_text=scenario,
        steps=[_step(1, first), _step(2, second)],
    )
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 2000, "Driver icon", _interaction(
            "Driver icon", operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
        _obs(2, 1000, "Maps icon", _interaction(
            "Maps icon", operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Maps",
        )),
    ])
    audit = apply_base_equivalence(trace, observed, _failed_audit(trace.steps, observed.actions), scenario_text=scenario)
    by_step = {item.expected_step_index: item for item in audit.equivalence_decisions}
    assert by_step[1].status == "committed"
    assert by_step[1].observed_index == 1
    assert by_step[2].status == "chronology_violation"
    assert audit.step_verdicts[0].passed is True
    assert audit.step_verdicts[1].passed is False


def test_click_after_the_next_legacy_proof_is_outside_the_fence():
    scenario = SCENARIO
    intent = _open_intent()
    later = ExpectedStep(index=2, action=ActionType.assert_visible, target="home", assertion="You're offline")
    trace = ExpectedTrace(scenario_title="t", source_text=scenario, steps=[_step(1, intent), later])
    target = "Driver icon"
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 2500, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
    ])
    audit = AuditResult(
        fidelity_score=0.5,
        findings=[],
        step_verdicts=[
            StepVerdict(expected_step_index=1, passed=False, finding_types=[FindingType.missing_action]),
            StepVerdict(
                expected_step_index=2, passed=True, matched_observed_index=None,
                evidence_timestamp_ms=1500, finding_types=[],
            ),
        ],
    )
    result = apply_base_equivalence(trace, observed, audit, scenario_text=scenario)
    assert result.equivalence_decisions[0].status == "no_candidate"
    assert result.step_verdicts[0].passed is False


def test_action_match_does_not_clear_an_assertion():
    scenario = SCENARIO
    intent = _open_intent()
    step = _step(1, intent, assertion="You're offline")
    trace = ExpectedTrace(scenario_title="t", source_text=scenario, steps=[step])
    target = "Driver icon"
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 1000, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
    ])
    audit = AuditResult(
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.advisory,
                expected_step_index=1,
                observed_action_index=1,
                description="wrong",
            ),
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                description="assertion",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                finding_types=[FindingType.wrong_target, FindingType.assertion_not_observed],
            ),
        ],
    )
    result = apply_base_equivalence(trace, observed, audit, scenario_text=scenario)
    verdict = result.step_verdicts[0]
    assert verdict.passed is False
    assert FindingType.assertion_not_observed in verdict.finding_types
    assert FindingType.wrong_target not in verdict.finding_types
    assert any(finding.type == FindingType.assertion_not_observed for finding in result.findings)


def test_realization_rows_positive_and_negative():
    cases = [
        ("same_kind", "Activate the Submit button.", ObjectKind.button, "button", "Submit button", ElementKind.button, "button", "Submit", "same_kind@1"),
        ("application_icon", SCENARIO, ObjectKind.application, "application", "Driver icon", ElementKind.icon, "icon", "Driver", "application_icon@1"),
        ("control_button", "Activate the Submit control.", ObjectKind.control, "control", "Submit button", ElementKind.button, "button", "Submit", "control_button@1"),
        ("control_link", "Activate the Submit control.", ObjectKind.control, "control", "Submit link", ElementKind.link, "link", "Submit", "control_link@1"),
    ]
    for _name, scenario, kind, kind_text, target, element, element_text, label, relation in cases:
        operation = IntentOperation.traverse if kind == ObjectKind.application else IntentOperation.activate
        operation_text = "Open" if kind == ObjectKind.application else "Activate"
        intent = _intent(
            scenario, operation=operation, operation_text=operation_text,
            kind=kind, kind_text=kind_text, name=label,
        )
        match = evaluate(
            _step(1, intent),
            _obs(1, 1000, target, _interaction(
                target, operation=IntentOperation.activate, kind=element,
                kind_text=element_text, name=label,
            )),
            scenario_text=scenario,
            fence=(None, None),
            claimed=set(),
        )
        assert getattr(match, "relation_id", match) == relation

    scenario = SCENARIO
    intent = _open_intent()
    page = evaluate(
        _step(1, intent),
        _obs(1, 1000, "Driver page", _interaction(
            "Driver page", operation=IntentOperation.activate, kind=ElementKind.page,
            kind_text="page", name="Driver",
        )),
        scenario_text=scenario, fence=(None, None), claimed=set(),
    )
    assert page == "type_not_realized"
    missing_icon = bind_observed_interaction(
        surface_text="Driver application",
        surface="target",
        operation="activate",
        kind="icon",
        kind_text="icon",
        cited_frame_index=1,
        name="Driver",
    )
    assert missing_icon is None
    mismatched = evaluate(
        _step(1, _intent(
            "Open the Uber application.",
            operation=IntentOperation.traverse,
            operation_text="Open",
            kind=ObjectKind.application,
            kind_text="application",
            name="Uber",
            product="Uber",
        )),
        _obs(1, 1000, "Lyft icon", _interaction(
            "Lyft icon", operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Lyft", product="Lyft",
        )),
        scenario_text="Open the Uber application.",
        fence=(None, None),
        claimed=set(),
    )
    assert mismatched == "identity_mismatch"
    button_for_application = evaluate(
        _step(1, intent),
        _obs(1, 1000, "Driver button", _interaction(
            "Driver button", operation=IntentOperation.activate, kind=ElementKind.button,
            kind_text="button", name="Driver",
        )),
        scenario_text=scenario, fence=(None, None), claimed=set(),
    )
    assert button_for_application == "type_not_realized"
    link_and_name = evaluate(
        _step(1, _intent(
            "Activate the Submit control.",
            operation=IntentOperation.activate,
            operation_text="Activate",
            kind=ObjectKind.control,
            kind_text="control",
            name="Submit",
        )),
        _obs(1, 1000, "Cancel link", _interaction(
            "Cancel link", operation=IntentOperation.activate, kind=ElementKind.link,
            kind_text="link", name="Cancel",
        )),
        scenario_text="Activate the Submit control.",
        fence=(None, None),
        claimed=set(),
    )
    assert link_and_name == "identity_mismatch"


def test_legacy_traces_produce_no_equivalence_result():
    expected = ExpectedTrace.model_validate({
        "scenario_title": "legacy",
        "steps": [{"index": 1, "action": "click", "target": "Login"}],
    })
    observed = ObservedTrace.model_validate({
        "source_file": "r.mp4",
        "actions": [{
            "index": 1,
            "timestamp_ms": 1000,
            "action": "click",
            "target": "Login",
            "confidence": 0.9,
        }],
    })
    assert expected.steps[0].action_intent is None
    assert observed.actions[0].interaction is None
    audit = align(expected, observed)
    assert audit.equivalence_decisions == []
    assert audit.step_verdicts[0].passed is True
    assert not actions_compatible(ActionType.other, ActionType.click)


def test_align_uses_structured_icon_without_changing_action_compatibility():
    scenario = SCENARIO
    intent = _open_intent()
    expected = ExpectedTrace(
        scenario_title="t",
        source_text=scenario,
        steps=[_step(1, intent, target="Driver application")],
    )
    target = "Driver icon"
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 1000, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
    ])
    audit = align(expected, observed)
    assert audit.step_verdicts[0].passed is True
    assert audit.step_verdicts[0].evidence_id == "O1"
    assert audit.step_verdicts[0].evidence_timestamp_ms == 1000
    assert audit.step_verdicts[0].recovery_reason == "application_icon@1"
    assert not actions_compatible(ActionType.other, ActionType.click)


def test_assist_does_not_replace_a_base_equivalence_commit():
    scenario = SCENARIO
    intent = _open_intent()
    expected = ExpectedTrace(scenario_title="t", source_text=scenario, steps=[_step(1, intent)])
    target = "Driver icon"
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _obs(1, 1000, target, _interaction(
            target, operation=IntentOperation.activate, kind=ElementKind.icon,
            kind_text="icon", name="Driver",
        )),
    ])
    assisted = align_with_assist(
        expected,
        observed,
        equivalence_agent_candidates={1: [1]},
    )
    verdict = assisted.result.step_verdicts[0]
    assert verdict.passed is True
    assert verdict.evidence_id == "O1"
    assert assisted.deterministic_result.equivalence_decisions[0].status == "committed"


def _driver_icon(index: int = 1, timestamp_ms: int = 1000) -> ObservedAction:
    target = "Driver icon"
    return _obs(index, timestamp_ms, target, _interaction(
        target, operation=IntentOperation.activate, kind=ElementKind.icon,
        kind_text="icon", name="Driver",
    ))


def test_graph_commit_releases_a_failed_verdict_citation_of_the_same_observation():
    scenario = SCENARIO
    expected = ExpectedTrace(
        scenario_title="t",
        source_text=scenario,
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Save button"),
            _step(2, _open_intent(), target="Driver application"),
        ],
    )
    audit = align(expected, ObservedTrace(source_file="r.mp4", actions=[_driver_icon()]))
    save, opened = audit.step_verdicts

    assert opened.passed is True
    assert opened.matched_observed_index == 1
    assert opened.evidence_id == "O1"
    assert opened.evidence_timestamp_ms == 1000
    assert opened.recovery_reason == "application_icon@1"
    assert save.passed is False
    assert save.matched_observed_index is None
    assert save.evidence_id is None
    assert save.evidence_timestamp_ms == 1000
    assert save.finding_types == [FindingType.wrong_target]
    wrong_target = next(
        finding for finding in audit.findings if finding.type == FindingType.wrong_target
    )
    assert wrong_target.expected_step_index == 1
    assert wrong_target.observed_action_index is None
    assert wrong_target.evidence_timestamp_ms == 1000
    assert audit.equivalence_decisions[0].status == "committed"
    assert audit.equivalence_decisions[0].observed_index == 1


def test_failed_citation_of_an_unclaimed_observation_survives_a_graph_commit():
    scenario = SCENARIO
    finding = Finding(
        type=FindingType.wrong_target,
        severity=FindingSeverity.advisory,
        expected_step_index=1,
        observed_action_index=2,
        evidence_timestamp_ms=2000,
        description="Step 1 was paired with the Delete button.",
    )
    audit = AuditResult(
        fidelity_score=0.0,
        findings=[
            finding,
            Finding(
                type=FindingType.missing_action,
                severity=FindingSeverity.advisory,
                expected_step_index=2,
                description="missing",
            ),
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=2,
                evidence_timestamp_ms=2000,
                finding_types=[FindingType.wrong_target],
                evidence_id="O2",
                explanation="Diagnostic pair only.",
            ),
            StepVerdict(
                expected_step_index=2,
                passed=False,
                finding_types=[FindingType.missing_action],
            ),
        ],
    )
    expected = ExpectedTrace(
        scenario_title="t",
        source_text=scenario,
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Save button"),
            _step(2, _open_intent(), target="Driver application"),
        ],
    )
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _driver_icon(),
        ObservedAction(
            index=2, timestamp_ms=2000, action=ActionType.click,
            target="Delete button", confidence=0.4,
        ),
    ])

    result = apply_base_equivalence(expected, observed, audit, scenario_text=scenario)
    failed = result.step_verdicts[0]
    opened = result.step_verdicts[1]

    assert opened.passed is True
    assert opened.matched_observed_index == 1
    assert failed.passed is False
    assert failed.matched_observed_index == 2
    assert failed.evidence_id == "O2"
    assert failed.evidence_timestamp_ms == 2000
    assert failed.explanation == "Diagnostic pair only."
    assert failed.finding_types == [FindingType.wrong_target]
    assert result.findings[0] is finding


def test_graph_matches_do_not_give_one_observation_to_two_passed_steps():
    first = "Open the Driver application."
    second = "Activate the Save control."
    scenario = f"{first} {second}"
    save_intent = _intent(
        scenario,
        operation=IntentOperation.activate,
        operation_text="Activate",
        kind=ObjectKind.control,
        kind_text="control",
        name="Save",
        sentence=(len(first) + 1, len(scenario)),
    )
    expected = ExpectedTrace(
        scenario_title="t",
        source_text=scenario,
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Quarterly invoice link"),
            _step(2, _open_intent(scenario), target="Driver application"),
            _step(3, save_intent, target="Save control"),
        ],
    )
    save_target = "Save button"
    observed = ObservedTrace(source_file="r.mp4", actions=[
        _driver_icon(),
        _obs(2, 2000, save_target, _interaction(
            save_target, operation=IntentOperation.activate, kind=ElementKind.button,
            kind_text="button", name="Save",
        )),
    ])

    audit = align(expected, observed)
    refund, opened, save = audit.step_verdicts
    claimed = [
        verdict.matched_observed_index
        for verdict in audit.step_verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    ]

    assert opened.passed is True and opened.matched_observed_index == 1
    assert save.passed is True and save.matched_observed_index == 2
    assert len(claimed) == len(set(claimed))
    assert refund.passed is False
    assert refund.matched_observed_index is None
    assert refund.evidence_id is None
    assert FindingType.wrong_target in refund.finding_types


def test_ordinary_alignment_without_intent_keeps_its_citations():
    expected = ExpectedTrace(
        scenario_title="Languages",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="English checkbox"),
            ExpectedStep(index=2, action=ActionType.click, target="Spanish checkbox"),
        ],
    )
    observed = ObservedTrace(source_file="languages.mp4", actions=[
        ObservedAction(
            index=1, timestamp_ms=100, action=ActionType.click,
            target="Spanish checkbox", confidence=0.9,
        ),
        ObservedAction(
            index=2, timestamp_ms=200, action=ActionType.click,
            target="English checkbox", confidence=0.9,
        ),
    ])
    audit = align(expected, observed)

    assert audit.equivalence_decisions == []
    assert audit.step_verdicts[0].matched_observed_index == 2
    assert audit.step_verdicts[1].matched_observed_index is None
    assert audit.step_verdicts[1].finding_types == [FindingType.wrong_target]


def test_stage2c_frame_recovery_does_not_claim_an_observation():
    client = MagicMock()
    client.chat.completions.create.return_value = _AssistBatchResponse(decisions=[])
    expected = ExpectedTrace(
        scenario_title="Offer",
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="English checkbox"),
            ExpectedStep(
                index=2, action=ActionType.assert_text, target="heading",
                assertion="Choose your guarantee",
            ),
        ],
    )
    observed = ObservedTrace(source_file="offer.mp4", actions=[
        ObservedAction(
            index=1, timestamp_ms=1000, action=ActionType.click,
            target="Spanish checkbox", confidence=0.9,
        ),
        ObservedAction(
            index=2, timestamp_ms=2000, action=ActionType.assert_text, target="home map",
            confidence=0.9,
            assertion_evidence=AssertionEvidence(description="Map", visible_text="Home map"),
        ),
    ])
    assisted = align_with_assist(
        expected,
        observed,
        openai_client=client,
        frame_evidence=True,
        frame_manifest=build_frame_manifest([1000, 8000]),
        stage2c_raw_batch=RawFrameEvidenceBatch(candidates=[
            RawFrameEvidenceCandidate(
                expected_step_id="E2",
                frame_ids=["F2"],
                evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
                observed_description="model prose is not the evidence",
                visible_text=["Choose your guarantee"],
                confidence=0.95,
                search_status=CandidateSearchStatus.candidate_found,
            ),
        ]),
    )
    click, assertion = assisted.result.step_verdicts

    assert click.passed is False
    assert click.matched_observed_index == 1
    assert assertion.passed is True
    assert assertion.matched_observed_index is None
    assert assertion.evidence_id == "F2"
    assert assertion.recovery_reason == "frame_evidence"


def test_agent_path_receives_the_corrected_base_result():
    scenario = SCENARIO
    expected = ExpectedTrace(
        scenario_title="t",
        source_text=scenario,
        steps=[
            ExpectedStep(index=1, action=ActionType.click, target="Save button"),
            _step(2, _open_intent(), target="Driver application"),
        ],
    )
    observed = ObservedTrace(source_file="r.mp4", actions=[_driver_icon()])
    assisted = align_with_assist(
        expected,
        observed,
        equivalence_agent_candidates={2: [1]},
    )
    save = assisted.deterministic_result.step_verdicts[0]
    opened = assisted.result.step_verdicts[1]

    assert assisted.deterministic_result.equivalence_decisions[0].status == "committed"
    assert assisted.deterministic_result.equivalence_decisions[0].agent_status == (
        "ignored_after_base_commit"
    )
    assert save.passed is False
    assert save.matched_observed_index is None
    assert save.evidence_timestamp_ms == 1000
    assert save.finding_types == [FindingType.wrong_target]
    assert opened.passed is True
    assert opened.matched_observed_index == 1
    assert opened.evidence_id == "O1"


# ---------------------------------------------------------------------------
# Stage 1 intents produced by the parser reaching the engine
# ---------------------------------------------------------------------------

SIGN_IN_SCENARIO = "1. Click the **Sign in** button.\n"
SIGN_IN_TARGET = "Sign in button"


def _parsed_sign_in(intent: ReportedIntent | None) -> ExpectedTrace:
    step = _LLMStep(
        action=ActionType.click,
        target=SIGN_IN_TARGET,
        raw_text="Click the **Sign in** button.",
        intent=intent,
    )
    return _build_expected_trace(
        _LLMTrace(scenario_title="t", steps=[step]), scenario_text=SIGN_IN_SCENARIO
    )


def _sign_in_reported(**overrides: str) -> ReportedIntent:
    fields = {
        "operation": "activate",
        "operation_text": "Click",
        "kind": "button",
        "kind_text": "button",
        "name": "Sign in",
    }
    fields.update(overrides)
    return ReportedIntent(**fields)


def _button_observation(name: str, *, with_interaction: bool = True) -> ObservedTrace:
    target = f"{name} button"
    interaction = _interaction(
        target,
        operation=IntentOperation.activate,
        kind=ElementKind.button,
        kind_text="button",
        name=name,
    )
    return ObservedTrace(
        source_file="r.mp4",
        actions=[_obs(1, 1000, target, interaction if with_interaction else None)],
    )


def _apply_base(trace: ExpectedTrace, observed: ObservedTrace) -> AuditResult:
    return apply_base_equivalence(
        trace,
        observed,
        _failed_audit(trace.steps, observed.actions),
        scenario_text=SIGN_IN_SCENARIO,
    )


def test_parsed_intent_reaches_the_engine_and_commits_on_observed_evidence():
    trace = _parsed_sign_in(_sign_in_reported())

    result = _apply_base(trace, _button_observation("Sign in"))

    assert (result.equivalence_decisions[0].status, result.step_verdicts[0].passed) == (
        "committed", True,
    )


def test_parsed_intent_with_capitalised_operation_reaches_the_engine():
    trace = _parsed_sign_in(_sign_in_reported(operation="Activate", kind="Button"))

    result = _apply_base(trace, _button_observation("Sign in"))

    assert result.step_verdicts[0].passed is True


def test_parsed_intent_reaches_the_engine_for_evaluation():
    trace = _parsed_sign_in(_sign_in_reported())
    observation = _button_observation("Sign in").actions[0]

    match = evaluate(
        trace.steps[0], observation, scenario_text=SIGN_IN_SCENARIO,
        fence=(None, None), claimed=set(),
    )

    assert getattr(match, "evidence_id", None) == "O1"


def test_valid_intent_alone_does_not_pass_a_step_without_observed_interaction():
    trace = _parsed_sign_in(_sign_in_reported())

    result = _apply_base(trace, _button_observation("Sign in", with_interaction=False))

    assert result.step_verdicts[0].passed is False


def test_valid_intent_alone_does_not_pass_a_step_observed_on_another_control():
    trace = _parsed_sign_in(_sign_in_reported())

    result = _apply_base(trace, _button_observation("Log out"))

    assert result.step_verdicts[0].passed is False


def test_valid_intent_alone_does_not_pass_a_step_with_no_observed_action():
    trace = _parsed_sign_in(_sign_in_reported())
    nothing = ObservedTrace(source_file="r.mp4", actions=[])

    result = align(trace, nothing)

    assert result.step_verdicts[0].passed is False


def test_rejected_intent_never_creates_a_pass():
    trace = _parsed_sign_in(_sign_in_reported(kind="website"))
    audit = _failed_audit(trace.steps, [])

    result = apply_base_equivalence(
        trace, _button_observation("Sign in"), audit, scenario_text=SIGN_IN_SCENARIO
    )

    assert (trace.steps[0].action_intent, result.step_verdicts[0].passed) == (None, False)


def test_missing_intent_never_creates_a_pass():
    trace = _parsed_sign_in(None)
    audit = _failed_audit(trace.steps, [])

    result = apply_base_equivalence(
        trace, _button_observation("Sign in"), audit, scenario_text=SIGN_IN_SCENARIO
    )

    assert (trace.steps[0].action_intent, result.step_verdicts[0].passed) == (None, False)


# ---------------------------------------------------------------------------
# Stage 2 interactions bound by the extractor reaching the engine
# ---------------------------------------------------------------------------

CLICK_SCENARIO = "Click the Sign in button."


def _click_step() -> ExpectedStep:
    intent = bind_expected_intent(
        CLICK_SCENARIO,
        sentence_start=0,
        sentence_end=len(CLICK_SCENARIO),
        operation="activate",
        operation_text="Click",
        kind="button",
        kind_text="button",
        name="Sign in",
    )
    return ExpectedStep(index=1, action=ActionType.click, target="Sign in button", action_intent=intent)


def _extracted(target: str, action: ActionType = ActionType.click, **fields: str) -> ObservedAction:
    reported = {
        "operation": "activate",
        "kind": "button",
        "kind_text": "button",
        "surface": "target",
    }
    reported.update(fields)
    outcome = bind_reported_interaction(
        ReportedInteraction(**{f"interaction_{k}": v for k, v in reported.items()}),
        action=action,
        surfaces={"target": target, "input_value": None, "visible_text": None},
        frame_index=1,
    )
    return ObservedAction(
        index=1,
        timestamp_ms=1000,
        action=action,
        target=target,
        confidence=0.9,
        interaction=outcome.interaction,
        interaction_diagnostic=outcome.diagnostic,
    )


def _committed(observation: ObservedAction) -> bool:
    step = _click_step()
    trace = ExpectedTrace(scenario_title="t", source_text=CLICK_SCENARIO, steps=[step])
    observed = ObservedTrace(source_file="r.mp4", actions=[observation])
    result = apply_base_equivalence(
        trace, observed, _failed_audit([step], []), scenario_text=CLICK_SCENARIO
    )
    return result.step_verdicts[0].passed


def test_stage1_intent_commits_on_a_grounded_stage2_interaction():
    assert _committed(_extracted("Sign in button", name="Sign in")) is True


def test_stage1_intent_commits_on_a_stage2_interaction_with_normalised_casing():
    assert _committed(_extracted("Sign in button", name="Sign in", operation="Activate", kind="Button"))


def test_stage1_intent_rejects_a_stage2_interaction_on_another_control():
    assert _committed(_extracted("Log out button", name="Log out")) is False


def test_stage1_intent_rejects_a_stage2_interaction_of_an_incompatible_kind():
    assert _committed(_extracted("Sign in link", name="Sign in", kind="link", kind_text="link")) is False


def test_stage1_intent_rejects_a_stage2_dismissal_of_the_same_control():
    assert _committed(_extracted("Sign in button", name="Sign in", operation="dismiss")) is False


def test_rejected_stage2_interaction_never_creates_a_pass():
    observation = _extracted("Sign in", name="Sign in")

    assert observation.interaction is None
    assert _committed(observation) is False


def test_stage2_interaction_contradicting_its_action_never_creates_a_pass():
    observation = _extracted("Sign in button", ActionType.hover, name="Sign in")

    assert observation.interaction is None
    assert _committed(observation) is False


def test_omitted_stage2_interaction_never_creates_a_pass():
    observation = ObservedAction(
        index=1, timestamp_ms=1000, action=ActionType.click, target="Sign in button",
        confidence=0.9,
    )

    assert _committed(observation) is False
