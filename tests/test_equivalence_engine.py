"""Predicate engine: citations, assignment, and legacy compatibility."""

import pytest
from pydantic import ValidationError

from reckon.aligner import align
from reckon.assist import align_with_assist
from reckon.parser import _LLMIntent, _LLMStep, _LLMTrace, _build_expected_trace
from reckon.equivalence.citations import bind_expected_intent, bind_observed_interaction
from reckon.equivalence.engine import apply_agent_equivalence, apply_base_equivalence, evaluate
from reckon.equivalence.table import TABLE_VERSION, TYPE_REALIZATIONS
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditResult,
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
    intent = _LLMIntent(
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
