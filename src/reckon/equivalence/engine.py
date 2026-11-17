"""Deterministic equivalence. Base commits before any Agent proposal is considered."""

from reckon.equivalence.grounding import citation_inside, citation_matches, slices_adjacent
from reckon.equivalence.table import TABLE_VERSION, matching_rule
from reckon.schema import (
    ActionType,
    AuditResult,
    EquivalenceDecision,
    EquivalenceMatch,
    ExpectedActionIntent,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    IntentOperation,
    ObjectKind,
    ObservedAction,
    ObservedInteraction,
    ObservedTrace,
    SlotCitation,
    StepVerdict,
)
from reckon.tolerance import finding_severity, verdict_for_steps

_ACTION_OPERATION: dict[ActionType, IntentOperation] = {
    ActionType.click: IntentOperation.activate,
    ActionType.type_text: IntentOperation.enter,
    ActionType.select: IntentOperation.choose,
    ActionType.navigate: IntentOperation.traverse,
    ActionType.assert_visible: IntentOperation.assert_,
    ActionType.assert_text: IntentOperation.assert_,
    ActionType.assert_state: IntentOperation.assert_,
}

_NAME_KINDS = {
    ObjectKind.application,
    ObjectKind.control,
    ObjectKind.page,
    ObjectKind.dialog,
}

_ACTION_FINDINGS = {FindingType.missing_action, FindingType.wrong_target}


def evaluate(
    step: ExpectedStep,
    observation: ObservedAction,
    *,
    scenario_text: str,
    fence: tuple[int | None, int | None],
    claimed: set[int],
) -> EquivalenceMatch | str:
    """Return a match, or one rejection reason. Does not read confidence or screen text."""
    intent = step.action_intent
    interaction = observation.interaction
    if intent is None or interaction is None:
        return "structured_fields_absent"
    if intent.schema_version != "1" or interaction.schema_version != "1":
        return "structured_fields_absent"
    reason = _citations_exact(intent, interaction, scenario_text, observation)
    if reason is not None:
        return reason
    if not _operation_consistent(step.action, intent.operation):
        return "contradicted_slot"
    if not _required_identity(intent):
        return "missing_identity"
    rule = matching_rule(
        intent.operation,
        intent.object.kind,
        interaction.operation,
        interaction.element.kind,
    )
    if rule is None:
        return "type_not_realized"
    if not _identity_equal(intent, interaction):
        return "identity_mismatch"
    previous, upcoming = fence
    if previous is not None and observation.timestamp_ms <= previous:
        return "outside_fence"
    if upcoming is not None and observation.timestamp_ms > upcoming:
        return "outside_fence"
    if observation.index in claimed:
        return "observation_claimed"
    compared = _compared_slots(intent)
    return EquivalenceMatch(
        relation_id=f"{rule.rule_id}@{TABLE_VERSION}",
        expected_step_index=step.index,
        terminal_observed_index=observation.index,
        evidence_timestamp_ms=observation.timestamp_ms,
        evidence_id=f"O{observation.index}",
        cited_frame_index=interaction.cited_frame_index,
        evidence_kind="action",
        compared_slots=compared,
        context_episode_id=interaction.context_episode_id,
    )


def apply_base_equivalence(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    *,
    scenario_text: str,
) -> AuditResult:
    """Commit Base matches. Steps with no structured intent are left untouched."""
    if not any(step.action_intent is not None for step in expected.steps):
        return audit
    decisions, matches = _base_assignment(expected, observed, audit, scenario_text)
    if not matches:
        return audit.model_copy(update={"equivalence_decisions": decisions})
    return _apply_matches(expected, observed, audit, decisions, matches)


def apply_agent_equivalence(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    proposals: dict[int, list[int]],
    *,
    scenario_text: str,
) -> AuditResult:
    """Consider Agent indexes only for steps Base left at ``no_candidate``."""
    if not audit.equivalence_decisions and not proposals:
        return audit
    by_step = {item.expected_step_index: item for item in audit.equivalence_decisions}
    committed = {
        item.expected_step_index
        for item in audit.equivalence_decisions
        if item.status == "committed"
    }
    claimed = _claimed(audit)
    fences = _fences(audit)
    actions = {action.index: action for action in observed.actions}
    steps = {step.index: step for step in expected.steps}
    updated: dict[int, EquivalenceDecision] = {}
    matches: dict[int, EquivalenceMatch] = {}
    open_hits: dict[int, list[EquivalenceMatch]] = {}

    for step_index, indexes in sorted(proposals.items()):
        current = by_step.get(step_index)
        if current is not None and current.status == "committed":
            updated[step_index] = current.model_copy(update={
                "agent_status": "ignored_after_base_commit",
            })
            continue
        if current is None or current.status != "no_candidate":
            if current is not None:
                updated[step_index] = current.model_copy(update={
                    "agent_status": "ignored_after_base_commit",
                    "agent_rejection": current.status,
                })
            continue
        verified: list[EquivalenceMatch] = []
        rejection: str | None = None
        for index in indexes:
            action = actions.get(index)
            step = steps.get(step_index)
            if action is None or step is None:
                rejection = "structured_fields_absent"
                continue
            result = evaluate(
                step,
                action,
                scenario_text=scenario_text,
                fence=fences.get(step_index, (None, None)),
                claimed=claimed,
            )
            if isinstance(result, EquivalenceMatch):
                verified.append(result)
            else:
                rejection = result
        open_hits[step_index] = verified
        if len(verified) != 1:
            updated[step_index] = current.model_copy(update={
                "agent_status": "unmatched",
                "agent_rejection": rejection if len(verified) == 0 else "ambiguous_candidates",
            })

    owners: dict[int, list[int]] = {}
    for step_index, hits in open_hits.items():
        if len(hits) == 1:
            owners.setdefault(hits[0].terminal_observed_index, []).append(step_index)

    last_ts: int | None = None
    for step_index in sorted(open_hits):
        hits = open_hits[step_index]
        current = by_step[step_index]
        if len(hits) != 1:
            continue
        match = hits[0]
        if len(owners.get(match.terminal_observed_index, [])) != 1:
            updated[step_index] = current.model_copy(update={
                "agent_status": "unmatched",
                "agent_rejection": "contested_observation",
            })
            continue
        if last_ts is not None and match.evidence_timestamp_ms < last_ts:
            updated[step_index] = current.model_copy(update={
                "agent_status": "unmatched",
                "agent_rejection": "chronology_violation",
            })
            continue
        if not _within_neighbor_commit(audit, step_index, match.evidence_timestamp_ms):
            updated[step_index] = current.model_copy(update={
                "agent_status": "unmatched",
                "agent_rejection": "chronology_violation",
            })
            continue
        updated[step_index] = current.model_copy(update={
            "agent_status": "committed",
            "agent_observed_index": match.terminal_observed_index,
        })
        matches[step_index] = match
        last_ts = match.evidence_timestamp_ms

    decisions = [
        updated.get(item.expected_step_index, item)
        for item in audit.equivalence_decisions
    ]
    if not matches:
        return audit.model_copy(update={"equivalence_decisions": decisions})
    return _apply_matches(expected, observed, audit, decisions, matches)


def _base_assignment(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    scenario_text: str,
) -> tuple[list[EquivalenceDecision], dict[int, EquivalenceMatch]]:
    fences = _fences(audit)
    claimed = _claimed(audit)
    verdicts = {item.expected_step_index: item for item in audit.step_verdicts}
    candidates: dict[int, list[EquivalenceMatch]] = {}
    for step in expected.steps:
        verdict = verdicts.get(step.index)
        if step.action_intent is None or verdict is None or verdict.passed:
            continue
        hits: list[EquivalenceMatch] = []
        for action in observed.actions:
            result = evaluate(
                step,
                action,
                scenario_text=scenario_text,
                fence=fences.get(step.index, (None, None)),
                claimed=claimed,
            )
            if isinstance(result, EquivalenceMatch):
                hits.append(result)
        candidates[step.index] = hits

    unique: dict[int, EquivalenceMatch] = {}
    decisions: dict[int, EquivalenceDecision] = {}
    for step_index, hits in candidates.items():
        if len(hits) == 0:
            decisions[step_index] = EquivalenceDecision(
                expected_step_index=step_index,
                status="no_candidate",
            )
        elif len(hits) > 1:
            decisions[step_index] = EquivalenceDecision(
                expected_step_index=step_index,
                status="ambiguous_candidates",
            )
        else:
            unique[step_index] = hits[0]

    pointed: dict[int, list[int]] = {}
    for step_index, match in unique.items():
        pointed.setdefault(match.terminal_observed_index, []).append(step_index)
    contested = {
        step_index
        for indexes in pointed.values()
        if len(indexes) > 1
        for step_index in indexes
    }
    for step_index in contested:
        match = unique.pop(step_index)
        decisions[step_index] = EquivalenceDecision(
            expected_step_index=step_index,
            status="contested_observation",
            observed_index=match.terminal_observed_index,
        )

    matches: dict[int, EquivalenceMatch] = {}
    last_ts: int | None = None
    for step_index in sorted(unique):
        match = unique[step_index]
        if last_ts is not None and match.evidence_timestamp_ms < last_ts:
            decisions[step_index] = EquivalenceDecision(
                expected_step_index=step_index,
                status="chronology_violation",
                observed_index=match.terminal_observed_index,
            )
            continue
        decisions[step_index] = EquivalenceDecision(
            expected_step_index=step_index,
            status="committed",
            observed_index=match.terminal_observed_index,
            relation_id=match.relation_id,
        )
        matches[step_index] = match
        last_ts = match.evidence_timestamp_ms
    ordered = [decisions[index] for index in sorted(decisions)]
    return ordered, matches


def _citations_exact(
    intent: ExpectedActionIntent,
    interaction: ObservedInteraction,
    scenario: str,
    observation: ObservedAction,
) -> str | None:
    if intent.sentence_end > len(scenario):
        return "bad_citation"
    surface = _surface_text(observation, interaction.citation_surface.value)
    if surface is None:
        return "bad_citation"
    if intent.operation_citation is None or intent.object.kind_citation is None:
        return "bad_citation"
    expected_pairs: list[tuple[str | None, SlotCitation | None]] = [
        (None, intent.operation_citation),
        (None, intent.object.kind_citation),
        (intent.object.name, intent.object.name_citation),
        (intent.object.product, intent.object.product_citation),
        (intent.object.instance, intent.object.instance_citation),
        (intent.object.value, intent.object.value_citation),
    ]
    for value, citation in expected_pairs:
        if citation is None:
            if value is not None:
                return "bad_citation"
            continue
        if value is not None and value != citation.text:
            return "bad_citation"
        if not citation_inside(citation, intent.sentence_start, intent.sentence_end):
            return "bad_citation"
        if not citation_matches(scenario, citation):
            return "bad_citation"
    name = intent.object.name_citation
    kind = intent.object.kind_citation
    if name is not None and not slices_adjacent(scenario, name, kind):
        return "bad_citation"
    observed_pairs: list[tuple[str | None, SlotCitation | None]] = [
        (None, interaction.operation_citation),
        (None, interaction.element.kind_citation),
        (interaction.element.name, interaction.element.name_citation),
        (interaction.element.product, interaction.element.product_citation),
        (interaction.element.instance, interaction.element.instance_citation),
        (interaction.element.value, interaction.element.value_citation),
    ]
    if interaction.element.kind_citation is None:
        return "bad_citation"
    for value, citation in observed_pairs:
        if citation is None:
            if value is not None:
                return "bad_citation"
            continue
        if value is not None and value != citation.text:
            return "bad_citation"
        if not citation_matches(surface, citation):
            return "bad_citation"
    return None


def _operation_consistent(action: ActionType, operation: IntentOperation) -> bool:
    expected = _ACTION_OPERATION.get(action)
    if expected is None:
        return True
    return expected == operation


def _required_identity(intent: ExpectedActionIntent) -> bool:
    obj = intent.object
    if obj.kind in _NAME_KINDS and not obj.name:
        return False
    if obj.kind in {ObjectKind.field, ObjectKind.option} and obj.value is not None:
        return bool(obj.value_citation is not None and obj.value == obj.value_citation.text)
    if obj.kind in {ObjectKind.field, ObjectKind.option} and not obj.name and obj.value is None:
        return False
    return True


def _identity_equal(intent: ExpectedActionIntent, interaction: ObservedInteraction) -> bool:
    expected = intent.object
    observed = interaction.element
    if expected.name is not None and observed.name != expected.name:
        return False
    if expected.product is not None and observed.product != expected.product:
        return False
    if expected.instance is not None and observed.instance != expected.instance:
        return False
    if expected.value is not None and observed.value != expected.value:
        return False
    return True


def _compared_slots(intent: ExpectedActionIntent) -> list[str]:
    slots = ["operation", "kind"]
    obj = intent.object
    if obj.name is not None:
        slots.append("name")
    if obj.product is not None:
        slots.append("product")
    if obj.instance is not None:
        slots.append("instance")
    if obj.value is not None:
        slots.append("value")
    return slots


def _surface_text(observation: ObservedAction, surface: str) -> str | None:
    if surface == "target":
        return observation.target
    if surface == "input_value":
        return observation.input_value
    if surface == "visible_text" and observation.assertion_evidence is not None:
        return observation.assertion_evidence.visible_text
    return None


def _claimed(audit: AuditResult) -> set[int]:
    return {
        verdict.matched_observed_index
        for verdict in audit.step_verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    }


def _fences(audit: AuditResult) -> dict[int, tuple[int | None, int | None]]:
    upcoming: int | None = None
    nxt: dict[int, int | None] = {}
    for verdict in reversed(audit.step_verdicts):
        nxt[verdict.expected_step_index] = upcoming
        if verdict.passed and verdict.evidence_timestamp_ms is not None:
            upcoming = verdict.evidence_timestamp_ms
    previous: int | None = None
    fences: dict[int, tuple[int | None, int | None]] = {}
    for verdict in audit.step_verdicts:
        fences[verdict.expected_step_index] = (previous, nxt.get(verdict.expected_step_index))
        if verdict.passed and verdict.evidence_timestamp_ms is not None:
            previous = verdict.evidence_timestamp_ms
    return fences


def _within_neighbor_commit(audit: AuditResult, step_index: int, timestamp: int) -> bool:
    previous, upcoming = _fences(audit).get(step_index, (None, None))
    if previous is not None and timestamp <= previous:
        return False
    if upcoming is not None and timestamp > upcoming:
        return False
    return True


def _apply_matches(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    decisions: list[EquivalenceDecision],
    matches: dict[int, EquivalenceMatch],
) -> AuditResult:
    actions = {action.index: action for action in observed.actions}
    released: set[int] = set()
    verdicts: list[StepVerdict] = []
    for verdict in audit.step_verdicts:
        match = matches.get(verdict.expected_step_index)
        if match is None:
            verdicts.append(verdict)
            continue
        if verdict.matched_observed_index not in (None, match.terminal_observed_index):
            released.add(verdict.matched_observed_index)
        remaining = [
            finding
            for finding in verdict.finding_types
            if finding not in _ACTION_FINDINGS
        ]
        action = actions[match.terminal_observed_index]
        verdicts.append(verdict.model_copy(update={
            "passed": not remaining,
            "matched_observed_index": match.terminal_observed_index,
            "evidence_timestamp_ms": action.timestamp_ms,
            "finding_types": remaining,
            "recovery_reason": match.relation_id,
            "evidence_id": match.evidence_id,
            "frame_ids": [f"F{match.cited_frame_index}"],
            "explanation": (
                f"Observed action {match.evidence_id} at {action.timestamp_ms} ms "
                f"realizes this step ({match.relation_id})."
            ),
        }))
    taken = {match.terminal_observed_index for match in matches.values()}
    findings = [
        finding
        for finding in audit.findings
        if finding.expected_step_index not in matches
        or finding.type not in _ACTION_FINDINGS
    ]
    findings = [
        finding
        for finding in findings
        if not (
            finding.type == FindingType.extra_action
            and finding.observed_action_index in taken
        )
    ]
    used = _claimed(audit) | taken
    for index in sorted(released):
        if index in used:
            continue
        action = actions.get(index)
        if action is None:
            continue
        findings.append(Finding(
            type=FindingType.extra_action,
            severity=finding_severity(FindingType.extra_action, None),
            expected_step_index=None,
            observed_action_index=index,
            evidence_timestamp_ms=action.timestamp_ms,
            description=(
                f"Observed action {index} ({action.action.value} on {action.target!r}) "
                "was released by action equivalence."
            ),
        ))
    passed = sum(1 for verdict in verdicts if verdict.passed)
    total = len(expected.steps)
    return audit.model_copy(update={
        "verdict": verdict_for_steps(expected.steps, findings, verdicts),
        "fidelity_score": (passed / total) if total else 1.0,
        "findings": findings,
        "step_verdicts": verdicts,
        "equivalence_decisions": decisions,
    })
