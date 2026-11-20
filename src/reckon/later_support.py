"""Replace a failed alignment with the one observation that fully supports the step.

The search runs between the previous passed step and the next one. A failed
pair does not move either bound. Evidence before that pair is recorded as
fenced_observed_action; evidence after it stays later_observed_action.
"""

from reckon.aligner import (
    _dismissal_click_supports_target,
    _order_navigation_supports_open,
)
from reckon.control_semantics import (
    choice_identity_conflicts,
    choice_pair_supports_step,
    is_targetless_dismiss,
    state_contradicts_instruction,
    targetless_dismissal_supported,
)
from reckon.ownership import drop_shared_citations
from reckon.page_context import opening_navigation_target
from reckon.schema import (
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)
from reckon.tolerance import (
    _contains_assertion,
    _introduces_negation,
    actions_compatible,
    assertion_satisfied,
    finding_severity,
    values_equivalent,
    verdict_for_steps,
)


def recover_later_support(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
) -> AuditResult:
    """Use the one in-fence observation that independently supports a failed step.

    The window is after the previous passed step and before the next one.
    The failed pair's own timestamp is not a bound. Zero supporters, or more
    than one, leave the step unmatched. The replacement keeps that
    observation's index, timestamp, and explanation.
    """
    steps = {step.index: step for step in expected.steps}
    used = {
        verdict.matched_observed_index
        for verdict in audit.step_verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    }
    next_limit: dict[int, int | None] = {}
    upcoming: int | None = None
    for verdict in reversed(audit.step_verdicts):
        next_limit[verdict.expected_step_index] = upcoming
        if verdict.passed and verdict.evidence_timestamp_ms is not None:
            upcoming = verdict.evidence_timestamp_ms

    floor = -1
    verdicts: list[StepVerdict] = []
    replaced: dict[int, ObservedAction] = {}
    released: set[int] = set()
    for verdict in audit.step_verdicts:
        step = steps[verdict.expected_step_index]
        if verdict.passed:
            if verdict.evidence_timestamp_ms is not None:
                floor = max(floor, verdict.evidence_timestamp_ms)
            verdicts.append(verdict)
            continue
        limit = next_limit.get(verdict.expected_step_index)
        supporters = [
            action
            for action in observed.actions
            if _within_passed_fence(action, verdict, used, floor, limit)
            and _validated_support(step, action, observed.actions, floor_ms=floor)
        ]
        if len(supporters) != 1:
            verdicts.append(verdict)
            continue
        action = supporters[0]
        used.add(action.index)
        floor = max(floor, action.timestamp_ms)
        if verdict.matched_observed_index is not None:
            released.add(verdict.matched_observed_index)
        replaced[step.index] = action
        verdicts.append(StepVerdict(
            expected_step_index=step.index,
            passed=True,
            matched_observed_index=action.index,
            evidence_timestamp_ms=action.timestamp_ms,
            finding_types=[],
            recovery_reason=_recovery_reason(verdict, action),
            evidence_id=f"O{action.index}",
            explanation=(
                f"Observed action O{action.index} at {action.timestamp_ms} ms "
                "supports this step."
            ),
        ))
    if not replaced:
        verdicts = drop_shared_citations(audit.step_verdicts)
        return audit.model_copy(update={
            "step_verdicts": verdicts,
            "verdict": verdict_for_steps(expected.steps, audit.findings, verdicts),
        })

    taken = {action.index for action in replaced.values()}
    findings = [
        finding
        for finding in audit.findings
        if finding.expected_step_index not in replaced
        and not (
            finding.type == FindingType.extra_action
            and finding.observed_action_index in taken
        )
    ]
    actions = {action.index: action for action in observed.actions}
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
                "was not in the expected scenario."
            ),
        ))
    passed = sum(1 for verdict in verdicts if verdict.passed)
    total = len(verdicts)
    verdicts = drop_shared_citations(verdicts)
    return audit.model_copy(update={
        "step_verdicts": verdicts,
        "findings": findings,
        "fidelity_score": (passed / total) if total else 1.0,
        "verdict": verdict_for_steps(expected.steps, findings, verdicts),
    })


def _without_description(action: ObservedAction) -> ObservedAction:
    """Description text is the model's account, so it cannot recover a step."""
    evidence = action.assertion_evidence
    if evidence is None or not evidence.description:
        return action
    return action.model_copy(update={
        "assertion_evidence": evidence.model_copy(update={"description": None}),
    })


def _assertion_text_supports(
    step: ExpectedStep,
    action: ObservedAction,
    context_target: str | None,
) -> bool:
    """The asserted words, or the asserted screen, are in recorded visible text.

    A navigate target or a model description alone is not assertion evidence,
    so it cannot recover the step.
    """
    checked = _without_description(action)
    evidence = checked.assertion_evidence
    if step.assertion is None or evidence is None or not evidence.visible_text:
        return False
    if _introduces_negation(step.assertion, evidence.visible_text):
        return False
    return assertion_satisfied(
        step.assertion,
        checked,
        expected_target=step.target,
        context_target=context_target,
    )


def _validated_support(
    step: ExpectedStep,
    action: ObservedAction,
    observed_actions: list[ObservedAction] | None = None,
    *,
    floor_ms: int = -1,
) -> bool:
    """The action has to satisfy the step on its own recorded evidence."""
    if is_targetless_dismiss(step):
        if (
            step.input_value is not None
            and action.input_value is not None
            and not values_equivalent(step.input_value, action.input_value)
        ):
            return False
        return targetless_dismissal_supported(step, action)
    if (
        step.input_value is not None
        and action.input_value is not None
        and not values_equivalent(step.input_value, action.input_value)
    ):
        return False
    if step.assertion is not None:
        context_target = (
            opening_navigation_target(observed_actions, action, floor_ms=floor_ms)
            if observed_actions is not None
            else None
        )
        return _assertion_text_supports(step, action, context_target)
    if state_contradicts_instruction(step, action):
        return False
    if choice_identity_conflicts(step, action):
        return False
    if choice_pair_supports_step(step, action, observed_actions=observed_actions):
        return True
    compatible = (
        actions_compatible(step.action, action.action)
        or _order_navigation_supports_open(step, action)
        or _dismissal_click_supports_target(step, action)
    )
    if not compatible:
        return False
    if (
        _order_navigation_supports_open(step, action)
        or _dismissal_click_supports_target(step, action)
    ):
        return True
    if step.target is None:
        return True
    if action.target is None:
        return False
    return _contains_assertion(step.target, action.target)


def _within_passed_fence(
    action: ObservedAction,
    verdict: StepVerdict,
    used: set[int | None],
    floor: int,
    limit: int | None,
) -> bool:
    """Free observation between the previous passed step and the next one.

    The failed pair's timestamp is not a bound: that pair is the alignment's
    diagnosis, and a valid occurrence may sit earlier. The observation the
    pair already cites is not a second occurrence of this step.
    """
    if action.index in used or action.index == verdict.matched_observed_index:
        return False
    if action.timestamp_ms < floor:
        return False
    return limit is None or action.timestamp_ms < limit


def _recovery_reason(verdict: StepVerdict, action: ObservedAction) -> str:
    """Evidence before the failed pair is fenced; evidence after it is later."""
    if (
        verdict.evidence_timestamp_ms is not None
        and action.timestamp_ms < verdict.evidence_timestamp_ms
    ):
        return "fenced_observed_action"
    return "later_observed_action"
