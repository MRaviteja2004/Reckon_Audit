"""Replace a failed early alignment with one later observation that fully supports the step."""

from reckon.aligner import (
    _dismissal_click_supports_target,
    _order_navigation_supports_open,
)
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
    """Use a later clean observation when the paired one does not support the step.

    A replacement has to be the only later observation that independently
    supports the step, after the previous passed step and before the next one.
    Two supporters stay unresolved. The new verdict uses that observation's
    index, timestamp, and explanation together.
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
            if _later_support(action, verdict, used, floor, limit)
            and _validated_support(step, action)
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
            recovery_reason="later_observed_action",
            evidence_id=f"O{action.index}",
            explanation=(
                f"Observed action O{action.index} at {action.timestamp_ms} ms "
                "supports this step."
            ),
        ))
    if not replaced:
        return audit.model_copy(update={
            "verdict": verdict_for_steps(expected.steps, audit.findings, audit.step_verdicts),
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


def _validated_support(step: ExpectedStep, action: ObservedAction) -> bool:
    """The later action has to satisfy the step on its own recorded evidence."""
    if (
        step.input_value is not None
        and action.input_value is not None
        and not values_equivalent(step.input_value, action.input_value)
    ):
        return False
    if step.assertion is not None:
        return assertion_satisfied(
            step.assertion,
            _without_description(action),
            expected_target=step.target,
        )
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


def _later_support(
    action: ObservedAction,
    verdict: StepVerdict,
    used: set[int | None],
    floor: int,
    limit: int | None,
) -> bool:
    if action.index in used or action.index == verdict.matched_observed_index:
        return False
    if action.timestamp_ms < floor:
        return False
    if (
        verdict.evidence_timestamp_ms is not None
        and action.timestamp_ms <= verdict.evidence_timestamp_ms
    ):
        return False
    return limit is None or action.timestamp_ms < limit
