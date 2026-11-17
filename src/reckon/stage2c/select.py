"""Select expected steps that need Stage 2C frame-level evidence discovery."""

from dataclasses import dataclass

from reckon.schema import (
    ActionType,
    AuditResult,
    CandidateEvidenceType,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedTrace,
)
from reckon.tolerance import assertion_satisfied

# Finding types Stage 2C is allowed to attempt recovery for.
_RECOVERABLE = frozenset({
    FindingType.missing_action,
    FindingType.assertion_not_observed,
})

_ASSERTION_ACTIONS = frozenset({
    ActionType.assert_visible,
    ActionType.assert_text,
    ActionType.assert_state,
})


@dataclass(frozen=True)
class Stage2CSearchTarget:
    """One expected step to search for in labelled frames."""

    expected_step_id: str
    expected_step_index: int
    evidence_type: CandidateEvidenceType
    expected_action: str
    expected_text: str
    expected_assertion: str | None
    finding_types: tuple[FindingType, ...]


def _step_text(step: ExpectedStep) -> str:
    parts = [
        step.target or "",
        step.input_value or "",
        step.assertion or "",
        step.raw_text or "",
    ]
    return " ".join(part for part in parts if part).strip()


def _preferred_evidence_type(step: ExpectedStep) -> CandidateEvidenceType:
    if step.assertion is not None or step.action in _ASSERTION_ACTIONS:
        return CandidateEvidenceType.ASSERTION_OUTCOME
    return CandidateEvidenceType.ACTION


def select_stage2c_targets(
    expected: ExpectedTrace,
    deterministic: AuditResult,
    observed: ObservedTrace | None = None,
) -> list[Stage2CSearchTarget]:
    """Return unmatched / suspicious assertion steps eligible for frame evidence search.

    Selection is finding-driven and independent of whether DP paired a step to a
    plausible observation. Suspicious passes (assertion not satisfied on the
    matched observation) are included so drifted early matches still get Stage 2C.
    """
    finding_by_step: dict[int, list[FindingType]] = {}
    for finding in deterministic.findings:
        if finding.expected_step_index is None:
            continue
        if finding.type not in _RECOVERABLE:
            continue
        finding_by_step.setdefault(finding.expected_step_index, []).append(finding.type)

    action_by_index = {}
    if observed is not None:
        action_by_index = {action.index: action for action in observed.actions}

    targets: list[Stage2CSearchTarget] = []
    for step in expected.steps:
        types = list(finding_by_step.get(step.index, []))
        verdict = next(
            (
                item
                for item in deterministic.step_verdicts
                if item.expected_step_index == step.index
            ),
            None,
        )
        suspicious = False
        if (
            verdict is not None
            and verdict.passed
            and step.assertion is not None
            and verdict.matched_observed_index is not None
            and action_by_index
        ):
            matched = action_by_index.get(verdict.matched_observed_index)
            if matched is not None and not assertion_satisfied(
                step.assertion, matched, expected_target=step.target
            ):
                suspicious = True
                if FindingType.assertion_not_observed not in types:
                    types.append(FindingType.assertion_not_observed)
        if not types and not suspicious:
            continue
        unique = tuple(dict.fromkeys(types))
        evidence_type = _preferred_evidence_type(step)
        if FindingType.assertion_not_observed in unique or suspicious:
            evidence_type = CandidateEvidenceType.ASSERTION_OUTCOME
        elif FindingType.missing_action in unique and step.assertion is not None:
            # Unmatched assertion steps surface as missing_action from align().
            evidence_type = CandidateEvidenceType.ASSERTION_OUTCOME
        elif FindingType.missing_action in unique and step.assertion is None:
            evidence_type = CandidateEvidenceType.ACTION
        targets.append(
            Stage2CSearchTarget(
                expected_step_id=f"E{step.index}",
                expected_step_index=step.index,
                evidence_type=evidence_type,
                expected_action=step.action.value,
                expected_text=_step_text(step),
                expected_assertion=step.assertion,
                finding_types=unique,
            )
        )
    # Prefer unmatched / failed steps before suspicious passes (stable order by index,
    # with missing_action / assertion_not_observed findings sorted first within ties).
    targets.sort(
        key=lambda item: (
            0 if any(ft in _RECOVERABLE for ft in item.finding_types) else 1,
            item.expected_step_index,
        )
    )
    return targets
