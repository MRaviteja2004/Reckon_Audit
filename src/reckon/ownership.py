"""One observation, one owner: reconcile verdict citations with extra-action findings.

A passed verdict owns the observation it cites. A failed verdict may keep its
diagnostic finding but must not cite an owned observation, and an owned
observation is not an extra action. An observation a step stopped citing is an
extra action again unless something else still accounts for it.
"""

from collections.abc import Collection

from reckon.schema import (
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    ObservedTrace,
    StepVerdict,
)
from reckon.tolerance import ASSERTION_ACTIONS, finding_severity

OWNED_BY_ANOTHER_STEP = "observation_owned_by_another_step"


def claimed_observations(verdicts: list[StepVerdict]) -> set[int]:
    """Observation indexes cited by verdicts that passed."""
    return {
        verdict.matched_observed_index
        for verdict in verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    }


def drop_shared_citations(verdicts: list[StepVerdict]) -> list[StepVerdict]:
    """Stop a failed step citing an observation a passed step already owns.

    The finding keeps the diagnostic pair. Only the verdict's evidence
    citation is cleared, and only when that observation was claimed.
    """
    claimed = claimed_observations(verdicts)
    if not claimed:
        return verdicts
    cleaned: list[StepVerdict] = []
    for verdict in verdicts:
        index = verdict.matched_observed_index
        if verdict.passed or index is None or index not in claimed:
            cleaned.append(verdict)
            continue
        cleaned.append(verdict.model_copy(update={
            "matched_observed_index": None,
            "evidence_id": None,
        }))
    return cleaned


def drop_shared_finding_citations(
    findings: list[Finding],
    verdicts: list[StepVerdict],
) -> list[Finding]:
    """Stop a failed step's finding citing an observation another passed step owns.

    The finding stays, with its type and description: the step really did fail.
    Only the observation reference is cleared, so an owned observation is never
    offered as the evidence for a different step's failure.
    """
    owners = {
        verdict.matched_observed_index: verdict.expected_step_index
        for verdict in verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    }
    if not owners:
        return findings
    cleaned: list[Finding] = []
    for finding in findings:
        owner = owners.get(finding.observed_action_index)
        if (
            finding.type == FindingType.extra_action
            or finding.expected_step_index is None
            or owner is None
            or owner == finding.expected_step_index
        ):
            cleaned.append(finding)
            continue
        cleaned.append(finding.model_copy(update={"observed_action_index": None}))
    return cleaned


def with_consistent_citations(audit: AuditResult) -> AuditResult:
    """The audit with verdict and finding citations of owned observations cleared."""
    verdicts = drop_shared_citations(audit.step_verdicts)
    findings = drop_shared_finding_citations(audit.findings, verdicts)
    return audit.model_copy(update={"step_verdicts": verdicts, "findings": findings})


def _proposal_claim(
    record: AssistProposalRecord,
    step: ExpectedStep | None,
    verdict: StepVerdict | None,
) -> int | None:
    """The observation the step would pass on once this proposal is applied."""
    baseline = verdict.matched_observed_index if verdict is not None else None
    if record.kind == AssistCandidateKind.episode_verification:
        return (
            record.supporting_observed_index
            or record.completion_observed_index
            or record.matched_observed_index
        )
    if record.kind in (
        AssistCandidateKind.delayed_assertion,
        AssistCandidateKind.evidence_bundle,
    ):
        if record.challenged_false_pass:
            return None
        outcome_step = step is not None and (
            step.assertion is not None or step.action in ASSERTION_ACTIONS
        )
        if outcome_step and record.supporting_observed_index is not None:
            return record.supporting_observed_index
        return baseline
    if record.kind == AssistCandidateKind.wrong_target:
        return baseline
    if record.kind == AssistCandidateKind.frame_evidence:
        return record.supporting_observed_index or record.matched_observed_index
    return None


def as_owned_conflict(record: AssistProposalRecord) -> AssistProposalRecord:
    """The rejected form of a proposal whose observation another step owns."""
    return record.model_copy(update={
        "status": AssistProposalStatus.rejected,
        "reason": OWNED_BY_ANOTHER_STEP,
        "matching_method": AssistMatchMethod.rejected_proposal,
    })


def split_conflicting_claims(
    deterministic: AuditResult,
    accepted: list[AssistProposalRecord],
    expected: ExpectedTrace,
) -> tuple[list[AssistProposalRecord], list[AssistProposalRecord]]:
    """Separate proposals that would pass a step on another step's observation.

    Passed deterministic verdicts own their observations first; proposals then
    claim in order. Returns (kept, conflicting); see ``as_owned_conflict``.
    """
    steps = {step.index: step for step in expected.steps}
    verdicts = {verdict.expected_step_index: verdict for verdict in deterministic.step_verdicts}
    owners = {
        verdict.matched_observed_index: verdict.expected_step_index
        for verdict in deterministic.step_verdicts
        if verdict.passed and verdict.matched_observed_index is not None
    }
    kept: list[AssistProposalRecord] = []
    rejected: list[AssistProposalRecord] = []
    for record in accepted:
        step_index = record.expected_step_index
        claim = _proposal_claim(record, steps.get(step_index), verdicts.get(step_index))
        if claim is not None and owners.get(claim, step_index) != step_index:
            rejected.append(record)
            continue
        if claim is not None:
            owners[claim] = step_index
        kept.append(record)
    return kept, rejected


def reconcile_ownership(
    audit: AuditResult,
    observed: ObservedTrace,
    *,
    released: Collection[int] = (),
    covered: Collection[int] = (),
) -> AuditResult:
    """Make citations and extra-action findings agree with the passed verdicts.

    ``released`` are observations a step stopped citing. ``covered`` are
    observations inside an accepted episode span; they belong to that step even
    though no verdict cites them. Extra actions are advisory, so the verdict and
    score are unchanged.
    """
    verdicts = drop_shared_citations(audit.step_verdicts)
    owned = claimed_observations(verdicts) | set(covered)
    findings = [
        finding
        for finding in drop_shared_finding_citations(audit.findings, verdicts)
        if not (
            finding.type == FindingType.extra_action
            and finding.observed_action_index in owned
        )
    ]
    accounted = owned | {
        verdict.matched_observed_index
        for verdict in verdicts
        if verdict.matched_observed_index is not None
    } | {
        finding.observed_action_index
        for finding in findings
        if finding.type == FindingType.extra_action
    }
    actions = {action.index: action for action in observed.actions}
    for index in sorted(set(released) - accounted):
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
    return audit.model_copy(update={"step_verdicts": verdicts, "findings": findings})
