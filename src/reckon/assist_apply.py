"""Apply validated Stage 3 assist proposals to a copy of the deterministic AuditResult."""

from reckon.schema import (
    AssistCandidateKind,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    CandidateEvidenceType,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    ObservedTrace,
    StepVerdict,
)
from reckon.tolerance import ASSERTION_ACTIONS, finding_severity, verdict_for_steps


def _clear_finding_types(
    finding_types: list[FindingType],
    to_clear: set[FindingType],
) -> list[FindingType]:
    return [ft for ft in finding_types if ft not in to_clear]


def _is_assertion_outcome_step(step: ExpectedStep | None) -> bool:
    """True when the step's obligation is verifying on-screen state, not a user action.

    Assertion evidence may recover ``missing_action`` only for these steps. It must
    never satisfy a genuine click/type/select/navigate requirement.
    """
    if step is None:
        return False
    if step.assertion is not None:
        return True
    return step.action in ASSERTION_ACTIONS


def _apply_assertion_outcome_recovery(
    *,
    step_index: int,
    step: ExpectedStep | None,
    support_index: int | None,
    support_timestamp_ms: int | None,
    confidence: float | None,
    clear_assertion: dict[int, int],
    clear_assertion_ts: dict[int, int],
    clear_missing: set[int],
    clear_missing_ts: dict[int, int],
    clear_wrong_target: set[int],
    remap_matched: dict[int, int],
    assist_meta: dict[int, float | None],
) -> None:
    """Shared ASSERTION_OUTCOME recovery used by evidence_bundle / delayed / Stage 2C.

    Clears assertion_not_observed. For assertion-outcome steps also clears
    missing_action (aligner often reports unmatched asserts that way) and remaps
    the evidence pointer. Never clears wrong_input_value.
    """
    if support_index is not None:
        clear_assertion[step_index] = support_index
        if _is_assertion_outcome_step(step):
            clear_missing.add(step_index)
            remap_matched[step_index] = support_index
            # Wording-only wrong_target on the old DP pair is superseded by the
            # remapped assertion proof; genuine action mismatches stay elsewhere.
            clear_wrong_target.add(step_index)
    elif support_timestamp_ms is not None:
        clear_assertion_ts[step_index] = support_timestamp_ms
        if _is_assertion_outcome_step(step):
            clear_missing.add(step_index)
            clear_missing_ts[step_index] = support_timestamp_ms
            # Stage 2C often recovers by timestamp only (no ObservedAction index).
            # Wording-only wrong_target on the old DP pair is still superseded.
            clear_wrong_target.add(step_index)
    else:
        return
    assist_meta[step_index] = confidence


def apply_assist_proposals(
    deterministic: AuditResult,
    accepted: list[AssistProposalRecord],
    expected: ExpectedTrace,
    observed: ObservedTrace,
) -> AuditResult:
    """Return a refined audit. Never mutates the deterministic baseline object graph."""
    if not accepted:
        return deterministic.model_copy(deep=True)

    step_by_index = {step.index: step for step in expected.steps}
    action_by_index = {action.index: action for action in observed.actions}

    clear_wrong_target: set[int] = set()
    clear_assertion: dict[int, int] = {}  # expected_step_index -> supporting obs index
    clear_assertion_ts: dict[int, int] = {}  # step -> frame-derived timestamp only
    clear_missing: set[int] = set()
    clear_missing_ts: dict[int, int] = {}  # step -> frame-derived timestamp
    remap_matched: dict[int, int] = {}  # expected_step_index -> terminal obs index
    challenge_assertion: dict[int, int | None] = {}  # step -> matched obs for evidence ts
    assist_meta: dict[int, float | None] = {}  # step -> assist confidence (None = deterministic)
    frame_evidence: dict[int, AssistProposalRecord] = {}

    for record in accepted:
        if record.status != AssistProposalStatus.accepted:
            continue
        if record.kind == AssistCandidateKind.wrong_target:
            clear_wrong_target.add(record.expected_step_index)
            assist_meta[record.expected_step_index] = record.confidence
        elif record.kind == AssistCandidateKind.delayed_assertion:
            if record.supporting_observed_index is None:
                continue
            _apply_assertion_outcome_recovery(
                step_index=record.expected_step_index,
                step=step_by_index.get(record.expected_step_index),
                support_index=record.supporting_observed_index,
                support_timestamp_ms=None,
                confidence=record.confidence,
                clear_assertion=clear_assertion,
                clear_assertion_ts=clear_assertion_ts,
                clear_missing=clear_missing,
                clear_missing_ts=clear_missing_ts,
                clear_wrong_target=clear_wrong_target,
                remap_matched=remap_matched,
                assist_meta=assist_meta,
            )
        elif record.kind == AssistCandidateKind.evidence_bundle:
            if record.challenged_false_pass:
                challenge_assertion[record.expected_step_index] = (
                    record.matched_observed_index
                )
            elif record.supporting_observed_index is not None:
                # Validated assertion support — never initiation alone.
                # For assertion-outcome steps, also clear missing_action so
                # assist metadata stays consistent with findings.
                _apply_assertion_outcome_recovery(
                    step_index=record.expected_step_index,
                    step=step_by_index.get(record.expected_step_index),
                    support_index=record.supporting_observed_index,
                    support_timestamp_ms=None,
                    confidence=record.confidence,
                    clear_assertion=clear_assertion,
                    clear_assertion_ts=clear_assertion_ts,
                    clear_missing=clear_missing,
                    clear_missing_ts=clear_missing_ts,
                    clear_wrong_target=clear_wrong_target,
                    remap_matched=remap_matched,
                    assist_meta=assist_meta,
                )
        elif record.kind == AssistCandidateKind.episode_verification:
            terminal = (
                record.supporting_observed_index
                or record.completion_observed_index
                or record.matched_observed_index
            )
            if terminal is None:
                continue
            # Episode verification already validated a terminal proof / remap.
            remap_matched[record.expected_step_index] = terminal
            clear_assertion[record.expected_step_index] = terminal
            clear_wrong_target.add(record.expected_step_index)
            clear_missing.add(record.expected_step_index)
            assist_meta[record.expected_step_index] = record.confidence
        elif record.kind == AssistCandidateKind.frame_evidence:
            # Stage 2C: only Python-validated frame evidence may clear findings.
            # ACTION recovers missing_action. ASSERTION_OUTCOME uses the shared
            # assertion-outcome recovery path. Never clears wrong_input_value.
            if record.evidence_timestamp_ms is None:
                continue
            evidence_type = record.candidate_evidence_type
            if evidence_type == CandidateEvidenceType.ACTION:
                clear_missing.add(record.expected_step_index)
                clear_missing_ts[record.expected_step_index] = (
                    record.evidence_timestamp_ms
                )
                assist_meta[record.expected_step_index] = record.confidence
            elif evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME:
                frame_evidence[record.expected_step_index] = record
                _apply_assertion_outcome_recovery(
                    step_index=record.expected_step_index,
                    step=step_by_index.get(record.expected_step_index),
                    support_index=None,
                    support_timestamp_ms=record.evidence_timestamp_ms,
                    confidence=record.confidence,
                    clear_assertion=clear_assertion,
                    clear_assertion_ts=clear_assertion_ts,
                    clear_missing=clear_missing,
                    clear_missing_ts=clear_missing_ts,
                    clear_wrong_target=clear_wrong_target,
                    remap_matched=remap_matched,
                    assist_meta=assist_meta,
                )

    new_findings: list[Finding] = []
    for finding in deterministic.findings:
        step_index = finding.expected_step_index
        if step_index is None:
            new_findings.append(finding)
            continue
        if (
            finding.type == FindingType.wrong_target
            and step_index in clear_wrong_target
        ):
            continue
        if (
            finding.type == FindingType.assertion_not_observed
            and (
                step_index in clear_assertion or step_index in clear_assertion_ts
            )
        ):
            continue
        if (
            finding.type == FindingType.missing_action
            and step_index in clear_missing
        ):
            continue
        # Never clear wrong_input_value or other types via assist.
        # Remap evidence pointers on surviving findings for remapped steps.
        if step_index in remap_matched and finding.observed_action_index is not None:
            terminal = remap_matched[step_index]
            support = action_by_index.get(terminal)
            new_findings.append(
                finding.model_copy(
                    update={
                        "observed_action_index": terminal,
                        "evidence_timestamp_ms": (
                            support.timestamp_ms if support is not None else None
                        ),
                    }
                )
            )
            continue
        new_findings.append(finding)

    challenged_steps_with_finding = {
        f.expected_step_index
        for f in new_findings
        if f.type == FindingType.assertion_not_observed
        and f.expected_step_index in challenge_assertion
    }
    for step_index, matched_index in challenge_assertion.items():
        if step_index in challenged_steps_with_finding:
            continue
        step = step_by_index.get(step_index)
        evidence_ts = None
        if matched_index is not None:
            matched = action_by_index.get(matched_index)
            if matched is not None:
                evidence_ts = matched.timestamp_ms
        new_findings.append(
            Finding(
                type=FindingType.assertion_not_observed,
                severity=finding_severity(FindingType.assertion_not_observed, step),
                expected_step_index=step_index,
                observed_action_index=matched_index,
                evidence_timestamp_ms=evidence_ts,
                description=(
                    f"Step {step_index}: evidence-aware assist challenged a suspicious "
                    "pass; assertion was not supported by validated observation text."
                ),
            )
        )

    new_verdicts: list[StepVerdict] = []
    for verdict in deterministic.step_verdicts:
        types = list(verdict.finding_types)
        evidence_ts = verdict.evidence_timestamp_ms
        matched_index = verdict.matched_observed_index
        to_clear: set[FindingType] = set()
        if verdict.expected_step_index in clear_wrong_target:
            to_clear.add(FindingType.wrong_target)
        if verdict.expected_step_index in clear_missing:
            to_clear.add(FindingType.missing_action)
            if verdict.expected_step_index in clear_missing_ts:
                evidence_ts = clear_missing_ts[verdict.expected_step_index]
        if verdict.expected_step_index in clear_assertion:
            to_clear.add(FindingType.assertion_not_observed)
            support_index = clear_assertion[verdict.expected_step_index]
            support = action_by_index.get(support_index)
            if support is not None:
                evidence_ts = support.timestamp_ms
        if verdict.expected_step_index in clear_assertion_ts:
            to_clear.add(FindingType.assertion_not_observed)
            evidence_ts = clear_assertion_ts[verdict.expected_step_index]
        if verdict.expected_step_index in remap_matched:
            matched_index = remap_matched[verdict.expected_step_index]
            support = action_by_index.get(matched_index)
            if support is not None:
                evidence_ts = support.timestamp_ms
        types = _clear_finding_types(types, to_clear)
        if verdict.expected_step_index in challenge_assertion:
            if FindingType.assertion_not_observed not in types:
                types.append(FindingType.assertion_not_observed)
        cleared = verdict.expected_step_index in assist_meta
        frames = frame_evidence.get(verdict.expected_step_index)
        evidence_id = verdict.evidence_id
        frame_ids = list(verdict.frame_ids)
        explanation = verdict.explanation
        recovery_reason = None if cleared else verdict.recovery_reason
        if frames is not None and len(types) == 0:
            # The frames are the evidence. Do not keep the earlier alignment pair.
            matched_index = None
            evidence_ts = frames.evidence_timestamp_ms
            frame_ids = list(frames.frame_ids)
            evidence_id = ",".join(frame_ids) or None
            recovery_reason = "frame_evidence"
            explanation = (
                f"Frames {evidence_id} at {evidence_ts} ms show the asserted text."
            )
        elif matched_index is not None and evidence_ts is not None and len(types) == 0:
            support = action_by_index.get(matched_index)
            if support is not None and support.timestamp_ms == evidence_ts:
                evidence_id = f"O{matched_index}"
                explanation = (
                    f"Observed action O{matched_index} at {evidence_ts} ms "
                    "supports this step."
                )
        new_verdicts.append(
            StepVerdict(
                expected_step_index=verdict.expected_step_index,
                passed=len(types) == 0,
                matched_observed_index=matched_index,
                evidence_timestamp_ms=evidence_ts,
                finding_types=types,
                cleared_by_assist=cleared,
                assist_confidence=(
                    assist_meta.get(verdict.expected_step_index) if cleared else None
                ),
                recovery_reason=recovery_reason,
                evidence_id=evidence_id,
                frame_ids=frame_ids,
                explanation=explanation,
            )
        )

    rebuilt: list[Finding] = []
    for finding in new_findings:
        step = (
            step_by_index.get(finding.expected_step_index)
            if finding.expected_step_index is not None
            else None
        )
        rebuilt.append(
            finding.model_copy(
                update={"severity": finding_severity(finding.type, step)}
            )
        )

    total = len(expected.steps)
    passed = sum(1 for verdict in new_verdicts if verdict.passed)
    fidelity_score = passed / total if total > 0 else 1.0

    return AuditResult(
        verdict=verdict_for_steps(expected.steps, rebuilt, new_verdicts),
        fidelity_score=fidelity_score,
        findings=rebuilt,
        step_verdicts=new_verdicts,
    )
