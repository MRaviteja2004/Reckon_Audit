"""Deterministic validation of Stage 2C frame evidence candidates.

Candidate evidence is never proof. This module materializes timestamps from the
frame manifest and applies the five mandatory rules before Stage 3 may use a
candidate.
"""

import re

from reckon.schema import (
    AuditResult,
    CandidateEvidenceItem,
    CandidateEvidenceTrace,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedTrace,
    FindingType,
    FrameManifest,
    ObservedTrace,
)
from reckon.stage2c.manifest import materialize_timestamp_range, normalize_frame_id
from reckon.stage2c.propose import RawFrameEvidenceCandidate
from reckon.stage2c.roles import (
    expectation_semantic_role,
    evidence_semantic_role,
    role_mismatch_reason,
)
from reckon.stage2c.select import Stage2CSearchTarget
from reckon.tolerance import assertion_satisfied
from reckon.workflow_reentry import reentry_lower_timestamp_ms

STAGE2C_CONFIDENCE_FLOOR = 0.7

# Terminal / blocking error phrases that must not "prove" a positive success claim.
_BLOCKING_ERROR_RE = re.compile(
    r"\b(?:"
    r"invalid\s+(?:password|credentials?|email|username)|"
    r"incorrect\s+(?:password|credentials?|email|username)|"
    r"authentication\s+failed|"
    r"login\s+failed|"
    r"sign[- ]?in\s+failed|"
    r"something\s+went\s+wrong|"
    r"error\s+occurred|"
    r"unexpected\s+error|"
    r"connection\s+failed|"
    r"network\s+(?:error|failure)|"
    r"access\s+denied|"
    r"crash(?:ed)?|"
    r"unable\s+to\s+(?:authenticate|log\s*in|sign\s*in)"
    r")\b",
    re.IGNORECASE,
)

_ERROR_CLAIM_RE = re.compile(
    r"\b(?:error|invalid|fail(?:ed|ure)?|denied|unable|wrong)\b",
    re.IGNORECASE,
)

# Candidate text that admits the claimed surface/control is not present.
_CONTRADICTORY_ABSENCE_RE = re.compile(
    r"\b(?:"
    r"no\s+page\s+explicitly|"
    r"not\s+(?:a\s+)?(?:program|mentorship\s+)?settings\s+page|"
    r"no\s+(?:\w+\s+){0,3}(?:labelled|labeled)\s+settings|"
    r"settings\s+(?:page\s+)?(?:is\s+)?not\s+(?:shown|visible|present|found)|"
    r"not\s+(?:shown|visible|present|found)\b.*\bsettings"
    r")\b",
    re.IGNORECASE,
)


def _parse_step_index(expected_step_id: str) -> int | None:
    text = expected_step_id.strip().upper()
    if not text.startswith("E") or not text[1:].isdigit():
        return None
    return int(text[1:])


def _is_reliable_anchor(
    verdict,
    *,
    observed: ObservedTrace | None = None,
    expected: ExpectedTrace | None = None,
) -> bool:
    """Passed DP/assist matches only — failed or suspicious pairs are not anchors.

    A step that ``passed`` while still carrying wrong_target / assertion failure,
    or whose matched observation does not satisfy its assertion claim, must not
    bound other candidates.
    """
    if not verdict.passed:
        return False
    if verdict.matched_observed_index is None and verdict.evidence_timestamp_ms is None:
        return False
    findings = set(verdict.finding_types or ())
    if findings & {
        FindingType.wrong_target,
        FindingType.assertion_not_observed,
        FindingType.missing_action,
        FindingType.wrong_input_value,
    }:
        return False
    if verdict.evidence_timestamp_ms is None:
        return False
    if observed is None or expected is None or verdict.matched_observed_index is None:
        return True

    step = next(
        (
            item
            for item in expected.steps
            if item.index == verdict.expected_step_index
        ),
        None,
    )
    action = next(
        (
            item
            for item in observed.actions
            if item.index == verdict.matched_observed_index
        ),
        None,
    )
    if step is None or action is None:
        return False
    if step.assertion is None:
        return True
    return assertion_satisfied(
        step.assertion, action, expected_target=step.target
    )


def _anchor_bounds(
    deterministic: AuditResult,
    step_index: int,
    *,
    observed: ObservedTrace | None = None,
    expected: ExpectedTrace | None = None,
) -> tuple[int | None, int | None]:
    """Lower/upper timestamp bounds from surrounding *reliable* anchors.

    Unreliable DP matches (failed assertions / wrong targets / suspicious passes)
    must not make valid later evidence look premature or out-of-order. When a
    workflow re-entry (deeplink) precedes this step's expected section, the lower
    bound may rise to that re-entry so pre-entry noise cannot validate post-entry
    assertions — without equating the re-entry to a feed-card click.
    """
    lower: int | None = None
    upper: int | None = None
    for verdict in deterministic.step_verdicts:
        if not _is_reliable_anchor(
            verdict, observed=observed, expected=expected
        ):
            continue
        ts = verdict.evidence_timestamp_ms
        if ts is None:
            continue
        if verdict.expected_step_index < step_index:
            if lower is None or ts > lower:
                lower = ts
        elif verdict.expected_step_index > step_index:
            if upper is None or ts < upper:
                upper = ts

    if expected is not None and observed is not None:
        reentry_ts = reentry_lower_timestamp_ms(
            expected, observed, step_index=step_index
        )
        if reentry_ts is not None:
            lower = reentry_ts if lower is None else max(lower, reentry_ts)
    return lower, upper


def _frame_reuse_conflict(
    *,
    candidate_type: CandidateEvidenceType,
    owner_type: CandidateEvidenceType,
) -> bool:
    """True when claiming a frame already owned by another step is disallowed.

    Independent ACTION evidence stays exclusive. ASSERTION_OUTCOME may share
    persistent co-visible screen context with other assertions, and may share a
    terminal ACTION frame (outcome on the same still). ACTION may not steal a
    frame already owned by another ACTION.
    """
    if candidate_type == CandidateEvidenceType.ACTION:
        return owner_type == CandidateEvidenceType.ACTION
    # ASSERTION_OUTCOME: share with ACTION or ASSERTION_OUTCOME.
    return False


def _combined_visible(candidate: RawFrameEvidenceCandidate) -> str:
    parts = [candidate.observed_description, *candidate.visible_text]
    return " ".join(part for part in parts if part)


def _expects_error(target: Stage2CSearchTarget) -> bool:
    text = " ".join(
        part
        for part in (target.expected_assertion, target.expected_text)
        if part
    )
    return bool(_ERROR_CLAIM_RE.search(text))


def _reject(
    *,
    expected_step_id: str,
    expected_step_index: int,
    raw: RawFrameEvidenceCandidate,
    reason: str,
    frame_ids: list[str] | None = None,
    timestamp_start_ms: int | None = None,
    timestamp_end_ms: int | None = None,
) -> CandidateEvidenceItem:
    return CandidateEvidenceItem(
        expected_step_id=expected_step_id,
        expected_step_index=expected_step_index,
        frame_ids=list(frame_ids or raw.frame_ids),
        timestamp_start_ms=timestamp_start_ms,
        timestamp_end_ms=timestamp_end_ms,
        evidence_type=raw.evidence_type,
        observed_description=raw.observed_description,
        visible_text=list(raw.visible_text),
        confidence=raw.confidence,
        search_status=raw.search_status,
        validation_status=CandidateValidationStatus.rejected,
        rejection_reason=reason,
    )


def _accept(
    *,
    expected_step_id: str,
    expected_step_index: int,
    raw: RawFrameEvidenceCandidate,
    frame_ids: list[str],
    timestamp_start_ms: int,
    timestamp_end_ms: int,
) -> CandidateEvidenceItem:
    return CandidateEvidenceItem(
        expected_step_id=expected_step_id,
        expected_step_index=expected_step_index,
        frame_ids=frame_ids,
        timestamp_start_ms=timestamp_start_ms,
        timestamp_end_ms=timestamp_end_ms,
        evidence_type=raw.evidence_type,
        observed_description=raw.observed_description,
        visible_text=list(raw.visible_text),
        confidence=raw.confidence,
        search_status=raw.search_status,
        validation_status=CandidateValidationStatus.accepted,
        rejection_reason=None,
    )


def validate_raw_candidate(
    raw: RawFrameEvidenceCandidate,
    target: Stage2CSearchTarget | None,
    manifest: FrameManifest,
    deterministic: AuditResult,
    *,
    claimed_frames: dict[str, tuple[int, CandidateEvidenceType]] | None = None,
    peer_action_end_ms: int | None = None,
    observed: ObservedTrace | None = None,
    expected: ExpectedTrace | None = None,
) -> CandidateEvidenceItem:
    """Validate one raw Stage 2C candidate against the five deterministic rules."""
    step_index = _parse_step_index(raw.expected_step_id)
    if step_index is None:
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=1,
            raw=raw,
            reason="invalid_expected_step_id",
        )
    if target is None or target.expected_step_index != step_index:
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="unexpected_step_id",
        )

    if raw.search_status != CandidateSearchStatus.candidate_found:
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason=f"search_{raw.search_status.value}",
        )

    if raw.confidence < STAGE2C_CONFIDENCE_FLOOR:
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="low_confidence",
        )

    # Rule 1 — frame ID existence and timestamp materialization.
    materialized = materialize_timestamp_range(manifest, raw.frame_ids)
    if materialized is None:
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="unknown_frame_id",
        )
    frame_ids, start_ms, end_ms = materialized

    # Rule 2 — temporal monotonicity against surrounding matched anchors.
    lower, upper = _anchor_bounds(
        deterministic, step_index, observed=observed, expected=expected
    )
    if lower is not None and end_ms < lower:
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="before_previous_anchor",
            frame_ids=frame_ids,
            timestamp_start_ms=start_ms,
            timestamp_end_ms=end_ms,
        )
    if upper is not None and start_ms > upper:
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="after_next_anchor",
            frame_ids=frame_ids,
            timestamp_start_ms=start_ms,
            timestamp_end_ms=end_ms,
        )

    # Rule 3 — action-before-outcome sequencing.
    if (
        raw.evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME
        and peer_action_end_ms is not None
        and start_ms < peer_action_end_ms
    ):
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="outcome_before_action",
            frame_ids=frame_ids,
            timestamp_start_ms=start_ms,
            timestamp_end_ms=end_ms,
        )

    # Rule 4 — evidence reuse. ACTION stays exclusive among ACTIONs;
    # ASSERTION_OUTCOME may share persistent / terminal frames.
    if claimed_frames is not None:
        for fid in frame_ids:
            owner = claimed_frames.get(normalize_frame_id(fid))
            if owner is None:
                continue
            owner_step, owner_type = owner
            if owner_step == step_index:
                continue
            if _frame_reuse_conflict(
                candidate_type=raw.evidence_type,
                owner_type=owner_type,
            ):
                return _reject(
                    expected_step_id=raw.expected_step_id,
                    expected_step_index=step_index,
                    raw=raw,
                    reason="frame_reuse_conflict",
                    frame_ids=frame_ids,
                    timestamp_start_ms=start_ms,
                    timestamp_end_ms=end_ms,
                )

    # Rule 5 — terminal / blocking error collision protection.
    combined = _combined_visible(raw)
    if _BLOCKING_ERROR_RE.search(combined) and not _expects_error(target):
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="blocking_error_collision",
            frame_ids=frame_ids,
            timestamp_start_ms=start_ms,
            timestamp_end_ms=end_ms,
        )

    # Candidate description that admits the expected surface is missing cannot
    # prove a positive display assertion (e.g. opt-out screen ≠ settings page).
    if (
        raw.evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME
        and not _expects_error(target)
        and _CONTRADICTORY_ABSENCE_RE.search(combined)
    ):
        return _reject(
            expected_step_id=raw.expected_step_id,
            expected_step_index=step_index,
            raw=raw,
            reason="contradictory_description",
            frame_ids=frame_ids,
            timestamp_start_ms=start_ms,
            timestamp_end_ms=end_ms,
        )

    # Semantic role exclusivity for ACTION evidence only. ASSERTION_OUTCOME and
    # unrelated OTHER roles keep prior behavior.
    if raw.evidence_type == CandidateEvidenceType.ACTION and expected is not None:
        step = next(
            (item for item in expected.steps if item.index == step_index),
            None,
        )
        if step is not None:
            exp_role = expectation_semantic_role(step, expected)
            ev_role = evidence_semantic_role(
                raw.observed_description, list(raw.visible_text)
            )
            mismatch = role_mismatch_reason(exp_role, ev_role)
            if mismatch is not None:
                return _reject(
                    expected_step_id=raw.expected_step_id,
                    expected_step_index=step_index,
                    raw=raw,
                    reason=mismatch,
                    frame_ids=frame_ids,
                    timestamp_start_ms=start_ms,
                    timestamp_end_ms=end_ms,
                )

    if claimed_frames is not None:
        for fid in frame_ids:
            key = normalize_frame_id(fid)
            existing = claimed_frames.get(key)
            # Prefer ACTION ownership when sharing so a later ACTION still
            # conflicts with the first ACTION on that frame.
            if existing is None or (
                existing[1] != CandidateEvidenceType.ACTION
                and raw.evidence_type == CandidateEvidenceType.ACTION
            ):
                claimed_frames[key] = (step_index, raw.evidence_type)

    return _accept(
        expected_step_id=raw.expected_step_id,
        expected_step_index=step_index,
        raw=raw,
        frame_ids=frame_ids,
        timestamp_start_ms=start_ms,
        timestamp_end_ms=end_ms,
    )


def validate_stage2c_batch(
    raw_candidates: list[RawFrameEvidenceCandidate],
    targets: list[Stage2CSearchTarget],
    manifest: FrameManifest,
    deterministic: AuditResult,
    *,
    source_file: str,
    expected: ExpectedTrace | None = None,
    observed: ObservedTrace | None = None,
) -> CandidateEvidenceTrace:
    """Validate a full Stage 2C batch; preserve every rejection reason."""
    by_id = {target.expected_step_id.upper(): target for target in targets}
    claimed_frames: dict[str, tuple[int, CandidateEvidenceType]] = {}

    # Process ACTION candidates first so ASSERTION_OUTCOME can enforce sequencing.
    ordered = sorted(
        raw_candidates,
        key=lambda item: (
            0 if item.evidence_type == CandidateEvidenceType.ACTION else 1,
            _parse_step_index(item.expected_step_id) or 10**9,
        ),
    )

    action_end_by_step: dict[int, int] = {}
    items: list[CandidateEvidenceItem] = []
    for raw in ordered:
        target = by_id.get(raw.expected_step_id.strip().upper())
        peer_end: int | None = None
        step_index = _parse_step_index(raw.expected_step_id)
        if (
            raw.evidence_type == CandidateEvidenceType.ASSERTION_OUTCOME
            and step_index is not None
        ):
            # Prefer same-step action end; else nearest prior action end.
            peer_end = action_end_by_step.get(step_index)
            if peer_end is None:
                prior = [
                    end
                    for idx, end in action_end_by_step.items()
                    if idx < step_index
                ]
                peer_end = max(prior) if prior else None

        item = validate_raw_candidate(
            raw,
            target,
            manifest,
            deterministic,
            claimed_frames=claimed_frames,
            peer_action_end_ms=peer_end,
            observed=observed,
            expected=expected,
        )
        items.append(item)
        if (
            item.validation_status == CandidateValidationStatus.accepted
            and item.evidence_type == CandidateEvidenceType.ACTION
            and item.timestamp_end_ms is not None
        ):
            action_end_by_step[item.expected_step_index] = item.timestamp_end_ms

    # Stable audit order by expected step then evidence type.
    items.sort(
        key=lambda item: (
            item.expected_step_index,
            0 if item.evidence_type == CandidateEvidenceType.ACTION else 1,
        )
    )
    return CandidateEvidenceTrace(source_file=source_file, candidates=items)
