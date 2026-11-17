"""Opt-in evidence-aware Stage 3 assist orchestration (experimental)."""

import logging

import instructor
import openai
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from reckon.assist_evidence_propose import (
    ALIGN_EVIDENCE_MODEL,
    propose_evidence_batch,
)
from reckon.assist_evidence_select import (
    EvidenceBundleCandidate,
    select_evidence_bundle_candidates,
)
from reckon.assist_evidence_validate import validate_evidence_decision
from reckon.latency import elapsed_ms, monotonic_ms
from reckon.parser import make_openai_client
from reckon.retry import RetryCallback
from reckon.schema import (
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    ExpectedTrace,
    ObservedTrace,
    ScreenEpisode,
)

logger = logging.getLogger(__name__)


def _failed_evidence_record(
    candidate: EvidenceBundleCandidate,
    error_type: str,
) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=candidate.kind,
        status=AssistProposalStatus.failed,
        expected_step_index=candidate.expected_step_index,
        matched_observed_index=candidate.matched_observed_index,
        finding_types=list(candidate.finding_types),
        reason=f"llm_{error_type}",
        matching_method=AssistMatchMethod.rejected_proposal,
    )


def run_evidence_assist(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    deterministic: AuditResult,
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = ALIGN_EVIDENCE_MODEL,
    on_retry: RetryCallback | None = None,
    episodes: list[ScreenEpisode] | None = None,
) -> tuple[list[EvidenceBundleCandidate], list[AssistProposalRecord], int, float, str | None]:
    """Select evidence windows, ask Claude for roles, validate. Never applies."""
    candidates = select_evidence_bundle_candidates(
        expected,
        observed,
        deterministic,
        episodes=episodes,
    )
    if not candidates:
        return [], [], 0, 0.0, None

    llm_started = monotonic_ms()
    try:
        if openai_client is None:
            openai_client = make_openai_client()
        batch = propose_evidence_batch(
            candidates,
            openai_client=openai_client,
            model=model,
            on_retry=on_retry,
        )
        by_id = {item.candidate_id: item for item in batch.decisions}
        proposals = [
            validate_evidence_decision(
                candidate,
                by_id.get(candidate.candidate_id),
                expected,
                observed,
                parse_error=(
                    None
                    if candidate.candidate_id in by_id
                    else "missing_decision"
                ),
            )
            for candidate in candidates
        ]
        return candidates, proposals, 1, elapsed_ms(llm_started), None
    except (
        openai.APIError,
        openai.APITimeoutError,
        InstructorRetryException,
        ValidationError,
        ValueError,
        TypeError,
    ) as exc:
        logger.warning(
            "stage3.evidence llm_failed error_type=%s", type(exc).__name__
        )
        proposals = [
            _failed_evidence_record(candidate, type(exc).__name__)
            for candidate in candidates
        ]
        return (
            candidates,
            proposals,
            1,
            elapsed_ms(llm_started),
            f"llm_{type(exc).__name__}",
        )
