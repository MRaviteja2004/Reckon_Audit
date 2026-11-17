"""Opt-in Stage 3 LLM assist on top of the unchanged deterministic aligner.

Ownership
---------
Stage 3 assist (this module):
  - Episode-level state verification for coarse assertion steps.
  - Temporal remapping of drifted click/select/type_text steps onto episode
    actions between verified anchors (independent of DP 1:1 drift).
  - Semantic wrong_target resolution on already-matched pairs.
  - Delayed assertion support using later ObservedTrace *text* evidence.
  - Optional experimental evidence-aware bundles for coarse expected steps.

Stage 4 cross-check retains:
  - Visual frame verification for remaining assertion_not_observed findings.
  - Semantic wrong_target review only for pairs Stage 3 assist did not already review.

When assist is disabled, callers keep using align() and run_cross_check() unchanged.

Safety
------
Claude proposes; Python validates and applies. On any LLM / parse failure the
deterministic AuditResult is retained. Clean deterministic matches and
wrong_input_value findings are never overwritten. Initiation alone never clears
assertions. reviewed_semantic_pairs includes accepted high-confidence
resolutions only (LLM confidence >= 0.85, or deterministic validation).
"""

import logging

import instructor
import openai
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from pathlib import Path

from reckon.aligner import align
from reckon.assist_apply import apply_assist_proposals
from reckon.assist_evidence import run_evidence_assist
from reckon.assist_propose import ALIGN_ASSIST_MODEL, propose_assist_batch
from reckon.assist_select import AssistCandidate, select_assist_candidates
from reckon.assist_validate import validate_assist_decision, validate_deterministic_delayed
from reckon.episode_verifier import verify_trace_against_episodes
from reckon.episodes import cluster_observed_episodes
from reckon.latency import elapsed_ms, monotonic_ms
from reckon.parser import make_openai_client
from reckon.retry import RetryCallback
from reckon.schema import (
    AssistCandidateKind,
    AssistMatchMethod,
    AssistMetrics,
    AssistProposalRecord,
    AssistProposalStatus,
    AssistReport,
    AssistReviewedPair,
    AssistedAlignResult,
    CandidateEvidenceTrace,
    ExpectedTrace,
    FrameManifest,
    ObservedTrace,
)
from reckon.stage2c import (
    discover_candidate_evidence,
    high_confidence_visual_skip_steps,
    proposals_from_candidate_trace,
)
from reckon.stage2c.propose import RawFrameEvidenceBatch

logger = logging.getLogger(__name__)

# Stage 4 semantic skip only for high-confidence, structurally validated pairs.
ASSIST_SEMANTIC_SKIP_CONFIDENCE = 0.85


def _empty_metrics(
    *,
    expected_n: int,
    observed_n: int,
    deterministic_matches: int,
    deterministic_ms: float,
    total_ms: float,
) -> AssistMetrics:
    return AssistMetrics(
        expected_event_count=expected_n,
        observed_event_count=observed_n,
        deterministic_match_count=deterministic_matches,
        candidate_count=0,
        llm_request_count=0,
        proposals_accepted=0,
        proposals_rejected=0,
        unresolved_count=0,
        deterministic_ms=deterministic_ms,
        llm_ms=0.0,
        total_ms=total_ms,
    )


def _metrics_from_proposals(
    *,
    expected_n: int,
    observed_n: int,
    deterministic_matches: int,
    candidate_count: int,
    llm_request_count: int,
    proposals: list[AssistProposalRecord],
    deterministic_ms: float,
    llm_ms: float,
    total_ms: float,
) -> AssistMetrics:
    accepted = sum(1 for p in proposals if p.status == AssistProposalStatus.accepted)
    rejected = sum(
        1
        for p in proposals
        if p.status in {AssistProposalStatus.rejected, AssistProposalStatus.failed}
    )
    unresolved = sum(1 for p in proposals if p.reason == "model_unresolved")
    return AssistMetrics(
        expected_event_count=expected_n,
        observed_event_count=observed_n,
        deterministic_match_count=deterministic_matches,
        candidate_count=candidate_count,
        llm_request_count=llm_request_count,
        proposals_accepted=accepted,
        proposals_rejected=rejected,
        unresolved_count=unresolved,
        deterministic_ms=deterministic_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
    )


def _reviewed_pairs_from_proposals(
    proposals: list[AssistProposalRecord],
) -> list[AssistReviewedPair]:
    """Pairs Stage 4 semantic review should skip.

    ``skip_semantic_pairs`` is populated ONLY from ``status == accepted`` records.
    Acceptance itself is granted only after deterministic Python validation
    (``assertion_satisfied``, ``values_equivalent``, protected contradictions,
    action/target floors) — high LLM confidence alone never enters this set.

    Additionally, LLM-assisted accepts require confidence >= 0.85. Deterministic
    accepts (no LLM confidence) are included. Rejected / failed / unresolved
    proposals and low-confidence LLM accepts remain Stage-4 eligible.
    """
    reviewed: list[AssistReviewedPair] = []
    seen: set[tuple[int, int]] = set()
    for record in proposals:
        # Hard gate: never skip Stage 4 for non-accepted proposals.
        if record.status != AssistProposalStatus.accepted:
            continue
        if record.matched_observed_index is None:
            continue
        if record.challenged_false_pass:
            continue
        if record.matching_method == AssistMatchMethod.llm_assisted:
            if (
                record.confidence is None
                or record.confidence < ASSIST_SEMANTIC_SKIP_CONFIDENCE
            ):
                continue
        key = (record.expected_step_index, record.matched_observed_index)
        if key in seen:
            continue
        seen.add(key)
        reviewed.append(
            AssistReviewedPair(
                expected_step_index=record.expected_step_index,
                observed_action_index=record.matched_observed_index,
                confidence=record.confidence,
            )
        )
    return reviewed


def _failed_record(candidate: AssistCandidate, error_type: str) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=candidate.kind,
        status=AssistProposalStatus.failed,
        expected_step_index=candidate.expected_step_index,
        matched_observed_index=candidate.matched_observed_index,
        finding_types=list(candidate.finding_types),
        reason=f"llm_{error_type}",
        matching_method=AssistMatchMethod.rejected_proposal,
    )


def align_with_assist(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = ALIGN_ASSIST_MODEL,
    on_retry: RetryCallback | None = None,
    evidence_aware: bool = False,
    frame_evidence: bool = False,
    frame_manifest: FrameManifest | None = None,
    frame_video_path: Path | None = None,
    frame_jpeg_by_id: dict[str, bytes] | None = None,
    stage2c_raw_batch: RawFrameEvidenceBatch | None = None,
    equivalence_agent_candidates: dict[int, list[int]] | None = None,
) -> AssistedAlignResult:
    """Run deterministic align(), then optionally apply validated Claude proposals.

    Existing align() callers are unaffected. This entry point is opt-in.
    evidence_aware enables experimental multi-observation evidence bundles in
    addition to wrong_target / delayed_assertion assist.

    frame_evidence enables Stage 2C expected-guided frame discovery. Requires
    ``frame_manifest``. LLM returns frame IDs only; Python validates and applies.
    """
    started = monotonic_ms()
    deterministic = align(expected, observed)
    if equivalence_agent_candidates:
        from reckon.equivalence.engine import apply_agent_equivalence

        deterministic = apply_agent_equivalence(
            expected,
            observed,
            deterministic,
            equivalence_agent_candidates,
            scenario_text=expected.source_text or "",
        )
    deterministic_ms = elapsed_ms(started)
    deterministic_matches = sum(1 for v in deterministic.step_verdicts if v.passed)
    # Episode clusters are the assist view of observed micro-actions. Deterministic
    # align() remains 1:1; episode verification remaps coarse steps to terminal proof.
    episodes = cluster_observed_episodes(observed)

    proposals: list[AssistProposalRecord] = []
    llm_ms = 0.0
    llm_request_count = 0
    assist_error: str | None = None
    candidate_evidence: CandidateEvidenceTrace | None = None
    stage2c_skip_visual: set[int] = set()

    # Phase 3: episode-level verification before classic pair assist.
    episode_started = monotonic_ms()
    try:
        episode_proposals, episode_llm_requests = verify_trace_against_episodes(
            expected,
            observed,
            deterministic,
            episodes,
            openai_client=openai_client,
            model=model,
            on_retry=on_retry,
            allow_llm=True,
        )
    except Exception as exc:
        logger.warning(
            "stage3.episode_verify failed error_type=%s", type(exc).__name__
        )
        episode_proposals = []
        episode_llm_requests = 0
        assist_error = f"episode_verify_{type(exc).__name__}"
    else:
        if episode_llm_requests:
            if openai_client is None:
                # verify_trace may have created a client internally; keep counting.
                pass
            llm_ms += elapsed_ms(episode_started)
        llm_request_count += episode_llm_requests
    proposals.extend(episode_proposals)
    episode_candidate_count = len(episode_proposals)
    episode_accepted_steps = {
        record.expected_step_index
        for record in episode_proposals
        if record.status == AssistProposalStatus.accepted
    }

    # Classic assist runs against the deterministic baseline. Steps already proven
    # by episode verification are skipped to avoid conflicting remaps.
    candidates = [
        candidate
        for candidate in select_assist_candidates(expected, observed, deterministic)
        if candidate.expected_step_index not in episode_accepted_steps
    ]
    llm_candidates: list[AssistCandidate] = []

    for candidate in candidates:
        if (
            candidate.kind == AssistCandidateKind.delayed_assertion
            and candidate.deterministic_support_index is not None
        ):
            proposals.append(
                validate_deterministic_delayed(candidate, expected, observed)
            )
        else:
            llm_candidates.append(candidate)

    if llm_candidates and assist_error is None:
        llm_started = monotonic_ms()
        llm_request_count += 1
        try:
            if openai_client is None:
                openai_client = make_openai_client()
            batch = propose_assist_batch(
                llm_candidates,
                openai_client=openai_client,
                model=model,
                on_retry=on_retry,
            )
            by_id = {item.candidate_id: item for item in batch.decisions}
            for candidate in llm_candidates:
                proposals.append(
                    validate_assist_decision(
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
                )
        except (
            openai.APIError,
            openai.APITimeoutError,
            InstructorRetryException,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "stage3.assist llm_failed error_type=%s", type(exc).__name__
            )
            llm_ms += elapsed_ms(llm_started)
            proposals.extend(
                _failed_record(candidate, type(exc).__name__)
                for candidate in llm_candidates
            )
            assist_error = f"llm_{type(exc).__name__}"
        else:
            llm_ms += elapsed_ms(llm_started)

    evidence_candidates = []
    if evidence_aware and assist_error is None:
        ev_candidates, ev_proposals, ev_requests, ev_ms, ev_error = run_evidence_assist(
            expected,
            observed,
            deterministic,
            openai_client=openai_client,
            model=model,
            on_retry=on_retry,
            episodes=episodes,
        )
        evidence_candidates = [
            candidate
            for candidate in ev_candidates
            if candidate.expected_step_index not in episode_accepted_steps
        ]
        proposals.extend(
            proposal
            for proposal in ev_proposals
            if proposal.expected_step_index not in episode_accepted_steps
        )
        llm_request_count += ev_requests
        llm_ms += ev_ms
        if ev_error:
            assist_error = ev_error

    stage2c_candidate_count = 0
    if frame_evidence and assist_error is None:
        if frame_manifest is None:
            assist_error = "stage2c_missing_frame_manifest"
        else:
            # Apply classic/evidence accepts first so Stage 2C only searches remaining gaps.
            interim_accepted = [
                p for p in proposals if p.status == AssistProposalStatus.accepted
            ]
            interim_audit = (
                apply_assist_proposals(
                    deterministic, interim_accepted, expected, observed
                )
                if interim_accepted
                else deterministic
            )
            try:
                (
                    candidate_evidence,
                    s2c_requests,
                    s2c_ms,
                    s2c_error,
                ) = discover_candidate_evidence(
                    expected,
                    interim_audit,
                    frame_manifest,
                    source_file=observed.source_file,
                    observed=observed,
                    video_path=frame_video_path,
                    jpeg_by_frame_id=frame_jpeg_by_id,
                    raw_batch=stage2c_raw_batch,
                    openai_client=openai_client,
                    model=model,
                    on_retry=on_retry,
                )
            except Exception as exc:
                logger.warning(
                    "stage2c.discover failed error_type=%s", type(exc).__name__
                )
                candidate_evidence = CandidateEvidenceTrace(
                    source_file=observed.source_file, candidates=[]
                )
                s2c_requests = 0
                s2c_ms = 0.0
                s2c_error = f"stage2c_{type(exc).__name__}"
            stage2c_proposals = proposals_from_candidate_trace(
                candidate_evidence, expected
            )
            # Skip steps already recovered earlier in this assist pass.
            recovered = {
                p.expected_step_index
                for p in proposals
                if p.status == AssistProposalStatus.accepted
            }
            stage2c_proposals = [
                p
                for p in stage2c_proposals
                if p.expected_step_index not in recovered
                and p.expected_step_index not in episode_accepted_steps
            ]
            proposals.extend(stage2c_proposals)
            stage2c_candidate_count = len(stage2c_proposals)
            llm_request_count += s2c_requests
            llm_ms += s2c_ms
            supported_steps = {
                proposal.expected_step_index
                for proposal in stage2c_proposals
                if proposal.status == AssistProposalStatus.accepted
            }
            stage2c_skip_visual = high_confidence_visual_skip_steps(
                candidate_evidence
            ) & supported_steps
            if s2c_error and assist_error is None:
                assist_error = s2c_error

    candidate_count = (
        episode_candidate_count
        + len(candidates)
        + len(evidence_candidates)
        + stage2c_candidate_count
    )
    if candidate_count == 0:
        total_ms = elapsed_ms(started)
        return AssistedAlignResult(
            deterministic_result=deterministic,
            result=deterministic,
            assist=AssistReport(
                model=model,
                metrics=_empty_metrics(
                    expected_n=len(expected.steps),
                    observed_n=len(observed.actions),
                    deterministic_matches=deterministic_matches,
                    deterministic_ms=deterministic_ms,
                    total_ms=total_ms,
                ),
                candidate_evidence=candidate_evidence,
                stage2c_skip_visual_steps=sorted(stage2c_skip_visual),
            ),
        )

    base_committed = {
        decision.expected_step_index
        for decision in deterministic.equivalence_decisions
        if decision.status == "committed" or decision.agent_status == "committed"
    }
    accepted = [
        proposal
        for proposal in proposals
        if proposal.status == AssistProposalStatus.accepted
        and proposal.expected_step_index not in base_committed
    ]
    try:
        refined = (
            apply_assist_proposals(deterministic, accepted, expected, observed)
            if accepted
            else deterministic
        )
    except Exception as exc:
        logger.warning("stage3.assist apply_failed error_type=%s", type(exc).__name__)
        total_ms = elapsed_ms(started)
        return AssistedAlignResult(
            deterministic_result=deterministic,
            result=deterministic,
            assist=AssistReport(
                model=model,
                proposals=proposals,
                metrics=_metrics_from_proposals(
                    expected_n=len(expected.steps),
                    observed_n=len(observed.actions),
                    deterministic_matches=deterministic_matches,
                    candidate_count=candidate_count,
                    llm_request_count=llm_request_count,
                    proposals=proposals,
                    deterministic_ms=deterministic_ms,
                    llm_ms=llm_ms,
                    total_ms=total_ms,
                ),
                reviewed_semantic_pairs=_reviewed_pairs_from_proposals(proposals),
                error=f"apply_{type(exc).__name__}",
                candidate_evidence=candidate_evidence,
                stage2c_skip_visual_steps=sorted(stage2c_skip_visual),
            ),
        )

    total_ms = elapsed_ms(started)
    return AssistedAlignResult(
        deterministic_result=deterministic,
        result=refined,
        assist=AssistReport(
            model=model,
            proposals=proposals,
            metrics=_metrics_from_proposals(
                expected_n=len(expected.steps),
                observed_n=len(observed.actions),
                deterministic_matches=deterministic_matches,
                candidate_count=candidate_count,
                llm_request_count=llm_request_count,
                proposals=proposals,
                deterministic_ms=deterministic_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
            ),
            reviewed_semantic_pairs=_reviewed_pairs_from_proposals(proposals),
            error=assist_error,
            candidate_evidence=candidate_evidence,
            stage2c_skip_visual_steps=sorted(stage2c_skip_visual),
        ),
    )
