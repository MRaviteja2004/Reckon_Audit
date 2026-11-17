"""Episode-level state invariant verification for Stage 3 assist.

After deterministic DP alignment, coarse expected steps with
``assertion_not_observed`` or ``missing_action`` are checked against causal
``ScreenEpisode`` windows. Fast-path uses ``tolerance`` text invariants; an
optional LLM path returns VERIFIED / FAILED / INSUFFICIENT_EVIDENCE without
guessing. Claude never mutates the audit — callers validate and apply.
"""

import logging

import instructor
import openai
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from reckon.episodes import episode_containing_index
from reckon.episode_interaction import (
    is_passive_observation,
    remap_interactions_against_episodes,
)
from reckon.episode_verify_fast import (
    candidate_episodes_for_step,
    episode_has_negative_invariant,
    episode_probe_texts,
    fast_path_verify_episode,
    findings_for_step,
    proof_is_premature,
    quote_appears_in_texts,
    resolve_action_for_quote,
    select_episodes_for_llm,
    steps_needing_episode_verification,
)
from reckon.tolerance import _introduces_negation, assertion_target_relevant
from reckon.episode_verify_propose import (
    EPISODE_VERIFY_CONFIDENCE_FLOOR,
    EPISODE_VERIFY_MODEL,
    EpisodeVerifyDecision,
    EpisodeVerifyVerdict,
    propose_episode_verification,
)
from reckon.retry import RetryCallback
from reckon.schema import (
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedTrace,
    ScreenEpisode,
)
logger = logging.getLogger(__name__)

# Re-export for callers/tests that import from this module.
__all__ = [
    "candidate_episodes_for_step",
    "episode_has_negative_invariant",
    "fast_path_verify_episode",
    "steps_needing_episode_verification",
    "verify_step_against_episodes",
    "verify_trace_against_episodes",
]


def _record(
    *,
    step: ExpectedStep,
    status: AssistProposalStatus,
    matched_index: int | None,
    support_index: int | None,
    finding_types: list[FindingType],
    reason: str | None,
    matching_method: AssistMatchMethod,
    model_answer: str | None = None,
    confidence: float | None = None,
    rationale: str | None = None,
) -> AssistProposalRecord:
    return AssistProposalRecord(
        kind=AssistCandidateKind.episode_verification,
        status=status,
        expected_step_index=step.index,
        matched_observed_index=matched_index,
        supporting_observed_index=support_index,
        completion_observed_index=support_index,
        finding_types=finding_types,
        model_answer=model_answer,
        confidence=confidence,
        rationale=rationale,
        reason=reason,
        matching_method=matching_method,
    )


def _validate_llm_decision(
    step: ExpectedStep,
    decision: EpisodeVerifyDecision,
    episodes: list[ScreenEpisode],
    observed: ObservedTrace,
    findings: list[FindingType],
    baseline_matched: int | None,
    expected: ExpectedTrace,
) -> AssistProposalRecord:
    allowed = {
        action.index
        for episode in episodes
        for action in observed.actions
        if episode.start_index <= action.index <= episode.end_index
    }
    if decision.verdict == EpisodeVerifyVerdict.insufficient_evidence:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=None,
            finding_types=findings,
            reason="model_insufficient_evidence",
            matching_method=AssistMatchMethod.unresolved,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    if decision.verdict == EpisodeVerifyVerdict.failed:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=None,
            finding_types=findings,
            reason="model_failed",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    if decision.confidence < EPISODE_VERIFY_CONFIDENCE_FLOOR:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=None,
            finding_types=findings,
            reason="low_confidence",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    quote = (decision.evidence_quote or "").strip()
    if not quote:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=decision.best_action_index,
            finding_types=findings,
            reason="missing_evidence_quote",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    if decision.best_action_index is None or decision.best_action_index not in allowed:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=decision.best_action_index,
            finding_types=findings,
            reason="best_action_not_in_candidates",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    episode = episode_containing_index(episodes, decision.best_action_index)
    if episode is None or step.assertion is None:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=decision.best_action_index,
            finding_types=findings,
            reason="missing_trace_refs",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    corpus = episode_probe_texts(episode, observed)
    if not quote_appears_in_texts(quote, corpus):
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=decision.best_action_index,
            finding_types=findings,
            reason="ungrounded_verification",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    if _introduces_negation(step.assertion, quote):
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=decision.best_action_index,
            finding_types=findings,
            reason="negative_quote",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    if episode_has_negative_invariant(step, episode, observed):
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=decision.best_action_index,
            finding_types=findings,
            reason="negative_invariant",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    action = resolve_action_for_quote(
        quote,
        episode,
        observed,
        preferred_index=decision.best_action_index,
    )
    if action is None:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=decision.best_action_index,
            finding_types=findings,
            reason="ungrounded_verification",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    # Named component assertions must land on a compatible UI role / screen —
    # a Join Mentorship button must not prove a Mentorship title, and a
    # carousel page must not prove a dashboard CTA or menu option.
    if not assertion_target_relevant(step.target, action):
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=action.index,
            finding_types=findings,
            reason="target_role_mismatch",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    if proof_is_premature(step, expected, action, observed):
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=action.index,
            finding_types=findings,
            reason="premature_proof",
            matching_method=AssistMatchMethod.rejected_proposal,
            model_answer=decision.verdict.value,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
    return _record(
        step=step,
        status=AssistProposalStatus.accepted,
        matched_index=action.index,
        support_index=action.index,
        finding_types=findings,
        reason="episode_verified",
        matching_method=AssistMatchMethod.llm_assisted,
        model_answer=decision.verdict.value,
        confidence=decision.confidence,
        rationale=decision.rationale,
    )


def verify_step_against_episodes(
    step: ExpectedStep,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    episodes: list[ScreenEpisode],
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = EPISODE_VERIFY_MODEL,
    on_retry: RetryCallback | None = None,
    allow_llm: bool = True,
) -> AssistProposalRecord:
    """Verify one expected step against a causal episode window."""
    findings = sorted(
        findings_for_step(audit, step.index),
        key=lambda item: item.value,
    )
    baseline_matched = next(
        (
            verdict.matched_observed_index
            for verdict in audit.step_verdicts
            if verdict.expected_step_index == step.index
        ),
        None,
    )
    candidates = candidate_episodes_for_step(
        step, expected, observed, audit, episodes
    )
    if not candidates:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=None,
            finding_types=findings,
            reason="no_candidate_episodes",
            matching_method=AssistMatchMethod.rejected_proposal,
        )

    non_ambient = [
        episode for episode in candidates if not episode.is_ambient_system
    ] or candidates
    # Skip fatal-failure episodes for positive proof; do not fail the whole step
    # just because an intermediate form shows ordinary UI chrome.
    fatal_episodes = [
        episode
        for episode in non_ambient
        if episode_has_negative_invariant(step, episode, observed)
    ]
    viable_candidates = [
        episode for episode in non_ambient if episode not in fatal_episodes
    ]

    # Scan the full forward window first. Prefer a passive proof (assert/navigate)
    # over an interaction that merely opens the screen, so a Terms *click* cannot
    # consume the Terms *page* assertion and starve the preceding click step.
    fast_proofs = []
    for episode in viable_candidates:
        proof = fast_path_verify_episode(
            step, episode, observed, expected=expected
        )
        if proof is not None:
            fast_proofs.append((proof, episode))
    if fast_proofs:
        passive = [
            item for item in fast_proofs if is_passive_observation(item[0].action)
        ]
        chosen_proof, chosen_episode = min(
            passive or fast_proofs,
            key=lambda item: item[0].index,
        )
        return _record(
            step=step,
            status=AssistProposalStatus.accepted,
            matched_index=chosen_proof.index,
            support_index=chosen_proof.index,
            finding_types=findings,
            reason="episode_verified",
            matching_method=AssistMatchMethod.deterministic,
            rationale=(
                f"Assertion satisfied by episode {chosen_episode.episode_id} "
                f"action {chosen_proof.index}"
            ),
        )

    # No success proof: a genuine failure cue in the forward window is decisive.
    if fatal_episodes:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=None,
            finding_types=findings,
            reason="negative_invariant",
            matching_method=AssistMatchMethod.deterministic,
            rationale=(
                f"Episode {fatal_episodes[0].episode_id} contains a contradictory "
                "failure outcome"
            ),
        )

    if not allow_llm:
        return _record(
            step=step,
            status=AssistProposalStatus.rejected,
            matched_index=baseline_matched,
            support_index=None,
            finding_types=findings,
            reason="insufficient_evidence",
            matching_method=AssistMatchMethod.unresolved,
        )

    # Expand LLM view across the full forward assertion window, preferring
    # episodes whose visible text overlaps the expected assertion.
    llm_episodes = select_episodes_for_llm(
        step,
        viable_candidates or non_ambient,
    )
    try:
        decision = propose_episode_verification(
            step,
            llm_episodes,
            openai_client=openai_client,
            model=model,
            on_retry=on_retry,
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
            "stage3.episode_verify llm_failed error_type=%s", type(exc).__name__
        )
        return _record(
            step=step,
            status=AssistProposalStatus.failed,
            matched_index=baseline_matched,
            support_index=None,
            finding_types=findings,
            reason=f"llm_{type(exc).__name__}",
            matching_method=AssistMatchMethod.rejected_proposal,
        )
    return _validate_llm_decision(
        step,
        decision,
        llm_episodes,
        observed,
        findings,
        baseline_matched,
        expected,
    )


def verify_trace_against_episodes(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    episodes: list[ScreenEpisode],
    *,
    openai_client: instructor.Instructor | None = None,
    model: str = EPISODE_VERIFY_MODEL,
    on_retry: RetryCallback | None = None,
    allow_llm: bool = True,
) -> tuple[list[AssistProposalRecord], int]:
    """Run assertion episode verification, then interaction temporal remaps.

    Assertion steps establish verified anchors. Interaction steps with
    ``wrong_target`` / ``missing_action`` then search episode actions inside
    the window between the previous verified step and the next verified
    assertion — without locking to DP's single matched index.

    Returns (proposals, llm_request_count).
    """
    steps = steps_needing_episode_verification(expected, audit)
    proposals: list[AssistProposalRecord] = []
    llm_requests = 0
    for step in steps:
        candidates = candidate_episodes_for_step(
            step, expected, observed, audit, episodes
        )
        non_ambient = [
            episode for episode in candidates if not episode.is_ambient_system
        ] or candidates
        viable = [
            episode
            for episode in non_ambient
            if not episode_has_negative_invariant(step, episode, observed)
        ]
        needs_llm = True
        for episode in viable:
            if fast_path_verify_episode(
                step, episode, observed, expected=expected
            ) is not None:
                needs_llm = False
                break
        if needs_llm and any(
            episode_has_negative_invariant(step, episode, observed)
            for episode in non_ambient
        ):
            # Fatal failure without success proof → deterministic reject, no LLM.
            needs_llm = False
        if needs_llm and allow_llm and (viable or non_ambient):
            llm_requests += 1
        proposals.append(
            verify_step_against_episodes(
                step,
                expected,
                observed,
                audit,
                episodes,
                openai_client=openai_client,
                model=model,
                on_retry=on_retry,
                allow_llm=allow_llm,
            )
        )

    accepted_assertion_remaps = {
        record.expected_step_index: terminal
        for record in proposals
        if record.status == AssistProposalStatus.accepted
        for terminal in (
            record.supporting_observed_index
            or record.completion_observed_index
            or record.matched_observed_index,
        )
        if terminal is not None
    }
    interaction_proposals = remap_interactions_against_episodes(
        expected,
        observed,
        audit,
        episodes,
        accepted_assertion_remaps=accepted_assertion_remaps,
    )
    proposals.extend(interaction_proposals)
    return proposals, llm_requests
