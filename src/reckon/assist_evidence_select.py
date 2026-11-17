"""Deterministic evidence-window retrieval for experimental Stage 3 evidence-aware assist.

Does not change align(). Candidates are ranked proposals only — never auto-accepted.
"""

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from reckon.assist_select import AssistEvidenceEvent, texts_have_protected_contradiction
from reckon.episodes import cluster_observed_episodes, expand_indices_to_episode
from reckon.schema import (
    ActionType,
    AssistCandidateKind,
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedAction,
    ObservedTrace,
    ScreenEpisode,
)
from reckon.tolerance import ASSERTION_ACTIONS, actions_compatible, assertion_satisfied
from reckon.workflow_reentry import reentry_floor_for_step

# Cap candidates shown to the model per expected step.
MAX_EVIDENCE_WINDOW_EVENTS = 12

# Soft initiation allow-list when expected action is coarse `other`.
_INITIATION_ACTIONS = frozenset({
    ActionType.click,
    ActionType.navigate,
    ActionType.other,
})

_COARSE_ACTIONS = frozenset({
    ActionType.other,
    ActionType.navigate,
})


@dataclass(frozen=True)
class EvidenceBundleCandidate:
    """One coarse expected step with a retrieved observation window."""

    candidate_id: str
    kind: AssistCandidateKind
    expected_step_index: int
    matched_observed_index: int | None
    finding_types: tuple[FindingType, ...]
    expected_action: str
    expected_text: str
    expected_assertion: str | None
    evidence_events: tuple[AssistEvidenceEvent, ...] = field(default_factory=tuple)
    # Baseline passed but assertion text is not satisfied on the matched action.
    suspicious_pass: bool = False


def _is_coarse_step(step: ExpectedStep) -> bool:
    if step.action in _COARSE_ACTIONS:
        return True
    if step.assertion is not None:
        return True
    raw = (step.raw_text or "").strip()
    return len(raw.split()) >= 6 and step.action == ActionType.other


def _event_from_action(action: ObservedAction) -> AssistEvidenceEvent:
    evidence = action.assertion_evidence
    return AssistEvidenceEvent(
        observed_index=action.index,
        timestamp_ms=action.timestamp_ms,
        action=action.action.value,
        target=action.target,
        visible_text=evidence.visible_text if evidence else None,
        evidence_description=evidence.description if evidence else None,
        screen_description=action.screen_description,
    )


def _text_overlap_score(step: ExpectedStep, action: ObservedAction) -> float:
    parts = [
        step.target or "",
        step.raw_text or "",
        step.assertion or "",
    ]
    expected_blob = " ".join(p for p in parts if p).lower()
    observed_parts = [
        action.target or "",
        action.screen_description or "",
    ]
    if action.assertion_evidence is not None:
        observed_parts.append(action.assertion_evidence.description or "")
        observed_parts.append(action.assertion_evidence.visible_text or "")
    observed_blob = " ".join(p for p in observed_parts if p).lower()
    if not expected_blob or not observed_blob:
        return 0.0
    return fuzz.token_set_ratio(expected_blob, observed_blob) / 100.0


def _eligible_for_window(step: ExpectedStep, action: ObservedAction) -> bool:
    if actions_compatible(step.action, action.action):
        return True
    if step.action == ActionType.other and action.action in _INITIATION_ACTIONS:
        return True
    if step.assertion is not None and action.assertion_evidence is not None:
        return True
    if step.assertion is not None and action.action in ASSERTION_ACTIONS:
        return True
    return _text_overlap_score(step, action) >= 0.35


def _rank_key(step: ExpectedStep, action: ObservedAction) -> tuple:
    has_evidence = 1 if action.assertion_evidence is not None else 0
    satisfied = (
        1
        if step.assertion is not None
        and assertion_satisfied(
            step.assertion, action, expected_target=step.target
        )
        else 0
    )
    overlap = _text_overlap_score(step, action)
    compatible = 1 if actions_compatible(step.action, action.action) else 0
    # Prefer evidence that can satisfy assertions; never prefer spinner-like distractors
    # purely by order — ranking only, acceptance is elsewhere.
    spinner_penalty = (
        1
        if action.target and "spinner" in action.target.lower()
        else 0
    )
    return (-satisfied, -has_evidence, -compatible, -overlap, spinner_penalty, action.index)


def _reliable_match_index(verdict) -> int | None:
    """Use only passed matches as window anchors — drifted DP fails are ignored."""
    if verdict is None or not verdict.passed:
        return None
    if FindingType.wrong_target in (verdict.finding_types or ()):
        return None
    if FindingType.assertion_not_observed in (verdict.finding_types or ()):
        return None
    return verdict.matched_observed_index


def _window_bounds(
    step: ExpectedStep,
    expected: ExpectedTrace,
    audit: AuditResult,
    observed: ObservedTrace,
) -> tuple[int, int]:
    """Inclusive observed-index bounds for this step's evidence window.

    Bounds come from surrounding *reliable* matches only. Incorrect DP pairings
    must not shrink the window away from the real evidence.
    """
    verdicts = {v.expected_step_index: v for v in audit.step_verdicts}
    min_index = 1
    max_index = observed.actions[-1].index if observed.actions else 1

    for prev in reversed([s for s in expected.steps if s.index < step.index]):
        anchor = _reliable_match_index(verdicts.get(prev.index))
        if anchor is not None:
            min_index = anchor
            break

    for nxt in [s for s in expected.steps if s.index > step.index]:
        anchor = _reliable_match_index(verdicts.get(nxt.index))
        if anchor is not None:
            max_index = anchor
            break

    if min_index > max_index:
        return 1, observed.actions[-1].index if observed.actions else 1
    # Deeplink / URL re-entry can raise the floor when assertion proof follows it,
    # so early noise + bad DP anchors do not dominate the window.
    min_index = reentry_floor_for_step(
        step, observed, min_index=min_index, max_index=max_index
    )
    if min_index > max_index:
        return 1, observed.actions[-1].index if observed.actions else 1
    return min_index, max_index


def retrieve_evidence_window(
    step: ExpectedStep,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    *,
    matched_observed_index: int | None = None,
    episodes: list[ScreenEpisode] | None = None,
) -> list[AssistEvidenceEvent]:
    """Return a capped, ranked window of supporting observations for one step.

    When episodes are provided, the DP-matched observation's whole screen episode
    is retained so Claude can label micro-actions that share one UI state.
    """
    lo, hi = _window_bounds(step, expected, audit, observed)
    action_by_index = {action.index: action for action in observed.actions}
    must_keep: set[int] = set()
    if matched_observed_index is not None:
        must_keep.add(matched_observed_index)
        if episodes:
            span = expand_indices_to_episode(episodes, matched_observed_index)
            if span is not None:
                ep_lo, ep_hi = span
                for action in observed.actions:
                    if ep_lo <= action.index <= ep_hi and lo <= action.index <= hi:
                        must_keep.add(action.index)

    pool = [
        action
        for action in observed.actions
        if lo <= action.index <= hi and (
            _eligible_for_window(step, action) or action.index in must_keep
        )
    ]
    # DP-independent: also pull high-overlap assertion evidence from outside the
    # reliable-anchor window so drifted early matches cannot hide later proof.
    if step.assertion is not None:
        in_pool = {action.index for action in pool}
        for action in observed.actions:
            if action.index in in_pool:
                continue
            if action.assertion_evidence is None and action.action not in ASSERTION_ACTIONS:
                continue
            if _text_overlap_score(step, action) < 0.55:
                continue
            pool.append(action)
            in_pool.add(action.index)
    pool.sort(key=lambda action: _rank_key(step, action))
    selected = pool[:MAX_EVIDENCE_WINDOW_EVENTS]
    selected_indices = {action.index for action in selected}
    for index in sorted(must_keep):
        if index in selected_indices:
            continue
        action = action_by_index.get(index)
        if action is not None and lo <= action.index <= hi:
            selected.append(action)
            selected_indices.add(index)
    selected.sort(key=lambda action: action.index)
    return [_event_from_action(action) for action in selected]


def select_evidence_bundle_candidates(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    *,
    episodes: list[ScreenEpisode] | None = None,
) -> list[EvidenceBundleCandidate]:
    """Select coarse expected steps that may benefit from evidence-aware assist."""
    if not observed.actions:
        return []

    episode_list = (
        episodes if episodes is not None else cluster_observed_episodes(observed)
    )
    step_by_index = {step.index: step for step in expected.steps}
    action_by_index = {action.index: action for action in observed.actions}
    findings_by_step: dict[int, set[FindingType]] = {}
    for finding in audit.findings:
        if finding.expected_step_index is None:
            continue
        findings_by_step.setdefault(finding.expected_step_index, set()).add(finding.type)

    candidates: list[EvidenceBundleCandidate] = []
    next_id = 1
    for verdict in audit.step_verdicts:
        step = step_by_index.get(verdict.expected_step_index)
        if step is None or not _is_coarse_step(step):
            continue

        matched = (
            action_by_index.get(verdict.matched_observed_index)
            if verdict.matched_observed_index is not None
            else None
        )
        pair_findings = findings_by_step.get(step.index, set())
        suspicious_pass = False
        if (
            verdict.passed
            and step.assertion is not None
            and matched is not None
            and not assertion_satisfied(
                step.assertion, matched, expected_target=step.target
            )
        ):
            suspicious_pass = True

        needs_help = bool(
            pair_findings
            & {
                FindingType.wrong_target,
                FindingType.assertion_not_observed,
                FindingType.missing_action,
            }
        ) or suspicious_pass
        if not needs_help:
            continue

        if (
            matched is not None
            and step.target is not None
            and matched.target is not None
            and texts_have_protected_contradiction(step.target, matched.target)
            and FindingType.wrong_target in pair_findings
            and not suspicious_pass
            and FindingType.assertion_not_observed not in pair_findings
        ):
            # Pure protected target contradictions stay with wrong_target assist.
            continue

        window = retrieve_evidence_window(
            step,
            expected,
            observed,
            audit,
            matched_observed_index=verdict.matched_observed_index,
            episodes=episode_list,
        )
        if not window:
            continue

        expected_text = step.assertion or step.target or step.raw_text or ""
        candidates.append(
            EvidenceBundleCandidate(
                candidate_id=f"E{next_id}",
                kind=AssistCandidateKind.evidence_bundle,
                expected_step_index=step.index,
                matched_observed_index=verdict.matched_observed_index,
                finding_types=tuple(sorted(pair_findings, key=lambda ft: ft.value)),
                expected_action=step.action.value,
                expected_text=expected_text,
                expected_assertion=step.assertion,
                evidence_events=tuple(window),
                suspicious_pass=suspicious_pass,
            )
        )
        next_id += 1

    return candidates
