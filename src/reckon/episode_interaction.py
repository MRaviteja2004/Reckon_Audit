"""Deterministic remapping of click/select/type_text steps onto episode actions.

Assertions may already be remapped by episode verification. Interaction steps
still suffer sequential DP drift: a wrong earlier pair cascades into
``wrong_target`` / ``missing_action`` on later clicks. This module searches
observed actions in the temporal window between trusted anchors (previous
verified step → next verified assertion), ignoring each step's own drifted
DP ``matched_observed_index``.
"""

import logging
import re
from collections.abc import Mapping

from rapidfuzz import fuzz

from reckon.schema import (
    ActionType,
    AssistCandidateKind,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingType,
    ObservedAction,
    ObservedTrace,
    ScreenEpisode,
    StepVerdict,
)
from reckon.tolerance import (
    ASSERTION_ACTIONS,
    FUZZY_THRESHOLD,
    actions_compatible,
    finding_severity,
    is_choice_control_target,
    values_equivalent,
    verdict_for,
)
from reckon.auth_cta import (
    AUTH_CTA_RECOVERY_REASON,
    auth_cta_canonical,
    find_auth_cta_synonym_match,
)
from reckon.typeahead import find_typeahead_selection_proof

# Re-export for callers/tests that historically imported from this module.
__all__ = [
    "INTERACTION_ACTIONS",
    "REMAP_MISSING_ACTION_FLOOR",
    "REMAP_WRONG_TARGET_FLOOR",
    "apply_deterministic_auth_cta_recovery",
    "apply_deterministic_typeahead_recovery",
    "find_matching_interaction",
    "is_choice_control_target",
    "is_interaction_action",
    "is_passive_observation",
    "remap_interactions_against_episodes",
    "remap_targets_compatible",
    "shares_canonical_ui_nouns",
    "target_similarity",
]

TYPEAHEAD_RECOVERY_REASON = "typeahead_selection_recovery"
APP_LAUNCH_RECOVERY_REASON = "app_launch_recovery"

logger = logging.getLogger(__name__)

INTERACTION_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.click,
        ActionType.select,
        ActionType.type_text,
    }
)

# Coarse "open the X app" expectations (action=other) may bind to app-icon clicks.
# Require a discriminative noun beyond bare "app" so "Log into the app" stays
# assertion-driven (home/offline proof) rather than binding the launcher icon.
_APP_LAUNCH_TARGET_RE = re.compile(r"\b(?:app|application)\b", re.IGNORECASE)
_APP_LAUNCH_GENERIC_TOKENS = frozenset({"app", "application", "the", "a", "an"})
_APP_LAUNCH_OBS_RE = re.compile(
    r"\b(?:app\s+icon|application\s+icon|welcome\s+to\s+the|launching)\b",
    re.IGNORECASE,
)
# App-icon recovery is only for Mentorship-style split macros where account
# creation is a sibling step. A lone "Open the driver app" other-step stays
# initiation-only so coarse-login evidence goldens are not false-cleared.
_APP_LAUNCH_ACCOUNT_TARGET_RE = re.compile(
    r"\b(?:test\s+)?account\b",
    re.IGNORECASE,
)

_CHOICE_CONTROL_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.click, ActionType.select}
)

# Floors for remapping drifted interactions onto observed controls.
REMAP_WRONG_TARGET_FLOOR = 60.0
REMAP_MISSING_ACTION_FLOOR = 80.0

# Scenario authors often write "join mentorship/Done cta" for alternatives.
_TARGET_ALTERNATIVE_SPLIT_RE = re.compile(r"\s*/\s*")

# Noise tokens that should not dominate substring matching.
_TARGET_NOISE_RE = re.compile(
    r"\b(?:button|btn|cta|link|control|field|input)\b",
    re.IGNORECASE,
)

# Explicit synonym groups for remapping (terms acceptance ↔ agree checkbox, …).
_REMAP_TARGET_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({
        "terms",
        "conditions",
        "tos",
        "agreement",
        "accept",
        "acceptance",
        "agree",
        "checkbox",
    }),
    frozenset({"continue", "next", "proceed"}),
    frozenset({"submit", "confirm", "save"}),
    frozenset({"cancel", "dismiss", "close"}),
    frozenset({"language", "locale", "lang"}),
)

# Controls that must never remap onto each other even when fuzzy score clears.
_REMAP_INCOMPATIBLE_ROLE_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset({
            "terms",
            "conditions",
            "tos",
            "agreement",
            "accept",
            "acceptance",
            "agree",
            "checkbox",
        }),
        frozenset({"continue", "next", "proceed", "submit", "confirm", "done"}),
    ),
    (
        frozenset({"cancel", "dismiss", "close"}),
        frozenset({"continue", "next", "proceed", "submit", "confirm", "done"}),
    ),
)

_INTERACTION_FINDINGS: frozenset[FindingType] = frozenset(
    {
        FindingType.wrong_target,
        FindingType.missing_action,
    }
)


def is_interaction_action(action: ActionType) -> bool:
    return action in INTERACTION_ACTIONS


def is_passive_observation(action: ActionType) -> bool:
    """True for observations that are not physical user interactions."""
    return action not in INTERACTION_ACTIONS


def _scenario_has_split_account_step(
    expected: ExpectedTrace,
    step: ExpectedStep,
) -> bool:
    """True when another expected step names account creation/use separately."""
    for other in expected.steps:
        if other.index == step.index or not other.target:
            continue
        if _APP_LAUNCH_ACCOUNT_TARGET_RE.search(other.target):
            return True
    return False


def _is_app_launch_expectation(
    step: ExpectedStep,
    expected: ExpectedTrace | None = None,
) -> bool:
    """True for coarse other-steps that name launching a *specific* application.

    Requires a sibling account step so split macros (account + driver app)
    can bind the launcher icon, while a lone "Open the driver app" other-step
    remains unrecovered initiation for coarse-login evidence cases.
    """
    if step.action != ActionType.other or not step.target:
        return False
    if not _APP_LAUNCH_TARGET_RE.search(step.target):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", step.target.lower()))
    if not (tokens - _APP_LAUNCH_GENERIC_TOKENS):
        return False
    if expected is None or not _scenario_has_split_account_step(expected, step):
        return False
    return True


def steps_needing_interaction_remap(
    expected: ExpectedTrace,
    audit: AuditResult,
) -> list[int]:
    """Return expected indices for interaction steps with wrong_target/missing."""
    needed: list[int] = []
    for verdict in audit.step_verdicts:
        step = next(
            (s for s in expected.steps if s.index == verdict.expected_step_index),
            None,
        )
        if step is None:
            continue
        if step.action not in INTERACTION_ACTIONS and not _is_app_launch_expectation(
            step, expected
        ):
            continue
        if any(ft in _INTERACTION_FINDINGS for ft in verdict.finding_types):
            needed.append(verdict.expected_step_index)
    return needed


def find_app_launch_match(
    step: ExpectedStep,
    observed: ObservedTrace,
    *,
    window: tuple[int, int],
    claimed_obs_indices: set[int],
    expected: ExpectedTrace | None = None,
) -> ObservedAction | None:
    """Bind an app-launch ``other`` step to a click/navigate on that app.

    Requires shared discriminative nouns (e.g. driver) plus an app-icon /
    welcome / launching cue. Does not infer launch from a later home/map screen.
    """
    if not _is_app_launch_expectation(step, expected):
        return None
    start, end = window
    expected_target = step.target or ""
    candidates: list[tuple[float, ObservedAction]] = []
    for action in observed.actions:
        if action.index < start or action.index > end:
            continue
        if action.index in claimed_obs_indices:
            continue
        if action.action not in {ActionType.click, ActionType.navigate}:
            continue
        blob = _action_context_blob(action)
        if not _APP_LAUNCH_OBS_RE.search(blob):
            continue
        if not shares_canonical_ui_nouns(expected_target, action.target or ""):
            # Allow "driver app" ↔ "Uber Driver app icon" via target text alone.
            if not remap_targets_compatible(expected_target, action.target or ""):
                continue
        score = target_similarity(expected_target, action.target or "")
        if score < REMAP_MISSING_ACTION_FLOOR and not remap_targets_compatible(
            expected_target, action.target or ""
        ):
            continue
        if score < REMAP_WRONG_TARGET_FLOOR:
            continue
        candidates.append((score, action))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].index))
    return candidates[0][1]


def _verdict_map(audit: AuditResult) -> dict[int, StepVerdict]:
    return {
        verdict.expected_step_index: verdict for verdict in audit.step_verdicts
    }


def _is_trusted_anchor(
    step_index: int,
    *,
    expected: ExpectedTrace,
    verdicts: Mapping[int, StepVerdict],
    accepted_remaps: Mapping[int, int],
    needs_remap: set[int],
) -> bool:
    """True when this step's matched index may bound an interaction window.

    Drifted interaction steps (wrong_target / missing_action) are never anchors:
    their DP indices often lie *outside* the chronological bracket and would
    collapse or shift the search window.
    """
    if step_index in accepted_remaps:
        return True
    if step_index in needs_remap:
        return False
    verdict = verdicts.get(step_index)
    if verdict is None or verdict.matched_observed_index is None:
        return False
    if verdict.passed:
        return True
    step = next((s for s in expected.steps if s.index == step_index), None)
    if step is None:
        return False
    # Unremapped assertion failures are not reliable chronological anchors.
    if step.action in ASSERTION_ACTIONS or step.assertion is not None:
        return False
    if any(ft in _INTERACTION_FINDINGS for ft in verdict.finding_types):
        return False
    return True


def _trusted_matched_index_map(
    expected: ExpectedTrace,
    audit: AuditResult,
    *,
    accepted_remaps: Mapping[int, int] | None = None,
    needs_remap: set[int] | None = None,
) -> dict[int, int]:
    """Observed indices that may serve as chronological anchors only."""
    remaps = dict(accepted_remaps or {})
    pending = needs_remap if needs_remap is not None else set(
        steps_needing_interaction_remap(expected, audit)
    )
    verdicts = _verdict_map(audit)
    trusted: dict[int, int] = {}
    for step in expected.steps:
        if step.index in remaps:
            trusted[step.index] = remaps[step.index]
            continue
        if not _is_trusted_anchor(
            step.index,
            expected=expected,
            verdicts=verdicts,
            accepted_remaps=remaps,
            needs_remap=pending,
        ):
            continue
        verdict = verdicts[step.index]
        if verdict.matched_observed_index is not None:
            trusted[step.index] = verdict.matched_observed_index
    return trusted


def _previous_trusted_obs_index(
    step_index: int,
    expected: ExpectedTrace,
    trusted: Mapping[int, int],
) -> int | None:
    prior = [s.index for s in expected.steps if s.index < step_index and s.index in trusted]
    if not prior:
        return None
    return trusted[max(prior)]


def _next_trusted_assertion_obs_index(
    step_index: int,
    expected: ExpectedTrace,
    trusted: Mapping[int, int],
) -> int | None:
    later = [
        s.index
        for s in expected.steps
        if s.index > step_index
        and (s.action in ASSERTION_ACTIONS or s.assertion is not None)
        and s.index in trusted
    ]
    if not later:
        return None
    return trusted[min(later)]


def interaction_search_window(
    step_index: int,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    trusted: Mapping[int, int],
) -> tuple[int, int]:
    """Inclusive observed-index window between trusted anchors.

    The step's own DP ``matched_observed_index`` is never consulted. Only
    previously verified / remapped steps and the next verified assertion bound
    the search.
    """
    if not observed.actions:
        return (0, -1)
    lo = _previous_trusted_obs_index(step_index, expected, trusted)
    hi = _next_trusted_assertion_obs_index(step_index, expected, trusted)
    start = (lo + 1) if lo is not None else observed.actions[0].index
    end = (hi - 1) if hi is not None else observed.actions[-1].index
    if end < start:
        # Degenerate bracket (anchors inverted): still search the open interval
        # toward the assertion when one exists, else the remainder of the trace.
        if hi is not None and lo is not None:
            start = min(lo + 1, hi)
            end = max(lo + 1, hi) - 1 if hi > lo + 1 else hi
            if end < start:
                start, end = lo + 1, hi
        elif hi is not None:
            start, end = observed.actions[0].index, hi
        else:
            end = observed.actions[-1].index
    return (start, end)


def _normalize_target(text: str) -> str:
    cleaned = _TARGET_NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _target_variants(expected_target: str) -> list[str]:
    """Expand slash-separated alternatives and noise-stripped forms."""
    raw = (expected_target or "").strip()
    if not raw:
        return []
    parts = [raw, *_TARGET_ALTERNATIVE_SPLIT_RE.split(raw)]
    variants: list[str] = []
    seen: set[str] = set()
    for part in parts:
        candidate = part.strip()
        if not candidate:
            continue
        for form in (candidate, _normalize_target(candidate)):
            key = form.casefold()
            if form and key not in seen:
                seen.add(key)
                variants.append(form)
    return variants


def target_similarity(expected_target: str, observed_target: str) -> float:
    """0..100 fuzzy score; prefer partial_ratio substring matching."""
    obs = (observed_target or "").strip()
    if not obs:
        return 0.0
    obs_norm = _normalize_target(obs) or obs
    best = 0.0
    for variant in _target_variants(expected_target):
        if variant.casefold() in obs.casefold() or obs.casefold() in variant.casefold():
            return 100.0
        if (
            variant.casefold() in obs_norm.casefold()
            or obs_norm.casefold() in variant.casefold()
        ):
            return 100.0
        best = max(
            best,
            float(fuzz.partial_ratio(variant, obs)),
            float(fuzz.partial_ratio(variant, obs_norm)),
            float(fuzz.token_set_ratio(variant, obs)),
            float(fuzz.token_set_ratio(variant, obs_norm)),
        )
    return best


def _canonical_noun_tokens(text: str | None) -> set[str]:
    """Meaningful UI nouns after stripping structural noise tokens."""
    if not text:
        return set()
    cleaned = _TARGET_NOISE_RE.sub(" ", text)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", cleaned.lower())
        if len(token) > 1
    }


def shares_canonical_ui_nouns(expected_target: str, observed_target: str) -> bool:
    """True when expected and observed share at least one non-noise UI noun."""
    expected_nouns = _canonical_noun_tokens(expected_target)
    if not expected_nouns:
        return True
    return bool(expected_nouns & _canonical_noun_tokens(observed_target))


def _remap_alias_overlap(expected_target: str, observed_target: str) -> bool:
    """True when expected and observed share a configured alias group."""
    expected_nouns = _canonical_noun_tokens(expected_target)
    observed_nouns = _canonical_noun_tokens(observed_target)
    if not expected_nouns or not observed_nouns:
        return False
    for group in _REMAP_TARGET_ALIAS_GROUPS:
        if expected_nouns & group and observed_nouns & group:
            return True
    return False


def _remap_roles_incompatible(expected_target: str, observed_target: str) -> bool:
    """True when expected/observed fall into known-incompatible control roles."""
    expected_nouns = _canonical_noun_tokens(expected_target)
    observed_nouns = _canonical_noun_tokens(observed_target)
    for left, right in _REMAP_INCOMPATIBLE_ROLE_PAIRS:
        if expected_nouns & left and observed_nouns & right:
            return True
        if expected_nouns & right and observed_nouns & left:
            return True
    return False


def remap_targets_compatible(
    expected_target: str,
    observed_target: str,
    *,
    context: str | None = None,
) -> bool:
    """Require shared discriminative nouns or configured aliases for remapping.

    Rejects incompatible controls (e.g. Terms acceptance ↔ Continue) even when
    temporal windows and fuzzy similarity thresholds would otherwise pass.

    ``context`` may include screen/evidence text so a choice-control option
    (e.g. English checkbox on a Language dropdown picker) can still bind when
    the expected field label appears around the control.
    """
    if not (expected_target or "").strip():
        return True
    if not (observed_target or "").strip():
        return False
    if _remap_roles_incompatible(expected_target, observed_target):
        return False
    if _remap_alias_overlap(expected_target, observed_target):
        return True
    if shares_canonical_ui_nouns(expected_target, observed_target):
        return True
    if context:
        blob = context.casefold()
        for variant in _target_variants(expected_target):
            if variant.casefold() in blob:
                return True
    return False


def _action_context_blob(action: ObservedAction) -> str:
    parts = [action.target or "", action.screen_description or ""]
    if action.assertion_evidence is not None:
        parts.append(action.assertion_evidence.visible_text or "")
        parts.append(action.assertion_evidence.description or "")
    return " ".join(parts)


def choice_control_match_score(
    expected_action: ActionType,
    expected_target: str,
    action: ObservedAction,
) -> float:
    """Score select↔click pairs when *both* targets denote choice controls."""
    if expected_action not in _CHOICE_CONTROL_ACTIONS:
        return 0.0
    if action.action not in _CHOICE_CONTROL_ACTIONS:
        return 0.0
    if not is_choice_control_target(expected_target):
        return 0.0
    if not is_choice_control_target(action.target):
        return 0.0
    base = target_similarity(expected_target, action.target or "")
    blob = _action_context_blob(action)
    for variant in _target_variants(expected_target):
        if variant.casefold() in blob.casefold():
            return max(base, float(FUZZY_THRESHOLD))
    return base


def _actions_in_episodes(
    observed: ObservedTrace,
    episodes: list[ScreenEpisode],
) -> list[ObservedAction]:
    """Observed actions that fall inside at least one episode span."""
    if not episodes:
        return list(observed.actions)
    covered: set[int] = set()
    for episode in episodes:
        for index in range(episode.start_index, episode.end_index + 1):
            covered.add(index)
    return [action for action in observed.actions if action.index in covered]


def find_matching_interaction(
    *,
    expected_action: ActionType,
    expected_target: str,
    expected_input_value: str | None,
    observed: ObservedTrace,
    episodes: list[ScreenEpisode],
    window: tuple[int, int],
    claimed_obs_indices: set[int],
    min_score: float,
    require_canonical_nouns: bool,
) -> ObservedAction | None:
    """Best semantically matching interaction inside the window, if any.

    Searches every in-window action. The step's drifted DP match is irrelevant
    here — callers must not pass it as a hard constraint.
    """
    start, end = window
    if end < start:
        return None

    candidates: list[tuple[float, ObservedAction]] = []
    for action in _actions_in_episodes(observed, episodes):
        if action.index < start or action.index > end:
            continue
        if action.index in claimed_obs_indices:
            continue
        if not actions_compatible(
            expected_action,
            action.action,
            allow_choice_control=True,
            expected_target=expected_target,
            observed_target=action.target,
        ):
            continue
        if action.action not in INTERACTION_ACTIONS:
            continue
        if expected_input_value is not None:
            if action.input_value is None or not values_equivalent(
                expected_input_value, action.input_value
            ):
                continue
        sort_score = float(
            fuzz.token_sort_ratio(
                (expected_target or "").lower(),
                (action.target or "").lower(),
            )
        )
        similarity = target_similarity(expected_target, action.target or "")
        choice_score = choice_control_match_score(
            expected_action, expected_target, action
        )
        score = max(sort_score, similarity, choice_score)
        # Alias overlap is semantic compatibility even when raw fuzzy is weak
        # (Terms acceptance ↔ I agree Terms checkbox ~43).
        if _remap_alias_overlap(expected_target, action.target or ""):
            score = max(score, float(FUZZY_THRESHOLD))
        context_blob = _action_context_blob(action)
        # Always require target compatibility for remapping. Fuzzy floors alone
        # previously rebound Terms acceptance onto Continue (partial_ratio ~62).
        if not remap_targets_compatible(
            expected_target,
            action.target or "",
            context=context_blob,
        ):
            continue
        if score < min_score:
            continue
        if require_canonical_nouns and not shares_canonical_ui_nouns(
            expected_target, action.target or ""
        ):
            # Alias overlap or choice-control context may still clear
            # missing_action when the field label surrounds the option.
            if (
                choice_score < min_score
                and not _remap_alias_overlap(
                    expected_target, action.target or ""
                )
            ):
                continue
        candidates.append((score, action))

    if not candidates:
        return None
    # Prefer stronger fuzzy score; earlier index breaks ties (chronological).
    candidates.sort(key=lambda item: (item[0], -item[1].index), reverse=True)
    return candidates[0][1]


def apply_deterministic_typeahead_recovery(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
) -> AuditResult:
    """Recover missing/wrong-target selects via proven typeahead sequences.

    Reuses ``find_typeahead_selection_proof`` and the same trusted-anchor windows
    as interaction remapping. Does not change ``actions_compatible`` or DP
    pairing for steps without a valid proof. Recovered observations are claimed
    so they cannot satisfy a later expected step.
    """
    candidates = [
        step
        for step in expected.steps
        if step.action == ActionType.select and step.input_value
    ]
    if not candidates:
        return audit

    verdicts = _verdict_map(audit)
    recoverable: list[int] = []
    for step in candidates:
        verdict = verdicts.get(step.index)
        if verdict is None or verdict.passed:
            continue
        if not any(ft in _INTERACTION_FINDINGS for ft in verdict.finding_types):
            continue
        recoverable.append(step.index)
    if not recoverable:
        return audit

    needs_remap = set(recoverable)
    trusted = _trusted_matched_index_map(
        expected,
        audit,
        accepted_remaps={},
        needs_remap=needs_remap,
    )
    claimed = set(trusted.values())
    action_by_index = {action.index: action for action in observed.actions}
    recoveries: dict[int, tuple[int, frozenset[int]]] = {}

    for step_index in sorted(recoverable):
        step = next(s for s in expected.steps if s.index == step_index)
        window = interaction_search_window(step_index, expected, observed, trusted)
        start, end = window
        if end < start:
            continue
        proof = find_typeahead_selection_proof(
            step,
            observed,
            window=window,
            claimed_obs_indices=claimed,
        )
        if proof is None:
            continue
        if proof.claimed_indices & claimed:
            continue
        if any(index < start or index > end for index in proof.claimed_indices):
            continue
        terminal = action_by_index.get(proof.terminal_index)
        if terminal is None:
            continue
        claimed.update(proof.claimed_indices)
        trusted[step_index] = proof.terminal_index
        recoveries[step_index] = (proof.terminal_index, proof.claimed_indices)
        logger.info(
            "stage3.typeahead_baseline_recovery step=%s terminal_obs=%s "
            "type=%s click=%s confirm=%s window=%s..%s",
            step_index,
            proof.terminal_index,
            proof.type_index,
            proof.click_index,
            proof.confirm_index,
            start,
            end,
        )

    if not recoveries:
        return audit
    return _apply_interaction_recoveries(
        expected,
        audit,
        recoveries=recoveries,
        action_by_index=action_by_index,
        recovery_reason=TYPEAHEAD_RECOVERY_REASON,
    )


def apply_deterministic_auth_cta_recovery(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
) -> AuditResult:
    """Clear click wrong_target/missing when auth CTAs are explicitly synonymous.

    Sign in ↔ Login only, click↔click, inside trusted-anchor windows. Does not
    change ``actions_compatible`` or equate Cancel/Submit/Continue.
    """
    candidates = [
        step
        for step in expected.steps
        if step.action == ActionType.click
        and auth_cta_canonical(step.target) is not None
    ]
    if not candidates:
        return audit

    verdicts = _verdict_map(audit)
    recoverable: list[int] = []
    for step in candidates:
        verdict = verdicts.get(step.index)
        if verdict is None or verdict.passed:
            continue
        if not any(ft in _INTERACTION_FINDINGS for ft in verdict.finding_types):
            continue
        recoverable.append(step.index)
    if not recoverable:
        return audit

    needs_remap = set(recoverable)
    trusted = _trusted_matched_index_map(
        expected,
        audit,
        accepted_remaps={},
        needs_remap=needs_remap,
    )
    claimed = set(trusted.values())
    action_by_index = {action.index: action for action in observed.actions}
    recoveries: dict[int, tuple[int, frozenset[int]]] = {}

    for step_index in sorted(recoverable):
        step = next(s for s in expected.steps if s.index == step_index)
        window = interaction_search_window(step_index, expected, observed, trusted)
        start, end = window
        if end < start:
            continue
        preferred = (
            verdicts[step_index].matched_observed_index
            if step_index in verdicts
            else None
        )
        match = find_auth_cta_synonym_match(
            step,
            observed,
            window=window,
            claimed_obs_indices=claimed,
            preferred_obs_index=preferred,
        )
        if match is None:
            continue
        if match.index < start or match.index > end or match.index in claimed:
            continue
        claimed.add(match.index)
        trusted[step_index] = match.index
        recoveries[step_index] = (match.index, frozenset({match.index}))
        logger.info(
            "stage3.auth_cta_baseline_recovery step=%s -> obs=%s "
            "expected=%r observed=%r window=%s..%s",
            step_index,
            match.index,
            step.target,
            match.target,
            start,
            end,
        )

    if not recoveries:
        return audit
    return _apply_interaction_recoveries(
        expected,
        audit,
        recoveries=recoveries,
        action_by_index=action_by_index,
        recovery_reason=AUTH_CTA_RECOVERY_REASON,
    )


def _apply_interaction_recoveries(
    expected: ExpectedTrace,
    audit: AuditResult,
    *,
    recoveries: dict[int, tuple[int, frozenset[int]]],
    action_by_index: dict[int, ObservedAction],
    recovery_reason: str,
) -> AuditResult:
    """Shared finding/verdict rewrite for deterministic baseline recoveries."""
    step_by_index = {step.index: step for step in expected.steps}
    clear_types = {FindingType.missing_action, FindingType.wrong_target}
    new_findings: list[Finding] = []
    for finding in audit.findings:
        if (
            finding.expected_step_index in recoveries
            and finding.type in clear_types
        ):
            continue
        if (
            finding.expected_step_index in recoveries
            and finding.observed_action_index is not None
        ):
            terminal_index = recoveries[finding.expected_step_index][0]
            support = action_by_index.get(terminal_index)
            new_findings.append(
                finding.model_copy(
                    update={
                        "observed_action_index": terminal_index,
                        "evidence_timestamp_ms": (
                            support.timestamp_ms if support is not None else None
                        ),
                    }
                )
            )
            continue
        new_findings.append(finding)

    new_verdicts: list[StepVerdict] = []
    for verdict in audit.step_verdicts:
        if verdict.expected_step_index not in recoveries:
            new_verdicts.append(verdict)
            continue
        terminal_index, _claimed = recoveries[verdict.expected_step_index]
        support = action_by_index[terminal_index]
        remaining = [
            ft for ft in verdict.finding_types if ft not in clear_types
        ]
        new_verdicts.append(
            StepVerdict(
                expected_step_index=verdict.expected_step_index,
                passed=len(remaining) == 0,
                matched_observed_index=terminal_index,
                evidence_timestamp_ms=support.timestamp_ms,
                finding_types=remaining,
                cleared_by_assist=False,
                assist_confidence=None,
                recovery_reason=recovery_reason,
            )
        )

    rebuilt = [
        finding.model_copy(
            update={
                "severity": finding_severity(
                    finding.type,
                    step_by_index.get(finding.expected_step_index)
                    if finding.expected_step_index is not None
                    else None,
                )
            }
        )
        for finding in new_findings
    ]
    total = len(expected.steps)
    passed = sum(1 for verdict in new_verdicts if verdict.passed)
    return AuditResult(
        verdict=verdict_for(rebuilt),
        fidelity_score=(passed / total) if total else 1.0,
        findings=rebuilt,
        step_verdicts=new_verdicts,
    )


def remap_interactions_against_episodes(
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    episodes: list[ScreenEpisode],
    *,
    accepted_assertion_remaps: Mapping[int, int] | None = None,
) -> list[AssistProposalRecord]:
    """Remap drifted interaction steps onto actions inside trusted brackets.

    For each ``wrong_target`` / ``missing_action`` interaction step:
    1. Ignore that step's DP ``matched_observed_index`` entirely.
    2. Search all actions in ``(prev_trusted_anchor, next_trusted_assertion)``.
    3. Bind the best fuzzy target match and clear the drifted findings.
    """
    needed = steps_needing_interaction_remap(expected, audit)
    if not needed:
        return []

    needs_remap = set(needed)
    remaps = dict(accepted_assertion_remaps or {})
    trusted = _trusted_matched_index_map(
        expected,
        audit,
        accepted_remaps=remaps,
        needs_remap=needs_remap,
    )
    # Only trusted anchors occupy observed slots. Drifted DP matches outside
    # (or inside) the bracket must not block in-window candidates.
    claimed = set(trusted.values())

    verdicts = _verdict_map(audit)
    records: list[AssistProposalRecord] = []
    # Process in expected order so remapped indices stay strictly monotonic.
    for step_index in sorted(needed):
        step = next(s for s in expected.steps if s.index == step_index)
        window = interaction_search_window(step_index, expected, observed, trusted)
        start, end = window
        if end < start:
            continue
        dp_matched = (
            verdicts[step_index].matched_observed_index
            if step_index in verdicts
            else None
        )
        finding_types = set(verdicts[step_index].finding_types) if step_index in verdicts else set()
        clearing_missing = FindingType.missing_action in finding_types
        min_score = (
            REMAP_MISSING_ACTION_FLOOR
            if clearing_missing
            else REMAP_WRONG_TARGET_FLOOR
        )
        match = None
        app_launch_recovery = False
        if _is_app_launch_expectation(step, expected):
            match = find_app_launch_match(
                step,
                observed,
                window=window,
                claimed_obs_indices=claimed,
                expected=expected,
            )
            app_launch_recovery = match is not None
        elif step.action in INTERACTION_ACTIONS:
            match = find_matching_interaction(
                expected_action=step.action,
                expected_target=step.target or "",
                expected_input_value=step.input_value,
                observed=observed,
                episodes=episodes,
                window=window,
                claimed_obs_indices=claimed,
                min_score=min_score,
                require_canonical_nouns=clearing_missing,
            )
        typeahead_proof = None
        if match is None and step.action == ActionType.select and step.input_value:
            typeahead_proof = find_typeahead_selection_proof(
                step,
                observed,
                window=window,
                claimed_obs_indices=claimed,
            )
            if typeahead_proof is not None:
                match = next(
                    (
                        action
                        for action in observed.actions
                        if action.index == typeahead_proof.terminal_index
                    ),
                    None,
                )
        if match is None:
            logger.info(
                "stage3.interaction_remap miss step=%s window=%s..%s "
                "dp_matched=%s (ignored)",
                step_index,
                start,
                end,
                dp_matched,
            )
            continue
        # Same-slot DP pairs stay with classic wrong_target LLM assist.
        # Interpolation only rebinds when a *different* in-bracket action wins.
        if (
            typeahead_proof is None
            and dp_matched is not None
            and match.index == dp_matched
        ):
            logger.info(
                "stage3.interaction_remap skip_same_dp step=%s obs=%s "
                "window=%s..%s",
                step_index,
                match.index,
                start,
                end,
            )
            continue
        # Monotonic bracket: remapped index must lie strictly inside trusted bounds.
        if match.index < start or match.index > end or match.index in claimed:
            continue
        if typeahead_proof is not None and (
            typeahead_proof.claimed_indices & claimed
            or any(
                index < start or index > end
                for index in typeahead_proof.claimed_indices
            )
        ):
            continue

        if typeahead_proof is not None:
            claimed.update(typeahead_proof.claimed_indices)
        else:
            claimed.add(match.index)
        trusted[step_index] = match.index
        episode = next(
            (
                ep
                for ep in episodes
                if ep.start_index <= match.index <= ep.end_index
            ),
            None,
        )
        cleared_types = [
            ft
            for ft in (FindingType.wrong_target, FindingType.missing_action)
            if ft in finding_types
        ] or [FindingType.wrong_target]
        if typeahead_proof is not None:
            rationale = (
                f"Recovered select on '{step.target}' via typeahead sequence "
                f"type@{typeahead_proof.type_index} → "
                f"click@{typeahead_proof.click_index}"
                + (
                    f" → confirm@{typeahead_proof.confirm_index}"
                    if typeahead_proof.confirm_index is not None
                    else ""
                )
                + f" inside trusted window {start}..{end}."
            )
            reason = TYPEAHEAD_RECOVERY_REASON
        elif app_launch_recovery:
            rationale = (
                f"Recovered app launch '{step.target}' via "
                f"{match.action.value} on '{match.target}' "
                f"inside trusted window {start}..{end}."
            )
            reason = APP_LAUNCH_RECOVERY_REASON
        else:
            rationale = (
                f"Remapped {step.action.value} on '{step.target}' to "
                f"observed action {match.index} "
                f"({match.action.value} on '{match.target}') "
                f"inside trusted window {start}..{end}, "
                f"ignoring DP match."
            )
            reason = "interaction_temporal_remap"
        records.append(
            AssistProposalRecord(
                kind=AssistCandidateKind.episode_verification,
                status=AssistProposalStatus.accepted,
                expected_step_index=step_index,
                matched_observed_index=match.index,
                supporting_observed_index=match.index,
                completion_observed_index=match.index,
                finding_types=cleared_types,
                rationale=rationale,
                reason=reason,
                matching_method=AssistMatchMethod.deterministic,
            )
        )
        logger.info(
            "stage3.interaction_remap hit step=%s -> obs=%s window=%s..%s "
            "episode=%s reason=%s",
            step_index,
            match.index,
            start,
            end,
            episode.episode_id if episode is not None else None,
            reason,
        )
    return records
