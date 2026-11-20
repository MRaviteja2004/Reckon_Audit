"""Deterministic episode search and fast-path assertion checks."""

import re

from rapidfuzz import fuzz

from reckon.schema import (
    AuditResult,
    ExpectedStep,
    ExpectedTrace,
    FindingType,
    ObservedAction,
    ObservedTrace,
    ScreenEpisode,
)
from reckon.episode_interaction import INTERACTION_ACTIONS, is_passive_observation
from reckon.page_context import opening_navigation_target
from reckon.tolerance import (
    ASSERTION_ACTIONS,
    FUZZY_THRESHOLD,
    actions_compatible,
    assertion_satisfied,
)
from reckon.workflow_reentry import reentry_floor_for_step

# Genuine failure / crash cues only. Ordinary UI chrome — disabled, greyed,
# unchecked, close, dismiss, cancel — must never trip episode verification.
_NEGATIVE_OUTCOME_RE = re.compile(
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
    r"try\s+again\s+later|"
    r"session\s+(?:expired|ended|terminated)|"
    r"access\s+denied|"
    r"crash(?:ed)?|"
    r"unable\s+to\s+(?:authenticate|log\s*in|sign\s*in)"
    r")\b",
    re.IGNORECASE,
)

# Cap LLM episode payload while still preferring assertion-relevant screens.
MAX_LLM_EPISODES = 24

_TRANSIENT_STATE_RE = re.compile(
    r"^\s*(?:"
    r"spinner|loading(?:\s+spinner)?|progress|please\s+wait|fetching|"
    r"still\s+loading|loading\s+spinner"
    r")\s*$",
    re.IGNORECASE,
)

_DELAYED_TOAST_RE = re.compile(
    r"\b(?:toast|banner|notification|snackbar|alert)\b",
    re.IGNORECASE,
)

# Delayed toast/banner evidence must appear within this window after the trigger.
DELAYED_TOAST_WINDOW_MS = 10_000

_TARGET_NOISE_TOKENS = frozenset(
    {
        "button",
        "btn",
        "cta",
        "link",
        "control",
        "field",
        "input",
        "the",
        "a",
        "an",
        "to",
        "of",
        "and",
        "or",
        "is",
        "are",
        "on",
        "in",
    }
)


def is_transient_evidence_text(text: str | None) -> bool:
    """True when text is solely a loading/spinner/progress indicator."""
    if text is None or not text.strip():
        return False
    return bool(_TRANSIENT_STATE_RE.match(text.strip()))


def is_delayed_toast_assertion(step: ExpectedStep) -> bool:
    """True when the expected assert is explicitly a delayed toast/banner cue."""
    haystack = " ".join(
        part for part in (step.assertion, step.target, step.raw_text) if part
    )
    return bool(_DELAYED_TOAST_RE.search(haystack))


def _meaningful_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _TARGET_NOISE_TOKENS and len(token) > 1
    }


def delayed_toast_target_relevant(step: ExpectedStep, action: ObservedAction) -> bool:
    """Require shared target/assertion nouns so unrelated success banners cannot match."""
    expected_tokens = _meaningful_tokens(step.target) | _meaningful_tokens(step.assertion)
    if not expected_tokens:
        return True
    observed_parts = [
        action.target,
        action.screen_description,
    ]
    if action.assertion_evidence is not None:
        observed_parts.extend(
            [
                action.assertion_evidence.visible_text,
                action.assertion_evidence.description,
            ]
        )
    observed_tokens: set[str] = set()
    for part in observed_parts:
        observed_tokens |= _meaningful_tokens(part)
    return bool(expected_tokens & observed_tokens)


def is_assertion_step(step: ExpectedStep) -> bool:
    return step.assertion is not None or step.action in ASSERTION_ACTIONS


def findings_for_step(audit: AuditResult, step_index: int) -> set[FindingType]:
    return {
        finding.type
        for finding in audit.findings
        if finding.expected_step_index == step_index
    }


def steps_needing_episode_verification(
    expected: ExpectedTrace,
    audit: AuditResult,
) -> list[ExpectedStep]:
    """Select assertion steps whose baseline DP result needs episode rescue."""
    selected: list[ExpectedStep] = []
    for step in expected.steps:
        if step.assertion is None:
            continue
        findings = findings_for_step(audit, step.index)
        if FindingType.wrong_input_value in findings:
            continue
        if findings & {
            FindingType.assertion_not_observed,
            FindingType.missing_action,
        }:
            selected.append(step)
    return selected


def search_start_observed_index(
    step: ExpectedStep,
    expected: ExpectedTrace,
    audit: AuditResult,
    episodes: list[ScreenEpisode],
    observed: ObservedTrace | None = None,
) -> int:
    """Inclusive lower bound from the nearest prior *passed* step, else 1.

    Failed / misaligned DP anchors are ignored so a bad earlier match cannot
    hide earlier-but-still-forward evidence for the current assertion. When
    ``observed`` is provided, a URL/deeplink re-entry may raise the floor if
    assertion proof appears after that re-entry.
    """
    start = 1
    prev_steps = [item for item in expected.steps if item.index < step.index]
    for prev in reversed(prev_steps):
        prev_verdict = next(
            (
                verdict
                for verdict in audit.step_verdicts
                if verdict.expected_step_index == prev.index
            ),
            None,
        )
        if (
            prev_verdict is None
            or not prev_verdict.passed
            or prev_verdict.matched_observed_index is None
        ):
            continue
        # Forward from the confirmed match (inclusive) — not capped to a
        # narrow ±N window around the current step's DP anchor.
        start = prev_verdict.matched_observed_index
        break
    if observed is None or not observed.actions:
        return start
    max_index = observed.actions[-1].index
    return reentry_floor_for_step(
        step, observed, min_index=start, max_index=max_index
    )


def _trusted_mutator_boundary(
    step: ExpectedStep,
    expected: ExpectedTrace,
    audit: AuditResult,
    observed: ObservedTrace,
    start_index: int,
) -> int | None:
    """Next verified interaction match that chronologically bounds this assert.

    Only a passed later interaction whose matched observation genuinely looks
    like that step's control may cap the search. Misaligned DP anchors that
    land on unrelated controls do not truncate evidence.
    """
    action_by_index = {action.index: action for action in observed.actions}
    for nxt in expected.steps:
        if nxt.index <= step.index:
            continue
        if nxt.action not in INTERACTION_ACTIONS:
            continue
        nxt_verdict = next(
            (
                verdict
                for verdict in audit.step_verdicts
                if verdict.expected_step_index == nxt.index
            ),
            None,
        )
        if (
            nxt_verdict is None
            or not nxt_verdict.passed
            or nxt_verdict.matched_observed_index is None
        ):
            continue
        matched = action_by_index.get(nxt_verdict.matched_observed_index)
        if matched is None or matched.index < start_index:
            continue
        if matched.action not in INTERACTION_ACTIONS:
            continue
        if nxt.target and matched.target:
            overlap = fuzz.token_set_ratio(
                nxt.target.lower(), matched.target.lower()
            )
            if overlap < FUZZY_THRESHOLD:
                continue
        return matched.index
    return None


def _trigger_mutator_timestamp_ms(
    start_index: int,
    observed: ObservedTrace,
) -> int:
    """Timestamp of the observation that opens the delayed-toast search window."""
    for action in observed.actions:
        if action.index == start_index:
            return action.timestamp_ms
    return observed.actions[0].timestamp_ms


def _delayed_toast_end_index(
    start_index: int,
    observed: ObservedTrace,
) -> int:
    """Last observed index with timestamp <= trigger_mutator_time + 10000ms."""
    trigger_ms = _trigger_mutator_timestamp_ms(start_index, observed)
    deadline = trigger_ms + DELAYED_TOAST_WINDOW_MS
    end_index = start_index
    for action in observed.actions:
        if action.index < start_index:
            continue
        if action.timestamp_ms <= deadline:
            end_index = action.index
        else:
            break
    return end_index


def candidate_episodes_for_step(
    step: ExpectedStep,
    expected: ExpectedTrace,
    observed: ObservedTrace,
    audit: AuditResult,
    episodes: list[ScreenEpisode],
) -> list[ScreenEpisode]:
    """Causal forward window of episodes for one expected step.

    Passive assertions stop at the next verified mutator/interaction unless the
    expected step is explicitly a delayed toast/banner. Delayed toasts are still
    bounded to ``trigger_mutator_time + 10000ms``. Non-assertion steps still stop
    at the next step's DP match to preserve action-order locality.
    """
    if not episodes or not observed.actions:
        return []
    start_index = search_start_observed_index(
        step, expected, audit, episodes, observed=observed
    )
    end_index = observed.actions[-1].index
    if is_assertion_step(step) and is_delayed_toast_assertion(step):
        end_index = max(
            _delayed_toast_end_index(start_index, observed),
            start_index,
        )
    elif is_assertion_step(step):
        boundary = _trusted_mutator_boundary(
            step, expected, audit, observed, start_index
        )
        if boundary is not None:
            end_index = max(boundary, start_index)
    else:
        next_steps = [item for item in expected.steps if item.index > step.index]
        if next_steps:
            nxt = next_steps[0]
            nxt_verdict = next(
                (
                    verdict
                    for verdict in audit.step_verdicts
                    if verdict.expected_step_index == nxt.index
                ),
                None,
            )
            if (
                nxt_verdict is not None
                and nxt_verdict.matched_observed_index is not None
            ):
                end_index = max(nxt_verdict.matched_observed_index, start_index)
    return [
        episode
        for episode in episodes
        if episode.end_index >= start_index and episode.start_index <= end_index
    ]


def _assertion_overlap_score(step: ExpectedStep, episode: ScreenEpisode) -> int:
    assertion = step.assertion or step.target or ""
    tokens = re.findall(r"[a-z0-9]+", assertion.lower())
    if not tokens:
        return 0
    blob = " ".join(
        [
            episode.screen_summary,
            *episode.visible_texts,
            *episode.actions_summary,
        ]
    ).lower()
    return sum(1 for token in tokens if token in blob)


def select_episodes_for_llm(
    step: ExpectedStep,
    episodes: list[ScreenEpisode],
    *,
    max_episodes: int = MAX_LLM_EPISODES,
) -> list[ScreenEpisode]:
    """Prefer assertion-overlapping episodes, then fill chronologically."""
    if len(episodes) <= max_episodes:
        return list(episodes)
    ranked = sorted(
        episodes,
        key=lambda episode: (
            -_assertion_overlap_score(step, episode),
            episode.start_index,
        ),
    )
    chosen = ranked[:max_episodes]
    return sorted(chosen, key=lambda episode: episode.start_index)


def _append_unique(texts: list[str], seen: set[str], raw: str | None) -> None:
    if not raw:
        return
    cleaned = raw.strip()
    if cleaned and cleaned not in seen:
        seen.add(cleaned)
        texts.append(cleaned)


def episode_probe_texts(
    episode: ScreenEpisode,
    observed: ObservedTrace,
) -> list[str]:
    """All grounded strings Python may use to validate an LLM evidence_quote."""
    texts: list[str] = []
    seen: set[str] = set()
    for raw in [*episode.visible_texts, episode.screen_summary]:
        _append_unique(texts, seen, raw)
    for action in observed.actions:
        if not (episode.start_index <= action.index <= episode.end_index):
            continue
        _append_unique(texts, seen, action.target)
        _append_unique(texts, seen, action.screen_description)
        if action.assertion_evidence is not None:
            _append_unique(texts, seen, action.assertion_evidence.visible_text)
            _append_unique(texts, seen, action.assertion_evidence.description)
    return texts


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def quote_appears_in_texts(quote: str, texts: list[str]) -> bool:
    """True when quote is an exact substring of some corpus text (whitespace-normalized)."""
    needle = normalize_quote(quote)
    if not needle:
        return False
    return any(needle in normalize_quote(text) for text in texts if text)


def action_texts(action: ObservedAction) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    _append_unique(texts, seen, action.target)
    _append_unique(texts, seen, action.screen_description)
    if action.assertion_evidence is not None:
        _append_unique(texts, seen, action.assertion_evidence.visible_text)
        _append_unique(texts, seen, action.assertion_evidence.description)
    return texts


def resolve_action_for_quote(
    quote: str,
    episode: ScreenEpisode,
    observed: ObservedTrace,
    preferred_index: int | None = None,
    *,
    prefer_passive: bool = True,
) -> ObservedAction | None:
    """Resolve quote to an action; prefer passive observations over clicks/selects."""
    in_episode = [
        action
        for action in observed.actions
        if episode.start_index <= action.index <= episode.end_index
    ]
    quote_matches = [
        action
        for action in in_episode
        if quote_appears_in_texts(quote, action_texts(action))
    ]
    if prefer_passive:
        passive_matches = [
            action for action in quote_matches if is_passive_observation(action.action)
        ]
        if passive_matches:
            if preferred_index is not None:
                preferred = next(
                    (a for a in passive_matches if a.index == preferred_index),
                    None,
                )
                if preferred is not None:
                    return preferred
            return passive_matches[0]
    if preferred_index is not None:
        preferred = next(
            (action for action in quote_matches if action.index == preferred_index),
            None,
        )
        if preferred is not None:
            return preferred
    if quote_matches:
        return quote_matches[0]
    if quote_appears_in_texts(quote, episode_probe_texts(episode, observed)):
        if prefer_passive:
            passive = [
                action
                for action in in_episode
                if is_passive_observation(action.action)
            ]
            if passive:
                if preferred_index is not None:
                    preferred = next(
                        (a for a in passive if a.index == preferred_index),
                        None,
                    )
                    if preferred is not None:
                        return preferred
                return passive[0]
        if preferred_index is not None:
            preferred = next(
                (action for action in in_episode if action.index == preferred_index),
                None,
            )
            if preferred is not None:
                return preferred
        return in_episode[-1] if in_episode else None
    return None


def assertion_claims_negative(assertion: str) -> bool:
    return _NEGATIVE_OUTCOME_RE.search(assertion) is not None


def episode_has_negative_invariant(
    step: ExpectedStep,
    episode: ScreenEpisode,
    observed: ObservedTrace,
) -> bool:
    """True when the episode shows a genuine failure opposing the assertion.

    Ordinary UI state language (disabled, unchecked, close, dismiss, cancel)
    is ignored. Broad token negation is not used here — it false-triggered on
    intermediate form chrome.
    """
    assertion = step.assertion
    if assertion is None or assertion_claims_negative(assertion):
        return False
    for text in episode_probe_texts(episode, observed):
        if _NEGATIVE_OUTCOME_RE.search(text):
            return True
    return False


def _action_satisfies_assertion(
    assertion: str,
    action: ObservedAction,
    *,
    expected_target: str | None = None,
    context_target: str | None = None,
) -> bool:
    """Accept structured assertion_evidence, or screen identity in visible text.

    Naked screen_description alone is never enough. ``context_target`` is the URL
    of the navigation that opened the screen, inside the same episode.
    """
    if not assertion_satisfied(
        assertion,
        action,
        expected_target=expected_target,
        context_target=context_target,
    ):
        return False
    evidence = action.assertion_evidence
    if evidence is None:
        return True
    for text in (evidence.visible_text, evidence.description):
        if text and is_transient_evidence_text(text):
            return False
    return True


def best_proof_action(
    step: ExpectedStep,
    episode: ScreenEpisode,
    observed: ObservedTrace,
) -> ObservedAction | None:
    """Earliest proving action; prefer passive observations over click/select/type.

    Assertion steps must not consume a physical interaction when a passive
    observation (assert_*, navigate, other, …) already carries the proof.
    Interactions are used only when no passive proof exists. Delayed toast
    asserts additionally require target-token relevance so unrelated later
    success banners cannot satisfy an earlier expect.
    """
    assertion = step.assertion
    if assertion is None:
        return None
    require_target = is_delayed_toast_assertion(step)
    in_episode = [
        action
        for action in observed.actions
        if episode.start_index <= action.index <= episode.end_index
    ]
    matches = [
        action
        for action in in_episode
        if _action_satisfies_assertion(
            assertion,
            action,
            expected_target=step.target,
            context_target=opening_navigation_target(
                in_episode, action, floor_ms=episode.start_ms
            ),
        )
        and (not require_target or delayed_toast_target_relevant(step, action))
    ]
    if not matches:
        return None
    passive = [action for action in matches if is_passive_observation(action.action)]
    if passive:
        return passive[0]
    return matches[0]


def _target_overlap(expected_target: str | None, observed_target: str | None) -> float:
    if not expected_target or not observed_target:
        return 0.0
    return fuzz.token_set_ratio(expected_target.lower(), observed_target.lower())


# Lookahead for a deferred initiation after candidate proof. Longer windows
# false-trigger on later same-label clicks (e.g. a second Sign-in submit).
PREMATURE_LOOKAHEAD_MS = 12_000


def proof_is_premature(
    step: ExpectedStep,
    expected: ExpectedTrace,
    proof: ObservedAction,
    observed: ObservedTrace,
) -> bool:
    """Reject completion evidence that appears before a deferred initiation.

    When the previous expected step is an open/launch action (no assertion), a
    later observed action that better matches that step's target *and* no
    plausible initiation already occurred at/before the proof means the
    candidate ran before the flow started.

    Later same-label clicks after a valid initiation (e.g. a second Sign-in
    submit) must not mark an earlier login-page proof as premature.
    """
    prev_steps = [item for item in expected.steps if item.index < step.index]
    if not prev_steps:
        return False
    prev = prev_steps[-1]
    if prev.assertion is not None or prev.target is None:
        return False

    def _matches_prev_initiation(action: ObservedAction) -> bool:
        if action.target is None:
            return False
        compatible = (
            actions_compatible(prev.action, action.action)
            or action.action.value in {"click", "navigate", "other"}
        )
        if not compatible:
            return False
        return _target_overlap(prev.target, action.target) >= FUZZY_THRESHOLD

    # Reliable prior initiation already observed — later clones are not "deferred".
    if any(
        action.index <= proof.index and _matches_prev_initiation(action)
        for action in observed.actions
    ):
        return False

    deadline_ms = proof.timestamp_ms + PREMATURE_LOOKAHEAD_MS
    for action in observed.actions:
        if action.index <= proof.index:
            continue
        if action.timestamp_ms > deadline_ms:
            break
        if _matches_prev_initiation(action):
            return True
    return False


def fast_path_verify_episode(
    step: ExpectedStep,
    episode: ScreenEpisode,
    observed: ObservedTrace,
    *,
    expected: ExpectedTrace | None = None,
) -> ObservedAction | None:
    """Deterministic verify via structured assertion_evidence only."""
    assertion = step.assertion
    if assertion is None or episode.is_ambient_system:
        return None
    if episode_has_negative_invariant(step, episode, observed):
        return None
    # Reject episodes whose only visible texts are transient loading cues.
    structured_texts = [
        text
        for text in episode.visible_texts
        if text and not is_transient_evidence_text(text)
    ]
    if episode.visible_texts and not structured_texts:
        return None
    proof = best_proof_action(step, episode, observed)
    if proof is None:
        return None
    if expected is not None and proof_is_premature(step, expected, proof, observed):
        return None
    return proof
