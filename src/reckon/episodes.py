"""Domain-agnostic clustering of ObservedAction items into ScreenEpisode objects.

Stage 3 assist evaluates high-level expected steps against these episodes rather
than assuming each expected step maps to exactly one micro-action. Clustering is
deterministic and never invents timestamps or text.
"""

import re

from reckon.schema import ActionType, ObservedAction, ObservedTrace, ScreenEpisode

# Start a new episode when the gap between consecutive actions exceeds this.
EPISODE_GAP_MS = 2500

# Fuzzy token overlap at/above this → same screen episode.
_SCREEN_CONTINUITY_RATIO = 0.65
# Soft band [soft, hard): keep static screens together despite OCR noise unless
# a mutator (click / scroll / navigate) lands on the newer action.
_SOFT_CONTINUITY_RATIO = 0.45

_SCREEN_MUTATOR_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.click,
        ActionType.scroll,
        ActionType.navigate,
    }
)

_AMBIENT_SYSTEM_RE = re.compile(
    r"\b(?:"
    r"notification(?:s)?(?:\s+shade)?|"
    r"lock\s*screen|lockscreen|"
    r"control\s+cent(?:er|re)|"
    r"status\s+bar|"
    r"system\s+(?:ui|shade|dialog|permission)|"
    r"permission\s+(?:dialog|prompt|request)|"
    r"quick\s+settings|"
    r"os[- ]level|"
    r"home\s+indicator|"
    r"recents?\s+(?:screen|view|switcher)"
    r")\b",
    re.IGNORECASE,
)


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _screen_key(action: ObservedAction) -> str:
    """Stable screen identity — prefer screen_description over per-control target."""
    if action.screen_description and action.screen_description.strip():
        return action.screen_description.strip().lower()
    if action.target and action.target.strip():
        return action.target.strip().lower()
    return ""


def _jaccard(prev_tokens: set[str], current_tokens: set[str]) -> float:
    if not prev_tokens or not current_tokens:
        return 0.0
    return len(prev_tokens & current_tokens) / len(prev_tokens | current_tokens)


def _is_screen_mutator(action: ObservedAction) -> bool:
    return action.action in _SCREEN_MUTATOR_ACTIONS


def _screen_continuity(prev: ObservedAction, current: ObservedAction) -> bool:
    prev_key = _screen_key(prev)
    current_key = _screen_key(current)
    if not prev_key or not current_key:
        # Missing screen identity is not continuity — only time/gap rules apply.
        return False
    if prev_key == current_key:
        return True
    overlap = _jaccard(_tokens(prev_key), _tokens(current_key))
    if overlap >= _SCREEN_CONTINUITY_RATIO:
        return True
    # Soft OCR band: do not over-split static screens when no mutator occurred.
    if (
        _SOFT_CONTINUITY_RATIO <= overlap < _SCREEN_CONTINUITY_RATIO
        and not _is_screen_mutator(current)
        and not _is_screen_mutator(prev)
    ):
        return True
    return False


def _forces_episode_split(current: ObservedAction) -> bool:
    """Navigate / high-displacement scroll always starts a new episode."""
    if current.action == ActionType.navigate:
        return True
    if current.action == ActionType.scroll:
        return True
    return False


def _should_split(prev: ObservedAction, current: ObservedAction) -> bool:
    if current.timestamp_ms - prev.timestamp_ms > EPISODE_GAP_MS:
        return True
    if _forces_episode_split(current):
        return True
    return not _screen_continuity(prev, current)


def _is_ambient_system(actions: list[ObservedAction]) -> bool:
    for action in actions:
        haystack = " ".join(
            part
            for part in (
                action.target,
                action.screen_description,
                (
                    action.assertion_evidence.visible_text
                    if action.assertion_evidence
                    else None
                ),
                (
                    action.assertion_evidence.description
                    if action.assertion_evidence
                    else None
                ),
            )
            if part
        )
        if _AMBIENT_SYSTEM_RE.search(haystack):
            return True
    return False


def _action_label(action: ObservedAction) -> str:
    target = action.target or "(no target)"
    return f"{action.action.value} on {target}"


def _visible_texts(actions: list[ObservedAction]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for action in actions:
        candidates: list[str] = []
        if action.assertion_evidence is not None:
            if action.assertion_evidence.visible_text:
                candidates.append(action.assertion_evidence.visible_text)
            if action.assertion_evidence.description:
                candidates.append(action.assertion_evidence.description)
        for text in candidates:
            cleaned = text.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                texts.append(cleaned)
    return texts


def _screen_summary(actions: list[ObservedAction]) -> str:
    for action in reversed(actions):
        if action.screen_description and action.screen_description.strip():
            return action.screen_description.strip()
    for action in reversed(actions):
        if action.target and action.target.strip():
            return action.target.strip()
    return "unknown screen"


def _build_episode(episode_id: int, actions: list[ObservedAction]) -> ScreenEpisode:
    return ScreenEpisode(
        episode_id=episode_id,
        start_ms=actions[0].timestamp_ms,
        end_ms=actions[-1].timestamp_ms,
        start_index=actions[0].index,
        end_index=actions[-1].index,
        screen_summary=_screen_summary(actions),
        visible_texts=_visible_texts(actions),
        actions_summary=[_action_label(action) for action in actions],
        is_ambient_system=_is_ambient_system(actions),
    )


def cluster_observed_episodes(observed: ObservedTrace) -> list[ScreenEpisode]:
    """Cluster ordered observed actions into temporal screen episodes.

    Rules (domain-agnostic):
    - Preserve action order; never invent events.
    - Split on large timestamp gaps or clear screen-identity changes.
    - Mark ambient/system chrome via lexical OS cues only.
    """
    if not observed.actions:
        return []

    episodes: list[ScreenEpisode] = []
    bucket: list[ObservedAction] = [observed.actions[0]]

    for action in observed.actions[1:]:
        if _should_split(bucket[-1], action):
            episodes.append(_build_episode(len(episodes) + 1, bucket))
            bucket = [action]
        else:
            bucket.append(action)

    episodes.append(_build_episode(len(episodes) + 1, bucket))
    return episodes


def episode_containing_index(
    episodes: list[ScreenEpisode],
    observed_index: int,
) -> ScreenEpisode | None:
    """Return the episode whose [start_index, end_index] covers observed_index."""
    for episode in episodes:
        if episode.start_index <= observed_index <= episode.end_index:
            return episode
    return None


def expand_indices_to_episode(
    episodes: list[ScreenEpisode],
    observed_index: int,
) -> tuple[int, int] | None:
    """Return inclusive (start_index, end_index) for the episode covering index."""
    episode = episode_containing_index(episodes, observed_index)
    if episode is None:
        return None
    return episode.start_index, episode.end_index
