"""Collapse the one action a split-window seam can report twice.

When derive_action_windows splits a long window, the next sub-window resumes
SPLIT_SEAM_GAP_MS after the previous one ends, and select_window_frames always
samples both window edges. The two edge frames are effectively the same screen,
seen by two independent extractions, so the same interaction can surface once
per window. Only that exact shape is collapsed: anything else is a genuine
repeat and is preserved.
"""

import re

from reckon.schema import ObservedAction, ObservedTrace

SPLIT_SEAM_GAP_MS = 1


def _tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _values_compatible(first: str | None, second: str | None) -> bool:
    if first is None or second is None:
        return first == second
    a, b = first.strip().lower(), second.strip().lower()
    return a.startswith(b) or b.startswith(a)


def _same_seam_action(earlier: ObservedAction, later: ObservedAction) -> bool:
    if earlier.action != later.action:
        return False
    earlier_tokens, later_tokens = _tokens(earlier.target), _tokens(later.target)
    if not earlier_tokens or not later_tokens:
        return False
    if not (earlier_tokens <= later_tokens or later_tokens <= earlier_tokens):
        return False
    return _values_compatible(earlier.input_value, later.input_value)


def _specificity(action: ObservedAction) -> tuple[int, int]:
    return len(_tokens(action.target)), len(action.input_value or "")


def _seam_pair(
    earlier: ObservedTrace, later: ObservedTrace
) -> tuple[ObservedAction, ObservedAction] | None:
    if earlier.clip_end_ms is None or not earlier.actions or not later.actions:
        return None
    if later.clip_offset_ms - earlier.clip_end_ms != SPLIT_SEAM_GAP_MS:
        return None
    last, first = earlier.actions[-1], later.actions[0]
    on_earlier_edge = last.timestamp_ms + earlier.clip_offset_ms == earlier.clip_end_ms
    on_later_edge = first.timestamp_ms == 0
    if not (on_earlier_edge and on_later_edge and _same_seam_action(last, first)):
        return None
    return last, first


def drop_split_seam_duplicates(chunks: list[ObservedTrace]) -> list[ObservedTrace]:
    """Return chunks with each split-seam duplicate reported only once.

    The more specific report is kept with its own frame timestamp; on a tie the
    earlier window's report wins.
    """
    ordered = sorted(chunks, key=lambda c: c.clip_offset_ms)
    for position in range(1, len(ordered)):
        earlier, later = ordered[position - 1], ordered[position]
        pair = _seam_pair(earlier, later)
        if pair is None:
            continue
        last, first = pair
        if _specificity(first) > _specificity(last):
            ordered[position - 1] = earlier.model_copy(update={"actions": earlier.actions[:-1]})
        else:
            ordered[position] = later.model_copy(update={"actions": later.actions[1:]})
    return ordered
