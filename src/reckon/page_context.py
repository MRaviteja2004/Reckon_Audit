"""The navigation that opened the screen an observation was recorded on."""

from reckon.schema import ActionType, ObservedAction


def opening_navigation_target(
    actions: list[ObservedAction],
    obs: ObservedAction,
    *,
    floor_ms: int,
) -> str | None:
    """URL of the latest navigation before ``obs``, if it is at or after ``floor_ms``.

    The latest navigation is the one whose screen ``obs`` was recorded on; an
    earlier one was left behind by it. ``floor_ms`` keeps a navigation from an
    earlier part of the flow from naming a later screen. A navigation is its own
    opening, so it borrows nothing.
    """
    if obs.action == ActionType.navigate:
        return None
    for action in reversed(actions):
        if action.index >= obs.index or action.action != ActionType.navigate:
            continue
        if action.timestamp_ms < floor_ms:
            return None
        return action.target
    return None
