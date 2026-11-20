from reckon.page_context import opening_navigation_target
from reckon.schema import ActionType, ObservedAction


def _action(index: int, action: ActionType, target: str, timestamp_ms: int) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=action,
        target=target,
        confidence=0.9,
    )


def _trace() -> list[ObservedAction]:
    return [
        _action(1, ActionType.navigate, "example.invalid", 1000),
        _action(2, ActionType.click, "Sign in", 2000),
        _action(3, ActionType.navigate, "example.invalid/home", 3000),
        _action(4, ActionType.assert_visible, "Dashboard", 3500),
        _action(5, ActionType.navigate, "example.invalid/settings", 4000),
    ]


def test_opening_navigation_target_returns_latest_earlier_navigation():
    actions = _trace()
    assert opening_navigation_target(actions, actions[3], floor_ms=2000) == (
        "example.invalid/home"
    )


def test_opening_navigation_target_rejects_navigation_before_floor():
    actions = _trace()
    assert opening_navigation_target(actions, actions[3], floor_ms=3200) is None


def test_opening_navigation_target_ignores_later_navigation():
    actions = _trace()
    assert opening_navigation_target(actions, actions[1], floor_ms=0) == "example.invalid"


def test_opening_navigation_target_is_none_for_a_navigation():
    actions = _trace()
    assert opening_navigation_target(actions, actions[2], floor_ms=0) is None
