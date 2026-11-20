from reckon.schema import (
    ActionType,
    ObservedAction,
    ObservedTrace,
    merge_observed_chunks,
)
from reckon.seams import drop_split_seam_duplicates

SOURCE = "recording.mp4"


def _action(
    timestamp_ms: int,
    target: str | None,
    action: ActionType = ActionType.click,
    input_value: str | None = None,
) -> ObservedAction:
    return ObservedAction(
        index=1,
        timestamp_ms=timestamp_ms,
        action=action,
        target=target,
        input_value=input_value,
        confidence=0.9,
    )


def _chunk(start: int, end: int, *actions: ObservedAction) -> ObservedTrace:
    return ObservedTrace(
        source_file=SOURCE,
        clip_offset_ms=start,
        clip_end_ms=end,
        actions=[
            action.model_copy(update={"index": i}) for i, action in enumerate(actions, start=1)
        ],
    )


def _merged(*chunks: ObservedTrace) -> list[tuple[int, str | None]]:
    trace = merge_observed_chunks(drop_split_seam_duplicates(list(chunks)))
    return [(action.timestamp_ms, action.target) for action in trace.actions]


def test_drop_split_seam_duplicates_collapses_compatible_click_at_seam_keeping_specific_target():
    earlier = _chunk(
        0,
        22133,
        _action(21266, "Enter location", ActionType.type_text, "LB Nagar"),
        _action(22133, "L. B. Nagar Hyderabad, Telangana, India"),
    )
    later = _chunk(22134, 44266, _action(0, "L. B. Nagar"), _action(3866, "Enter destination"))

    assert _merged(earlier, later) == [
        (21266, "Enter location"),
        (22133, "L. B. Nagar Hyderabad, Telangana, India"),
        (26000, "Enter destination"),
    ]


def test_drop_split_seam_duplicates_keeps_later_report_when_it_is_more_specific():
    earlier = _chunk(0, 1000, _action(1000, "Save"))
    later = _chunk(1001, 2000, _action(0, "Save changes button"))

    assert _merged(earlier, later) == [(1001, "Save changes button")]


def test_drop_split_seam_duplicates_keeps_different_action_types():
    earlier = _chunk(0, 1000, _action(1000, "Search", ActionType.click))
    later = _chunk(1001, 2000, _action(0, "Search", ActionType.type_text, "uber"))

    assert len(_merged(earlier, later)) == 2


def test_drop_split_seam_duplicates_keeps_unrelated_targets():
    earlier = _chunk(0, 1000, _action(1000, "Save button"))
    later = _chunk(1001, 2000, _action(0, "Cancel button"))

    assert len(_merged(earlier, later)) == 2


def test_drop_split_seam_duplicates_keeps_incompatible_typed_values():
    earlier = _chunk(0, 1000, _action(1000, "Search", ActionType.type_text, "uber"))
    later = _chunk(1001, 2000, _action(0, "Search", ActionType.type_text, "lyft"))

    assert len(_merged(earlier, later)) == 2


def test_drop_split_seam_duplicates_keeps_repeated_clicks_away_from_seam_edge():
    earlier = _chunk(0, 1000, _action(700, "Next"))
    later = _chunk(1001, 2000, _action(0, "Next"))

    assert len(_merged(earlier, later)) == 2


def test_drop_split_seam_duplicates_keeps_repeated_clicks_within_one_window():
    chunk = _chunk(0, 2000, _action(1000, "Next"), _action(1300, "Next"))

    assert len(_merged(chunk)) == 2


def test_drop_split_seam_duplicates_ignores_windows_that_are_not_split_neighbours():
    earlier = _chunk(9766, 12500, _action(2734, "Uber rider home page", ActionType.assert_visible))
    later = _chunk(13266, 16766, _action(0, "Uber rider home page", ActionType.assert_visible))

    assert len(_merged(earlier, later)) == 2


def test_drop_split_seam_duplicates_keeps_actions_without_targets():
    earlier = _chunk(0, 1000, _action(1000, None, ActionType.scroll))
    later = _chunk(1001, 2000, _action(0, None, ActionType.scroll))

    assert len(_merged(earlier, later)) == 2
