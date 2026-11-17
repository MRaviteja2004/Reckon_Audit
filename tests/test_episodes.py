"""Unit tests for Stage 3 ScreenEpisode clustering."""

from reckon.episodes import (
    EPISODE_GAP_MS,
    cluster_observed_episodes,
    episode_containing_index,
    expand_indices_to_episode,
)
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    ObservedAction,
    ObservedTrace,
    ScreenEpisode,
)


def _action(
    index: int,
    *,
    timestamp_ms: int,
    target: str,
    screen: str,
    action: ActionType = ActionType.click,
    visible_text: str | None = None,
) -> ObservedAction:
    evidence = None
    if visible_text is not None:
        evidence = AssertionEvidence(
            description=visible_text,
            visible_text=visible_text,
        )
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=action,
        target=target,
        confidence=0.9,
        screen_description=screen,
        assertion_evidence=evidence,
    )


def test_cluster_merges_close_same_screen_actions():
    observed = ObservedTrace(
        source_file="rec.mp4",
        actions=[
            _action(1, timestamp_ms=1000, target="Email", screen="Login form"),
            _action(2, timestamp_ms=1500, target="Password", screen="Login form"),
            _action(3, timestamp_ms=2000, target="Sign in", screen="Login form"),
        ],
    )

    episodes = cluster_observed_episodes(observed)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.episode_id == 1
    assert episode.start_index == 1
    assert episode.end_index == 3
    assert episode.start_ms == 1000
    assert episode.end_ms == 2000
    assert "Login form" in episode.screen_summary
    assert len(episode.actions_summary) == 3
    assert episode.is_ambient_system is False


def test_cluster_splits_on_large_time_gap():
    observed = ObservedTrace(
        source_file="rec.mp4",
        actions=[
            _action(1, timestamp_ms=1000, target="A", screen="Screen A"),
            _action(
                2,
                timestamp_ms=1000 + EPISODE_GAP_MS + 1,
                target="B",
                screen="Screen A",
            ),
        ],
    )

    episodes = cluster_observed_episodes(observed)

    assert len(episodes) == 2
    assert episodes[0].end_index == 1
    assert episodes[1].start_index == 2


def test_cluster_splits_on_distinct_screen_identity():
    observed = ObservedTrace(
        source_file="rec.mp4",
        actions=[
            _action(1, timestamp_ms=1000, target="Home", screen="Driver home map"),
            _action(
                2,
                timestamp_ms=1200,
                target="Studio Dev Settings",
                screen="Studio Dev Settings with Launch Deeplink",
            ),
        ],
    )

    episodes = cluster_observed_episodes(observed)

    assert len(episodes) == 2
    assert "home" in episodes[0].screen_summary.lower()
    assert "studio" in episodes[1].screen_summary.lower()


def test_cluster_marks_ambient_system_episodes():
    observed = ObservedTrace(
        source_file="rec.mp4",
        actions=[
            _action(
                1,
                timestamp_ms=500,
                target="Notification shade",
                screen="System notification shade expanded",
            ),
        ],
    )

    episodes = cluster_observed_episodes(observed)

    assert len(episodes) == 1
    assert episodes[0].is_ambient_system is True


def test_cluster_collects_visible_texts():
    observed = ObservedTrace(
        source_file="rec.mp4",
        actions=[
            _action(
                1,
                timestamp_ms=1000,
                target="Banner",
                screen="Home",
                visible_text="You're offline",
            ),
            _action(
                2,
                timestamp_ms=1100,
                target="Banner",
                screen="Home",
                visible_text="You're offline",
            ),
        ],
    )

    episodes = cluster_observed_episodes(observed)

    assert episodes[0].visible_texts == ["You're offline"]


def test_episode_lookup_helpers():
    episodes = [
        ScreenEpisode(
            episode_id=1,
            start_ms=0,
            end_ms=1000,
            start_index=1,
            end_index=2,
            screen_summary="A",
        ),
        ScreenEpisode(
            episode_id=2,
            start_ms=2000,
            end_ms=3000,
            start_index=3,
            end_index=4,
            screen_summary="B",
        ),
    ]

    assert episode_containing_index(episodes, 2).episode_id == 1
    assert episode_containing_index(episodes, 3).episode_id == 2
    assert episode_containing_index(episodes, 9) is None
    assert expand_indices_to_episode(episodes, 4) == (3, 4)


def test_cluster_empty_trace():
    observed = ObservedTrace(source_file="rec.mp4", actions=[])
    assert cluster_observed_episodes(observed) == []
