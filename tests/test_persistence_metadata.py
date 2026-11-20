import platform
from datetime import datetime, timezone
from pathlib import Path

from reckon.persistence.metadata import build_metadata, git_revision


def test_git_revision_outside_a_repository_is_none(tmp_path: Path):
    assert git_revision(tmp_path) == (None, None)


def test_build_metadata_records_models_config_and_runtime():
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    metadata = build_metadata(
        started,
        models={"stage4_crosscheck": "model-x"},
        config={"frames_per_target": 3},
        duration_ms=12.5,
    )
    assert metadata.started_at == started
    assert metadata.models == {"stage4_crosscheck": "model-x"}
    assert metadata.config == {"frames_per_target": 3}
    assert metadata.python_version == platform.python_version()
    assert metadata.duration_ms == 12.5


def test_build_metadata_records_git_commit_of_source_tree():
    commit, _dirty = git_revision()
    metadata = build_metadata(datetime.now(timezone.utc), models={}, config={})
    assert metadata.git_commit == commit
