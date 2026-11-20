import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

from reckon.persistence import (
    STAGE1_FILENAME,
    STAGE2_FILENAME,
    ArtifactIntegrityError,
    ExecutionMetadata,
    MissingArtifactError,
    RecordingStore,
    RunStatus,
    Stage1Artifact,
    VideoRetention,
    build_stage2_artifact,
    frame_manifest_for,
    run_stage2,
)
from reckon.persistence import extraction as persistence_extraction
from reckon.schema import ActionType, ExpectedStep, ExpectedTrace, ObservedAction, ObservedTrace

pytestmark = pytest.mark.usefixtures("no_live_gateway")

_METADATA = ExecutionMetadata(
    started_at=datetime(2026, 1, 1, tzinfo=timezone.utc), python_version="3.13.0"
)


def _observed(source: str = "login.mp4", target: str = "Login") -> ObservedTrace:
    return ObservedTrace(
        source_file=source,
        actions=[
            ObservedAction(index=1, timestamp_ms=1500, action=ActionType.click, target=target, confidence=0.9),
            ObservedAction(index=2, timestamp_ms=500, action=ActionType.navigate, target="/", confidence=0.9),
        ],
    )


def _store(tmp_path: Path, retention: VideoRetention = VideoRetention.keep) -> tuple[RecordingStore, str]:
    store = RecordingStore(tmp_path / "recordings", video_retention=retention)
    recording_id = store.create_recording(
        title="Login", source_filename="login.mp4", scenario_text="Click Login"
    ).recording_id
    expected = ExpectedTrace(
        scenario_title="Login", steps=[ExpectedStep(index=1, action=ActionType.click, target="Login")]
    )
    store.save_stage1(recording_id, Stage1Artifact(metadata=_METADATA, expected_trace=expected))
    return store, recording_id


def _fake_jpegs(_video_path: Path, manifest: Any) -> dict[str, bytes]:
    return {entry.frame_id: f"jpeg-{entry.timestamp_ms}".encode() for entry in manifest.frames}


def test_frame_manifest_for_labels_each_distinct_observed_timestamp_in_order():
    manifest = frame_manifest_for(_observed())
    assert [(f.frame_id, f.timestamp_ms) for f in manifest.frames] == [("F1", 500), ("F2", 1500)]


def test_run_stage2_extracts_from_the_saved_video_without_touching_stage1(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    video = store.write_video(recording_id, lambda handle: handle.write(b"video"))
    stage1_sha = store.artifact_sha256(recording_id, STAGE1_FILENAME)
    client = MagicMock()
    with patch.object(persistence_extraction, "extract_observed_trace", return_value=_observed()) as extract:
        artifact = run_stage2(store, recording_id, extraction_client=client)
    extract.assert_called_once_with(client, video, source_file="login.mp4", on_retry=None)
    assert artifact.version == 1
    assert artifact.video_sha256 == store.load_manifest(recording_id).video_sha256
    assert store.artifact_sha256(recording_id, STAGE1_FILENAME) == stage1_sha


def test_run_stage2_retry_saves_a_new_version_and_keeps_the_previous_one(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with patch.object(persistence_extraction, "extract_observed_trace", return_value=_observed()):
        run_stage2(store, recording_id, extraction_client=MagicMock())
    first_sha = store.artifact_sha256(recording_id, STAGE2_FILENAME)
    retried = _observed(target="Sign in")
    with patch.object(persistence_extraction, "extract_observed_trace", return_value=retried):
        second = run_stage2(store, recording_id, extraction_client=MagicMock())
    assert second.version == 2
    assert store.artifact_sha256(recording_id, STAGE2_FILENAME) == first_sha
    assert store.load_stage2(recording_id, 1).observed_trace == _observed()
    assert store.load_stage2(recording_id, 2).observed_trace == retried


def test_run_stage2_failure_saves_nothing(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with patch.object(persistence_extraction, "extract_observed_trace", side_effect=RuntimeError("down")):
        with pytest.raises(RuntimeError):
            run_stage2(store, recording_id, extraction_client=MagicMock())
    assert store.stage2_versions(recording_id) == []


def test_run_stage2_requires_the_video(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    with pytest.raises(MissingArtifactError):
        run_stage2(store, recording_id, extraction_client=MagicMock())


def test_run_stage2_rejects_a_video_that_no_longer_matches_its_checksum(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    video = store.write_video(recording_id, lambda handle: handle.write(b"video"))
    video.write_bytes(b"replaced")
    with patch.object(persistence_extraction, "extract_observed_trace") as extract:
        with pytest.raises(ArtifactIntegrityError):
            run_stage2(store, recording_id, extraction_client=MagicMock())
    extract.assert_not_called()


def test_run_stage2_with_discard_retention_saves_evidence_stills(tmp_path: Path):
    store, recording_id = _store(tmp_path, VideoRetention.discard)
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    store.record_unretained_video(recording_id, upload)
    with (
        patch.object(persistence_extraction, "extract_observed_trace", return_value=_observed()),
        patch.object(persistence_extraction, "load_frame_jpegs", side_effect=_fake_jpegs),
    ):
        artifact = run_stage2(store, recording_id, extraction_client=MagicMock(), video_path=upload)
    assert [(f.frame_id, f.timestamp_ms) for f in artifact.evidence_frames] == [("F1", 500), ("F2", 1500)]
    assert artifact.evidence_frames[0].sha256 == hashlib.sha256(b"jpeg-500").hexdigest()
    assert store.load_evidence_frames(recording_id, artifact.evidence_frames)["F2"] == b"jpeg-1500"
    assert store.video_path(recording_id) is None


def test_run_stage2_with_keep_retention_saves_no_stills(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with (
        patch.object(persistence_extraction, "extract_observed_trace", return_value=_observed()),
        patch.object(persistence_extraction, "load_frame_jpegs") as decode,
    ):
        artifact = run_stage2(store, recording_id, extraction_client=MagicMock())
    assert artifact.evidence_frames == []
    decode.assert_not_called()


def test_build_stage2_artifact_records_extraction_configuration():
    artifact = build_stage2_artifact(
        _observed(), started_at=_METADATA.started_at, duration_ms=1.0, video_sha256="a" * 64
    )
    assert "stage2_event_selection_policy" in artifact.metadata.config
    assert artifact.metadata.models["stage2_extraction"]
    assert artifact.video_sha256 == "a" * 64


_SENTINEL = "example-usso-sentinel-0000"


def test_run_stage2_failure_records_a_failed_attempt_without_an_observed_trace(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with patch.object(persistence_extraction, "extract_observed_trace", side_effect=RuntimeError("down")):
        with pytest.raises(RuntimeError):
            run_stage2(store, recording_id, extraction_client=MagicMock())
    [attempt] = store.list_stage2_attempts(recording_id)
    assert attempt.status == RunStatus.failed
    assert attempt.error == "RuntimeError: down"
    assert attempt.stage2_version is None
    assert attempt.video_sha256 == store.load_manifest(recording_id).video_sha256
    assert (attempt.stage1_version, attempt.stage1_sha256) == (1, store.artifact_sha256(recording_id, STAGE1_FILENAME))
    assert attempt.metadata.completed_at >= attempt.metadata.started_at
    assert "observed_trace" not in attempt.model_dump()
    assert store.stage2_versions(recording_id) == []


def test_run_stage2_interruption_records_an_interrupted_attempt(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with patch.object(persistence_extraction, "extract_observed_trace", side_effect=StopException()):
        with pytest.raises(StopException):
            run_stage2(store, recording_id, extraction_client=MagicMock())
    [attempt] = store.list_stage2_attempts(recording_id)
    assert attempt.status == RunStatus.interrupted
    assert "StopException" in attempt.error


def test_run_stage2_success_links_the_attempt_and_the_new_version(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with patch.object(persistence_extraction, "extract_observed_trace", return_value=_observed()):
        artifact = run_stage2(store, recording_id, extraction_client=MagicMock())
    [attempt] = store.list_stage2_attempts(recording_id)
    assert (attempt.status, attempt.stage2_version) == (RunStatus.completed, artifact.version)
    assert store.load_stage2(recording_id, artifact.version).attempt_id == attempt.attempt_id


def test_run_stage2_failed_retry_preserves_the_successful_version(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with patch.object(persistence_extraction, "extract_observed_trace", return_value=_observed()):
        run_stage2(store, recording_id, extraction_client=MagicMock())
    first_sha = store.artifact_sha256(recording_id, STAGE2_FILENAME)
    with patch.object(persistence_extraction, "extract_observed_trace", side_effect=TimeoutError("slow")):
        with pytest.raises(TimeoutError):
            run_stage2(store, recording_id, extraction_client=MagicMock())
    assert store.stage2_versions(recording_id) == [1]
    assert store.artifact_sha256(recording_id, STAGE2_FILENAME) == first_sha
    assert [a.status for a in store.list_stage2_attempts(recording_id)] == [RunStatus.completed, RunStatus.failed]


def test_run_stage2_redacts_credentials_from_attempt_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USSO_TOKEN", _SENTINEL)
    store, recording_id = _store(tmp_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    with patch.object(
        persistence_extraction, "extract_observed_trace", side_effect=PermissionError(f"bad token {_SENTINEL}")
    ):
        with pytest.raises(PermissionError):
            run_stage2(store, recording_id, extraction_client=MagicMock())
    for path in store.recording_dir(recording_id).rglob("*.json"):
        assert _SENTINEL not in path.read_text()


def test_run_stage2_without_the_video_records_no_attempt(tmp_path: Path):
    store, recording_id = _store(tmp_path)
    with pytest.raises(MissingArtifactError):
        run_stage2(store, recording_id, extraction_client=MagicMock())
    assert store.list_stage2_attempts(recording_id) == []
