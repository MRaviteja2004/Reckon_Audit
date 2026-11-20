import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from reckon.persistence import (
    RECORDINGS_DIR_ENV,
    STAGE1_FILENAME,
    STAGE2_ATTEMPTS_DIRNAME,
    STAGE2_FILENAME,
    STALE_RUN_SECONDS_ENV,
    VIDEO_RETENTION_ENV,
    ArtifactIntegrityError,
    EvidenceFrame,
    ExecutionMetadata,
    MissingArtifactError,
    RecordingBusyError,
    RecordingStore,
    RunOwner,
    RunStatus,
    Stage1Artifact,
    Stage2Artifact,
    Stage2Attempt,
    Stage2CArtifact,
    Stage3Evaluation,
    Stage3Inputs,
    Stage3Mode,
    VideoRetention,
    stage2c_filename,
    utc_now,
)
from persistence_fakes import make_legacy, tamper_json
from reckon.persistence.files import file_sha256
from reckon.persistence.integrity import INTEGRITY_FIELD, legacy_checksum_path, verify_json_artifact
from reckon.schema import (
    ActionType,
    AuditResult,
    AuditVerdict,
    CandidateEvidenceItem,
    CandidateEvidenceTrace,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedStep,
    ExpectedTrace,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)
from reckon.stage2c import build_frame_manifest
from reckon.stage2c.propose import RawFrameEvidenceBatch, RawFrameEvidenceCandidate

_STARTED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _metadata() -> ExecutionMetadata:
    return ExecutionMetadata(started_at=_STARTED, python_version="3.13.0")


def _store(tmp_path: Path) -> RecordingStore:
    return RecordingStore(tmp_path / "recordings")


def _new_recording(store: RecordingStore, title: str = "Login") -> str:
    return store.create_recording(
        title=title, source_filename="login.mp4", scenario_text="Click Login"
    ).recording_id


def _expected() -> ExpectedTrace:
    return ExpectedTrace(
        scenario_title="Login",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Login")],
    )


def _observed() -> ObservedTrace:
    return ObservedTrace(
        source_file="login.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1500,
                action=ActionType.click,
                target="Login",
                confidence=0.9,
            )
        ],
    )


def _audit() -> AuditResult:
    return AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1500,
            )
        ],
    )


def _inputs() -> Stage3Inputs:
    return Stage3Inputs(stage1_sha256="a" * 64, stage2_sha256="b" * 64)


def _completed_stage3(store: RecordingStore, recording_id: str) -> int:
    started = store.start_stage3(
        recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata()
    )
    store.finish_stage3(
        recording_id,
        started.model_copy(update={"status": RunStatus.completed, "result": _audit()}),
    )
    return started.version


def _stage2c_artifact() -> Stage2CArtifact:
    return Stage2CArtifact(
        metadata=_metadata(),
        produced_by_stage3_version=1,
        frame_manifest=build_frame_manifest([500, 2000, 3500]),
        raw_batch=RawFrameEvidenceBatch(
            candidates=[
                RawFrameEvidenceCandidate(
                    expected_step_id="E2",
                    frame_ids=["F3"],
                    evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
                    observed_description="Home screen",
                    visible_text=["Welcome"],
                    confidence=0.93,
                    search_status=CandidateSearchStatus.candidate_found,
                )
            ]
        ),
        candidate_evidence=CandidateEvidenceTrace(
            source_file="login.mp4",
            candidates=[
                CandidateEvidenceItem(
                    expected_step_id="E2",
                    expected_step_index=2,
                    frame_ids=["F3"],
                    timestamp_start_ms=3500,
                    timestamp_end_ms=3500,
                    evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
                    observed_description="Home screen",
                    visible_text=["Welcome"],
                    confidence=0.93,
                    search_status=CandidateSearchStatus.candidate_found,
                    validation_status=CandidateValidationStatus.accepted,
                ),
                CandidateEvidenceItem(
                    expected_step_id="E1",
                    expected_step_index=1,
                    frame_ids=["F99"],
                    evidence_type=CandidateEvidenceType.ACTION,
                    confidence=0.95,
                    search_status=CandidateSearchStatus.candidate_found,
                    validation_status=CandidateValidationStatus.rejected,
                    rejection_reason="unknown_frame_id",
                ),
            ],
        ),
        stage2c_skip_visual_steps=[2],
    )


def test_create_recording_sanitizes_scenario_heading(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store, "TC 29: Login / logout")
    assert recording_id == "TC_29_Login_logout"
    assert store.load_manifest(recording_id).title == "TC 29: Login / logout"


def test_create_recording_falls_back_to_video_heading(tmp_path: Path):
    store = _store(tmp_path)
    manifest = store.create_recording(
        title="  ??  ", source_filename="login flow.mp4", scenario_text="x"
    )
    assert manifest.recording_id == "login_flow"


def test_create_recording_never_reuses_an_existing_folder(tmp_path: Path):
    store = _store(tmp_path)
    assert [_new_recording(store), _new_recording(store)] == ["Login", "Login-2"]


def test_recording_dir_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError):
        _store(tmp_path).recording_dir("../outside")


def test_stage1_and_stage2_round_trip(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage1(recording_id, Stage1Artifact(metadata=_metadata(), expected_trace=_expected()))
    store.save_stage2(recording_id, Stage2Artifact(metadata=_metadata(), observed_trace=_observed()))
    assert store.load_stage1(recording_id).expected_trace == _expected()
    assert store.load_stage2(recording_id).observed_trace == _observed()


def test_stage_artifact_is_write_once(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage1(recording_id, Stage1Artifact(metadata=_metadata(), expected_trace=_expected()))
    before = store.artifact_sha256(recording_id, STAGE1_FILENAME)
    other = _expected().model_copy(update={"scenario_title": "Changed"})
    with pytest.raises(FileExistsError):
        store.save_stage1(recording_id, Stage1Artifact(metadata=_metadata(), expected_trace=other))
    assert store.artifact_sha256(recording_id, STAGE1_FILENAME) == before


def test_load_stage2_missing_raises_missing_artifact(tmp_path: Path):
    store = _store(tmp_path)
    with pytest.raises(MissingArtifactError):
        store.load_stage2(_new_recording(store))


def test_load_stage2c_missing_returns_none(tmp_path: Path):
    store = _store(tmp_path)
    assert store.load_stage2c(_new_recording(store)) is None


def test_stage2c_round_trip_preserves_evidence_and_validation_decisions(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage2c(recording_id, _stage2c_artifact())
    loaded = store.load_stage2c(recording_id)
    assert loaded == _stage2c_artifact()
    accepted, rejected = loaded.candidate_evidence.candidates
    assert (accepted.frame_ids, accepted.timestamp_start_ms) == (["F3"], 3500)
    assert rejected.rejection_reason == "unknown_frame_id"
    assert loaded.frame_manifest.timestamp_for("F3") == 3500


def test_write_video_records_checksum_once(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    path = store.write_video(recording_id, lambda handle: handle.write(b"video"), suffix=".MOV")
    manifest = store.load_manifest(recording_id)
    assert path.name == "recording.mov"
    assert manifest.video_sha256 == file_sha256(path)
    assert manifest.video_size_bytes == 5
    assert store.video_path(recording_id) == path
    with pytest.raises(FileExistsError):
        store.write_video(recording_id, lambda handle: handle.write(b"x"))


def test_stage3_versions_increment_and_keep_history(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    versions = [_completed_stage3(store, recording_id) for _ in range(2)]
    history = store.list_stage3(recording_id)
    assert versions == [1, 2]
    assert [(item.version, item.status) for item in history] == [
        (1, RunStatus.completed),
        (2, RunStatus.completed),
    ]


def test_finish_stage3_never_overwrites_a_finished_evaluation(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    version = _completed_stage3(store, recording_id)
    finished = store.load_stage3(recording_id, version)
    with pytest.raises(FileExistsError):
        store.finish_stage3(recording_id, finished.model_copy(update={"error": "x"}))


def test_unfinished_stage3_stays_running_and_next_run_gets_new_version(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.start_stage3(
        recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata()
    )
    assert _completed_stage3(store, recording_id) == 2
    assert [item.status for item in store.list_stage3(recording_id)] == [
        RunStatus.running,
        RunStatus.completed,
    ]


def test_leftover_temp_files_are_ignored(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    stage3_dir = store.recording_dir(recording_id) / "stage3"
    stage3_dir.mkdir()
    (stage3_dir / ".0001.json.abc123.tmp").write_text("{partial")
    assert store.list_stage3(recording_id) == []
    assert _completed_stage3(store, recording_id) == 1


def test_unreadable_evaluation_is_skipped_and_its_version_never_reused(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    stage3_dir = store.recording_dir(recording_id) / "stage3"
    stage3_dir.mkdir()
    (stage3_dir / "0001.json").write_text("{")
    assert store.list_stage3(recording_id) == []
    assert _completed_stage3(store, recording_id) == 2


def test_list_recordings_skips_folders_without_manifest_newest_first(tmp_path: Path):
    store = _store(tmp_path)
    older = _new_recording(store, "Older")
    newer = _new_recording(store, "Newer")
    manifest_path = store.recording_dir(older) / "recording.json"
    manifest = store.load_manifest(older)
    manifest_path.write_text(
        manifest.model_copy(
            update={"created_at": manifest.created_at - timedelta(days=1)}
        ).model_dump_json()
    )
    (store.root / "stray_folder").mkdir()
    assert [item.recording_id for item in store.list_recordings()] == [newer, older]


def test_stage4_records_link_to_the_stage3_version(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    version = _completed_stage3(store, recording_id)
    started = store.start_stage4(
        recording_id,
        stage3_version=version,
        stage3_sha256=store.stage3_sha256(recording_id, version),
        stage2_version=1,
        video_sha256=None,
        unverified_artifacts=[],
        skip_semantic_pairs=[(1, 1)],
        skip_visual_steps=None,
        metadata=_metadata(),
    )
    loaded = store.load_stage4(recording_id, started.version)
    assert loaded.stage3_version == version
    assert loaded.stage2_version == 1
    assert loaded.stage3_sha256 == store.stage3_sha256(recording_id, version)
    assert loaded.skip_semantic_pairs == [(1, 1)]
    assert loaded.status == RunStatus.running


def _stage2(observed: ObservedTrace | None = None) -> Stage2Artifact:
    return Stage2Artifact(metadata=_metadata(), observed_trace=observed or _observed())


def _write_running_stage3(store: RecordingStore, recording_id: str, record: Stage3Evaluation) -> None:
    directory = store.recording_dir(recording_id) / "stage3"
    directory.mkdir(exist_ok=True)
    (directory / f"{record.version:04d}.json").write_text(record.model_dump_json())


def _running_stage3(*, owner: RunOwner | None, started_at: datetime = _STARTED) -> Stage3Evaluation:
    return Stage3Evaluation(
        version=1,
        status=RunStatus.running,
        mode=Stage3Mode.deterministic,
        owner=owner,
        inputs=_inputs(),
        metadata=ExecutionMetadata(started_at=started_at, python_version="3.13.0"),
    )


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_save_stage2_assigns_new_versions_and_preserves_previous(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    first = store.save_stage2(recording_id, _stage2())
    first_sha = store.artifact_sha256(recording_id, STAGE2_FILENAME)
    retried = _observed().model_copy(update={"source_file": "retry.mp4"})
    second = store.save_stage2(recording_id, _stage2(retried))
    assert (first.version, second.version) == (1, 2)
    assert store.stage2_versions(recording_id) == [1, 2]
    assert store.artifact_sha256(recording_id, STAGE2_FILENAME) == first_sha
    assert store.load_stage2(recording_id, 1).observed_trace == _observed()
    assert store.load_stage2(recording_id, 2).observed_trace == retried


def test_load_stage2_defaults_to_latest_version(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage2(recording_id, _stage2())
    store.save_stage2(recording_id, _stage2())
    assert store.load_stage2(recording_id).version == 2


def test_load_stage2_unknown_version_raises_missing_artifact(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage2(recording_id, _stage2())
    with pytest.raises(MissingArtifactError):
        store.load_stage2(recording_id, 3)


def test_stage2c_is_stored_per_stage2_version(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage2c(recording_id, _stage2c_artifact().model_copy(update={"stage2_version": 2}))
    assert store.load_stage2c(recording_id, 1) is None
    assert store.load_stage2c(recording_id, 2).stage2_version == 2
    assert store.has_artifact(recording_id, stage2c_filename(2))


def test_saved_artifacts_verify_against_their_checksum_sidecar(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage1(recording_id, Stage1Artifact(metadata=_metadata(), expected_trace=_expected()))
    digest, verified = store.verify_artifact(recording_id, STAGE1_FILENAME)
    assert verified is True
    assert digest == store.artifact_sha256(recording_id, STAGE1_FILENAME)


def test_verify_artifact_rejects_a_modified_stage1(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.save_stage1(recording_id, Stage1Artifact(metadata=_metadata(), expected_trace=_expected()))
    path = store.recording_dir(recording_id) / STAGE1_FILENAME
    path.write_text(path.read_text().replace("Login", "Logout"))
    with pytest.raises(ArtifactIntegrityError):
        store.verify_artifact(recording_id, STAGE1_FILENAME)


def test_verify_video_rejects_a_modified_video(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    path = store.write_video(recording_id, lambda handle: handle.write(b"video"))
    assert store.verify_video(recording_id, path) == file_sha256(path)
    path.write_bytes(b"other")
    with pytest.raises(ArtifactIntegrityError):
        store.verify_video(recording_id, path)


def test_record_unretained_video_keeps_checksum_without_storing_the_file(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    digest = store.record_unretained_video(recording_id, upload)
    manifest = store.load_manifest(recording_id)
    assert (manifest.video_sha256, manifest.video_size_bytes) == (digest, 5)
    assert manifest.video_retained is False
    assert store.video_path(recording_id) is None
    assert not (store.recording_dir(recording_id) / "recording.mp4").exists()
    assert store.verify_video(recording_id, upload) == digest


def test_video_checksum_can_only_be_recorded_once(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    store.record_unretained_video(recording_id, upload)
    with pytest.raises(FileExistsError):
        store.write_video(recording_id, lambda handle: handle.write(b"video"))


def test_evidence_frames_round_trip_with_checksums(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    frames = store.save_evidence_frames(recording_id, "stage-2.v1", [("F1", 500, b"jpeg-1")])
    [frame] = frames
    assert (frame.frame_id, frame.timestamp_ms) == ("F1", 500)
    assert frame.filename == "evidence/stage-2_v1/F1.jpg"
    assert frame.sha256 == hashlib.sha256(b"jpeg-1").hexdigest()
    assert store.load_evidence_frames(recording_id, frames) == {"F1": b"jpeg-1"}


def test_load_evidence_frames_rejects_a_modified_frame(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    frames = store.save_evidence_frames(recording_id, "stage-2.v1", [("F1", 500, b"jpeg-1")])
    (store.recording_dir(recording_id) / frames[0].filename).write_bytes(b"swapped")
    with pytest.raises(ArtifactIntegrityError):
        store.load_evidence_frames(recording_id, frames)


def test_load_evidence_frames_rejects_paths_outside_the_recording(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    outside = EvidenceFrame(frame_id="F1", timestamp_ms=0, filename="../../x.jpg", sha256="a" * 64)
    with pytest.raises(MissingArtifactError):
        store.load_evidence_frames(recording_id, [outside])


def test_save_evidence_frames_rejects_unsafe_frame_ids(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    with pytest.raises(ValueError):
        store.save_evidence_frames(recording_id, "g", [("../F1", 0, b"x")])


def test_completed_evaluation_is_sealed_with_a_checksum(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    version = _completed_stage3(store, recording_id)
    digest, verified = store.verify_version(recording_id, "stage3", version)
    assert verified is True
    assert digest == store.stage3_sha256(recording_id, version)


def test_recover_stale_runs_marks_a_dead_process_run_interrupted(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    owner = RunOwner(token="t", pid=_dead_pid(), host=socket.gethostname())
    _write_running_stage3(store, recording_id, _running_stage3(owner=owner, started_at=utc_now()))
    assert store.recover_stale_runs(recording_id) == [("stage3", 1)]
    [record] = store.list_stage3(recording_id)
    assert record.status == RunStatus.interrupted
    assert "no longer running" in record.error


def test_recover_stale_runs_marks_an_orphaned_run_of_this_process_interrupted(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    owner = RunOwner(token="not-active", pid=os.getpid(), host=socket.gethostname())
    _write_running_stage3(store, recording_id, _running_stage3(owner=owner, started_at=utc_now()))
    assert store.recover_stale_runs(recording_id) == [("stage3", 1)]


def test_recover_stale_runs_leaves_an_active_run_running(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.start_stage3(recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata().model_copy(update={"started_at": utc_now()}))
    assert store.recover_stale_runs(recording_id) == []
    assert store.list_stage3(recording_id)[0].status == RunStatus.running


def test_recover_stale_runs_uses_age_for_runs_without_an_owner(tmp_path: Path):
    store = RecordingStore(tmp_path / "recordings", stale_after_s=60)
    recording_id = _new_recording(store)
    _write_running_stage3(store, recording_id, _running_stage3(owner=None, started_at=utc_now() - timedelta(minutes=5)))
    assert store.recover_stale_runs(recording_id) == [("stage3", 1)]
    assert "still running after 60s" in store.list_stage3(recording_id)[0].error


def test_recovered_version_is_never_reused(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    owner = RunOwner(token="t", pid=_dead_pid(), host=socket.gethostname())
    _write_running_stage3(store, recording_id, _running_stage3(owner=owner))
    store.recover_stale_runs(recording_id)
    assert _completed_stage3(store, recording_id) == 2


def test_owner_can_still_finish_a_run_marked_interrupted_while_alive(tmp_path: Path):
    store = RecordingStore(tmp_path / "recordings", stale_after_s=0)
    recording_id = _new_recording(store)
    started = store.start_stage3(recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata())
    assert store.recover_stale_runs(recording_id) == [("stage3", started.version)]
    store.finish_stage3(recording_id, started.model_copy(update={"status": RunStatus.completed, "result": _audit()}))
    assert store.load_stage3(recording_id, started.version).status == RunStatus.completed


def test_another_run_cannot_finish_an_interrupted_record(tmp_path: Path):
    store = RecordingStore(tmp_path / "recordings", stale_after_s=0)
    recording_id = _new_recording(store)
    started = store.start_stage3(recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata())
    store.recover_stale_runs(recording_id)
    impostor = started.model_copy(
        update={
            "status": RunStatus.completed,
            "result": _audit(),
            "owner": RunOwner(token="other", pid=os.getpid(), host=socket.gethostname()),
        }
    )
    with pytest.raises(FileExistsError):
        store.finish_stage3(recording_id, impostor)


def test_from_env_reads_retention_and_stale_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(RECORDINGS_DIR_ENV, str(tmp_path / "r"))
    monkeypatch.setenv(VIDEO_RETENTION_ENV, "discard")
    monkeypatch.setenv(STALE_RUN_SECONDS_ENV, "120")
    store = RecordingStore.from_env()
    assert (store.root, store.video_retention) == (tmp_path / "r", VideoRetention.discard)
    assert store.stale_after == timedelta(seconds=120)


_PARALLEL = 8


def _busy_peer(store: RecordingStore) -> RecordingStore:
    return RecordingStore(store.root, lock_timeout_s=0.1)


def test_recording_lock_rejects_a_second_holder_with_a_busy_error(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    with store.recording_lock(recording_id):
        with pytest.raises(RecordingBusyError, match="busy"):
            with _busy_peer(store).recording_lock(recording_id):
                pass


def test_recording_lock_is_released_after_the_operation(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    with store.recording_lock(recording_id):
        pass
    with _busy_peer(store).recording_lock(recording_id):
        pass


def test_video_checksum_update_is_rejected_while_the_recording_is_busy(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    with store.recording_lock(recording_id):
        with pytest.raises(RecordingBusyError):
            _busy_peer(store).record_unretained_video(recording_id, upload)
    assert store.load_manifest(recording_id).video_sha256 is None


def test_finishing_a_run_is_rejected_while_the_recording_is_busy(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    started = store.start_stage3(recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata())
    with store.recording_lock(recording_id):
        with pytest.raises(RecordingBusyError):
            _busy_peer(store).finish_stage3(
                recording_id, started.model_copy(update={"status": RunStatus.completed, "result": _audit()})
            )
    assert store.load_stage3(recording_id, started.version).status == RunStatus.running


def test_stale_recovery_is_rejected_while_the_recording_is_busy(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    with store.recording_lock(recording_id):
        with pytest.raises(RecordingBusyError):
            _busy_peer(store).recover_stale_runs(recording_id)


def test_simultaneous_video_metadata_updates_keep_exactly_one_checksum(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    uploads = []
    for index in range(_PARALLEL):
        upload = tmp_path / f"upload-{index}.mp4"
        upload.write_bytes(f"video-{index}".encode())
        uploads.append(upload)
    barrier = threading.Barrier(_PARALLEL)

    def record(upload: Path) -> str:
        barrier.wait()
        try:
            store.record_unretained_video(recording_id, upload)
        except FileExistsError:
            return "rejected"
        return file_sha256(upload)

    with ThreadPoolExecutor(_PARALLEL) as pool:
        outcomes = list(pool.map(record, uploads))
    winners = [outcome for outcome in outcomes if outcome != "rejected"]
    assert len(winners) == 1
    assert store.load_manifest(recording_id).video_sha256 == winners[0]


def test_simultaneous_video_writes_keep_the_file_and_checksum_consistent(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    barrier = threading.Barrier(_PARALLEL)

    def write(index: int) -> bool:
        barrier.wait()
        try:
            store.write_video(recording_id, lambda handle: handle.write(f"video-{index}".encode()))
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(_PARALLEL) as pool:
        outcomes = list(pool.map(write, range(_PARALLEL)))
    assert outcomes.count(True) == 1
    assert store.verify_video(recording_id, store.video_path(recording_id))


def test_simultaneous_evaluations_get_distinct_versions_and_all_finish(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    barrier = threading.Barrier(_PARALLEL)

    def evaluate(_index: int) -> int:
        barrier.wait()
        return _completed_stage3(store, recording_id)

    with ThreadPoolExecutor(_PARALLEL) as pool:
        versions = list(pool.map(evaluate, range(_PARALLEL)))
    assert sorted(versions) == list(range(1, _PARALLEL + 1))
    assert [record.status for record in store.list_stage3(recording_id)] == [RunStatus.completed] * _PARALLEL


def _start_attempt(store: RecordingStore, recording_id: str) -> Stage2Attempt:
    return store.start_stage2_attempt(
        recording_id, video_sha256="c" * 64, stage1_sha256="a" * 64, metadata=_metadata()
    )


def test_stage2_attempt_is_recorded_as_running_with_its_inputs(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    attempt = _start_attempt(store, recording_id)
    assert store.list_stage2_attempts(recording_id) == [attempt]
    assert (attempt.status, attempt.video_sha256, attempt.stage1_version) == (RunStatus.running, "c" * 64, 1)
    assert (store.recording_dir(recording_id) / STAGE2_ATTEMPTS_DIRNAME / f"{attempt.attempt_id}.json").is_file()


def test_failed_stage2_attempt_is_sealed_with_its_error(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    attempt = _start_attempt(store, recording_id)
    store.finish_stage2_attempt(
        recording_id, attempt.model_copy(update={"status": RunStatus.failed, "error": "RuntimeError: down"})
    )
    [saved] = store.list_stage2_attempts(recording_id)
    assert (saved.status, saved.error, saved.stage2_version) == (RunStatus.failed, "RuntimeError: down", None)
    assert store.verify_artifact(recording_id, f"{STAGE2_ATTEMPTS_DIRNAME}/{attempt.attempt_id}.json")[1]


def test_failed_stage2_attempt_requires_an_error():
    with pytest.raises(ValueError):
        Stage2Attempt(
            attempt_id="20260101T000000Z-0000abcd",
            status=RunStatus.failed,
            metadata=_metadata(),
            video_sha256="c" * 64,
        )


def test_finished_stage2_attempt_cannot_be_overwritten(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    attempt = _start_attempt(store, recording_id)
    failed = attempt.model_copy(update={"status": RunStatus.failed, "error": "RuntimeError: down"})
    store.finish_stage2_attempt(recording_id, failed)
    with pytest.raises(FileExistsError):
        store.finish_stage2_attempt(recording_id, failed.model_copy(update={"error": "rewritten"}))


def test_stage2_attempt_ids_reject_path_traversal(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    attempt = _start_attempt(store, recording_id)
    with pytest.raises(ValueError):
        store.finish_stage2_attempt(
            recording_id,
            attempt.model_copy(update={"attempt_id": "../stage1", "status": RunStatus.failed, "error": "x"}),
        )


def test_recover_stale_runs_marks_an_orphaned_stage2_attempt_interrupted(tmp_path: Path):
    store = RecordingStore(tmp_path / "recordings", stale_after_s=0)
    recording_id = _new_recording(store)
    attempt = _start_attempt(store, recording_id)
    assert store.recover_stale_runs(recording_id) == [(STAGE2_ATTEMPTS_DIRNAME, attempt.attempt_id)]
    [saved] = store.list_stage2_attempts(recording_id)
    assert saved.status == RunStatus.interrupted


def test_a_run_that_could_not_finish_because_of_a_busy_lock_is_recoverable(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    started = store.start_stage3(
        recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata().model_copy(update={"started_at": utc_now()})
    )
    with store.recording_lock(recording_id):
        with pytest.raises(RecordingBusyError):
            _busy_peer(store).finish_stage3(
                recording_id, started.model_copy(update={"status": RunStatus.completed, "result": _audit()})
            )
    assert store.recover_stale_runs(recording_id) == [("stage3", started.version)]


def _embedded(path: Path) -> str:
    return json.loads(path.read_text())[INTEGRITY_FIELD]["sha256"]


def _populated(store: RecordingStore) -> str:
    recording_id = _new_recording(store)
    store.save_stage1(recording_id, Stage1Artifact(metadata=_metadata(), expected_trace=_expected()))
    store.save_stage2(recording_id, _stage2())
    store.save_stage2(recording_id, _stage2())
    store.save_stage2c(recording_id, _stage2c_artifact())
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    stage3 = _completed_stage3(store, recording_id)
    started4 = store.start_stage4(
        recording_id,
        stage3_version=stage3,
        stage3_sha256=store.stage3_sha256(recording_id, stage3),
        stage2_version=1,
        video_sha256=None,
        unverified_artifacts=[],
        skip_semantic_pairs=None,
        skip_visual_steps=None,
        metadata=_metadata(),
    )
    store.finish_stage4(recording_id, started4.model_copy(update={"status": RunStatus.failed, "error": "x"}))
    attempt = store.start_stage2_attempt(recording_id, video_sha256="c" * 64, stage1_sha256=None, metadata=_metadata())
    store.finish_stage2_attempt(recording_id, attempt.model_copy(update={"status": RunStatus.failed, "error": "x"}))
    return recording_id


def test_every_saved_json_artifact_embeds_a_verifying_checksum(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _populated(store)
    paths = sorted(store.recording_dir(recording_id).rglob("*.json"))
    assert {p.relative_to(store.recording_dir(recording_id)).as_posix() for p in paths} >= {
        "recording.json",
        "stage-1.json",
        "stage-2.json",
        "stage-2.v2.json",
        "stage-2c.json",
        "stage3/0001.json",
        "stage4/0001.json",
    }
    for path in paths:
        assert verify_json_artifact(path) == (_embedded(path), True), path.name


def test_new_saves_never_create_sidecar_files(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _populated(store)
    store.save_evidence_frames(recording_id, "g", [("F1", 0, b"jpeg")])
    assert list(store.recording_dir(recording_id).rglob("*.sha256")) == []


def test_artifact_sha256_is_the_embedded_checksum(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _populated(store)
    directory = store.recording_dir(recording_id)
    assert store.artifact_sha256(recording_id, STAGE1_FILENAME) == _embedded(directory / STAGE1_FILENAME)
    assert store.stage3_sha256(recording_id, 1) == _embedded(directory / "stage3" / "0001.json")


def test_stage4_records_the_embedded_checksum_of_its_stage3_input(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _populated(store)
    [stage4] = store.list_stage4(recording_id)
    assert stage4.stage3_sha256 == _embedded(store.recording_dir(recording_id) / "stage3" / "0001.json")


def test_recording_metadata_update_keeps_its_own_checksum_valid(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    manifest_path = store.recording_dir(recording_id) / "recording.json"
    before = _embedded(manifest_path)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    assert _embedded(manifest_path) != before
    assert verify_json_artifact(manifest_path) == (_embedded(manifest_path), True)
    assert store.load_manifest(recording_id).video_sha256 == hashlib.sha256(b"video").hexdigest()


def test_a_modified_recording_manifest_is_refused(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    manifest_path = store.recording_dir(recording_id) / "recording.json"
    payload = json.loads(manifest_path.read_text())
    payload["video_sha256"] = hashlib.sha256(b"replaced").hexdigest()
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactIntegrityError):
        store.load_manifest(recording_id)
    assert store.list_recordings() == []


def test_a_modified_stage2_version_is_detected(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _populated(store)
    tamper_json(store.recording_dir(recording_id) / "stage-2.v2.json")
    with pytest.raises(ArtifactIntegrityError):
        store.verify_artifact(recording_id, "stage-2.v2.json")
    assert store.verify_artifact(recording_id, STAGE2_FILENAME)[1] is True


def test_legacy_recording_with_sidecars_still_loads_and_verifies(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _populated(store)
    directory = store.recording_dir(recording_id)
    make_legacy(directory)
    assert store.list_recordings()[0].recording_id == recording_id
    digest, verified = store.verify_artifact(recording_id, STAGE1_FILENAME)
    assert (digest, verified) == (file_sha256(directory / STAGE1_FILENAME), True)
    assert store.verify_version(recording_id, "stage3", 1)[1] is True
    assert store.load_stage2(recording_id, 2) == _stage2().model_copy(update={"version": 2})


def test_legacy_artifact_modified_after_its_sidecar_is_still_refused(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _populated(store)
    directory = store.recording_dir(recording_id)
    make_legacy(directory)
    stage1 = directory / STAGE1_FILENAME
    stage1.write_text(stage1.read_text().replace('"Login"', '"Logout"'))
    with pytest.raises(ArtifactIntegrityError, match="modified after it was saved"):
        store.verify_artifact(recording_id, STAGE1_FILENAME)


def test_legacy_manifest_gains_an_embedded_checksum_on_its_next_write(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    manifest_path = store.recording_dir(recording_id) / "recording.json"
    make_legacy(store.recording_dir(recording_id))
    assert verify_json_artifact(manifest_path)[1] is False
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    store.record_unretained_video(recording_id, upload)
    assert verify_json_artifact(manifest_path) == (_embedded(manifest_path), True)


def test_legacy_running_evaluation_gains_an_embedded_checksum_when_it_finishes(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    started = store.start_stage3(recording_id, mode=Stage3Mode.deterministic, inputs=_inputs(), metadata=_metadata())
    path = store.recording_dir(recording_id) / "stage3" / "0001.json"
    make_legacy(store.recording_dir(recording_id))
    assert verify_json_artifact(path)[1] is False
    store.finish_stage3(recording_id, started.model_copy(update={"status": RunStatus.completed, "result": _audit()}))
    assert verify_json_artifact(path) == (_embedded(path), True)
    assert not legacy_checksum_path(path).exists()


def test_legacy_running_evaluation_gains_an_embedded_checksum_when_recovered(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    owner = RunOwner(token="t", pid=_dead_pid(), host=socket.gethostname())
    _write_running_stage3(store, recording_id, _running_stage3(owner=owner, started_at=utc_now()))
    path = store.recording_dir(recording_id) / "stage3" / "0001.json"
    assert store.recover_stale_runs(recording_id) == [("stage3", 1)]
    assert verify_json_artifact(path) == (_embedded(path), True)


def test_legacy_completed_evaluations_keep_their_history_and_version_numbers(tmp_path: Path):
    store = _store(tmp_path)
    recording_id = _new_recording(store)
    _completed_stage3(store, recording_id)
    make_legacy(store.recording_dir(recording_id))
    legacy_bytes = (store.recording_dir(recording_id) / "stage3" / "0001.json").read_bytes()
    assert _completed_stage3(store, recording_id) == 2
    assert (store.recording_dir(recording_id) / "stage3" / "0001.json").read_bytes() == legacy_bytes
    assert [record.version for record in store.list_stage3(recording_id)] == [1, 2]
