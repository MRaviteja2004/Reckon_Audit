"""Click the real Streamlit buttons against a temporary store with mocked model clients."""

import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from persistence_fakes import (
    FakeGateway,
    fake_jpegs,
    frozen_stage2c_for_other_step,
    raw_batch,
    tamper_json,
    traces,
)
from reckon.persistence import (
    RECORDINGS_DIR_ENV,
    STAGE1_FILENAME,
    VIDEO_RETENTION_ENV,
    ExecutionMetadata,
    RecordingBusyError,
    RecordingStore,
    RunOwner,
    RunStatus,
    Stage1Artifact,
    Stage2Artifact,
    Stage2CSource,
    Stage3Evaluation,
    Stage3Inputs,
    Stage3Mode,
    VideoRetention,
)
from reckon.schema import AuditResult, AuditVerdict, CrossCheckReport
from reckon.stage2c.propose import RawFrameEvidenceBatch

pytestmark = pytest.mark.usefixtures("no_live_gateway")

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_TIMEOUT_S = 60
EVIDENCE_MODE_LABEL = "LLM-assisted evidence-aware (experimental)"
_METADATA = ExecutionMetadata(
    started_at=datetime(2026, 1, 1, tzinfo=timezone.utc), python_version="3.13.0"
)


class _Doubles:
    """Every model-facing entry point the app can reach, replaced for one test."""

    def __init__(self) -> None:
        self.gateway = FakeGateway()
        expected, observed = traces()
        self.parse_scenario = MagicMock(return_value=expected)
        self.extract = MagicMock(return_value=observed)
        self.cross_check = MagicMock(
            return_value=CrossCheckReport(
                verdicts=[],
                refined_result=AuditResult(
                    verdict=AuditVerdict.faithful, fidelity_score=1.0, findings=[], step_verdicts=[]
                ),
            )
        )

    def install(self, stack: ExitStack) -> None:
        stack.enter_context(patch("reckon.parser.make_openai_client", return_value=self.gateway))
        stack.enter_context(patch("reckon.parser.parse_scenario", self.parse_scenario))
        stack.enter_context(patch("reckon.persistence.extraction.make_client", return_value=MagicMock()))
        stack.enter_context(patch("reckon.persistence.extraction.extract_observed_trace", self.extract))
        stack.enter_context(patch("reckon.persistence.extraction.load_frame_jpegs", side_effect=fake_jpegs))
        stack.enter_context(patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs))
        stack.enter_context(patch("reckon.persistence.runner.run_cross_check", self.cross_check))


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RecordingStore:
    monkeypatch.setenv(RECORDINGS_DIR_ENV, str(tmp_path / "recordings"))
    monkeypatch.delenv(VIDEO_RETENTION_ENV, raising=False)
    return RecordingStore.from_env()


@pytest.fixture
def doubles() -> Iterator[_Doubles]:
    installed = _Doubles()
    with ExitStack() as stack:
        installed.install(stack)
        yield installed


def _app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_S).run()
    assert not app.exception
    return app


def _workspace() -> AppTest:
    """The Saved Recordings workspace, opened from the main screen."""
    return _click(_app(), key="use_saved_recording")


def _click(app: AppTest, *, key: str | None = None, label: str | None = None) -> AppTest:
    button = app.button(key=key) if key else next(b for b in app.button if b.label == label)
    app = button.click().run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _seed(store: RecordingStore) -> str:
    expected, observed = traces()
    recording_id = store.create_recording(
        title="Login", source_filename="login.mp4", scenario_text="Click Login"
    ).recording_id
    store.save_stage1(recording_id, Stage1Artifact(metadata=_METADATA, expected_trace=expected))
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    return recording_id


def _run_full_pipeline(app: AppTest) -> AppTest:
    app.file_uploader[0].upload("login.mp4", b"video", "video/mp4")
    app.text_area[0].input("1. Click the Login button\n2. Driver home screen is displayed")
    app.radio[0].set_value(EVIDENCE_MODE_LABEL)
    return _click(app.run(), label="Run Full Pipeline")


def test_run_full_pipeline_button_persists_every_stage(store: RecordingStore, doubles: _Doubles):
    _run_full_pipeline(_app())
    [recording] = store.list_recordings()
    rid = recording.recording_id
    assert store.stage2_versions(rid) == [1]
    assert store.video_path(rid).read_bytes() == b"video"
    assert store.load_stage2c(rid).raw_batch == raw_batch()
    [stage3] = store.list_stage3(rid)
    [stage4] = store.list_stage4(rid)
    assert (stage3.status, stage3.mode, stage3.inputs.stage2c_source) == (
        RunStatus.completed,
        Stage3Mode.evidence_aware,
        Stage2CSource.live,
    )
    assert len(stage3.model_calls) == len(doubles.gateway.requested)
    assert (stage4.status, stage4.stage3_version) == (RunStatus.completed, stage3.version)
    assert stage4.stage3_sha256 == store.stage3_sha256(rid, stage3.version)


def test_run_full_pipeline_with_discard_retention_keeps_checksum_and_evidence_stills(
    store: RecordingStore, doubles: _Doubles, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(VIDEO_RETENTION_ENV, VideoRetention.discard.value)
    _run_full_pipeline(_app())
    [recording] = store.list_recordings()
    rid = recording.recording_id
    assert recording.video_retained is False and recording.video_sha256 is not None
    assert store.video_path(rid) is None
    stills = store.load_stage2(rid).evidence_frames
    assert store.load_evidence_frames(rid, stills) == {f.frame_id: b"jpeg" for f in stills}
    assert Path(doubles.cross_check.call_args.args[3]).name == "upload.mp4"
    assert store.list_stage4(rid)[0].video_sha256 == recording.video_sha256


def test_run_stage3_button_uses_saved_artifacts_without_extraction(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    _click(_workspace(), key="run_stage3")
    doubles.extract.assert_not_called()
    doubles.parse_scenario.assert_not_called()
    [stage3] = store.list_stage3(rid)
    assert (stage3.status, stage3.inputs.stage2c_source) == (RunStatus.completed, Stage2CSource.live)
    assert store.load_stage2c(rid) is not None


def test_second_run_stage3_replays_stage2c_without_a_stage2c_model_call(
    store: RecordingStore, doubles: _Doubles
):
    rid = _seed(store)
    app = _click(_workspace(), key="run_stage3")
    calls_after_first = doubles.gateway.requested.count(RawFrameEvidenceBatch)
    _click(app, key="run_stage3")
    assert doubles.gateway.requested.count(RawFrameEvidenceBatch) == calls_after_first == 1
    first, second = store.list_stage3(rid)
    assert second.inputs.stage2c_source == Stage2CSource.replayed
    assert second.stage2c_replay.unmatched_targets == []
    assert store.load_stage3(rid, first.version) == first


def test_run_stage4_button_cross_checks_the_selected_stage3(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    app = _click(_click(_workspace(), key="run_stage3"), key="run_stage3")
    app.selectbox(key=f"stage3_version_{rid}").set_value(1)
    _click(app.run(), key="run_stage4")
    [stage4] = store.list_stage4(rid)
    assert stage4.stage3_version == 1
    assert doubles.cross_check.call_args.args[2] == store.load_stage3(rid, 1).result


def test_retry_stage2_button_adds_a_version_without_running_stage1(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    stage1_sha = store.artifact_sha256(rid, STAGE1_FILENAME)
    _click(_workspace(), key="retry_stage2")
    doubles.parse_scenario.assert_not_called()
    doubles.extract.assert_called_once()
    assert store.stage2_versions(rid) == [1, 2]
    assert store.artifact_sha256(rid, STAGE1_FILENAME) == stage1_sha


def test_run_stage3_button_uses_the_selected_stage2_version(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    app = _click(_workspace(), key="retry_stage2")
    app.selectbox(key=f"stage2_version_{rid}").set_value(1)
    _click(app.run(), key="run_stage3")
    [stage3] = store.list_stage3(rid)
    assert stage3.inputs.stage2_version == 1


def test_run_stage3_button_reports_a_modified_artifact_and_saves_nothing(
    store: RecordingStore, doubles: _Doubles
):
    rid = _seed(store)
    tamper_json(store.recording_dir(rid) / "stage-2.json")
    app = _click(_workspace(), key="run_stage3")
    assert any("modified after it was saved" in error.value for error in app.error)
    assert store.list_stage3(rid) == []


def test_opening_a_recording_marks_a_dead_run_interrupted(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    stale = Stage3Evaluation(
        version=1,
        status=RunStatus.running,
        mode=Stage3Mode.evidence_aware,
        owner=RunOwner(token="gone", pid=process.pid, host=socket.gethostname()),
        inputs=Stage3Inputs(stage1_sha256="a" * 64, stage2_sha256="b" * 64),
        metadata=_METADATA,
    )
    (store.recording_dir(rid) / "stage3").mkdir()
    (store.recording_dir(rid) / "stage3" / "0001.json").write_text(stale.model_dump_json())
    app = _workspace()
    assert store.list_stage3(rid)[0].status == RunStatus.interrupted
    _click(app, key="run_stage3")
    statuses = [(item.version, item.status) for item in store.list_stage3(rid)]
    assert statuses == [(1, RunStatus.interrupted), (2, RunStatus.completed)]


def test_app_tests_never_reach_the_live_gateway(store: RecordingStore, doubles: _Doubles):
    _seed(store)
    with patch("reckon.parser.make_openai_client", side_effect=AssertionError("live client")):
        app = _click(_workspace(), key="run_stage3")
    assert any("live client" in error.value for error in app.error)



def _seed_without_video(store: RecordingStore, tmp_path: Path) -> str:
    expected, observed = traces()
    recording_id = store.create_recording(
        title="Login", source_filename="login.mp4", scenario_text="Click Login"
    ).recording_id
    store.save_stage1(recording_id, Stage1Artifact(metadata=_METADATA, expected_trace=expected))
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    store.record_unretained_video(recording_id, upload)
    return recording_id


def test_run_stage4_is_disabled_with_a_reason_when_the_video_was_discarded(
    store: RecordingStore, doubles: _Doubles, tmp_path: Path
):
    _seed_without_video(store, tmp_path)
    app = _workspace()
    assert app.button(key="run_stage4").disabled
    assert any("Stage 4 needs the original video" in caption.value for caption in app.caption)


def test_retry_stage2_is_disabled_with_a_reason_when_the_video_was_discarded(
    store: RecordingStore, doubles: _Doubles, tmp_path: Path
):
    _seed_without_video(store, tmp_path)
    app = _workspace()
    assert app.button(key="retry_stage2").disabled
    assert any("Stage 2 retry needs the original video" in caption.value for caption in app.caption)


def test_run_stage3_is_disabled_with_a_reason_when_no_frame_source_survives(
    store: RecordingStore, doubles: _Doubles, tmp_path: Path
):
    _seed_without_video(store, tmp_path)
    app = _workspace()
    assert app.button(key="run_stage3").disabled
    assert any("Stage 3 evidence review needs frames" in caption.value for caption in app.caption)


def test_a_busy_recording_is_reported_and_nothing_is_overwritten(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    busy = RecordingBusyError(f"Recording {rid} is busy with another operation; try again shortly")
    with patch.object(RecordingStore, "finish_stage3", side_effect=busy):
        app = _click(_workspace(), key="run_stage3")
    assert any("is busy with another operation" in error.value for error in app.error)
    assert [item.status for item in store.list_stage3(rid)] == [RunStatus.running]


def test_failed_stage2_retry_reports_the_error_and_records_the_attempt(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    doubles.extract.side_effect = RuntimeError("extraction gateway down")
    app = _click(_workspace(), key="retry_stage2")
    assert any("extraction gateway down" in error.value for error in app.error)
    assert app.selectbox(key=f"stage2_version_{rid}").options == ["v1"]
    [attempt] = store.list_stage2_attempts(rid)
    assert (attempt.status, attempt.error) == (RunStatus.failed, "RuntimeError: extraction gateway down")
    assert store.stage2_versions(rid) == [1]


def test_replay_warning_reports_unanswered_steps_as_not_searched(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    _expected, observed = traces()
    store.save_stage2c(rid, frozen_stage2c_for_other_step(observed))
    app = _click(_workspace(), key="run_stage3")
    [warning] = [w.value for w in app.warning if "Stage 2C replay" in w.value]
    assert "E2" in warning and "not searched" in warning
    assert "treated as not found" not in warning


_PANEL_BUTTON_KEYS = {"run_stage3", "run_stage4", "retry_stage2"}


def test_saved_recordings_shows_only_the_selectors_and_three_actions(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    app = _click(_workspace(), key="run_stage3")
    assert {button.key for button in app.button if button.key} == _PANEL_BUTTON_KEYS | {
        "saved_recordings_back"
    }
    assert [box.key for box in app.selectbox] == [
        "selected_recording",
        f"stage2_version_{rid}",
        f"stage3_version_{rid}",
    ]
    assert len(app.dataframe) == 0
    panel_text = " ".join(c.value for c in app.caption)
    assert "sha256" not in panel_text and "stage-1.json" not in panel_text


def test_selecting_a_recording_updates_its_stage2_versions_and_stage3_results(
    store: RecordingStore, doubles: _Doubles
):
    first = _seed(store)
    _expected, observed = traces()
    store.save_stage2(first, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    second = _seed(store)
    app = _workspace()
    app.selectbox(key="selected_recording").set_value(first)
    app = app.run()
    assert app.selectbox(key=f"stage2_version_{first}").options == ["v2", "v1"]
    app.selectbox(key="selected_recording").set_value(second)
    app = app.run()
    assert app.selectbox(key=f"stage2_version_{second}").options == ["v1"]
    assert not any(box.key == f"stage3_version_{second}" for box in app.selectbox)


def test_retry_stage2_refreshes_the_selector_with_the_new_version_selected(
    store: RecordingStore, doubles: _Doubles
):
    rid = _seed(store)
    app = _click(_workspace(), key="retry_stage2")
    selector = app.selectbox(key=f"stage2_version_{rid}")
    assert (selector.options, selector.value) == (["v2", "v1"], 2)
    assert any("Saved Stage 2 v2" in message.value for message in app.success)
    assert store.stage2_versions(rid) == [1, 2]


def test_run_stage3_uses_the_selected_stage2_and_selects_the_new_result_for_stage4(
    store: RecordingStore, doubles: _Doubles
):
    rid = _seed(store)
    app = _click(_workspace(), key="retry_stage2")
    app.selectbox(key=f"stage2_version_{rid}").set_value(1)
    app = _click(app.run(), key="run_stage3")
    assert app.selectbox(key=f"stage3_version_{rid}").value == 1
    _click(app, key="run_stage3")
    app = _workspace()
    app.selectbox(key=f"stage3_version_{rid}").set_value(1)
    _click(app.run(), key="run_stage4")
    assert [item.inputs.stage2_version for item in store.list_stage3(rid)] == [1, 1]
    [stage4] = store.list_stage4(rid)
    assert (stage4.stage3_version, stage4.stage2_version) == (1, 1)


def test_run_stage3_keeps_previous_results(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    app = _click(_click(_workspace(), key="run_stage3"), key="run_stage3")
    assert app.selectbox(key=f"stage3_version_{rid}").value == 2
    first, second = store.list_stage3(rid)
    assert (first.version, second.version) == (1, 2)
    assert first.status == second.status == RunStatus.completed


def test_main_screen_hides_saved_recordings_until_opened(store: RecordingStore, doubles: _Doubles):
    _seed(store)
    app = _app()
    assert app.button(key="use_saved_recording").label == "Use Saved Recording"
    assert not any(button.key in _PANEL_BUTTON_KEYS for button in app.button)
    assert not any(box.key == "selected_recording" for box in app.selectbox)
    assert any("Upload a video" in block.label for block in app.file_uploader)


def test_opening_saved_recordings_makes_no_pipeline_or_model_call(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    app = _workspace()
    doubles.extract.assert_not_called()
    doubles.parse_scenario.assert_not_called()
    doubles.cross_check.assert_not_called()
    assert doubles.gateway.requested == []
    assert app.header[0].value == "Saved Recordings"
    assert app.selectbox(key="selected_recording").value == rid
    assert {button.key for button in app.button if button.key in _PANEL_BUTTON_KEYS} == _PANEL_BUTTON_KEYS


def test_back_returns_to_the_main_screen_without_changing_recordings(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    saved = sorted(path.relative_to(store.recording_dir(rid)).as_posix() for path in store.recording_dir(rid).rglob("*") if path.is_file())
    app = _click(_workspace(), key="saved_recordings_back")
    assert any(button.key == "use_saved_recording" for button in app.button)
    assert not any(button.key in _PANEL_BUTTON_KEYS for button in app.button)
    assert sorted(
        path.relative_to(store.recording_dir(rid)).as_posix()
        for path in store.recording_dir(rid).rglob("*")
        if path.is_file()
    ) == saved
    assert store.stage2_versions(rid) == [1]


def test_reopening_saved_recordings_refreshes_versions_from_disk(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    _expected, observed = traces()
    app = _click(_workspace(), key="saved_recordings_back")
    store.save_stage2(rid, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    app = _click(app, key="use_saved_recording")
    assert app.selectbox(key=f"stage2_version_{rid}").options == ["v2", "v1"]


def test_version_selection_is_kept_while_the_workspace_stays_open(store: RecordingStore, doubles: _Doubles):
    rid = _seed(store)
    _expected, observed = traces()
    store.save_stage2(rid, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    app = _workspace()
    app.selectbox(key=f"stage2_version_{rid}").set_value(1)
    app = app.run()
    assert not app.exception
    assert app.selectbox(key=f"stage2_version_{rid}").value == 1


def test_empty_recordings_directory_shows_a_message(store: RecordingStore, doubles: _Doubles):
    app = _workspace()
    assert any("No saved recordings yet" in item.value for item in app.info)
    assert app.button(key="saved_recordings_back").label == "Back"
    assert not any(box.key == "selected_recording" for box in app.selectbox)
