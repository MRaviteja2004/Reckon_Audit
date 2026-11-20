import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from persistence_fakes import traces  # noqa: E402
from reckon.persistence import RECORDINGS_DIR_ENV, RecordingStore, RunStatus  # noqa: E402
from reckon.schema import AuditResult, AuditVerdict, CrossCheckReport  # noqa: E402
from scripts import run_pipeline  # noqa: E402

pytestmark = pytest.mark.usefixtures("no_live_gateway")


def _report() -> CrossCheckReport:
    return CrossCheckReport(
        verdicts=[],
        refined_result=AuditResult(
            verdict=AuditVerdict.faithful, fidelity_score=1.0, findings=[], step_verdicts=[]
        ),
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "login.mp4"
    video.write_bytes(b"video")
    scenario = tmp_path / "scenario.txt"
    scenario.write_text("1. Click the Login button")
    return video, scenario


def _run(argv: list[str]) -> None:
    expected, observed = traces()
    with (
        patch.object(sys, "argv", ["run_pipeline.py", *argv]),
        patch.object(run_pipeline, "parse_scenario", return_value=expected),
        patch.object(run_pipeline, "make_client", return_value=MagicMock()),
        patch.object(run_pipeline, "extract_observed_trace", return_value=observed),
        patch.object(run_pipeline, "run_cross_check", return_value=_report()),
        patch("reckon.persistence.extraction.make_client", return_value=MagicMock()),
        patch("reckon.persistence.extraction.extract_observed_trace", return_value=observed),
        patch("reckon.persistence.runner.run_cross_check", return_value=_report()),
    ):
        run_pipeline.main()


def test_run_pipeline_without_save_writes_no_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(RECORDINGS_DIR_ENV, str(tmp_path / "recordings"))
    video, scenario = _inputs(tmp_path)
    _run([str(video), str(scenario)])
    assert not (tmp_path / "recordings").exists()


def test_run_pipeline_save_persists_through_the_shared_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(RECORDINGS_DIR_ENV, str(tmp_path / "recordings"))
    video, scenario = _inputs(tmp_path)
    _run([str(video), str(scenario), "--save"])
    store = RecordingStore.from_env()
    [recording] = store.list_recordings()
    rid = recording.recording_id
    assert store.video_path(rid).read_bytes() == b"video"
    assert store.stage2_versions(rid) == [1]
    [stage3] = store.list_stage3(rid)
    [stage4] = store.list_stage4(rid)
    assert stage3.status == RunStatus.completed
    assert stage4.stage3_version == stage3.version
