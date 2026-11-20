import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

from conftest import LiveGatewayCallError
from persistence_fakes import (
    FakeGateway,
    fake_jpegs,
    frozen_stage2c_for_other_step,
    make_legacy,
    raw_batch,
    strip_integrity,
    tamper_json,
    traces,
)
from reckon.aligner import align
from reckon.persistence import (
    STAGE1_FILENAME,
    STAGE2_FILENAME,
    STAGE2C_FILENAME,
    STAGE2C_NOT_SEARCHED_REASON,
    ArtifactIntegrityError,
    ExecutionMetadata,
    MissingArtifactError,
    RecordingBusyError,
    RecordingClient,
    RecordingStore,
    RunStatus,
    Stage1Artifact,
    Stage2Artifact,
    Stage2CSource,
    Stage3Evaluation,
    Stage3Inputs,
    Stage3Mode,
    VideoRetention,
    frame_manifest_for,
    reviewed_skip_pairs,
    run_stage3,
    run_stage4,
    stage2_filename,
)
from reckon.persistence import runner as persistence_runner
from reckon.schema import (
    ActionType,
    AssistCandidateKind,
    AssistMetrics,
    AssistProposalStatus,
    AssistReport,
    AssistReviewedPair,
    AuditResult,
    AuditVerdict,
    CandidateValidationStatus,
    CrossCheckReport,
    ExpectedStep,
    ExpectedTrace,
    ObservedAction,
    ObservedTrace,
)
from reckon.stage2c.propose import RawFrameEvidenceBatch

pytestmark = pytest.mark.usefixtures("no_live_gateway")

_METADATA = ExecutionMetadata(
    started_at=datetime(2026, 1, 1, tzinfo=timezone.utc), python_version="3.13.0"
)


def _seeded_store(tmp_path: Path, *, with_stage2: bool = True, with_video: bool = True) -> tuple[RecordingStore, str]:
    store = RecordingStore(tmp_path / "recordings")
    expected, observed = traces()
    recording_id = store.create_recording(
        title="Login", source_filename="login.mp4", scenario_text="Click Login"
    ).recording_id
    store.save_stage1(recording_id, Stage1Artifact(metadata=_METADATA, expected_trace=expected))
    if with_stage2:
        store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    if with_video:
        store.write_video(recording_id, lambda handle: handle.write(b"video"))
    return store, recording_id


def _audit() -> AuditResult:
    return AuditResult(verdict=AuditVerdict.faithful, fidelity_score=1.0, findings=[], step_verdicts=[])


def _assist_report() -> AssistReport:
    return AssistReport(
        model="claude-opus-5-5",
        metrics=AssistMetrics(
            expected_event_count=1,
            observed_event_count=1,
            deterministic_match_count=0,
            candidate_count=1,
            llm_request_count=1,
            proposals_accepted=1,
            proposals_rejected=0,
            unresolved_count=0,
            deterministic_ms=1.0,
            llm_ms=1.0,
            total_ms=2.0,
        ),
        reviewed_semantic_pairs=[AssistReviewedPair(expected_step_index=1, observed_action_index=1)],
        stage2c_skip_visual_steps=[2],
    )


def test_run_stage3_requires_stage2_artifact(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path, with_stage2=False)
    with pytest.raises(MissingArtifactError):
        run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert store.list_stage3(recording_id) == []


def test_run_stage3_deterministic_matches_align_and_records_inputs(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    expected, observed = traces()
    evaluation = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert evaluation.status == RunStatus.completed
    assert evaluation.result == align(expected, observed)
    assert evaluation.inputs.stage1_sha256 == store.artifact_sha256(recording_id, STAGE1_FILENAME)
    assert evaluation.inputs.stage2_sha256 == store.artifact_sha256(recording_id, STAGE2_FILENAME)
    assert evaluation.metadata.duration_ms is not None
    assert store.load_stage3(recording_id, evaluation.version) == evaluation


def test_run_stage3_live_stage2c_is_saved_with_evidence_and_frame_refs(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    gateway = FakeGateway()
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        evaluation = run_stage3(store, recording_id, openai_client=gateway)
    artifact = store.load_stage2c(recording_id)
    assert gateway.requested.count(RawFrameEvidenceBatch) == 1
    assert evaluation.inputs.stage2c_source == Stage2CSource.live
    assert artifact is not None and artifact.produced_by_stage3_version == evaluation.version
    assert artifact.raw_batch == raw_batch()
    candidate = next(c for c in artifact.candidate_evidence.candidates if c.expected_step_index == 2)
    assert candidate.validation_status == CandidateValidationStatus.accepted
    assert (candidate.frame_ids, candidate.timestamp_start_ms) == (["F3"], 3500)
    assert artifact.frame_manifest.timestamp_for("F3") == 3500
    recovered = next(v for v in evaluation.result.step_verdicts if v.expected_step_index == 2)
    assert (recovered.passed, recovered.evidence_timestamp_ms) == (True, 3500)


def test_run_stage3_replays_saved_stage2c_without_a_stage2c_model_call(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        first = run_stage3(store, recording_id, openai_client=FakeGateway())
    first_sha = store.stage3_sha256(recording_id, first.version)
    stage2c_sha = store.artifact_sha256(recording_id, STAGE2C_FILENAME)
    gateway = FakeGateway()
    second = run_stage3(store, recording_id, openai_client=gateway)
    assert RawFrameEvidenceBatch not in gateway.requested
    assert second.version == first.version + 1
    assert second.inputs.stage2c_source == Stage2CSource.replayed
    assert second.inputs.stage2c_sha256 == stage2c_sha
    assert second.result.step_verdicts == first.result.step_verdicts
    assert store.stage3_sha256(recording_id, first.version) == first_sha
    assert store.artifact_sha256(recording_id, STAGE2C_FILENAME) == stage2c_sha


def test_run_stage3_failed_stage2c_discovery_is_not_frozen(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    gateway = FakeGateway(stage2c_error=ValueError("bad batch"))
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        evaluation = run_stage3(store, recording_id, openai_client=gateway)
    assert evaluation.status == RunStatus.completed
    assert evaluation.assist.error == "llm_ValueError"
    assert store.load_stage2c(recording_id) is None


def test_run_stage3_failure_is_recorded_and_reraised(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    with patch.object(persistence_runner, "align", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    [failed] = store.list_stage3(recording_id)
    assert failed.status == RunStatus.failed
    assert "RuntimeError: boom" in failed.error


def test_run_stage4_uses_the_exact_selected_stage3_evaluation(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    first = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    report = CrossCheckReport(verdicts=[], refined_result=_audit())
    with patch.object(persistence_runner, "run_cross_check", return_value=report) as cross:
        evaluation = run_stage4(store, recording_id, first.version)
    args, kwargs = cross.call_args
    assert args[2] == first.result
    assert args[3] == store.video_path(recording_id)
    assert isinstance(kwargs.pop("openai_client"), RecordingClient)
    assert kwargs == {"skip_semantic_pairs": None, "skip_visual_steps": None}
    assert evaluation.stage3_version == first.version
    assert evaluation.stage3_sha256 == store.stage3_sha256(recording_id, first.version)
    assert store.load_stage4(recording_id, evaluation.version).report == report


def test_run_stage4_passes_reviewed_pairs_and_stage2c_visual_skips(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    started = store.start_stage3(
        recording_id,
        mode=Stage3Mode.evidence_aware,
        inputs=Stage3Inputs(
            stage1_sha256=store.artifact_sha256(recording_id, STAGE1_FILENAME),
            stage2_sha256=store.artifact_sha256(recording_id, STAGE2_FILENAME),
        ),
        metadata=_METADATA,
    )
    store.finish_stage3(
        recording_id,
        started.model_copy(
            update={"status": RunStatus.completed, "result": _audit(), "assist": _assist_report()}
        ),
    )
    report = CrossCheckReport(verdicts=[], refined_result=_audit())
    with patch.object(persistence_runner, "run_cross_check", return_value=report) as cross:
        evaluation = run_stage4(store, recording_id, started.version)
    kwargs = dict(cross.call_args.kwargs)
    assert isinstance(kwargs.pop("openai_client"), RecordingClient)
    assert kwargs == {"skip_semantic_pairs": {(1, 1)}, "skip_visual_steps": {2}}
    assert evaluation.skip_semantic_pairs == [(1, 1)]
    assert evaluation.skip_visual_steps == [2]


def test_run_stage4_rejects_an_interrupted_stage3(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    started = store.start_stage3(
        recording_id,
        mode=Stage3Mode.deterministic,
        inputs=Stage3Inputs(stage1_sha256="a" * 64, stage2_sha256="b" * 64),
        metadata=_METADATA,
    )
    with pytest.raises(ValueError, match="running"):
        run_stage4(store, recording_id, started.version)
    assert store.list_stage4(recording_id) == []


def test_run_stage4_requires_the_recording_video(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path, with_video=False)
    stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    with pytest.raises(MissingArtifactError):
        run_stage4(store, recording_id, stage3.version)


def test_run_stage4_requires_an_existing_stage3_version(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    with pytest.raises(MissingArtifactError):
        run_stage4(store, recording_id, 7)


def test_run_stage4_keeps_every_run_in_history(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    report = CrossCheckReport(verdicts=[], refined_result=_audit())
    with patch.object(persistence_runner, "run_cross_check", return_value=report):
        run_stage4(store, recording_id, stage3.version)
        run_stage4(store, recording_id, stage3.version)
    history = store.list_stage4(recording_id)
    assert [(item.version, item.stage3_version, item.status) for item in history] == [
        (1, stage3.version, RunStatus.completed),
        (2, stage3.version, RunStatus.completed),
    ]


def test_run_stage4_failure_is_recorded_and_reraised(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    with patch.object(persistence_runner, "run_cross_check", side_effect=RuntimeError("gateway down")):
        with pytest.raises(RuntimeError):
            run_stage4(store, recording_id, stage3.version)
    [failed] = store.list_stage4(recording_id)
    assert failed.status == RunStatus.failed
    assert failed.stage3_version == stage3.version
    assert "gateway down" in failed.error


def test_reviewed_skip_pairs_only_includes_reviewed_pairs():
    assert reviewed_skip_pairs(_assist_report()) == {(1, 1)}
    empty = _assist_report().model_copy(update={"reviewed_semantic_pairs": []})
    assert reviewed_skip_pairs(empty) == set()


def test_run_stage3_creates_default_client_only_when_none_given(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    with (
        patch.object(persistence_runner, "make_openai_client", return_value=FakeGateway()) as factory,
        patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs),
    ):
        run_stage3(store, recording_id)
        run_stage3(store, recording_id, openai_client=MagicMock(wraps=FakeGateway()))
    factory.assert_called_once_with()


def _matchingtraces() -> tuple[ExpectedTrace, ObservedTrace]:
    expected = ExpectedTrace(
        scenario_title="Login",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Login", raw_text="Click Login")],
    )
    observed = ObservedTrace(
        source_file="login.mp4",
        actions=[ObservedAction(index=1, timestamp_ms=500, action=ActionType.click, target="Login", confidence=0.95)],
    )
    return expected, observed


def test_run_stage3_uses_the_selected_stage2_version(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    expected, observed = traces()
    _matching_expected, retried = _matchingtraces()
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=retried))
    evaluation = run_stage3(store, recording_id, mode=Stage3Mode.deterministic, stage2_version=1)
    assert evaluation.inputs.stage2_version == 1
    assert evaluation.inputs.stage2_sha256 == store.artifact_sha256(recording_id, STAGE2_FILENAME)
    assert evaluation.result == align(expected, observed)


def test_run_stage3_defaults_to_the_latest_stage2_version(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    expected, _observed = traces()
    _matching_expected, retried = _matchingtraces()
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=retried))
    evaluation = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert evaluation.inputs.stage2_version == 2
    assert evaluation.inputs.stage2_sha256 == store.artifact_sha256(recording_id, stage2_filename(2))
    assert evaluation.result == align(expected, retried)


def test_run_stage3_freezes_stage2c_separately_for_each_stage2_version(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    _expected, observed = traces()
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        run_stage3(store, recording_id, openai_client=FakeGateway(), stage2_version=1)
        gateway = FakeGateway()
        evaluation = run_stage3(store, recording_id, openai_client=gateway, stage2_version=2)
    assert gateway.requested.count(RawFrameEvidenceBatch) == 1
    assert evaluation.inputs.stage2c_source == Stage2CSource.live
    frozen_v2 = store.load_stage2c(recording_id, 2)
    assert (frozen_v2.stage2_version, frozen_v2.stage2_sha256) == (
        2,
        store.artifact_sha256(recording_id, stage2_filename(2)),
    )
    assert store.load_stage2c(recording_id, 1).stage2_version == 1


def test_run_stage3_refuses_a_modified_stage2_and_creates_no_evaluation(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    path = store.recording_dir(recording_id) / STAGE2_FILENAME
    path.write_text(path.read_text().replace("Splash", "Login"))
    with pytest.raises(ArtifactIntegrityError):
        run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert store.list_stage3(recording_id) == []


def test_run_stage3_refuses_a_modified_video(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    store.video_path(recording_id).write_bytes(b"replaced")
    with pytest.raises(ArtifactIntegrityError):
        run_stage3(store, recording_id, openai_client=FakeGateway())
    assert store.list_stage3(recording_id) == []


def test_run_stage3_records_inputs_without_checksums_as_unverified(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    strip_integrity(store.recording_dir(recording_id) / STAGE1_FILENAME)
    evaluation = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert evaluation.inputs.unverified_artifacts == [STAGE1_FILENAME]


def test_run_stage3_records_every_model_call_with_prompt_hash_and_response(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    gateway = FakeGateway()
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        evaluation = run_stage3(store, recording_id, openai_client=gateway)
    assert [call.response_model for call in evaluation.model_calls] == [
        model.__name__ for model in gateway.requested
    ]
    assert all(call.system_prompt_sha256 and call.response for call in evaluation.model_calls)
    stage2c_call = next(c for c in evaluation.model_calls if c.response_model == "RawFrameEvidenceBatch")
    assert stage2c_call.image_sha256 == [hashlib.sha256(b"jpeg").hexdigest()] * 3
    assert stage2c_call.response == raw_batch().model_dump(mode="json")
    assert store.load_stage3(recording_id, evaluation.version).model_calls == evaluation.model_calls


def test_run_stage3_replay_check_reports_matching_targets(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        run_stage3(store, recording_id, openai_client=FakeGateway())
    replayed = run_stage3(store, recording_id, openai_client=FakeGateway())
    assert replayed.stage2c_replay.unmatched_targets == []
    assert replayed.stage2c_replay.current_targets == replayed.stage2c_replay.frozen_targets


def test_run_stage3_replay_check_flags_targets_the_frozen_batch_never_answered(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    _expected, observed = traces()
    store.save_stage2c(recording_id, frozen_stage2c_for_other_step(observed))
    gateway = FakeGateway()
    evaluation = run_stage3(store, recording_id, openai_client=gateway)
    assert RawFrameEvidenceBatch not in gateway.requested
    assert evaluation.stage2c_replay.frozen_targets == ["E1"]
    assert "E2" in evaluation.stage2c_replay.unmatched_targets


def test_run_stage3_without_stage2c_targets_freezes_nothing(tmp_path: Path):
    store = RecordingStore(tmp_path / "recordings")
    expected, observed = _matchingtraces()
    recording_id = store.create_recording(title="Login", source_filename="login.mp4", scenario_text="x").recording_id
    store.save_stage1(recording_id, Stage1Artifact(metadata=_METADATA, expected_trace=expected))
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    store.write_video(recording_id, lambda handle: handle.write(b"video"))
    gateway = FakeGateway()
    evaluation = run_stage3(store, recording_id, openai_client=gateway)
    assert evaluation.status == RunStatus.completed
    assert RawFrameEvidenceBatch not in gateway.requested
    assert store.load_stage2c(recording_id) is None


def test_run_stage3_interruption_is_recorded_with_partial_model_calls(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    gateway = FakeGateway(stage2c_error=StopException())
    with (
        patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs),
        pytest.raises(StopException),
    ):
        run_stage3(store, recording_id, openai_client=gateway)
    [record] = store.list_stage3(recording_id)
    assert record.status == RunStatus.interrupted
    assert "StopException" in record.error
    assert len(record.model_calls) == len(gateway.requested)
    assert record.model_calls[-1].error.startswith("StopException")
    assert store.load_stage2c(recording_id) is None
    assert run_stage3(store, recording_id, mode=Stage3Mode.deterministic).version == 2


def test_run_stage3_uses_saved_evidence_stills_when_the_video_was_not_retained(tmp_path: Path):
    store = RecordingStore(tmp_path / "recordings", video_retention=VideoRetention.discard)
    expected, observed = traces()
    recording_id = store.create_recording(title="Login", source_filename="login.mp4", scenario_text="x").recording_id
    store.save_stage1(recording_id, Stage1Artifact(metadata=_METADATA, expected_trace=expected))
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    store.record_unretained_video(recording_id, upload)
    stills = store.save_evidence_frames(
        recording_id,
        "stage-2.v1",
        [(f.frame_id, f.timestamp_ms, f"still-{f.frame_id}".encode()) for f in frame_manifest_for(observed).frames],
    )
    store.save_stage2(
        recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=observed, evidence_frames=stills)
    )
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=AssertionError("video decoded")):
        evaluation = run_stage3(store, recording_id, openai_client=FakeGateway())
    stage2c_call = next(c for c in evaluation.model_calls if c.response_model == "RawFrameEvidenceBatch")
    assert stage2c_call.image_sha256 == [still.sha256 for still in stills]
    recovered = next(v for v in evaluation.result.step_verdicts if v.expected_step_index == 2)
    assert (recovered.passed, recovered.evidence_timestamp_ms) == (True, 3500)
    assert evaluation.inputs.video_sha256 is None


def test_run_stage4_uses_the_stage2_version_its_stage3_consumed(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    _expected, observed = traces()
    stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    _matching_expected, retried = _matchingtraces()
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=retried))
    report = CrossCheckReport(verdicts=[], refined_result=_audit())
    with patch.object(persistence_runner, "run_cross_check", return_value=report) as cross:
        evaluation = run_stage4(store, recording_id, stage3.version, openai_client=MagicMock())
    assert cross.call_args.args[1] == observed
    assert evaluation.stage2_version == 1
    assert evaluation.video_sha256 == store.load_manifest(recording_id).video_sha256


def test_run_stage4_refuses_a_modified_stage3_record(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    tamper_json(store.recording_dir(recording_id) / "stage3" / "0001.json")
    with (
        patch.object(persistence_runner, "run_cross_check") as cross,
        pytest.raises(ArtifactIntegrityError),
    ):
        run_stage4(store, recording_id, stage3.version, openai_client=MagicMock())
    cross.assert_not_called()
    assert store.list_stage4(recording_id) == []


def test_run_stage4_records_its_model_calls(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    report = CrossCheckReport(verdicts=[], refined_result=_audit())

    def cross_check(*_args: Any, openai_client: Any, **_kwargs: Any) -> CrossCheckReport:
        openai_client.chat.completions.create(
            model="crosscheck-model", messages=[{"role": "system", "content": "VERIFY"}]
        )
        return report

    with patch.object(persistence_runner, "run_cross_check", side_effect=cross_check):
        evaluation = run_stage4(store, recording_id, stage3.version, openai_client=MagicMock())
    [call] = evaluation.model_calls
    assert call.requested_model == "crosscheck-model"
    assert call.system_prompt_sha256 == hashlib.sha256(b"VERIFY").hexdigest()


def test_run_stage4_interruption_is_recorded(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    with (
        patch.object(persistence_runner, "run_cross_check", side_effect=StopException()),
        pytest.raises(StopException),
    ):
        run_stage4(store, recording_id, stage3.version, openai_client=MagicMock())
    [record] = store.list_stage4(recording_id)
    assert (record.status, record.stage3_version) == (RunStatus.interrupted, stage3.version)


def test_default_client_cannot_reach_the_live_gateway_in_tests():
    with pytest.raises(LiveGatewayCallError):
        persistence_runner.make_openai_client()


_SENTINEL = "example-usso-sentinel-0000"


def _replay_with_unanswered_target(tmp_path: Path) -> tuple[RecordingStore, str]:
    store, recording_id = _seeded_store(tmp_path)
    _expected, observed = traces()
    store.save_stage2c(recording_id, frozen_stage2c_for_other_step(observed))
    return store, recording_id


def _frame_proposal(evaluation: Any, step: int) -> Any:
    return next(
        p
        for p in evaluation.assist.proposals
        if p.kind == AssistCandidateKind.frame_evidence and p.expected_step_index == step
    )


def test_replay_marks_a_target_the_frozen_batch_never_answered_as_not_searched(tmp_path: Path):
    store, recording_id = _replay_with_unanswered_target(tmp_path)
    evaluation = run_stage3(store, recording_id, openai_client=FakeGateway())
    proposal = _frame_proposal(evaluation, 2)
    assert (proposal.status, proposal.reason) == (AssistProposalStatus.skipped, STAGE2C_NOT_SEARCHED_REASON)
    assert "E2" not in {c.expected_step_id for c in evaluation.assist.candidate_evidence.candidates}


def test_replay_keeps_a_searched_but_not_found_target_rejected(tmp_path: Path):
    store, recording_id = _replay_with_unanswered_target(tmp_path)
    evaluation = run_stage3(store, recording_id, openai_client=FakeGateway())
    proposal = _frame_proposal(evaluation, 1)
    assert (proposal.status, proposal.reason) == (AssistProposalStatus.rejected, "search_not_found")
    [searched] = evaluation.assist.candidate_evidence.candidates
    assert (searched.expected_step_id, searched.validation_status) == ("E1", CandidateValidationStatus.rejected)


def test_replay_not_searched_relabel_leaves_the_audit_result_unchanged(tmp_path: Path):
    store, recording_id = _replay_with_unanswered_target(tmp_path)
    relabelled = run_stage3(store, recording_id, openai_client=FakeGateway())
    with patch.object(persistence_runner, "_mark_not_searched", side_effect=lambda bundle, _targets: bundle):
        unlabelled = run_stage3(store, recording_id, openai_client=FakeGateway())
    assert relabelled.result == unlabelled.result
    assert _frame_proposal(unlabelled, 2).reason == "search_not_found"


def test_replay_not_searched_target_stays_eligible_for_stage4(tmp_path: Path):
    store, recording_id = _replay_with_unanswered_target(tmp_path)
    evaluation = run_stage3(store, recording_id, openai_client=FakeGateway())
    assert 2 not in evaluation.assist.stage2c_skip_visual_steps
    assert all(step != 2 for step, _action in reviewed_skip_pairs(evaluation.assist))


def test_replay_not_searched_relabel_keeps_proposal_metrics_consistent(tmp_path: Path):
    store, recording_id = _replay_with_unanswered_target(tmp_path)
    assist = run_stage3(store, recording_id, openai_client=FakeGateway()).assist
    rejected = sum(p.status == AssistProposalStatus.rejected for p in assist.proposals)
    assert assist.metrics.proposals_rejected == rejected


def test_live_stage2c_not_found_targets_are_not_relabelled(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        evaluation = run_stage3(store, recording_id, openai_client=FakeGateway())
    assert evaluation.stage2c_replay is None
    assert all(p.reason != STAGE2C_NOT_SEARCHED_REASON for p in evaluation.assist.proposals)


def _discarded_video_store(tmp_path: Path) -> tuple[RecordingStore, str]:
    store = RecordingStore(tmp_path / "recordings", video_retention=VideoRetention.discard)
    expected, observed = traces()
    recording_id = store.create_recording(title="Login", source_filename="login.mp4", scenario_text="x").recording_id
    store.save_stage1(recording_id, Stage1Artifact(metadata=_METADATA, expected_trace=expected))
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(b"video")
    store.record_unretained_video(recording_id, upload)
    store.save_stage2(recording_id, Stage2Artifact(metadata=_METADATA, observed_trace=observed))
    return store, recording_id


def test_run_stage3_evidence_mode_refuses_when_no_frame_source_survives(tmp_path: Path):
    store, recording_id = _discarded_video_store(tmp_path)
    gateway = FakeGateway()
    with pytest.raises(MissingArtifactError, match="not retained"):
        run_stage3(store, recording_id, openai_client=gateway)
    assert gateway.requested == []
    assert store.list_stage3(recording_id) == []


def test_run_stage3_deterministic_mode_runs_without_any_frame_source(tmp_path: Path):
    store, recording_id = _discarded_video_store(tmp_path)
    assert run_stage3(store, recording_id, mode=Stage3Mode.deterministic).status == RunStatus.completed


def test_run_stage4_refuses_a_discarded_video_and_explains_why(tmp_path: Path):
    store, recording_id = _discarded_video_store(tmp_path)
    version = run_stage3(store, recording_id, mode=Stage3Mode.deterministic).version
    with pytest.raises(MissingArtifactError, match="not retained"):
        run_stage4(store, recording_id, stage3_version=version)
    assert store.list_stage4(recording_id) == []


def test_run_stage3_tolerates_a_concurrent_run_freezing_stage2c_first(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    with (
        patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs),
        patch.object(store, "save_stage2c", side_effect=FileExistsError("frozen by another run")),
    ):
        evaluation = run_stage3(store, recording_id, openai_client=FakeGateway())
    assert evaluation.status == RunStatus.completed
    assert evaluation.inputs.stage2c_source == Stage2CSource.live


def test_run_stage3_saves_no_credentials_from_model_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USSO_TOKEN", _SENTINEL)
    store, recording_id = _seeded_store(tmp_path)
    gateway = FakeGateway(stage2c_error=PermissionError(f"gateway rejected {_SENTINEL}"))
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        evaluation = run_stage3(store, recording_id, openai_client=gateway)
    assert any(call.error for call in evaluation.model_calls)
    for path in store.recording_dir(recording_id).rglob("*.json"):
        assert _SENTINEL not in path.read_text()


def test_run_stage3_hashes_mode_saves_no_prompt_or_response_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(persistence_runner.MODEL_IO_CAPTURE_ENV, "hashes")
    store, recording_id = _seeded_store(tmp_path)
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs):
        evaluation = run_stage3(store, recording_id, openai_client=FakeGateway())
    assert evaluation.model_calls
    assert all(call.user_text is None and call.response is None for call in evaluation.model_calls)
    assert all(call.user_text_sha256 and call.response_sha256 for call in evaluation.model_calls)


def _busy_during(store: RecordingStore, recording_id: str) -> tuple[RecordingStore, Any]:
    """A store that gives up on the lock quickly, and the lock a peer holds meanwhile."""
    return RecordingStore(store.root, lock_timeout_s=0.1), store.recording_lock(recording_id)


def test_failed_final_write_preserves_the_model_calls_of_a_completed_run(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    impatient, held = _busy_during(store, recording_id)
    gateway = FakeGateway()
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs), held:
        with pytest.raises(RecordingBusyError):
            run_stage3(impatient, recording_id, openai_client=gateway)
    unsaved = Stage3Evaluation.model_validate_json(
        (store.recording_dir(recording_id) / "stage3" / "0001.unsaved.json").read_text()
    )
    assert unsaved.status == RunStatus.completed
    assert len(unsaved.model_calls) == len(gateway.requested) > 0
    assert [record.status for record in store.list_stage3(recording_id)] == [RunStatus.running]


def test_stale_recovery_copies_preserved_model_calls_into_the_interrupted_record(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    impatient, held = _busy_during(store, recording_id)
    with patch("reckon.stage2c.discover.load_frame_jpegs", side_effect=fake_jpegs), held:
        with pytest.raises(RecordingBusyError):
            run_stage3(impatient, recording_id, openai_client=FakeGateway())
    assert store.recover_stale_runs(recording_id) == [("stage3", 1)]
    [record] = store.list_stage3(recording_id)
    unsaved = Stage3Evaluation.model_validate_json(
        (store.recording_dir(recording_id) / "stage3" / "0001.unsaved.json").read_text()
    )
    assert record.status == RunStatus.interrupted
    assert record.model_calls == unsaved.model_calls
    assert "0001.unsaved.json" in record.error


def test_failed_final_write_keeps_the_original_run_error(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    impatient, held = _busy_during(store, recording_id)
    with patch.object(persistence_runner, "align_with_assist", side_effect=RuntimeError("boom")), held:
        with pytest.raises(RuntimeError, match="boom"):
            run_stage3(impatient, recording_id, openai_client=FakeGateway())
    unsaved = Stage3Evaluation.model_validate_json(
        (store.recording_dir(recording_id) / "stage3" / "0001.unsaved.json").read_text()
    )
    assert (unsaved.status, unsaved.error) == (RunStatus.failed, "RuntimeError: boom")


def test_run_stage4_refuses_a_stage2_that_changed_since_stage3_consumed_it(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    stage2 = store.recording_dir(recording_id) / STAGE2_FILENAME
    strip_integrity(stage2)
    version = run_stage3(store, recording_id, mode=Stage3Mode.deterministic).version
    stage2.write_text(stage2.read_text() + "\n")
    with pytest.raises(ArtifactIntegrityError, match="no longer matches what Stage 3"):
        run_stage4(store, recording_id, stage3_version=version)
    assert store.list_stage4(recording_id) == []


def test_run_stage3_refuses_a_frozen_stage2c_made_from_different_stage2_content(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    _expected, observed = traces()
    store.save_stage2c(
        recording_id,
        frozen_stage2c_for_other_step(observed).model_copy(update={"stage2_sha256": "0" * 64}),
    )
    gateway = FakeGateway()
    with pytest.raises(ArtifactIntegrityError, match="different Stage 2"):
        run_stage3(store, recording_id, openai_client=gateway)
    assert gateway.requested == []
    assert store.list_stage3(recording_id) == []


def _embedded(path: Path) -> str:
    return json.loads(path.read_text())["integrity"]["sha256"]


def _stage4(store: RecordingStore, recording_id: str, version: int) -> Any:
    report = CrossCheckReport(verdicts=[], refined_result=_audit())
    with patch.object(persistence_runner, "run_cross_check", return_value=report):
        return run_stage4(store, recording_id, version, openai_client=MagicMock())


def test_run_stage3_records_the_embedded_checksums_of_its_inputs(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    directory = store.recording_dir(recording_id)
    evaluation = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert evaluation.inputs.stage1_sha256 == _embedded(directory / STAGE1_FILENAME)
    assert evaluation.inputs.stage2_sha256 == _embedded(directory / STAGE2_FILENAME)
    assert evaluation.inputs.unverified_artifacts == []


def test_run_stage4_records_the_embedded_checksum_of_the_selected_stage3(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    version = run_stage3(store, recording_id, mode=Stage3Mode.deterministic).version
    stage4 = _stage4(store, recording_id, version)
    assert stage4.stage3_sha256 == _embedded(store.recording_dir(recording_id) / "stage3" / "0001.json")
    assert stage4.unverified_artifacts == []


def test_run_stage4_refuses_a_stage1_modified_after_stage3(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    version = run_stage3(store, recording_id, mode=Stage3Mode.deterministic).version
    tamper_json(store.recording_dir(recording_id) / STAGE1_FILENAME)
    with pytest.raises(ArtifactIntegrityError):
        _stage4(store, recording_id, version)
    assert store.list_stage4(recording_id) == []


def test_legacy_recording_runs_stage3_and_stage4_from_its_sidecars(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    directory = store.recording_dir(recording_id)
    make_legacy(directory)
    legacy_stage2_sha = hashlib.sha256((directory / STAGE2_FILENAME).read_bytes()).hexdigest()
    legacy_stage3 = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert (legacy_stage3.inputs.stage2_sha256, legacy_stage3.inputs.unverified_artifacts) == (legacy_stage2_sha, [])
    make_legacy(directory)
    stage4 = _stage4(store, recording_id, legacy_stage3.version)
    assert (stage4.status, stage4.unverified_artifacts) == (RunStatus.completed, [])
    assert stage4.stage3_sha256 == hashlib.sha256((directory / "stage3" / "0001.json").read_bytes()).hexdigest()
    newer = run_stage3(store, recording_id, mode=Stage3Mode.deterministic)
    assert newer.inputs.stage2_sha256 == legacy_stage2_sha
    assert _stage4(store, recording_id, newer.version).status == RunStatus.completed


def test_legacy_stage3_record_modified_after_its_sidecar_is_refused_by_stage4(tmp_path: Path):
    store, recording_id = _seeded_store(tmp_path)
    make_legacy(store.recording_dir(recording_id))
    version = run_stage3(store, recording_id, mode=Stage3Mode.deterministic).version
    make_legacy(store.recording_dir(recording_id))
    record = store.recording_dir(recording_id) / "stage3" / "0001.json"
    record.write_text(record.read_text().replace('"deterministic"', '"evidence_aware"', 1))
    with pytest.raises(ArtifactIntegrityError):
        _stage4(store, recording_id, version)
