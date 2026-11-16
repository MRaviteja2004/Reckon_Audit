import importlib.util
import json
import re
import tomllib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from reckon.latency import LatencyMetrics, WindowLatency
from reckon.schema import (
    ActionType,
    AssertionEvidence,
    AuditResult,
    AuditVerdict,
    CrossCheckAnswer,
    CrossCheckReport,
    CrossCheckVerdict,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
    ObservedTrace,
    SemanticDiagnostic,
    StepVerdict,
)

_ROOT = Path(__file__).parents[1]
_APP_PATH = _ROOT / "app.py"
_CONFIG_PATH = _ROOT / ".streamlit" / "config.toml"
_APP_SPEC = importlib.util.spec_from_file_location("reckon_streamlit_app", _APP_PATH)
assert _APP_SPEC is not None and _APP_SPEC.loader is not None
app = importlib.util.module_from_spec(_APP_SPEC)
_APP_SPEC.loader.exec_module(app)


def _export_models() -> tuple[
    ExpectedTrace,
    ObservedTrace,
    AuditResult,
    AuditResult,
    CrossCheckReport,
]:
    expected = ExpectedTrace(
        scenario_title="TC 29",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.click,
                target="Submit",
                input_value=None,
                assertion="Confirmation is visible",
                preconditions="Form is complete",
                raw_text="Click Submit and verify confirmation",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        clip_offset_ms=500,
        clip_end_ms=2000,
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Submit",
                input_value=None,
                assertion_evidence=AssertionEvidence(
                    description="Confirmation panel is visible",
                    visible_text="Saved",
                ),
                confidence=0.95,
                screen_description="Completed form with confirmation panel",
            ),
        ],
    )
    finding = Finding(
        type=FindingType.assertion_not_observed,
        severity=FindingSeverity.critical,
        expected_step_index=1,
        observed_action_index=1,
        evidence_timestamp_ms=1000,
        description="Confirmation was not established by the initial alignment",
    )
    original = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[finding],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
                finding_types=[FindingType.assertion_not_observed],
            ),
        ],
    )
    refined = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
            ),
        ],
    )
    report = CrossCheckReport(
        verdicts=[
            CrossCheckVerdict(
                question="Is the confirmation visible?",
                answer=CrossCheckAnswer.present,
                confidence=0.97,
                rationale="The confirmation panel is visible.",
                frame_timestamps_ms=[900, 1000, 1100],
                expected_step_index=1,
                observed_action_index=1,
            ),
        ],
        refined_result=refined,
        semantic_diagnostics=[
            SemanticDiagnostic(
                expected_step_index=1,
                observed_action_index=1,
                comparison_kind="assertion",
                answer="equivalent",
                confidence=0.92,
                applied=True,
                error_type=None,
            ),
        ],
    )
    return expected, observed, original, refined, report


def test_streamlit_config_matches_app_upload_limit():
    config = tomllib.loads(_CONFIG_PATH.read_text())
    assert config["server"]["maxUploadSize"] == app.MAX_UPLOAD_SIZE_MB
    assert config["server"]["maxMessageSize"] == app.MAX_UPLOAD_SIZE_MB


def test_write_recording_to_temp_from_bytes():
    destination = BytesIO()
    app._write_recording_to_temp(b"abc123", destination)
    assert destination.getvalue() == b"abc123"


def test_write_recording_to_temp_from_file_like():
    source = BytesIO(b"chunked-upload")
    source.seek(4)
    destination = BytesIO()
    app._write_recording_to_temp(source, destination)
    assert destination.getvalue() == b"chunked-upload"


def test_pipeline_uses_semantic_only_refined_result():
    expected = ExpectedTrace(
        scenario_title="test",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Submit")],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Submit",
                confidence=0.95,
            ),
        ],
    )
    initial = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[],
        step_verdicts=[
            StepVerdict(expected_step_index=1, passed=False),
        ],
    )
    refined = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
            ),
        ],
    )
    diagnostic = SemanticDiagnostic(
        expected_step_index=1,
        observed_action_index=1,
        comparison_kind="target",
        answer="equivalent",
        confidence=0.95,
        applied=True,
    )
    report = CrossCheckReport(
        verdicts=[],
        refined_result=refined,
        semantic_diagnostics=[diagnostic],
    )

    streamlit = MagicMock()
    streamlit.session_state = {}
    status = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.return_value = status
    status_context.__exit__.return_value = False
    streamlit.status.return_value = status_context

    with (
        patch.object(app, "st", streamlit),
        patch.object(app, "make_openai_client", return_value=MagicMock()),
        patch.object(app, "parse_scenario", return_value=expected),
        patch.object(app, "make_client", return_value=MagicMock()),
        patch.object(app, "extract_observed_trace", return_value=observed),
        patch.object(app, "align", return_value=initial),
        patch.object(app, "run_cross_check", return_value=report),
    ):
        result = app._run_pipeline(
            b"video",
            "recording.mp4",
            "Click Submit",
            run_crosscheck=True,
        )

    assert result is not None
    assert result[0] is refined
    assert streamlit.session_state["debug"]["crosscheck"] is not None
    assert streamlit.session_state["semantic_diagnostics"] == [diagnostic]
    completion_messages = [
        call.args[0]
        for call in streamlit.write.call_args_list
        if call.args and str(call.args[0]).startswith("Stage 4 complete")
    ]
    assert completion_messages == [
        "Stage 4 complete — 1 semantic diagnostic(s), 0 visual verdict(s); "
        "refined fidelity score: 100%"
    ]
    export = streamlit.session_state["export"]
    assert export["original_audit_result"]["verdict"] == "unfaithful"
    assert export["cross_check"]["refined_audit_result"]["verdict"] == "faithful"


def test_stage4_completion_message_includes_score_when_no_checks_run():
    report = CrossCheckReport(
        verdicts=[],
        refined_result=AuditResult(
            verdict=AuditVerdict.unfaithful,
            fidelity_score=1 / 3,
            findings=[],
            step_verdicts=[],
        ),
    )

    assert app._stage4_completion_message(report) == (
        "Stage 4 complete — nothing required a second opinion; "
        "refined fidelity score: 33%"
    )


def test_render_semantic_diagnostics_shows_applied_result():
    diagnostic = SemanticDiagnostic(
        expected_step_index=2,
        observed_action_index=3,
        comparison_kind="target",
        answer="equivalent",
        confidence=0.95,
        applied=True,
    )
    streamlit = MagicMock()

    with patch.object(app, "st", streamlit):
        app._render_semantic_diagnostics([diagnostic])

    streamlit.subheader.assert_called_once_with("Semantic Cross-checks  (1)")
    streamlit.expander.assert_called_once_with(
        "Step 2  —  target  —  applied",
        expanded=False,
    )


def test_pipeline_accepts_file_like_recording():
    expected = ExpectedTrace(
        scenario_title="test",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Submit")],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Submit",
                confidence=0.95,
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
            ),
        ],
    )

    streamlit = MagicMock()
    streamlit.session_state = {}
    status = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.return_value = status
    status_context.__exit__.return_value = False
    streamlit.status.return_value = status_context
    written: list[Path] = []

    def _capture_extract(_client, video_path, source_file, on_retry=None, **_kwargs):
        written.append(Path(video_path))
        assert video_path.read_bytes() == b"file-like-video"
        return observed

    with (
        patch.object(app, "st", streamlit),
        patch.object(app, "make_openai_client", return_value=MagicMock()),
        patch.object(app, "parse_scenario", return_value=expected),
        patch.object(app, "make_client", return_value=MagicMock()),
        patch.object(app, "extract_observed_trace", side_effect=_capture_extract),
        patch.object(app, "align", return_value=audit),
    ):
        result = app._run_pipeline(
            BytesIO(b"file-like-video"),
            "recording.mp4",
            "Click Submit",
            run_crosscheck=False,
        )

    assert result is not None
    assert result[0] is audit
    assert written
    assert not written[0].exists()


def test_pipeline_does_not_pass_latency_kwarg_to_extract_observed_trace():
    """Regression: Stage 2 failed when app passed latency= to a signature without it.

    Streamlit can reload app.py while keeping an older extract_observed_trace in
    sys.modules. The pipeline must not pass latency=; metrics use latency_scope.
    """
    expected = ExpectedTrace(
        scenario_title="test",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Submit")],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Submit",
                confidence=0.95,
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1000,
            ),
        ],
    )

    streamlit = MagicMock()
    streamlit.session_state = {}
    status = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.return_value = status
    status_context.__exit__.return_value = False
    streamlit.status.return_value = status_context
    calls: list[dict] = []

    def extract_without_latency_kwarg(
        _client, video_path, source_file, *, on_retry=None
    ):
        # Intentionally no latency parameter — matches the pre-instrumentation signature.
        calls.append({"source_file": source_file, "on_retry": on_retry})
        return observed

    with (
        patch.object(app, "st", streamlit),
        patch.object(app, "make_openai_client", return_value=MagicMock()),
        patch.object(app, "parse_scenario", return_value=expected),
        patch.object(app, "make_client", return_value=MagicMock()),
        patch.object(app, "extract_observed_trace", side_effect=extract_without_latency_kwarg),
        patch.object(app, "align", return_value=audit),
    ):
        result = app._run_pipeline(
            b"video-bytes",
            "recording.mp4",
            "Click Submit",
            run_crosscheck=False,
        )

    assert result is not None
    assert result[0] is audit
    assert len(calls) == 1
    assert streamlit.session_state.get("debug", {}).get("errors", []) == []
    assert "latency" in streamlit.session_state


def test_format_duration_ms_seconds_and_minutes():
    assert app._format_duration_ms(None) == "—"
    assert app._format_duration_ms(1500) == "1.5 s"
    assert app._format_duration_ms(125_000) == "2 m 5.0 s (125.0 s)"
    assert app._format_duration_ms("bad") == "—"


def test_format_bytes_for_payload_display():
    assert app._format_bytes(None) == "—"
    assert app._format_bytes(512) == "512 B"
    assert app._format_bytes(2048) == "2.0 KB"


def test_latency_session_payload_includes_summary_and_windows():
    metrics = LatencyMetrics(
        total_ms=10_000.0,
        stage1_ms=100.0,
        stage2a_ms=200.0,
        stage2b_ms=300.0,
        window_count=1,
        frame_count=2,
        windows=[
            WindowLatency(
                index=0,
                duration_ms=250.0,
                frame_count=2,
                image_payload_bytes=4096,
            ),
        ],
    )
    payload = app._latency_session_payload(metrics)
    assert payload["total_ms"] == 10_000.0
    assert payload["stage1_ms"] == 100.0
    assert payload["windows"] == [
        {
            "index": 0,
            "duration_ms": 250.0,
            "frame_count": 2,
            "image_payload_bytes": 4096,
        },
    ]


def test_latency_export_payload_preserves_retry_sleep_precision():
    metrics = LatencyMetrics(retry_sleep_s=1.23456789)

    assert app._latency_session_payload(metrics)["retry_sleep_s"] == 1.235
    assert app._latency_export_payload(metrics)["retry_sleep_s"] == 1.23456789


def test_export_payload_includes_complete_result():
    expected, observed, original, refined, report = _export_models()
    metrics = LatencyMetrics(
        total_ms=12_500.125,
        retry_sleep_s=1.23456789,
        windows=[
            WindowLatency(
                index=0,
                duration_ms=5000.25,
                frame_count=3,
                image_payload_bytes=4096,
            ),
        ],
    )
    payload = app._build_export_payload(
        scenario_title=expected.scenario_title,
        recording_filename=observed.source_file,
        original_audit_result=original,
        expected_trace=expected,
        observed_trace=observed,
        crosscheck_report=report,
        latency_metrics=app._latency_export_payload(metrics),
    )

    assert payload["scenario_title"] == "TC 29"
    assert payload["recording_filename"] == "recording.mp4"
    assert payload["expected_trace"] == expected.model_dump(mode="json")
    assert payload["observed_trace"] == observed.model_dump(mode="json")
    assert payload["original_audit_result"] == original.model_dump(mode="json")
    assert payload["cross_check"] == {
        "refined_audit_result": refined.model_dump(mode="json"),
        "verdicts": [
            verdict.model_dump(mode="json") for verdict in report.verdicts
        ],
        "semantic_diagnostics": [
            diagnostic.model_dump(mode="json")
            for diagnostic in report.semantic_diagnostics
        ],
    }
    assert payload["latency_metrics"] == app._latency_export_payload(metrics)
    assert payload["latency_metrics"]["retry_sleep_s"] == 1.23456789


def test_export_payload_handles_missing_crosscheck():
    expected, observed, original, _refined, _report = _export_models()
    payload = app._build_export_payload(
        scenario_title=expected.scenario_title,
        recording_filename=observed.source_file,
        original_audit_result=original,
        expected_trace=expected,
        observed_trace=observed,
        crosscheck_report=None,
        latency_metrics={"total_ms": 12_500.0},
    )

    assert payload["cross_check"] is None
    assert payload["original_audit_result"]["verdict"] == "unfaithful"


def test_export_payload_handles_missing_latency():
    expected, observed, original, _refined, report = _export_models()
    payload = app._build_export_payload(
        scenario_title=expected.scenario_title,
        recording_filename=observed.source_file,
        original_audit_result=original,
        expected_trace=expected,
        observed_trace=observed,
        crosscheck_report=report,
        latency_metrics=None,
    )

    assert payload["latency_metrics"] is None
    assert payload["observed_trace"] is not None


def test_export_payload_handles_missing_traces():
    expected, observed, original, _refined, report = _export_models()
    payload = app._build_export_payload(
        scenario_title=expected.scenario_title,
        recording_filename=observed.source_file,
        original_audit_result=original,
        expected_trace=None,
        observed_trace=None,
        crosscheck_report=report,
        latency_metrics=None,
    )

    assert payload["expected_trace"] is None
    assert payload["observed_trace"] is None


def test_export_payload_handles_empty_crosscheck_diagnostics():
    expected, observed, original, refined, _report = _export_models()
    report = CrossCheckReport(verdicts=[], refined_result=refined)
    payload = app._build_export_payload(
        scenario_title=expected.scenario_title,
        recording_filename=observed.source_file,
        original_audit_result=original,
        expected_trace=expected,
        observed_trace=observed,
        crosscheck_report=report,
        latency_metrics=None,
    )

    assert payload["cross_check"]["verdicts"] == []
    assert payload["cross_check"]["semantic_diagnostics"] == []


def test_render_export_download_uses_valid_json_and_useful_filename():
    expected, observed, original, _refined, report = _export_models()
    payload = app._build_export_payload(
        scenario_title=expected.scenario_title,
        recording_filename=observed.source_file,
        original_audit_result=original,
        expected_trace=expected,
        observed_trace=observed,
        crosscheck_report=report,
        latency_metrics=None,
    )
    streamlit = MagicMock()
    streamlit.session_state = {"export": payload}

    with patch.object(app, "st", streamlit):
        app._render_export_download()

    streamlit.download_button.assert_called_once()
    call = streamlit.download_button.call_args
    assert call.args[0] == "Save Results (JSON)"
    downloaded = json.loads(call.kwargs["data"])
    assert downloaded == payload
    assert type(downloaded["scenario_title"]) is str
    assert type(downloaded["original_audit_result"]["fidelity_score"]) is float
    assert type(downloaded["observed_trace"]["actions"][0]["timestamp_ms"]) is int
    assert type(downloaded["original_audit_result"]["step_verdicts"][0]["passed"]) is bool
    assert downloaded["expected_trace"]["steps"][0]["input_value"] is None
    assert type(downloaded["expected_trace"]["steps"]) is list
    assert type(downloaded["cross_check"]) is dict
    assert call.kwargs["file_name"] == "reckon_audit_TC_29_recording.json"
    assert call.kwargs["mime"] == "application/json"


def test_export_filename_distinguishes_recordings_for_same_scenario():
    first = app._export_filename("TC 29", "recording_one.mp4")
    second = app._export_filename("TC 29", "recording_two.mp4")

    assert first == "reckon_audit_TC_29_recording_one.json"
    assert second == "reckon_audit_TC_29_recording_two.json"
    assert first != second


def test_export_filename_sanitizes_paths_and_unsupported_characters():
    filename = app._export_filename("../TC:29?", "../../recordings/run #1.mp4")

    assert filename == "reckon_audit_TC_29_run_1.json"
    assert "/" not in filename
    assert ":" not in filename
    assert "?" not in filename
    assert "#" not in filename


def test_export_filename_preserves_meaningful_unicode():
    assert (
        app._export_filename("场景 九", "录制 一.mp4")
        == "reckon_audit_场景_九_录制_一.json"
    )


def test_export_filename_uses_stable_fallback_for_blank_input():
    assert app._export_filename("  ", "  ") == "reckon_audit_result.json"


def test_export_filename_limits_excessively_long_input():
    filename = app._export_filename("A" * 300, f"{'B' * 300}.mp4")
    base = filename.removeprefix("reckon_audit_").removesuffix(".json")

    assert len(filename) == app.MAX_EXPORT_FILENAME_LENGTH
    assert "A" in base
    assert "B" in base


def test_render_latency_skips_when_metrics_missing():
    streamlit = MagicMock()
    with patch.object(app, "st", streamlit):
        app._render_latency(None)
        app._render_latency({})
    streamlit.subheader.assert_not_called()


def test_render_latency_shows_stages_and_window_table():
    streamlit = MagicMock()
    column_mocks: list[MagicMock] = []

    def _columns(count, **_kwargs):
        cols = [MagicMock(name=f"col{i}") for i in range(count)]
        column_mocks.extend(cols)
        return cols

    streamlit.columns.side_effect = _columns
    metrics = {
        "total_ms": 125_000.0,
        "stage1_ms": 1000.0,
        "stage2a_ms": 2000.0,
        "stage2b_ms": 90_000.0,
        "stage3_ms": 50.0,
        "stage4_semantic_ms": 500.0,
        "stage4_visual_ms": 1500.0,
        "window_count": 1,
        "frame_count": 3,
        "retry_count": 2,
        "retry_sleep_s": 12.5,
        "windows": [
            {
                "index": 0,
                "duration_ms": 45_000.0,
                "frame_count": 3,
                "image_payload_bytes": 8192,
            },
        ],
    }
    with patch.object(app, "st", streamlit):
        app._render_latency(metrics)

    streamlit.subheader.assert_called_once_with("Evaluation Performance")
    metric_labels = [call.args[0] for call in streamlit.metric.call_args_list]
    for col in column_mocks:
        metric_labels.extend(call.args[0] for call in col.metric.call_args_list)
    assert "Total evaluation time" in metric_labels
    assert "Stage 2a — Change detect" in metric_labels
    assert "Stage 4 — Visual" in metric_labels
    assert "Retry sleep" in metric_labels
    total_call = streamlit.metric.call_args_list[0]
    assert total_call.args[1] == "2 m 5.0 s (125.0 s)"
    streamlit.dataframe.assert_called_once()
    table = streamlit.dataframe.call_args.args[0]
    assert table[0]["Window"] == 0
    assert table[0]["JPEG payload"] == "8.0 KB"


def test_help_text_does_not_mention_gemini_file_api():
    source = _APP_PATH.read_text()
    assert "Gemini File API" not in source
    assert re.search(r"OpenAI-compatible gateway", source)
