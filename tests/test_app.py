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
    AssistCandidateKind,
    AssistMatchMethod,
    AssistMetrics,
    AssistProposalRecord,
    AssistProposalStatus,
    AssistReport,
    AssistReviewedPair,
    AssistedAlignResult,
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


def _assist_report(*, accepted: bool = True) -> AssistReport:
    proposals = []
    reviewed = []
    if accepted:
        proposals.append(
            AssistProposalRecord(
                kind=AssistCandidateKind.wrong_target,
                status=AssistProposalStatus.accepted,
                expected_step_index=1,
                matched_observed_index=1,
                finding_types=[FindingType.wrong_target],
                model_answer="equivalent",
                confidence=0.91,
                rationale="same control",
                matching_method=AssistMatchMethod.llm_assisted,
            )
        )
        reviewed.append(
            AssistReviewedPair(expected_step_index=1, observed_action_index=1)
        )
    else:
        proposals.append(
            AssistProposalRecord(
                kind=AssistCandidateKind.wrong_target,
                status=AssistProposalStatus.rejected,
                expected_step_index=1,
                matched_observed_index=1,
                finding_types=[FindingType.wrong_target],
                model_answer="different",
                confidence=0.8,
                rationale="distinct controls",
                reason="model_different",
                matching_method=AssistMatchMethod.rejected_proposal,
            )
        )
    return AssistReport(
        model="claude-opus-5-5",
        proposals=proposals,
        metrics=AssistMetrics(
            expected_event_count=1,
            observed_event_count=1,
            deterministic_match_count=0,
            candidate_count=1,
            llm_request_count=1,
            proposals_accepted=1 if accepted else 0,
            proposals_rejected=0 if accepted else 1,
            unresolved_count=0,
            deterministic_ms=12.0,
            llm_ms=34.0,
            total_ms=50.0,
        ),
        reviewed_semantic_pairs=reviewed,
    )


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
        patch.object(app, "align", return_value=initial) as align_mock,
        patch.object(app, "align_with_assist") as assist_mock,
        patch.object(app, "run_cross_check", return_value=report) as cross_mock,
    ):
        result = app._run_pipeline(
            b"video",
            "recording.mp4",
            "Click Submit",
            run_crosscheck=True,
            stage3_mode=app.STAGE3_MODE_DETERMINISTIC,
        )

    assert result is not None
    assert result[0] is refined
    align_mock.assert_called_once_with(expected, observed)
    assist_mock.assert_not_called()
    cross_kwargs = cross_mock.call_args.kwargs
    assert cross_kwargs.get("skip_semantic_pairs") is None
    assert streamlit.session_state["debug"]["crosscheck"] is not None
    assert streamlit.session_state["semantic_diagnostics"] == [diagnostic]
    assert streamlit.session_state["stage3_assist"] is None
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
    assert export["stage3_mode"] == app.STAGE3_MODE_DETERMINISTIC
    assert export["stage3_assist"] is None


def test_pipeline_evidence_aware_mode_passes_evidence_aware_flag():
    expected = ExpectedTrace(
        scenario_title="test",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Sign in")],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1500,
                action=ActionType.click,
                target="Login",
                confidence=0.9,
            ),
        ],
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[],
        step_verdicts=[StepVerdict(expected_step_index=1, passed=False)],
    )
    assist = _assist_report(accepted=False)
    bundle = AssistedAlignResult(
        deterministic_result=deterministic,
        result=deterministic,
        assist=assist,
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
        patch.object(app, "align_with_assist", return_value=bundle) as assist_mock,
        patch.object(app, "run_cross_check", return_value=CrossCheckReport(
            verdicts=[], refined_result=deterministic
        )),
    ):
        result = app._run_pipeline(
            b"video",
            "recording.mp4",
            "Click Sign in",
            run_crosscheck=True,
            stage3_mode=app.STAGE3_MODE_EVIDENCE,
        )

    assert result is not None
    assert assist_mock.call_args.kwargs.get("evidence_aware") is True
    assert streamlit.session_state["export"]["stage3_mode"] == app.STAGE3_MODE_EVIDENCE


def test_pipeline_assisted_mode_calls_align_with_assist_and_skips_accepted_pairs():
    expected = ExpectedTrace(
        scenario_title="test",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Sign in")],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1500,
                action=ActionType.click,
                target="Login",
                confidence=0.9,
            ),
        ],
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=1500,
                description="Target mismatch",
            )
        ],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=False,
                matched_observed_index=1,
                evidence_timestamp_ms=1500,
                finding_types=[FindingType.wrong_target],
            ),
        ],
    )
    assisted = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[
            StepVerdict(
                expected_step_index=1,
                passed=True,
                matched_observed_index=1,
                evidence_timestamp_ms=1500,
            ),
        ],
    )
    assist = _assist_report(accepted=True)
    bundle = AssistedAlignResult(
        deterministic_result=deterministic,
        result=assisted,
        assist=assist,
    )
    refined = assisted
    report = CrossCheckReport(verdicts=[], refined_result=refined)

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
        patch.object(app, "align") as align_mock,
        patch.object(app, "align_with_assist", return_value=bundle) as assist_mock,
        patch.object(app, "run_cross_check", return_value=report) as cross_mock,
    ):
        result = app._run_pipeline(
            b"video",
            "recording.mp4",
            "Click Sign in",
            run_crosscheck=True,
            stage3_mode=app.STAGE3_MODE_ASSISTED,
        )

    assert result is not None
    assert result[0] is refined
    align_mock.assert_not_called()
    assist_mock.assert_called_once()
    assist_kwargs = assist_mock.call_args.kwargs
    assert assist_kwargs["model"] == app.ALIGN_ASSIST_MODEL
    assert assist_kwargs.get("evidence_aware") is False
    assert assist_mock.call_args.args[0] is expected
    assert assist_mock.call_args.args[1] is observed
    skip_pairs = cross_mock.call_args.kwargs.get("skip_semantic_pairs")
    assert skip_pairs == {(1, 1)}
    assert streamlit.session_state["stage3_assist"] is assist
    assert streamlit.session_state["deterministic_stage3"] is deterministic
    assert streamlit.session_state["assisted_stage3"] is assisted
    export = streamlit.session_state["export"]
    assert export["stage3_mode"] == app.STAGE3_MODE_ASSISTED
    assert export["stage3_assist"]["assist"]["model"] == "claude-opus-5-5"
    assert export["stage3_assist"]["deterministic_audit_result"]["verdict"] == "unfaithful"
    assert export["stage3_assist"]["assisted_audit_result"]["verdict"] == "faithful"
    assert export["original_audit_result"]["verdict"] == "faithful"


def test_pipeline_assisted_rejected_pairs_are_not_skipped_in_stage4():
    expected = ExpectedTrace(
        scenario_title="test",
        steps=[ExpectedStep(index=1, action=ActionType.click, target="Sign in")],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1500,
                action=ActionType.click,
                target="Cancel",
                confidence=0.9,
            ),
        ],
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[],
        step_verdicts=[StepVerdict(expected_step_index=1, passed=False)],
    )
    assist = _assist_report(accepted=False)
    bundle = AssistedAlignResult(
        deterministic_result=deterministic,
        result=deterministic,
        assist=assist,
    )
    report = CrossCheckReport(verdicts=[], refined_result=deterministic)

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
        patch.object(app, "align_with_assist", return_value=bundle),
        patch.object(app, "run_cross_check", return_value=report) as cross_mock,
    ):
        result = app._run_pipeline(
            b"video",
            "recording.mp4",
            "Click Sign in",
            run_crosscheck=True,
            stage3_mode=app.STAGE3_MODE_ASSISTED,
        )

    assert result is not None
    assert cross_mock.call_args.kwargs.get("skip_semantic_pairs") == set()


def test_reviewed_skip_pairs_only_includes_accepted():
    assist = _assist_report(accepted=True)
    assert app._reviewed_skip_pairs(assist) == {(1, 1)}
    rejected = _assist_report(accepted=False)
    assert app._reviewed_skip_pairs(rejected) == set()


def test_render_stage3_assist_panel_shows_comparison_and_metrics():
    assist = _assist_report(accepted=True)
    assist.proposals.append(
        AssistProposalRecord(
            kind=AssistCandidateKind.delayed_assertion,
            status=AssistProposalStatus.accepted,
            expected_step_index=2,
            matched_observed_index=2,
            supporting_observed_index=3,
            finding_types=[FindingType.assertion_not_observed],
            matching_method=AssistMatchMethod.deterministic,
        )
    )
    assist.proposals.append(
        AssistProposalRecord(
            kind=AssistCandidateKind.wrong_target,
            status=AssistProposalStatus.rejected,
            expected_step_index=3,
            matched_observed_index=4,
            finding_types=[FindingType.wrong_target],
            model_answer="unresolved",
            reason="model_unresolved",
            matching_method=AssistMatchMethod.unresolved,
        )
    )
    deterministic = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.wrong_target,
                severity=FindingSeverity.critical,
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=1500,
                description="mismatch",
            )
        ],
        step_verdicts=[],
    )
    assisted = AuditResult(
        verdict=AuditVerdict.faithful,
        fidelity_score=1.0,
        findings=[],
        step_verdicts=[],
    )
    observed = ObservedTrace(
        source_file="recording.mp4",
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1500,
                action=ActionType.click,
                target="Login",
                confidence=1.0,
            ),
            ObservedAction(
                index=3,
                timestamp_ms=3200,
                action=ActionType.assert_visible,
                target="Done",
                confidence=1.0,
            ),
        ],
    )
    streamlit = MagicMock()
    streamlit.columns.side_effect = lambda n, **_k: [MagicMock() for _ in range(n)]

    with patch.object(app, "st", streamlit):
        app._render_stage3_assist_panel(
            assist=assist,
            deterministic=deterministic,
            assisted=assisted,
            token_usage={
                "request_count": 1,
                "usage_available_count": 1,
                "prompt_tokens_sum": 10,
                "completion_tokens_sum": 4,
                "total_tokens_sum": 14,
                "requests": [],
            },
            observed=observed,
        )

    streamlit.subheader.assert_called_once_with("Stage 3 — LLM Assist (experimental)")
    markdowns = [call.args[0] for call in streamlit.markdown.call_args_list]
    assert "**Resolved by deterministic delayed evidence**" in markdowns
    assert "**Resolved by Claude (validated)**" in markdowns
    assert "**Rejected or unresolved proposals**" in markdowns
    assert "**Deterministic Stage 3**" in markdowns
    assert "**Assisted Stage 3 (pre–Stage 4)**" in markdowns
    writes = [str(call.args[0]) for call in streamlit.write.call_args_list if call.args]
    assert any("O3 @ 3.200s" in text for text in writes)
    captions = [str(call.args[0]) for call in streamlit.caption.call_args_list if call.args]
    assert any("Token usage" in text for text in captions)


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
