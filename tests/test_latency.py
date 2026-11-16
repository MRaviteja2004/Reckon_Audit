"""Focused tests for optional pipeline latency instrumentation."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from reckon.crosscheck import run_cross_check
from reckon.extractor import extract_window
from reckon.latency import (
    LatencyMetrics,
    bind_retry_callback,
    elapsed_ms,
    latency_scope,
    log_metrics,
    monotonic_ms,
    resolve_latency,
)
from reckon.sampler import extract_observed_trace
from reckon.schema import (
    ActionType,
    AuditResult,
    AuditVerdict,
    ExpectedStep,
    ExpectedTrace,
    Finding,
    FindingSeverity,
    FindingType,
    ObservedAction,
    ObservedTrace,
    StepVerdict,
)

_VIDEO = Path("recording.mp4")
_SOURCE = "recording.mp4"


def _chunk(clip_offset_ms: int, clip_end_ms: int, actions: list[ObservedAction]) -> ObservedTrace:
    return ObservedTrace(
        source_file=_SOURCE,
        clip_offset_ms=clip_offset_ms,
        clip_end_ms=clip_end_ms,
        actions=actions,
    )


def _observed(index: int = 1, timestamp_ms: int = 0) -> ObservedAction:
    return ObservedAction(
        index=index,
        timestamp_ms=timestamp_ms,
        action=ActionType.click,
        target="Submit",
        confidence=0.9,
    )


def test_bind_retry_callback_without_metrics_preserves_identity():
    def on_retry(attempt: int, code: int, delay_s: float) -> None:
        return None

    assert bind_retry_callback(None, on_retry) is on_retry
    assert bind_retry_callback(None, None) is None


def test_bind_retry_callback_records_and_forwards():
    metrics = LatencyMetrics()
    seen: list[tuple[int, int, float]] = []

    def on_retry(attempt: int, code: int, delay_s: float) -> None:
        seen.append((attempt, code, delay_s))

    wrapped = bind_retry_callback(metrics, on_retry)
    assert wrapped is not on_retry
    wrapped(1, 429, 12.5)
    assert metrics.retry_count == 1
    assert metrics.retry_sleep_s == 12.5
    assert seen == [(1, 429, 12.5)]


def test_elapsed_ms_uses_monotonic_clock():
    start = monotonic_ms()
    assert elapsed_ms(start) >= 0.0


def test_extract_observed_trace_records_stage2_timings():
    metrics = LatencyMetrics()
    with (
        patch("reckon.sampler.detect_change_events", return_value=[5000, 60_000]),
        patch("reckon.sampler.video_duration_ms", return_value=90_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.side_effect = [
            _chunk(3000, 8000, [_observed()]),
            _chunk(58_000, 63_000, [_observed()]),
        ]
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE, latency=metrics)

    assert metrics.stage2a_ms is not None and metrics.stage2a_ms >= 0.0
    assert metrics.stage2b_ms is not None and metrics.stage2b_ms >= 0.0
    assert metrics.window_count == 2
    assert mock_extract.call_count == 2
    assert mock_extract.call_args_list[0].kwargs["latency"] is metrics
    assert mock_extract.call_args_list[0].kwargs["window_index"] == 0
    assert mock_extract.call_args_list[1].kwargs["window_index"] == 1


def test_extract_observed_trace_picks_up_latency_scope_without_kwarg():
    """Regression: callers must not need latency= (Streamlit can cache an older signature)."""
    metrics = LatencyMetrics()
    with (
        patch("reckon.sampler.detect_change_events", return_value=[5000]),
        patch("reckon.sampler.video_duration_ms", return_value=20_000),
        patch("reckon.sampler.extract_window") as mock_extract,
        latency_scope(metrics),
    ):
        mock_extract.return_value = _chunk(3000, 8000, [_observed()])
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE)

    assert metrics.stage2a_ms is not None
    assert metrics.window_count == 1
    assert mock_extract.call_args.kwargs["latency"] is metrics


def test_resolve_latency_prefers_explicit_over_scope():
    scoped = LatencyMetrics()
    explicit = LatencyMetrics()
    with latency_scope(scoped):
        assert resolve_latency(explicit) is explicit
        assert resolve_latency(None) is scoped
    assert resolve_latency(None) is None


def test_extract_observed_trace_without_latency_forwards_on_retry_identity():
    def on_retry(attempt: int, code: int, delay_s: float) -> None:
        return None

    with (
        patch("reckon.sampler.detect_change_events", return_value=[5000]),
        patch("reckon.sampler.video_duration_ms", return_value=20_000),
        patch("reckon.sampler.extract_window") as mock_extract,
    ):
        mock_extract.return_value = _chunk(3000, 8000, [])
        extract_observed_trace(MagicMock(), _VIDEO, _SOURCE, on_retry=on_retry)
    assert mock_extract.call_args.kwargs["on_retry"] is on_retry


def test_extract_window_records_frame_and_payload_metrics():
    metrics = LatencyMetrics()
    samples = [
        (0, Image.new("RGB", (8, 8), color=(10, 20, 30))),
        (100, Image.new("RGB", (8, 8), color=(40, 50, 60))),
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(actions=[])

    with patch("reckon.extractor.extract_frames_at", return_value=samples):
        extract_window(
            client,
            _VIDEO,
            _SOURCE,
            [0, 100],
            latency=metrics,
            window_index=3,
        )

    assert len(metrics.windows) == 1
    window = metrics.windows[0]
    assert window.index == 3
    assert window.frame_count == 2
    assert window.image_payload_bytes > 0
    assert window.duration_ms >= 0.0
    assert metrics.frame_count == 2


def test_log_metrics_emits_numeric_fields_only(caplog):
    metrics = LatencyMetrics(
        total_ms=1000.0,
        stage1_ms=10.0,
        stage2a_ms=20.0,
        stage2b_ms=30.0,
        stage3_ms=5.0,
        stage4_semantic_ms=8.0,
        stage4_visual_ms=12.0,
        window_count=2,
        frame_count=6,
        retry_count=1,
        retry_sleep_s=12.0,
        stage4_semantic_candidate_count=1,
        stage4_visual_target_count=2,
    )
    with caplog.at_level(logging.INFO, logger="reckon.latency"):
        log_metrics(metrics, where="test")
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "latency[test]" in text
    assert "stage2b_ms=30.0" in text
    assert "USSO" not in text
    assert "Bearer" not in text
    assert "data:image" not in text


def test_run_cross_check_records_stage4_counts_and_durations():
    expected = ExpectedTrace(
        scenario_title="Login",
        steps=[
            ExpectedStep(
                index=1,
                action=ActionType.assert_visible,
                target="error",
                assertion="Invalid credentials",
                raw_text="Assert Invalid credentials",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file=_SOURCE,
        actions=[
            ObservedAction(
                index=1,
                timestamp_ms=1000,
                action=ActionType.click,
                target="Login",
                confidence=0.5,
            ),
        ],
    )
    audit = AuditResult(
        verdict=AuditVerdict.unfaithful,
        fidelity_score=0.0,
        findings=[
            Finding(
                type=FindingType.assertion_not_observed,
                severity=FindingSeverity.critical,
                description="missing assertion",
                expected_step_index=1,
                observed_action_index=1,
                evidence_timestamp_ms=1000,
            ),
        ],
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
    metrics = LatencyMetrics()
    client = MagicMock()

    with (
        patch("reckon.crosscheck.select_semantic_candidates", return_value=[]),
        patch("reckon.crosscheck.verify_target") as mock_verify,
    ):
        from reckon.schema import CrossCheckAnswer, CrossCheckVerdict

        mock_verify.return_value = CrossCheckVerdict(
            question="q",
            answer=CrossCheckAnswer.absent,
            confidence=0.9,
            rationale="not shown",
            frame_timestamps_ms=[500, 1000, 1500],
            expected_step_index=1,
            observed_action_index=1,
        )
        run_cross_check(
            expected,
            observed,
            audit,
            _VIDEO,
            openai_client=client,
            latency=metrics,
        )

    assert metrics.stage4_semantic_candidate_count == 0
    assert metrics.stage4_semantic_ms is not None and metrics.stage4_semantic_ms >= 0.0
    assert metrics.stage4_visual_target_count == 1
    assert metrics.stage4_visual_ms is not None and metrics.stage4_visual_ms >= 0.0
    assert mock_verify.call_args.kwargs["on_retry"] is not None
