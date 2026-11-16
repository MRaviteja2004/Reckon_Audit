"""Streamlit demo UI for the Recording Fidelity POC.

Run with:
    streamlit run app.py

Stage 1 (parse scenario) uses USSO_TOKEN through Uber AI Gateway.
Stage 2 (extract trace)  uses the same gateway; frames are decoded locally.
Stage 3 (align traces)   is deterministic — no API key required.
"""

import json
import logging
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import BinaryIO, IO

import streamlit as st
from dotenv import load_dotenv

# load_dotenv must run before reckon imports because extractor.py reads
# EXTRACTION_MODEL at module level via os.environ.get.
load_dotenv()

from reckon.aligner import align  # noqa: E402
from reckon.crosscheck import run_cross_check  # noqa: E402
from reckon.extractor import make_client  # noqa: E402
from reckon.latency import (  # noqa: E402
    LatencyMetrics,
    elapsed_ms,
    latency_scope,
    log_metrics,
    monotonic_ms,
)
from reckon.parser import make_openai_client, parse_scenario  # noqa: E402
from reckon.sampler import extract_observed_trace  # noqa: E402
from reckon.schema import (  # noqa: E402
    AuditResult,
    AuditVerdict,
    CrossCheckReport,
    CrossCheckVerdict,
    ExpectedTrace,
    FindingSeverity,
    FindingType,
    ObservedTrace,
    SemanticDiagnostic,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Keep in sync with server.maxUploadSize / maxMessageSize in .streamlit/config.toml.
# This is the Streamlit demo UI ceiling only; host/proxy limits are separate.
MAX_UPLOAD_SIZE_MB = 500
MAX_EXPORT_FILENAME_LENGTH = 120

_FINDING_LABELS: dict[FindingType, str] = {
    FindingType.missing_action: "Missing Action",
    FindingType.extra_action: "Extra Action",
    FindingType.wrong_target: "Wrong Target",
    FindingType.wrong_input_value: "Wrong Input Value",
    FindingType.assertion_not_observed: "Assertion Not Observed",
}

# Severity is not decided here. It comes from the aligner on each finding, so the UI
# cannot drift from the policy the verdict is computed with — a wrong input value is
# critical because a refund off by a factor of ten is critical, and a badge reading
# "Medium" beside it would be actively misleading.
_SEVERITY_LABEL: dict[FindingSeverity, str] = {
    FindingSeverity.critical: "Critical",
    FindingSeverity.advisory: "Advisory",
}

_SCENARIO_PLACEHOLDER = (
    "Example:\n"
    "1. Navigate to the login page\n"
    "2. Enter 'user@example.invalid' in the Email field\n"
    "3. Enter 'wrong-password' in the Password field\n"
    "4. Click the Login button\n"
    "5. Assert that an 'Invalid credentials' error message is shown"
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ms_to_timestamp(ms: int | None) -> str:
    """Convert a millisecond offset to a human-readable timestamp string."""
    if ms is None:
        return "—"
    total_s, remainder_ms = divmod(ms, 1000)
    minutes, seconds = divmod(total_s, 60)
    if minutes:
        return f"{minutes}:{seconds:02d}.{remainder_ms:03d}"
    return f"{seconds}.{remainder_ms:03d}s"


def _format_duration_ms(ms: float | int | None) -> str:
    """Format a wall-clock duration in milliseconds as seconds and minutes."""
    if ms is None:
        return "—"
    try:
        seconds = float(ms) / 1000.0
    except (TypeError, ValueError):
        return "—"
    if seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes} m {remainder:.1f} s ({seconds:.1f} s)"


def _format_bytes(size: float | int | None) -> str:
    """Format a byte count for display without exposing contents."""
    if size is None:
        return "—"
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "—"
    if value < 1024:
        return f"{int(value)} B"
    return f"{value / 1024.0:.1f} KB"


def _latency_session_payload(latency: LatencyMetrics) -> dict:
    """Build the session-state payload for UI display (numeric fields + windows)."""
    return {
        **latency.summary_fields(),
        "windows": [
            {
                "index": window.index,
                "duration_ms": window.duration_ms,
                "frame_count": window.frame_count,
                "image_payload_bytes": window.image_payload_bytes,
            }
            for window in latency.windows
        ],
    }


def _latency_export_payload(latency: LatencyMetrics) -> dict:
    """Build export metrics while preserving raw collector precision."""
    payload = _latency_session_payload(latency)
    payload["retry_sleep_s"] = latency.retry_sleep_s
    return payload


def _build_export_payload(
    *,
    scenario_title: str,
    recording_filename: str,
    original_audit_result: AuditResult,
    expected_trace: ExpectedTrace | None,
    observed_trace: ObservedTrace | None,
    crosscheck_report: CrossCheckReport | None,
    latency_metrics: dict | None,
) -> dict[str, object]:
    """Build a JSON-safe export from results already held in memory."""
    cross_check = None
    if crosscheck_report is not None:
        cross_check = {
            "refined_audit_result": crosscheck_report.refined_result.model_dump(mode="json"),
            "verdicts": [
                verdict.model_dump(mode="json") for verdict in crosscheck_report.verdicts
            ],
            "semantic_diagnostics": [
                diagnostic.model_dump(mode="json")
                for diagnostic in crosscheck_report.semantic_diagnostics
            ],
        }

    return {
        "scenario_title": scenario_title,
        "recording_filename": recording_filename,
        "original_audit_result": original_audit_result.model_dump(mode="json"),
        "cross_check": cross_check,
        "expected_trace": (
            expected_trace.model_dump(mode="json") if expected_trace is not None else None
        ),
        "observed_trace": (
            observed_trace.model_dump(mode="json") if observed_trace is not None else None
        ),
        "latency_metrics": latency_metrics,
    }


def _export_filename(scenario_title: str, recording_filename: str) -> str:
    """Return a filesystem-safe, useful name for an audit export."""
    prefix = "reckon_audit_"
    suffix = ".json"
    max_base_length = MAX_EXPORT_FILENAME_LENGTH - len(prefix) - len(suffix)

    def _sanitize(value: str) -> str:
        return re.sub(r"[^\w-]+", "_", value.strip()).strip("_-")

    title = _sanitize(scenario_title)
    recording_stem = _sanitize(Path(recording_filename.strip()).stem)
    components = [component for component in (title, recording_stem) if component]
    if not components:
        return f"{prefix}result{suffix}"

    if len(components) == 2 and len("_".join(components)) > max_base_length:
        component_length = (max_base_length - 1) // 2
        components = [
            components[0][:component_length].rstrip("_-"),
            components[1][:(max_base_length - component_length - 1)].rstrip("_-"),
        ]

    base = "_".join(components)[:max_base_length].rstrip("_-")
    return f"{prefix}{base or 'result'}{suffix}"


def _render_export_download() -> None:
    """Render a download for the completed in-memory audit, when available."""
    payload = st.session_state.get("export")
    if not payload:
        return
    st.download_button(
        "Save Results (JSON)",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=_export_filename(
            str(payload.get("scenario_title") or ""),
            str(payload.get("recording_filename") or ""),
        ),
        mime="application/json",
    )


def _stage4_completion_message(report: CrossCheckReport) -> str:
    """Summarize both Stage 4 paths and always include the refined score."""
    semantic_count = len(report.semantic_diagnostics)
    visual_count = len(report.verdicts)
    score = report.refined_result.fidelity_score
    if semantic_count or visual_count:
        return (
            f"Stage 4 complete — {semantic_count} semantic diagnostic(s), "
            f"{visual_count} visual verdict(s); refined fidelity score: {score:.0%}"
        )
    return (
        "Stage 4 complete — nothing required a second opinion; "
        f"refined fidelity score: {score:.0%}"
    )


def _severity_badge(severity: str) -> str:
    """Return an inline HTML badge for the given severity label."""
    palette: dict[str, tuple[str, str]] = {
        "Critical": ("#7f1d1d", "#fee2e2"),
        "High":     ("#7c2d12", "#ffedd5"),
        "Medium":   ("#78350f", "#fef3c7"),
        "Low":      ("#1e3a5f", "#dbeafe"),
    }
    fg, bg = palette.get(severity, ("#374151", "#f3f4f6"))
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'padding:1px 8px;border-radius:2px;font-size:11px;font-weight:700;'
        f'letter-spacing:0.06em;font-family:monospace">{severity.upper()}</span>'
    )


def _verdict_badge(passed: bool) -> str:
    if passed:
        return (
            '<span style="display:inline-block;background:#dcfce7;color:#14532d;'
            'padding:1px 8px;border-radius:2px;font-size:11px;font-weight:700;'
            'letter-spacing:0.06em;font-family:monospace">PASS</span>'
        )
    return (
        '<span style="display:inline-block;background:#fee2e2;color:#7f1d1d;'
        'padding:1px 8px;border-radius:2px;font-size:11px;font-weight:700;'
        'letter-spacing:0.06em;font-family:monospace">FAIL</span>'
    )


# ── UI sections ───────────────────────────────────────────────────────────────


def _render_findings(audit_result: AuditResult) -> None:
    findings = audit_result.findings
    if not findings:
        st.success("No findings — the recording faithfully demonstrates the expected scenario.")
        return

    st.subheader(f"Findings  ({len(findings)})")

    sorted_findings = sorted(
        findings,
        key=lambda f: 0 if f.severity == FindingSeverity.critical else 1,
    )

    for finding in sorted_findings:
        label = _FINDING_LABELS.get(finding.type, finding.type.value)
        severity = _SEVERITY_LABEL[finding.severity]
        ts = _ms_to_timestamp(finding.evidence_timestamp_ms)

        step_part = (
            f"  —  Step {finding.expected_step_index}"
            if finding.expected_step_index is not None
            else ""
        )
        ts_part = f"  —  {ts}" if finding.evidence_timestamp_ms is not None else ""
        header = f"{label}{step_part}{ts_part}"

        auto_expand = finding.severity == FindingSeverity.critical
        with st.expander(header, expanded=auto_expand):
            st.markdown(
                _severity_badge(severity) + f"&nbsp;&nbsp;`{finding.type.value}`",
                unsafe_allow_html=True,
            )
            st.write(finding.description)
            meta_cols = st.columns(3)
            meta_cols[0].caption(f"Severity: {severity}")
            if finding.expected_step_index is not None:
                meta_cols[1].caption(f"Expected step: {finding.expected_step_index}")
            if finding.observed_action_index is not None:
                meta_cols[2].caption(f"Observed action: {finding.observed_action_index}")
            if finding.evidence_timestamp_ms is not None:
                st.caption(f"Video timestamp: {ts}")


def _render_step_verdicts(audit_result: AuditResult, expected_trace: ExpectedTrace) -> None:
    st.subheader("Step-by-step Verdicts")

    verdict_by_step = {v.expected_step_index: v for v in audit_result.step_verdicts}

    for step in expected_trace.steps:
        verdict = verdict_by_step.get(step.index)
        passed = verdict is not None and verdict.passed

        ts_str = _ms_to_timestamp(verdict.evidence_timestamp_ms if verdict else None)
        target_suffix = f"  —  {step.target}" if step.target else ""
        status_text = "PASS" if passed else "FAIL"
        header = (
            f"Step {step.index}"
            f"  —  {step.action.value}{target_suffix}"
            f"  —  {status_text}"
            f"  —  {ts_str}"
        )

        with st.expander(header, expanded=False):
            st.markdown(
                _verdict_badge(passed),
                unsafe_allow_html=True,
            )
            if step.raw_text:
                st.caption(step.raw_text)

            detail_cols = st.columns(2)
            detail_cols[0].markdown(f"**Action:** `{step.action.value}`")
            if step.target:
                detail_cols[1].markdown(f"**Target:** `{step.target}`")
            if step.input_value:
                st.markdown(f"**Input value:** `{step.input_value}`")
            if step.assertion:
                st.markdown(f"**Assertion:** {step.assertion}")

            if verdict is None:
                st.error("No verdict recorded — step treated as missing.")
            elif not verdict.passed:
                issues = "  ·  ".join(
                    _FINDING_LABELS.get(ft, ft.value) for ft in verdict.finding_types
                )
                st.warning(f"Issues: {issues}")
            else:
                st.success("Step passed")


def _render_crosscheck(verdicts: list[CrossCheckVerdict]) -> None:
    answer_styles: dict[str, str] = {
        "present": "✅ present",
        "absent": "❌ absent",
        "unclear": "❔ unclear",
    }
    st.subheader(f"Cross-check Verdicts  ({len(verdicts)})")
    st.caption(
        "A second vision model re-examined these on targeted frames. A confident "
        "'present' overturns an assertion the first pass could not confirm."
    )
    for verdict in verdicts:
        answer = answer_styles.get(verdict.answer.value, verdict.answer.value)
        frames = ", ".join(_ms_to_timestamp(t) for t in verdict.frame_timestamps_ms)
        step_part = (
            f"Step {verdict.expected_step_index}"
            if verdict.expected_step_index is not None
            else f"Observed action {verdict.observed_action_index}"
        )
        header = f"{answer}  —  {step_part}  —  confidence {verdict.confidence:.0%}"
        with st.expander(header, expanded=False):
            st.write(verdict.rationale)
            st.caption(f"Question: {verdict.question}")
            if frames:
                st.caption(f"Frames examined: {frames}")


def _render_semantic_diagnostics(
    diagnostics: list[SemanticDiagnostic],
) -> None:
    """Render text-only Stage 4 comparisons without changing visual session data."""
    st.subheader(f"Semantic Cross-checks  ({len(diagnostics)})")
    st.caption(
        "Text comparisons can resolve advisory target wording. Assertion comparisons "
        "remain diagnostic until direct visual evidence confirms them."
    )
    for diagnostic in diagnostics:
        if diagnostic.error_type:
            status = f"error: {diagnostic.error_type}"
        elif diagnostic.applied:
            status = "applied"
        else:
            status = "diagnostic only"
        header = (
            f"Step {diagnostic.expected_step_index}  —  "
            f"{diagnostic.comparison_kind}  —  {status}"
        )
        with st.expander(header, expanded=False):
            st.write(f"Answer: {diagnostic.answer or 'unavailable'}")
            if diagnostic.confidence is not None:
                st.caption(f"Confidence: {diagnostic.confidence:.0%}")
            st.caption(f"Observed action: {diagnostic.observed_action_index}")


def _render_latency(metrics: dict | None) -> None:
    """Render evaluation timing from session-state latency metrics, if present."""
    if not metrics:
        return

    st.subheader("Evaluation Performance")
    st.caption("Wall-clock timings from this run. Counts and durations only.")

    total_label = _format_duration_ms(metrics.get("total_ms"))
    st.metric("Total evaluation time", total_label)

    stage_cols = st.columns(3)
    stage_cols[0].metric("Stage 1 — Parse", _format_duration_ms(metrics.get("stage1_ms")))
    stage_cols[1].metric("Stage 2a — Change detect", _format_duration_ms(metrics.get("stage2a_ms")))
    stage_cols[2].metric("Stage 2b — Extract", _format_duration_ms(metrics.get("stage2b_ms")))

    stage_cols2 = st.columns(3)
    stage_cols2[0].metric("Stage 3 — Align", _format_duration_ms(metrics.get("stage3_ms")))
    stage_cols2[1].metric(
        "Stage 4 — Semantic", _format_duration_ms(metrics.get("stage4_semantic_ms"))
    )
    stage_cols2[2].metric(
        "Stage 4 — Visual", _format_duration_ms(metrics.get("stage4_visual_ms"))
    )

    count_cols = st.columns(4)
    count_cols[0].metric("Windows", metrics.get("window_count", 0))
    count_cols[1].metric("Frames extracted", metrics.get("frame_count", 0))
    count_cols[2].metric("Retries", metrics.get("retry_count", 0))
    retry_sleep = metrics.get("retry_sleep_s")
    if retry_sleep is None:
        retry_sleep_label = "—"
    else:
        try:
            retry_sleep_label = f"{float(retry_sleep):.1f} s"
        except (TypeError, ValueError):
            retry_sleep_label = "—"
    count_cols[3].metric("Retry sleep", retry_sleep_label)

    windows = metrics.get("windows") or []
    if windows:
        st.markdown("**Per-window extraction**")
        table_rows = [
            {
                "Window": window.get("index", "—"),
                "Duration": _format_duration_ms(window.get("duration_ms")),
                "Frames": window.get("frame_count", "—"),
                "JPEG payload": _format_bytes(window.get("image_payload_bytes")),
            }
            for window in windows
            if isinstance(window, dict)
        ]
        if table_rows:
            st.dataframe(table_rows, use_container_width=True, hide_index=True)


def _render_results(audit_result: AuditResult, expected_trace: ExpectedTrace) -> None:
    st.divider()
    st.header("Audit Results")
    _render_export_download()

    score = audit_result.fidelity_score
    total = len(expected_trace.steps)
    passed_count = sum(1 for v in audit_result.step_verdicts if v.passed)
    finding_count = len(audit_result.findings)
    critical = audit_result.critical_findings
    advisory_count = finding_count - len(critical)

    # The verdict, not the score, is the headline. Measured on a 90-second synthetic
    # flow, a faithful recording's score overlapped the range of deliberately broken
    # ones, while the critical findings separated them every time.
    if audit_result.verdict == AuditVerdict.faithful and finding_count == 0:
        st.success("FAITHFUL — the recording demonstrates every step of the scenario.")
    elif audit_result.verdict == AuditVerdict.faithful:
        st.success(
            f"FAITHFUL — every claim in the scenario was demonstrated. "
            f"{advisory_count} advisory note(s) on wording or grain, none affecting fidelity."
        )
    else:
        st.error(
            f"UNFAITHFUL — {len(critical)} critical finding(s): the recording does not "
            "demonstrate what the scenario claims."
        )

    # Metrics row
    verdict_col, critical_col, steps_col = st.columns(3)
    verdict_col.metric("Verdict", audit_result.verdict.value.title())
    critical_col.metric("Critical Findings", len(critical))
    steps_col.metric(
        "Steps Passed", f"{passed_count} / {total}",
        help=(
            f"Fidelity {score:.0%}. Secondary detail — it moves with how finely the model "
            "happened to describe the recording, so it is not the verdict."
        ),
    )

    # Finding-type summary line, critical types first.
    if finding_count:
        type_counts: dict[FindingType, int] = {}
        worst: dict[FindingType, FindingSeverity] = {}
        for f in audit_result.findings:
            type_counts[f.type] = type_counts.get(f.type, 0) + 1
            if f.severity == FindingSeverity.critical:
                worst[f.type] = FindingSeverity.critical
        summary_parts = [
            f"{_FINDING_LABELS[ft]}: {c}"
            for ft, c in sorted(
                type_counts.items(),
                key=lambda x: 0 if worst.get(x[0]) == FindingSeverity.critical else 1,
            )
        ]
        st.caption("  ·  ".join(summary_parts))

    st.divider()
    _render_findings(audit_result)

    semantic_diagnostics = st.session_state.get("semantic_diagnostics")
    crosscheck_verdicts = st.session_state.get("crosscheck")
    if semantic_diagnostics or crosscheck_verdicts:
        st.divider()
        if semantic_diagnostics:
            _render_semantic_diagnostics(semantic_diagnostics)
        if crosscheck_verdicts:
            _render_crosscheck(crosscheck_verdicts)

    st.divider()
    _render_step_verdicts(audit_result, expected_trace)

    latency_metrics = st.session_state.get("latency")
    if latency_metrics:
        st.divider()
        _render_latency(latency_metrics)


# ── Debug panel ───────────────────────────────────────────────────────────────


def _render_debug_panel() -> None:
    """Render a collapsible panel with raw JSON output from each pipeline stage.

    Only shown after at least one pipeline run. Useful for diagnosing issues:
    share the contents of this panel to get a full picture of what each stage
    produced or where it failed.
    """
    debug = st.session_state.get("debug")
    if not debug:
        return

    st.divider()
    with st.expander("Pipeline Debug Output", expanded=False):
        st.caption(
            "Raw output from each pipeline stage. Copy and share this to diagnose issues."
        )

        if debug.get("stage1"):
            st.markdown("**Stage 1 — Expected Trace (parsed scenario)**")
            st.code(debug["stage1"], language="json")
        else:
            st.markdown("**Stage 1 — Expected Trace:** not available")

        if debug.get("stage2"):
            st.markdown("**Stage 2 — Observed Trace (extracted from video)**")
            st.code(debug["stage2"], language="json")
        else:
            st.markdown("**Stage 2 — Observed Trace:** not available")

        if debug.get("stage3"):
            st.markdown("**Stage 3 — Audit Result**")
            st.code(debug["stage3"], language="json")
        else:
            st.markdown("**Stage 3 — Audit Result:** not available")

        if debug.get("crosscheck"):
            st.markdown("**Stage 4 — Cross-check Report (verdicts + refined result)**")
            st.code(debug["crosscheck"], language="json")

        for err in debug.get("errors", []):
            st.markdown(f"**{err['stage']} — Full Traceback**")
            st.code(err["tb"], language="python")


# ── Pipeline ──────────────────────────────────────────────────────────────────


def _write_recording_to_temp(video: BinaryIO | bytes, destination: IO[bytes]) -> None:
    """Persist an uploaded recording to an open temp file without an extra full copy.

    bytes inputs (used by tests) are written directly. File-like inputs are copied
    in chunks via shutil.copyfileobj so callers can pass a Streamlit UploadedFile
    without calling getbuffer().tobytes() first.
    """
    if isinstance(video, (bytes, bytearray, memoryview)):
        destination.write(video)
        return
    video.seek(0)
    shutil.copyfileobj(video, destination)


def _run_pipeline(
    video: BinaryIO | bytes,
    video_name: str,
    scenario_text: str,
    run_crosscheck: bool,
) -> tuple[AuditResult, ExpectedTrace] | None:
    """Run Stage 1 → Stage 2 → Stage 3 (→ optional cross-check) with a live status widget.

    Returns (AuditResult, ExpectedTrace) on success, or None on failure
    (error messages are shown via st.error before returning). When run_crosscheck is
    set, the cross-check verdicts are stored in st.session_state["crosscheck"] and the
    returned AuditResult is the refined one.
    Intermediate results and full tracebacks are written to st.session_state["debug"]
    so the debug panel can display them regardless of whether the run succeeded.
    """
    debug: dict = {"stage1": None, "stage2": None, "stage3": None, "crosscheck": None, "errors": []}
    st.session_state["debug"] = debug
    st.session_state["crosscheck"] = None
    st.session_state["semantic_diagnostics"] = None
    st.session_state["export"] = None
    latency = LatencyMetrics()
    pipeline_started = monotonic_ms()

    with st.status("Running audit pipeline…", expanded=True) as status:
        # ── Stage 1: parse scenario ───────────────────────────────────────────
        st.write("Stage 1 — Parsing expected scenario")
        try:
            openai_client = make_openai_client()
            stage1_started = monotonic_ms()
            expected_trace = parse_scenario(scenario_text, openai_client=openai_client)
            latency.stage1_ms = elapsed_ms(stage1_started)
            logger.info("stage1 duration_ms=%.1f step_count=%d", latency.stage1_ms, len(expected_trace.steps))
            debug["stage1"] = expected_trace.model_dump_json(indent=2)
        except Exception as exc:
            debug["errors"].append({"stage": "Stage 1", "error": str(exc), "tb": traceback.format_exc()})
            status.update(label="Stage 1 failed", state="error")
            st.error(f"Stage 1 failed: {exc}")
            latency.total_ms = elapsed_ms(pipeline_started)
            log_metrics(latency, where="app")
            return None
        st.write(
            f"Stage 1 complete — {len(expected_trace.steps)} step(s) parsed: "
            f"{expected_trace.scenario_title}"
        )

        # ── Stage 2: extract observed trace ──────────────────────────────────
        # The temp video is kept until the end of the run: the optional cross-check
        # pass reads frames from it locally. It is removed in the outer finally.
        st.write("Stage 2 — Detecting screen changes in the recording")
        tmp_path: Path | None = None
        try:
            suffix = Path(video_name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                _write_recording_to_temp(video, tmp)
                tmp_path = Path(tmp.name)

            st.write("Stage 2 — Extracting observed trace from change-event frames")

            def _on_extraction_retry(attempt: int, code: int, delay_s: float) -> None:
                st.write(
                    f"⏳ Extraction model is busy (HTTP {code}) — retry {attempt} in {delay_s:.0f}s…"
                )

            # Propagate metrics via latency_scope rather than a latency= kwarg so a
            # stale cached extract_observed_trace (common under Streamlit reload)
            # cannot raise TypeError for an unexpected keyword.
            with latency_scope(latency):
                observed_trace = extract_observed_trace(
                    make_client(),
                    tmp_path,
                    source_file=video_name,
                    on_retry=_on_extraction_retry,
                )
            debug["stage2"] = observed_trace.model_dump_json(indent=2)

            st.write(
                f"Stage 2 complete — {len(observed_trace.actions)} observed action(s) extracted"
            )

            # ── Stage 3: align traces ─────────────────────────────────────────
            st.write("Stage 3 — Aligning expected and observed traces")
            stage3_started = monotonic_ms()
            original_audit_result = align(expected_trace, observed_trace)
            audit_result = original_audit_result
            latency.stage3_ms = elapsed_ms(stage3_started)
            logger.info(
                "stage3 duration_ms=%.1f finding_count=%d",
                latency.stage3_ms,
                len(audit_result.findings),
            )
            debug["stage3"] = audit_result.model_dump_json(indent=2)
            st.write(
                f"Stage 3 complete — fidelity score: {audit_result.fidelity_score:.0%}, "
                f"{len(audit_result.findings)} finding(s)"
            )

            # ── Stage 4 (optional): cross-check pass ──────────────────────────
            crosscheck_report: CrossCheckReport | None = None
            if run_crosscheck:
                st.write("Stage 4 — Cross-checking assertions and low-confidence steps")
                with latency_scope(latency):
                    crosscheck_report = run_cross_check(
                        expected_trace,
                        observed_trace,
                        audit_result,
                        tmp_path,
                    )
                debug["crosscheck"] = crosscheck_report.model_dump_json(indent=2)
                audit_result = crosscheck_report.refined_result
                if crosscheck_report.semantic_diagnostics:
                    st.session_state["semantic_diagnostics"] = (
                        crosscheck_report.semantic_diagnostics
                    )
                if crosscheck_report.verdicts:
                    st.session_state["crosscheck"] = crosscheck_report.verdicts
                st.write(_stage4_completion_message(crosscheck_report))
        except Exception as exc:
            stage = "Stage 2/3/4"
            debug["errors"].append({"stage": stage, "error": str(exc), "tb": traceback.format_exc()})
            status.update(label=f"{stage} failed", state="error")
            st.error(f"{stage} failed: {exc}")
            latency.total_ms = elapsed_ms(pipeline_started)
            log_metrics(latency, where="app")
            return None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        latency.total_ms = elapsed_ms(pipeline_started)
        log_metrics(latency, where="app")
        latency_payload = _latency_session_payload(latency)
        st.session_state["latency"] = latency_payload
        st.session_state["export"] = _build_export_payload(
            scenario_title=expected_trace.scenario_title,
            recording_filename=video_name,
            original_audit_result=original_audit_result,
            expected_trace=expected_trace,
            observed_trace=observed_trace,
            crosscheck_report=crosscheck_report,
            latency_metrics=_latency_export_payload(latency),
        )
        status.update(label="Audit complete", state="complete", expanded=False)

    return audit_result, expected_trace


# ── App entry point ───────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Recording Fidelity Audit",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Suppress the Streamlit hamburger menu and footer in demo contexts.
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Recording Fidelity Audit")
    st.caption(
        "Verify that a UI test recording faithfully demonstrates its expected scenario. "
        "Stage 1 parses the scenario into a structured trace; Stage 2 extracts what "
        "actually happened in the video; Stage 3 aligns the two traces deterministically."
    )

    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────────────
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("Test Recording")
        uploaded_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "webm"],
            max_upload_size=MAX_UPLOAD_SIZE_MB,
            help=(
                "MP4 recommended. The recording is written to a temporary local file; "
                "only extracted JPEG frames are sent through the OpenAI-compatible gateway."
            ),
            label_visibility="collapsed",
        )
        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"{uploaded_file.name}  ·  {size_mb:.1f} MB")

    with right_col:
        st.subheader("Expected Scenario")
        scenario_text = st.text_area(
            "Paste the expected scenario",
            height=220,
            placeholder=_SCENARIO_PLACEHOLDER,
            label_visibility="collapsed",
            help="Free text, numbered steps, or Gherkin — all three formats are supported.",
        )
        if scenario_text.strip():
            line_count = len([line for line in scenario_text.splitlines() if line.strip()])
            st.caption(f"{line_count} non-empty line(s)")

    run_crosscheck = st.checkbox(
        "Run cross-check pass (second model)",
        value=True,
        help=(
            "After alignment, route unconfirmed assertions and low-confidence steps to a "
            "second vision model on targeted frames. A confident second opinion can overturn "
            "an assertion the first pass missed. On by default: Stage 2 is not told the "
            "scenario, so it cannot know which on-screen evidence an assertion cares about, "
            "and without this pass faithful recordings draw false assertion findings."
        ),
    )

    run_disabled = not (uploaded_file and scenario_text.strip())
    run_clicked = st.button(
        "Run Audit",
        type="primary",
        disabled=run_disabled,
        use_container_width=False,
    )

    # Persist results across rerenders so that expanding a finding or verdict
    # does not discard the current audit output.
    if "result" not in st.session_state:
        st.session_state["result"] = None

    if run_clicked and uploaded_file is not None and scenario_text.strip():
        # Clear any previous result before starting a fresh run.
        st.session_state["result"] = None
        st.session_state["latency"] = None
        st.session_state["export"] = None
        outcome = _run_pipeline(
            video=uploaded_file,
            video_name=uploaded_file.name,
            scenario_text=scenario_text,
            run_crosscheck=run_crosscheck,
        )
        if outcome is not None:
            st.session_state["result"] = outcome

    if st.session_state.get("result") is not None:
        audit_result, expected_trace = st.session_state["result"]
        _render_results(audit_result, expected_trace)
    elif not run_clicked:
        if not uploaded_file:
            st.info("Upload a video recording to get started.")
        elif not scenario_text.strip():
            st.info("Paste the expected scenario to get started.")

    _render_debug_panel()


if __name__ == "__main__":
    main()
