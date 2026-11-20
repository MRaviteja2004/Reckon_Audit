"""Streamlit demo UI for the Recording Fidelity POC.

Run with:
    streamlit run app.py

Stage 1 (parse scenario) uses USSO_TOKEN through Uber AI Gateway.
Stage 2 (extract trace)  uses the same gateway; frames are decoded locally.
Stage 3 (align traces)   is deterministic by default; an optional experimental
evidence-aware agent review (with Stage 2C frame evidence) is available in the UI only.

Every run is persisted under RECKON_RECORDINGS_DIR (default ``recordings/``): one
folder per recording with stage-1, versioned stage-2/2c artifacts and versioned
Stage 3 / Stage 4 evaluations, so Stage 2, Stage 3 and Stage 4 can be re-run
independently from saved data.
"""

import json
import logging
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, BinaryIO, IO

import streamlit as st
from dotenv import load_dotenv

# load_dotenv must run before reckon imports because extractor.py reads
# EXTRACTION_MODEL at module level via os.environ.get.
load_dotenv()

from reckon.assist_propose import ALIGN_ASSIST_MODEL  # noqa: E402
from reckon.latency import (  # noqa: E402
    LatencyMetrics,
    elapsed_ms,
    log_metrics,
    monotonic_ms,
)
from reckon.parser import make_openai_client, parse_scenario  # noqa: E402
from reckon.persistence import (  # noqa: E402
    STAGE1_FILENAME,
    ArtifactIntegrityError,
    MissingArtifactError,
    RecordingBusyError,
    RecordingManifest,
    RecordingStore,
    RunStatus,
    Stage3Evaluation,
    Stage3Mode,
    Stage4Evaluation,
    VideoRetention,
    build_stage1_artifact,
    run_stage2,
    run_stage3,
    run_stage4,
    stage2c_filename,
    utc_now,
)
from reckon.schema import (  # noqa: E402
    AssistMatchMethod,
    AssistProposalStatus,
    AssistReport,
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

STAGE3_MODE_DETERMINISTIC = "Deterministic (default)"
STAGE3_MODE_EVIDENCE = "LLM-assisted evidence-aware (experimental)"
_STAGE3_MODES: dict[str, Stage3Mode] = {
    STAGE3_MODE_DETERMINISTIC: Stage3Mode.deterministic,
    STAGE3_MODE_EVIDENCE: Stage3Mode.evidence_aware,
}
_STAGE3_MODE_LABELS: dict[Stage3Mode, str] = {
    mode: label for label, mode in _STAGE3_MODES.items()
}

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
    stage3_mode: str = STAGE3_MODE_DETERMINISTIC,
    deterministic_stage3: AuditResult | None = None,
    assisted_stage3: AuditResult | None = None,
    stage3_assist: AssistReport | None = None,
    stage3_token_usage: dict | None = None,
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

    stage3_assist_payload = None
    if stage3_assist is not None:
        stage3_assist_payload = {
            "mode": stage3_mode,
            "assist": stage3_assist.model_dump(mode="json"),
            "deterministic_audit_result": (
                deterministic_stage3.model_dump(mode="json")
                if deterministic_stage3 is not None
                else None
            ),
            "assisted_audit_result": (
                assisted_stage3.model_dump(mode="json")
                if assisted_stage3 is not None
                else None
            ),
            "token_usage": stage3_token_usage,
        }

    return {
        "scenario_title": scenario_title,
        "recording_filename": recording_filename,
        "stage3_mode": stage3_mode,
        "original_audit_result": original_audit_result.model_dump(mode="json"),
        "cross_check": cross_check,
        "stage3_assist": stage3_assist_payload,
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


def _observation_timestamp_label(
    observed: ObservedTrace | None,
    index: int | None,
) -> str:
    """Format an observation id with its decoder timestamp when available."""
    if index is None:
        return "—"
    if observed is None:
        return f"O{index}"
    for action in observed.actions:
        if action.index == index:
            return f"O{index} @ {_ms_to_timestamp(action.timestamp_ms)}"
    return f"O{index}"


def _render_stage3_assist_panel(
    *,
    assist: AssistReport,
    deterministic: AuditResult,
    assisted: AuditResult,
    token_usage: dict[str, Any] | None,
    observed: ObservedTrace | None = None,
) -> None:
    """Render Stage 3 assist diagnostics and a deterministic comparison."""
    st.subheader("Stage 3 — LLM Assist (experimental)")
    st.caption(
        f"Model: `{assist.model}`. Claude proposes; Python validates before apply. "
        "Only accepted resolutions are excluded from Stage 4 semantic review."
    )
    if assist.error:
        st.warning(f"Assist reported an error and fell back where needed: `{assist.error}`")

    metric_cols = st.columns(4)
    metric_cols[0].metric("LLM requests", assist.metrics.llm_request_count)
    metric_cols[1].metric("Accepted", assist.metrics.proposals_accepted)
    metric_cols[2].metric("Rejected / failed", assist.metrics.proposals_rejected)
    metric_cols[3].metric("Assist total", _format_duration_ms(assist.metrics.total_ms))
    st.caption(
        "Deterministic align "
        f"{_format_duration_ms(assist.metrics.deterministic_ms)} · "
        f"LLM {_format_duration_ms(assist.metrics.llm_ms)} · "
        f"Unresolved {assist.metrics.unresolved_count}"
    )
    if token_usage and token_usage.get("usage_available_count"):
        st.caption(
            "Token usage — "
            f"prompt {token_usage.get('prompt_tokens_sum', 0)}, "
            f"completion {token_usage.get('completion_tokens_sum', 0)}, "
            f"total {token_usage.get('total_tokens_sum', 0)}"
        )
    elif token_usage:
        st.caption("Token usage unavailable from the gateway response.")

    det_resolved = [
        p for p in assist.proposals
        if p.status == AssistProposalStatus.accepted
        and p.matching_method == AssistMatchMethod.deterministic
    ]
    llm_resolved = [
        p for p in assist.proposals
        if p.status == AssistProposalStatus.accepted
        and p.matching_method == AssistMatchMethod.llm_assisted
        and not p.challenged_false_pass
    ]
    challenged = [
        p for p in assist.proposals
        if p.status == AssistProposalStatus.accepted and p.challenged_false_pass
    ]
    rejected = [
        p for p in assist.proposals
        if p.status != AssistProposalStatus.accepted
        and (
            p.status in {AssistProposalStatus.rejected, AssistProposalStatus.failed}
            or p.matching_method
            in {AssistMatchMethod.unresolved, AssistMatchMethod.rejected_proposal}
            or p.reason == "model_unresolved"
        )
    ]

    if det_resolved:
        st.markdown("**Resolved by deterministic delayed evidence**")
        for proposal in det_resolved:
            matched = _observation_timestamp_label(
                observed, proposal.matched_observed_index
            )
            support = _observation_timestamp_label(
                observed, proposal.supporting_observed_index
            )
            st.write(
                f"- Step {proposal.expected_step_index}: "
                f"{', '.join(ft.value for ft in proposal.finding_types) or 'finding'} "
                f"cleared (matched {matched}, supporting {support})"
            )
    if llm_resolved:
        st.markdown("**Resolved by Claude (validated)**")
        for proposal in llm_resolved:
            matched = _observation_timestamp_label(
                observed, proposal.matched_observed_index
            )
            support = _observation_timestamp_label(
                observed, proposal.supporting_observed_index
            )
            initiation = _observation_timestamp_label(
                observed, proposal.initiation_observed_index
            )
            support_bit = (
                f", supporting {support}"
                if proposal.supporting_observed_index is not None
                else ""
            )
            initiation_bit = (
                f", initiation {initiation}"
                if proposal.initiation_observed_index is not None
                else ""
            )
            st.write(
                f"- Step {proposal.expected_step_index}: "
                f"{', '.join(ft.value for ft in proposal.finding_types) or 'finding'} "
                f"→ `{proposal.model_answer}` "
                f"(confidence {proposal.confidence if proposal.confidence is not None else '—'}; "
                f"matched {matched}{support_bit}{initiation_bit})"
            )
            if proposal.rationale:
                st.caption(proposal.rationale)
    if challenged:
        st.markdown("**Challenged suspicious Stage 3 passes**")
        for proposal in challenged:
            matched = _observation_timestamp_label(
                observed, proposal.matched_observed_index
            )
            st.write(
                f"- Step {proposal.expected_step_index}: "
                f"challenged false pass on matched {matched} "
                f"(reason `{proposal.reason or 'challenge_false_pass'}`)"
            )
            if proposal.rationale:
                st.caption(proposal.rationale)
    if rejected:
        st.markdown("**Rejected or unresolved proposals**")
        for proposal in rejected:
            matched = _observation_timestamp_label(
                observed, proposal.matched_observed_index
            )
            st.write(
                f"- Step {proposal.expected_step_index}: "
                f"status `{proposal.status.value}`, "
                f"answer `{proposal.model_answer or '—'}`, "
                f"reason `{proposal.reason or proposal.matching_method.value}`, "
                f"matched {matched}"
            )

    left, right = st.columns(2)
    with left:
        st.markdown("**Deterministic Stage 3**")
        st.write(
            f"Verdict: `{deterministic.verdict.value}` · "
            f"findings: {len(deterministic.findings)} · "
            f"fidelity: {deterministic.fidelity_score:.0%}"
        )
        if deterministic.findings:
            for finding in deterministic.findings:
                st.caption(
                    f"{finding.type.value} @ step {finding.expected_step_index} "
                    f"(ts {_ms_to_timestamp(finding.evidence_timestamp_ms)}; "
                    f"obs {finding.observed_action_index})"
                )
        else:
            st.caption("No findings.")
    with right:
        st.markdown("**Assisted Stage 3 (pre–Stage 4)**")
        st.write(
            f"Verdict: `{assisted.verdict.value}` · "
            f"findings: {len(assisted.findings)} · "
            f"fidelity: {assisted.fidelity_score:.0%}"
        )
        if assisted.findings:
            for finding in assisted.findings:
                st.caption(
                    f"{finding.type.value} @ step {finding.expected_step_index} "
                    f"(ts {_ms_to_timestamp(finding.evidence_timestamp_ms)}; "
                    f"obs {finding.observed_action_index})"
                )
        else:
            st.caption("No findings.")


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

    stage3_assist = st.session_state.get("stage3_assist")
    deterministic_stage3 = st.session_state.get("deterministic_stage3")
    assisted_stage3 = st.session_state.get("assisted_stage3")
    if (
        stage3_assist is not None
        and deterministic_stage3 is not None
        and assisted_stage3 is not None
    ):
        st.divider()
        _render_stage3_assist_panel(
            assist=stage3_assist,
            deterministic=deterministic_stage3,
            assisted=assisted_stage3,
            token_usage=st.session_state.get("stage3_token_usage"),
            observed=st.session_state.get("observed_trace"),
        )

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

        if debug.get("stage3_assist"):
            st.markdown("**Stage 3 — Assist Report**")
            st.code(debug["stage3_assist"], language="json")

        if debug.get("crosscheck"):
            st.markdown("**Stage 4 — Cross-check Report (verdicts + refined result)**")
            st.code(debug["crosscheck"], language="json")

        for err in debug.get("errors", []):
            st.markdown(f"**{err['stage']} — Full Traceback**")
            st.code(err["tb"], language="python")


# ── Pipeline ──────────────────────────────────────────────────────────────────


def _write_recording_to_temp(video: BinaryIO | bytes, destination: IO[bytes]) -> None:
    """Copy an uploaded recording into an open file without an extra full copy.

    bytes inputs (used by tests) are written directly. File-like inputs are copied
    in chunks via shutil.copyfileobj so callers can pass a Streamlit UploadedFile
    without calling getbuffer().tobytes() first.
    """
    if isinstance(video, (bytes, bytearray, memoryview)):
        destination.write(video)
        return
    video.seek(0)
    shutil.copyfileobj(video, destination)


def _reset_run_state(stage3_mode: str) -> dict:
    """Clear per-run session state and return a fresh debug dict."""
    debug: dict = {
        "stage1": None,
        "stage2": None,
        "stage3": None,
        "stage3_assist": None,
        "crosscheck": None,
        "errors": [],
    }
    st.session_state["debug"] = debug
    st.session_state["crosscheck"] = None
    st.session_state["semantic_diagnostics"] = None
    st.session_state["export"] = None
    st.session_state["stage3_mode"] = stage3_mode
    st.session_state["stage3_assist"] = None
    st.session_state["deterministic_stage3"] = None
    st.session_state["assisted_stage3"] = None
    st.session_state["stage3_token_usage"] = None
    st.session_state["observed_trace"] = None
    return debug


def _on_assist_retry(attempt: int, code: int, delay_s: float) -> None:
    st.write(f"⏳ Assist model is busy (HTTP {code}) — retry {attempt} in {delay_s:.0f}s…")


def _replay_warning(evaluation: Stage3Evaluation) -> str | None:
    replay = evaluation.stage2c_replay
    if replay is None or not replay.unmatched_targets:
        return None
    return (
        "Stage 2C replay: the saved batch never answered "
        f"{', '.join(replay.unmatched_targets)}. These steps were not searched "
        '(not "searched, not found"); retry Stage 2 to search them.'
    )


def _execute_stage3(
    store: RecordingStore,
    recording_id: str,
    mode: Stage3Mode,
    debug: dict,
    *,
    stage2_version: int | None = None,
    video_path: Path | None = None,
) -> Stage3Evaluation:
    """Run and persist Stage 3 from saved artifacts, reporting progress in the status box."""
    s2_version = (
        stage2_version if stage2_version is not None else store.latest_stage2_version(recording_id)
    )
    if mode == Stage3Mode.evidence_aware:
        st.write(
            "Stage 3 — Aligning with LLM assist + evidence-aware + Stage 2C "
            f"({ALIGN_ASSIST_MODEL}) on Stage 2 v{s2_version}"
        )
        if store.has_artifact(recording_id, stage2c_filename(s2_version)):
            st.write("Stage 2C — replaying saved frame evidence (no new Stage 2C model call)")
        evaluation = run_stage3(
            store,
            recording_id,
            mode=mode,
            stage2_version=s2_version,
            openai_client=make_openai_client(),
            on_retry=_on_assist_retry,
            video_path=video_path,
        )
        assist = evaluation.assist
        assert assist is not None and evaluation.result is not None
        n_cand = (
            len(assist.candidate_evidence.candidates)
            if assist.candidate_evidence is not None
            else 0
        )
        debug["stage3_assist"] = assist.model_dump_json(indent=2)
        st.write(
            "Stage 3 complete (LLM-assisted evidence-aware) — "
            f"fidelity score: {evaluation.result.fidelity_score:.0%}, "
            f"{len(evaluation.result.findings)} finding(s); "
            f"accepted {assist.metrics.proposals_accepted}, "
            f"LLM requests {assist.metrics.llm_request_count}; "
            f"Stage 2C candidates {n_cand}"
        )
        warning = _replay_warning(evaluation)
        if warning is not None:
            st.warning(warning)
    else:
        st.write(f"Stage 3 — Aligning expected and observed traces (Stage 2 v{s2_version})")
        evaluation = run_stage3(store, recording_id, mode=mode, stage2_version=s2_version)
        assert evaluation.result is not None
        st.write(
            f"Stage 3 complete — fidelity score: {evaluation.result.fidelity_score:.0%}, "
            f"{len(evaluation.result.findings)} finding(s)"
        )
    if evaluation.inputs.unverified_artifacts:
        st.caption(
            "Inputs saved before checksums were recorded: "
            + ", ".join(evaluation.inputs.unverified_artifacts)
        )
    debug["stage3"] = evaluation.result.model_dump_json(indent=2)
    st.write(f"Saved Stage 3 evaluation v{evaluation.version}")
    return evaluation


def _execute_stage4(
    store: RecordingStore,
    recording_id: str,
    stage3_version: int,
    debug: dict,
    latency: LatencyMetrics,
    *,
    video_path: Path | None = None,
) -> Stage4Evaluation:
    st.write(
        "Stage 4 — Cross-checking assertions and low-confidence steps "
        f"(from Stage 3 v{stage3_version})"
    )
    evaluation = run_stage4(
        store, recording_id, stage3_version, latency=latency, video_path=video_path
    )
    assert evaluation.report is not None
    debug["crosscheck"] = evaluation.report.model_dump_json(indent=2)
    st.write(_stage4_completion_message(evaluation.report))
    st.write(f"Saved Stage 4 evaluation v{evaluation.version}")
    return evaluation


def _on_extraction_retry(attempt: int, code: int, delay_s: float) -> None:
    st.write(f"⏳ Extraction model is busy (HTTP {code}) — retry {attempt} in {delay_s:.0f}s…")


def _execute_stage2(
    store: RecordingStore,
    recording_id: str,
    debug: dict,
    latency: LatencyMetrics,
    *,
    video_path: Path | None = None,
) -> ObservedTrace:
    st.write("Stage 2 — Extracting observed trace from change-event frames")
    artifact = run_stage2(
        store,
        recording_id,
        video_path=video_path,
        on_retry=_on_extraction_retry,
        latency=latency,
    )
    observed_trace = artifact.observed_trace
    debug["stage2"] = observed_trace.model_dump_json(indent=2)
    st.write(
        f"Stage 2 complete — {len(observed_trace.actions)} observed action(s) extracted; "
        f"saved as Stage 2 v{artifact.version}"
    )
    return observed_trace


def _present_evaluation(
    *,
    recording: RecordingManifest,
    expected_trace: ExpectedTrace,
    observed_trace: ObservedTrace,
    stage3: Stage3Evaluation,
    stage4: Stage4Evaluation | None,
    latency: LatencyMetrics | None,
) -> tuple[AuditResult, ExpectedTrace]:
    """Populate session state from persisted records; return what _render_results shows."""
    assert stage3.result is not None
    assist = stage3.assist
    report = stage4.report if stage4 is not None else None
    mode_label = _STAGE3_MODE_LABELS[stage3.mode]
    st.session_state["stage3_mode"] = mode_label
    st.session_state["stage3_assist"] = assist
    st.session_state["deterministic_stage3"] = (
        stage3.deterministic_result if assist is not None else None
    )
    st.session_state["assisted_stage3"] = stage3.result if assist is not None else None
    st.session_state["stage3_token_usage"] = stage3.token_usage
    st.session_state["observed_trace"] = observed_trace
    if report is not None and report.semantic_diagnostics:
        st.session_state["semantic_diagnostics"] = report.semantic_diagnostics
    if report is not None and report.verdicts:
        st.session_state["crosscheck"] = report.verdicts
    st.session_state["latency"] = (
        _latency_session_payload(latency) if latency is not None else None
    )
    st.session_state["export"] = _build_export_payload(
        scenario_title=expected_trace.scenario_title,
        recording_filename=recording.source_filename,
        original_audit_result=stage3.result,
        expected_trace=expected_trace,
        observed_trace=observed_trace,
        crosscheck_report=report,
        latency_metrics=_latency_export_payload(latency) if latency is not None else None,
        stage3_mode=mode_label,
        deterministic_stage3=st.session_state["deterministic_stage3"],
        assisted_stage3=st.session_state["assisted_stage3"],
        stage3_assist=assist,
        stage3_token_usage=stage3.token_usage,
    )
    final = report.refined_result if report is not None else stage3.result
    return final, expected_trace


def _ingest_video(
    store: RecordingStore,
    recording_id: str,
    video: BinaryIO | bytes,
    video_name: str,
    scratch_dir: Path,
) -> Path:
    """Store the upload (retention "keep") or only its checksum (retention "discard").

    With "discard" the file lives in ``scratch_dir`` for this run only.
    """
    suffix = Path(video_name).suffix or ".mp4"
    if store.video_retention == VideoRetention.keep:
        return store.write_video(
            recording_id,
            lambda handle: _write_recording_to_temp(video, handle),
            suffix=suffix,
        )
    path = scratch_dir / f"upload{suffix}"
    with path.open("wb") as handle:
        _write_recording_to_temp(video, handle)
    store.record_unretained_video(recording_id, path)
    return path


def _run_pipeline(
    video: BinaryIO | bytes,
    video_name: str,
    scenario_text: str,
    run_crosscheck: bool,
    stage3_mode: str = STAGE3_MODE_DETERMINISTIC,
) -> tuple[AuditResult, ExpectedTrace] | None:
    """Run Stage 1 → Stage 2 → Stage 3 (→ optional cross-check) for a new recording.

    Each stage's output is persisted as soon as it completes: the recording folder,
    stage-1.json and the video after Stage 1, stage-2.json after Stage 2, stage-2c.json
    and a new Stage 3 version after Stage 3, and a new Stage 4 version after Stage 4.
    An interrupted run therefore keeps everything that finished.

    Returns (AuditResult, ExpectedTrace) on success, or None on failure (error messages
    are shown via st.error). Intermediate results and full tracebacks are written to
    st.session_state["debug"] for the debug panel.
    """
    debug = _reset_run_state(stage3_mode)
    latency = LatencyMetrics()
    pipeline_started = monotonic_ms()
    store = RecordingStore.from_env()

    with st.status("Running audit pipeline…", expanded=True) as status:
        # ── Stage 1: parse scenario ───────────────────────────────────────────
        st.write("Stage 1 — Parsing expected scenario")
        try:
            openai_client = make_openai_client()
            stage1_started_at = utc_now()
            stage1_started = monotonic_ms()
            expected_trace = parse_scenario(scenario_text, openai_client=openai_client)
            latency.stage1_ms = elapsed_ms(stage1_started)
            logger.info("stage1 duration_ms=%.1f step_count=%d", latency.stage1_ms, len(expected_trace.steps))
            debug["stage1"] = expected_trace.model_dump_json(indent=2)
            recording = store.create_recording(
                title=expected_trace.scenario_title,
                source_filename=video_name,
                scenario_text=scenario_text,
            )
            recording_id = recording.recording_id
            store.save_stage1(
                recording_id,
                build_stage1_artifact(
                    expected_trace,
                    started_at=stage1_started_at,
                    duration_ms=latency.stage1_ms,
                ),
            )
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
        st.write(f"Saved recording `{recording_id}` to `{store.recording_dir(recording_id)}`")

        with tempfile.TemporaryDirectory(prefix="reckon-upload-") as scratch:
            try:
                # ── Stage 2: extract observed trace ──────────────────────────
                st.write("Stage 2 — Detecting screen changes in the recording")
                video_path = _ingest_video(store, recording_id, video, video_name, Path(scratch))
                run_video = None if store.video_retention == VideoRetention.keep else video_path
                observed_trace = _execute_stage2(
                    store, recording_id, debug, latency, video_path=run_video
                )

                # ── Stage 3: align traces ─────────────────────────────────────
                stage3_started = monotonic_ms()
                stage3 = _execute_stage3(
                    store, recording_id, _STAGE3_MODES[stage3_mode], debug, video_path=run_video
                )
                latency.stage3_ms = elapsed_ms(stage3_started)
                logger.info(
                    "stage3 duration_ms=%.1f finding_count=%d mode=%s",
                    latency.stage3_ms,
                    len(stage3.result.findings) if stage3.result else 0,
                    stage3_mode,
                )

                # ── Stage 4 (optional): cross-check pass ──────────────────────
                stage4: Stage4Evaluation | None = None
                if run_crosscheck:
                    stage4 = _execute_stage4(
                        store, recording_id, stage3.version, debug, latency, video_path=run_video
                    )
            except Exception as exc:
                stage = "Stage 2/3/4"
                debug["errors"].append({"stage": stage, "error": str(exc), "tb": traceback.format_exc()})
                status.update(label=f"{stage} failed", state="error")
                st.error(f"{stage} failed: {exc}")
                latency.total_ms = elapsed_ms(pipeline_started)
                log_metrics(latency, where="app")
                return None

        latency.total_ms = elapsed_ms(pipeline_started)
        log_metrics(latency, where="app")
        outcome = _present_evaluation(
            recording=recording,
            expected_trace=expected_trace,
            observed_trace=observed_trace,
            stage3=stage3,
            stage4=stage4,
            latency=latency,
        )
        status.update(label="Audit complete", state="complete", expanded=False)

    return outcome


_SAVED_STAGE_LABELS = {"stage2": "Stage 2 retry", "stage3": "Stage 3", "stage4": "Stage 4"}


def _run_saved_stage(
    store: RecordingStore,
    recording_id: str,
    *,
    action: str,
    stage2_version: int | None = None,
    stage3_version: int | None = None,
) -> tuple[AuditResult, ExpectedTrace] | None:
    """Run one stage on a saved recording: ``stage2`` retry, ``stage3``, or ``stage4``.

    Stage 3 uses ``stage2_version`` (latest when None); Stage 4 needs ``stage3_version``.
    A Stage 2 retry shows no audit result, so it returns None.
    """
    label = _SAVED_STAGE_LABELS[action]
    debug = _reset_run_state(STAGE3_MODE_EVIDENCE)
    latency = LatencyMetrics()
    started = monotonic_ms()
    with st.status(f"Running {label} for {recording_id}…", expanded=True) as status:
        try:
            recording = store.load_manifest(recording_id)
            expected_trace = store.load_stage1(recording_id).expected_trace
            if action == "stage2":
                _execute_stage2(store, recording_id, debug, latency)
                status.update(label=f"{label} complete", state="complete", expanded=False)
                return None
            if action == "stage3":
                stage3 = _execute_stage3(
                    store,
                    recording_id,
                    Stage3Mode.evidence_aware,
                    debug,
                    stage2_version=stage2_version,
                )
                latency.stage3_ms = elapsed_ms(started)
                stage4 = None
            else:
                if stage3_version is None:
                    raise ValueError("Select a Stage 3 evaluation to cross-check")
                stage3 = store.load_stage3(recording_id, stage3_version)
                stage4 = _execute_stage4(store, recording_id, stage3_version, debug, latency)
            observed_trace = store.load_stage2(
                recording_id, stage3.inputs.stage2_version
            ).observed_trace
        except (MissingArtifactError, ArtifactIntegrityError, RecordingBusyError) as exc:
            status.update(label=f"{label} cannot run", state="error")
            st.error(f"{label} cannot run: {exc}")
            return None
        except Exception as exc:
            debug["errors"].append({"stage": label, "error": str(exc), "tb": traceback.format_exc()})
            status.update(label=f"{label} failed", state="error")
            st.error(f"{label} failed: {exc}")
            return None
        latency.total_ms = elapsed_ms(started)
        log_metrics(latency, where=f"app.{action}")
        outcome = _present_evaluation(
            recording=recording,
            expected_trace=expected_trace,
            observed_trace=observed_trace,
            stage3=stage3,
            stage4=stage4,
            latency=latency,
        )
        status.update(label=f"{label} complete", state="complete", expanded=False)
    return outcome


# ── Saved recordings panel ────────────────────────────────────────────────────

_PANEL_NOTICES_KEY = "saved_recordings_notices"
_PANEL_PENDING_KEY = "saved_recordings_pending_selection"
_SAVED_WORKSPACE_KEY = "saved_recordings_open"


def _recording_label(recording: RecordingManifest) -> str:
    created = recording.created_at.strftime("%Y-%m-%d %H:%M")
    return f"{recording.recording_id}  ·  {recording.source_filename}  ·  {created} UTC"


def _stage3_label(evaluation: Stage3Evaluation) -> str:
    verdict = evaluation.result.verdict.value if evaluation.result is not None else "—"
    return f"v{evaluation.version} · Stage 2 v{evaluation.inputs.stage2_version} · {verdict}"


def _refresh_panel(selection: dict[str, int], notices: list[tuple[str, str]]) -> None:
    """Rerun so the selectors list (and select) the version a run just saved.

    Widget values can only be set before their widget is drawn, so the selection is
    applied at the top of the next run; results stay in session state.
    """
    st.session_state[_PANEL_PENDING_KEY] = selection
    st.session_state[_PANEL_NOTICES_KEY] = notices
    st.rerun()


def _render_recording_panel(store: RecordingStore) -> None:
    """Pick a recording, its Stage 2 version and Stage 3 result; run Stage 3, Stage 4 or retry Stage 2."""
    st.divider()
    st.header("Saved Recordings")
    recordings = store.list_recordings()
    if not recordings:
        st.info("No saved recordings yet. Run the full pipeline to create one.")
        return

    by_id = {recording.recording_id: recording for recording in recordings}
    recording_id = st.selectbox(
        "Recording",
        list(by_id),
        format_func=lambda rid: _recording_label(by_id[rid]),
        key="selected_recording",
    )
    try:
        store.recover_stale_runs(recording_id)
    except RecordingBusyError:
        logger.info("stale-run check skipped: %s is busy", recording_id)
    for key, value in st.session_state.pop(_PANEL_PENDING_KEY, {}).items():
        st.session_state[key] = value

    has_stage1 = store.has_artifact(recording_id, STAGE1_FILENAME)
    stage2_versions = store.stage2_versions(recording_id)
    video_path = store.video_path(recording_id)
    completed3 = {
        item.version: item
        for item in reversed(store.list_stage3(recording_id))
        if item.status == RunStatus.completed
    }
    stage2_key = f"stage2_version_{recording_id}"
    stage3_key = f"stage3_version_{recording_id}"

    stage2_col, stage3_col = st.columns(2)
    with stage2_col:
        stage2_version = st.selectbox(
            "Stage 2 version",
            list(reversed(stage2_versions)),
            format_func=lambda version: f"v{version}",
            key=stage2_key,
            help="The observed trace Run Stage 3 aligns against.",
        ) if stage2_versions else None
        if stage2_version is None:
            st.caption("No Stage 2 extraction saved yet.")
    with stage3_col:
        stage3_version = st.selectbox(
            "Stage 3 result",
            list(completed3),
            format_func=lambda version: _stage3_label(completed3[version]),
            key=stage3_key,
            help="The evaluation Run Stage 4 cross-checks.",
        ) if completed3 else None
        if stage3_version is None:
            st.caption("No completed Stage 3 result yet.")

    has_stage2c = stage2_version is not None and store.has_artifact(
        recording_id, stage2c_filename(stage2_version)
    )
    has_stills = stage2_version is not None and bool(
        store.load_stage2(recording_id, stage2_version).evidence_frames
    )
    frames_available = video_path is not None or has_stage2c or has_stills

    run3_col, run4_col, retry2_col = st.columns(3)
    with run3_col:
        run3 = st.button(
            "Run Stage 3",
            disabled=not has_stage1 or stage2_version is None or not frames_available,
            key="run_stage3",
            help="Align the selected Stage 2 version and save a new Stage 3 result.",
        )
        if stage2_version is not None and not frames_available:
            st.caption(
                "Stage 3 evidence review needs frames: no video, no saved evidence stills "
                "and no frozen Stage 2C for this Stage 2 version."
            )
    with run4_col:
        run4 = st.button(
            "Run Stage 4",
            disabled=stage3_version is None or video_path is None,
            key="run_stage4",
            help="Cross-check the selected Stage 3 result.",
        )
        if video_path is None:
            st.caption(
                "Stage 4 needs the original video: it decodes frame windows around each "
                "target, which saved evidence stills do not contain."
            )
    with retry2_col:
        retry2 = st.button(
            "Retry Stage 2",
            disabled=not has_stage1 or video_path is None,
            key="retry_stage2",
            help="Extract again from the saved video into a new Stage 2 version; "
            "earlier versions are kept.",
        )
        if video_path is None:
            st.caption("Stage 2 retry needs the original video, which is not available.")

    for kind, message in st.session_state.pop(_PANEL_NOTICES_KEY, []):
        getattr(st, kind)(message)

    if not (retry2 or run3 or run4):
        return
    st.session_state["result"] = None
    st.session_state["latency"] = None
    st.session_state["export"] = None
    if retry2:
        _run_saved_stage(store, recording_id, action="stage2")
        added = sorted(set(store.stage2_versions(recording_id)) - set(stage2_versions))
        if added:
            _refresh_panel({stage2_key: added[-1]}, [("success", f"Saved Stage 2 v{added[-1]}.")])
        return
    if run3:
        st.session_state["result"] = _run_saved_stage(
            store, recording_id, action="stage3", stage2_version=stage2_version
        )
        added3 = [
            item
            for item in store.list_stage3(recording_id)
            if item.status == RunStatus.completed and item.version not in completed3
        ]
        if added3:
            latest = added3[-1]
            notices = [("success", f"Saved Stage 3 v{latest.version} (Stage 2 v{latest.inputs.stage2_version}).")]
            warning = _replay_warning(latest)
            if warning is not None:
                notices.append(("warning", warning))
            _refresh_panel({stage3_key: latest.version}, notices)
        return
    if stage3_version is not None:
        st.session_state["result"] = _run_saved_stage(
            store, recording_id, action="stage4", stage3_version=stage3_version
        )


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
        "actually happened in the video; Stage 3 aligns the two traces (deterministic by "
        "default, with an optional evidence-aware agent review); Stage 4 cross-checks. "
        "Every run is saved. Use Saved Recording re-runs Stage 2, Stage 3 and Stage 4 from one."
    )

    st.divider()

    if st.session_state.get(_SAVED_WORKSPACE_KEY):
        if st.button("Back", key="saved_recordings_back"):
            st.session_state[_SAVED_WORKSPACE_KEY] = False
            st.rerun()
        _render_recording_panel(RecordingStore.from_env())
        if st.session_state.get("result") is not None:
            audit_result, expected_trace = st.session_state["result"]
            _render_results(audit_result, expected_trace)
        _render_debug_panel()
        return

    # ── Inputs ────────────────────────────────────────────────────────────────
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("Test Recording")
        uploaded_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "webm"],
            max_upload_size=MAX_UPLOAD_SIZE_MB,
            help=(
                "MP4 recommended. The recording is saved in its recording folder; "
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

    stage3_mode = st.radio(
        "Stage 3 alignment mode",
        options=[
            STAGE3_MODE_DETERMINISTIC,
            STAGE3_MODE_EVIDENCE,
        ],
        index=0,
        horizontal=True,
        help=(
            "Deterministic (default) uses the existing align() path. "
            "Evidence-aware runs the same deterministic base, then an agent review "
            f"with {ALIGN_ASSIST_MODEL}: Claude proposes, Python validates before apply. "
            "It retrieves multi-observation windows for coarse steps, asks Claude to "
            "label initiation/progress/completion/distractor roles, and runs Stage 2C "
            "expected-guided frame evidence discovery (frame IDs only; Python validates "
            "timestamps). Initiation alone never clears assertions. Only accepted "
            "resolutions are excluded from Stage 4 semantic review."
        ),
    )

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
        "Run Full Pipeline",
        type="primary",
        disabled=run_disabled,
        use_container_width=False,
        help="Extract a new video (Stages 1–2) and run the complete evaluation.",
    )

    # Persist results across rerenders so that expanding a finding or verdict
    # does not discard the current audit output.
    if "result" not in st.session_state:
        st.session_state["result"] = None

    if st.button("Use Saved Recording", key="use_saved_recording"):
        st.session_state[_SAVED_WORKSPACE_KEY] = True
        st.rerun()

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
            stage3_mode=stage3_mode,
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
