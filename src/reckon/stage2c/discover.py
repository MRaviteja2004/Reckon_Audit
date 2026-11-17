"""Orchestrate Stage 2C expected-guided frame evidence discovery."""

import base64
import logging
from pathlib import Path

import instructor
from instructor.core import InstructorRetryException
from openai import APIError, APITimeoutError
from pydantic import ValidationError

from reckon.frames import extract_frames_at, frame_to_jpeg_bytes
from reckon.latency import elapsed_ms, monotonic_ms
from reckon.retry import RetryCallback
from reckon.schema import (
    AuditResult,
    CandidateEvidenceTrace,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedTrace,
    FrameManifest,
    ObservedTrace,
)
from reckon.stage2c.propose import (
    STAGE2C_MODEL,
    RawFrameEvidenceBatch,
    RawFrameEvidenceCandidate,
    propose_frame_evidence_batch,
)
from reckon.stage2c.select import Stage2CSearchTarget, select_stage2c_targets
from reckon.stage2c.validate import validate_stage2c_batch

logger = logging.getLogger(__name__)

MAX_FRAME_DIMENSION = 1536


def _frame_content_from_jpegs(
    manifest: FrameManifest,
    jpeg_by_frame_id: dict[str, bytes],
) -> list[dict]:
    content: list[dict] = []
    for entry in manifest.frames:
        jpeg = jpeg_by_frame_id.get(entry.frame_id) or jpeg_by_frame_id.get(
            entry.frame_id.upper()
        )
        if jpeg is None:
            continue
        encoded = base64.b64encode(jpeg).decode("ascii")
        content.append({"type": "text", "text": f"{entry.frame_id}:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                },
            }
        )
    return content


def load_frame_jpegs(
    video_path: Path,
    manifest: FrameManifest,
) -> dict[str, bytes]:
    """Decode manifest frames from video into JPEG bytes keyed by frame_id."""
    timestamps = [entry.timestamp_ms for entry in manifest.frames]
    samples = extract_frames_at(video_path, timestamps)
    out: dict[str, bytes] = {}
    for entry, (_, image) in zip(manifest.frames, samples, strict=True):
        out[entry.frame_id] = frame_to_jpeg_bytes(
            image, max_dimension=MAX_FRAME_DIMENSION
        )
    return out


def _observed_context(observed: ObservedTrace | None) -> str | None:
    if observed is None or not observed.actions:
        return None
    lines: list[str] = []
    for action in observed.actions[:24]:
        evidence = action.assertion_evidence
        lines.append(
            f"- O{action.index}: action={action.action.value} "
            f"target={action.target!r} ts_ms={action.timestamp_ms} "
            f"visible_text={evidence.visible_text if evidence else None!r} "
            f"screen={action.screen_description!r}"
        )
    return "\n".join(lines)


def _not_found_stub(target: Stage2CSearchTarget) -> RawFrameEvidenceCandidate:
    return RawFrameEvidenceCandidate(
        expected_step_id=target.expected_step_id,
        frame_ids=[],
        evidence_type=target.evidence_type,
        observed_description="",
        visible_text=[],
        confidence=0.0,
        search_status=CandidateSearchStatus.not_found,
    )


def discover_candidate_evidence(
    expected: ExpectedTrace,
    deterministic: AuditResult,
    manifest: FrameManifest,
    *,
    source_file: str,
    observed: ObservedTrace | None = None,
    video_path: Path | None = None,
    jpeg_by_frame_id: dict[str, bytes] | None = None,
    raw_batch: RawFrameEvidenceBatch | None = None,
    openai_client: instructor.Instructor | None = None,
    model: str = STAGE2C_MODEL,
    on_retry: RetryCallback | None = None,
    targets: list[Stage2CSearchTarget] | None = None,
) -> tuple[CandidateEvidenceTrace, int, float, str | None]:
    """Run Stage 2C discovery + validation.

    Returns (trace, llm_request_count, llm_ms, error).
    Pass ``raw_batch`` to skip the LLM (tests / offline). Never mutates ObservedTrace.
    """
    search_targets = targets if targets is not None else select_stage2c_targets(
        expected, deterministic, observed=observed
    )
    if not search_targets:
        return (
            CandidateEvidenceTrace(source_file=source_file, candidates=[]),
            0,
            0.0,
            None,
        )

    error: str | None = None
    llm_ms = 0.0
    llm_requests = 0
    batch = raw_batch

    if batch is None:
        jpegs = jpeg_by_frame_id
        if jpegs is None:
            if video_path is None:
                stubs = [_not_found_stub(target) for target in search_targets]
                trace = validate_stage2c_batch(
                    stubs,
                    search_targets,
                    manifest,
                    deterministic,
                    source_file=source_file,
                    expected=expected,
                    observed=observed,
                )
                return trace, 0, 0.0, "missing_frame_payload"
            jpegs = load_frame_jpegs(video_path, manifest)
        frame_content = _frame_content_from_jpegs(manifest, jpegs)
        if not frame_content:
            stubs = [_not_found_stub(target) for target in search_targets]
            trace = validate_stage2c_batch(
                stubs,
                search_targets,
                manifest,
                deterministic,
                source_file=source_file,
                expected=expected,
                observed=observed,
            )
            return trace, 0, 0.0, "empty_frame_content"

        llm_started = monotonic_ms()
        llm_requests = 1
        try:
            batch = propose_frame_evidence_batch(
                search_targets,
                manifest,
                frame_content,
                openai_client=openai_client,
                model=model,
                on_retry=on_retry,
                observed_context=_observed_context(observed),
            )
        except (
            APIError,
            APITimeoutError,
            InstructorRetryException,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "stage2c.discover llm_failed error_type=%s", type(exc).__name__
            )
            llm_ms = elapsed_ms(llm_started)
            stubs = [_not_found_stub(target) for target in search_targets]
            trace = validate_stage2c_batch(
                stubs,
                search_targets,
                manifest,
                deterministic,
                source_file=source_file,
                expected=expected,
                observed=observed,
            )
            # Mark stubs as LLM failure for auditability.
            failed = [
                item.model_copy(
                    update={
                        "validation_status": CandidateValidationStatus.rejected,
                        "rejection_reason": f"llm_{type(exc).__name__}",
                    }
                )
                for item in trace.candidates
            ]
            return (
                CandidateEvidenceTrace(source_file=source_file, candidates=failed),
                llm_requests,
                llm_ms,
                f"llm_{type(exc).__name__}",
            )
        else:
            llm_ms = elapsed_ms(llm_started)

    # Ensure every search target appears in the audit trail.
    by_id = {
        item.expected_step_id.strip().upper(): item for item in batch.candidates
    }
    raw_list: list[RawFrameEvidenceCandidate] = []
    for target in search_targets:
        raw = by_id.get(target.expected_step_id.upper())
        if raw is None:
            raw_list.append(_not_found_stub(target))
        else:
            # Prefer the selector's evidence_type when LLM omits a coherent type.
            if (
                raw.evidence_type
                not in {
                    CandidateEvidenceType.ACTION,
                    CandidateEvidenceType.ASSERTION_OUTCOME,
                }
            ):
                raw = raw.model_copy(update={"evidence_type": target.evidence_type})
            raw_list.append(raw)

    trace = validate_stage2c_batch(
        raw_list,
        search_targets,
        manifest,
        deterministic,
        source_file=source_file,
        expected=expected,
        observed=observed,
    )
    return trace, llm_requests, llm_ms, error
