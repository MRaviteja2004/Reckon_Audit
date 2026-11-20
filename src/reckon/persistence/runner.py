"""Run Stage 3 / Stage 4 from persisted artifacts and record each run as a new version.

Evaluation logic is not reimplemented here: Stage 3 is ``align`` or
``align_with_assist``; Stage 4 is ``run_cross_check``. This module only loads and
verifies inputs, invokes them with the same arguments the Streamlit pipeline uses,
observes their model calls, and saves.
"""

import logging
import os
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from reckon.aligner import align
from reckon.assist import ASSIST_SEMANTIC_SKIP_CONFIDENCE, align_with_assist
from reckon.assist_propose import ALIGN_ASSIST_MODEL, ASSIST_CONFIDENCE_FLOOR
from reckon.crosscheck import (
    CROSSCHECK_MODEL,
    FRAMES_PER_TARGET,
    LOW_CONFIDENCE_THRESHOLD,
    WINDOW_PAD_MS,
    run_cross_check,
)
from reckon.latency import LatencyMetrics, elapsed_ms, latency_scope, monotonic_ms
from reckon.parser import make_openai_client
from reckon.persistence.capture import RecordingClient, token_usage_summary
from reckon.persistence.extraction import frame_manifest_for
from reckon.persistence.files import ArtifactIntegrityError
from reckon.persistence.metadata import build_metadata, utc_now
from reckon.persistence.redact import redact_secrets
from reckon.persistence.schema import (
    ExecutionMetadata,
    ModelIOCapture,
    RunStatus,
    Stage2CArtifact,
    Stage2CReplayCheck,
    Stage2CSource,
    Stage3Evaluation,
    Stage3Inputs,
    Stage3Mode,
    Stage4Evaluation,
)
from reckon.persistence.store import (
    STAGE1_FILENAME,
    STAGE3_DIRNAME,
    STAGE4_DIRNAME,
    MissingArtifactError,
    RecordingStore,
    stage2_filename,
    stage2c_filename,
)
from reckon.retry import RetryCallback
from reckon.schema import (
    AssistCandidateKind,
    AssistedAlignResult,
    AssistMatchMethod,
    AssistProposalRecord,
    AssistProposalStatus,
    AssistReport,
    CandidateEvidenceTrace,
    CandidateValidationStatus,
    FrameManifest,
)
from reckon.stage2c import STAGE2C_CONFIDENCE_FLOOR, RawFrameEvidenceBatch

logger = logging.getLogger(__name__)

MODEL_IO_CAPTURE_ENV = "RECKON_CAPTURE_MODEL_IO"
STAGE2C_STUB_REASON = "search_not_found"
STAGE2C_NOT_SEARCHED_REASON = "stage2c_not_searched_frozen_replay"

EvaluationT = TypeVar("EvaluationT", Stage3Evaluation, Stage4Evaluation)


def model_io_capture_from_env() -> ModelIOCapture:
    return ModelIOCapture(os.environ.get(MODEL_IO_CAPTURE_ENV) or ModelIOCapture.full.value)


def reviewed_skip_pairs(assist: AssistReport) -> set[tuple[int, int]]:
    """Only accepted Stage 3 resolutions are excluded from Stage 4 semantic review."""
    return {
        (pair.expected_step_index, pair.observed_action_index)
        for pair in assist.reviewed_semantic_pairs
    }


def _finished_metadata(metadata: ExecutionMetadata, started_ms: float) -> ExecutionMetadata:
    return metadata.model_copy(
        update={"completed_at": utc_now(), "duration_ms": elapsed_ms(started_ms)}
    )


def _error_text(exc: BaseException) -> str:
    detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return redact_secrets(detail or type(exc).__name__)


def _recording_client(openai_client: Any | None, capture: ModelIOCapture) -> RecordingClient:
    if openai_client is not None:
        return RecordingClient(openai_client, capture=capture)
    return RecordingClient(factory=make_openai_client, capture=capture)


def _finish_or_preserve(
    finish: Callable[[EvaluationT], None],
    preserve: Callable[[EvaluationT], object],
    record: EvaluationT,
) -> None:
    try:
        finish(record)
    except Exception:
        preserve(record)
        raise


def _execute(
    finish: Callable[[EvaluationT], None],
    preserve: Callable[[EvaluationT], object],
    evaluation: EvaluationT,
    started_ms: float,
    client: RecordingClient | None,
    work: Callable[[], dict[str, Any]],
) -> EvaluationT:
    """Run ``work`` and finish the reserved record as completed, failed or interrupted.

    ``BaseException`` covers Ctrl-C and Streamlit's stop/rerun, which are not
    ``Exception`` subclasses; model calls captured before the interruption are kept.
    If the final write fails, the record goes to ``preserve`` so its model calls survive.
    """
    try:
        update = work()
    except BaseException as exc:
        status = RunStatus.failed if isinstance(exc, Exception) else RunStatus.interrupted
        try:
            _finish_or_preserve(
                finish,
                preserve,
                evaluation.model_copy(
                    update={
                        "status": status,
                        "error": _error_text(exc),
                        "metadata": _finished_metadata(evaluation.metadata, started_ms),
                        "model_calls": list(client.calls) if client is not None else [],
                    }
                ),
            )
        except Exception:
            logger.exception("could not record the %s run; re-raising its error", status.value)
        raise
    completed = evaluation.model_copy(
        update={
            **update,
            "status": RunStatus.completed,
            "metadata": _finished_metadata(evaluation.metadata, started_ms),
            "model_calls": list(client.calls) if client is not None else [],
        }
    )
    _finish_or_preserve(finish, preserve, completed)
    return completed


def _video_absence(store: RecordingStore, recording_id: str) -> str:
    manifest = store.load_manifest(recording_id)
    if manifest.video_sha256 is not None and not manifest.video_retained:
        return "not retained (RECKON_VIDEO_RETENTION=discard)"
    return "missing from the recording folder"


def _verified(
    store: RecordingStore, recording_id: str, filename: str, unverified: list[str]
) -> str:
    digest, verified = store.verify_artifact(recording_id, filename)
    if not verified:
        unverified.append(filename)
    return digest


def _stage3_metadata(
    mode: Stage3Mode, started_at: datetime, capture: ModelIOCapture
) -> ExecutionMetadata:
    models: dict[str, str] = {}
    run_config: dict[str, str | int | float | bool | None] = {"stage3_mode": mode.value}
    if mode == Stage3Mode.evidence_aware:
        models = {"stage3_agent_review": ALIGN_ASSIST_MODEL, "stage2c": ALIGN_ASSIST_MODEL}
        run_config |= {
            "assist_confidence_floor": ASSIST_CONFIDENCE_FLOOR,
            "assist_semantic_skip_confidence": ASSIST_SEMANTIC_SKIP_CONFIDENCE,
            "stage2c_confidence_floor": STAGE2C_CONFIDENCE_FLOOR,
            "model_io_capture": capture.value,
        }
    return build_metadata(started_at, models=models, config=run_config)


def _target_ids(trace: CandidateEvidenceTrace | None) -> set[str]:
    if trace is None:
        return set()
    return {item.expected_step_id.strip().upper() for item in trace.candidates}


def _replay_check(
    frozen: Stage2CArtifact, current: CandidateEvidenceTrace | None
) -> Stage2CReplayCheck:
    frozen_ids = _target_ids(frozen.candidate_evidence)
    current_ids = _target_ids(current)
    return Stage2CReplayCheck(
        frozen_targets=sorted(frozen_ids),
        current_targets=sorted(current_ids),
        unmatched_targets=sorted(current_ids - frozen_ids),
    )


def _mark_not_searched(
    bundle: AssistedAlignResult, unmatched_targets: list[str]
) -> AssistedAlignResult:
    """Relabel replay targets the frozen Stage 2C batch never answered.

    Discovery validates such targets as ``search_not_found`` stubs, which reads as
    "searched, nothing found". No search happened, so they are removed from the
    candidate trace (still listed in ``stage2c_replay.unmatched_targets``) and their
    rejected proposals become ``skipped``. Rejected proposals are never applied, so the
    audit result and Stage 4 eligibility are unchanged.
    """
    assist = bundle.assist
    trace = assist.candidate_evidence
    unmatched = set(unmatched_targets)
    if trace is None or not unmatched:
        return bundle
    stubs = [
        item
        for item in trace.candidates
        if item.expected_step_id.strip().upper() in unmatched
        and item.validation_status == CandidateValidationStatus.rejected
        and item.rejection_reason == STAGE2C_STUB_REASON
    ]
    stub_steps = {item.expected_step_index for item in stubs}
    proposals: list[AssistProposalRecord] = []
    relabelled = 0
    for proposal in assist.proposals:
        if (
            proposal.kind == AssistCandidateKind.frame_evidence
            and proposal.status == AssistProposalStatus.rejected
            and proposal.expected_step_index in stub_steps
        ):
            proposal = proposal.model_copy(
                update={
                    "status": AssistProposalStatus.skipped,
                    "reason": STAGE2C_NOT_SEARCHED_REASON,
                    "matching_method": AssistMatchMethod.unresolved,
                }
            )
            relabelled += 1
        proposals.append(proposal)
    return bundle.model_copy(
        update={
            "assist": assist.model_copy(
                update={
                    "proposals": proposals,
                    "metrics": assist.metrics.model_copy(
                        update={"proposals_rejected": assist.metrics.proposals_rejected - relabelled}
                    ),
                    "candidate_evidence": trace.model_copy(
                        update={"candidates": [c for c in trace.candidates if c not in stubs]}
                    ),
                }
            )
        }
    )


def _stage2c_artifact(
    bundle: AssistedAlignResult,
    captured: RawFrameEvidenceBatch | None,
    manifest: FrameManifest | None,
    *,
    stage2_version: int,
    stage2_sha256: str,
    video_sha256: str | None,
    stage3_version: int,
    started_at: datetime,
) -> Stage2CArtifact | None:
    """Freeze Stage 2C only when the model answered and discovery finished without error.

    A run with no search targets makes no Stage 2C call; nothing is frozen, so a later
    run whose targets differ still searches. A failed discovery is never frozen.
    """
    assist = bundle.assist
    if (
        captured is None
        or manifest is None
        or assist.candidate_evidence is None
        or assist.error is not None
    ):
        return None
    return Stage2CArtifact(
        metadata=build_metadata(
            started_at,
            completed_at=utc_now(),
            models={"stage2c": assist.model},
            config={"stage2c_confidence_floor": STAGE2C_CONFIDENCE_FLOOR},
        ),
        stage2_version=stage2_version,
        stage2_sha256=stage2_sha256,
        video_sha256=video_sha256,
        produced_by_stage3_version=stage3_version,
        frame_manifest=manifest,
        raw_batch=captured,
        candidate_evidence=assist.candidate_evidence,
        stage2c_skip_visual_steps=assist.stage2c_skip_visual_steps,
    )


def run_stage3(
    store: RecordingStore,
    recording_id: str,
    *,
    mode: Stage3Mode = Stage3Mode.evidence_aware,
    stage2_version: int | None = None,
    openai_client: Any | None = None,
    on_retry: RetryCallback | None = None,
    video_path: Path | None = None,
) -> Stage3Evaluation:
    """Deterministic base alignment, then (evidence-aware mode) the agent review.

    Uses stage-1 and the chosen Stage 2 version (latest by default), verifying their
    checksums first. Stage 2C for that Stage 2 version is replayed when frozen; the first
    Stage 2C model answer is frozen. Frames come from the video (``video_path`` or the
    stored one) or, when the video was not retained, from the saved evidence stills.
    Every call creates a new Stage 3 version; failures and interruptions are recorded.
    """
    s2_version = (
        stage2_version if stage2_version is not None else store.latest_stage2_version(recording_id)
    )
    unverified: list[str] = []
    stage1_sha = _verified(store, recording_id, STAGE1_FILENAME, unverified)
    stage2_sha = _verified(store, recording_id, stage2_filename(s2_version), unverified)
    expected = store.load_stage1(recording_id).expected_trace
    stage2 = store.load_stage2(recording_id, s2_version)
    observed = stage2.observed_trace
    stage2c = (
        store.load_stage2c(recording_id, s2_version)
        if mode == Stage3Mode.evidence_aware
        else None
    )
    stage2c_sha = (
        _verified(store, recording_id, stage2c_filename(s2_version), unverified)
        if stage2c
        else None
    )
    if stage2c is not None and stage2c.stage2_sha256 not in (None, stage2_sha):
        raise ArtifactIntegrityError(
            f"Frozen Stage 2C for Stage 2 v{s2_version} was made from different Stage 2 "
            f"content (recorded {stage2c.stage2_sha256[:12]}…, now {stage2_sha[:12]}…)"
        )
    video = video_path or store.video_path(recording_id)
    if (
        mode == Stage3Mode.evidence_aware
        and video is None
        and stage2c is None
        and not stage2.evidence_frames
    ):
        raise MissingArtifactError(
            f"Stage 2C has no frames to search for {recording_id}: the video is "
            f"{_video_absence(store, recording_id)}, no evidence stills were saved for "
            f"Stage 2 v{s2_version}, and no Stage 2C batch is frozen for it"
        )
    video_sha = store.verify_video(recording_id, video) if video is not None else None
    frame_jpegs = (
        store.load_evidence_frames(recording_id, stage2.evidence_frames)
        if mode == Stage3Mode.evidence_aware and video is None and stage2.evidence_frames
        else None
    )
    capture = model_io_capture_from_env()
    inputs = Stage3Inputs(
        stage1_sha256=stage1_sha,
        stage2_version=s2_version,
        stage2_sha256=stage2_sha,
        stage2c_sha256=stage2c_sha,
        stage2c_source=(
            None
            if mode == Stage3Mode.deterministic
            else Stage2CSource.replayed if stage2c else Stage2CSource.live
        ),
        video_sha256=video_sha,
        unverified_artifacts=unverified,
    )
    started_at = utc_now()
    started_ms = monotonic_ms()
    evaluation = store.start_stage3(
        recording_id,
        mode=mode,
        inputs=inputs,
        metadata=_stage3_metadata(mode, started_at, capture),
    )
    client = (
        _recording_client(openai_client, capture)
        if mode == Stage3Mode.evidence_aware
        else None
    )

    def work() -> dict[str, Any]:
        if client is None:
            result = align(expected, observed)
            return {"deterministic_result": result, "result": result}
        manifest = stage2c.frame_manifest if stage2c else frame_manifest_for(observed)
        bundle = align_with_assist(
            expected,
            observed,
            openai_client=client,
            model=ALIGN_ASSIST_MODEL,
            on_retry=on_retry,
            evidence_aware=True,
            frame_evidence=True,
            frame_manifest=manifest,
            frame_video_path=video,
            frame_jpeg_by_id=frame_jpegs,
            stage2c_raw_batch=stage2c.raw_batch if stage2c else None,
        )
        replay = _replay_check(stage2c, bundle.assist.candidate_evidence) if stage2c else None
        if replay is not None:
            bundle = _mark_not_searched(bundle, replay.unmatched_targets)
        if stage2c is None:
            artifact = _stage2c_artifact(
                bundle,
                client.stage2c_batch,
                manifest,
                stage2_version=s2_version,
                stage2_sha256=stage2_sha,
                video_sha256=video_sha,
                stage3_version=evaluation.version,
                started_at=started_at,
            )
            if artifact is not None:
                try:
                    store.save_stage2c(recording_id, artifact)
                except FileExistsError:
                    logger.info("stage-2c already frozen by a concurrent Stage 3 run; keeping it")
        return {
            "deterministic_result": bundle.deterministic_result,
            "result": bundle.result,
            "assist": bundle.assist,
            "token_usage": token_usage_summary(client.requests),
            "stage2c_replay": replay,
        }

    return _execute(
        lambda record: store.finish_stage3(recording_id, record),
        lambda record: store.preserve_unfinished(recording_id, STAGE3_DIRNAME, record),
        evaluation,
        started_ms,
        client,
        work,
    )


def run_stage4(
    store: RecordingStore,
    recording_id: str,
    stage3_version: int,
    *,
    latency: LatencyMetrics | None = None,
    openai_client: Any | None = None,
    video_path: Path | None = None,
) -> Stage4Evaluation:
    """Cross-check the exact Stage 3 evaluation ``stage3_version``; new version per call.

    Uses the Stage 2 version that Stage 3 evaluation consumed, and verifies the Stage 3
    record, its inputs, and the video before running.
    """
    stage3 = store.load_stage3(recording_id, stage3_version)
    if stage3.status != RunStatus.completed or stage3.result is None:
        raise ValueError(
            f"Stage 3 v{stage3_version} is {stage3.status.value}; "
            "only completed evaluations can be cross-checked"
        )
    unverified: list[str] = []
    stage3_sha, stage3_verified = store.verify_version(recording_id, STAGE3_DIRNAME, stage3_version)
    if not stage3_verified:
        unverified.append(f"{STAGE3_DIRNAME}/v{stage3_version}")
    s2_version = stage3.inputs.stage2_version
    for filename, recorded in (
        (STAGE1_FILENAME, stage3.inputs.stage1_sha256),
        (stage2_filename(s2_version), stage3.inputs.stage2_sha256),
    ):
        digest = _verified(store, recording_id, filename, unverified)
        if digest != recorded:
            raise ArtifactIntegrityError(
                f"{filename} no longer matches what Stage 3 v{stage3_version} consumed "
                f"(recorded {recorded[:12]}…, now {digest[:12]}…)"
            )
    expected = store.load_stage1(recording_id).expected_trace
    observed = store.load_stage2(recording_id, s2_version).observed_trace
    video = video_path or store.video_path(recording_id)
    if video is None:
        raise MissingArtifactError(
            f"Stage 4 cannot run for {recording_id}: the video is "
            f"{_video_absence(store, recording_id)}. Stage 4 decodes evenly spaced frames "
            "from a window around each target, which saved evidence stills do not contain"
        )
    video_sha = store.verify_video(recording_id, video)

    skip_semantic: set[tuple[int, int]] | None = None
    skip_visual: set[int] | None = None
    if stage3.assist is not None:
        skip_semantic = reviewed_skip_pairs(stage3.assist)
        if stage3.assist.stage2c_skip_visual_steps:
            skip_visual = set(stage3.assist.stage2c_skip_visual_steps)

    capture = model_io_capture_from_env()
    started_at = utc_now()
    started_ms = monotonic_ms()
    evaluation = store.start_stage4(
        recording_id,
        stage3_version=stage3_version,
        stage3_sha256=stage3_sha,
        stage2_version=s2_version,
        video_sha256=video_sha,
        unverified_artifacts=unverified,
        skip_semantic_pairs=sorted(skip_semantic) if skip_semantic is not None else None,
        skip_visual_steps=sorted(skip_visual) if skip_visual is not None else None,
        metadata=build_metadata(
            started_at,
            models={"stage4_crosscheck": CROSSCHECK_MODEL},
            config={
                "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
                "frames_per_target": FRAMES_PER_TARGET,
                "window_pad_ms": WINDOW_PAD_MS,
                "model_io_capture": capture.value,
            },
        ),
    )
    client = _recording_client(openai_client, capture)
    audit = stage3.result

    def work() -> dict[str, Any]:
        with latency_scope(latency):
            report = run_cross_check(
                expected,
                observed,
                audit,
                video,
                openai_client=client,
                skip_semantic_pairs=skip_semantic,
                skip_visual_steps=skip_visual,
            )
        return {"report": report}

    return _execute(
        lambda record: store.finish_stage4(recording_id, record),
        lambda record: store.preserve_unfinished(recording_id, STAGE4_DIRNAME, record),
        evaluation,
        started_ms,
        client,
        work,
    )
