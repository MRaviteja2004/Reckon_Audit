"""Persist Stage 1 / Stage 2 outputs and retry Stage 2 from a saved recording.

Extraction itself is ``extract_observed_trace``, unchanged; this module only supplies
the saved video, records metadata, and saves each attempt as a new Stage 2 version.
"""

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from reckon import config
from reckon.extractor import EXTRACTION_MODEL, make_client
from reckon.latency import LatencyMetrics, elapsed_ms, latency_scope, monotonic_ms
from reckon.parser import OPENAI_MODEL
from reckon.persistence.metadata import build_metadata, utc_now
from reckon.persistence.redact import redact_secrets
from reckon.persistence.schema import (
    EvidenceFrame,
    RunStatus,
    Stage1Artifact,
    Stage2Artifact,
    VideoRetention,
)
from reckon.persistence.store import (
    STAGE1_FILENAME,
    MissingArtifactError,
    RecordingStore,
)
from reckon.retry import RetryCallback
from reckon.sampler import extract_observed_trace
from reckon.schema import ExpectedTrace, FrameManifest, ObservedTrace
from reckon.stage2c import build_frame_manifest
from reckon.stage2c.discover import load_frame_jpegs


def frame_manifest_for(observed: ObservedTrace) -> FrameManifest | None:
    """The frame manifest Stage 2C uses: one labelled frame per sampled timestamp.

    Observed timestamps come first; gap-coverage frames are added so Stage 2C can search
    a quiet stretch even when extraction reported nothing there. Without gap coverage
    the manifest is exactly the observed timestamps, as before.
    """
    timestamps = sorted(
        {action.timestamp_ms for action in observed.actions}
        | set(observed.gap_sample_timestamps_ms)
    )
    return build_frame_manifest(timestamps) if timestamps else None


def build_stage1_artifact(
    expected: ExpectedTrace, *, started_at: datetime, duration_ms: float
) -> Stage1Artifact:
    return Stage1Artifact(
        metadata=build_metadata(
            started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms,
            models={"stage1_parser": OPENAI_MODEL},
            config={},
        ),
        expected_trace=expected,
    )


def build_stage2_artifact(
    observed: ObservedTrace,
    *,
    started_at: datetime,
    duration_ms: float,
    latency: LatencyMetrics | None = None,
    video_sha256: str | None = None,
    evidence_frames: list[EvidenceFrame] | None = None,
) -> Stage2Artifact:
    metrics: dict[str, float | int | None] = {}
    if latency is not None:
        metrics = {
            "stage2a_ms": latency.stage2a_ms,
            "stage2b_ms": latency.stage2b_ms,
            "window_count": latency.window_count,
            "frame_count": latency.frame_count,
            "retry_count": latency.retry_count,
        }
    return Stage2Artifact(
        metadata=build_metadata(
            started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms,
            models={"stage2_extraction": EXTRACTION_MODEL},
            config={
                "stage2_event_selection_policy": config.STAGE2_EVENT_SELECTION_POLICY,
                "stage2_adaptive_sampling_policy": config.STAGE2_ADAPTIVE_SAMPLING_POLICY,
                "stage2_adaptive_sample_interval_ms": config.STAGE2_ADAPTIVE_SAMPLE_INTERVAL_MS,
                "stage2_adaptive_max_extra_frames": config.STAGE2_ADAPTIVE_MAX_EXTRA_FRAMES,
                "stage2_adaptive_max_frames_per_window": (
                    config.STAGE2_ADAPTIVE_MAX_FRAMES_PER_WINDOW
                ),
                "stage2_adaptive_interaction_window_ms": (
                    config.STAGE2_ADAPTIVE_INTERACTION_WINDOW_MS
                ),
                "stage2_gap_sampling_policy": config.STAGE2_GAP_SAMPLING_POLICY,
                "stage2_gap_min_ms": config.STAGE2_GAP_MIN_MS,
                "stage2_gap_max_extra_frames": config.STAGE2_GAP_MAX_EXTRA_FRAMES,
            },
            metrics=metrics,
        ),
        video_sha256=video_sha256,
        observed_trace=observed,
        evidence_frames=evidence_frames or [],
    )


def snapshot_evidence_frames(
    store: RecordingStore,
    recording_id: str,
    observed: ObservedTrace,
    video_path: Path,
    *,
    group: str,
) -> list[EvidenceFrame]:
    """Decode and save the Stage 2C manifest frames when the video itself won't be kept.

    Uses the same decoder call Stage 2C uses, so a later Stage 2C run can be fed these
    stills instead of the video. No model call.
    """
    if store.video_retention != VideoRetention.discard:
        return []
    manifest = frame_manifest_for(observed)
    if manifest is None:
        return []
    jpegs = load_frame_jpegs(video_path, manifest)
    return store.save_evidence_frames(
        recording_id,
        group,
        [(entry.frame_id, entry.timestamp_ms, jpegs[entry.frame_id]) for entry in manifest.frames],
    )


def _error_text(exc: BaseException) -> str:
    detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return redact_secrets(detail or type(exc).__name__)


def run_stage2(
    store: RecordingStore,
    recording_id: str,
    *,
    extraction_client: Any | None = None,
    video_path: Path | None = None,
    on_retry: RetryCallback | None = None,
    latency: LatencyMetrics | None = None,
) -> Stage2Artifact:
    """Extract from the saved (or supplied) video and save a new Stage 2 version.

    Stage 1 is not needed and not re-run. Earlier Stage 2 versions are preserved. Every
    attempt is recorded under ``stage2-attempts/``; a failed or interrupted attempt
    keeps its error and produces no ObservedTrace.
    """
    manifest = store.load_manifest(recording_id)
    video = video_path or store.video_path(recording_id)
    if video is None:
        raise MissingArtifactError(
            f"Recording video is not available for {recording_id}"
            + ("" if manifest.video_retained else " (retention policy discarded it)")
        )
    video_sha256 = store.verify_video(recording_id, video)
    stage1_sha256 = (
        store.verify_artifact(recording_id, STAGE1_FILENAME)[0]
        if store.has_artifact(recording_id, STAGE1_FILENAME)
        else None
    )
    started_at = utc_now()
    started_ms = monotonic_ms()
    attempt = store.start_stage2_attempt(
        recording_id,
        video_sha256=video_sha256,
        stage1_sha256=stage1_sha256,
        metadata=build_metadata(
            started_at,
            models={"stage2_extraction": EXTRACTION_MODEL},
            config={},
        ),
    )
    try:
        with latency_scope(latency):
            observed = extract_observed_trace(
                extraction_client or make_client(),
                video,
                source_file=manifest.source_filename,
                on_retry=on_retry,
            )
        duration_ms = elapsed_ms(started_ms)
        artifact = build_stage2_artifact(
            observed,
            started_at=started_at,
            duration_ms=duration_ms,
            latency=latency,
            video_sha256=video_sha256,
            evidence_frames=snapshot_evidence_frames(
                store, recording_id, observed, video, group=f"stage-2-{attempt.attempt_id}"
            ),
        ).model_copy(update={"attempt_id": attempt.attempt_id})
        saved = store.save_stage2(recording_id, artifact)
    except BaseException as exc:
        store.finish_stage2_attempt(
            recording_id,
            attempt.model_copy(
                update={
                    "status": (
                        RunStatus.failed if isinstance(exc, Exception) else RunStatus.interrupted
                    ),
                    "error": _error_text(exc),
                    "metadata": attempt.metadata.model_copy(
                        update={"completed_at": utc_now(), "duration_ms": elapsed_ms(started_ms)}
                    ),
                }
            ),
        )
        raise
    store.finish_stage2_attempt(
        recording_id,
        attempt.model_copy(
            update={
                "status": RunStatus.completed,
                "stage2_version": saved.version,
                "metadata": attempt.metadata.model_copy(
                    update={"completed_at": utc_now(), "duration_ms": elapsed_ms(started_ms)}
                ),
            }
        ),
    )
    return saved
