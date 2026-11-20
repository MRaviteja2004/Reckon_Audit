"""Persisted records for recordings, stage artifacts, and versioned evaluations."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

from reckon.schema import (
    AssistReport,
    AuditResult,
    CandidateEvidenceTrace,
    CrossCheckReport,
    ExpectedTrace,
    FrameManifest,
    ObservedTrace,
)
from reckon.stage2c.propose import RawFrameEvidenceBatch

RECORD_SCHEMA_VERSION = 1

Sha256 = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]


class RunStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    interrupted = "interrupted"


class Stage3Mode(str, Enum):
    deterministic = "deterministic"
    evidence_aware = "evidence_aware"


class Stage2CSource(str, Enum):
    live = "live"
    replayed = "replayed"


class VideoRetention(str, Enum):
    keep = "keep"
    discard = "discard"


class ModelIOCapture(str, Enum):
    """How much of each model request/response is persisted. Images are always hashed only."""

    full = "full"
    hashes = "hashes"


def _check_status(status: RunStatus, payload: object, error: str | None, stage: str) -> None:
    if status == RunStatus.completed and payload is None:
        raise ValueError(f"completed {stage} evaluations require a result")
    if status in (RunStatus.failed, RunStatus.interrupted) and not (error or "").strip():
        raise ValueError(f"{status.value} {stage} evaluations require an error")


class ExecutionMetadata(BaseModel):
    """Who, what, and when for one stage execution. Never holds secrets."""

    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    git_commit: str | None = None
    git_dirty: bool | None = None
    reckon_version: str | None = None
    python_version: str
    models: dict[str, str] = Field(default_factory=dict)
    config: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    metrics: dict[str, float | int | None] = Field(default_factory=dict)


class RunOwner(BaseModel):
    """Identifies the process that reserved a version, for stale-run detection."""

    token: str = Field(min_length=1)
    pid: int = Field(ge=0)
    host: str


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ModelCallRecord(BaseModel):
    """One ``chat.completions.create`` attempt, as sent and as answered."""

    sequence: int = Field(ge=1)
    started_at: datetime
    duration_ms: float = Field(ge=0)
    requested_model: str | None = None
    response_model: str | None = None
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    system_prompt_sha256: str | None = None
    user_text_sha256: str | None = None
    user_text: str | None = None
    image_sha256: list[str] = Field(default_factory=list)
    served_model: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None
    response_sha256: str | None = None
    response: dict[str, Any] | None = None
    usage: TokenUsage | None = None
    error: str | None = None


class EvidenceFrame(BaseModel):
    """A decoded still saved beside the artifacts, so evidence survives without the video."""

    frame_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    filename: str = Field(min_length=1, description="Path relative to the recording folder")
    sha256: Sha256


class RecordingManifest(BaseModel):
    """Identity of one permanent recording folder."""

    schema_version: int = RECORD_SCHEMA_VERSION
    recording_id: str = Field(min_length=1, description="Folder name under the store root")
    title: str
    source_filename: str
    scenario_text: str
    created_at: datetime
    video_filename: str | None = None
    video_sha256: Sha256 | None = None
    video_size_bytes: int | None = Field(default=None, ge=0)
    video_retained: bool = Field(
        default=True, description="False when the retention policy kept only the checksum"
    )


class Stage1Artifact(BaseModel):
    schema_version: int = RECORD_SCHEMA_VERSION
    metadata: ExecutionMetadata
    expected_trace: ExpectedTrace


class Stage2Artifact(BaseModel):
    schema_version: int = RECORD_SCHEMA_VERSION
    version: int = Field(default=1, ge=1, description="Extraction version; set by the store")
    attempt_id: str | None = None
    metadata: ExecutionMetadata
    video_sha256: Sha256 | None = Field(default=None, description="Video this trace was extracted from")
    observed_trace: ObservedTrace
    evidence_frames: list[EvidenceFrame] = Field(default_factory=list)


class Stage2Attempt(BaseModel):
    """One Stage 2 extraction attempt. A failed attempt never produces an ObservedTrace."""

    schema_version: int = RECORD_SCHEMA_VERSION
    attempt_id: str = Field(min_length=1)
    status: RunStatus
    owner: RunOwner | None = None
    metadata: ExecutionMetadata
    video_sha256: Sha256
    stage1_version: int = Field(default=1, ge=1)
    stage1_sha256: Sha256 | None = None
    stage2_version: int | None = Field(
        default=None, ge=1, description="Stage 2 version saved by a completed attempt"
    )
    error: str | None = None

    @model_validator(mode="after")
    def status_matches_payload(self) -> "Stage2Attempt":
        _check_status(self.status, self.stage2_version, self.error, "Stage 2")
        return self


class Stage2CArtifact(BaseModel):
    """Frame evidence captured on the first successful Stage 2C run for one Stage 2 version.

    ``raw_batch`` is the model's frame-ID proposal exactly as returned; replaying it
    re-runs Python validation without another model call. ``candidate_evidence`` is the
    validated trace from that first run (timestamps, frame refs, accept/reject reasons).
    """

    schema_version: int = RECORD_SCHEMA_VERSION
    metadata: ExecutionMetadata
    stage2_version: int = Field(default=1, ge=1)
    stage2_sha256: Sha256 | None = None
    video_sha256: Sha256 | None = None
    produced_by_stage3_version: int = Field(ge=1)
    frame_manifest: FrameManifest
    raw_batch: RawFrameEvidenceBatch
    candidate_evidence: CandidateEvidenceTrace
    stage2c_skip_visual_steps: list[int] = Field(default_factory=list)


class Stage3Inputs(BaseModel):
    """The exact artifact versions and checksums a Stage 3 evaluation consumed.

    Each checksum is the artifact's embedded ``integrity.sha256``, or for files saved
    before embedded checksums, the SHA-256 of the file's bytes.
    """

    stage1_sha256: Sha256
    stage2_version: int = Field(default=1, ge=1)
    stage2_sha256: Sha256
    stage2c_sha256: Sha256 | None = None
    stage2c_source: Stage2CSource | None = None
    video_sha256: Sha256 | None = None
    unverified_artifacts: list[str] = Field(
        default_factory=list,
        description="Inputs saved with no checksum at all; read but not verifiable",
    )


class Stage2CReplayCheck(BaseModel):
    """Did this run search the same expected steps the frozen Stage 2C batch answered?

    Replay supplies frozen proposals only for the frozen targets. Targets selected now
    but absent from the frozen run were never searched: they are kept out of the
    evaluation's candidate trace, their proposals are ``skipped`` with reason
    ``stage2c_not_searched_frozen_replay``, and they stay eligible for Stage 4.
    """

    frozen_targets: list[str]
    current_targets: list[str]
    unmatched_targets: list[str]


class Stage3Evaluation(BaseModel):
    schema_version: int = RECORD_SCHEMA_VERSION
    version: int = Field(ge=1)
    status: RunStatus
    mode: Stage3Mode
    owner: RunOwner | None = None
    metadata: ExecutionMetadata
    inputs: Stage3Inputs
    deterministic_result: AuditResult | None = None
    result: AuditResult | None = Field(
        default=None, description="Stage 3 AuditResult handed to Stage 4"
    )
    assist: AssistReport | None = None
    token_usage: dict[str, Any] | None = None
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    stage2c_replay: Stage2CReplayCheck | None = None
    error: str | None = None

    @model_validator(mode="after")
    def status_matches_payload(self) -> "Stage3Evaluation":
        _check_status(self.status, self.result, self.error, "Stage 3")
        return self


class Stage4Evaluation(BaseModel):
    schema_version: int = RECORD_SCHEMA_VERSION
    version: int = Field(ge=1)
    status: RunStatus
    owner: RunOwner | None = None
    metadata: ExecutionMetadata
    stage3_version: int = Field(ge=1)
    stage3_sha256: Sha256
    stage2_version: int = Field(default=1, ge=1)
    video_sha256: Sha256 | None = None
    unverified_artifacts: list[str] = Field(default_factory=list)
    skip_semantic_pairs: list[tuple[int, int]] | None = None
    skip_visual_steps: list[int] | None = None
    report: CrossCheckReport | None = None
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def status_matches_payload(self) -> "Stage4Evaluation":
        _check_status(self.status, self.report, self.error, "Stage 4")
        return self
