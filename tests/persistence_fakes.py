"""Shared model-gateway double and sample traces for persistence and app tests."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reckon.assist_evidence_propose import _EvidenceBatchResponse
from reckon.assist_propose import _AssistBatchResponse
from reckon.episode_verify_propose import EpisodeVerifyDecision, EpisodeVerifyVerdict
from reckon.persistence import ExecutionMetadata, Stage2CArtifact, frame_manifest_for
from reckon.persistence.integrity import INTEGRITY_FIELD, legacy_checksum_path
from reckon.schema import (
    ActionType,
    CandidateEvidenceItem,
    CandidateEvidenceTrace,
    CandidateEvidenceType,
    CandidateSearchStatus,
    CandidateValidationStatus,
    ExpectedStep,
    ExpectedTrace,
    ObservedAction,
    ObservedTrace,
)
from reckon.stage2c.propose import RawFrameEvidenceBatch, RawFrameEvidenceCandidate


def traces() -> tuple[ExpectedTrace, ObservedTrace]:
    """Stage 2 missed the login click and the home assertion; Stage 2C can recover the latter."""
    expected = ExpectedTrace(
        scenario_title="Login",
        steps=[
            ExpectedStep(
                index=1, action=ActionType.click, target="Login", raw_text="Click the Login button"
            ),
            ExpectedStep(
                index=2,
                action=ActionType.assert_state,
                target="home",
                assertion="You're offline",
                raw_text="Driver home screen is displayed",
            ),
        ],
    )
    observed = ObservedTrace(
        source_file="login.mp4",
        actions=[
            ObservedAction(index=1, timestamp_ms=500, action=ActionType.other, target="Splash", confidence=0.8, screen_description="Splash screen"),
            ObservedAction(index=2, timestamp_ms=2000, action=ActionType.other, target="Spinner", confidence=0.8, screen_description="Loading"),
            ObservedAction(index=3, timestamp_ms=3500, action=ActionType.other, target="Screen", confidence=0.8, screen_description="Unclear"),
        ],
    )
    return expected, observed


def raw_batch() -> RawFrameEvidenceBatch:
    return RawFrameEvidenceBatch(
        candidates=[
            RawFrameEvidenceCandidate(
                expected_step_id="E2",
                frame_ids=["F3"],
                evidence_type=CandidateEvidenceType.ASSERTION_OUTCOME,
                observed_description="Driver home",
                visible_text=["You're offline"],
                confidence=0.91,
                search_status=CandidateSearchStatus.candidate_found,
            )
        ]
    )


class FakeGateway:
    """Answers each Stage 3 model call by response_model; records which were requested."""

    def __init__(self, *, stage2c_error: BaseException | None = None) -> None:
        self.requested: list[type] = []
        self.stage2c_error = stage2c_error
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> Any:
        model = kwargs["response_model"]
        self.requested.append(model)
        if model is RawFrameEvidenceBatch:
            if self.stage2c_error is not None:
                raise self.stage2c_error
            return raw_batch()
        if model is EpisodeVerifyDecision:
            return EpisodeVerifyDecision(
                verdict=EpisodeVerifyVerdict.insufficient_evidence,
                confidence=0.1,
                rationale="not enough evidence",
            )
        if model is _EvidenceBatchResponse:
            return _EvidenceBatchResponse(decisions=[])
        return _AssistBatchResponse(decisions=[])


def fake_jpegs(_video_path: Path, manifest: Any) -> dict[str, bytes]:
    return {entry.frame_id: b"jpeg" for entry in manifest.frames}


def frozen_stage2c_for_other_step(observed: ObservedTrace) -> Stage2CArtifact:
    """A frozen batch that answered only E1, so a replay asking for E2 never searched it."""
    return Stage2CArtifact(
        metadata=ExecutionMetadata(
            started_at=datetime(2026, 1, 1, tzinfo=UTC), python_version="3.13.0"
        ),
        produced_by_stage3_version=1,
        frame_manifest=frame_manifest_for(observed),
        raw_batch=RawFrameEvidenceBatch(candidates=[]),
        candidate_evidence=CandidateEvidenceTrace(
            source_file="login.mp4",
            candidates=[
                CandidateEvidenceItem(
                    expected_step_id="E1",
                    expected_step_index=1,
                    evidence_type=CandidateEvidenceType.ACTION,
                    confidence=0.0,
                    search_status=CandidateSearchStatus.not_found,
                    validation_status=CandidateValidationStatus.rejected,
                    rejection_reason="not_found",
                )
            ],
        ),
    )


def strip_integrity(path: Path) -> None:
    """Rewrite ``path`` the way it looked before checksums existed: no embedded checksum."""
    payload = json.loads(path.read_text())
    payload.pop(INTEGRITY_FIELD)
    path.write_text(json.dumps(payload, indent=2))


def tamper_json(path: Path) -> None:
    """Change one value but keep the recorded checksum, as an edit after saving would."""
    payload = json.loads(path.read_text())
    payload["metadata"]["python_version"] = "tampered"
    path.write_text(json.dumps(payload, indent=2))


def make_legacy(recording_dir: Path) -> None:
    """Convert a recording to the sidecar layout: no embedded checksums, ``.sha256`` files.

    ``recording.json`` and ``running`` records never had sidecars, so they get none. Files
    already in the legacy layout are left untouched.
    """
    for path in recording_dir.rglob("*.json"):
        if INTEGRITY_FIELD not in json.loads(path.read_text()):
            continue
        strip_integrity(path)
        payload = json.loads(path.read_text())
        if path.name != "recording.json" and payload.get("status", "completed") != "running":
            legacy_checksum_path(path).write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
