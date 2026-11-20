"""Filesystem store: one permanent folder per recording, versioned evaluations.

Layout::

    <root>/<recording_id>/
        recording.json              RecordingManifest
        recording.<ext>             the uploaded video (absent when retention is "discard")
        stage-1.json                Stage1Artifact   (write-once)
        stage-2.json                Stage2Artifact v1 (write-once)
        stage-2.v2.json …           Stage 2 retries   (write-once, one per attempt)
        stage-2c.json               Stage2CArtifact for Stage 2 v1 (write-once)
        stage-2c.v2.json …          Stage2CArtifact for Stage 2 v2 …
        evidence/<artifact>/F1.jpg  decoded evidence stills (write-once)
        stage2-attempts/<id>.json   Stage2Attempt, one per extraction attempt (incl. failures)
        stage3/0001.json …          Stage3Evaluation (one file per run, never overwritten)
        stage4/0001.json …          Stage4Evaluation (one file per run, never overwritten)
        .lock                       per-recording lock for read-check-write updates

Every JSON file carries its own SHA-256 in a top-level ``integrity`` field, recomputed on
each write (see ``integrity.py``); ``verify_*`` checks it before reuse. Files saved before
embedded checksums fall back to their ``<name>.sha256`` sidecar; none are written now.
Evidence stills are checked against the ``sha256`` stored in their Stage 2 artifact.

New files are published with exclusive creates, so concurrent writers never replace
each other. Updates that read a record and rewrite it (the manifest, finishing a run,
stale-run recovery) hold the recording's ``.lock`` (``fcntl.flock``; an in-process lock
where ``fcntl`` is unavailable) and raise RecordingBusyError if it can't be taken.
"""

import hashlib
import logging
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import IO, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

try:
    import fcntl
except ImportError:  # Windows: fall back to an in-process lock
    fcntl = None  # type: ignore[assignment]

from reckon.persistence.files import (
    ArtifactIntegrityError,
    atomic_create,
    file_sha256,
    temp_sibling,
)
from reckon.persistence.integrity import (
    create_sealed,
    json_artifact_sha256,
    replace_sealed,
    verify_json_artifact,
)
from reckon.persistence.schema import (
    EvidenceFrame,
    ExecutionMetadata,
    RecordingManifest,
    RunOwner,
    RunStatus,
    Stage1Artifact,
    Stage2Artifact,
    Stage2Attempt,
    Stage2CArtifact,
    Stage3Evaluation,
    Stage3Inputs,
    Stage3Mode,
    Stage4Evaluation,
    VideoRetention,
)

logger = logging.getLogger(__name__)

RECORDINGS_DIR_ENV = "RECKON_RECORDINGS_DIR"
DEFAULT_RECORDINGS_DIR = "recordings"
VIDEO_RETENTION_ENV = "RECKON_VIDEO_RETENTION"
STALE_RUN_SECONDS_ENV = "RECKON_STALE_RUN_SECONDS"
DEFAULT_STALE_RUN_SECONDS = 6 * 60 * 60

MANIFEST_FILENAME = "recording.json"
STAGE1_FILENAME = "stage-1.json"
STAGE2_FILENAME = "stage-2.json"
STAGE2C_FILENAME = "stage-2c.json"
STAGE3_DIRNAME = "stage3"
STAGE4_DIRNAME = "stage4"
STAGE2_ATTEMPTS_DIRNAME = "stage2-attempts"
LOCK_FILENAME = ".lock"
UNSAVED_SUFFIX = ".unsaved.json"
DEFAULT_LOCK_TIMEOUT_S = 10.0
LOCK_POLL_S = 0.05
EVIDENCE_DIRNAME = "evidence"
VIDEO_STEM = "recording"
DEFAULT_VIDEO_SUFFIX = ".mp4"

VERSION_WIDTH = 4
MAX_RECORDING_ID_LENGTH = 80

_RECORDING_ID_RE = re.compile(r"^[\w][\w-]*$")
_VERSION_RE = re.compile(r"^\d+$")
_STAGE2_RE = re.compile(r"^stage-2(?:\.v(\d+))?\.json$")
_FRAME_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ATTEMPT_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")

RecordT = TypeVar("RecordT", bound=BaseModel)
EvaluationT = TypeVar("EvaluationT", Stage3Evaluation, Stage4Evaluation)

_ACTIVE_RUN_TOKENS: set[str] = set()
_PROCESS_LOCKS: dict[Path, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class MissingArtifactError(FileNotFoundError):
    """A stage artifact or evaluation required for the requested run is absent."""


class RecordingBusyError(RuntimeError):
    """Another operation held the recording's lock for longer than the timeout."""


def sanitize_heading(value: str) -> str:
    """Filesystem-safe folder name; same character policy as the JSON export name."""
    cleaned = re.sub(r"[^\w-]+", "_", value.strip()).strip("_-")
    return cleaned[:MAX_RECORDING_ID_LENGTH].rstrip("_-")


def stage2_filename(version: int) -> str:
    return STAGE2_FILENAME if version == 1 else f"stage-2.v{version}.json"


def stage2c_filename(stage2_version: int) -> str:
    return STAGE2C_FILENAME if stage2_version == 1 else f"stage-2c.v{stage2_version}.json"


def _version_filename(version: int) -> str:
    return f"{version:0{VERSION_WIDTH}d}.json"


def _unsaved_filename(version: int) -> str:
    return f"{version:0{VERSION_WIDTH}d}{UNSAVED_SUFFIX}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _new_owner() -> RunOwner:
    owner = RunOwner(token=uuid4().hex, pid=os.getpid(), host=socket.gethostname())
    _ACTIVE_RUN_TOKENS.add(owner.token)
    return owner


def _stale_reason(
    record: Stage2Attempt | Stage3Evaluation | Stage4Evaluation, stale_after: timedelta
) -> str | None:
    owner = record.owner
    if owner is not None and owner.host == socket.gethostname():
        if owner.pid == os.getpid() and owner.token not in _ACTIVE_RUN_TOKENS:
            return "interrupted: run is no longer active in its process"
        if owner.pid != os.getpid() and not _pid_alive(owner.pid):
            return f"interrupted: owning process {owner.pid} is no longer running"
    if datetime.now(timezone.utc) - record.metadata.started_at > stale_after:
        return f"interrupted: still running after {int(stale_after.total_seconds())}s"
    return None


class RecordingStore:
    """All reads and writes of persisted recordings go through this class."""

    def __init__(
        self,
        root: Path,
        *,
        video_retention: VideoRetention = VideoRetention.keep,
        stale_after_s: int = DEFAULT_STALE_RUN_SECONDS,
        lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    ) -> None:
        self.root = root
        self.video_retention = video_retention
        self.stale_after = timedelta(seconds=stale_after_s)
        self.lock_timeout_s = lock_timeout_s

    @classmethod
    def from_env(cls) -> "RecordingStore":
        return cls(
            Path(os.environ.get(RECORDINGS_DIR_ENV) or DEFAULT_RECORDINGS_DIR),
            video_retention=VideoRetention(
                os.environ.get(VIDEO_RETENTION_ENV) or VideoRetention.keep.value
            ),
            stale_after_s=int(
                os.environ.get(STALE_RUN_SECONDS_ENV) or DEFAULT_STALE_RUN_SECONDS
            ),
        )

    @contextmanager
    def recording_lock(self, recording_id: str) -> Iterator[None]:
        """Exclusive lock for one recording's read-check-write updates. Not re-entrant."""
        path = self.recording_dir(recording_id) / LOCK_FILENAME
        if not path.parent.is_dir():
            raise MissingArtifactError(f"Recording not found: {recording_id}")
        busy = RecordingBusyError(
            f"Recording {recording_id} is busy with another operation; try again shortly"
        )
        if fcntl is None:
            with _PROCESS_LOCKS_GUARD:
                lock = _PROCESS_LOCKS.setdefault(path.resolve(), threading.Lock())
            if not lock.acquire(timeout=self.lock_timeout_s):
                raise busy
            try:
                yield
            finally:
                lock.release()
            return
        deadline = time.monotonic() + self.lock_timeout_s
        with path.open("a") as handle:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise busy from None
                    time.sleep(LOCK_POLL_S)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # ── Recordings ──────────────────────────────────────────────────────────

    def recording_dir(self, recording_id: str) -> Path:
        if not _RECORDING_ID_RE.match(recording_id):
            raise ValueError(f"Invalid recording id: {recording_id!r}")
        return self.root / recording_id

    def create_recording(
        self,
        *,
        title: str,
        source_filename: str,
        scenario_text: str,
    ) -> RecordingManifest:
        """Create a new folder named after the scenario (or video) heading.

        An existing folder is never reused: collisions get ``-2``, ``-3``, … suffixes.
        """
        base = (
            sanitize_heading(title)
            or sanitize_heading(Path(source_filename).stem)
            or VIDEO_STEM
        )
        self.root.mkdir(parents=True, exist_ok=True)
        for attempt in count(1):
            recording_id = base if attempt == 1 else f"{base}-{attempt}"
            try:
                self.recording_dir(recording_id).mkdir()
            except FileExistsError:
                continue
            break
        manifest = RecordingManifest(
            recording_id=recording_id,
            title=title,
            source_filename=source_filename,
            scenario_text=scenario_text,
            created_at=datetime.now(timezone.utc),
        )
        create_sealed(self.recording_dir(recording_id) / MANIFEST_FILENAME, manifest)
        return manifest

    def load_manifest(self, recording_id: str) -> RecordingManifest:
        path = self.recording_dir(recording_id) / MANIFEST_FILENAME
        if not path.is_file():
            raise MissingArtifactError(f"Recording not found: {recording_id}")
        verify_json_artifact(path)
        return RecordingManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list_recordings(self) -> list[RecordingManifest]:
        """Valid recordings, newest first. Folders without a readable manifest are skipped."""
        if not self.root.is_dir():
            return []
        manifests: list[RecordingManifest] = []
        for child in self.root.iterdir():
            if not child.is_dir() or not _RECORDING_ID_RE.match(child.name):
                continue
            try:
                manifests.append(self.load_manifest(child.name))
            except (MissingArtifactError, ArtifactIntegrityError, ValidationError) as exc:
                logger.warning("skipping recording folder %s: %s", child.name, exc)
        return sorted(manifests, key=lambda m: m.created_at, reverse=True)

    def _record_video(
        self, recording_id: str, publish: Callable[[], dict[str, object]]
    ) -> RecordingManifest:
        """Under the lock: refuse if a video is already recorded, publish, update manifest."""
        with self.recording_lock(recording_id):
            manifest = self.load_manifest(recording_id)
            if manifest.video_sha256 is not None:
                raise FileExistsError(f"Video already recorded for {recording_id}")
            updated = manifest.model_copy(update=publish())
            replace_sealed(self.recording_dir(recording_id) / MANIFEST_FILENAME, updated)
            return updated

    def write_video(
        self,
        recording_id: str,
        write: Callable[[IO[bytes]], None],
        *,
        suffix: str = DEFAULT_VIDEO_SUFFIX,
    ) -> Path:
        """Store the recording's video once and record its checksum in the manifest."""
        if self.load_manifest(recording_id).video_sha256 is not None:
            raise FileExistsError(f"Video already recorded for {recording_id}")
        clean_suffix = suffix if re.fullmatch(r"\.\w+", suffix or "") else DEFAULT_VIDEO_SUFFIX
        target = self.recording_dir(recording_id) / f"{VIDEO_STEM}{clean_suffix.lower()}"
        tmp = temp_sibling(target)
        try:
            with tmp.open("wb") as handle:
                write(handle)
                handle.flush()
                os.fsync(handle.fileno())

            def publish() -> dict[str, object]:
                os.link(tmp, target)
                return {
                    "video_filename": target.name,
                    "video_sha256": file_sha256(target),
                    "video_size_bytes": target.stat().st_size,
                    "video_retained": True,
                }

            self._record_video(recording_id, publish)
        finally:
            tmp.unlink(missing_ok=True)
        return target

    def record_unretained_video(self, recording_id: str, video_path: Path) -> str:
        """Retention "discard": keep only the checksum and size of a video stored elsewhere."""
        digest = file_sha256(video_path)
        size = video_path.stat().st_size
        self._record_video(
            recording_id,
            lambda: {
                "video_filename": None,
                "video_sha256": digest,
                "video_size_bytes": size,
                "video_retained": False,
            },
        )
        return digest

    def video_path(self, recording_id: str) -> Path | None:
        manifest = self.load_manifest(recording_id)
        if manifest.video_filename is None:
            return None
        path = self.recording_dir(recording_id) / manifest.video_filename
        return path if path.is_file() else None

    def verify_video(self, recording_id: str, video_path: Path) -> str:
        """Checksum ``video_path`` against the manifest; ArtifactIntegrityError on mismatch."""
        recorded = self.load_manifest(recording_id).video_sha256
        digest = file_sha256(video_path)
        if recorded is not None and recorded != digest:
            raise ArtifactIntegrityError(
                f"Video for {recording_id} does not match the recorded checksum "
                f"(recorded {recorded[:12]}…, got {digest[:12]}…)"
            )
        return digest

    # ── Write-once stage artifacts ──────────────────────────────────────────

    def _artifact_path(self, recording_id: str, filename: str) -> Path:
        return self.recording_dir(recording_id) / filename

    def _save_artifact(self, recording_id: str, filename: str, record: BaseModel) -> Path:
        self.load_manifest(recording_id)
        path = self._artifact_path(recording_id, filename)
        create_sealed(path, record)
        return path

    def _load_artifact(
        self, recording_id: str, filename: str, model: type[RecordT]
    ) -> RecordT:
        path = self._artifact_path(recording_id, filename)
        if not path.is_file():
            raise MissingArtifactError(f"{filename} is missing for recording {recording_id}")
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    def has_artifact(self, recording_id: str, filename: str) -> bool:
        return self._artifact_path(recording_id, filename).is_file()

    def artifact_sha256(self, recording_id: str, filename: str) -> str:
        path = self._artifact_path(recording_id, filename)
        if not path.is_file():
            raise MissingArtifactError(f"{filename} is missing for recording {recording_id}")
        return json_artifact_sha256(path)

    def verify_artifact(self, recording_id: str, filename: str) -> tuple[str, bool]:
        """(sha256, verified); raises ArtifactIntegrityError if the file was modified."""
        path = self._artifact_path(recording_id, filename)
        if not path.is_file():
            raise MissingArtifactError(f"{filename} is missing for recording {recording_id}")
        return verify_json_artifact(path)

    def save_stage1(self, recording_id: str, artifact: Stage1Artifact) -> Path:
        return self._save_artifact(recording_id, STAGE1_FILENAME, artifact)

    def load_stage1(self, recording_id: str) -> Stage1Artifact:
        return self._load_artifact(recording_id, STAGE1_FILENAME, Stage1Artifact)

    def stage2_versions(self, recording_id: str) -> list[int]:
        directory = self.recording_dir(recording_id)
        if not directory.is_dir():
            return []
        versions = []
        for path in directory.iterdir():
            match = _STAGE2_RE.match(path.name)
            if match:
                versions.append(int(match.group(1) or 1))
        return sorted(versions)

    def latest_stage2_version(self, recording_id: str) -> int:
        versions = self.stage2_versions(recording_id)
        if not versions:
            raise MissingArtifactError(f"{STAGE2_FILENAME} is missing for recording {recording_id}")
        return versions[-1]

    def save_stage2(self, recording_id: str, artifact: Stage2Artifact) -> Stage2Artifact:
        """Save as the next Stage 2 version; earlier versions are never touched."""
        self.load_manifest(recording_id)
        versions = self.stage2_versions(recording_id)
        version = (versions[-1] if versions else 0) + 1
        while True:
            record = artifact.model_copy(update={"version": version})
            try:
                create_sealed(self._artifact_path(recording_id, stage2_filename(version)), record)
            except FileExistsError:
                version += 1
                continue
            return record

    def load_stage2(self, recording_id: str, version: int | None = None) -> Stage2Artifact:
        resolved = version if version is not None else self.latest_stage2_version(recording_id)
        return self._load_artifact(recording_id, stage2_filename(resolved), Stage2Artifact)

    def save_stage2c(self, recording_id: str, artifact: Stage2CArtifact) -> Path:
        return self._save_artifact(
            recording_id, stage2c_filename(artifact.stage2_version), artifact
        )

    def load_stage2c(self, recording_id: str, stage2_version: int = 1) -> Stage2CArtifact | None:
        filename = stage2c_filename(stage2_version)
        if not self.has_artifact(recording_id, filename):
            return None
        return self._load_artifact(recording_id, filename, Stage2CArtifact)

    # ── Evidence stills ─────────────────────────────────────────────────────

    def save_evidence_frames(
        self,
        recording_id: str,
        group: str,
        frames: list[tuple[str, int, bytes]],
    ) -> list[EvidenceFrame]:
        """Write ``(frame_id, timestamp_ms, jpeg)`` stills under ``evidence/<group>/``."""
        directory = self.recording_dir(recording_id) / EVIDENCE_DIRNAME / sanitize_heading(group)
        directory.mkdir(parents=True, exist_ok=True)
        saved: list[EvidenceFrame] = []
        for frame_id, timestamp_ms, jpeg in frames:
            if not _FRAME_ID_RE.match(frame_id):
                raise ValueError(f"Invalid frame id: {frame_id!r}")
            path = directory / f"{frame_id}.jpg"
            atomic_create(path, jpeg)
            saved.append(
                EvidenceFrame(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    filename=path.relative_to(self.recording_dir(recording_id)).as_posix(),
                    sha256=hashlib.sha256(jpeg).hexdigest(),
                )
            )
        return saved

    def load_evidence_frames(
        self, recording_id: str, frames: list[EvidenceFrame]
    ) -> dict[str, bytes]:
        """JPEG bytes by frame id, each checked against the checksum stored in the artifact."""
        out: dict[str, bytes] = {}
        root = self.recording_dir(recording_id).resolve()
        for frame in frames:
            path = (root / frame.filename).resolve()
            if root not in path.parents or not path.is_file():
                raise MissingArtifactError(f"Evidence frame {frame.filename} is missing")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != frame.sha256:
                raise ArtifactIntegrityError(f"Evidence frame {frame.filename} was modified")
            out[frame.frame_id] = data
        return out

    # ── Versioned evaluations ───────────────────────────────────────────────

    def _versions_dir(self, recording_id: str, dirname: str) -> Path:
        return self.recording_dir(recording_id) / dirname

    def _version_path(self, recording_id: str, dirname: str, version: int) -> Path:
        return self._versions_dir(recording_id, dirname) / _version_filename(version)

    def _existing_versions(self, directory: Path) -> list[int]:
        if not directory.is_dir():
            return []
        return sorted(
            int(path.stem)
            for path in directory.glob("*.json")
            if _VERSION_RE.match(path.stem)
        )

    def _reserve(
        self,
        recording_id: str,
        dirname: str,
        build: Callable[[int], RecordT],
    ) -> RecordT:
        """Claim the next version number with an exclusive create; never reuses one."""
        self.load_manifest(recording_id)
        directory = self._versions_dir(recording_id, dirname)
        directory.mkdir(exist_ok=True)
        versions = self._existing_versions(directory)
        version = (versions[-1] if versions else 0) + 1
        while True:
            record = build(version)
            try:
                create_sealed(directory / _version_filename(version), record)
            except FileExistsError:
                version += 1
                continue
            return record

    def _finish(self, recording_id: str, dirname: str, record: EvaluationT) -> None:
        """Replace this run's own placeholder; completed and failed runs are immutable.

        A ``running`` placeholder may be replaced by anyone finishing it. An
        ``interrupted`` one only by its own owner (it was marked stale while still alive).
        """
        path = self._version_path(recording_id, dirname, record.version)
        self._replace_owned(recording_id, path, record, f"{dirname} v{record.version}")

    def _replace_owned(
        self,
        recording_id: str,
        path: Path,
        record: Stage2Attempt | Stage3Evaluation | Stage4Evaluation,
        label: str,
    ) -> None:
        try:
            with self.recording_lock(recording_id):
                if not path.is_file():
                    raise MissingArtifactError(f"{label} was never started")
                current = type(record).model_validate_json(path.read_text(encoding="utf-8"))
                same_owner = (
                    current.owner is not None
                    and record.owner is not None
                    and current.owner.token == record.owner.token
                )
                replaceable = current.status == RunStatus.running or (
                    current.status == RunStatus.interrupted and same_owner
                )
                if not replaceable:
                    raise FileExistsError(f"{label} is already {current.status.value}")
                replace_sealed(path, record)
        finally:
            # A run whose final write failed can never finish, so stale recovery must see it.
            if record.owner is not None and record.status != RunStatus.running:
                _ACTIVE_RUN_TOKENS.discard(record.owner.token)

    def preserve_unfinished(self, recording_id: str, dirname: str, record: EvaluationT) -> Path:
        """Keep a run's final record when writing it over the placeholder failed.

        Needs no lock: the name belongs to this run's reserved version. Stale recovery
        copies its model calls into the interrupted placeholder.
        """
        path = self._versions_dir(recording_id, dirname) / _unsaved_filename(record.version)
        create_sealed(path, record)
        return path

    def _load_version(
        self, recording_id: str, dirname: str, version: int, model: type[RecordT]
    ) -> RecordT:
        path = self._version_path(recording_id, dirname, version)
        if not path.is_file():
            raise MissingArtifactError(
                f"{dirname} v{version} not found for recording {recording_id}"
            )
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    def _list_versions(
        self, recording_id: str, dirname: str, model: type[RecordT]
    ) -> list[RecordT]:
        """All readable runs, oldest first. Unreadable files are logged and skipped."""
        records: list[RecordT] = []
        directory = self._versions_dir(recording_id, dirname)
        for version in self._existing_versions(directory):
            try:
                records.append(self._load_version(recording_id, dirname, version, model))
            except (ValidationError, ValueError) as exc:
                logger.warning("skipping unreadable %s v%d: %s", dirname, version, exc)
        return records

    def version_sha256(self, recording_id: str, dirname: str, version: int) -> str:
        path = self._version_path(recording_id, dirname, version)
        if not path.is_file():
            raise MissingArtifactError(f"{dirname} v{version} not found")
        return json_artifact_sha256(path)

    def verify_version(self, recording_id: str, dirname: str, version: int) -> tuple[str, bool]:
        path = self._version_path(recording_id, dirname, version)
        if not path.is_file():
            raise MissingArtifactError(f"{dirname} v{version} not found")
        return verify_json_artifact(path)

    def recover_stale_runs(self, recording_id: str) -> list[tuple[str, int]]:
        """Mark ``running`` records whose execution can no longer finish as ``interrupted``.

        Version numbers stay reserved; any partial payload in the record is kept.
        """
        recovered: list[tuple[str, int | str]] = []
        with self.recording_lock(recording_id):
            candidates: list[tuple[str, int | str, Path, Stage2Attempt | Stage3Evaluation | Stage4Evaluation]] = [
                (dirname, record.version, self._version_path(recording_id, dirname, record.version), record)
                for dirname, model in (
                    (STAGE3_DIRNAME, Stage3Evaluation),
                    (STAGE4_DIRNAME, Stage4Evaluation),
                )
                for record in self._list_versions(recording_id, dirname, model)
            ]
            candidates += [
                (STAGE2_ATTEMPTS_DIRNAME, attempt.attempt_id, self._attempt_path(recording_id, attempt.attempt_id), attempt)
                for attempt in self.list_stage2_attempts(recording_id)
            ]
            for dirname, key, path, record in candidates:
                if record.status != RunStatus.running:
                    continue
                reason = _stale_reason(record, self.stale_after)
                if reason is None:
                    continue
                update: dict[str, object] = {"status": RunStatus.interrupted, "error": reason}
                unsaved = path.with_name(path.name.removesuffix(".json") + UNSAVED_SUFFIX)
                if isinstance(record, Stage3Evaluation | Stage4Evaluation) and unsaved.is_file():
                    preserved = type(record).model_validate_json(unsaved.read_text(encoding="utf-8"))
                    update |= {
                        "model_calls": preserved.model_calls,
                        "error": f"{reason}; its final write failed, last record kept in {unsaved.name}",
                    }
                replace_sealed(path, record.model_copy(update=update))
                logger.warning("%s %s marked interrupted: %s", dirname, key, reason)
                recovered.append((dirname, key))
        return recovered

    # ── Stage 2 attempts ────────────────────────────────────────────────────

    def _attempt_path(self, recording_id: str, attempt_id: str) -> Path:
        if not _ATTEMPT_ID_RE.match(attempt_id):
            raise ValueError(f"Invalid attempt id: {attempt_id!r}")
        return self._versions_dir(recording_id, STAGE2_ATTEMPTS_DIRNAME) / f"{attempt_id}.json"

    def start_stage2_attempt(
        self,
        recording_id: str,
        *,
        video_sha256: str,
        stage1_sha256: str | None,
        metadata: ExecutionMetadata,
    ) -> Stage2Attempt:
        self.load_manifest(recording_id)
        attempt = Stage2Attempt(
            attempt_id=f"{metadata.started_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
            status=RunStatus.running,
            owner=_new_owner(),
            metadata=metadata,
            video_sha256=video_sha256,
            stage1_sha256=stage1_sha256,
        )
        path = self._attempt_path(recording_id, attempt.attempt_id)
        path.parent.mkdir(exist_ok=True)
        create_sealed(path, attempt)
        return attempt

    def finish_stage2_attempt(self, recording_id: str, attempt: Stage2Attempt) -> None:
        path = self._attempt_path(recording_id, attempt.attempt_id)
        self._replace_owned(recording_id, path, attempt, f"Stage 2 attempt {attempt.attempt_id}")

    def list_stage2_attempts(self, recording_id: str) -> list[Stage2Attempt]:
        """All readable attempts, oldest first."""
        directory = self._versions_dir(recording_id, STAGE2_ATTEMPTS_DIRNAME)
        if not directory.is_dir():
            return []
        attempts: list[Stage2Attempt] = []
        for path in sorted(directory.glob("*.json")):
            if not _ATTEMPT_ID_RE.match(path.stem):
                continue
            try:
                attempts.append(Stage2Attempt.model_validate_json(path.read_text(encoding="utf-8")))
            except ValidationError as exc:
                logger.warning("skipping unreadable Stage 2 attempt %s: %s", path.name, exc)
        return sorted(attempts, key=lambda attempt: (attempt.metadata.started_at, attempt.attempt_id))

    def start_stage3(
        self,
        recording_id: str,
        *,
        mode: Stage3Mode,
        inputs: Stage3Inputs,
        metadata: ExecutionMetadata,
    ) -> Stage3Evaluation:
        owner = _new_owner()
        return self._reserve(
            recording_id,
            STAGE3_DIRNAME,
            lambda version: Stage3Evaluation(
                version=version,
                status=RunStatus.running,
                mode=mode,
                owner=owner,
                inputs=inputs,
                metadata=metadata,
            ),
        )

    def finish_stage3(self, recording_id: str, evaluation: Stage3Evaluation) -> None:
        self._finish(recording_id, STAGE3_DIRNAME, evaluation)

    def load_stage3(self, recording_id: str, version: int) -> Stage3Evaluation:
        return self._load_version(recording_id, STAGE3_DIRNAME, version, Stage3Evaluation)

    def list_stage3(self, recording_id: str) -> list[Stage3Evaluation]:
        return self._list_versions(recording_id, STAGE3_DIRNAME, Stage3Evaluation)

    def stage3_sha256(self, recording_id: str, version: int) -> str:
        return self.version_sha256(recording_id, STAGE3_DIRNAME, version)

    def start_stage4(
        self,
        recording_id: str,
        *,
        stage3_version: int,
        stage3_sha256: str,
        stage2_version: int,
        video_sha256: str | None,
        unverified_artifacts: list[str],
        skip_semantic_pairs: list[tuple[int, int]] | None,
        skip_visual_steps: list[int] | None,
        metadata: ExecutionMetadata,
    ) -> Stage4Evaluation:
        owner = _new_owner()
        return self._reserve(
            recording_id,
            STAGE4_DIRNAME,
            lambda version: Stage4Evaluation(
                version=version,
                status=RunStatus.running,
                owner=owner,
                stage3_version=stage3_version,
                stage3_sha256=stage3_sha256,
                stage2_version=stage2_version,
                video_sha256=video_sha256,
                unverified_artifacts=unverified_artifacts,
                skip_semantic_pairs=skip_semantic_pairs,
                skip_visual_steps=skip_visual_steps,
                metadata=metadata,
            ),
        )

    def finish_stage4(self, recording_id: str, evaluation: Stage4Evaluation) -> None:
        self._finish(recording_id, STAGE4_DIRNAME, evaluation)

    def load_stage4(self, recording_id: str, version: int) -> Stage4Evaluation:
        return self._load_version(recording_id, STAGE4_DIRNAME, version, Stage4Evaluation)

    def list_stage4(self, recording_id: str) -> list[Stage4Evaluation]:
        return self._list_versions(recording_id, STAGE4_DIRNAME, Stage4Evaluation)
