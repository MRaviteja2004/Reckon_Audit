"""SHA-256 checksums embedded in JSON artifacts, with a fallback for legacy ``.sha256`` sidecars.

Every JSON artifact is written with a top-level ``integrity`` object holding the SHA-256 of
its canonical JSON (keys sorted, no whitespace, UTF-8) with ``integrity`` itself removed.
Verification re-parses the file and hashes it the same way, so formatting changes don't
matter but any change to a value does. Files written before embedded checksums keep their
old identity, the SHA-256 of their bytes, checked against the sidecar when one exists.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from reckon.persistence.files import (
    ArtifactIntegrityError,
    atomic_create,
    atomic_replace,
    file_sha256,
)

INTEGRITY_FIELD = "integrity"
LEGACY_CHECKSUM_SUFFIX = ".sha256"


def canonical_sha256(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != INTEGRITY_FIELD}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sealed_json(record: BaseModel) -> tuple[str, str]:
    """``record`` as JSON text carrying its own checksum; returns (text, sha256)."""
    payload = json.loads(record.model_dump_json(exclude={INTEGRITY_FIELD}))
    digest = canonical_sha256(payload)
    payload[INTEGRITY_FIELD] = {"sha256": digest}
    return json.dumps(payload, indent=2, ensure_ascii=False), digest


def create_sealed(path: Path, record: BaseModel) -> str:
    """Write-once ``path`` with an embedded checksum; FileExistsError if it exists."""
    text, digest = sealed_json(record)
    atomic_create(path, text)
    return digest


def replace_sealed(path: Path, record: BaseModel) -> str:
    """Atomically rewrite ``path``; the checksum is recomputed for the new content."""
    text, digest = sealed_json(record)
    atomic_replace(path, text)
    return digest


def legacy_checksum_path(path: Path) -> Path:
    return path.with_name(path.name + LEGACY_CHECKSUM_SUFFIX)


def _read_payload(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"{path.name} is not readable JSON: {exc}") from exc


def _embedded_sha256(path: Path, payload: Any) -> str | None:
    if not isinstance(payload, dict) or INTEGRITY_FIELD not in payload:
        return None
    integrity = payload[INTEGRITY_FIELD]
    recorded = integrity.get("sha256") if isinstance(integrity, dict) else None
    if not isinstance(recorded, str):
        raise ArtifactIntegrityError(f"{path.name} has a malformed {INTEGRITY_FIELD} field")
    return recorded


def json_artifact_sha256(path: Path) -> str:
    """The artifact's identity: canonical digest if embedded, else the SHA-256 of its bytes."""
    payload = _read_payload(path)
    if _embedded_sha256(path, payload) is not None:
        return canonical_sha256(payload)
    return file_sha256(path)


def verify_json_artifact(path: Path) -> tuple[str, bool]:
    """Return (sha256, verified). ``verified`` is False when no checksum was ever recorded.

    Raises ArtifactIntegrityError when the content no longer matches its embedded checksum
    or, for legacy files, its sidecar.
    """
    payload = _read_payload(path)
    recorded = _embedded_sha256(path, payload)
    if recorded is not None:
        digest = canonical_sha256(payload)
    else:
        digest = file_sha256(path)
        sidecar = legacy_checksum_path(path)
        if not sidecar.is_file():
            return digest, False
        recorded = sidecar.read_text(encoding="utf-8").strip()
    if recorded != digest:
        raise ArtifactIntegrityError(
            f"{path.name} was modified after it was saved "
            f"(recorded sha256 {recorded[:12]}…, now {digest[:12]}…)"
        )
    return digest, True
