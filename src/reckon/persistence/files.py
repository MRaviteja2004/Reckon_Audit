"""Atomic file primitives used by the recording store."""

import hashlib
import os
from pathlib import Path
from uuid import uuid4

HASH_CHUNK_BYTES = 1024 * 1024


class ArtifactIntegrityError(ValueError):
    """A persisted file no longer matches the checksum recorded when it was written."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _write_temp(path: Path, data: bytes) -> Path:
    tmp = temp_sibling(path)
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return tmp


def atomic_replace(path: Path, text: str) -> None:
    tmp = _write_temp(path, text.encode("utf-8"))
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_create(path: Path, data: str | bytes) -> None:
    """Publish a complete file at ``path``; FileExistsError if it already exists."""
    tmp = _write_temp(path, data.encode("utf-8") if isinstance(data, str) else data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
