import hashlib
from pathlib import Path

import pytest

from reckon.persistence.files import atomic_create, atomic_replace, file_sha256


def test_atomic_create_refuses_to_overwrite(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    atomic_create(path, "first")
    with pytest.raises(FileExistsError):
        atomic_create(path, "second")
    assert path.read_text() == "first"


def test_atomic_create_leaves_no_temp_files(tmp_path: Path):
    atomic_create(tmp_path / "stage-1.json", b"bytes")
    assert [p.name for p in tmp_path.iterdir()] == ["stage-1.json"]


def test_atomic_replace_overwrites_in_place(tmp_path: Path):
    path = tmp_path / "recording.json"
    path.write_text("old")
    atomic_replace(path, "new")
    assert path.read_text() == "new"


def test_file_sha256_hashes_the_file_bytes(tmp_path: Path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"jpeg")
    assert file_sha256(path) == hashlib.sha256(b"jpeg").hexdigest()
