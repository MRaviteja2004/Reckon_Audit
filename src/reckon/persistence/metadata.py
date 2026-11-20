"""Execution metadata: Git revision, versions, models, and configuration."""

import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from reckon.persistence.schema import ExecutionMetadata

PACKAGE_NAME = "reckon"
GIT_TIMEOUT_S = 5
_REPO_DIR = Path(__file__).resolve().parents[3]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _git(args: list[str], repo_dir: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def git_revision(repo_dir: Path = _REPO_DIR) -> tuple[str | None, bool | None]:
    """Return (commit, dirty) for the source tree, or (None, None) outside Git."""
    commit = _git(["rev-parse", "HEAD"], repo_dir)
    if commit is None:
        return None, None
    status = _git(["status", "--porcelain"], repo_dir)
    return commit.strip(), (None if status is None else bool(status.strip()))


def _package_version() -> str | None:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def build_metadata(
    started_at: datetime,
    *,
    models: dict[str, str],
    config: dict[str, str | int | float | bool | None],
    completed_at: datetime | None = None,
    duration_ms: float | None = None,
    metrics: dict[str, float | int | None] | None = None,
) -> ExecutionMetadata:
    commit, dirty = git_revision()
    return ExecutionMetadata(
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        git_commit=commit,
        git_dirty=dirty,
        reckon_version=_package_version(),
        python_version=platform.python_version(),
        models=models,
        config=config,
        metrics=metrics or {},
    )
