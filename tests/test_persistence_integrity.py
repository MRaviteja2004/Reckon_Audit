import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from reckon.persistence import (
    ArtifactIntegrityError,
    ExecutionMetadata,
    Stage1Artifact,
    Stage2Artifact,
)
from reckon.persistence.integrity import (
    INTEGRITY_FIELD,
    canonical_sha256,
    create_sealed,
    json_artifact_sha256,
    legacy_checksum_path,
    replace_sealed,
    sealed_json,
    verify_json_artifact,
)
from reckon.schema import ActionType, ExpectedStep, ExpectedTrace


def _artifact(target: str = "Login") -> Stage1Artifact:
    return Stage1Artifact(
        metadata=ExecutionMetadata(
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            duration_ms=1234.5678,
            python_version="3.13.0",
        ),
        expected_trace=ExpectedTrace(
            scenario_title="Login — café",
            steps=[ExpectedStep(index=1, action=ActionType.click, target=target)],
        ),
    )


def _payload(path: Path) -> dict:
    return json.loads(path.read_text())


def test_create_sealed_embeds_the_checksum_in_the_artifact(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    digest = create_sealed(path, _artifact())
    assert _payload(path)[INTEGRITY_FIELD] == {"sha256": digest}


def test_embedded_checksum_excludes_the_integrity_field_itself(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    digest = create_sealed(path, _artifact())
    body = {key: value for key, value in _payload(path).items() if key != INTEGRITY_FIELD}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert digest == expected


def test_sealed_json_is_deterministic():
    assert sealed_json(_artifact()) == sealed_json(_artifact())


def test_canonical_sha256_ignores_key_order():
    assert canonical_sha256({"a": 1, "b": [1.5, "x"]}) == canonical_sha256({"b": [1.5, "x"], "a": 1})


def test_verify_accepts_an_unmodified_artifact(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    digest = create_sealed(path, _artifact())
    assert verify_json_artifact(path) == (digest, True)


def test_verify_accepts_an_artifact_after_json_reload_and_reformatting(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    digest = create_sealed(path, _artifact())
    path.write_text(json.dumps(_payload(path), indent=None, sort_keys=True))
    assert verify_json_artifact(path) == (digest, True)


def test_verify_accepts_a_model_round_trip_through_the_loader(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    create_sealed(path, _artifact())
    reloaded = Stage1Artifact.model_validate_json(path.read_text())
    assert reloaded == _artifact()
    assert sealed_json(reloaded)[1] == json_artifact_sha256(path)


def test_verify_detects_modified_content(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    create_sealed(path, _artifact())
    path.write_text(path.read_text().replace('"Login"', '"Logout"'))
    with pytest.raises(ArtifactIntegrityError, match="modified after it was saved"):
        verify_json_artifact(path)


def test_verify_detects_a_forged_checksum(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    create_sealed(path, _artifact())
    payload = _payload(path)
    payload[INTEGRITY_FIELD] = {"sha256": "0" * 64}
    path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactIntegrityError):
        verify_json_artifact(path)


def test_verify_rejects_a_malformed_integrity_field(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    create_sealed(path, _artifact())
    payload = _payload(path)
    payload[INTEGRITY_FIELD] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactIntegrityError, match="malformed"):
        verify_json_artifact(path)


def test_verify_rejects_unreadable_json(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    path.write_text("{truncated")
    with pytest.raises(ArtifactIntegrityError, match="not readable JSON"):
        verify_json_artifact(path)


def test_create_sealed_is_write_once_and_writes_no_sidecar(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    create_sealed(path, _artifact())
    with pytest.raises(FileExistsError):
        create_sealed(path, _artifact("Other"))
    assert not legacy_checksum_path(path).exists()


def test_replace_sealed_recomputes_the_checksum_for_new_content(tmp_path: Path):
    path = tmp_path / "recording.json"
    first = create_sealed(path, _artifact())
    second = replace_sealed(path, _artifact("Sign in"))
    assert first != second
    assert verify_json_artifact(path) == (second, True)


def test_legacy_artifact_with_a_matching_sidecar_verifies_by_file_bytes(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    path.write_text(_artifact().model_dump_json(indent=2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    legacy_checksum_path(path).write_text(digest + "\n")
    assert verify_json_artifact(path) == (digest, True)
    assert json_artifact_sha256(path) == digest


def test_legacy_artifact_modified_after_its_sidecar_is_rejected(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    path.write_text(_artifact().model_dump_json(indent=2))
    legacy_checksum_path(path).write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    path.write_text(_artifact("Logout").model_dump_json(indent=2))
    with pytest.raises(ArtifactIntegrityError, match="modified after it was saved"):
        verify_json_artifact(path)


def test_legacy_artifact_without_any_checksum_is_reported_unverified(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    path.write_text(_artifact().model_dump_json(indent=2))
    assert verify_json_artifact(path) == (hashlib.sha256(path.read_bytes()).hexdigest(), False)


def test_embedded_checksum_takes_precedence_over_a_stale_sidecar(tmp_path: Path):
    path = tmp_path / "stage-1.json"
    digest = create_sealed(path, _artifact())
    legacy_checksum_path(path).write_text("0" * 64 + "\n")
    assert verify_json_artifact(path) == (digest, True)


def _legacy_stage2_payload() -> dict:
    """A Stage 2 artifact as written before observations carried interaction diagnostics."""
    return {
        "schema_version": 1,
        "version": 1,
        "metadata": {
            "started_at": "2026-01-01T00:00:00Z",
            "duration_ms": 10.0,
            "python_version": "3.13.0",
        },
        "observed_trace": {
            "source_file": "recording.mp4",
            "actions": [
                {
                    "index": 1,
                    "timestamp_ms": 500,
                    "action": "click",
                    "target": "Sign in",
                    "confidence": 0.9,
                    "interaction": None,
                }
            ],
        },
    }


def test_legacy_stage2_artifact_without_interaction_diagnostics_still_verifies(tmp_path: Path):
    payload = _legacy_stage2_payload()
    payload[INTEGRITY_FIELD] = {"sha256": canonical_sha256(payload)}
    path = tmp_path / "stage-2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert verify_json_artifact(path)[1] is True


def test_legacy_stage2_artifact_loads_without_interaction_diagnostics():
    loaded = Stage2Artifact.model_validate(_legacy_stage2_payload())

    action = loaded.observed_trace.actions[0]
    assert (action.interaction, action.interaction_diagnostic) == (None, None)
