import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reckon.persistence import ModelIOCapture, RecordingClient, token_usage_summary
from reckon.schema import CandidateEvidenceType, CandidateSearchStatus
from reckon.stage2c.propose import RawFrameEvidenceBatch, RawFrameEvidenceCandidate


def _batch() -> RawFrameEvidenceBatch:
    return RawFrameEvidenceBatch(
        candidates=[
            RawFrameEvidenceCandidate(
                expected_step_id="E1",
                frame_ids=["F2"],
                evidence_type=CandidateEvidenceType.ACTION,
                confidence=0.9,
                search_status=CandidateSearchStatus.candidate_found,
            )
        ]
    )


def test_recording_client_keeps_stage2c_batch_and_returns_it_unchanged():
    inner = MagicMock()
    batch = _batch()
    inner.chat.completions.create.return_value = batch
    client = RecordingClient(inner)
    response = client.chat.completions.create(model="m", response_model=RawFrameEvidenceBatch)
    assert response is batch
    assert client.stage2c_batch is batch
    inner.chat.completions.create.assert_called_once_with(
        model="m", response_model=RawFrameEvidenceBatch
    )


def test_recording_client_ignores_non_stage2c_responses():
    inner = MagicMock()
    inner.chat.completions.create.return_value = SimpleNamespace(usage=None)
    client = RecordingClient(inner)
    client.chat.completions.create(model="m")
    assert client.stage2c_batch is None


def test_recording_client_records_token_usage():
    inner = MagicMock()
    inner.chat.completions.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    )
    client = RecordingClient(inner)
    client.chat.completions.create(model="m")
    assert client.requests == [
        {
            "model": "m",
            "usage": {
                "available": True,
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }
    ]


def test_token_usage_summary_sums_only_available_usage():
    summary = token_usage_summary(
        [
            {"model": "m", "usage": {"available": True, "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
            {"model": "m", "usage": {"available": False}},
        ]
    )
    assert (summary["request_count"], summary["usage_available_count"]) == (2, 1)
    assert summary["total_tokens_sum"] == 5


def _messages(jpeg: bytes = b"jpeg-bytes") -> list[dict]:
    encoded = base64.b64encode(jpeg).decode("ascii")
    return [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "F1:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                {"type": "text", "text": "Which frame shows E1?"},
            ],
        },
    ]


def _instructor_like_response() -> RawFrameEvidenceBatch:
    batch = _batch()
    batch._raw_response = SimpleNamespace(
        id="resp-1",
        model="served-model-2026",
        system_fingerprint="fp-1",
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )
    return batch


def test_recording_client_records_prompt_hashes_and_full_text():
    inner = MagicMock()
    inner.chat.completions.create.return_value = _instructor_like_response()
    client = RecordingClient(inner)
    client.chat.completions.create(
        model="m", response_model=RawFrameEvidenceBatch, messages=_messages(), max_retries=2
    )
    [call] = client.calls
    assert call.requested_model == "m"
    assert call.response_model == "RawFrameEvidenceBatch"
    assert call.parameters == {"max_retries": 2}
    assert call.system_prompt_sha256 == hashlib.sha256(b"SYSTEM PROMPT").hexdigest()
    assert call.user_text == "F1:\nWhich frame shows E1?"
    assert call.response == _batch().model_dump(mode="json")


def test_recording_client_records_served_model_response_id_and_usage():
    inner = MagicMock()
    inner.chat.completions.create.return_value = _instructor_like_response()
    client = RecordingClient(inner)
    client.chat.completions.create(model="m", messages=_messages())
    [call] = client.calls
    assert (call.served_model, call.response_id, call.system_fingerprint) == (
        "served-model-2026",
        "resp-1",
        "fp-1",
    )
    assert call.usage.total_tokens == 10


def test_recording_client_hashes_images_by_their_decoded_bytes():
    inner = MagicMock()
    inner.chat.completions.create.return_value = _batch()
    client = RecordingClient(inner)
    client.chat.completions.create(model="m", messages=_messages(b"frame-3"))
    assert client.calls[0].image_sha256 == [hashlib.sha256(b"frame-3").hexdigest()]


def test_recording_client_hashes_only_mode_omits_text_and_response_bodies():
    inner = MagicMock()
    inner.chat.completions.create.return_value = _batch()
    client = RecordingClient(inner, capture=ModelIOCapture.hashes)
    client.chat.completions.create(model="m", messages=_messages())
    [call] = client.calls
    assert (call.user_text, call.response) == (None, None)
    assert call.user_text_sha256 is not None and call.response_sha256 is not None


def test_recording_client_records_a_failed_call_and_reraises():
    inner = MagicMock()
    inner.chat.completions.create.side_effect = TimeoutError("gateway timeout")
    client = RecordingClient(inner)
    with pytest.raises(TimeoutError):
        client.chat.completions.create(model="m", messages=_messages())
    [call] = client.calls
    assert call.error == "TimeoutError: gateway timeout"
    assert client.requests == []


def test_recording_client_factory_is_not_called_until_the_first_request():
    factory = MagicMock(return_value=MagicMock())
    client = RecordingClient(factory=factory)
    factory.assert_not_called()
    client.chat.completions.create(model="m", messages=_messages())
    client.chat.completions.create(model="m", messages=_messages())
    factory.assert_called_once_with()


def test_recording_client_requires_exactly_one_source():
    with pytest.raises(ValueError):
        RecordingClient()


_SENTINEL = "example-usso-sentinel-0000"


def test_recording_client_hashes_only_mode_persists_no_prompt_or_response_content():
    inner = MagicMock()
    inner.chat.completions.create.return_value = _batch()
    client = RecordingClient(inner, capture=ModelIOCapture.hashes)
    client.chat.completions.create(model="m", messages=_messages())
    serialized = client.calls[0].model_dump_json()
    assert "Which frame shows E1" not in serialized
    assert "SYSTEM PROMPT" not in serialized
    assert '"F2"' not in serialized


def test_recording_client_redacts_credentials_from_captured_prompt_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USSO_TOKEN", _SENTINEL)
    inner = MagicMock()
    inner.chat.completions.create.return_value = _batch()
    client = RecordingClient(inner)
    messages = _messages()
    messages[1]["content"].append({"type": "text", "text": f"token={_SENTINEL}"})
    client.chat.completions.create(model="m", messages=messages)
    assert _SENTINEL not in client.calls[0].model_dump_json()


def test_recording_client_redacts_credentials_from_recorded_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USSO_TOKEN", _SENTINEL)
    inner = MagicMock()
    inner.chat.completions.create.side_effect = PermissionError(f"rejected {_SENTINEL}")
    client = RecordingClient(inner)
    with pytest.raises(PermissionError):
        client.chat.completions.create(model="m", messages=_messages())
    assert client.calls[0].error == "PermissionError: rejected [REDACTED]"


def test_recording_client_records_only_allowlisted_parameters():
    inner = MagicMock()
    inner.chat.completions.create.return_value = _batch()
    client = RecordingClient(inner)
    client.chat.completions.create(
        model="m",
        messages=_messages(),
        temperature=0.0,
        api_key="placeholder-key",
        base_url="https://gateway.example.invalid",
        user="someone@example.invalid",
        extra_headers={"Authorization": "Bearer placeholder"},
    )
    assert client.calls[0].parameters == {"temperature": 0.0}
