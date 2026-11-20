"""Client wrapper that observes model calls without altering them."""

import base64
import binascii
import hashlib
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from reckon.latency import elapsed_ms, monotonic_ms
from reckon.persistence.metadata import utc_now
from reckon.persistence.redact import redact_secrets
from reckon.persistence.schema import ModelCallRecord, ModelIOCapture, TokenUsage
from reckon.stage2c.propose import RawFrameEvidenceBatch

_DATA_URL_MARKER = ";base64,"
_SCALAR_TYPES = (str, int, float, bool, type(None))
RECORDED_PARAMETERS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "max_retries",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
    }
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _image_sha256(url: str) -> str:
    """Hash of the decoded image bytes, so it matches a saved evidence frame's checksum."""
    _, marker, payload = url.partition(_DATA_URL_MARKER)
    if marker:
        try:
            return hashlib.sha256(base64.b64decode(payload, validate=True)).hexdigest()
        except binascii.Error:
            pass
    return _sha256_text(url)


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    system: list[str] = []
    user: list[str] = []
    images: list[str] = []
    for message in messages:
        sink = system if message.get("role") == "system" else user
        content = message.get("content")
        if isinstance(content, str):
            sink.append(content)
            continue
        for part in content or []:
            if part.get("type") == "text":
                sink.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                images.append(_image_sha256((part.get("image_url") or {}).get("url", "")))
    return "\n".join(system), "\n".join(user), images


def _raw_response(response: Any) -> Any:
    return getattr(response, "_raw_response", None) or getattr(response, "completion", None)


def _usage(response: Any) -> TokenUsage | None:
    raw = _raw_response(response)
    usage = getattr(raw, "usage", None) if raw is not None else None
    if usage is None:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=_optional_int(getattr(usage, "prompt_tokens", None)),
        completion_tokens=_optional_int(getattr(usage, "completion_tokens", None)),
        total_tokens=_optional_int(getattr(usage, "total_tokens", None)),
    )


def _redacted_body(body: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(body)
    cleaned = redact_secrets(text)
    return body if cleaned == text else json.loads(cleaned)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class RecordingClient:
    """Pass-through for ``client.chat.completions.create``.

    Records every attempt as a ``ModelCallRecord``. With ``ModelIOCapture.hashes`` no
    prompt text or response body is stored, only SHA-256 digests. In both modes images
    are hashed only, only ``RECORDED_PARAMETERS`` are kept (never headers, keys or
    client options), and stored text is passed through ``redact_secrets``. Keeps the Stage 2C
    ``RawFrameEvidenceBatch`` exactly as the model returned it, so it can be persisted
    and replayed later. With ``factory`` the real client is created on first use, so
    wrapping never requires credentials for a run that makes no model call.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        factory: Callable[[], Any] | None = None,
        capture: ModelIOCapture = ModelIOCapture.full,
    ) -> None:
        if (client is None) == (factory is None):
            raise ValueError("RecordingClient needs exactly one of client or factory")
        self._client = client
        self._factory = factory
        self._capture = capture
        self.requests: list[dict[str, Any]] = []
        self.calls: list[ModelCallRecord] = []
        self.stage2c_batch: RawFrameEvidenceBatch | None = None

    @property
    def chat(self) -> Any:
        return self

    @property
    def completions(self) -> Any:
        return self

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if self._client is None:
            assert self._factory is not None
            self._client = self._factory()
        started_at = utc_now()
        started_ms = monotonic_ms()
        try:
            response = self._client.chat.completions.create(*args, **kwargs)
        except BaseException as exc:
            self.calls.append(
                self._record(
                    kwargs,
                    started_at,
                    started_ms,
                    error=redact_secrets(f"{type(exc).__name__}: {exc}"),
                )
            )
            raise
        if isinstance(response, RawFrameEvidenceBatch):
            self.stage2c_batch = response
        record = self._record(kwargs, started_at, started_ms, response=response)
        self.calls.append(record)
        self.requests.append(
            {
                "model": kwargs.get("model"),
                "usage": (
                    {"available": True, **record.usage.model_dump()}
                    if record.usage is not None
                    else {"available": False}
                ),
            }
        )
        return response

    def _record(
        self,
        kwargs: dict[str, Any],
        started_at: Any,
        started_ms: float,
        *,
        response: Any = None,
        error: str | None = None,
    ) -> ModelCallRecord:
        system, user, images = _split_messages(kwargs.get("messages") or [])
        response_model = kwargs.get("response_model")
        body: dict[str, Any] | None = None
        if isinstance(response, BaseModel):
            body = response.model_dump(mode="json")
        raw = _raw_response(response) if response is not None else None
        full = self._capture == ModelIOCapture.full
        return ModelCallRecord(
            sequence=len(self.calls) + 1,
            started_at=started_at,
            duration_ms=elapsed_ms(started_ms),
            requested_model=_optional_str(kwargs.get("model")),
            response_model=getattr(response_model, "__name__", None),
            parameters={
                key: value
                for key, value in kwargs.items()
                if key in RECORDED_PARAMETERS and isinstance(value, _SCALAR_TYPES)
            },
            system_prompt_sha256=_sha256_text(system) if system else None,
            user_text_sha256=_sha256_text(user) if user else None,
            user_text=redact_secrets(user) if full and user else None,
            image_sha256=images,
            served_model=_optional_str(getattr(raw, "model", None)),
            response_id=_optional_str(getattr(raw, "id", None)),
            system_fingerprint=_optional_str(getattr(raw, "system_fingerprint", None)),
            response_sha256=(
                _sha256_text(json.dumps(body, sort_keys=True)) if body is not None else None
            ),
            response=_redacted_body(body) if full and body is not None else None,
            usage=_usage(response) if response is not None else None,
            error=error,
        )


def token_usage_summary(requests: list[dict[str, Any]]) -> dict[str, Any]:
    available = [req for req in requests if req.get("usage", {}).get("available")]
    return {
        "request_count": len(requests),
        "usage_available_count": len(available),
        "prompt_tokens_sum": sum(
            (req.get("usage") or {}).get("prompt_tokens") or 0 for req in available
        ),
        "completion_tokens_sum": sum(
            (req.get("usage") or {}).get("completion_tokens") or 0 for req in available
        ),
        "total_tokens_sum": sum(
            (req.get("usage") or {}).get("total_tokens") or 0 for req in available
        ),
        "requests": requests,
    }
