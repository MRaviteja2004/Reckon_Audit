"""Shared retry handling for model API calls."""

import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TypeVar

import openai

T = TypeVar("T")
RetryCallback = Callable[[int, int, float], None]

MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 12.0
MAX_BACKOFF_DELAY_S = 60.0
MAX_SERVER_DELAY_S = 120.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_DURATION_MULTIPLIERS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


def _find_api_status_error(exc: Exception) -> openai.APIStatusError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, openai.APIStatusError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _bounded_server_delay(delay_s: float) -> float | None:
    if 0 < delay_s <= MAX_SERVER_DELAY_S:
        return delay_s
    return None


def _parse_duration(value: str) -> float | None:
    compact = value.strip().lower().replace(" ", "")
    matches = list(_DURATION_PART_RE.finditer(compact))
    if not matches or "".join(match.group(0) for match in matches) != compact:
        return None
    return sum(
        float(match.group(1)) * _DURATION_MULTIPLIERS[match.group(2)]
        for match in matches
    )


def _retry_after_seconds(error: openai.APIStatusError) -> float | None:
    headers = error.response.headers

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            delay = _bounded_server_delay(float(retry_after_ms) / 1000)
        except ValueError:
            delay = None
        if delay is not None:
            return delay

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            delay = _bounded_server_delay(float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = _bounded_server_delay(
                    (retry_at - datetime.now(timezone.utc)).total_seconds()
                )
            except (TypeError, ValueError):
                delay = None
        if delay is not None:
            return delay

    reset_delays = [
        delay
        for name in (
            "x-ratelimit-reset-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset",
        )
        if (value := headers.get(name)) is not None
        if (delay := _parse_duration(value)) is not None
        if _bounded_server_delay(delay) is not None
    ]
    return max(reset_delays, default=None)


def _fallback_delay(attempt: int) -> float:
    return min(
        RETRY_BASE_DELAY_S * (2 ** (attempt - 1)),
        MAX_BACKOFF_DELAY_S,
    )


def call_with_retry(
    call: Callable[[], T],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    on_retry: RetryCallback | None = None,
) -> T:
    """Run a model call with bounded retries for rate limits and transient failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as exc:
            api_error = _find_api_status_error(exc)
            if (
                api_error is None
                or api_error.status_code not in RETRYABLE_STATUS_CODES
                or attempt == max_attempts
            ):
                raise

            delay_s = _retry_after_seconds(api_error) or _fallback_delay(attempt)
            if on_retry is not None:
                on_retry(attempt, api_error.status_code, delay_s)
            time.sleep(delay_s)

    raise RuntimeError("retry loop exited unexpectedly")
