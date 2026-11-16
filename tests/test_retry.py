from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from instructor.core.exceptions import InstructorRetryException

from reckon.retry import MAX_BACKOFF_DELAY_S, call_with_retry


def _api_error(
    status_code: int,
    headers: dict[str, str] | None = None,
) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://genai-api.uberinternal.com/v1/chat/completions")
    response = httpx.Response(status_code, headers=headers, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


def _instructor_wrapped(error: openai.APIStatusError) -> InstructorRetryException:
    try:
        raise error
    except openai.APIStatusError as cause:
        try:
            raise InstructorRetryException(
                str(cause),
                n_attempts=1,
                total_usage=0,
            ) from cause
        except InstructorRetryException as wrapped:
            return wrapped


def test_call_with_retry_retries_instructor_wrapped_429():
    call = MagicMock(
        side_effect=[
            _instructor_wrapped(_api_error(429)),
            "success",
        ]
    )
    with patch("reckon.retry.time.sleep"):
        result = call_with_retry(call)

    assert result == "success"
    assert call.call_count == 2


def test_call_with_retry_respects_retry_after_header():
    call = MagicMock(side_effect=[_api_error(429, {"Retry-After": "10.992"}), "success"])
    with patch("reckon.retry.time.sleep") as sleep:
        call_with_retry(call)

    sleep.assert_called_once_with(10.992)


def test_call_with_retry_respects_token_reset_header():
    call = MagicMock(
        side_effect=[
            _api_error(429, {"x-ratelimit-reset-tokens": "10s992ms"}),
            "success",
        ]
    )
    with patch("reckon.retry.time.sleep") as sleep:
        call_with_retry(call)

    sleep.assert_called_once_with(10.992)


def test_call_with_retry_does_not_retry_non_retryable_error():
    error = _api_error(400)
    call = MagicMock(side_effect=error)

    with pytest.raises(openai.APIStatusError) as raised:
        call_with_retry(call)

    assert raised.value is error
    assert call.call_count == 1


def test_call_with_retry_reraises_final_retryable_error():
    error = _api_error(429)
    call = MagicMock(side_effect=error)

    with patch("reckon.retry.time.sleep"), pytest.raises(openai.APIStatusError) as raised:
        call_with_retry(call, max_attempts=3)

    assert raised.value is error
    assert call.call_count == 3


def test_call_with_retry_attempt_count_is_configurable():
    call = MagicMock(side_effect=_api_error(503))

    with patch("reckon.retry.time.sleep"), pytest.raises(openai.APIStatusError):
        call_with_retry(call, max_attempts=2)

    assert call.call_count == 2


def test_call_with_retry_bounds_exponential_backoff():
    call = MagicMock(
        side_effect=[
            _api_error(429),
            _api_error(429),
            _api_error(429),
            _api_error(429),
            "success",
        ]
    )
    with patch("reckon.retry.time.sleep") as sleep:
        call_with_retry(call, max_attempts=5)

    delays = [args[0] for args, _ in sleep.call_args_list]
    assert delays == [12.0, 24.0, 48.0, MAX_BACKOFF_DELAY_S]
