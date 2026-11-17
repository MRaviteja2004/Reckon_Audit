import os
from unittest.mock import patch

import pytest

from reckon.gateway import (
    GENAI_BASE_URL_ENV,
    UBER_AI_GATEWAY_BASE_URL,
    make_gateway_client,
)


def test_make_gateway_client_requires_usso_token():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="USSO_TOKEN"):
            make_gateway_client()


def test_make_gateway_client_uses_uber_endpoint_and_usso_token():
    with patch("reckon.gateway.OpenAI") as mock_openai, patch.dict(
        os.environ, {"USSO_TOKEN": "test-token"}, clear=True
    ):
        make_gateway_client()

    mock_openai.assert_called_once_with(
        base_url=UBER_AI_GATEWAY_BASE_URL,
        api_key="test-token",
    )


def test_make_gateway_client_honors_genai_base_url_override():
    with patch("reckon.gateway.OpenAI") as mock_openai, patch.dict(
        os.environ,
        {
            "USSO_TOKEN": "test-token",
            GENAI_BASE_URL_ENV: "https://genai-api.example.invalid/v1",
        },
        clear=True,
    ):
        make_gateway_client()

    mock_openai.assert_called_once_with(
        base_url="https://genai-api.example.invalid/v1",
        api_key="test-token",
    )


def test_make_gateway_client_blank_genai_base_url_falls_back_to_default():
    with patch("reckon.gateway.OpenAI") as mock_openai, patch.dict(
        os.environ,
        {"USSO_TOKEN": "test-token", GENAI_BASE_URL_ENV: "   "},
        clear=True,
    ):
        make_gateway_client()

    mock_openai.assert_called_once_with(
        base_url=UBER_AI_GATEWAY_BASE_URL,
        api_key="test-token",
    )
