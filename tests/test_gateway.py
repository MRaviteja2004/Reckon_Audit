import os
from unittest.mock import patch

import pytest

from reckon.gateway import UBER_AI_GATEWAY_BASE_URL, make_gateway_client


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
