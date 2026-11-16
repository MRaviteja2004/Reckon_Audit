"""Uber AI Gateway client configuration."""

import os

from openai import OpenAI

UBER_AI_GATEWAY_BASE_URL = "https://genai-api.uberinternal.com/v1"


def make_gateway_client() -> OpenAI:
    """Create an OpenAI-compatible client routed only through Uber AI Gateway."""
    token = os.environ.get("USSO_TOKEN")
    if not token:
        raise ValueError("USSO_TOKEN environment variable is not set")
    return OpenAI(base_url=UBER_AI_GATEWAY_BASE_URL, api_key=token)
