"""Uber AI Gateway client configuration."""

import os

from openai import OpenAI

UBER_AI_GATEWAY_BASE_URL = "https://genai-api.uberinternal.com/v1"
GENAI_BASE_URL_ENV = "GENAI_BASE_URL"


def make_gateway_client() -> OpenAI:
    """Create an OpenAI-compatible client routed only through Uber AI Gateway."""
    token = os.environ.get("USSO_TOKEN")
    if not token:
        raise ValueError("USSO_TOKEN environment variable is not set")
    base_url = (os.environ.get(GENAI_BASE_URL_ENV) or "").strip() or UBER_AI_GATEWAY_BASE_URL
    return OpenAI(base_url=base_url, api_key=token)
