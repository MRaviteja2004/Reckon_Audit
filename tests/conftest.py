from collections.abc import Iterator
from unittest.mock import patch

import pytest


class LiveGatewayCallError(AssertionError):
    """Raised when a test would construct a real model gateway client."""


def _refuse_gateway(*_args: object, **_kwargs: object) -> None:
    raise LiveGatewayCallError("a test tried to create a live model gateway client")


@pytest.fixture
def no_live_gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail loudly if code under test reaches the real gateway instead of a test double."""
    monkeypatch.setenv("USSO_TOKEN", "test-token-not-real")
    with patch("reckon.gateway.OpenAI", side_effect=_refuse_gateway):
        yield
