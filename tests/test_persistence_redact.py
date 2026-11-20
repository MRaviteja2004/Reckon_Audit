import pytest

from reckon.persistence import redact_secrets
from reckon.persistence.redact import REDACTED

_SENTINEL = "example-usso-sentinel-0000"


def test_redact_secrets_removes_values_of_credential_environment_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USSO_TOKEN", _SENTINEL)
    assert redact_secrets(f"401 for token {_SENTINEL} at gateway") == f"401 for token {REDACTED} at gateway"


def test_redact_secrets_ignores_short_credential_environment_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEY", "abc")
    assert redact_secrets("abc and more") == "abc and more"


def test_redact_secrets_removes_bearer_tokens():
    assert redact_secrets("Authorization: Bearer abcdefgh12345678") == f"Authorization: Bearer {REDACTED}"


def test_redact_secrets_removes_api_key_shaped_values():
    assert redact_secrets("key sk-exampleexampleexample0") == f"key {REDACTED}"


def test_redact_secrets_removes_credential_assignments():
    assert redact_secrets('{"api_key": "placeholder-value"}') == f'{{"api_key": "{REDACTED}"}}'


def test_redact_secrets_leaves_ordinary_text_unchanged():
    text = "Click the Login button; expect You're offline at F3"
    assert redact_secrets(text) == text
