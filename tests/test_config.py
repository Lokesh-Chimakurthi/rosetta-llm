"""Unit tests for ProviderConfig header/env resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rosetta.config import ProviderConfig


def _base_provider(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "format": "openai_chat",
        "base_url": "https://upstream.test/v1",
        "api_key": "sk-literal",
        "models": [{"id": "m1"}],
    }
    base.update(overrides)
    return base


def test_extra_headers_env_resolves_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET_HEADER", "secret-value")
    prov = ProviderConfig.model_validate(
        _base_provider(
            extra_headers_env={"X-Custom-Secret": "MY_SECRET_HEADER"},
        )
    )
    assert prov.extra_headers["X-Custom-Secret"] == "secret-value"


def test_extra_headers_env_unset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)
    with pytest.raises(ValidationError, match="MISSING_ENV_VAR"):
        ProviderConfig.model_validate(
            _base_provider(
                extra_headers_env={"X-Custom-Secret": "MISSING_ENV_VAR"},
            )
        )


def test_extra_headers_env_collision_with_literal_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_SECRET_HEADER", "secret-value")
    with pytest.raises(ValidationError, match="X-Custom-Secret"):
        ProviderConfig.model_validate(
            _base_provider(
                extra_headers={"X-Custom-Secret": "literal-value"},
                extra_headers_env={"X-Custom-Secret": "MY_SECRET_HEADER"},
            )
        )


def test_extra_headers_literal_only_still_works() -> None:
    prov = ProviderConfig.model_validate(_base_provider(extra_headers={"X-Custom": "value"}))
    assert prov.extra_headers["X-Custom"] == "value"
    assert prov.extra_headers_env == {}


def test_extra_headers_env_empty_is_noop() -> None:
    prov = ProviderConfig.model_validate(_base_provider())
    assert prov.extra_headers == {}


def test_extra_headers_env_value_hidden_from_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The secret value must never appear in the raised error message.
    monkeypatch.setenv("LEAK_CHECK", "do-not-leak-me")
    # Collision forces a ValidationError; assert the secret is absent from the message.
    with pytest.raises(ValidationError) as exc_info:
        ProviderConfig.model_validate(
            _base_provider(
                extra_headers={"X-Secret": "literal"},
                extra_headers_env={"X-Secret": "LEAK_CHECK"},
            )
        )
    assert "do-not-leak-me" not in str(exc_info.value)
    assert "LEAK_CHECK" in str(exc_info.value)


@pytest.mark.parametrize("fmt", ["anthropic", "openai_responses"])
def test_forward_cache_control_rejected_outside_openai_chat(fmt: str) -> None:
    models = [{"id": "m1", "forward_cache_control": False}]
    with pytest.raises(ValidationError, match=r"\['m1'\].*only to 'openai_chat'"):
        ProviderConfig.model_validate(_base_provider(format=fmt, models=models))
