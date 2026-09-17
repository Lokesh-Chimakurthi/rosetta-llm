"""Tests for the optional ``Config.default_provider`` fallback.

Motivated by clients that drop the ``<provider>/`` prefix before forwarding
(e.g. OpenAI Codex CLI: ``--model anthropic/claude-haiku-4-5-20251001``
arrives at the bridge as ``claude-haiku-4-5-20251001``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rosetta.config import Config, ProviderConfig
from rosetta.pipeline import _resolve_model


def _make_config(default_provider: str | None = None) -> Config:
    return Config(
        providers={
            "anthropic": ProviderConfig(
                format="anthropic",
                base_url="https://example.com/v1",
                api_key="sk-test",
                models=[{"id": "claude-haiku-4-5-20251001"}],
            ),
        },
        default_provider=default_provider,
    )


class TestDefaultProvider:
    def test_bare_model_resolves_when_default_set(self):
        config = _make_config(default_provider="anthropic")
        provider, key, upstream = _resolve_model("claude-haiku-4-5-20251001", config)
        assert key == "anthropic"
        assert upstream == "claude-haiku-4-5-20251001"
        assert provider is config.providers["anthropic"]

    def test_bare_model_raises_when_no_default(self):
        config = _make_config(default_provider=None)
        with pytest.raises(ValueError, match="must be in format"):
            _resolve_model("claude-haiku-4-5-20251001", config)

    def test_error_message_mentions_default_provider_remedy(self):
        config = _make_config(default_provider=None)
        with pytest.raises(ValueError, match="default_provider"):
            _resolve_model("claude-haiku-4-5-20251001", config)

    def test_prefixed_model_still_works_with_default(self):
        config = _make_config(default_provider="anthropic")
        _, key, upstream = _resolve_model("anthropic/claude-haiku-4-5-20251001", config)
        assert key == "anthropic"
        assert upstream == "claude-haiku-4-5-20251001"

    def test_config_rejects_unknown_default_provider(self):
        with pytest.raises(ValidationError, match="not declared in providers"):
            Config(
                providers={
                    "anthropic": ProviderConfig(
                        format="anthropic",
                        base_url="https://example.com/v1",
                        api_key="sk-test",
                    ),
                },
                default_provider="openai",
            )

    def test_claude_code_gateway_prefix_still_requires_inner_slash(self):
        """default_provider does NOT bail out the claude-code/ gateway path."""
        config = _make_config(default_provider="anthropic")
        with pytest.raises(ValueError, match="has gateway prefix but no provider"):
            _resolve_model("claude-code/claude-haiku-4-5-20251001", config)

    def test_empty_model_id_not_routed_even_with_default(self):
        """An empty model id (the handle() default for a missing "model" field)
        must raise the clear format error instead of being routed to
        ``<default_provider>/`` and forwarding ``model=""`` upstream."""
        config = _make_config(default_provider="anthropic")
        with pytest.raises(ValueError, match="must be in format"):
            _resolve_model("", config)
