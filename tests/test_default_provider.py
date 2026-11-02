"""Tests for the optional ``Config.default_provider`` fallback.

Motivated by clients that drop the ``<provider>/`` prefix before forwarding
(e.g. OpenAI Codex CLI: ``--model anthropic/claude-haiku-4-5-20251001``
arrives at the bridge as ``claude-haiku-4-5-20251001``).
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rosetta.app import create_app
from rosetta.config import Config
from rosetta.pipeline import _resolve_model


def _make_config(default_provider: str | None = None) -> Config:
    os.environ["UP_KEY"] = "sk-up"
    return Config.model_validate(
        {
            "providers": {
                "abc": {
                    "format": "openai_chat",
                    "base_url": "https://upstream.test/v1",
                    "api_key_env": "UP_KEY",
                    "models": [{"id": "m1"}],
                },
            },
            "default_provider": default_provider,
        }
    )


def _chat(config: Config, model: str) -> tuple[httpx.Response, respx.Route]:
    upstream_resp = {
        "id": "chatcmpl-up",
        "object": "chat.completion",
        "model": "m1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    with (
        TestClient(create_app(config)) as c,
        respx.mock(base_url="https://upstream.test/v1", assert_all_called=False) as mock,
    ):
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=upstream_resp)
        )
        r = c.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
    return r, route


def test_e2e_bare_model_routed_to_default_provider() -> None:
    r, route = _chat(_make_config(default_provider="abc"), "m1")
    assert r.status_code == 200
    assert json.loads(route.calls.last.request.content)["model"] == "m1"


def test_e2e_bare_model_without_default_is_400_with_remedy() -> None:
    r, route = _chat(_make_config(), "m1")
    assert r.status_code == 400
    assert "default_provider" in r.json()["error"]["message"]
    assert not route.called


def test_bare_model_resolves_when_default_set() -> None:
    config = _make_config(default_provider="abc")
    provider, key, upstream = _resolve_model("m1", config)
    assert (key, upstream) == ("abc", "m1")
    assert provider is config.providers["abc"]


def test_bare_model_raises_with_remedy_when_no_default() -> None:
    with pytest.raises(ValueError, match=r"must be in format.*default_provider"):
        _resolve_model("m1", _make_config())


def test_prefixed_model_still_works_with_default() -> None:
    _, key, upstream = _resolve_model("abc/m1", _make_config(default_provider="abc"))
    assert (key, upstream) == ("abc", "m1")


def test_config_rejects_unknown_default_provider() -> None:
    with pytest.raises(ValidationError, match="not declared in providers"):
        _make_config(default_provider="openai")


def test_claude_code_gateway_prefix_still_requires_inner_slash() -> None:
    """default_provider does not rescue the claude-code/ gateway path."""
    with pytest.raises(ValueError, match="has gateway prefix but no provider"):
        _resolve_model("claude-code/m1", _make_config(default_provider="abc"))


@pytest.mark.parametrize("model_id", ["", "abc/"])
def test_empty_model_name_not_forwarded(model_id: str) -> None:
    """An empty model name (missing "model" field, or "abc/") raises the format
    error instead of forwarding model="" upstream."""
    with pytest.raises(ValueError, match="must be in format"):
        _resolve_model(model_id, _make_config(default_provider="abc"))
