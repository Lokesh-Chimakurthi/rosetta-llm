"""End-to-end smoke tests via FastAPI TestClient with mocked upstream."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from rosetta.app import create_app
from rosetta.config import Config


def _parse_sse(raw: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Split a rendered Anthropic SSE byte stream into (event_name, payload) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.split(b"\n\n"):
        if not block.strip():
            continue
        name, data_str = "", ""
        for line in block.decode().split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].lstrip()
        events.append((name, json.loads(data_str)))
    return events


@pytest.fixture
def client() -> TestClient:
    os.environ["UP_KEY"] = "sk-up"
    cfg = Config.model_validate(
        {
            "providers": {
                "abc": {
                    "format": "openai_chat",
                    "base_url": "https://upstream.test/v1",
                    "api_key_env": "UP_KEY",
                    "models": [{"id": "m1"}],
                },
                "anth": {
                    "format": "anthropic",
                    "base_url": "https://anth.test/v1",
                    "api_key_env": "UP_KEY",
                    "models": [{"id": "claude"}],
                },
            },
        }
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_static(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    ids = sorted(m["id"] for m in data["data"])
    assert ids == ["abc/m1", "anth/claude"]


def test_count_tokens_local(client: TestClient) -> None:
    body = {"messages": [{"role": "user", "content": "hello world"}]}
    r = client.post("/v1/messages/count_tokens", json=body)
    assert r.status_code == 200
    assert r.json()["input_tokens"] >= 1


def test_chat_completions_passthrough(client: TestClient) -> None:
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
    with respx.mock(base_url="https://upstream.test/v1") as mock:
        mock.post("/chat/completions").mock(return_value=httpx.Response(200, json=upstream_resp))
        r = client.post(
            "/v1/chat/completions",
            json={"model": "abc/m1", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_anthropic_to_chat_translation(client: TestClient) -> None:
    """Anthropic /v1/messages → openai_chat provider should translate both ways."""
    upstream_resp = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "m1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    with respx.mock(base_url="https://upstream.test/v1") as mock:
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=upstream_resp)
        )
        r = client.post(
            "/v1/messages",
            json={
                "model": "abc/m1",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert route.called, "should have hit /chat/completions"
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "hello back"
    assert body["stop_reason"] == "end_turn"


def test_unknown_provider_400(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"model": "ghost/missing", "messages": []})
    assert r.status_code == 400
    assert "error" in r.json()


def test_upstream_error_passthrough_format(client: TestClient) -> None:
    """Upstream 4xx should be wrapped in inbound-format error envelope when translating."""
    with respx.mock(base_url="https://upstream.test/v1") as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "rate limited"}}),
        )
        r = client.post(
            "/v1/messages",
            json={
                "model": "abc/m1",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 429
    body = r.json()
    # Must be Anthropic-shaped because /v1/messages was hit.
    assert body["type"] == "error"
    assert body["error"]["message"] == "rate limited"


def test_auth_required_when_configured() -> None:
    os.environ["UP_KEY"] = "sk-up"
    cfg = Config.model_validate(
        {
            "proxy": {"api_keys": ["sk-proxy"]},
            "providers": {
                "abc": {
                    "format": "openai_chat",
                    "base_url": "https://upstream.test/v1",
                    "api_key_env": "UP_KEY",
                    "models": [{"id": "m1"}],
                },
            },
        }
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={"model": "abc/m1", "messages": []})
        assert r.status_code == 401
        with respx.mock(base_url="https://upstream.test/v1") as mock:
            mock.post("/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "x",
                        "object": "chat.completion",
                        "model": "m1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                ),
            )
            r = c.post(
                "/v1/chat/completions",
                json={"model": "abc/m1", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer sk-proxy"},
            )
        assert r.status_code != 401


def test_chat_stream_passthrough(client: TestClient) -> None:
    upstream_sse = (
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    with respx.mock(base_url="https://upstream.test/v1") as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=upstream_sse,
                headers={"content-type": "text/event-stream"},
            ),
        )
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "abc/m1",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            chunks = b"".join(r.iter_bytes())
    assert b"[DONE]" in chunks
    assert b"hi" in chunks


def test_anthropic_stream_translation_carries_usage(client: TestClient) -> None:
    """Anthropic-inbound streaming translation: one usage-bearing terminal message_delta."""
    upstream_sse = (
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":5,"cache_write_tokens":0}}}\n\n'
        b"data: [DONE]\n\n"
    )
    with respx.mock(base_url="https://upstream.test/v1") as mock:
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=upstream_sse,
                headers={"content-type": "text/event-stream"},
            ),
        )
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "abc/m1",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            body = b"".join(r.iter_bytes())

    # The upstream saw the ask for usage.
    sent = json.loads(route.calls.last.request.content)
    assert sent["stream_options"] == {"include_usage": True}

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert names.count("message_delta") == 1
    delta_idx = names.index("message_delta")
    assert names[delta_idx + 1] == "message_stop"

    usage = events[delta_idx][1]["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["cache_read_input_tokens"] == 5

    start = next(data for name, data in events if name == "message_start")
    assert start["message"]["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_anthropic_stream_translation_plain_usage_no_cache(client: TestClient) -> None:
    """Plain case: no cached tokens — input_tokens passes through unchanged."""
    upstream_sse = (
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
        b"data: [DONE]\n\n"
    )
    with respx.mock(base_url="https://upstream.test/v1") as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=upstream_sse,
                headers={"content-type": "text/event-stream"},
            ),
        )
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "abc/m1",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            body = b"".join(r.iter_bytes())

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert names.count("message_delta") == 1
    delta_idx = names.index("message_delta")
    assert names[delta_idx + 1] == "message_stop"

    usage = events[delta_idx][1]["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage.get("cache_read_input_tokens", 0) == 0


def test_env_resolved_custom_header_reaches_upstream() -> None:
    """A header declared via extra_headers_env must be sent upstream after env resolution."""
    os.environ["UP_KEY"] = "sk-up"
    os.environ["ABC_SECRET_HEADER"] = "secret-from-env"
    cfg = Config.model_validate(
        {
            "providers": {
                "abc": {
                    "format": "openai_chat",
                    "base_url": "https://upstream.test/v1",
                    "api_key_env": "UP_KEY",
                    "extra_headers": {"X-Literal-Header": "literal-val"},
                    "extra_headers_env": {"X-Secret-Header": "ABC_SECRET_HEADER"},
                    "models": [{"id": "m1"}],
                },
            },
        }
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        upstream_resp = {
            "id": "chatcmpl-h",
            "object": "chat.completion",
            "model": "m1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        with respx.mock(base_url="https://upstream.test/v1") as mock:
            route = mock.post("/chat/completions").mock(
                return_value=httpx.Response(200, json=upstream_resp)
            )
            r = c.post(
                "/v1/chat/completions",
                json={"model": "abc/m1", "messages": [{"role": "user", "content": "hi"}]},
            )
    assert route.called, "should have hit /chat/completions"
    assert r.status_code == 200
    sent_headers = route.calls.last.request.headers
    assert sent_headers["x-secret-header"] == "secret-from-env"
    assert sent_headers["x-literal-header"] == "literal-val"
